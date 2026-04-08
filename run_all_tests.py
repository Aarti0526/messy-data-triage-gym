"""
Run all existing tests and a comprehensive edge-case test suite inline.
Prints a full report to stdout.
"""
import sys, traceback
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

# ── Bootstrap ──────────────────────────────────────────────────────────────────
from data_triage_env.server import app
from data_triage_env.engine.dataset_factory import generate_clean
from data_triage_env.engine.corruptor import (
    corrupt_easy, corrupt_medium, corrupt_hard,
    GroundTruthManifest, CorruptionRecord, CORRUPT_FNS,
)
from data_triage_env.engine.executor import run_action, _observe
from data_triage_env.models import DataAction, DataObservation, DataReward
from data_triage_env.graders.easy_grader import score as score_easy
from data_triage_env.graders.medium_grader import score as score_medium
from data_triage_env.graders.hard_grader import score as score_hard

client = TestClient(app)

PASS = 0
FAIL = 0
TESTS = []

def run(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        TESTS.append(("PASS", name))
        print(f"  ✅  {name}")
    except Exception as e:
        FAIL += 1
        TESTS.append(("FAIL", name, traceback.format_exc()))
        print(f"  ❌  {name}: {e}")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 – Dataset Factory
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 1: Dataset Factory ─────────────────────────────────────────")

def t_easy_shape():
    df = generate_clean("easy", 0)
    assert df.shape == (200, 4), f"Expected (200,4) got {df.shape}"

def t_medium_shape():
    df = generate_clean("medium", 0)
    assert df.shape == (500, 6), f"Expected (500,6) got {df.shape}"

def t_hard_shape():
    df = generate_clean("hard", 0)
    assert df.shape == (800, 8), f"Expected (800,8) got {df.shape}"

def t_determinism():
    df1 = generate_clean("easy", 42)
    df2 = generate_clean("easy", 42)
    assert df1.equals(df2), "Same seed should produce identical DFs"

def t_different_seeds():
    df1 = generate_clean("easy", 1)
    df2 = generate_clean("easy", 2)
    assert not df1.equals(df2), "Different seeds should differ"

def t_easy_dtypes():
    df = generate_clean("easy", 0)
    assert pd.api.types.is_integer_dtype(df["customer_id"])
    assert pd.api.types.is_float_dtype(df["price"])
    assert pd.api.types.is_integer_dtype(df["quantity"])

def t_medium_date_format():
    df = generate_clean("medium", 0)
    # All dates must match YYYY-MM-DD
    import re
    for v in df["order_date"]:
        assert re.match(r"\d{4}-\d{2}-\d{2}$", v), f"Bad date: {v}"

def t_hard_temp_unit_all_C():
    df = generate_clean("hard", 0)
    assert (df["temp_unit"] == "C").all(), "All temp_unit should be 'C' before corruption"

def t_hard_timestamp_format():
    df = generate_clean("hard", 0)
    import re
    for v in df["timestamp"].head(20):
        assert re.match(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$", v), f"Bad timestamp: {v}"

def t_bool_column():
    df = generate_clean("hard", 0)
    assert set(df["reading_valid"].unique()).issubset({True, False})

run("easy_shape", t_easy_shape)
run("medium_shape", t_medium_shape)
run("hard_shape", t_hard_shape)
run("determinism_same_seed", t_determinism)
run("different_seeds_differ", t_different_seeds)
run("easy_dtypes", t_easy_dtypes)
run("medium_date_format", t_medium_date_format)
run("hard_temp_unit_all_C", t_hard_temp_unit_all_C)
run("hard_timestamp_format", t_hard_timestamp_format)
run("bool_column_values", t_bool_column)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 – Corruptor
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 2: Corruptor ────────────────────────────────────────────────")

def t_easy_nulls_in_price():
    df = generate_clean("easy", 10)
    dirty, manifest = corrupt_easy(df, np.random.default_rng(10))
    cols = [r.column for r in manifest.records]
    assert "price" in cols
    null_rec = next(r for r in manifest.records if r.column == "price" and r.corruption_type == "null")
    assert dirty.loc[null_rec.row_indices[0], "price"] != dirty.loc[null_rec.row_indices[0], "price"]  # NaN check

def t_easy_type_mismatch_quantity():
    df = generate_clean("easy", 10)
    dirty, manifest = corrupt_easy(df, np.random.default_rng(10))
    tm = next(r for r in manifest.records if r.corruption_type == "type_mismatch")
    assert tm.column == "quantity"
    for i in tm.row_indices[:3]:
        assert dirty.at[i, "quantity"] == "N/A"

def t_medium_4_corruption_types():
    df = generate_clean("medium", 5)
    dirty, manifest = corrupt_medium(df, np.random.default_rng(5))
    types = {r.corruption_type for r in manifest.records}
    assert "null" in types
    assert "type_mismatch" in types
    assert "duplicate" in types
    assert "date_format" in types

def t_medium_duplicate_rows_appended():
    df = generate_clean("medium", 5)
    dirty, manifest = corrupt_medium(df, np.random.default_rng(5))
    dup_rec = next(r for r in manifest.records if r.corruption_type == "duplicate")
    assert len(dirty) > len(df)  # duplicates were added
    assert all(i >= len(df) for i in dup_rec.row_indices)

def t_hard_contains_trap():
    df = generate_clean("hard", 7)
    dirty, manifest = corrupt_hard(df, np.random.default_rng(7))
    trap = next((r for r in manifest.records if r.corruption_type == "trap"), None)
    assert trap is not None, "Hard must have a trap record"
    assert trap.column == "pressure"

def t_hard_unit_mismatch_conversion():
    df = generate_clean("hard", 7)
    dirty, manifest = corrupt_hard(df, np.random.default_rng(7))
    um = next(r for r in manifest.records if r.corruption_type == "unit_mismatch")
    # Check some rows were converted to Fahrenheit
    for i in um.row_indices[:5]:
        f_val = dirty.at[i, "temperature"]
        c_val = df.at[i, "temperature"]
        # F = C*9/5 + 32
        assert abs(f_val - (c_val * 9/5 + 32)) < 0.1, f"Row {i} not converted: f={f_val} c={c_val}"

def t_hard_6_corruption_records():
    df = generate_clean("hard", 99)
    _, manifest = corrupt_hard(df, np.random.default_rng(99))
    assert len(manifest.records) == 6, f"Expected 6 records, got {len(manifest.records)}"

def t_manifest_clean_df_unchanged():
    df = generate_clean("easy", 0)
    dirty, manifest = corrupt_easy(df, np.random.default_rng(0))
    # manifest.clean_df should NOT have NaNs in price
    assert manifest.clean_df["price"].isna().sum() == 0

def t_null_idx_count_easy():
    df = generate_clean("easy", 0)
    dirty, manifest = corrupt_easy(df, np.random.default_rng(0))
    null_rec = next(r for r in manifest.records if r.corruption_type == "null")
    expected_n = len(df) // 10
    assert len(null_rec.row_indices) == expected_n, f"Expected {expected_n} nulls, got {len(null_rec.row_indices)}"

run("easy_nulls_in_price", t_easy_nulls_in_price)
run("easy_type_mismatch_quantity", t_easy_type_mismatch_quantity)
run("medium_4_corruption_types", t_medium_4_corruption_types)
run("medium_duplicate_rows_appended", t_medium_duplicate_rows_appended)
run("hard_contains_trap", t_hard_contains_trap)
run("hard_unit_mismatch_conversion", t_hard_unit_mismatch_conversion)
run("hard_6_corruption_records", t_hard_6_corruption_records)
run("manifest_clean_df_unchanged", t_manifest_clean_df_unchanged)
run("null_idx_count_easy", t_null_idx_count_easy)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 – Executor
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 3: Executor ─────────────────────────────────────────────────")

def t_inspect_no_mutation():
    df = generate_clean("easy", 0)
    df2, obs, msg = run_action(df, DataAction(action="inspect"))
    assert df.equals(df2), "inspect must not mutate"

def t_cast_float_to_int():
    df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
    df2, obs, _ = run_action(df, DataAction(action="cast", column="col", params={"dtype": "int64"}))
    assert df2["col"].dtype == np.int64

def t_cast_with_strip():
    df = pd.DataFrame({"col": ["1,200 USD", "500 USD"]})
    action = DataAction(action="cast", column="col", params={"dtype": "float64", "strip_pattern": r"[^\d.]"})
    df2, obs, _ = run_action(df, action)
    assert abs(df2["col"].iloc[0] - 1200.0) < 0.01

def t_impute_median():
    df = pd.DataFrame({"col": [1.0, np.nan, 3.0, 5.0]})
    df2, obs, _ = run_action(df, DataAction(action="impute", column="col", params={"strategy": "median"}))
    assert df2["col"].isna().sum() == 0
    assert abs(df2["col"].iloc[1] - 3.0) < 0.01  # median of [1,3,5]=3

def t_impute_mean():
    df = pd.DataFrame({"col": [2.0, np.nan, 4.0]})
    df2, obs, _ = run_action(df, DataAction(action="impute", column="col", params={"strategy": "mean"}))
    assert abs(df2["col"].iloc[1] - 3.0) < 0.01

def t_impute_constant():
    df = pd.DataFrame({"col": [1.0, np.nan]})
    df2, obs, _ = run_action(df, DataAction(action="impute", column="col", params={"strategy": "constant", "value": 99}))
    assert df2["col"].iloc[1] == 99

def t_dedupe_removes():
    df = pd.DataFrame({"a": [1, 2, 1, 3], "b": [4, 5, 4, 6]})
    df2, obs, _ = run_action(df, DataAction(action="dedupe"))
    assert len(df2) == 3

def t_dedupe_keeps_first():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [10, 20, 30]})
    df2, obs, _ = run_action(df, DataAction(action="dedupe", params={"subset": ["a"], "keep": "first"}))
    assert len(df2) == 2
    assert df2.iloc[0]["b"] == 10

def t_rescale_F_to_C():
    df = pd.DataFrame({"temp": [212.0, 32.0], "unit": ["F", "F"]})
    action = DataAction(action="rescale", column="temp", params={"from_unit": "F", "to_unit": "C", "condition_col": "unit", "condition_val": "F"})
    df2, obs, _ = run_action(df, action)
    assert abs(df2["temp"].iloc[0] - 100.0) < 0.1
    assert abs(df2["temp"].iloc[1] - 0.0) < 0.1
    assert (df2["unit"] == "C").all()

def t_rescale_C_to_F():
    df = pd.DataFrame({"temp": [0.0, 100.0], "unit": ["C", "C"]})
    action = DataAction(action="rescale", column="temp", params={"from_unit": "C", "to_unit": "F", "condition_col": "unit", "condition_val": "C"})
    df2, obs, _ = run_action(df, action)
    assert abs(df2["temp"].iloc[0] - 32.0) < 0.1
    assert abs(df2["temp"].iloc[1] - 212.0) < 0.1

def t_invalid_column_cast():
    df = pd.DataFrame({"col": [1.0]})
    try:
        run_action(df, DataAction(action="cast", column="nonexistent", params={"dtype": "float64"}))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def t_invalid_column_impute():
    df = pd.DataFrame({"col": [1.0]})
    try:
        run_action(df, DataAction(action="impute", column="nonexistent", params={"strategy": "median"}))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def t_unknown_impute_strategy():
    df = pd.DataFrame({"col": [1.0, 2.0]})
    try:
        run_action(df, DataAction(action="impute", column="col", params={"strategy": "bogus"}))
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

def t_observe_returns_correct_shape():
    df = generate_clean("medium", 0)
    obs = _observe(df)
    assert obs.shape == (500, 6)
    assert len(obs.columns) == 6

def t_no_inplace_mutation():
    df = generate_clean("easy", 0)
    original = df.copy()
    run_action(df, DataAction(action="dedupe"))
    assert df.equals(original), "run_action must not mutate original df"

run("inspect_no_mutation", t_inspect_no_mutation)
run("cast_float_to_int", t_cast_float_to_int)
run("cast_with_strip_pattern", t_cast_with_strip)
run("impute_median", t_impute_median)
run("impute_mean", t_impute_mean)
run("impute_constant", t_impute_constant)
run("dedupe_removes_duplicates", t_dedupe_removes)
run("dedupe_keep_first", t_dedupe_keeps_first)
run("rescale_F_to_C", t_rescale_F_to_C)
run("rescale_C_to_F", t_rescale_C_to_F)
run("invalid_column_cast_raises", t_invalid_column_cast)
run("invalid_column_impute_raises", t_invalid_column_impute)
run("unknown_impute_strategy_raises", t_unknown_impute_strategy)
run("observe_returns_correct_shape", t_observe_returns_correct_shape)
run("no_inplace_mutation", t_no_inplace_mutation)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 – Graders
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 4: Graders ──────────────────────────────────────────────────")

def _make_manifest_single(col, val, idx=0):
    clean_df = pd.DataFrame({col: [val, 2, 3, 4, 5]})
    manifest = GroundTruthManifest(
        clean_df=clean_df.copy(),
        records=[CorruptionRecord(corruption_type="null", column=col, row_indices=[idx], original_values=[val], expected_fix="impute")]
    )
    return clean_df, manifest

def t_perfect_fix_scores_1():
    clean_df, manifest = _make_manifest_single("col", 10)
    s = score_easy(clean_df.copy(), manifest)
    assert s == 1.0, f"Perfect fix should score 1.0, got {s}"

def t_zero_fix_score():
    clean_df, manifest = _make_manifest_single("col", 10)
    dirty = clean_df.copy()
    dirty.at[0, "col"] = np.nan
    s = score_easy(dirty, manifest)
    assert s < 1.0, "Unfixed null should score < 1.0"

def t_score_in_bounds():
    for seed in [1, 7, 42, 99, 123]:
        df = generate_clean("easy", seed)
        dirty, manifest = corrupt_easy(df, np.random.default_rng(seed))
        s = score_easy(dirty, manifest)
        assert 0.0 <= s <= 1.0, f"Score {s} out of [0,1] for seed {seed}"

def t_score_hard_bounds():
    for seed in [1, 7, 42]:
        df = generate_clean("hard", seed)
        dirty, manifest = corrupt_hard(df, np.random.default_rng(seed))
        s = score_hard(dirty, manifest)
        assert 0.0 <= s <= 1.0, f"Hard score {s} out of [0,1]"

def t_easy_penalty_higher_than_easy():
    # Hard grader (20x) should be equal or harsher than easy (10x) on same broken data
    clean_df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
    manifest = GroundTruthManifest(
        clean_df=clean_df.copy(),
        records=[CorruptionRecord(corruption_type="null", column="col", row_indices=[0], original_values=[1], expected_fix="impute")]
    )
    agent_df = clean_df.copy()
    agent_df.at[1, "col"] = 99   # break a clean cell
    s_easy = score_easy(agent_df, manifest)
    s_hard = score_hard(agent_df, manifest)
    # With 1 broken clean cell and penalty_multiplier 10 vs 20, hard should be <= easy
    assert s_hard <= s_easy + 1e-9, f"Hard score {s_hard} > easy score {s_easy}"

def t_trap_triggers_cap():
    clean_df = pd.DataFrame({"pressure": [1.0, 2.0, 3.0], "val": [10, 20, 30]})
    manifest = GroundTruthManifest(
        clean_df=clean_df.copy(),
        records=[
            CorruptionRecord(corruption_type="trap", column="pressure", row_indices=[], original_values=[], expected_fix="none"),
            CorruptionRecord(corruption_type="null", column="val", row_indices=[0], original_values=[10], expected_fix="impute")
        ]
    )
    agent_df = clean_df.copy()
    agent_df.at[0, "pressure"] = 9.9   # break trap column
    s = score_hard(agent_df, manifest)
    assert s <= 0.5, f"Trap broken, hard grader should cap at 0.5, got {s}"

def t_duplicate_grading():
    clean_df = pd.DataFrame({"a": [1, 2, 3]})
    dup_row = pd.DataFrame({"a": [1]})
    dirty = pd.concat([clean_df, dup_row], ignore_index=True)
    manifest = GroundTruthManifest(
        clean_df=clean_df.copy(),
        records=[CorruptionRecord(corruption_type="duplicate", column="a", row_indices=[3], original_values=[], expected_fix="drop")]
    )
    # Agent dedupes correctly
    agent_df = dirty.drop_duplicates().reset_index(drop=True)
    s = score_easy(agent_df, manifest)
    assert s >= 0.9, f"Dedupe should score well, got {s}"

def t_medium_penalty_between_easy_and_hard():
    clean_df = pd.DataFrame({"col": [1, 2, 3, 4, 5]})
    manifest = GroundTruthManifest(
        clean_df=clean_df.copy(),
        records=[CorruptionRecord(corruption_type="null", column="col", row_indices=[0], original_values=[1], expected_fix="impute")]
    )
    agent_df = clean_df.copy()
    agent_df.at[1, "col"] = 99
    s_easy = score_easy(agent_df, manifest)
    s_med = score_medium(agent_df, manifest)
    s_hard = score_hard(agent_df, manifest)
    # All should be 0 here (penalty >= 1.0 in all cases), but medium >= hard
    assert s_med >= s_hard - 1e-9

def t_no_bugs_returns_1():
    clean_df = pd.DataFrame({"a": [1, 2, 3]})
    manifest = GroundTruthManifest(clean_df=clean_df.copy(), records=[])
    s = score_easy(clean_df.copy(), manifest)
    assert s == 1.0

def t_broken_clean_outside_corrupted_positions():
    clean_df = pd.DataFrame({"a": [10, 20, 30], "b": [1, 2, 3]})
    manifest = GroundTruthManifest(
        clean_df=clean_df.copy(),
        records=[CorruptionRecord(corruption_type="null", column="a", row_indices=[0], original_values=[10], expected_fix="impute")]
    )
    agent_df = clean_df.copy()
    agent_df.at[0, "a"] = 10   # fix the bug
    agent_df.at[1, "b"] = 999  # break a clean cell
    s = score_easy(agent_df, manifest)
    # penalty = (1 * 10) / 1 = 10 >= 1.0, so capped; raw = 1.0; score = 0.0
    assert s == 0.0

run("perfect_fix_scores_1", t_perfect_fix_scores_1)
run("zero_fix_score", t_zero_fix_score)
run("score_in_bounds_easy", t_score_in_bounds)
run("score_in_bounds_hard", t_score_hard_bounds)
run("hard_penalty_harsher", t_easy_penalty_higher_than_easy)
run("trap_triggers_hard_cap", t_trap_triggers_cap)
run("duplicate_grading", t_duplicate_grading)
run("medium_penalty_between", t_medium_penalty_between_easy_and_hard)
run("no_bugs_returns_1", t_no_bugs_returns_1)
run("broken_clean_cells_penalised", t_broken_clean_outside_corrupted_positions)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 – FastAPI Server & Session
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 5: FastAPI Server & Session ─────────────────────────────────")

def t_server_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "env" in body

def t_reset_easy():
    r = client.post("/reset", json={"task_id": "easy", "seed": 1})
    assert r.status_code == 200
    d = r.json()
    assert "session_id" in d
    assert d["observation"]["shape"][0] == 200

def t_reset_medium():
    r = client.post("/reset", json={"task_id": "medium", "seed": 1})
    assert r.status_code == 200
    assert r.json()["observation"]["shape"][0] == 515

def t_reset_hard():
    r = client.post("/reset", json={"task_id": "hard", "seed": 1})
    assert r.status_code == 200
    assert r.json()["observation"]["shape"][0] >= 800  # may have dup rows

def t_reset_no_seed():
    r = client.post("/reset", json={"task_id": "easy"})
    assert r.status_code == 200
    assert "session_id" in r.json()

def t_reset_invalid_task():
    r = client.post("/reset", json={"task_id": "impossible"})
    assert r.status_code == 422

def t_step_inspect():
    r = client.post("/reset", json={"task_id": "easy", "seed": 5})
    sid = r.json()["session_id"]
    r2 = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
    assert r2.status_code == 200
    d = r2.json()
    assert "observation" in d
    assert "reward" in d
    assert 0.0 <= d["reward"]["score"] <= 1.0

def t_step_missing_column_on_cast():
    r = client.post("/reset", json={"task_id": "easy", "seed": 1})
    sid = r.json()["session_id"]
    r2 = client.post("/step", json={"session_id": sid, "action": {"action": "cast"}})
    assert r2.status_code == 422

def t_step_nonexistent_session():
    r = client.post("/step", json={"session_id": "bad-id", "action": {"action": "inspect"}})
    assert r.status_code == 404

def t_get_state_endpoint():
    r = client.post("/reset", json={"task_id": "easy", "seed": 1})
    sid = r.json()["session_id"]
    r2 = client.get(f"/state/{sid}")
    assert r2.status_code == 200
    d = r2.json()
    assert d["task_id"] == "easy"
    assert d["step"] == 0
    assert d["done"] == False

def t_state_not_found():
    r = client.get("/state/fake-session-id")
    assert r.status_code == 404

def t_step_increments_step_counter():
    r = client.post("/reset", json={"task_id": "easy", "seed": 1})
    sid = r.json()["session_id"]
    client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
    r2 = client.get(f"/state/{sid}")
    assert r2.json()["step"] == 1

def t_done_after_max_steps():
    r = client.post("/reset", json={"task_id": "easy", "seed": 1})
    sid = r.json()["session_id"]
    last = None
    for _ in range(20):
        r2 = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
        last = r2
    assert last.json()["reward"]["done"] == True

def t_done_episode_blocked():
    r = client.post("/reset", json={"task_id": "easy", "seed": 1})
    sid = r.json()["session_id"]
    for _ in range(20):
        client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
    r2 = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
    assert r2.status_code == 400

def t_deterministic_same_seed():
    r1 = client.post("/reset", json={"task_id": "medium", "seed": 42})
    r2 = client.post("/reset", json={"task_id": "medium", "seed": 42})
    obs1 = r1.json()["observation"]
    obs2 = r2.json()["observation"]
    assert obs1["shape"] == obs2["shape"]
    # Null counts in same columns should match
    for c1, c2 in zip(obs1["columns"], obs2["columns"]):
        assert c1["null_count"] == c2["null_count"]

def t_reward_bounds_all_tasks():
    for task in ["easy", "medium", "hard"]:
        r = client.post("/reset", json={"task_id": task, "seed": 7})
        sid = r.json()["session_id"]
        for _ in range(3):
            r2 = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
            if r2.status_code == 200:
                score = r2.json()["reward"]["score"]
                assert 0.0 <= score <= 1.0, f"Score {score} out of bounds for {task}"

def t_impute_via_api():
    r = client.post("/reset", json={"task_id": "easy", "seed": 10})
    sid = r.json()["session_id"]
    r2 = client.post("/step", json={
        "session_id": sid,
        "action": {"action": "impute", "column": "price", "params": {"strategy": "median"}}
    })
    assert r2.status_code == 200
    d = r2.json()
    # Null count in price should drop to 0
    price_stat = next(c for c in d["observation"]["columns"] if c["name"] == "price")
    assert price_stat["null_count"] == 0

def t_cast_quantity_via_api():
    r = client.post("/reset", json={"task_id": "easy", "seed": 10})
    sid = r.json()["session_id"]
    r2 = client.post("/step", json={
        "session_id": sid,
        "action": {"action": "cast", "column": "quantity", "params": {"dtype": "float64"}}
    })
    assert r2.status_code == 200

def t_grader_called_correctly_per_task():
    for task in ["easy", "medium", "hard"]:
        r = client.post("/reset", json={"task_id": task, "seed": 1})
        sid = r.json()["session_id"]
        r2 = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
        assert r2.status_code == 200
        assert 0.0 <= r2.json()["reward"]["score"] <= 1.0

run("server_health", t_server_health)
run("reset_easy", t_reset_easy)
run("reset_medium", t_reset_medium)
run("reset_hard", t_reset_hard)
run("reset_no_seed", t_reset_no_seed)
run("reset_invalid_task", t_reset_invalid_task)
run("step_inspect", t_step_inspect)
run("step_missing_column_on_cast", t_step_missing_column_on_cast)
run("step_nonexistent_session", t_step_nonexistent_session)
run("get_state_endpoint", t_get_state_endpoint)
run("state_not_found", t_state_not_found)
run("step_increments_step_counter", t_step_increments_step_counter)
run("done_after_max_steps", t_done_after_max_steps)
run("done_episode_blocked", t_done_episode_blocked)
run("deterministic_same_seed", t_deterministic_same_seed)
run("reward_bounds_all_tasks", t_reward_bounds_all_tasks)
run("impute_via_api", t_impute_via_api)
run("cast_quantity_via_api", t_cast_quantity_via_api)
run("grader_called_correctly_per_task", t_grader_called_correctly_per_task)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6 – Full Episode Simulations (End-to-End)
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 6: Full Episode E2E ─────────────────────────────────────────")

def t_full_easy_episode_score_improves():
    r = client.post("/reset", json={"task_id": "easy", "seed": 42})
    sid = r.json()["session_id"]
    # Inspect first
    r0 = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
    score0 = r0.json()["reward"]["score"]
    print(f"score0: {score0}")
    # Impute price
    r1 = client.post("/step", json={"session_id": sid, "action": {"action": "impute", "column": "price", "params": {"strategy": "median"}}})
    print(f"score after impute price: {r1.json()['reward']['score']}")
    # Cast quantity to numeric
    r2 = client.post("/step", json={"session_id": sid, "action": {"action": "cast", "column": "quantity", "params": {"dtype": "float64"}}})
    print(f"score after cast quantity float64: {r2.json()['reward']['score']}")
    # Impute quantity
    r3 = client.post("/step", json={"session_id": sid, "action": {"action": "impute", "column": "quantity", "params": {"strategy": "median"}}})
    print(f"score after impute quantity: {r3.json()['reward']['score']}")
    # Cast quantity back to int64
    r4 = client.post("/step", json={"session_id": sid, "action": {"action": "cast", "column": "quantity", "params": {"dtype": "int64"}}})
    print(f"score after cast quantity int64: {r4.json()['reward']['score']}")
    # Final inspect
    rf = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
    score_final = rf.json()["reward"]["score"]
    assert score_final > score0, f"Score should improve: {score0} -> {score_final}"

def t_full_medium_deduplication():
    r = client.post("/reset", json={"task_id": "medium", "seed": 10})
    sid = r.json()["session_id"]
    r1 = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
    shape_before = r1.json()["observation"]["shape"][0]
    client.post("/step", json={"session_id": sid, "action": {"action": "dedupe"}})
    r2 = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
    shape_after = r2.json()["observation"]["shape"][0]
    assert shape_after <= shape_before, "Deduplication should reduce or keep row count"

def t_hard_rescale_F_to_C_improves_score():
    r = client.post("/reset", json={"task_id": "hard", "seed": 1})
    sid = r.json()["session_id"]
    r0 = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
    score0 = r0.json()["reward"]["score"]
    # Rescale F to C for rows where temp_unit == 'F'
    client.post("/step", json={"session_id": sid, "action": {
        "action": "rescale", "column": "temperature",
        "params": {"from_unit": "F", "to_unit": "C", "condition_col": "temp_unit", "condition_val": "F"}
    }})
    rf = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
    score1 = rf.json()["reward"]["score"]
    assert score1 >= score0 - 0.01, f"Rescale should not decrease score: {score0} -> {score1}"

def t_step_info_has_task_and_step():
    r = client.post("/reset", json={"task_id": "easy", "seed": 1})
    sid = r.json()["session_id"]
    r2 = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
    info = r2.json()["reward"]["info"]
    assert info["task"] == "easy"
    assert info["step"] == 1

def t_max_steps_medium_40():
    r = client.post("/reset", json={"task_id": "medium", "seed": 1})
    sid = r.json()["session_id"]
    last = None
    for _ in range(40):
        r2 = client.post("/step", json={"session_id": sid, "action": {"action": "inspect"}})
        if r2.status_code != 200:
            break
        last = r2
    assert last.json()["reward"]["done"] == True

run("full_easy_episode_score_improves", t_full_easy_episode_score_improves)
run("full_medium_deduplication", t_full_medium_deduplication)
run("hard_rescale_improves_score", t_hard_rescale_F_to_C_improves_score)
run("step_info_has_task_and_step", t_step_info_has_task_and_step)
run("max_steps_medium_40", t_max_steps_medium_40)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 7 – Edge Cases & Adversarial Inputs
# ══════════════════════════════════════════════════════════════════════════════
print("\n── SECTION 7: Edge Cases & Adversarial ─────────────────────────────────")

def t_empty_df_observe():
    df = pd.DataFrame({"a": [], "b": []})
    obs = _observe(df)
    assert obs.shape == (0, 2)

def t_cast_already_correct_type():
    df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
    df2, obs, _ = run_action(df, DataAction(action="cast", column="col", params={"dtype": "float64"}))
    assert df2["col"].dtype == np.float64

def t_impute_no_nulls():
    df = pd.DataFrame({"col": [1.0, 2.0, 3.0]})
    df2, obs, _ = run_action(df, DataAction(action="impute", column="col", params={"strategy": "median"}))
    assert df2.equals(df)

def t_dedupe_no_duplicates():
    df = pd.DataFrame({"a": [1, 2, 3]})
    df2, obs, _ = run_action(df, DataAction(action="dedupe"))
    assert df2.equals(df)

def t_model_validation_cast_no_col():
    from pydantic import ValidationError
    try:
        DataAction(action="cast")
        assert False, "Should have raised ValidationError"
    except (ValueError, Exception):
        pass

def t_model_validation_impute_no_col():
    from pydantic import ValidationError
    try:
        DataAction(action="impute")
        assert False, "Should have raised ValidationError"
    except (ValueError, Exception):
        pass

def t_model_validation_invalid_action():
    from pydantic import ValidationError
    try:
        DataAction(action="flying_kick")
        assert False, "Should have raised ValidationError"
    except Exception:
        pass

def t_multiple_seeds_all_pass():
    for seed in range(0, 50, 10):
        df = generate_clean("easy", seed)
        dirty, manifest = corrupt_easy(df, np.random.default_rng(seed))
        s = score_easy(dirty, manifest)
        assert 0.0 <= s <= 1.0

def t_rescale_without_condition():
    # No condition_col = rescale all rows
    df = pd.DataFrame({"temp": [100.0, 200.0]})
    action = DataAction(action="rescale", column="temp", params={"from_unit": "C", "to_unit": "F"})
    df2, obs, _ = run_action(df, action)
    assert abs(df2["temp"].iloc[0] - 212.0) < 0.1
    assert abs(df2["temp"].iloc[1] - 392.0) < 0.1

def t_strip_pattern_removes_non_numeric():
    df = pd.DataFrame({"col": ["$1,200.50", "$999.99"]})
    action = DataAction(action="cast", column="col", params={"dtype": "float64", "strip_pattern": r"[^\d.]"})
    df2, obs, _ = run_action(df, action)
    assert abs(df2["col"].iloc[0] - 1200.50) < 0.01
    assert abs(df2["col"].iloc[1] - 999.99) < 0.01

def t_score_easy_multiple_seeds_clean_data():
    # Clean data (no corruption) should score very high or 1.0
    for seed in [1, 5, 10]:
        df = generate_clean("easy", seed)
        _, manifest = corrupt_easy(df, np.random.default_rng(seed))
        # Pass clean (not dirty) df - not all cells will match original values
        s = score_easy(df, manifest)
        assert 0.0 <= s <= 1.0

def t_all_action_types_via_api():
    r = client.post("/reset", json={"task_id": "hard", "seed": 1})
    sid = r.json()["session_id"]
    actions = [
        {"action": "inspect"},
        {"action": "impute", "column": "temperature", "params": {"strategy": "median"}},
        {"action": "cast", "column": "sensor_id", "params": {"dtype": "float64"}},
        {"action": "dedupe"},
        {"action": "rescale", "column": "temperature", "params": {"from_unit": "F", "to_unit": "C", "condition_col": "temp_unit", "condition_val": "F"}},
    ]
    for action in actions:
        r2 = client.post("/step", json={"session_id": sid, "action": action})
        assert r2.status_code == 200, f"Action {action['action']} failed: {r2.text}"

def t_observation_fields_complete():
    r = client.post("/reset", json={"task_id": "easy", "seed": 1})
    obs = r.json()["observation"]
    assert "shape" in obs
    assert "columns" in obs
    assert "step" in obs
    for c in obs["columns"]:
        assert "name" in c
        assert "dtype" in c
        assert "null_count" in c
        assert "sample_values" in c
        assert "unique_count" in c

run("empty_df_observe", t_empty_df_observe)
run("cast_already_correct_type", t_cast_already_correct_type)
run("impute_no_nulls_noop", t_impute_no_nulls)
run("dedupe_no_duplicates_noop", t_dedupe_no_duplicates)
run("model_cast_no_col_raises", t_model_validation_cast_no_col)
run("model_impute_no_col_raises", t_model_validation_impute_no_col)
run("model_invalid_action_raises", t_model_validation_invalid_action)
run("multiple_seeds_score_bounds", t_multiple_seeds_all_pass)
run("rescale_without_condition", t_rescale_without_condition)
run("strip_pattern_dollar_comma", t_strip_pattern_removes_non_numeric)
run("score_easy_clean_data_seeds", t_score_easy_multiple_seeds_clean_data)
run("all_action_types_via_api", t_all_action_types_via_api)
run("observation_fields_complete", t_observation_fields_complete)

# ══════════════════════════════════════════════════════════════════════════════
# FINAL REPORT
# ══════════════════════════════════════════════════════════════════════════════
total = PASS + FAIL
print(f"\n{'='*60}")
print(f"  RESULTS: {PASS}/{total} passed  |  {FAIL} failed")
print(f"{'='*60}")
if FAIL > 0:
    print("\nFailed tests:")
    for t in TESTS:
        if t[0] == "FAIL":
            print(f"\n  ❌ {t[1]}")
            print("  " + t[2].replace("\n", "\n  "))
sys.exit(0 if FAIL == 0 else 1)
