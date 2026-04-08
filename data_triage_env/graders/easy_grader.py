import pandas as pd
import numpy as np
from data_triage_env.engine.corruptor import GroundTruthManifest

def score(agent_df: pd.DataFrame, manifest: GroundTruthManifest, penalty_multiplier: float = 10.0, max_score_if_trap_broken: float = 1.0) -> float:
    clean = manifest.clean_df
    total_bugs = 0
    fixed_bugs = 0
    broken_clean = 0
    trap_broken = False

    for record in manifest.records:
        if record.corruption_type == "trap":
            # Penalise if agent modified the trap column
            trap_col = record.column
            if trap_col in agent_df.columns and trap_col in clean.columns:
                n = min(len(agent_df), len(clean))
                # Check for any change in the trap column
                changed_mask = (agent_df[trap_col].iloc[:n] != clean[trap_col].iloc[:n])
                if changed_mask.any():
                    trap_broken = True
                    broken_clean += int(changed_mask.sum())
            continue

        if record.corruption_type == "duplicate":
            clean_len = len(clean)
            # A bit simplistic: just check if ANY duplicates exist in agent_df
            agent_deduped = not any(agent_df.duplicated())
            total_bugs += 1
            if agent_deduped:
                fixed_bugs += 1
            continue

        col = record.column
        if col not in agent_df.columns or col not in clean.columns:
            total_bugs += len(record.row_indices)
            continue

        for idx, orig_val in zip(record.row_indices, record.original_values):
            if idx >= len(agent_df):
                continue
            total_bugs += 1
            agent_val = agent_df.iloc[idx][col] if idx < len(agent_df) else np.nan
            clean_val = clean.iloc[idx][col] if idx < len(clean) else np.nan
            # If the expected fix involves imputation, the original data is irrecoverable.
            # We consider it fixed if the agent replaced it with a valid numeric/non-empty value.
            if record.corruption_type == "null" or "median" in record.expected_fix.lower():
                if not pd.isna(agent_val) and str(agent_val).strip() not in ["", "N/A", "nan", "NaN"]:
                    fixed_bugs += 1
                continue

            # Fixed = agent_val is close to clean_val AND not NaN
            try:
                if pd.isna(agent_val):
                    pass  # not fixed
                elif abs(float(agent_val) - float(clean_val)) < 0.01:
                    fixed_bugs += 1
            except (ValueError, TypeError):
                if str(agent_val).strip() == str(clean_val).strip():
                    fixed_bugs += 1

    # Check for clean-cell breakage outside corrupted positions
    corrupted_positions = {(r.column, i) for r in manifest.records for i in r.row_indices}
    for col in clean.columns:
        if col not in agent_df.columns:
            continue
        n = min(len(agent_df), len(clean))
        for i in range(n):
            if (col, i) in corrupted_positions:
                continue
            c_val = clean.iloc[i][col]
            a_val = agent_df.iloc[i][col]
            try:
                same = pd.isna(c_val) and pd.isna(a_val)
                if not same:
                    try:
                        # Prevent unfair penalty when an int correctly becomes a float during imputation steps
                        same = abs(float(c_val) - float(a_val)) < 0.01
                    except (ValueError, TypeError):
                        same = str(c_val).strip() == str(a_val).strip()
            except Exception:
                # Fallback to string comparison
                same = str(c_val) == str(a_val)
            if not same:
                broken_clean += 1

    if total_bugs == 0:
        return 1.0 if not trap_broken else min(1.0, max_score_if_trap_broken)

    raw = fixed_bugs / total_bugs
    penalty = min(1.0, (broken_clean * penalty_multiplier) / max(total_bugs, 1))
    final_score = float(max(0.0, round(raw - penalty, 4)))

    if trap_broken:
        final_score = min(final_score, max_score_if_trap_broken)

    return final_score
