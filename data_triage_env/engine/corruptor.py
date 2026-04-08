import copy
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass, field

def _fmt_day(dt):
    if sys.platform == "win32":
        return dt.strftime("%#d %b %Y")
    return dt.strftime("%-d %b %Y")

@dataclass
class CorruptionRecord:
    corruption_type: str   # null | type_mismatch | duplicate | unit_mismatch | date_format
    column: str
    row_indices: list[int]
    original_values: list
    expected_fix: str      # human-readable description of the correct fix

@dataclass
class GroundTruthManifest:
    records: list[CorruptionRecord] = field(default_factory=list)
    clean_df: pd.DataFrame = None  # snapshot of the clean df before corruption

def corrupt_easy(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, GroundTruthManifest]:
    manifest = GroundTruthManifest(clean_df=df.copy())
    dirty = df.copy()

    # Bug 1: Nulls in 'price' column (10% of rows)
    null_idx = rng.choice(len(dirty), size=len(dirty) // 10, replace=False).tolist()
    manifest.records.append(CorruptionRecord(
        corruption_type="null",
        column="price",
        row_indices=null_idx,
        original_values=dirty.loc[null_idx, "price"].tolist(),
        expected_fix="impute with median of column"
    ))
    dirty.loc[null_idx, "price"] = np.nan

    # Bug 2: Type mismatch in 'quantity' — inject "N/A" strings into int column
    mismatch_idx = rng.choice(len(dirty), size=15, replace=False).tolist()
    dirty = dirty.astype({"quantity": object})
    for i in mismatch_idx:
        dirty.at[i, "quantity"] = "N/A"
    manifest.records.append(CorruptionRecord(
        corruption_type="type_mismatch",
        column="quantity",
        row_indices=mismatch_idx,
        original_values=[df.at[i, "quantity"] for i in mismatch_idx],
        expected_fix="cast column to int64, replace non-numeric with median"
    ))

    return dirty, manifest

def corrupt_medium(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, GroundTruthManifest]:
    manifest = GroundTruthManifest(clean_df=df.copy())
    dirty = df.copy()

    # Bug 1: Nulls in 'revenue'
    null_idx = rng.choice(len(dirty), size=40, replace=False).tolist()
    manifest.records.append(CorruptionRecord(
        corruption_type="null", column="revenue",
        row_indices=null_idx,
        original_values=dirty.loc[null_idx, "revenue"].tolist(),
        expected_fix="impute with median"
    ))
    dirty.loc[null_idx, "revenue"] = np.nan

    # Bug 2: Type mismatch — revenue has "1,200 USD" strings injected
    mismatch_idx = rng.choice([i for i in range(len(dirty)) if i not in null_idx], size=20, replace=False).tolist()
    dirty = dirty.astype({"revenue": object})
    for i in mismatch_idx:
        dirty.at[i, "revenue"] = f"{dirty.at[i, 'revenue']:,.0f} USD"
    manifest.records.append(CorruptionRecord(
        corruption_type="type_mismatch", column="revenue",
        row_indices=mismatch_idx,
        original_values=[df.at[i, "revenue"] for i in mismatch_idx],
        expected_fix="strip ' USD' and commas, cast to float64"
    ))

    # Bug 3: Duplicate customer_name rows (inject 15 exact duplicates)
    dup_source_idx = rng.choice(len(dirty), size=15, replace=False).tolist()
    dup_rows = dirty.iloc[dup_source_idx].copy()
    dirty = pd.concat([dirty, dup_rows], ignore_index=True)
    manifest.records.append(CorruptionRecord(
        corruption_type="duplicate", column="customer_name",
        row_indices=list(range(len(df), len(dirty))),
        original_values=[],
        expected_fix="drop exact duplicate rows, reset index"
    ))

    # Bug 4: Mixed date formats in 'order_date'
    fmt_idx = rng.choice(len(df), size=30, replace=False).tolist()
    for i in fmt_idx:
        original = dirty.at[i, "order_date"]
        try:
            dt = pd.Timestamp(original)
            dirty.at[i, "order_date"] = _fmt_day(dt)  # e.g., "5 Jan 2024"
        except Exception:
            pass
    manifest.records.append(CorruptionRecord(
        corruption_type="date_format", column="order_date",
        row_indices=fmt_idx,
        original_values=[df.at[i, "order_date"] for i in fmt_idx],
        expected_fix="parse all date strings to ISO 8601 YYYY-MM-DD"
    ))

    return dirty, manifest

def corrupt_hard(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, GroundTruthManifest]:
    manifest = GroundTruthManifest(clean_df=df.copy())
    dirty = df.copy()

    # Bug 1: Null temperatures
    null_idx = rng.choice(len(dirty), size=60, replace=False).tolist()
    manifest.records.append(CorruptionRecord(
        corruption_type="null", column="temperature",
        row_indices=null_idx,
        original_values=dirty.loc[null_idx, "temperature"].tolist(),
        expected_fix="impute with median"
    ))
    dirty.loc[null_idx, "temperature"] = np.nan

    # Bug 2: Unit mismatch — REAL mismatch: 80 rows converted to Fahrenheit BUT temp_unit still says "C"
    real_mismatch_idx = rng.choice([i for i in range(len(dirty)) if i not in null_idx], size=80, replace=False).tolist()
    for i in real_mismatch_idx:
        c_val = dirty.at[i, "temperature"]
        if pd.notna(c_val):
            dirty.at[i, "temperature"] = round(c_val * 9/5 + 32, 2)
            dirty.at[i, "temp_unit"] = "F"  # unit IS updated — agent must read both columns
    manifest.records.append(CorruptionRecord(
        corruption_type="unit_mismatch", column="temperature",
        row_indices=real_mismatch_idx,
        original_values=[df.at[i, "temperature"] for i in real_mismatch_idx],
        expected_fix="where temp_unit=='F', convert (val-32)*5/9, set temp_unit='C'"
    ))

    # Bug 3: TRAP — 'pressure' column looks suspicious (values 1-5) but is CORRECT — do NOT rescale
    # Manifest records this with expected_fix = "no action required"
    manifest.records.append(CorruptionRecord(
        corruption_type="trap", column="pressure",
        row_indices=[],
        original_values=[],
        expected_fix="no action required — pressure values are correct in their scale"
    ))

    # Bug 4: Type mismatch in sensor_id
    mismatch_idx = rng.choice(len(dirty), size=30, replace=False).tolist()
    dirty = dirty.astype({"sensor_id": object})
    for i in mismatch_idx:
        dirty.at[i, "sensor_id"] = f"SENSOR_{dirty.at[i, 'sensor_id']}"
    manifest.records.append(CorruptionRecord(
        corruption_type="type_mismatch", column="sensor_id",
        row_indices=mismatch_idx,
        original_values=[df.at[i, "sensor_id"] for i in mismatch_idx],
        expected_fix="strip 'SENSOR_' prefix, cast to int64"
    ))

    # Bug 5: Date format issues in 'timestamp'
    date_idx = rng.choice(len(df), size=40, replace=False).tolist()
    for i in date_idx:
        dt = pd.Timestamp(dirty.at[i, "timestamp"])
        # Mix formats: "1 Jan 2024", "2024/01/01", "01-01-2024"
        choice = rng.choice([0, 1, 2])
        if choice == 0:
            dirty.at[i, "timestamp"] = _fmt_day(dt)
        elif choice == 1:
            dirty.at[i, "timestamp"] = dt.strftime("%Y/%m/%d")
        else:
            dirty.at[i, "timestamp"] = dt.strftime("%d-%m-%Y")
    manifest.records.append(CorruptionRecord(
        corruption_type="date_format", column="timestamp",
        row_indices=date_idx,
        original_values=[df.at[i, "timestamp"] for i in date_idx],
        expected_fix="convert to ISO 8601 YYYY-MM-DDTHH:MM:SS"
    ))

    # Bug 6: Duplicate rows
    dup_idx = rng.choice(len(df), size=25, replace=False).tolist()
    dup_rows = dirty.iloc[dup_idx].copy()
    dirty = pd.concat([dirty, dup_rows], ignore_index=True)
    manifest.records.append(CorruptionRecord(
        corruption_type="duplicate", column="ALL",
        row_indices=list(range(len(df), len(dirty))),
        original_values=[],
        expected_fix="drop exact duplicate rows"
    ))

    return dirty, manifest

CORRUPT_FNS = {
    "easy": corrupt_easy,
    "medium": corrupt_medium,
    "hard": corrupt_hard,
}
