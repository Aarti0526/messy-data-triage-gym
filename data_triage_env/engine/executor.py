import re
import pandas as pd
import numpy as np
from data_triage_env.models import DataAction, DataObservation, ColumnStats

def run_action(df: pd.DataFrame, action: DataAction) -> tuple[pd.DataFrame, DataObservation, str]:
    """
    Execute one action on df. Returns (new_df, observation, message).
    Raises ValueError for invalid actions. Never modifies df in-place.
    """
    df = df.copy()
    msg = ""

    if action.action == "inspect":
        # No mutation — just observe
        pass

    elif action.action == "cast":
        col = action.column
        target_type = action.params.get("dtype", "float64")
        strip_pattern = action.params.get("strip_pattern", None)  # e.g. r"[^\d.]"
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
        if strip_pattern:
            df[col] = df[col].astype(str).str.replace(strip_pattern, "", regex=True)
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(target_type)
        msg = f"Cast '{col}' to {target_type}"

    elif action.action == "impute":
        col = action.column
        strategy = action.params.get("strategy", "median")
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        if strategy == "median":
            fill_val = numeric_col.median()
        elif strategy == "mean":
            fill_val = numeric_col.mean()
        elif strategy == "mode":
            fill_val = df[col].mode().iloc[0]
        elif strategy == "constant":
            fill_val = action.params.get("value", 0)
        else:
            raise ValueError(f"Unknown impute strategy: {strategy}")
        df[col] = df[col].fillna(fill_val)
        msg = f"Imputed '{col}' with {strategy}={fill_val}"

    elif action.action == "dedupe":
        subset = action.params.get("subset", None)
        keep = action.params.get("keep", "first")
        before = len(df)
        df = df.drop_duplicates(subset=subset, keep=keep).reset_index(drop=True)
        msg = f"Removed {before - len(df)} duplicate rows"

    elif action.action == "rescale":
        col = action.column
        from_unit = action.params.get("from_unit", "F")
        to_unit = action.params.get("to_unit", "C")
        condition_col = action.params.get("condition_col", None)
        condition_val = action.params.get("condition_val", None)
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found")
        mask = pd.Series([True] * len(df), index=df.index)
        if condition_col and condition_val is not None:
            mask = df[condition_col] == condition_val
        if from_unit == "F" and to_unit == "C":
            df.loc[mask, col] = ((df.loc[mask, col] - 32) * 5 / 9).round(2)
        elif from_unit == "C" and to_unit == "F":
            df.loc[mask, col] = ((df.loc[mask, col] * 9 / 5) + 32).round(2)
        if condition_col:
            df.loc[mask, condition_col] = to_unit
        msg = f"Rescaled '{col}' from {from_unit} to {to_unit}"

    obs = _observe(df)
    return df, obs, msg

def _observe(df: pd.DataFrame) -> DataObservation:
    stats = []
    for col in df.columns:
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")
            sample = df[col].dropna().head(5).tolist()
        except Exception:
            sample = []
        stats.append(ColumnStats(
            name=col,
            dtype=str(df[col].dtype),
            null_count=int(df[col].isna().sum()),
            sample_values=sample,
            unique_count=int(df[col].nunique()),
        ))
    return DataObservation(
        step=0,
        columns=stats,
        shape=(len(df), len(df.columns)),
    )
