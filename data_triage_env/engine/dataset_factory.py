import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class DatasetSpec:
    n_rows: int
    columns: list[dict]  # name, dtype, value_range

TASK_SPECS: dict[str, DatasetSpec] = {
    "easy": DatasetSpec(
        n_rows=200,
        columns=[
            {"name": "customer_id", "dtype": "int",    "range": (1000, 9999)},
            {"name": "price",       "dtype": "float",  "range": (5.0, 500.0)},
            {"name": "quantity",    "dtype": "int",    "range": (1, 100)},
            {"name": "region",      "dtype": "category","values": ["North","South","East","West"]},
        ]
    ),
    "medium": DatasetSpec(
        n_rows=500,
        columns=[
            {"name": "customer_name","dtype": "str",    "values": ["Alice","Bob","Carol","Dave","Eve"]},
            {"name": "revenue",      "dtype": "float",  "range": (100.0, 10000.0)},
            {"name": "order_date",   "dtype": "date",   "format": "%Y-%m-%d"},
            {"name": "product_code", "dtype": "str",    "values": ["A1","B2","C3","D4"]},
            {"name": "units",        "dtype": "int",    "range": (1, 50)},
            {"name": "discount",     "dtype": "float",  "range": (0.0, 0.5)},
        ]
    ),
    "hard": DatasetSpec(
        n_rows=800,
        columns=[
            {"name": "sensor_id",    "dtype": "int",    "range": (1, 200)},
            {"name": "temperature",  "dtype": "float",  "range": (15.0, 40.0)},  # Celsius
            {"name": "temp_unit",    "dtype": "str",    "values": ["C"]},        # all Celsius — trap column
            {"name": "pressure",     "dtype": "float",  "range": (1.0, 5.0)},
            {"name": "humidity",     "dtype": "float",  "range": (20.0, 90.0)},
            {"name": "timestamp",    "dtype": "date",   "format": "%Y-%m-%dT%H:%M:%S"},
            {"name": "station_name", "dtype": "str",    "values": ["Alpha","Bravo","Charlie","Delta","Echo"]},
            {"name": "reading_valid","dtype": "bool",   "values": [True, False]},
        ]
    )
}

def generate_clean(spec_name: str, seed: int) -> pd.DataFrame:
    """Generate a clean DataFrame from the given spec with fixed seed."""
    rng = np.random.default_rng(seed)
    spec = TASK_SPECS[spec_name]
    data: dict[str, list] = {}
    for col in spec.columns:
        n = spec.n_rows
        dtype = col["dtype"]
        if dtype == "int":
            lo, hi = col["range"]
            data[col["name"]] = rng.integers(lo, hi + 1, n).tolist()
        elif dtype == "float":
            lo, hi = col["range"]
            vals = rng.uniform(lo, hi, n)
            data[col["name"]] = np.round(vals, 2).tolist()
        elif dtype in ("str", "category"):
            choices = col["values"]
            idx = rng.integers(0, len(choices), n)
            data[col["name"]] = [choices[i] for i in idx]
        elif dtype == "date":
            # Generate sequential timestamps
            base = pd.Timestamp("2024-01-01")
            offsets = rng.integers(0, 365 * 24 * 3600, n)
            dates = [base + pd.Timedelta(seconds=int(s)) for s in offsets]
            fmt = col.get("format", "%Y-%m-%d")
            data[col["name"]] = [d.strftime(fmt) for d in dates]
        elif dtype == "bool":
            data[col["name"]] = rng.choice(col["values"], n).tolist()
    return pd.DataFrame(data)
