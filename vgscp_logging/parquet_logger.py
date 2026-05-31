"""Parquet per-sample logger.

Accumulates row dicts (or columnar arrays) and writes a single parquet file per run/split.
JSON-typed columns (per-label sets, concept lists) are serialized to strings on write.
"""
from __future__ import annotations

import json
import os
from typing import Any, Optional

import pandas as pd

_JSON_COLUMN_HINTS = (
    "_concepts",
    "in_set_",
    "merlin",
    "morgana",
)


def _is_json_col(col: str) -> bool:
    return any(h in col for h in _JSON_COLUMN_HINTS)


def _serialize_json_cols(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if _is_json_col(col):
            df[col] = df[col].map(
                lambda v: v if isinstance(v, str) or v is None else json.dumps(v)
            )
    return df


class ParquetLogger:
    """Per-sample logger. Build a DataFrame from a dict of columns or a list of rows."""

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def write_frame(self, df: pd.DataFrame, name: str) -> str:
        df = _serialize_json_cols(df.copy())
        path = os.path.join(self.out_dir, f"{name}.parquet")
        df.to_parquet(path, engine="pyarrow", index=False)
        return path

    def write_columns(self, columns: dict[str, Any], name: str) -> str:
        return self.write_frame(pd.DataFrame(columns), name)

    def write_rows(self, rows: list[dict], name: str) -> str:
        return self.write_frame(pd.DataFrame(rows), name)


def load_run_logs(run_dir: str, split: Optional[str] = None) -> pd.DataFrame:
    """Load and concatenate per-split parquet logs from a run directory."""
    import glob

    pattern = os.path.join(run_dir, "logs", "*.parquet")
    frames = []
    for p in sorted(glob.glob(pattern)):
        if split is not None and split not in os.path.basename(p):
            continue
        frames.append(pd.read_parquet(p))
    if not frames:
        raise FileNotFoundError(f"no parquet logs in {pattern}")
    return pd.concat(frames, ignore_index=True)
