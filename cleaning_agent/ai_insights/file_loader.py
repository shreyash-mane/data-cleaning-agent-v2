"""
file_loader.py
==============
In-memory file store for the AI Insights Engine.

Supports CSV, XLSX, XLS upload with UUID-keyed storage and LRU eviction.
Also stores detected insights keyed by file_id so detail lookups are fast.
"""

from __future__ import annotations

from collections import OrderedDict
import io
import uuid

import pandas as pd

MAX_STORE_SIZE = 100

_DF_STORE: OrderedDict = OrderedDict()      # file_id -> DataFrame
_INSIGHT_STORE: dict[str, list] = {}        # file_id -> ranked insights list


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_file(content: bytes, filename: str) -> tuple[str, pd.DataFrame]:
    """Parse bytes into a DataFrame and store it. Returns (file_id, df)."""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        if ext == "csv":
            df = pd.read_csv(io.BytesIO(content))
        elif ext == "xlsx":
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        elif ext == "xls":
            df = pd.read_excel(io.BytesIO(content), engine="xlrd")
        else:
            raise ValueError(f"Unsupported file type '.{ext}'. Use .csv, .xlsx, or .xls.")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse file: {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded file is empty.")
    if len(df.columns) < 2:
        raise ValueError("Dataset must have at least 2 columns for insight detection.")

    _evict_if_full()
    file_id = str(uuid.uuid4())
    _DF_STORE[file_id] = df
    return file_id, df


def get_dataframe(file_id: str) -> pd.DataFrame:
    if file_id not in _DF_STORE:
        raise KeyError(f"File '{file_id}' not found or expired. Please re-upload.")
    _DF_STORE.move_to_end(file_id)
    return _DF_STORE[file_id]


def store_insights(file_id: str, insights: list) -> None:
    _INSIGHT_STORE[file_id] = insights


def get_insights(file_id: str) -> list:
    if file_id not in _INSIGHT_STORE:
        raise KeyError(
            f"No insights for file '{file_id}'. Call /ai-insights/detect first."
        )
    return _INSIGHT_STORE[file_id]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _evict_if_full() -> None:
    while len(_DF_STORE) >= MAX_STORE_SIZE:
        evicted_id, _ = _DF_STORE.popitem(last=False)
        _INSIGHT_STORE.pop(evicted_id, None)
