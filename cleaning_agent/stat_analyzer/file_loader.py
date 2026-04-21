"""
file_loader.py
==============
Handles CSV / Excel ingestion and keeps parsed DataFrames alive in memory
for the duration of a session, keyed by a UUID.

Design notes
------------
* LRU-style eviction: when the store reaches MAX_SIZE, the oldest entry is
  dropped so memory stays bounded.
* No temp files are written to disk.
* Thread-safety is not required for single-worker deployments; if you scale
  to multiple workers, swap _STORE for a Redis / memcached backend.
"""

from __future__ import annotations

import io
import uuid
from collections import OrderedDict

import pandas as pd

# ---------------------------------------------------------------------------
# In-memory store
# ---------------------------------------------------------------------------

_STORE: OrderedDict[str, pd.DataFrame] = OrderedDict()
MAX_STORE_SIZE = 100          # evict oldest when this limit is exceeded


def _evict_if_full() -> None:
    while len(_STORE) >= MAX_STORE_SIZE:
        _STORE.popitem(last=False)   # remove oldest (FIFO)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_file(content: bytes, filename: str) -> tuple[str, pd.DataFrame]:
    """
    Parse *content* as CSV or Excel, persist the DataFrame, return (file_id, df).

    Raises
    ------
    ValueError  – unsupported extension or parse failure
    """
    name = (filename or "").strip().lower()

    try:
        if name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(content))
        elif name.endswith(".xlsx"):
            df = pd.read_excel(io.BytesIO(content), engine="openpyxl")
        elif name.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(content), engine="xlrd")
        else:
            raise ValueError(
                f"Unsupported file type '{filename}'. "
                "Please upload a .csv, .xlsx, or .xls file."
            )
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not parse '{filename}': {exc}") from exc

    if df.empty:
        raise ValueError("The uploaded file contains no data rows.")

    file_id = str(uuid.uuid4())
    _evict_if_full()
    _STORE[file_id] = df.copy()
    return file_id, df


def get_dataframe(file_id: str) -> pd.DataFrame:
    """
    Retrieve a previously uploaded DataFrame by *file_id*.

    Raises
    ------
    KeyError – file_id not found (expired or never uploaded)
    """
    df = _STORE.get(file_id)
    if df is None:
        raise KeyError(
            f"File '{file_id}' not found. "
            "The session may have expired — please re-upload your file."
        )
    return df


def file_exists(file_id: str) -> bool:
    return file_id in _STORE
