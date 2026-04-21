"""
routes.py
=========
FastAPI router for the Statistical Analyzer feature.

Endpoints
---------
POST /analyzer/upload
    Accept CSV / Excel, return file_id + column list with inferred types.

POST /analyzer/analyze-column
    Accept file_id + column name, return full statistical profile.
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from .file_loader      import load_file, get_dataframe
from .column_detector  import get_column_info, detect_type
from .numerical_analysis   import analyze as analyze_numerical
from .categorical_analysis import analyze as analyze_categorical

router = APIRouter(prefix="/analyzer", tags=["Statistical Analyzer"])


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    file_id: str
    column:  str


# ---------------------------------------------------------------------------
# Endpoint 1 — Upload file
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel file.

    Returns the file_id (needed for subsequent /analyze-column calls) and a
    list of columns with their inferred types.

    Example response
    ----------------
    {
        "file_id": "3f2a...",
        "row_count": 500,
        "col_count": 12,
        "columns": [
            {"name": "Age",       "type": "numerical"},
            {"name": "City",      "type": "categorical"},
            {"name": "Join_Date", "type": "date"}
        ]
    }
    """
    filename = file.filename or ""
    ext = filename.lower().split(".")[-1] if "." in filename else ""
    if ext not in {"csv", "xlsx", "xls"}:
        raise HTTPException(
            status_code=400,
            detail="Only .csv, .xlsx, and .xls files are accepted.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    try:
        file_id, df = load_file(content, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"File processing error: {exc}")

    return {
        "file_id":   file_id,
        "row_count": len(df),
        "col_count": len(df.columns),
        "columns":   get_column_info(df),
    }


# ---------------------------------------------------------------------------
# Endpoint 2 — Analyze a single column
# ---------------------------------------------------------------------------

@router.post("/analyze-column")
def analyze_column(req: AnalyzeRequest):
    """
    Analyze one column from a previously uploaded file.

    Input
    -----
    { "file_id": "...", "column": "Age" }

    Returns a full statistical profile whose shape depends on the column type:
    * numerical  → mean, median, std, quartiles, skewness, outliers, normality test, …
    * categorical → unique count, mode, top-10 frequencies, …
    * date        → treated as categorical (frequency of date strings)

    Raises 404 if the file_id or column name is not found.
    """
    # Retrieve DataFrame
    try:
        df = get_dataframe(req.file_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    # Validate column
    if req.column not in df.columns:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Column '{req.column}' not found. "
                f"Available columns: {list(df.columns)}"
            ),
        )

    series   = df[req.column]
    col_type = detect_type(series)

    if col_type == "numerical":
        return analyze_numerical(series, req.column)

    # date and categorical both use the categorical analyser;
    # the col_type field in the response tells the frontend which variant it is.
    return analyze_categorical(series, req.column, col_type=col_type)
