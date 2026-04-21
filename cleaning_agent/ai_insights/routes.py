"""
routes.py
=========
FastAPI router for the AI Insights Engine.

Endpoints
---------
POST /ai-insights/upload          — upload CSV/Excel, return file_id + column metadata
POST /ai-insights/detect          — run insight detection, return ranked top insights
POST /ai-insights/insight-details — return full detail for one insight (chart + explanation)
POST /ai-insights/chart-preview   — build custom chart data from an arbitrary config
"""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from .file_loader       import load_file, get_dataframe, store_insights, get_insights
from .column_inference  import get_column_types, get_column_info
from .insight_detector  import detect_all
from .insight_ranker    import rank_insights
from .chart_builder     import build_chart_data
from .explanation_engine import build_explanation, get_reliability_notes

router = APIRouter(prefix="/ai-insights", tags=["AI Insights Engine"])


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class DetectRequest(BaseModel):
    file_id: str


class InsightDetailsRequest(BaseModel):
    file_id:    str
    insight_id: str


class ChartPreviewRequest(BaseModel):
    file_id:     str
    chart_type:  str
    x:           str
    y:           str | None = None
    aggregation: str | None = None
    group_by:    str | None = None
    filters:     list[dict] | None = None
    sort_by:     str | None = None
    sort_asc:    bool = True
    top_n:       int | None = None
    title:       str | None = None


# ---------------------------------------------------------------------------
# Endpoint 1 — Upload
# ---------------------------------------------------------------------------

@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a CSV or Excel file.

    Returns file_id + column metadata + basic dataset summary.
    The file_id is required for all subsequent calls.
    """
    filename = file.filename or ""
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext not in {"csv", "xlsx", "xls"}:
        raise HTTPException(400, "Only .csv, .xlsx, and .xls files are accepted.")

    content = await file.read()
    if not content:
        raise HTTPException(400, "Uploaded file is empty.")

    try:
        file_id, df = load_file(content, filename)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(500, f"File processing error: {exc}")

    col_types = get_column_types(df)

    return {
        "file_id": file_id,
        "columns": get_column_info(df, col_types),
        "summary": {"rows": len(df), "columns": len(df.columns)},
    }


# ---------------------------------------------------------------------------
# Endpoint 2 — Detect Insights
# ---------------------------------------------------------------------------

@router.post("/detect")
def detect_insights(req: DetectRequest):
    """
    Run the full insight detection pipeline on a previously uploaded file.

    Returns the top 10 ranked insights with slim metadata (no chart data yet).
    Call /insight-details to fetch chart data + full explanation for one insight.
    """
    try:
        df = get_dataframe(req.file_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    col_types    = get_column_types(df)
    raw_insights = detect_all(df, col_types)
    ranked       = rank_insights(raw_insights, top_n=10)

    store_insights(req.file_id, ranked)

    slim = [
        {
            "insight_id":        ins["insight_id"],
            "type":              ins["type"],
            "title":             ins["title"],
            "summary":           ins["summary"],
            "columns":           ins["columns"],
            "strength":          ins["strength"],
            "confidence":        ins["confidence"],
            "score":             ins["score"],
            "recommended_chart": ins.get("recommended_chart"),
        }
        for ins in ranked
    ]

    return {
        "file_id":     req.file_id,
        "total_found": len(raw_insights),
        "total_shown": len(ranked),
        "insights":    slim,
    }


# ---------------------------------------------------------------------------
# Endpoint 3 — Insight Details
# ---------------------------------------------------------------------------

@router.post("/insight-details")
def insight_details(req: InsightDetailsRequest):
    """
    Return the full detail for a single insight identified by insight_id.

    Includes:
    • explanation text (rule-based, or LLM if wired up)
    • chart data ready for recharts
    • reliability notes / caveats
    • customization_options (available columns, chart types, aggregations)
    """
    try:
        df = get_dataframe(req.file_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    try:
        insights = get_insights(req.file_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    ins = next((i for i in insights if i["insight_id"] == req.insight_id), None)
    if ins is None:
        raise HTTPException(404, f"Insight '{req.insight_id}' not found.")

    # Build chart data from the recommended config
    rec = ins.get("recommended_chart") or {}
    if rec:
        chart_result = build_chart_data(
            df=df,
            chart_type=rec.get("chart_type", "bar"),
            x=rec.get("x", ""),
            y=rec.get("y"),
            aggregation=rec.get("aggregation"),
            group_by=rec.get("group_by"),
            title=rec.get("title"),
        )
    else:
        chart_result = {"data": [], "config": {}, "notes": ["No chart available for this insight type."]}

    explanation = build_explanation(ins)
    reliability = get_reliability_notes(ins)

    col_types = get_column_types(df)
    num_cols  = [c for c, t in col_types.items() if t == "numerical"]
    cat_cols  = [c for c, t in col_types.items() if t == "categorical"]

    return {
        "insight_id":        ins["insight_id"],
        "title":             ins["title"],
        "summary":           ins["summary"],
        "type":              ins["type"],
        "strength":          ins["strength"],
        "confidence":        ins["confidence"],
        "columns":           ins["columns"],
        "metadata":          ins.get("metadata", {}),
        "explanation":       explanation,
        "reliability_notes": reliability,
        "chart":             chart_result,
        "recommended_chart": rec,
        "customization_options": {
            "chart_types":   ["bar", "line", "scatter", "histogram", "grouped_bar", "pie", "box"],
            "x_columns":     list(df.columns),
            "y_columns":     num_cols,
            "group_columns": cat_cols,
            "aggregations":  ["mean", "sum", "count", "median", "min", "max"],
        },
    }


# ---------------------------------------------------------------------------
# Endpoint 4 — Chart Preview
# ---------------------------------------------------------------------------

@router.post("/chart-preview")
def chart_preview(req: ChartPreviewRequest):
    """
    Build chart data for any arbitrary configuration (used by the Customize panel).

    Returns {data, config, notes} or raises 400 on invalid config.
    """
    try:
        df = get_dataframe(req.file_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc))

    result = build_chart_data(
        df=df,
        chart_type=req.chart_type,
        x=req.x,
        y=req.y,
        aggregation=req.aggregation,
        group_by=req.group_by,
        filters=req.filters or [],
        sort_by=req.sort_by,
        sort_asc=req.sort_asc,
        top_n=req.top_n,
        title=req.title,
    )

    if "error" in result:
        raise HTTPException(400, result["error"])

    return result
