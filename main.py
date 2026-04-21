"""
FastAPI service for the Cleaning Agent v2
==========================================
Endpoints:
  GET  /                          health check
  POST /profile                   profile a CSV file
  POST /predict                   predict actions for all columns (no cleaning)
  POST /clean                     full pipeline: profile → predict → execute
  POST /clean-column              clean a single named column
  POST /clean-column/override     clean a column with a manually specified action
  GET  /actions                   list all available action names
  GET  /model/status              check if ML model is loaded and its metadata
"""

from __future__ import annotations

import io
import os
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from cleaning_agent.pipeline import CleaningPipeline
from cleaning_agent.profiler import profile_dataset, profile_column
from cleaning_agent.predictor import predict_action, _load_model
from cleaning_agent.detector import detect_dataset_type
from cleaning_agent.intelligence import refine_action
from cleaning_agent.analyzer import analyze_dataset
from cleaning_agent.stat_analyzer.routes import router as stat_analyzer_router

app = FastAPI(
    title="Cleaning Agent v2",
    description="ML-assisted data cleaning with rule-based fallback",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

PIPELINE = CleaningPipeline()

app.include_router(stat_analyzer_router)

AVAILABLE_ACTIONS = [
    "no_change",
    "impute_mean",
    "impute_median",
    "convert_and_flag",
    "parse_and_flag",
    "normalize_and_flag",
    "normalize_boolean",
    "flag_outliers",
    "fill_mode",
    "fill_unknown",
    "title_case",
    "date_manual_review",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_csv(file: UploadFile) -> pd.DataFrame:
    try:
        return pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")


def _df_to_preview(df: pd.DataFrame, max_rows: int = 50) -> list[dict]:
    import numpy as np
    import math
    preview = df.head(max_rows).copy()
    # Replace NaN / inf / -inf so they serialise to null, not crash JSON
    preview = preview.replace([np.inf, -np.inf], None)
    preview = preview.where(pd.notnull(preview), None)
    rows = preview.to_dict(orient="records")
    # Extra pass: catch any stray float nan that slipped through
    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        return v
    return [{k: _clean(v) for k, v in row.items()} for row in rows]


def _sanitise_report(report: list) -> list:
    """Recursively replace NaN/inf floats so the response is JSON-safe."""
    import math

    def _clean(v):
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return None
        if isinstance(v, dict):
            return {ik: _clean(iv) for ik, iv in v.items()}
        if isinstance(v, list):
            return [_clean(i) for i in v]
        return v

    return [_clean(row) for row in report]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "service": "cleaning-agent-v2"}


@app.get("/actions")
def list_actions():
    return {"actions": AVAILABLE_ACTIONS}


@app.get("/model/status")
def model_status():
    bundle = _load_model()
    if bundle is None:
        return {
            "model_loaded": False,
            "message": "Model not found. Run generate_training_data.py then train_model.py.",
        }
    le = bundle["label_encoder"]
    model = bundle["model"]
    return {
        "model_loaded": True,
        "action_classes": list(model.classes_),
        "col_type_classes": list(le.classes_),
        "n_features": len(bundle["features"]),
        "features": bundle["features"],
        "tree_depth": model.get_depth(),
        "tree_leaves": model.get_n_leaves(),
    }


@app.post("/detect")
async def detect_type(file: UploadFile = File(...)):
    """Detect the domain/type of a dataset and return a cleaning template."""
    df = _read_csv(file)
    profiles_result = profile_dataset(df)
    detection = detect_dataset_type(list(df.columns), profiles_result["profiles"])
    return detection


@app.post("/predict-with-template")
async def predict_with_template(file: UploadFile = File(...)):
    """
    Combined endpoint: profile + predict + detect dataset type in one call.
    Returns predictions for every column AND a domain template.
    """
    content = await file.read()
    import io

    df = _read_csv_bytes(content)
    profiles_result = profile_dataset(df)
    detection = detect_dataset_type(list(df.columns), profiles_result["profiles"])

    predictions = []
    for col in df.columns:
        profile = profile_column(df[col], col)
        profile["column_name"] = col
        prediction = predict_action(profile)
        refined_action, refinement = refine_action(
            prediction["predicted_action"], df[col], profile
        )
        template_rec = detection.get("column_recommendations", {}).get(col)
        predictions.append({
            "column": col,
            "col_type": profile["col_type"],
            "missing_rate": profile["missing_rate"],
            "predicted_action": refined_action,
            "raw_predicted_action": prediction["predicted_action"],
            "refined": refinement.get("refined", False),
            "refinement_reason": refinement.get("reason", None),
            "confidence": prediction["confidence"],
            "method": prediction["method"],
            "rule_action": prediction["rule_action"],
            "probabilities": prediction.get("probabilities", {}),
            "template_action": template_rec["suggested_action"] if template_rec else None,
            "template_reason": template_rec["reason"] if template_rec else None,
        })

    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "predictions": predictions,
        "template": detection,
    }


def _read_csv_bytes(content: bytes) -> "pd.DataFrame":
    import pandas as pd
    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail=f"Could not parse CSV: {e}")


@app.post("/profile")
async def profile_file(file: UploadFile = File(...)):
    df = _read_csv(file)
    result = profile_dataset(df)
    return result


@app.post("/predict")
async def predict_all_columns(file: UploadFile = File(...)):
    """Profile every column and return the predicted cleaning action (no actual cleaning)."""
    df = _read_csv(file)
    predictions = []
    for col in df.columns:
        profile = profile_column(df[col], col)
        prediction = predict_action(profile)
        predictions.append({
            "column": col,
            "col_type": profile["col_type"],
            "missing_rate": profile["missing_rate"],
            "predicted_action": prediction["predicted_action"],
            "confidence": prediction["confidence"],
            "method": prediction["method"],
            "rule_action": prediction["rule_action"],
            "probabilities": prediction.get("probabilities", {}),
        })
    return {
        "row_count": len(df),
        "column_count": len(df.columns),
        "predictions": predictions,
    }


@app.post("/clean")
async def clean_file(file: UploadFile = File(...)):
    """Run the full cleaning pipeline on the uploaded CSV."""
    df = _read_csv(file)
    result = PIPELINE.run(df)

    cleaned_df = result["cleaned_df"]

    return {
        "summary": result["summary"],
        "report": _sanitise_report(result["report"]),
        "preview": _df_to_preview(cleaned_df),
        "row_count_before": result["summary"]["rows_before"],
        "row_count_after": result["summary"]["rows_after"],
    }


@app.post("/clean/download")
async def clean_and_download(file: UploadFile = File(...)):
    """Run the full cleaning pipeline and return the cleaned CSV as a download."""
    df = _read_csv(file)
    result = PIPELINE.run(df)
    cleaned_df = result["cleaned_df"]

    output = io.StringIO()
    cleaned_df.to_csv(output, index=False)
    output.seek(0)

    return StreamingResponse(
        output,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=cleaned.csv"},
    )


@app.post("/clean-column")
async def clean_single_column(
    file: UploadFile = File(...),
    column_name: str = Form(...),
):
    """Profile and clean a single column using the ML-predicted (or rule-based) action."""
    df = _read_csv(file)
    if column_name not in df.columns:
        raise HTTPException(
            status_code=404,
            detail=f"Column '{column_name}' not found. Available: {list(df.columns)}",
        )

    result = PIPELINE.run_column(df, column_name)
    cleaned_df = result["cleaned_df"]

    return {
        "summary": result["summary"],
        "report": _sanitise_report(result["report"]),
        "preview": _df_to_preview(cleaned_df),
    }


@app.post("/clean-column/override")
async def clean_column_with_override(
    file: UploadFile = File(...),
    column_name: str = Form(...),
    action: str = Form(...),
):
    """Clean a single column using a manually specified action."""
    if action not in AVAILABLE_ACTIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action '{action}'. Valid actions: {AVAILABLE_ACTIONS}",
        )
    df = _read_csv(file)
    if column_name not in df.columns:
        raise HTTPException(
            status_code=404,
            detail=f"Column '{column_name}' not found.",
        )

    result = PIPELINE.run_column(df, column_name, action_override=action)
    cleaned_df = result["cleaned_df"]

    return {
        "summary": result["summary"],
        "report": _sanitise_report(result["report"]),
        "preview": _df_to_preview(cleaned_df),
    }


@app.post("/clean-columns")
async def clean_multiple_columns(
    file: UploadFile = File(...),
    column_names: str = Form(...),   # comma-separated
):
    """Clean a specific subset of columns."""
    df = _read_csv(file)
    cols = [c.strip() for c in column_names.split(",") if c.strip()]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise HTTPException(
            status_code=404,
            detail=f"Columns not found: {missing}",
        )

    result = PIPELINE.run(df, columns=cols)
    cleaned_df = result["cleaned_df"]

    return {
        "summary": result["summary"],
        "report": _sanitise_report(result["report"]),
        "preview": _df_to_preview(cleaned_df),
    }


@app.post("/analyze")
async def analyze_file(file: UploadFile = File(...)):
    """
    Full statistical analysis of every column in the uploaded CSV.

    Returns per-column stats (mean/median/std/quartiles/skewness for numerical;
    unique count / top-5 frequencies for categorical), a dataset-level summary,
    and a correlation matrix for numerical columns.
    """
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are accepted.")
    df = _read_csv(file)
    if df.empty:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    try:
        result = analyze_dataset(df)
        return result
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {exc}")
