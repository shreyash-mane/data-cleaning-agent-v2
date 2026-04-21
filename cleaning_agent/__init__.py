"""
Cleaning Agent v2
=================
ML-assisted data cleaning pipeline.

Usage
-----
    from cleaning_agent.pipeline import CleaningPipeline

    pipeline = CleaningPipeline()
    result = pipeline.run(df)
"""

from .pipeline import CleaningPipeline
from .profiler import profile_column, profile_dataset
from .predictor import predict_action
from .executor import apply_action

__all__ = [
    "CleaningPipeline",
    "profile_column",
    "profile_dataset",
    "predict_action",
    "apply_action",
]
