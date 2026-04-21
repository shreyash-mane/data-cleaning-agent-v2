"""
Cleaning Pipeline
=================
Orchestrates profiling → prediction → execution for every column
(or a selected subset) in one call.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from .profiler import profile_column, profile_dataset
from .predictor import predict_action
from .executor import apply_action
from .intelligence import refine_action


class CleaningPipeline:
    """
    End-to-end ML-assisted column cleaning.

    Parameters
    ----------
    confidence_threshold : float
        Passed to the predictor. If the ML model's top confidence < threshold
        the rule-based action is used instead. Default is 0.65.
    skip_id_columns : bool
        If True (default), columns detected as 'id' type are skipped.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        skip_id_columns: bool = True,
    ):
        self.confidence_threshold = confidence_threshold
        self.skip_id_columns = skip_id_columns

    # ------------------------------------------------------------------

    def run(
        self,
        df: pd.DataFrame,
        columns: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Clean *df* column by column.

        Parameters
        ----------
        df      : input dataframe (not modified — a copy is made)
        columns : optional subset of columns to clean (default: all)

        Returns
        -------
        {
            "cleaned_df": pd.DataFrame,
            "report": list of per-column result dicts,
            "summary": high-level counts,
        }
        """
        df_work = df.copy()
        cols_to_clean = columns or list(df_work.columns)

        # -- pre-flight: deduplicate & drop all-null rows --
        before_rows = len(df_work)
        df_work = df_work.drop_duplicates()
        df_work = df_work.replace(r"^\s*$", pd.NA, regex=True)
        df_work = df_work.dropna(how="all")
        dropped_rows = before_rows - len(df_work)

        report = []
        total_changed = 0
        total_flagged = 0

        for col in cols_to_clean:
            if col not in df_work.columns:
                continue

            profile = profile_column(df_work[col], col)

            if self.skip_id_columns and profile["col_type"] == "id":
                report.append({
                    "column": col,
                    "col_type": "id",
                    "action": "skipped",
                    "method": "skip_id",
                    "confidence": 1.0,
                    "cells_changed": 0,
                    "cells_flagged": 0,
                    "profile": profile,
                })
                continue

            prediction = predict_action(profile)
            ml_action = prediction["predicted_action"]

            # Intelligence layer: refine the ML decision using actual column data
            profile["column_name"] = col
            action, intel = refine_action(ml_action, df_work[col], profile)

            df_work, change_log = apply_action(df_work, col, action)

            total_changed += change_log.get("cells_changed", 0)
            total_flagged += change_log.get("cells_flagged", 0)

            report.append({
                "column": col,
                "col_type": profile["col_type"],
                "action": action,
                "ml_action": ml_action,
                "intelligence_refined": intel.get("refined", False),
                "intelligence_reason": intel.get("reason", ""),
                "method": prediction["method"],
                "confidence": prediction["confidence"],
                "rule_action": prediction["rule_action"],
                "cells_changed": change_log.get("cells_changed", 0),
                "cells_flagged": change_log.get("cells_flagged", 0),
                "profile": profile,
                "probabilities": prediction.get("probabilities", {}),
                "intelligence": intel,
                **{k: v for k, v in change_log.items()
                   if k not in ("column", "action", "cells_changed", "cells_flagged")},
            })

        summary = {
            "columns_processed": len(report),
            "columns_skipped": sum(1 for r in report if r["action"] == "skipped"),
            "total_cells_changed": total_changed,
            "total_cells_flagged": total_flagged,
            "rows_before": before_rows,
            "rows_after": len(df_work),
            "duplicate_rows_dropped": dropped_rows,
            "ml_used_count": sum(1 for r in report if r.get("method") == "ml"),
            "rule_used_count": sum(1 for r in report if r.get("method") in ("rule_based", "fallback_low_confidence")),
        }

        return {
            "cleaned_df": df_work,
            "report": report,
            "summary": summary,
        }

    def run_column(
        self,
        df: pd.DataFrame,
        col_name: str,
        action_override: str | None = None,
    ) -> dict[str, Any]:
        """
        Clean a single column.  Optionally override the predicted action.

        Returns the same dict shape as run() but with a single-item report.
        """
        df_copy = df.copy()
        profile = profile_column(df_copy[col_name], col_name)
        profile["column_name"] = col_name
        prediction = predict_action(profile)
        ml_action = prediction["predicted_action"]

        if action_override:
            action, intel = action_override, {"refined": False, "reason": "manual override"}
        else:
            action, intel = refine_action(ml_action, df_copy[col_name], profile)

        df_copy, change_log = apply_action(df_copy, col_name, action)

        return {
            "cleaned_df": df_copy,
            "report": [{
                "column": col_name,
                "col_type": profile["col_type"],
                "action": action,
                "ml_action": ml_action,
                "intelligence_refined": intel.get("refined", False),
                "intelligence_reason": intel.get("reason", ""),
                "method": "override" if action_override else prediction["method"],
                "confidence": prediction["confidence"],
                "rule_action": prediction["rule_action"],
                "cells_changed": change_log.get("cells_changed", 0),
                "cells_flagged": change_log.get("cells_flagged", 0),
                "profile": profile,
                "probabilities": prediction.get("probabilities", {}),
                "intelligence": intel,
                **{k: v for k, v in change_log.items()
                   if k not in ("column", "action", "cells_changed", "cells_flagged")},
            }],
            "summary": {
                "columns_processed": 1,
                "total_cells_changed": change_log.get("cells_changed", 0),
                "total_cells_flagged": change_log.get("cells_flagged", 0),
                "ml_used_count": 1 if prediction.get("method") == "ml" else 0,
            },
        }
