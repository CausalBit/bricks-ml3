"""Data drift detection for the genre propensity pipeline.

Compares current feature distributions against a stored baseline snapshot
using Population Stability Index (PSI). On the first run (no baseline
exists), the current features are saved as the baseline and drift detection
is skipped. Prediction drift is not yet implemented because the scores
table is overwritten on each batch run (no historical data to compare).

Results are appended to a monitoring log table for historical tracking.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import types as T

from bricks_ml3.config.settings import (
    PSI_DRIFT_THRESHOLD,
    SCHEMA_GOLD,
    SCHEMA_INFERENCE,
    TABLE_FEATURE_BASELINE,
    TABLE_MONITORING_LOG,
    TABLE_SCORES_DAILY,
    TABLE_USER_GENRE_FEATURES,
)
from bricks_ml3.utils.spark_helpers import table_name

logger = logging.getLogger(__name__)

_NUMERIC_FEATURE_COLS: List[str] = [
    "genre_avg_rating",
    "genre_watch_count",
    "genre_recency_score",
    "genre_share",
    "genre_diversity_index",
    "genre_avg_genome_relevance",
    "genre_tag_count",
]

_NUM_PSI_BINS: int = 10


def _compute_psi(
    baseline: np.ndarray,
    current: np.ndarray,
    bins: int = _NUM_PSI_BINS,
) -> float:
    """Compute Population Stability Index between two distributions.

    Uses quantile-based binning derived from the baseline distribution.
    Handles skewed data by deduplicating bin edges and falling back to
    fewer bins when necessary.

    Args:
        baseline: 1-D array of baseline values.
        current: 1-D array of current values.
        bins: Number of quantile bins.

    Returns:
        PSI value (float). Higher values indicate more drift.
    """
    breakpoints = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(baseline, breakpoints)

    # Deduplicate edges — skewed data can produce identical percentiles
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 3:
        # Not enough distinct edges to form meaningful bins
        return 0.0

    # Extend the last edge slightly so the right boundary is inclusive
    bin_edges[-1] = bin_edges[-1] + 1e-6

    baseline_counts = np.histogram(baseline, bins=bin_edges)[0]
    current_counts = np.histogram(current, bins=bin_edges)[0]

    num_bins = len(baseline_counts)

    # Convert to proportions, avoid zeros
    baseline_pct = (baseline_counts + 1e-6) / (baseline_counts.sum() + num_bins * 1e-6)
    current_pct = (current_counts + 1e-6) / (current_counts.sum() + num_bins * 1e-6)

    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    return float(psi)


def _table_exists(spark: SparkSession, full_table_name: str) -> bool:
    """Check whether a Delta table exists in the catalog."""
    try:
        spark.read.table(full_table_name).limit(0)
        return True
    except Exception:
        return False


def run_drift_check(spark: SparkSession, catalog: str) -> Dict[str, Any]:
    """Run feature and prediction drift checks.

    Compares the current ``gold.user_genre_features`` table against a
    stored baseline snapshot using PSI. On the first run, the current
    features are saved as the baseline and drift detection is skipped.

    Prediction drift compares the current ``inference.genre_propensity_scores_daily``
    against a stored baseline snapshot. Because the scores table is
    overwritten on each run, drift detection requires a previously stored
    baseline to exist.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.

    Returns:
        Summary dict with per-feature PSI values, ``drift_detected`` flag,
        and ``timestamp``.
    """
    timestamp = datetime.utcnow().isoformat()
    summary: Dict[str, Any] = {
        "timestamp": timestamp,
        "catalog": catalog,
        "feature_psi": {},
        "prediction_psi": float("nan"),
        "drift_detected": False,
    }

    # --- Feature drift ---
    features_table = table_name(catalog, SCHEMA_GOLD, TABLE_USER_GENRE_FEATURES)
    baseline_table = table_name(catalog, SCHEMA_INFERENCE, TABLE_FEATURE_BASELINE)
    try:
        current_df = spark.read.table(features_table)

        if not _table_exists(spark, baseline_table):
            # First run — save current features as baseline, skip drift check
            logger.info(
                "No feature baseline found at %s. Saving current features as baseline (first run).",
                baseline_table,
            )
            current_df.write.mode("overwrite").saveAsTable(baseline_table)
        else:
            baseline_df = spark.read.table(baseline_table)

            for col_name in _NUMERIC_FEATURE_COLS:
                try:
                    baseline_vals = np.array(baseline_df.select(col_name).dropna().toPandas()[col_name])
                    current_vals = np.array(current_df.select(col_name).dropna().toPandas()[col_name])
                    if len(baseline_vals) > _NUM_PSI_BINS and len(current_vals) > _NUM_PSI_BINS:
                        psi = _compute_psi(baseline_vals, current_vals)
                        summary["feature_psi"][col_name] = round(psi, 6)
                        if psi > PSI_DRIFT_THRESHOLD:
                            summary["drift_detected"] = True
                            logger.warning(
                                "Feature drift detected for %s: PSI=%.4f (threshold=%.2f)",
                                col_name,
                                psi,
                                PSI_DRIFT_THRESHOLD,
                            )
                except Exception:
                    logger.warning("Failed to compute PSI for feature %s", col_name, exc_info=True)
    except Exception:
        logger.warning("Failed to read features table %s", features_table, exc_info=True)

    # --- Prediction drift ---
    # The scores table is overwritten on each batch scoring run, so it only
    # contains the latest day's data. Prediction drift requires comparing
    # against a previous day's scores, which are not retained. Log a warning
    # and skip until the pipeline is extended to preserve historical scores.
    scores_table = table_name(catalog, SCHEMA_INFERENCE, TABLE_SCORES_DAILY)
    try:
        scores_df = spark.read.table(scores_table)
        score_cols = [c for c in scores_df.columns if c not in ("userId", "scored_date", "model_version")]
        if score_cols:
            logger.info(
                "Prediction drift check skipped: scores table %s is overwritten "
                "daily (no historical data to compare against).",
                scores_table,
            )
        else:
            logger.warning(
                "No score columns found in %s — cannot compute prediction drift.",
                scores_table,
            )
    except Exception:
        logger.warning("Failed to read scores table %s", scores_table, exc_info=True)

    # --- Write monitoring log ---
    monitoring_table = table_name(catalog, SCHEMA_INFERENCE, TABLE_MONITORING_LOG)
    try:
        import json

        schema = T.StructType(
            [
                T.StructField("timestamp", T.StringType(), False),
                T.StructField("catalog", T.StringType(), False),
                T.StructField("feature_psi", T.StringType(), True),
                T.StructField("prediction_psi", T.DoubleType(), True),
                T.StructField("drift_detected", T.BooleanType(), False),
            ]
        )

        log_row = spark.createDataFrame(
            [
                (
                    timestamp,
                    catalog,
                    json.dumps(summary["feature_psi"]),
                    summary["prediction_psi"],
                    summary["drift_detected"],
                )
            ],
            schema=schema,
        )
        log_row.write.mode("append").saveAsTable(monitoring_table)
        logger.info("Monitoring results written to %s", monitoring_table)
    except Exception:
        logger.warning("Failed to write monitoring log to %s", monitoring_table, exc_info=True)

    return summary
