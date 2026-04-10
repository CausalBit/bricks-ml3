"""Model validation checks for the genre propensity pipeline.

Implements all 10 validation checks from ``validation_step.md`` and
``project_design.md`` section 6. Loads the model by URI from task values,
runs format/metadata checks, performance thresholds, champion comparison,
data-slice checks, and governance tagging.

All thresholds are imported from ``bricks_ml3.config.settings`` — no
hardcoded numbers in this module.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score

from bricks_ml3.config.settings import (
    GENRES,
    LOW_ACTIVITY_MAX,
    MEDIUM_ACTIVITY_MAX,
    NOKIDS_GENRES,
    PER_GENRE_RMSE_THRESHOLD,
    R2_THRESHOLD,
    RMSE_THRESHOLD,
    SCHEMA_GOLD,
    SCHEMA_SILVER,
    SLICE_R2_THRESHOLD,
    TABLE_MOVIES_GENRE_EXPLODED,
    TABLE_MOVIES_GENRE_EXPLODED_NOKIDS,
    TABLE_RATINGS_CLEAN,
    TABLE_USER_GENRE_FEATURES,
    TABLE_USER_PROFILE_FEATURES,
)
from bricks_ml3.training.train import read_split_boundaries
from bricks_ml3.utils.spark_helpers import table_name

logger = logging.getLogger(__name__)

_USER_GENRE_FEATURE_COLS: List[str] = [
    "genre_avg_rating",
    "genre_watch_count",
    "genre_recency_score",
    "genre_share",
    "genre_diversity_index",
    "genre_avg_genome_relevance",
    "genre_tag_count",
]

_USER_PROFILE_FEATURE_COLS: List[str] = [
    "total_ratings",
    "avg_rating",
    "active_days",
    "distinct_genres",
    "diversity_index",
]


# ---------------------------------------------------------------------------
# Individual check helpers (pure logic, no table I/O)
# ---------------------------------------------------------------------------


def _check_artifacts(model_uri: str) -> Any:
    """Load model from URI and return the loaded model object.

    Args:
        model_uri: MLflow model URI (e.g. ``models:/name/version``).

    Returns:
        The loaded pyfunc model.

    Raises:
        Exception: If the model cannot be loaded.
    """
    model = mlflow.pyfunc.load_model(model_uri)
    logger.info("Check 1 PASSED: model loaded from %s", model_uri)
    return model


def _check_description(
    model_name: str,
    model_version: str,
) -> None:
    """Verify that the model version has a non-empty description.

    Args:
        model_name: Fully-qualified registered model name.
        model_version: Model version number as string.

    Raises:
        ValueError: If description is missing or empty.
    """
    mlflow_client = mlflow.MlflowClient()
    mv = mlflow_client.get_model_version(name=model_name, version=model_version)
    if not mv.description:
        raise ValueError(f"Check 2 FAILED: model {model_name} v{model_version} has no description")
    logger.info("Check 2 PASSED: model has description")


def _check_signature(model_uri: str) -> None:
    """Verify the model has an input signature.

    Args:
        model_uri: MLflow model URI.

    Raises:
        ValueError: If signature is None.
    """
    model_info = mlflow.models.get_model_info(model_uri)
    if model_info.signature is None:
        raise ValueError(f"Check 3 FAILED: model at {model_uri} has no input signature")
    logger.info("Check 3 PASSED: model has input signature")


def _smoke_test(model: Any, sample_pdf: pd.DataFrame) -> None:
    """Run a smoke test: predict on a small sample and verify non-null output.

    Args:
        model: Loaded pyfunc model.
        sample_pdf: A pandas DataFrame with at least 5 rows.

    Raises:
        ValueError: If predictions are null or empty.
    """
    predictions = model.predict(sample_pdf.head(5))
    if predictions is None:
        raise ValueError("Check 4 FAILED: model returned None predictions")
    arr = np.asarray(predictions)
    if arr.size == 0:
        raise ValueError("Check 4 FAILED: model returned empty predictions")
    if np.isnan(arr).all():
        raise ValueError("Check 4 FAILED: all predictions are NaN")
    logger.info("Check 4 PASSED: smoke test produced non-null predictions")


def _check_metrics(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute overall RMSE and R-squared.

    Args:
        y_true: True labels with one column per genre.
        y_pred: Predicted values as numpy array.

    Returns:
        Dict with keys ``rmse_overall`` and ``r2_overall``.
    """
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse_overall": rmse, "r2_overall": r2}


def _check_rmse_threshold(metrics: Dict[str, float]) -> None:
    """Assert overall RMSE is below the configured threshold.

    Args:
        metrics: Dict containing ``rmse_overall``.

    Raises:
        ValueError: If RMSE exceeds threshold.
    """
    rmse = metrics["rmse_overall"]
    if rmse >= RMSE_THRESHOLD:
        raise ValueError(f"Check 5 FAILED: RMSE {rmse:.4f} >= threshold {RMSE_THRESHOLD}")
    logger.info("Check 5 PASSED: RMSE %.4f < threshold %.2f", rmse, RMSE_THRESHOLD)


def _check_r2_threshold(metrics: Dict[str, float]) -> None:
    """Assert R-squared is above the configured threshold.

    Args:
        metrics: Dict containing ``r2_overall``.

    Raises:
        ValueError: If R2 is below threshold.
    """
    r2 = metrics["r2_overall"]
    if r2 <= R2_THRESHOLD:
        raise ValueError(f"Check 6 FAILED: R2 {r2:.4f} <= threshold {R2_THRESHOLD}")
    logger.info("Check 6 PASSED: R2 %.4f > threshold %.2f", r2, R2_THRESHOLD)


def _check_champion_comparison(
    model_name: str,
    challenger_rmse: float,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame,
) -> Optional[float]:
    """Compare challenger RMSE against the current Champion on the same holdout.

    Loads the current ``@Champion`` model and scores it on the identical
    test set used for the challenger, ensuring an apples-to-apples comparison.

    If no Champion alias exists, the check is skipped (not failed).

    Args:
        model_name: Fully-qualified registered model name.
        challenger_rmse: RMSE of the challenger model on the holdout set.
        X_test: Feature matrix for the holdout set.
        y_test: True labels for the holdout set.

    Returns:
        The champion's RMSE on the holdout set, or ``None`` if no champion.

    Raises:
        ValueError: If challenger RMSE is worse than champion RMSE.
    """
    mlflow_client = mlflow.MlflowClient()
    try:
        champion_mv = mlflow_client.get_model_version_by_alias(name=model_name, alias="Champion")
    except Exception:
        logger.info("Check 7 SKIPPED: no @Champion alias exists")
        return None

    champion_uri = f"models:/{model_name}/{champion_mv.version}"
    champion_model = mlflow.pyfunc.load_model(champion_uri)
    champion_preds = champion_model.predict(X_test)
    champion_rmse = float(np.sqrt(mean_squared_error(y_test, champion_preds)))

    if challenger_rmse > champion_rmse:
        raise ValueError(
            f"Check 7 FAILED: challenger RMSE {challenger_rmse:.4f} > "
            f"champion RMSE {champion_rmse:.4f} (both on same holdout)"
        )
    logger.info(
        "Check 7 PASSED: challenger RMSE %.4f <= champion RMSE %.4f (both evaluated on identical holdout set)",
        challenger_rmse,
        champion_rmse,
    )
    return champion_rmse


def _check_per_genre_rmse(per_genre_rmse: Dict[str, float]) -> None:
    """Assert no single genre exceeds the per-genre RMSE threshold.

    Args:
        per_genre_rmse: Dict mapping genre name to its RMSE.

    Raises:
        ValueError: If any genre's RMSE exceeds the threshold.
    """
    violations = {g: v for g, v in per_genre_rmse.items() if v > PER_GENRE_RMSE_THRESHOLD}
    if violations:
        raise ValueError(f"Check 8 FAILED: genres exceeding threshold {PER_GENRE_RMSE_THRESHOLD}: {violations}")
    logger.info(
        "Check 8 PASSED: all per-genre RMSE values below %.2f",
        PER_GENRE_RMSE_THRESHOLD,
    )


def _check_activity_slices(
    y_true_by_slice: Dict[str, pd.DataFrame],
    y_pred_by_slice: Dict[str, np.ndarray],
) -> None:
    """Evaluate R-squared on low/medium/high activity user slices.

    Args:
        y_true_by_slice: Dict mapping slice name to true label DataFrame.
        y_pred_by_slice: Dict mapping slice name to predicted array.

    Raises:
        ValueError: If any slice's R2 is below the threshold.
    """
    violations: Dict[str, float] = {}
    for slice_name in y_true_by_slice:
        yt = y_true_by_slice[slice_name]
        yp = y_pred_by_slice[slice_name]
        if len(yt) == 0:
            logger.info("Check 9: slice '%s' has no data, skipping", slice_name)
            continue
        r2 = float(r2_score(yt, yp))
        logger.info("Check 9: slice '%s' R2 = %.4f", slice_name, r2)
        if r2 < SLICE_R2_THRESHOLD:
            violations[slice_name] = r2

    if violations:
        raise ValueError(f"Check 9 FAILED: slices below R2 threshold {SLICE_R2_THRESHOLD}: {violations}")
    logger.info("Check 9 PASSED: all activity slices above R2 threshold")


def _set_governance_tags(
    mlflow_client: Any,
    model_name: str,
    model_version: str,
) -> None:
    """Set Unity Catalog governance tags on the model version.

    Tags set: ``model_validation_status``, ``model_owner``, ``use_case``,
    ``data_classification``.

    Args:
        mlflow_client: ``mlflow.MlflowClient`` instance configured for UC.
        model_name: Fully-qualified registered model name.
        model_version: Model version number as string.
    """
    tags = {
        "model_validation_status": "PASSED",
        "model_owner": "mlops-team",
        "use_case": "genre_propensity",
        "data_classification": "public",
    }
    for key, value in tags.items():
        mlflow_client.set_model_version_tag(
            name=model_name,
            version=model_version,
            key=key,
            value=value,
        )
    logger.info("Check 10 PASSED: governance tags set on %s v%s", model_name, model_version)


# ---------------------------------------------------------------------------
# Multi-output pivot helper (mirrors training.train._pivot_to_multi_output)
# ---------------------------------------------------------------------------


def _pivot_to_multi_output(
    pdf: pd.DataFrame,
    genres: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot per-(userId, genre) data into multi-output format for evaluation.

    Args:
        pdf: Pandas DataFrame with userId, genre, label, and feature columns.
        genres: Ordered list of genre names.

    Returns:
        ``(X, y)`` tuple of pandas DataFrames indexed by userId.
    """
    sorted_genres = sorted(genres)

    y = pdf.pivot_table(index="userId", columns="genre", values="label", fill_value=0.0)
    for g in sorted_genres:
        if g not in y.columns:
            y[g] = 0.0
    y = y[sorted_genres]

    profile_cols = [c for c in _USER_PROFILE_FEATURE_COLS if c in pdf.columns]
    user_profiles = pdf.groupby("userId")[profile_cols].first()

    X = user_profiles.copy()
    for feat in _USER_GENRE_FEATURE_COLS:
        if feat not in pdf.columns:
            continue
        feat_pivot = pdf.pivot_table(index="userId", columns="genre", values=feat, fill_value=0.0).astype(float)
        feat_pivot.columns = [f"{feat}_{g}" for g in feat_pivot.columns]
        X = X.join(feat_pivot, how="left")

    X = X.fillna(0.0)

    common = X.index.intersection(y.index)
    return X.loc[common], y.loc[common]


# ---------------------------------------------------------------------------
# Per-genre RMSE helper
# ---------------------------------------------------------------------------


def _compute_per_genre_metrics(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    genres: List[str],
) -> Dict[str, float]:
    """Compute RMSE for each genre column.

    Args:
        y_true: True labels with one column per genre (sorted order).
        y_pred: Predicted values array.
        genres: Ordered list of genre names.

    Returns:
        Dict mapping genre name to its RMSE.
    """
    sorted_genres = sorted(genres)
    result: Dict[str, float] = {}
    for i, genre in enumerate(sorted_genres):
        result[genre] = float(np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i])))
    return result


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_validation(
    spark: Any,
    catalog: str,
    model_variant: str,
    dbutils: Any,
    model_name: str = "",
    model_version: str = "",
) -> None:
    """Run the complete 10-check validation pipeline.

    Reads model name/version from task values set by the training step,
    reads split boundaries from ``ml.split_metadata``, runs all checks
    (including a fair head-to-head champion comparison on the same holdout
    set), logs dataset metadata and validation metrics to MLflow, and
    sets the appropriate tags and aliases on pass or failure.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.
        model_variant: ``"general"`` or ``"nokids"``.
        dbutils: Databricks ``dbutils`` object for reading task values.
        model_name: Override registered model name (skip taskValues lookup).
        model_version: Override model version (skip taskValues lookup).

    Raises:
        ValueError: If any validation check fails.
    """
    from pyspark.sql import functions as F

    mlflow.set_registry_uri("databricks-uc")
    mlflow_client = mlflow.MlflowClient()

    if not model_name or not model_version:
        task_key = f"train_{model_variant}" if model_variant != "general" else "train_general"
        model_name = dbutils.jobs.taskValues.get(taskKey=task_key, key="model_name")
        model_version = dbutils.jobs.taskValues.get(taskKey=task_key, key="model_version")
    model_uri = f"models:/{model_name}/{model_version}"

    # Resolve logged_model_id from the training run for metric linkage
    mv_info = mlflow_client.get_model_version(name=model_name, version=model_version)
    try:
        training_run = mlflow_client.get_run(mv_info.run_id)
        _ = training_run.data.tags.get("mlflow.loggedModels")
    except Exception:
        logger.warning("Could not resolve logged_model_id from training run")

    mlflow_client.set_model_version_tag(
        name=model_name,
        version=model_version,
        key="model_validation_status",
        value="PENDING",
    )

    try:
        # Check 1: Has artifacts
        loaded_model = _check_artifacts(model_uri)

        # Check 2: Has description
        _check_description(model_name, model_version)

        # Check 3: Has input signature
        _check_signature(model_uri)

        # Read persisted split boundaries for deterministic holdout
        boundaries = read_split_boundaries(spark, catalog)
        holdout_boundary = boundaries["holdout_boundary_ts"]

        # Build holdout/test set using the persisted boundary
        genres = GENRES if model_variant == "general" else NOKIDS_GENRES
        if model_variant == "nokids":
            genre_tbl = table_name(catalog, SCHEMA_SILVER, TABLE_MOVIES_GENRE_EXPLODED_NOKIDS)
        else:
            genre_tbl = table_name(catalog, SCHEMA_SILVER, TABLE_MOVIES_GENRE_EXPLODED)

        ratings_tbl = table_name(catalog, SCHEMA_SILVER, TABLE_RATINGS_CLEAN)
        ratings_df = spark.read.table(ratings_tbl)
        genre_exploded_df = spark.read.table(genre_tbl)

        genre_cols = genre_exploded_df.select("movieId", "genre")
        rated_genres = ratings_df.join(genre_cols, on="movieId")

        ts_long = F.col("timestamp").cast("long")
        test_data = rated_genres.filter(ts_long > F.lit(holdout_boundary))
        test_labels = test_data.groupBy("userId", "genre").agg(F.avg("rating").alias("label"))

        ug_tbl = table_name(catalog, SCHEMA_GOLD, TABLE_USER_GENRE_FEATURES)
        up_tbl = table_name(catalog, SCHEMA_GOLD, TABLE_USER_PROFILE_FEATURES)
        ug_df = spark.read.table(ug_tbl)
        up_df = spark.read.table(up_tbl)

        # Restrict to users present in the gold feature tables so that
        # total_ratings (LongType) is never null after the join.
        gold_users = up_df.select("userId").distinct()
        test_labels = test_labels.join(gold_users, on="userId", how="inner")

        test_with_features = test_labels.join(ug_df, on=["userId", "genre"], how="left").join(
            up_df, on=["userId"], how="left"
        )
        test_pdf = test_with_features.toPandas()

        X_test, y_test = _pivot_to_multi_output(test_pdf, genres)

        # Collect holdout set metadata for logging
        holdout_user_count = int(X_test.shape[0])
        holdout_row_count = len(test_pdf)
        holdout_stats = test_data.agg(
            F.min(ts_long).alias("min_ts"),
            F.max(ts_long).alias("max_ts"),
        ).collect()[0]

        # Check 4: Smoke test
        _smoke_test(loaded_model, X_test)

        # Compute metrics
        y_pred = loaded_model.predict(X_test)
        metrics = _check_metrics(y_test, y_pred)

        # Check 5: RMSE threshold
        _check_rmse_threshold(metrics)

        # Check 6: R-squared threshold
        _check_r2_threshold(metrics)

        # Check 7: Head-to-head champion comparison on same holdout
        champion_rmse = _check_champion_comparison(model_name, metrics["rmse_overall"], X_test, y_test)

        # Check 8: Per-genre RMSE
        per_genre_rmse = _compute_per_genre_metrics(y_test, y_pred, genres)
        _check_per_genre_rmse(per_genre_rmse)

        # Check 9: Activity-level slices
        user_total_ratings = ratings_df.groupBy("userId").agg(F.count("*").alias("total_user_ratings")).toPandas()
        test_users = pd.DataFrame({"userId": X_test.index})
        user_activity = test_users.merge(user_total_ratings, on="userId", how="left").fillna(0)

        slices = {
            "low": user_activity["total_user_ratings"] < LOW_ACTIVITY_MAX,
            "medium": (
                (user_activity["total_user_ratings"] >= LOW_ACTIVITY_MAX)
                & (user_activity["total_user_ratings"] <= MEDIUM_ACTIVITY_MAX)
            ),
            "high": user_activity["total_user_ratings"] > MEDIUM_ACTIVITY_MAX,
        }

        y_true_by_slice: Dict[str, pd.DataFrame] = {}
        y_pred_by_slice: Dict[str, np.ndarray] = {}
        for slice_name, mask in slices.items():
            user_ids = user_activity.loc[mask.values, "userId"].values
            idx_mask = y_test.index.isin(user_ids)
            y_true_by_slice[slice_name] = y_test.loc[idx_mask]
            y_pred_by_slice[slice_name] = y_pred[idx_mask]

        _check_activity_slices(y_true_by_slice, y_pred_by_slice)

        # Log validation metrics and holdout dataset metadata to MLflow
        try:
            val_exp_name = f"/Shared/genre_propensity/{catalog}/validation"
            val_exp = mlflow_client.get_experiment_by_name(val_exp_name)
            if val_exp is None:
                val_exp_id = mlflow_client.create_experiment(val_exp_name)
            else:
                val_exp_id = val_exp.experiment_id

            val_metrics = {
                "holdout_rmse_overall": metrics["rmse_overall"],
                "holdout_r2_overall": metrics["r2_overall"],
                **{f"holdout_rmse_{genre}": v for genre, v in per_genre_rmse.items()},
            }
            if champion_rmse is not None:
                val_metrics["champion_holdout_rmse"] = champion_rmse

            with mlflow.start_run(
                experiment_id=val_exp_id,
                run_name=f"validate_{model_variant}_v{model_version}",
            ):
                mlflow.log_params(
                    {
                        "model_name": model_name,
                        "model_version": model_version,
                        "holdout_boundary_ts": str(holdout_boundary),
                        "train_boundary_ts": str(boundaries["train_boundary_ts"]),
                    }
                )

                mlflow.log_metrics(val_metrics)

                # Log holdout dataset for digest tracking
                try:
                    holdout_dataset = mlflow.data.from_pandas(
                        test_pdf,
                        name=f"holdout_{model_variant}",
                    )
                    mlflow.log_input(holdout_dataset, context="validation")
                except Exception:
                    logger.warning(
                        "Failed to log holdout dataset via mlflow.data",
                        exc_info=True,
                    )

            logger.info("Validation metrics logged to experiment %s", val_exp_name)
        except Exception:
            logger.warning(
                "Failed to log validation metrics to experiment",
                exc_info=True,
            )

        # Tag the model version with holdout metadata for auditing
        holdout_tags = {
            "holdout_row_count": str(holdout_row_count),
            "holdout_user_count": str(holdout_user_count),
            "holdout_boundary_ts": str(holdout_boundary),
            "holdout_min_timestamp": str(holdout_stats["min_ts"]),
            "holdout_max_timestamp": str(holdout_stats["max_ts"]),
            "holdout_rmse_overall": f"{metrics['rmse_overall']:.4f}",
            "holdout_r2_overall": f"{metrics['r2_overall']:.4f}",
        }
        if champion_rmse is not None:
            holdout_tags["champion_holdout_rmse"] = f"{champion_rmse:.4f}"
        for key, value in holdout_tags.items():
            mlflow_client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key=key,
                value=value,
            )
        logger.info("Holdout metadata tags set on %s v%s", model_name, model_version)

        # Check 10: Governance tags
        _set_governance_tags(mlflow_client, model_name, model_version)

        # All checks passed — assign Challenger alias
        mlflow_client.set_registered_model_alias(name=model_name, alias="Challenger", version=model_version)
        logger.info(
            "VALIDATION PASSED: %s v%s assigned @Challenger",
            model_name,
            model_version,
        )

    except Exception as exc:
        try:
            mlflow_client.set_model_version_tag(
                name=model_name,
                version=model_version,
                key="model_validation_status",
                value="FAILED",
            )
        except Exception:
            pass
        logger.error("VALIDATION FAILED: %s", exc)
        raise
