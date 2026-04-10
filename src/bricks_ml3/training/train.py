"""LightGBM multi-output training with MLflow 3.x logging.

Builds training labels from silver tables, constructs a training set via
Feature Store lookups, fits a ``MultiOutputRegressor(LGBMRegressor())``,
logs metrics/artifacts to MLflow, and registers the model in Unity Catalog.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.multioutput import MultiOutputRegressor

from bricks_ml3.config.settings import (
    GENRES,
    MODEL_GENERAL,
    MODEL_NOKIDS,
    NOKIDS_GENRES,
    SCHEMA_GOLD,
    SCHEMA_ML,
    SCHEMA_SILVER,
    TABLE_MOVIES_GENRE_EXPLODED,
    TABLE_MOVIES_GENRE_EXPLODED_NOKIDS,
    TABLE_RATINGS_CLEAN,
    TABLE_SPLIT_METADATA,
    TABLE_USER_GENRE_FEATURES,
    TABLE_USER_PROFILE_FEATURES,
)
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
# Pure transformation helpers (no table I/O – used by unit tests)
# ---------------------------------------------------------------------------


def _build_training_labels_transform(
    ratings_df: DataFrame,
    genre_exploded_df: DataFrame,
    train_boundary: int,
    holdout_boundary: int,
) -> DataFrame:
    """Build per-(userId, genre) labels with chronological 3-way split.

    Joins cleaned ratings with genre-exploded movies and assigns each row
    to ``train``, ``val``, or ``test`` based on pre-computed timestamp
    boundaries from ``ml.split_metadata``.

    Args:
        ratings_df: Cleaned ratings with columns userId, movieId, rating,
            timestamp (TimestampType).
        genre_exploded_df: Genre-exploded movies with columns movieId,
            title, genre.
        train_boundary: Epoch-seconds boundary (60th pctl) between train
            and validation splits.
        holdout_boundary: Epoch-seconds boundary (80th pctl) between
            validation and test/holdout splits.

    Returns:
        DataFrame with columns: userId, genre, label, split_flag
        where split_flag is one of ``train``, ``val``, ``test``.
    """
    genre_cols = genre_exploded_df.select("movieId", "genre")
    rated_genres = ratings_df.join(genre_cols, on="movieId")

    ts_long = F.col("timestamp").cast("long")

    with_flag = rated_genres.withColumn(
        "split_flag",
        F.when(ts_long <= F.lit(train_boundary), F.lit("train"))
        .when(ts_long <= F.lit(holdout_boundary), F.lit("val"))
        .otherwise(F.lit("test")),
    )

    labels = with_flag.groupBy("userId", "genre", "split_flag").agg(F.avg("rating").alias("label"))
    return labels.select("userId", "genre", "label", "split_flag")


def _pivot_to_multi_output(
    pdf: pd.DataFrame,
    genres: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pivot per-(userId, genre) data into multi-output training format.

    Transforms one-row-per-(userId, genre) pandas DataFrame into
    one-row-per-user format suitable for ``MultiOutputRegressor``.

    Args:
        pdf: Pandas DataFrame from Feature Store training set with columns
            including userId, genre, label, and feature columns.
        genres: Ordered list of genre names (defines output column order).

    Returns:
        ``(X, y)`` tuple of pandas DataFrames indexed by userId.
        X has user-profile features plus pivoted genre features.
        y has one column per genre (the label).
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


def _compute_per_genre_metrics(
    y_true: pd.DataFrame,
    y_pred: np.ndarray,
    genres: List[str],
) -> Dict[str, float]:
    """Compute RMSE for each genre column.

    Args:
        y_true: True labels with one column per genre (sorted order).
        y_pred: Predicted values as numpy array, same shape as y_true.
        genres: Ordered list of genre names matching column order.

    Returns:
        Dictionary mapping genre name to its RMSE.
    """
    sorted_genres = sorted(genres)
    result: Dict[str, float] = {}
    for i, genre in enumerate(sorted_genres):
        result[genre] = float(np.sqrt(mean_squared_error(y_true.iloc[:, i], y_pred[:, i])))
    return result


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def read_split_boundaries(spark: SparkSession, catalog: str) -> Dict[str, int]:
    """Read the persisted split boundaries from ``ml.split_metadata``.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.

    Returns:
        Dict with ``train_boundary_ts`` and ``holdout_boundary_ts``.

    Raises:
        RuntimeError: If the metadata table is empty or missing.
    """
    meta_tbl = table_name(catalog, SCHEMA_ML, TABLE_SPLIT_METADATA)
    try:
        row = spark.read.table(meta_tbl).collect()[-1]
    except Exception as exc:
        raise RuntimeError(
            f"Split metadata table {meta_tbl} is missing or empty. Run feature engineering first."
        ) from exc
    return {
        "train_boundary_ts": int(row["train_boundary_ts"]),
        "holdout_boundary_ts": int(row["holdout_boundary_ts"]),
    }


def build_training_labels(
    spark: SparkSession,
    catalog: str,
    model_variant: str,
) -> DataFrame:
    """Build per-(userId, genre) training labels from silver tables.

    Reads split boundaries from ``ml.split_metadata`` and produces a
    3-way ``split_flag`` (train/val/test).

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.
        model_variant: ``"general"`` (all 18 genres) or ``"nokids"``
            (15 genres, excluding Children/Animation/Fantasy).

    Returns:
        DataFrame with columns: userId, genre, label, split_flag.
    """
    boundaries = read_split_boundaries(spark, catalog)

    ratings_tbl = table_name(catalog, SCHEMA_SILVER, TABLE_RATINGS_CLEAN)

    if model_variant == "nokids":
        genre_tbl = table_name(catalog, SCHEMA_SILVER, TABLE_MOVIES_GENRE_EXPLODED_NOKIDS)
    else:
        genre_tbl = table_name(catalog, SCHEMA_SILVER, TABLE_MOVIES_GENRE_EXPLODED)

    ratings_df = spark.read.table(ratings_tbl)
    genre_exploded_df = spark.read.table(genre_tbl)

    return _build_training_labels_transform(
        ratings_df,
        genre_exploded_df,
        train_boundary=boundaries["train_boundary_ts"],
        holdout_boundary=boundaries["holdout_boundary_ts"],
    )


def create_training_set(
    spark: SparkSession,
    catalog: str,
    training_labels_df: DataFrame,
) -> Any:
    """Create a Feature Store training set with feature lookups.

    Joins the training labels with the user-genre and user-profile
    feature tables via ``FeatureLookup``.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.
        training_labels_df: DataFrame with userId, genre, label, split_flag.

    Returns:
        Feature Store training set object.
    """
    from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

    fe = FeatureEngineeringClient()
    lookups = [
        FeatureLookup(
            table_name=table_name(catalog, SCHEMA_GOLD, TABLE_USER_GENRE_FEATURES),
            lookup_key=["userId", "genre"],
        ),
        FeatureLookup(
            table_name=table_name(catalog, SCHEMA_GOLD, TABLE_USER_PROFILE_FEATURES),
            lookup_key=["userId"],
        ),
    ]
    return fe.create_training_set(
        df=training_labels_df,
        feature_lookups=lookups,
        label="label",
    )


def train_model(
    spark: SparkSession,
    catalog: str,
    model_variant: str,
    hyperparams: Dict[str, Any],
    sample_fraction: float,
) -> Tuple[str, str]:
    """Train a LightGBM multi-output model and register in Unity Catalog.

    End-to-end pipeline: builds labels, constructs a Feature Store training
    set, pivots to multi-output format, fits
    ``MultiOutputRegressor(LGBMRegressor())``, logs everything to MLflow,
    and registers the model.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.
        model_variant: ``"general"`` or ``"nokids"``.
        hyperparams: LightGBM hyperparameters dict.
        sample_fraction: Fraction of users used (logged as param).

    Returns:
        ``(registered_model_name, model_version)`` tuple.
    """
    import lightgbm
    import mlflow
    from lightgbm import LGBMRegressor
    from mlflow.models import infer_signature

    genres = GENRES if model_variant == "general" else NOKIDS_GENRES
    model_name_short = MODEL_GENERAL if model_variant == "general" else MODEL_NOKIDS
    registered_model_name = table_name(catalog, SCHEMA_ML, model_name_short)

    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"/Shared/genre_propensity/{catalog}/training")

    boundaries = read_split_boundaries(spark, catalog)

    training_labels_df = build_training_labels(spark, catalog, model_variant)

    # Restrict labels to only users present in the gold feature tables
    # (i.e. the sample_fraction subset written by feature_engineering).
    # Without this, the Feature Store LEFT JOIN processes all ~162K silver
    # users (~2.9M label rows) and the resulting shuffle fills the JVM heap
    # long before toPandas() is even called.
    gold_users = (
        spark.read.table(table_name(catalog, SCHEMA_GOLD, TABLE_USER_PROFILE_FEATURES)).select("userId").distinct()
    )
    training_labels_df = training_labels_df.join(gold_users, on="userId", how="inner")

    # Only use train + val splits for training; test/holdout is reserved
    # for the validation step's head-to-head champion comparison.
    train_val_labels = training_labels_df.filter(F.col("split_flag").isin("train", "val"))

    training_set = create_training_set(spark, catalog, train_val_labels)
    training_df = training_set.load_df().dropna(subset=["total_ratings"])
    full_pdf = training_df.toPandas()
    train_pdf = full_pdf[full_pdf["split_flag"] == "train"].copy()
    val_pdf = full_pdf[full_pdf["split_flag"] == "val"].copy()

    X_train, y_train = _pivot_to_multi_output(train_pdf, genres)
    X_val, y_val = _pivot_to_multi_output(val_pdf, genres)

    with mlflow.start_run(run_name=f"genre_propensity_{model_variant}"):
        try:
            dataset = mlflow.data.load_delta(table_name=table_name(catalog, SCHEMA_GOLD, TABLE_USER_GENRE_FEATURES))
            mlflow.log_input(dataset, context="training")
        except Exception:
            logger.warning("Failed to log input dataset for lineage tracking", exc_info=True)

        model = MultiOutputRegressor(LGBMRegressor(**hyperparams))
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)

        val_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        val_r2 = float(r2_score(y_val, y_pred_val))
        val_per_genre_rmse = _compute_per_genre_metrics(y_val, y_pred_val, genres)

        signature = infer_signature(X_train, y_pred_train)

        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path=f"genre_propensity_{model_variant}",
            signature=signature,
            input_example=X_train.iloc[:5],
            registered_model_name=registered_model_name,
            extra_pip_requirements=[f"lightgbm=={lightgbm.__version__}"],
        )

        mlflow.log_params(
            {
                **{k: str(v) for k, v in hyperparams.items()},
                "sample_fraction": str(sample_fraction),
                "train_boundary_ts": str(boundaries["train_boundary_ts"]),
                "holdout_boundary_ts": str(boundaries["holdout_boundary_ts"]),
            }
        )

        mlflow.log_metrics(
            {
                "val_rmse_overall": val_rmse,
                "val_r2_overall": val_r2,
                **{f"val_rmse_{genre}": v for genre, v in val_per_genre_rmse.items()},
            }
        )

        mlflow_client = mlflow.MlflowClient()
        mlflow_client.update_model_version(
            name=registered_model_name,
            version=model_info.registered_model_version,
            description=f"Genre propensity model ({model_variant}), trained with "
            f"sample_fraction={sample_fraction}, "
            f"val_RMSE={val_rmse:.4f}, val_R2={val_r2:.4f}",
        )

    model_version = model_info.registered_model_version

    return registered_model_name, str(model_version)
