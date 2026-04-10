"""Batch inference using direct feature lookup + model predict.

Scores all users from ``gold.user_genre_features`` by loading features
directly, running the ``@Champion`` model, and writing per-user genre
propensity scores to the inference table.
"""

from __future__ import annotations

import logging

import pandas as pd
from pyspark.sql import SparkSession

from bricks_ml3.config.settings import (
    GENRES,
    MODEL_GENERAL,
    MODEL_NOKIDS,
    NOKIDS_GENRES,
    SCHEMA_GOLD,
    SCHEMA_INFERENCE,
    SCHEMA_ML,
    TABLE_SCORES_DAILY,
    TABLE_SCORES_DAILY_NOKIDS,
    TABLE_USER_GENRE_FEATURES,
    TABLE_USER_PROFILE_FEATURES,
)
from bricks_ml3.training.train import _pivot_to_multi_output
from bricks_ml3.utils.spark_helpers import table_name

logger = logging.getLogger(__name__)

_USER_GENRE_FEATURE_COLS = [
    "genre_avg_rating",
    "genre_watch_count",
    "genre_recency_score",
    "genre_share",
    "genre_diversity_index",
    "genre_avg_genome_relevance",
    "genre_tag_count",
]

_USER_PROFILE_FEATURE_COLS = [
    "total_ratings",
    "avg_rating",
    "active_days",
    "distinct_genres",
    "diversity_index",
]


def score_all_users(
    spark: SparkSession,
    catalog: str,
    model_variant: str,
) -> int:
    """Score all users with the ``@Champion`` model and write to inference table.

    Loads user features directly from the gold feature tables, pivots to
    multi-output format, runs the Champion model, and writes per-user genre
    propensity scores to the inference table.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.
        model_variant: ``"general"`` or ``"nokids"``.

    Returns:
        The number of rows written to the inference table.
    """
    import mlflow

    model_name_short = MODEL_GENERAL if model_variant == "general" else MODEL_NOKIDS
    registered_model_name = table_name(catalog, SCHEMA_ML, model_name_short)
    model_uri = f"models:/{registered_model_name}@Champion"
    genres = GENRES if model_variant == "general" else NOKIDS_GENRES
    sorted_genres = sorted(genres)

    if model_variant == "nokids":
        inference_table = table_name(catalog, SCHEMA_INFERENCE, TABLE_SCORES_DAILY_NOKIDS)
    else:
        inference_table = table_name(catalog, SCHEMA_INFERENCE, TABLE_SCORES_DAILY)

    # Load Champion model first so we know the exact feature columns it expects
    mlflow.set_registry_uri("databricks-uc")
    model_meta = mlflow.models.get_model_info(model_uri)
    expected_cols = [col.name for col in model_meta.signature.inputs]

    from pyspark.sql import functions as F

    ug_df = spark.read.table(table_name(catalog, SCHEMA_GOLD, TABLE_USER_GENRE_FEATURES))
    ug_df = ug_df.filter(F.col("genre").isin(genres))
    up_df = spark.read.table(table_name(catalog, SCHEMA_GOLD, TABLE_USER_PROFILE_FEATURES))

    # Join and collect to pandas
    features_df = ug_df.join(up_df, on="userId", how="left")
    features_pdf = features_df.toPandas()

    # Add dummy label column so _pivot_to_multi_output can work
    features_pdf["label"] = 0.0

    # Build feature matrix then align columns to exactly what the model expects
    X, _ = _pivot_to_multi_output(features_pdf, genres)
    # Add any missing columns (unseen genres) as 0, drop any extra columns
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0.0
    X = X[expected_cols]

    # Load Champion model and predict
    model = mlflow.pyfunc.load_model(model_uri)
    y_pred = model.predict(X)

    # Build output DataFrame: userId + one column per genre
    scores_pdf = pd.DataFrame(y_pred, index=X.index, columns=sorted_genres)
    scores_pdf.index.name = "userId"
    scores_pdf = scores_pdf.reset_index()

    scored_df = spark.createDataFrame(scores_pdf)
    scored_df.write.mode("overwrite").saveAsTable(inference_table)

    row_count = scored_df.count()
    logger.info(
        "Wrote %d rows to %s using model %s",
        row_count,
        inference_table,
        model_uri,
    )
    return row_count
