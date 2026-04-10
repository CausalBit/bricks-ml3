"""Unit tests for LightGBM multi-output model training.

All tests run on a local SparkSession -- no Databricks cluster required.
They exercise the pure helpers in ``bricks_ml3.training.train`` and verify
that ``MultiOutputRegressor(LGBMRegressor())`` fits/predicts correctly on
synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from lightgbm import LGBMRegressor
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)
from sklearn.multioutput import MultiOutputRegressor

from bricks_ml3.training.train import (
    _build_training_labels_transform,
    _compute_per_genre_metrics,
    _pivot_to_multi_output,
)

# -- helpers to build test DataFrames ----------------------------------------


def _make_ratings(spark: SparkSession, data: list) -> "DataFrame":
    """Create a ratings DataFrame with TimestampType timestamp."""
    schema = StructType(
        [
            StructField("userId", IntegerType()),
            StructField("movieId", IntegerType()),
            StructField("rating", DoubleType()),
            StructField("timestamp", LongType()),
        ]
    )
    df = spark.createDataFrame(data, schema)
    return df.withColumn("timestamp", F.col("timestamp").cast("timestamp"))


def _make_genre_exploded(spark: SparkSession, data: list) -> "DataFrame":
    """Create a genre-exploded movies DataFrame."""
    schema = StructType(
        [
            StructField("movieId", IntegerType()),
            StructField("title", StringType()),
            StructField("genre", StringType()),
        ]
    )
    return spark.createDataFrame(data, schema)


# -- test_model_fits_on_synthetic_data ---------------------------------------


@pytest.mark.unit
def test_model_fits_on_synthetic_data() -> None:
    """MultiOutputRegressor(LGBMRegressor) fits and predicts without error."""
    rng = np.random.RandomState(42)
    n_users = 10
    n_features = 5
    genres = ["Action", "Comedy", "Drama"]

    X_train = pd.DataFrame(
        rng.rand(n_users, n_features),
        columns=[f"feat_{i}" for i in range(n_features)],
    )
    y_train = pd.DataFrame(
        rng.rand(n_users, len(genres)) * 4.5 + 0.5,
        columns=sorted(genres),
    )

    model = MultiOutputRegressor(LGBMRegressor(n_estimators=10, verbosity=-1))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_train)

    assert y_pred.shape == (n_users, len(genres))
    assert not np.isnan(y_pred).any()


# -- test_label_construction_split_ratio -------------------------------------


@pytest.mark.unit
def test_label_construction_split_ratio(spark: SparkSession) -> None:
    """80/20 chronological split assigns earlier ratings to train.

    Uses users that appear in only one time period so the aggregated
    label rows clearly reflect the underlying 80/20 timestamp split.
    """
    base_ts = 1_000_000_000
    step = 1000

    ratings_data = []
    genres_data = []
    mid = 1
    for uid in range(1, 21):
        ts = base_ts + uid * step
        ratings_data.append((uid, mid, 4.0, ts))
        genres_data.append((mid, f"M{mid}", "Action"))
        mid += 1

    ratings_df = _make_ratings(spark, ratings_data)
    genre_df = _make_genre_exploded(spark, genres_data)

    # 60th / 80th percentile of the 20 timestamps
    train_boundary = base_ts + 12 * step
    holdout_boundary = base_ts + 16 * step

    labels = _build_training_labels_transform(ratings_df, genre_df, train_boundary, holdout_boundary)

    total = labels.count()
    train_count = labels.filter(F.col("split_flag") == "train").count()
    val_count = labels.filter(F.col("split_flag") == "val").count()
    test_count = labels.filter(F.col("split_flag") == "test").count()

    assert total == train_count + val_count + test_count
    assert train_count > 0, "Expected some train labels"
    assert test_count > 0, "Expected some test labels"
    train_ratio = train_count / total
    assert 0.5 <= train_ratio <= 0.95, f"Unexpected train ratio: {train_ratio}"


# -- test_per_genre_rmse_computed --------------------------------------------


@pytest.mark.unit
def test_per_genre_rmse_computed() -> None:
    """Per-genre RMSE dict contains one entry per genre, all positive."""
    genres = ["Action", "Comedy", "Drama"]
    sorted_genres = sorted(genres)
    rng = np.random.RandomState(42)

    y_true = pd.DataFrame(
        rng.rand(20, len(genres)) * 4.5 + 0.5,
        columns=sorted_genres,
    )
    y_pred = y_true.values + rng.randn(20, len(genres)) * 0.3

    rmse_dict = _compute_per_genre_metrics(y_true, y_pred, genres)

    assert set(rmse_dict.keys()) == set(genres)
    for genre, val in rmse_dict.items():
        assert val > 0.0, f"RMSE for {genre} should be positive"


# -- test_pivot_to_multi_output_shape ----------------------------------------


@pytest.mark.unit
def test_pivot_to_multi_output_shape() -> None:
    """Pivoted X and y have the expected row/column counts."""
    genres = ["Action", "Comedy", "Drama"]
    n_users = 5
    rows = []
    for uid in range(1, n_users + 1):
        for genre in genres:
            rows.append(
                {
                    "userId": uid,
                    "genre": genre,
                    "label": float(uid) / 2.0,
                    "total_ratings": 10.0,
                    "avg_rating": 3.5,
                    "active_days": 100.0,
                    "distinct_genres": 3.0,
                    "diversity_index": 1.0,
                    "genre_avg_rating": 3.0 + uid * 0.1,
                    "genre_watch_count": float(uid * 2),
                    "genre_recency_score": 2.5,
                    "genre_share": 1.0 / 3.0,
                    "genre_diversity_index": 1.0,
                    "genre_avg_genome_relevance": 0.5,
                    "genre_tag_count": 1.0,
                }
            )

    pdf = pd.DataFrame(rows)
    X, y = _pivot_to_multi_output(pdf, genres)

    assert X.shape[0] == n_users
    assert y.shape == (n_users, len(genres))

    expected_x_cols = 5 + 7 * len(genres)
    assert X.shape[1] == expected_x_cols


# -- test_label_columns_present -----------------------------------------------


@pytest.mark.unit
def test_label_columns_present(spark: SparkSession) -> None:
    """_build_training_labels_transform returns the four required columns."""
    ratings_data = [
        (1, 1, 4.0, 1_000_000_000),
        (1, 2, 3.0, 1_000_100_000),
        (2, 1, 5.0, 1_000_200_000),
    ]
    genres_data = [
        (1, "M1", "Action"),
        (2, "M2", "Drama"),
    ]

    ratings_df = _make_ratings(spark, ratings_data)
    genre_df = _make_genre_exploded(spark, genres_data)

    labels = _build_training_labels_transform(
        ratings_df,
        genre_df,
        train_boundary=1_000_100_000,
        holdout_boundary=1_000_150_000,
    )

    expected_cols = {"userId", "genre", "label", "split_flag"}
    assert set(labels.columns) == expected_cols
