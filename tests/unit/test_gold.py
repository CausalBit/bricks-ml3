"""Unit tests for silver-to-gold feature engineering.

All tests run on a local SparkSession -- no Databricks cluster required.
They exercise the pure-DataFrame helper functions in
``bricks_ml3.transformations.gold``.
"""

from __future__ import annotations

import math

import pytest
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

from bricks_ml3.config.settings import DECAY_LAMBDA
from bricks_ml3.transformations.gold import (
    _compute_user_genre_features_transform,
    _compute_user_profile_features_transform,
    _sample_users,
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


def _make_genome_agg(spark: SparkSession, data: list) -> "DataFrame":
    """Create a genome aggregation DataFrame."""
    schema = StructType(
        [
            StructField("movieId", IntegerType()),
            StructField("genre", StringType()),
            StructField("avg_genome_relevance", DoubleType()),
        ]
    )
    return spark.createDataFrame(data, schema)


def _make_tags(spark: SparkSession, data: list) -> "DataFrame":
    """Create a tags DataFrame."""
    schema = StructType(
        [
            StructField("userId", IntegerType()),
            StructField("movieId", IntegerType()),
            StructField("tag", StringType()),
            StructField("timestamp", LongType()),
        ]
    )
    return spark.createDataFrame(data, schema)


# -- recency score -----------------------------------------------------------


@pytest.mark.unit
def test_recency_score_decay_math(spark: SparkSession) -> None:
    """Recency score matches hand-calculated exponential decay.

    Two ratings at t=0 days ago (rating=4.0) and t=100 days ago (rating=3.0):
    (4.0 * exp(0) + 3.0 * exp(-0.001 * 100)) / 2
    """
    latest_ts = 1_000_000_000
    earlier_ts = latest_ts - 100 * 86400

    ratings = _make_ratings(
        spark,
        [
            (1, 1, 4.0, latest_ts),
            (1, 2, 3.0, earlier_ts),
        ],
    )
    genres = _make_genre_exploded(
        spark,
        [
            (1, "M1", "Action"),
            (2, "M2", "Action"),
        ],
    )
    genome = _make_genome_agg(spark, [])
    tags = _make_tags(spark, [])

    result = _compute_user_genre_features_transform(ratings, genres, genome, tags)

    row = result.filter((F.col("userId") == 1) & (F.col("genre") == "Action")).collect()[0]

    expected = (4.0 * math.exp(0) + 3.0 * math.exp(-DECAY_LAMBDA * 100)) / 2
    assert abs(row["genre_recency_score"] - expected) < 1e-4


# -- Shannon entropy ---------------------------------------------------------


@pytest.mark.unit
def test_shannon_entropy_single_genre(spark: SparkSession) -> None:
    """User with only one genre has diversity index = 0.0."""
    ratings = _make_ratings(
        spark,
        [
            (1, 1, 4.0, 1_000_000_000),
            (1, 2, 5.0, 1_000_000_100),
        ],
    )
    genres = _make_genre_exploded(
        spark,
        [
            (1, "M1", "Drama"),
            (2, "M2", "Drama"),
        ],
    )
    genome = _make_genome_agg(spark, [])
    tags = _make_tags(spark, [])

    result = _compute_user_genre_features_transform(ratings, genres, genome, tags)
    row = result.collect()[0]

    assert abs(row["genre_diversity_index"] - 0.0) < 1e-6


@pytest.mark.unit
def test_shannon_entropy_uniform(spark: SparkSession) -> None:
    """User with equal ratings across N genres has entropy = ln(N)."""
    ratings = _make_ratings(
        spark,
        [
            (1, 1, 4.0, 1_000_000_000),
            (1, 2, 4.0, 1_000_000_100),
            (1, 3, 4.0, 1_000_000_200),
        ],
    )
    genres = _make_genre_exploded(
        spark,
        [
            (1, "M1", "Action"),
            (2, "M2", "Comedy"),
            (3, "M3", "Drama"),
        ],
    )
    genome = _make_genome_agg(spark, [])
    tags = _make_tags(spark, [])

    result = _compute_user_genre_features_transform(ratings, genres, genome, tags)
    row = result.filter(F.col("genre") == "Action").collect()[0]

    expected_entropy = math.log(3)
    assert abs(row["genre_diversity_index"] - expected_entropy) < 1e-4


# -- genre share -------------------------------------------------------------


@pytest.mark.unit
def test_genre_share_sums_to_one(spark: SparkSession) -> None:
    """Sum of genre_share across all genres for any user equals 1.0."""
    ratings = _make_ratings(
        spark,
        [
            (1, 1, 4.0, 1_000_000_000),
            (1, 2, 3.5, 1_000_000_100),
            (1, 3, 5.0, 1_000_000_200),
            (1, 4, 4.5, 1_000_000_300),
        ],
    )
    genres = _make_genre_exploded(
        spark,
        [
            (1, "M1", "Action"),
            (1, "M1", "Adventure"),
            (2, "M2", "Comedy"),
            (3, "M3", "Action"),
            (3, "M3", "Drama"),
            (4, "M4", "Drama"),
        ],
    )
    genome = _make_genome_agg(spark, [])
    tags = _make_tags(spark, [])

    result = _compute_user_genre_features_transform(ratings, genres, genome, tags)

    total_share = result.filter(F.col("userId") == 1).agg(F.sum("genre_share").alias("total")).collect()[0]["total"]

    assert abs(total_share - 1.0) < 1e-6


# -- sample fraction ---------------------------------------------------------


@pytest.mark.unit
def test_sample_fraction_reduces_users(spark: SparkSession) -> None:
    """sample_fraction=0.5 on 20 users produces roughly 10 users."""
    schema = StructType(
        [
            StructField("userId", IntegerType()),
            StructField("value", DoubleType()),
        ]
    )
    data = [(i, float(i)) for i in range(1, 21)]
    df = spark.createDataFrame(data, schema)

    sampled = _sample_users(df, sample_fraction=0.5)
    n_users = sampled.select("userId").distinct().count()

    assert 3 <= n_users <= 17


# -- column checks -----------------------------------------------------------


@pytest.mark.unit
def test_all_feature_columns_present(spark: SparkSession) -> None:
    """Output DataFrame contains exactly the expected user-genre columns."""
    ratings = _make_ratings(spark, [(1, 1, 4.0, 1_000_000_000)])
    genres = _make_genre_exploded(spark, [(1, "M1", "Action")])
    genome = _make_genome_agg(spark, [])
    tags = _make_tags(spark, [])

    result = _compute_user_genre_features_transform(ratings, genres, genome, tags)

    expected_cols = {
        "userId",
        "genre",
        "genre_avg_rating",
        "genre_watch_count",
        "genre_recency_score",
        "genre_share",
        "genre_diversity_index",
        "genre_avg_genome_relevance",
        "genre_tag_count",
        "feature_timestamp",
    }
    assert set(result.columns) == expected_cols


@pytest.mark.unit
def test_profile_features_columns(spark: SparkSession) -> None:
    """Output DataFrame contains exactly the expected user-profile columns."""
    ratings = _make_ratings(
        spark,
        [
            (1, 1, 4.0, 1_000_000_000),
            (1, 2, 3.5, 1_000_100_000),
        ],
    )
    genres = _make_genre_exploded(
        spark,
        [
            (1, "M1", "Action"),
            (2, "M2", "Drama"),
        ],
    )

    result = _compute_user_profile_features_transform(ratings, genres)

    expected_cols = {
        "userId",
        "total_ratings",
        "avg_rating",
        "active_days",
        "distinct_genres",
        "diversity_index",
        "profile_timestamp",
    }
    assert set(result.columns) == expected_cols
