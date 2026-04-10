"""Unit tests for bronze-to-silver transformations.

All tests run on a local SparkSession -- no Databricks cluster required.
They exercise the pure-DataFrame helper functions in
``bricks_ml3.transformations.silver``.
"""

from __future__ import annotations

import pytest
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
)

from bricks_ml3.config.settings import NOKIDS_EXCLUDE_GENRES
from bricks_ml3.transformations.silver import (
    _aggregate_genome_transform,
    _clean_ratings_transform,
    _explode_genres_transform,
    _filter_nokids_transform,
    _split_holdout_transform,
)

# -- clean_ratings -----------------------------------------------------------


@pytest.mark.unit
def test_clean_ratings_dedup(spark: SparkSession) -> None:
    """Two ratings for the same (userId, movieId) -> keep only the latest."""
    schema = StructType(
        [
            StructField("userId", IntegerType()),
            StructField("movieId", IntegerType()),
            StructField("rating", DoubleType()),
            StructField("timestamp", LongType()),
        ]
    )
    data = [
        (1, 10, 3.0, 1000),
        (1, 10, 4.5, 2000),  # later timestamp – should survive
        (1, 20, 5.0, 1500),
    ]
    df = spark.createDataFrame(data, schema)
    result = _clean_ratings_transform(df)

    rows = result.filter((F.col("userId") == 1) & (F.col("movieId") == 10)).collect()

    assert len(rows) == 1
    assert rows[0]["rating"] == 4.5


@pytest.mark.unit
def test_clean_ratings_drops_nulls(spark: SparkSession) -> None:
    """Rows with null userId or movieId are removed."""
    schema = StructType(
        [
            StructField("userId", IntegerType()),
            StructField("movieId", IntegerType()),
            StructField("rating", DoubleType()),
            StructField("timestamp", LongType()),
        ]
    )
    data = [
        (1, 10, 4.0, 1000),
        (None, 20, 3.0, 1100),
        (2, None, 5.0, 1200),
        (3, 30, 4.5, 1300),
    ]
    df = spark.createDataFrame(data, schema)
    result = _clean_ratings_transform(df)

    assert result.count() == 2
    user_ids = {r["userId"] for r in result.collect()}
    assert user_ids == {1, 3}


# -- explode_genres ----------------------------------------------------------


@pytest.mark.unit
def test_explode_genres_row_count(spark: SparkSession) -> None:
    """A movie with 3 pipe-separated genres produces 3 rows."""
    schema = StructType(
        [
            StructField("movieId", IntegerType()),
            StructField("title", StringType()),
            StructField("genres", StringType()),
        ]
    )
    data = [(99, "Test Movie", "Action|Crime|Thriller")]
    df = spark.createDataFrame(data, schema)
    result = _explode_genres_transform(df)

    assert result.count() == 3
    genres = {r["genre"] for r in result.collect()}
    assert genres == {"Action", "Crime", "Thriller"}


@pytest.mark.unit
def test_explode_genres_excludes_no_genres(spark: SparkSession) -> None:
    """Movies with '(no genres listed)' produce zero rows."""
    schema = StructType(
        [
            StructField("movieId", IntegerType()),
            StructField("title", StringType()),
            StructField("genres", StringType()),
        ]
    )
    data = [
        (1, "Good Movie", "Drama|Comedy"),
        (2, "Unknown", "(no genres listed)"),
    ]
    df = spark.createDataFrame(data, schema)
    result = _explode_genres_transform(df)

    movie_ids = {r["movieId"] for r in result.collect()}
    assert 2 not in movie_ids
    assert result.count() == 2  # Drama + Comedy from movie 1


# -- nokids variant ----------------------------------------------------------


@pytest.mark.unit
def test_nokids_variant_excludes_three_genres(spark: SparkSession) -> None:
    """Children, Animation, and Fantasy are absent; other genres remain."""
    schema = StructType(
        [
            StructField("movieId", IntegerType()),
            StructField("title", StringType()),
            StructField("genre", StringType()),
        ]
    )
    data = [
        (1, "M1", "Action"),
        (1, "M1", "Animation"),
        (2, "M2", "Children"),
        (3, "M3", "Fantasy"),
        (4, "M4", "Drama"),
        (5, "M5", "Comedy"),
    ]
    df = spark.createDataFrame(data, schema)
    result = _filter_nokids_transform(df)

    remaining = {r["genre"] for r in result.collect()}
    for excluded in NOKIDS_EXCLUDE_GENRES:
        assert excluded not in remaining

    assert remaining == {"Action", "Drama", "Comedy"}


# -- genome aggregation ------------------------------------------------------


@pytest.mark.unit
def test_genome_aggregation_one_row_per_movie_genre(
    spark: SparkSession,
) -> None:
    """Output has exactly one row per unique (movieId, genre) pair."""
    scores_schema = StructType(
        [
            StructField("movieId", IntegerType()),
            StructField("tagId", IntegerType()),
            StructField("relevance", DoubleType()),
        ]
    )
    tags_schema = StructType(
        [
            StructField("tagId", IntegerType()),
            StructField("tag", StringType()),
        ]
    )
    genre_schema = StructType(
        [
            StructField("movieId", IntegerType()),
            StructField("title", StringType()),
            StructField("genre", StringType()),
        ]
    )

    scores = spark.createDataFrame(
        [
            (1, 1, 0.5),
            (1, 2, 0.8),
            (1, 3, 0.3),
            (3, 1, 0.9),
            (3, 2, 0.4),
        ],
        scores_schema,
    )
    tags = spark.createDataFrame(
        [(1, "atmospheric"), (2, "funny"), (3, "dark")],
        tags_schema,
    )
    genres = spark.createDataFrame(
        [
            (1, "Toy Story", "Adventure"),
            (1, "Toy Story", "Comedy"),
            (3, "Heat", "Action"),
            (3, "Heat", "Crime"),
        ],
        genre_schema,
    )

    result = _aggregate_genome_transform(scores, tags, genres)

    pairs = result.select("movieId", "genre").distinct().count()
    assert pairs == result.count()

    expected_pairs = {(1, "Adventure"), (1, "Comedy"), (3, "Action"), (3, "Crime")}
    actual_pairs = {(r["movieId"], r["genre"]) for r in result.collect()}
    assert actual_pairs == expected_pairs


# -- holdout split -----------------------------------------------------------


@pytest.mark.unit
def test_holdout_split_boundary(spark: SparkSession) -> None:
    """Rows at or before the 80th-percentile timestamp go to training."""
    schema = StructType(
        [
            StructField("userId", IntegerType()),
            StructField("movieId", IntegerType()),
            StructField("rating", DoubleType()),
            StructField("timestamp", LongType()),
        ]
    )
    # 10 rows with timestamps 1..10; 80th percentile = 8
    data = [(i, i, 4.0, i) for i in range(1, 11)]
    df = spark.createDataFrame(data, schema)
    df = df.withColumn("timestamp", F.col("timestamp").cast("timestamp"))

    training, holdout = _split_holdout_transform(df)

    # 80th percentile of [1..10] = 8 → training gets ts <= 8 (8 rows),
    # holdout gets ts > 8 (2 rows)
    assert training.count() == 8
    assert holdout.count() == 2


@pytest.mark.unit
def test_holdout_split_no_overlap(spark: SparkSession) -> None:
    """No (userId, movieId, timestamp) appears in both training and holdout."""
    schema = StructType(
        [
            StructField("userId", IntegerType()),
            StructField("movieId", IntegerType()),
            StructField("rating", DoubleType()),
            StructField("timestamp", LongType()),
        ]
    )
    data = [(u, m, 4.0, t) for u, m, t in zip(range(1, 21), range(101, 121), range(1000, 1200, 10))]
    df = spark.createDataFrame(data, schema)
    df = df.withColumn("timestamp", F.col("timestamp").cast("timestamp"))

    training, holdout = _split_holdout_transform(df)

    key_cols = ["userId", "movieId", "timestamp"]
    overlap = training.select(key_cols).intersect(holdout.select(key_cols))
    assert overlap.count() == 0
