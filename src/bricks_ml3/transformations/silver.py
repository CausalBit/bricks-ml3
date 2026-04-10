"""Bronze-to-silver transformations for the ML3 pipeline.

Five public functions orchestrate reading from bronze/silver tables, applying
transformations, and writing results.  Each delegates to a pure-DataFrame
helper (prefixed with ``_``) so the core logic is testable without table I/O.
"""

from __future__ import annotations

from typing import Tuple

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from bricks_ml3.config.settings import (
    HOLDOUT_PERCENTILE,
    NOKIDS_EXCLUDE_GENRES,
    SCHEMA_BRONZE,
    SCHEMA_SILVER,
    TABLE_GENOME_GENRE_AGG,
    TABLE_GENOME_SCORES,
    TABLE_GENOME_TAGS,
    TABLE_MOVIES,
    TABLE_MOVIES_GENRE_EXPLODED,
    TABLE_MOVIES_GENRE_EXPLODED_NOKIDS,
    TABLE_RATINGS,
    TABLE_RATINGS_CLEAN,
    TABLE_RATINGS_HOLDOUT,
)
from bricks_ml3.utils.spark_helpers import table_name

# ---------------------------------------------------------------------------
# Pure transformation helpers (no table I/O – used by unit tests)
# ---------------------------------------------------------------------------


def _clean_ratings_transform(ratings_df: DataFrame) -> DataFrame:
    """Drop nulls, deduplicate on (userId, movieId), and cast timestamp.

    Keeps the row with the latest ``timestamp`` for each (userId, movieId)
    pair, then casts the epoch-seconds ``timestamp`` to ``TimestampType``.

    Args:
        ratings_df: Raw ratings with columns userId, movieId, rating,
            timestamp (LongType).

    Returns:
        Cleaned DataFrame with TimestampType timestamp.
    """
    cleaned = ratings_df.dropna()

    window = Window.partitionBy("userId", "movieId").orderBy(F.col("timestamp").desc())
    deduped = cleaned.withColumn("_rn", F.row_number().over(window)).filter(F.col("_rn") == 1).drop("_rn")

    return deduped.withColumn("timestamp", F.col("timestamp").cast("timestamp"))


def _explode_genres_transform(movies_df: DataFrame) -> DataFrame:
    """Split pipe-delimited genres and explode into one row per genre.

    Rows where ``genre == "(no genres listed)"`` are excluded.

    Args:
        movies_df: Raw movies with columns movieId, title, genres.

    Returns:
        DataFrame with columns movieId, title, genre.
    """
    exploded = movies_df.withColumn("genre", F.explode(F.split(F.col("genres"), "\\|")))
    filtered = exploded.filter(F.col("genre") != "(no genres listed)")
    return filtered.select("movieId", "title", "genre")


def _filter_nokids_transform(genre_exploded_df: DataFrame) -> DataFrame:
    """Remove genres listed in ``NOKIDS_EXCLUDE_GENRES``.

    Args:
        genre_exploded_df: Output of :func:`_explode_genres_transform`.

    Returns:
        Filtered DataFrame excluding kid-oriented genres.
    """
    return genre_exploded_df.filter(~F.col("genre").isin(NOKIDS_EXCLUDE_GENRES))


def _aggregate_genome_transform(
    genome_scores_df: DataFrame,
    genome_tags_df: DataFrame,
    genre_exploded_df: DataFrame,
) -> DataFrame:
    """Average genome-tag relevance per (movieId, genre).

    Joins ``genome_scores`` → ``genome_tags`` → ``movies_genre_exploded`` and
    computes the mean relevance across all genome tags for each movie-genre
    pair.

    Args:
        genome_scores_df: Columns movieId, tagId, relevance.
        genome_tags_df: Columns tagId, tag.
        genre_exploded_df: Columns movieId, title, genre.

    Returns:
        DataFrame with columns movieId, genre, avg_genome_relevance.
    """
    joined = genome_scores_df.join(genome_tags_df, on="tagId").join(
        genre_exploded_df.select("movieId", "genre"), on="movieId"
    )
    return joined.groupBy("movieId", "genre").agg(F.avg("relevance").alias("avg_genome_relevance"))


def _split_holdout_transform(
    ratings_clean_df: DataFrame,
) -> Tuple[DataFrame, DataFrame]:
    """Split cleaned ratings at the 80th-percentile timestamp.

    Rows **at or before** the boundary go to the training set; rows
    **after** the boundary go to the holdout set.

    Args:
        ratings_clean_df: Cleaned ratings with TimestampType timestamp.

    Returns:
        ``(training_df, holdout_df)`` tuple.
    """
    ts_long = F.col("timestamp").cast("long")

    boundary = (ratings_clean_df.withColumn("_ts", ts_long).stat.approxQuantile("_ts", [HOLDOUT_PERCENTILE], 0.0))[0]

    training = ratings_clean_df.filter(ts_long <= boundary)
    holdout = ratings_clean_df.filter(ts_long > boundary)
    return training, holdout


# ---------------------------------------------------------------------------
# Public orchestration functions (read tables → transform → write tables)
# ---------------------------------------------------------------------------


def clean_ratings(spark: SparkSession, catalog: str) -> DataFrame:
    """Clean and deduplicate bronze ratings, write to silver.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.

    Returns:
        The cleaned ratings DataFrame.
    """
    src = table_name(catalog, SCHEMA_BRONZE, TABLE_RATINGS)
    dst = table_name(catalog, SCHEMA_SILVER, TABLE_RATINGS_CLEAN)

    raw_df = spark.read.table(src)
    clean_df = _clean_ratings_transform(raw_df)
    clean_df.write.mode("overwrite").saveAsTable(dst)
    return clean_df


def explode_genres(spark: SparkSession, catalog: str) -> DataFrame:
    """Explode movie genres into one row per genre, write to silver.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.

    Returns:
        The genre-exploded DataFrame.
    """
    src = table_name(catalog, SCHEMA_BRONZE, TABLE_MOVIES)
    dst = table_name(catalog, SCHEMA_SILVER, TABLE_MOVIES_GENRE_EXPLODED)

    movies_df = spark.read.table(src)
    exploded_df = _explode_genres_transform(movies_df)
    exploded_df.write.mode("overwrite").saveAsTable(dst)
    return exploded_df


def create_nokids_variant(spark: SparkSession, catalog: str) -> DataFrame:
    """Filter genre-exploded movies to exclude kid-oriented genres.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.

    Returns:
        The no-kids genre-exploded DataFrame.
    """
    src = table_name(catalog, SCHEMA_SILVER, TABLE_MOVIES_GENRE_EXPLODED)
    dst = table_name(catalog, SCHEMA_SILVER, TABLE_MOVIES_GENRE_EXPLODED_NOKIDS)

    genre_df = spark.read.table(src)
    nokids_df = _filter_nokids_transform(genre_df)
    nokids_df.write.mode("overwrite").saveAsTable(dst)
    return nokids_df


def aggregate_genome(spark: SparkSession, catalog: str) -> DataFrame:
    """Aggregate genome-tag relevance per (movieId, genre), write to silver.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.

    Returns:
        The aggregated genome DataFrame.
    """
    scores_tbl = table_name(catalog, SCHEMA_BRONZE, TABLE_GENOME_SCORES)
    tags_tbl = table_name(catalog, SCHEMA_BRONZE, TABLE_GENOME_TAGS)
    genre_tbl = table_name(catalog, SCHEMA_SILVER, TABLE_MOVIES_GENRE_EXPLODED)
    dst = table_name(catalog, SCHEMA_SILVER, TABLE_GENOME_GENRE_AGG)

    scores_df = spark.read.table(scores_tbl)
    tags_df = spark.read.table(tags_tbl)
    genre_df = spark.read.table(genre_tbl)

    agg_df = _aggregate_genome_transform(scores_df, tags_df, genre_df)
    agg_df.write.mode("overwrite").saveAsTable(dst)
    return agg_df


def split_holdout(spark: SparkSession, catalog: str) -> Tuple[DataFrame, DataFrame]:
    """Split cleaned ratings into training and holdout sets.

    Reads the already-written ``ratings_clean``, computes the 80th-percentile
    timestamp boundary, and overwrites ``ratings_clean`` with only the
    training portion.  The holdout portion is written to ``ratings_holdout``.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.

    Returns:
        ``(training_df, holdout_df)`` tuple.
    """
    src = table_name(catalog, SCHEMA_SILVER, TABLE_RATINGS_CLEAN)
    holdout_tbl = table_name(catalog, SCHEMA_SILVER, TABLE_RATINGS_HOLDOUT)

    ratings_df = spark.read.table(src).cache()

    training_df, holdout_df = _split_holdout_transform(ratings_df)

    training_df.write.mode("overwrite").saveAsTable(src)
    holdout_df.write.mode("overwrite").saveAsTable(holdout_tbl)

    ratings_df.unpersist()
    return training_df, holdout_df
