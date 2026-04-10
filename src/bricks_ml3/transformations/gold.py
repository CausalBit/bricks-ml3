"""Silver-to-gold feature engineering for the ML3 pipeline.

Two public functions build the Feature Store tables in the gold schema:
``build_user_genre_features`` and ``build_user_profile_features``.  Each
delegates to pure-DataFrame helpers (prefixed with ``_``) that are
testable without table I/O or Feature Store dependencies.
"""

from __future__ import annotations

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from bricks_ml3.config.settings import (
    DECAY_LAMBDA,
    SCHEMA_BRONZE,
    SCHEMA_GOLD,
    SCHEMA_ML,
    SCHEMA_SILVER,
    TABLE_GENOME_GENRE_AGG,
    TABLE_MOVIES_GENRE_EXPLODED,
    TABLE_RATINGS_CLEAN,
    TABLE_SPLIT_METADATA,
    TABLE_TAGS,
    TABLE_USER_GENRE_FEATURES,
    TABLE_USER_PROFILE_FEATURES,
    TRAIN_TEST_SPLIT_PERCENTILE,
    TRAIN_VAL_PERCENTILE,
)
from bricks_ml3.utils.spark_helpers import table_name

# ---------------------------------------------------------------------------
# Pure transformation helpers (no table I/O – used by unit tests)
# ---------------------------------------------------------------------------


def _sample_users(df: DataFrame, sample_fraction: float, seed: int = 42) -> DataFrame:
    """Filter DataFrame to a random fraction of distinct userIds.

    Args:
        df: DataFrame containing a ``userId`` column.
        sample_fraction: Fraction of distinct users to keep (0.0 to 1.0).
        seed: Random seed for reproducibility.

    Returns:
        Filtered DataFrame containing only the sampled users' rows.
    """
    if sample_fraction >= 1.0:
        return df
    user_ids = df.select("userId").distinct().sample(fraction=sample_fraction, seed=seed)
    return df.join(F.broadcast(user_ids), on="userId", how="inner")


def _compute_user_genre_features_transform(
    ratings_df: DataFrame,
    genre_exploded_df: DataFrame,
    genome_agg_df: DataFrame,
    tags_df: DataFrame,
) -> DataFrame:
    """Compute all seven user-genre features from pure DataFrames.

    Args:
        ratings_df: Cleaned ratings with columns userId, movieId, rating,
            timestamp (TimestampType).
        genre_exploded_df: Genre-exploded movies with columns movieId,
            title, genre.
        genome_agg_df: Genome-tag aggregation with columns movieId, genre,
            avg_genome_relevance.
        tags_df: Raw user tags with columns userId, movieId, tag, timestamp.

    Returns:
        DataFrame with columns: userId, genre, genre_avg_rating,
        genre_watch_count, genre_recency_score, genre_share,
        genre_diversity_index, genre_avg_genome_relevance,
        genre_tag_count, feature_timestamp.
    """
    genre_cols = genre_exploded_df.select("movieId", "genre")

    rated_genres = ratings_df.join(genre_cols, on="movieId")

    user_max_ts = (
        ratings_df.withColumn("_ts_long", F.col("timestamp").cast("long"))
        .groupBy("userId")
        .agg(F.max("_ts_long").alias("_user_max_ts"))
    )

    rated_with_ts = (
        rated_genres.join(user_max_ts, on="userId")
        .withColumn("_ts_long", F.col("timestamp").cast("long"))
        .withColumn(
            "_days_since",
            (F.col("_user_max_ts") - F.col("_ts_long")) / F.lit(86400),
        )
        .withColumn(
            "_weighted_rating",
            F.col("rating") * F.exp(-F.lit(DECAY_LAMBDA) * F.col("_days_since")),
        )
    )

    base = rated_with_ts.groupBy("userId", "genre").agg(
        F.avg("rating").alias("genre_avg_rating"),
        F.count("*").alias("genre_watch_count"),
        (F.sum("_weighted_rating") / F.count("*")).alias("genre_recency_score"),
        F.max("timestamp").alias("feature_timestamp"),
    )

    user_total = base.groupBy("userId").agg(F.sum("genre_watch_count").alias("_total_count"))
    features = base.join(user_total, on="userId").withColumn(
        "genre_share",
        F.col("genre_watch_count") / F.col("_total_count"),
    )

    entropy = (
        features.withColumn("_p_log_p", F.col("genre_share") * F.log(F.col("genre_share")))
        .groupBy("userId")
        .agg((-F.sum("_p_log_p")).alias("genre_diversity_index"))
    )
    features = features.join(entropy, on="userId")

    genome_join = (
        rated_genres.select("userId", "movieId", "genre")
        .join(genome_agg_df, on=["movieId", "genre"], how="left")
        .groupBy("userId", "genre")
        .agg(F.avg("avg_genome_relevance").alias("genre_avg_genome_relevance"))
    )
    features = features.join(genome_join, on=["userId", "genre"], how="left")

    tags_with_genre = tags_df.join(genre_cols, on="movieId")
    tag_counts = tags_with_genre.groupBy("userId", "genre").agg(F.countDistinct("tag").alias("genre_tag_count"))
    features = features.join(tag_counts, on=["userId", "genre"], how="left")
    features = features.fillna(0, subset=["genre_tag_count"])

    return features.select(
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
    )


def _compute_user_profile_features_transform(
    ratings_df: DataFrame,
    genre_exploded_df: DataFrame,
) -> DataFrame:
    """Compute user-level aggregate features from pure DataFrames.

    Args:
        ratings_df: Cleaned ratings with columns userId, movieId, rating,
            timestamp (TimestampType).
        genre_exploded_df: Genre-exploded movies with columns movieId,
            title, genre.

    Returns:
        DataFrame with columns: userId, total_ratings, avg_rating,
        active_days, distinct_genres, diversity_index, profile_timestamp.
    """
    user_stats = (
        ratings_df.withColumn("_ts_long", F.col("timestamp").cast("long"))
        .groupBy("userId")
        .agg(
            F.count("*").alias("total_ratings"),
            F.avg("rating").alias("avg_rating"),
            ((F.max("_ts_long") - F.min("_ts_long")) / F.lit(86400)).alias("active_days"),
            F.max("timestamp").alias("profile_timestamp"),
        )
    )

    rated_genres = ratings_df.join(genre_exploded_df.select("movieId", "genre"), on="movieId")
    genre_stats = rated_genres.groupBy("userId").agg(F.countDistinct("genre").alias("distinct_genres"))

    genre_counts = rated_genres.groupBy("userId", "genre").agg(F.count("*").alias("_gc"))
    user_total = genre_counts.groupBy("userId").agg(F.sum("_gc").alias("_total"))
    genre_shares = genre_counts.join(user_total, on="userId").withColumn("_p", F.col("_gc") / F.col("_total"))
    entropy = (
        genre_shares.withColumn("_p_log_p", F.col("_p") * F.log(F.col("_p")))
        .groupBy("userId")
        .agg((-F.sum("_p_log_p")).alias("diversity_index"))
    )

    result = user_stats.join(genre_stats, on="userId").join(entropy, on="userId")

    return result.select(
        "userId",
        "total_ratings",
        "avg_rating",
        "active_days",
        "distinct_genres",
        "diversity_index",
        "profile_timestamp",
    )


# ---------------------------------------------------------------------------
# Public orchestration functions (read tables → transform → register)
# ---------------------------------------------------------------------------


def build_user_genre_features(spark: SparkSession, catalog: str, sample_fraction: float = 1.0) -> DataFrame:
    """Build and register the user-genre feature table in the gold schema.

    Reads silver tables, samples users, computes all seven per-(userId, genre)
    features, and registers (or updates) the Feature Store table.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.
        sample_fraction: Fraction of distinct users to include.

    Returns:
        The computed user-genre features DataFrame.
    """
    from databricks.feature_engineering import FeatureEngineeringClient

    ratings_df = spark.read.table(table_name(catalog, SCHEMA_SILVER, TABLE_RATINGS_CLEAN))
    genre_exploded_df = spark.read.table(table_name(catalog, SCHEMA_SILVER, TABLE_MOVIES_GENRE_EXPLODED))
    genome_agg_df = spark.read.table(table_name(catalog, SCHEMA_SILVER, TABLE_GENOME_GENRE_AGG))
    tags_df = spark.read.table(table_name(catalog, SCHEMA_BRONZE, TABLE_TAGS))

    ratings_df = _sample_users(ratings_df, sample_fraction)
    sampled_user_ids = ratings_df.select("userId").distinct()
    tags_df = tags_df.join(F.broadcast(sampled_user_ids), on="userId", how="inner")

    features_df = _compute_user_genre_features_transform(ratings_df, genre_exploded_df, genome_agg_df, tags_df)

    fe = FeatureEngineeringClient()
    tbl = table_name(catalog, SCHEMA_GOLD, TABLE_USER_GENRE_FEATURES)

    try:
        fe.create_table(
            name=tbl,
            primary_keys=["userId", "genre"],
            df=features_df,
            description="Per-user per-genre features for genre propensity model",
        )
    except Exception:
        fe.write_table(name=tbl, df=features_df, mode="merge")

    return features_df


def build_user_profile_features(spark: SparkSession, catalog: str, sample_fraction: float = 1.0) -> DataFrame:
    """Build and register the user profile feature table in the gold schema.

    Reads silver tables, samples users, computes all five per-user features,
    and registers (or updates) the Feature Store table.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.
        sample_fraction: Fraction of distinct users to include.

    Returns:
        The computed user profile features DataFrame.
    """
    from databricks.feature_engineering import FeatureEngineeringClient

    ratings_df = spark.read.table(table_name(catalog, SCHEMA_SILVER, TABLE_RATINGS_CLEAN))
    genre_exploded_df = spark.read.table(table_name(catalog, SCHEMA_SILVER, TABLE_MOVIES_GENRE_EXPLODED))

    ratings_df = _sample_users(ratings_df, sample_fraction)

    features_df = _compute_user_profile_features_transform(ratings_df, genre_exploded_df)

    fe = FeatureEngineeringClient()
    tbl = table_name(catalog, SCHEMA_GOLD, TABLE_USER_PROFILE_FEATURES)

    try:
        fe.create_table(
            name=tbl,
            primary_keys=["userId"],
            df=features_df,
            description="User-level aggregate features",
        )
    except Exception:
        fe.write_table(name=tbl, df=features_df, mode="merge")

    return features_df


def compute_and_persist_split_boundaries(spark: SparkSession, catalog: str) -> dict:
    """Compute train/val/test timestamp boundaries from ``ratings_clean``.

    Reads the full ``ratings_clean`` population (no genre join, no
    downsampling) and computes the 60th and 80th percentile timestamp
    boundaries.  Writes a single-row metadata table to ``ml.split_metadata``
    so that both training and validation read the same deterministic splits.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name.

    Returns:
        Dict with keys ``train_boundary_ts``, ``holdout_boundary_ts``,
        ``ratings_clean_row_count``, ``min_timestamp``, ``max_timestamp``.
    """
    import datetime

    from pyspark.sql import Row
    from pyspark.sql import types as T

    ratings_tbl = table_name(catalog, SCHEMA_SILVER, TABLE_RATINGS_CLEAN)
    ratings_df = spark.read.table(ratings_tbl)

    ts_long = F.col("timestamp").cast("long")
    boundaries = ratings_df.withColumn("_ts", ts_long).stat.approxQuantile(
        "_ts",
        [TRAIN_VAL_PERCENTILE, TRAIN_TEST_SPLIT_PERCENTILE],
        0.0,
    )
    train_boundary_ts = int(boundaries[0])
    holdout_boundary_ts = int(boundaries[1])

    stats = ratings_df.agg(
        F.count("*").alias("row_count"),
        F.min(ts_long).alias("min_ts"),
        F.max(ts_long).alias("max_ts"),
    ).collect()[0]

    metadata = {
        "train_boundary_ts": train_boundary_ts,
        "holdout_boundary_ts": holdout_boundary_ts,
        "ratings_clean_row_count": int(stats["row_count"]),
        "min_timestamp": int(stats["min_ts"]),
        "max_timestamp": int(stats["max_ts"]),
    }

    schema = T.StructType(
        [
            T.StructField("computed_at", T.TimestampType(), False),
            T.StructField("train_boundary_ts", T.LongType(), False),
            T.StructField("holdout_boundary_ts", T.LongType(), False),
            T.StructField("ratings_clean_row_count", T.LongType(), False),
            T.StructField("min_timestamp", T.LongType(), False),
            T.StructField("max_timestamp", T.LongType(), False),
        ]
    )

    row = Row(
        computed_at=datetime.datetime.now(datetime.timezone.utc),
        **metadata,
    )
    meta_df = spark.createDataFrame([row], schema=schema)

    dst = table_name(catalog, SCHEMA_ML, TABLE_SPLIT_METADATA)
    meta_df.write.mode("overwrite").saveAsTable(dst)

    return metadata
