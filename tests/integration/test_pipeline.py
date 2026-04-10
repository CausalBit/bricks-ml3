"""Integration tests for the ML3 pipeline.

These tests require a running Databricks workspace and Databricks Connect.
They validate that bronze, silver, gold, model, and inference artifacts
exist and contain expected schemas/data after the pipeline has been run.

Run with:
    pytest tests/integration/ -m integration -v
"""

from __future__ import annotations

import pytest
from databricks.connect import DatabricksSession

from bricks_ml3.config.settings import (
    MODEL_GENERAL,
    MODEL_NOKIDS,
    SCHEMA_BRONZE,
    SCHEMA_GOLD,
    SCHEMA_INFERENCE,
    SCHEMA_ML,
    SCHEMA_SILVER,
    TABLE_GENOME_GENRE_AGG,
    TABLE_GENOME_SCORES,
    TABLE_GENOME_TAGS,
    TABLE_LINKS,
    TABLE_MOVIES,
    TABLE_MOVIES_GENRE_EXPLODED,
    TABLE_MOVIES_GENRE_EXPLODED_NOKIDS,
    TABLE_RATINGS,
    TABLE_RATINGS_CLEAN,
    TABLE_RATINGS_HOLDOUT,
    TABLE_SCORES_DAILY,
    TABLE_SCORES_DAILY_NOKIDS,
    TABLE_TAGS,
    TABLE_USER_GENRE_FEATURES,
    TABLE_USER_PROFILE_FEATURES,
)
from bricks_ml3.utils.spark_helpers import table_name

CATALOG = "dev"


@pytest.fixture(scope="module")
def spark():
    """Create a DatabricksSession via Databricks Connect (DEFAULT profile)."""
    return DatabricksSession.builder.getOrCreate()


@pytest.mark.integration
class TestBronzeTablesExist:
    """Verify all 6 bronze tables are readable."""

    @pytest.mark.parametrize(
        "tbl",
        [
            TABLE_RATINGS,
            TABLE_MOVIES,
            TABLE_TAGS,
            TABLE_GENOME_SCORES,
            TABLE_GENOME_TAGS,
            TABLE_LINKS,
        ],
    )
    def test_bronze_table_readable(self, spark, tbl):
        fqn = table_name(CATALOG, SCHEMA_BRONZE, tbl)
        df = spark.read.table(fqn)
        assert df.count() > 0, f"{fqn} is empty"


@pytest.mark.integration
class TestSilverTablesExist:
    """Verify silver tables have expected schemas."""

    def test_ratings_clean_schema(self, spark):
        fqn = table_name(CATALOG, SCHEMA_SILVER, TABLE_RATINGS_CLEAN)
        df = spark.read.table(fqn)
        expected_cols = {"userId", "movieId", "rating", "timestamp"}
        assert expected_cols.issubset(set(df.columns))
        assert df.count() > 0

    def test_movies_genre_exploded_schema(self, spark):
        fqn = table_name(CATALOG, SCHEMA_SILVER, TABLE_MOVIES_GENRE_EXPLODED)
        df = spark.read.table(fqn)
        assert "movieId" in df.columns
        assert "genre" in df.columns
        assert df.count() > 0

    def test_movies_genre_exploded_nokids_schema(self, spark):
        fqn = table_name(CATALOG, SCHEMA_SILVER, TABLE_MOVIES_GENRE_EXPLODED_NOKIDS)
        df = spark.read.table(fqn)
        assert "genre" in df.columns
        assert df.count() > 0

    def test_genome_genre_agg_schema(self, spark):
        fqn = table_name(CATALOG, SCHEMA_SILVER, TABLE_GENOME_GENRE_AGG)
        df = spark.read.table(fqn)
        expected_cols = {"movieId", "genre", "avg_genome_relevance"}
        assert expected_cols.issubset(set(df.columns))

    def test_ratings_holdout_exists(self, spark):
        fqn = table_name(CATALOG, SCHEMA_SILVER, TABLE_RATINGS_HOLDOUT)
        df = spark.read.table(fqn)
        assert df.columns is not None


@pytest.mark.integration
class TestFeatureTablesExist:
    """Verify gold Feature Store tables are populated."""

    def test_user_genre_features(self, spark):
        fqn = table_name(CATALOG, SCHEMA_GOLD, TABLE_USER_GENRE_FEATURES)
        df = spark.read.table(fqn)
        expected_cols = {"userId", "genre", "genre_avg_rating", "genre_watch_count"}
        assert expected_cols.issubset(set(df.columns))
        assert df.count() > 0

    def test_user_profile_features(self, spark):
        fqn = table_name(CATALOG, SCHEMA_GOLD, TABLE_USER_PROFILE_FEATURES)
        df = spark.read.table(fqn)
        expected_cols = {"userId", "total_ratings", "avg_rating"}
        assert expected_cols.issubset(set(df.columns))
        assert df.count() > 0


@pytest.mark.integration
class TestModelRegistered:
    """Verify models exist in Unity Catalog registry."""

    def test_general_model_registered(self):
        from mlflow import MlflowClient

        client = MlflowClient()
        model_fqn = table_name(CATALOG, SCHEMA_ML, MODEL_GENERAL)
        versions = client.search_model_versions(f"name='{model_fqn}'")
        assert len(versions) > 0, f"No versions found for {model_fqn}"

    def test_nokids_model_registered(self):
        from mlflow import MlflowClient

        client = MlflowClient()
        model_fqn = table_name(CATALOG, SCHEMA_ML, MODEL_NOKIDS)
        versions = client.search_model_versions(f"name='{model_fqn}'")
        assert len(versions) > 0, f"No versions found for {model_fqn}"


@pytest.mark.integration
class TestInferenceTableHasData:
    """Verify scoring output is non-empty."""

    def test_general_scores_non_empty(self, spark):
        fqn = table_name(CATALOG, SCHEMA_INFERENCE, TABLE_SCORES_DAILY)
        df = spark.read.table(fqn)
        assert df.count() > 0, f"{fqn} is empty"

    def test_nokids_scores_non_empty(self, spark):
        fqn = table_name(CATALOG, SCHEMA_INFERENCE, TABLE_SCORES_DAILY_NOKIDS)
        df = spark.read.table(fqn)
        assert df.count() > 0, f"{fqn} is empty"
