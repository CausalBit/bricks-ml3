"""Fixtures for unit tests — local SparkSession, no Databricks cluster needed.

``databricks-connect`` overrides PySpark and blocks local SparkSession
creation.  This module installs a vanilla PySpark to a temp directory and
prepends it to ``sys.path`` so that unit tests can create a local session.

This file intentionally lives under tests/unit/ (not tests/) so that the
path hack does NOT run when collecting integration tests, which need the
real databricks-connect SparkSession.Hook machinery.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

SAMPLE_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sample"

_STANDALONE_PYSPARK_DIR = "/tmp/standalone-pyspark"


def _ensure_standalone_pyspark() -> None:
    """Install standalone pyspark to /tmp/standalone-pyspark if not present."""
    marker = Path(_STANDALONE_PYSPARK_DIR) / "pyspark"
    if not marker.exists():
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pyspark", "--target", _STANDALONE_PYSPARK_DIR, "-q"],
            check=True,
        )


_ensure_standalone_pyspark()
if _STANDALONE_PYSPARK_DIR not in sys.path:
    sys.path.insert(0, _STANDALONE_PYSPARK_DIR)


@pytest.fixture(scope="session")
def spark():
    """Create a local SparkSession shared across the entire test session."""
    from pyspark.sql import SparkSession

    return (
        SparkSession.builder.master("local[*]")
        .appName("bricks-ml3-tests")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "4")
        .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse-test")
        .getOrCreate()
    )


@pytest.fixture()
def sample_ratings_df(spark):
    """Load ``data/sample/ratings_sample.csv`` as a DataFrame."""
    return spark.read.csv(
        str(SAMPLE_DIR / "ratings_sample.csv"),
        header=True,
        inferSchema=True,
    )


@pytest.fixture()
def sample_movies_df(spark):
    """Load ``data/sample/movies_sample.csv`` as a DataFrame."""
    return spark.read.csv(
        str(SAMPLE_DIR / "movies_sample.csv"),
        header=True,
        inferSchema=True,
    )


@pytest.fixture()
def sample_genome_scores_df(spark):
    """Load ``data/sample/genome_scores_sample.csv`` as a DataFrame."""
    return spark.read.csv(
        str(SAMPLE_DIR / "genome_scores_sample.csv"),
        header=True,
        inferSchema=True,
    )


@pytest.fixture()
def sample_genome_tags_df(spark):
    """Load ``data/sample/genome_tags_sample.csv`` as a DataFrame."""
    return spark.read.csv(
        str(SAMPLE_DIR / "genome_tags_sample.csv"),
        header=True,
        inferSchema=True,
    )


@pytest.fixture()
def sample_tags_df(spark):
    """Load ``data/sample/tags_sample.csv`` as a DataFrame."""
    return spark.read.csv(
        str(SAMPLE_DIR / "tags_sample.csv"),
        header=True,
        inferSchema=True,
    )


@pytest.fixture()
def sample_links_df(spark):
    """Load ``data/sample/links_sample.csv`` as a DataFrame."""
    return spark.read.csv(
        str(SAMPLE_DIR / "links_sample.csv"),
        header=True,
        inferSchema=True,
    )
