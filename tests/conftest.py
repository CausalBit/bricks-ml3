"""Root conftest — shared across all test directories.

Intentionally empty. The pyspark path hack and local SparkSession fixture
live in tests/unit/conftest.py so they don't interfere with integration
tests that need the real databricks-connect import chain.
"""
