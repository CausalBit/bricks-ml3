"""Utility functions for Spark session management and catalog/schema resolution.

Every notebook imports from this module to obtain the SparkSession and resolve
the catalog parameter that DABs injects via widget defaults.
"""

from __future__ import annotations

import os
from typing import Any, Optional


def get_spark_session(use_serverless: bool = True) -> Any:
    """Return a DatabricksSession connected via the DEFAULT profile.

    Args:
        use_serverless: If True, request serverless compute. Falls back to
            the cluster_id in the DEFAULT profile when False.

    Returns:
        A ``DatabricksSession`` (PySpark-compatible ``SparkSession``).
    """
    from databricks.connect import DatabricksSession

    builder = DatabricksSession.builder
    if use_serverless:
        builder = builder.serverless(True)
    return builder.getOrCreate()


def get_catalog(dbutils: Optional[Any] = None) -> str:
    """Resolve the ``catalog`` parameter from a notebook widget or env var.

    Resolution order:
    1. ``dbutils.widgets.get("catalog")`` if *dbutils* is provided.
    2. The ``CATALOG`` environment variable.
    3. Falls back to ``"dev"``.

    Args:
        dbutils: Databricks ``dbutils`` object (available inside notebooks).

    Returns:
        The catalog name string.
    """
    if dbutils is not None:
        try:
            return dbutils.widgets.get("catalog")
        except Exception:
            pass
    return os.getenv("CATALOG", "dev")


def get_sample_fraction(dbutils: Optional[Any] = None) -> float:
    """Resolve the ``sample_fraction`` parameter from a widget or env var.

    Resolution order:
    1. ``dbutils.widgets.get("sample_fraction")`` if *dbutils* is provided.
    2. The ``SAMPLE_FRACTION`` environment variable.
    3. Falls back to ``0.2``.

    Args:
        dbutils: Databricks ``dbutils`` object.

    Returns:
        A float between 0.0 and 1.0.
    """
    if dbutils is not None:
        try:
            return float(dbutils.widgets.get("sample_fraction"))
        except Exception:
            pass
    return float(os.getenv("SAMPLE_FRACTION", "0.2"))


def table_name(catalog: str, schema: str, table: str) -> str:
    """Build a fully-qualified Unity Catalog table name.

    Args:
        catalog: Catalog name (e.g. ``"dev"`` or ``"prod"``).
        schema: Schema name (e.g. ``"bronze"``).
        table: Table name (e.g. ``"ratings"``).

    Returns:
        ``"{catalog}.{schema}.{table}"``
    """
    return f"{catalog}.{schema}.{table}"


def volume_path(catalog: str, schema: str, volume: str) -> str:
    """Build the DBFS path for a Unity Catalog volume.

    Args:
        catalog: Catalog name.
        schema: Schema name.
        volume: Volume name.

    Returns:
        ``"/Volumes/{catalog}/{schema}/{volume}"``
    """
    return f"/Volumes/{catalog}/{schema}/{volume}"
