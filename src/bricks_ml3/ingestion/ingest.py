"""CSV-to-Delta ingestion for the ML-25M dataset.

Reads raw CSV files from a Unity Catalog volume and writes them as Delta
tables in the bronze schema.  No transformations are applied -- bronze
stores data exactly as ingested.
"""

from __future__ import annotations

from typing import Dict, List

from pyspark.sql import DataFrame, SparkSession

from bricks_ml3.config.settings import (
    CSV_FILES,
    SCHEMA_BRONZE,
    VOLUME_LANDING,
)
from bricks_ml3.utils.spark_helpers import table_name, volume_path


def ingest_csv_to_delta(
    spark: SparkSession,
    volume_base: str,
    filename: str,
    catalog: str,
    schema: str,
    target_table: str,
) -> DataFrame:
    """Read a single CSV from a Unity Catalog volume and persist as Delta.

    Args:
        spark: Active Spark session.
        volume_base: Base volume path (e.g. ``/Volumes/dev/bronze/landing``).
        filename: Name of the CSV file inside the volume.
        catalog: Unity Catalog catalog name.
        schema: Target schema name.
        target_table: Target table name (without catalog/schema prefix).

    Returns:
        The ingested DataFrame.
    """
    csv_path = f"{volume_base}/{filename}"
    df = spark.read.csv(csv_path, header=True, inferSchema=True)

    full_table = table_name(catalog, schema, target_table)
    df.write.mode("overwrite").saveAsTable(full_table)

    return df


def ingest_all(spark: SparkSession, catalog: str) -> Dict[str, DataFrame]:
    """Ingest all six ML-25M CSV files into bronze Delta tables.

    Reads each CSV from ``/Volumes/{catalog}/bronze/landing/`` and writes
    it as a Delta table in ``{catalog}.bronze.*``.

    Args:
        spark: Active Spark session.
        catalog: Unity Catalog catalog name (e.g. ``"dev"`` or ``"prod"``).

    Returns:
        A dictionary mapping table names to their ingested DataFrames.
    """
    vol_base = volume_path(catalog, SCHEMA_BRONZE, VOLUME_LANDING)

    table_map: List[tuple[str, str]] = [
        (CSV_FILES["ratings"], "ratings"),
        (CSV_FILES["movies"], "movies"),
        (CSV_FILES["tags"], "tags"),
        (CSV_FILES["genome_scores"], "genome_scores"),
        (CSV_FILES["genome_tags"], "genome_tags"),
        (CSV_FILES["links"], "links"),
    ]

    results: Dict[str, DataFrame] = {}
    for csv_filename, tbl in table_map:
        df = ingest_csv_to_delta(
            spark=spark,
            volume_base=vol_base,
            filename=csv_filename,
            catalog=catalog,
            schema=SCHEMA_BRONZE,
            target_table=tbl,
        )
        results[tbl] = df

    return results
