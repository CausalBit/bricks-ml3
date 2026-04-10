# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 00 — Download & Ingest MovieLens 25M Dataset
# MAGIC
# MAGIC Downloads the ML-25M zip from GroupLens directly onto the cluster,
# MAGIC extracts the 6 CSV files, and writes them as Delta tables in the
# MAGIC bronze schema. Reads CSVs from local disk to avoid volume upload
# MAGIC size limitations.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Unity Catalog catalog name")

# COMMAND ----------

import os
import zipfile
from urllib.request import urlretrieve

import pandas as pd

from bricks_ml3.config.settings import CSV_FILES, SCHEMA_BRONZE
from bricks_ml3.utils.spark_helpers import get_catalog, table_name

catalog = get_catalog(dbutils)
dataset_url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"

print(f"Catalog: {catalog}")

# COMMAND ----------

# Check if bronze tables already exist (idempotent)
tables_needed = []
for tbl in CSV_FILES.keys():
    full_name = table_name(catalog, SCHEMA_BRONZE, tbl)
    try:
        count = spark.table(full_name).count()
        print(f"  {full_name}: {count} rows (exists)")
    except Exception:
        tables_needed.append(tbl)
        print(f"  {full_name}: MISSING")

if not tables_needed:
    print("\nAll bronze tables already exist — skipping download.")
    dbutils.notebook.exit("skipped")

print(f"\nNeed to create {len(tables_needed)} table(s)")

# COMMAND ----------

# Download and extract to cluster-local temp directory
tmpdir = "/tmp/ml-25m-data"
os.makedirs(tmpdir, exist_ok=True)
zip_path = os.path.join(tmpdir, "ml-25m.zip")

if not os.path.exists(zip_path):
    print("Downloading ML-25M dataset (~250 MB)...")
    urlretrieve(dataset_url, zip_path)
    size_mb = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"Downloaded {size_mb:.0f} MB")
else:
    print("Using cached zip from previous run")

print("Extracting CSV files...")
with zipfile.ZipFile(zip_path, "r") as zf:
    for entry in CSV_FILES.values():
        zf.extract(f"ml-25m/{entry}", tmpdir)
        size_mb = os.path.getsize(f"{tmpdir}/ml-25m/{entry}") / (1024 * 1024)
        print(f"  {entry:25s} {size_mb:.0f} MB")

# COMMAND ----------

# Read CSVs from local disk and write as Delta tables
for tbl, csv_filename in CSV_FILES.items():
    full_name = table_name(catalog, SCHEMA_BRONZE, tbl)

    # Skip tables that already exist
    if tbl not in tables_needed:
        continue

    local_path = f"{tmpdir}/ml-25m/{csv_filename}"
    print(f"  {csv_filename} → {full_name}...", end="", flush=True)

    # Read with pandas on the driver (file is local to driver only),
    # then convert to Spark DataFrame for Delta write.
    pdf = pd.read_csv(local_path)
    df = spark.createDataFrame(pdf)
    df.write.mode("overwrite").saveAsTable(full_name)

    print(f" {len(pdf)} rows")

print("\nBronze ingestion complete.")
