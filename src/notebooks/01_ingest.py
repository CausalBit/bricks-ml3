# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 01 — Data Ingestion (Bronze Layer)
# MAGIC
# MAGIC Reads the six ML-25M CSV files from the Unity Catalog landing volume
# MAGIC and writes them as raw Delta tables in the bronze schema.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Unity Catalog catalog name")

# COMMAND ----------

from bricks_ml3.ingestion.ingest import ingest_all
from bricks_ml3.utils.spark_helpers import get_catalog

catalog = get_catalog(dbutils)
print(f"Ingesting into catalog: {catalog}")

# COMMAND ----------

results = ingest_all(spark, catalog)

for table, df in results.items():
    print(f"  {catalog}.bronze.{table}: {df.count()} rows")

print("Bronze ingestion complete.")
