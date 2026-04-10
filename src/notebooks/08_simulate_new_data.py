# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 08 — Simulate New Data
# MAGIC
# MAGIC Moves the next time-window slice from `silver.ratings_holdout` into
# MAGIC `silver.ratings_clean`, then updates the Feature Store tables so
# MAGIC downstream scoring reflects the newly arrived data.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Unity Catalog catalog name")
dbutils.widgets.text("days_window", "", "Number of days per simulation slice")
dbutils.widgets.text("sample_fraction", "0.2", "Fraction of users to include")

# COMMAND ----------

from pyspark.sql import functions as F

from bricks_ml3.config.settings import (
    SCHEMA_SILVER,
    SIMULATION_DAYS_WINDOW,
    TABLE_RATINGS_CLEAN,
    TABLE_RATINGS_HOLDOUT,
)
from bricks_ml3.utils.spark_helpers import get_catalog, get_sample_fraction, table_name

catalog = get_catalog(dbutils)
days_window_raw = dbutils.widgets.get("days_window")
days_window = int(days_window_raw) if days_window_raw else SIMULATION_DAYS_WINDOW
sample_fraction = get_sample_fraction(dbutils)
print(f"Simulating new data for catalog: {catalog}, window: {days_window} days")

# COMMAND ----------

holdout_tbl = table_name(catalog, SCHEMA_SILVER, TABLE_RATINGS_HOLDOUT)
clean_tbl = table_name(catalog, SCHEMA_SILVER, TABLE_RATINGS_CLEAN)

holdout_df = spark.read.table(holdout_tbl)

if holdout_df.count() == 0:
    print("No holdout data remaining — nothing to simulate.")
    dbutils.notebook.exit("NO_HOLDOUT_DATA")

min_ts = holdout_df.agg(F.min(F.col("timestamp").cast("long")).alias("min_ts")).collect()[0]["min_ts"]

boundary_ts = min_ts + (days_window * 86400)

slice_df = holdout_df.filter(F.col("timestamp").cast("long") <= boundary_ts)
remaining_df = holdout_df.filter(F.col("timestamp").cast("long") > boundary_ts)

slice_count = slice_df.count()
print(f"Moving {slice_count} ratings from holdout to clean")

# COMMAND ----------

slice_df.write.mode("append").saveAsTable(clean_tbl)
remaining_df.write.mode("overwrite").saveAsTable(holdout_tbl)

print(f"Appended {slice_count} rows to {clean_tbl}")
print(f"Remaining holdout rows: {remaining_df.count()}")

# COMMAND ----------

from bricks_ml3.transformations.gold import (
    build_user_genre_features,
    build_user_profile_features,
)

genre_features_df = build_user_genre_features(spark, catalog, sample_fraction)
print(f"Updated user_genre_features: {genre_features_df.count()} rows (mode=merge)")

profile_features_df = build_user_profile_features(spark, catalog, sample_fraction)
print(f"Updated user_profile_features: {profile_features_df.count()} rows (mode=merge)")

# COMMAND ----------

print("New data simulation complete.")
