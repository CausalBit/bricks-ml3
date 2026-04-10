# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 03 — Feature Engineering (Gold Layer)
# MAGIC
# MAGIC Builds the Feature Store tables in the gold schema:
# MAGIC - `user_genre_features`: 7 per-(userId, genre) features
# MAGIC - `user_profile_features`: 5 per-user aggregate features

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Unity Catalog catalog name")
dbutils.widgets.text("sample_fraction", "0.2", "Fraction of users to include")

# COMMAND ----------

from bricks_ml3.transformations.gold import (
    build_user_genre_features,
    build_user_profile_features,
    compute_and_persist_split_boundaries,
)
from bricks_ml3.utils.spark_helpers import get_catalog, get_sample_fraction

catalog = get_catalog(dbutils)
sample_fraction = get_sample_fraction(dbutils)
print(f"Building features for catalog: {catalog}, sample_fraction: {sample_fraction}")

# COMMAND ----------

genre_features_df = build_user_genre_features(spark, catalog, sample_fraction)
print(f"  {catalog}.gold.user_genre_features: {genre_features_df.count()} rows")

# COMMAND ----------

profile_features_df = build_user_profile_features(spark, catalog, sample_fraction)
print(f"  {catalog}.gold.user_profile_features: {profile_features_df.count()} rows")

# COMMAND ----------

split_meta = compute_and_persist_split_boundaries(spark, catalog)
print(f"  Split boundaries persisted to {catalog}.ml.split_metadata:")
print(f"    train_boundary_ts  = {split_meta['train_boundary_ts']}")
print(f"    holdout_boundary_ts = {split_meta['holdout_boundary_ts']}")
print(f"    ratings_clean rows  = {split_meta['ratings_clean_row_count']}")

# COMMAND ----------

print("Feature engineering complete.")
