# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 02 — Silver ETL (Bronze → Silver)
# MAGIC
# MAGIC Runs all bronze-to-silver transformations in order:
# MAGIC 1. Clean and deduplicate ratings
# MAGIC 2. Explode movie genres
# MAGIC 3. Create no-kids genre variant
# MAGIC 4. Aggregate genome-tag relevance per genre
# MAGIC 5. Split ratings into training and holdout sets

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Unity Catalog catalog name")

# COMMAND ----------

from bricks_ml3.transformations.silver import (
    aggregate_genome,
    clean_ratings,
    create_nokids_variant,
    explode_genres,
    split_holdout,
)
from bricks_ml3.utils.spark_helpers import get_catalog

catalog = get_catalog(dbutils)
print(f"Running silver ETL for catalog: {catalog}")

# COMMAND ----------

clean_df = clean_ratings(spark, catalog)
print(f"  {catalog}.silver.ratings_clean: {clean_df.count()} rows")

# COMMAND ----------

exploded_df = explode_genres(spark, catalog)
print(f"  {catalog}.silver.movies_genre_exploded: {exploded_df.count()} rows")

# COMMAND ----------

nokids_df = create_nokids_variant(spark, catalog)
print(f"  {catalog}.silver.movies_genre_exploded_nokids: {nokids_df.count()} rows")

# COMMAND ----------

genome_df = aggregate_genome(spark, catalog)
print(f"  {catalog}.silver.genome_genre_agg: {genome_df.count()} rows")

# COMMAND ----------

training_df, holdout_df = split_holdout(spark, catalog)
print(f"  {catalog}.silver.ratings_clean (training): {training_df.count()} rows")
print(f"  {catalog}.silver.ratings_holdout: {holdout_df.count()} rows")

# COMMAND ----------

print("Silver ETL complete.")
