# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 07 — Batch Scoring
# MAGIC
# MAGIC Scores all users with the `@Champion` model using Feature Store
# MAGIC `score_batch` and writes results to the inference table.
# MAGIC Parameterized by `model_variant` (`general` or `nokids`).

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Unity Catalog catalog name")
dbutils.widgets.text("model_variant", "general", "Model variant: general or nokids")

# COMMAND ----------

from bricks_ml3.inference.batch_score import score_all_users
from bricks_ml3.utils.spark_helpers import get_catalog

catalog = get_catalog(dbutils)
model_variant = dbutils.widgets.get("model_variant")
print(f"Batch scoring with {model_variant} model for catalog: {catalog}")

# COMMAND ----------

row_count = score_all_users(spark, catalog, model_variant)
print(f"Scored {row_count} users for {model_variant} model.")

# COMMAND ----------

print("Batch scoring complete.")
