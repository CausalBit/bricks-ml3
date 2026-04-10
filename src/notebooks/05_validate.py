# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 05 — Model Validation
# MAGIC
# MAGIC Runs all 10 validation checks on the freshly trained model.
# MAGIC Reads model name/version from task values set by the training notebook.
# MAGIC On pass, assigns the `@Challenger` alias and sets governance tags.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Unity Catalog catalog name")
dbutils.widgets.text("model_variant", "general", "Model variant: general or nokids")
dbutils.widgets.text("model_name", "", "Override: registered model name (skip taskValues lookup)")
dbutils.widgets.text("model_version", "", "Override: model version (skip taskValues lookup)")

# COMMAND ----------

from bricks_ml3.utils.spark_helpers import get_catalog
from bricks_ml3.validation.validate import run_validation

catalog = get_catalog(dbutils)
model_variant = dbutils.widgets.get("model_variant")
model_name = dbutils.widgets.get("model_name")
model_version = dbutils.widgets.get("model_version")
print(f"Validating {model_variant} model for catalog: {catalog}")

# COMMAND ----------

run_validation(spark, catalog, model_variant, dbutils, model_name=model_name, model_version=model_version)

print(f"Validation complete for {model_variant} model.")
