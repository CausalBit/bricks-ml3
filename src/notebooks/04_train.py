# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 04 — Model Training
# MAGIC
# MAGIC Trains a LightGBM multi-output genre propensity model, logs to MLflow,
# MAGIC and registers in Unity Catalog. Parameterized by `model_variant`
# MAGIC (`general` or `nokids`).

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Unity Catalog catalog name")
dbutils.widgets.text("model_variant", "general", "Model variant: general or nokids")
dbutils.widgets.text("sample_fraction", "0.2", "Fraction of users to include")
dbutils.widgets.text("hyperparams_profile", "dev", "Hyperparams profile: dev or prod")

# COMMAND ----------

from bricks_ml3.config.settings import HYPERPARAMS_DEV, HYPERPARAMS_PROD
from bricks_ml3.training.train import train_model
from bricks_ml3.utils.spark_helpers import get_catalog, get_sample_fraction

catalog = get_catalog(dbutils)
model_variant = dbutils.widgets.get("model_variant")
sample_fraction = get_sample_fraction(dbutils)
hyperparams_profile = dbutils.widgets.get("hyperparams_profile")

hyperparams = HYPERPARAMS_PROD if hyperparams_profile == "prod" else HYPERPARAMS_DEV

print(f"Training {model_variant} model for catalog: {catalog}")
print(f"Sample fraction: {sample_fraction}")
print(f"Hyperparams: {hyperparams}")

# COMMAND ----------

model_name, model_version = train_model(spark, catalog, model_variant, hyperparams, sample_fraction)

print(f"Registered model: {model_name} version {model_version}")

# COMMAND ----------

dbutils.jobs.taskValues.set(key="model_name", value=model_name)
dbutils.jobs.taskValues.set(key="model_version", value=model_version)

print("Task values set for downstream validation.")
