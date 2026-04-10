# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 11 — Copy Model to Prod
# MAGIC
# MAGIC Copies the `@Champion` model from a source catalog (staging) to a
# MAGIC destination catalog (prod) using `MlflowClient.copy_model_version()`.
# MAGIC Both model variants (general and nokids) are copied and assigned the
# MAGIC `@Champion` alias in the destination.

# COMMAND ----------

dbutils.widgets.text("src_catalog", "", "Source catalog (e.g. dsml_staging)")
dbutils.widgets.text("dst_catalog", "", "Destination catalog (e.g. dsml_prod)")

# COMMAND ----------

import mlflow

from bricks_ml3.deployment.deploy_model import copy_model_to_prod

mlflow.set_registry_uri("databricks-uc")

src_catalog = dbutils.widgets.get("src_catalog")
dst_catalog = dbutils.widgets.get("dst_catalog")

print(f"Copying @Champion models: {src_catalog} → {dst_catalog}")

# COMMAND ----------

general_version = copy_model_to_prod(src_catalog, dst_catalog, "general")
print(f"  general v{general_version} copied to {dst_catalog}")

nokids_version = copy_model_to_prod(src_catalog, dst_catalog, "nokids")
print(f"  nokids v{nokids_version} copied to {dst_catalog}")

print("Model promotion complete.")

# COMMAND ----------

dbutils.notebook.exit(f"general=v{general_version}, nokids=v{nokids_version}")
