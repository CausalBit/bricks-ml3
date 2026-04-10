# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 09 — Promote or Reject
# MAGIC
# MAGIC Checks the validation status tags for both model variants. If both
# MAGIC passed, promotes `@Challenger` to `@Champion`. If either failed,
# MAGIC logs a warning and skips promotion.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Unity Catalog catalog name")

# COMMAND ----------

import mlflow
from mlflow import MlflowClient

from bricks_ml3.config.settings import (
    MODEL_GENERAL,
    MODEL_NOKIDS,
    SCHEMA_ML,
)
from bricks_ml3.deployment.deploy_code import promote_to_champion
from bricks_ml3.utils.spark_helpers import get_catalog, table_name

catalog = get_catalog(dbutils)
print(f"Checking validation status for catalog: {catalog}")

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
mlflow_client = MlflowClient()

variants = {
    "general": table_name(catalog, SCHEMA_ML, MODEL_GENERAL),
    "nokids": table_name(catalog, SCHEMA_ML, MODEL_NOKIDS),
}

all_passed = True
for variant, model_name in variants.items():
    try:
        challenger_mv = mlflow_client.get_model_version_by_alias(name=model_name, alias="Challenger")
        tags = challenger_mv.tags or {}
        status = tags.get("model_validation_status", "UNKNOWN")
        print(f"  {variant}: @Challenger v{challenger_mv.version} — status={status}")
        if status != "PASSED":
            all_passed = False
    except Exception as exc:
        print(f"  {variant}: no @Challenger alias found — {exc}")
        all_passed = False

# COMMAND ----------

if all_passed:
    print("Both models passed validation — promoting to @Champion.")

    general_version = promote_to_champion(catalog, "general")
    print(f"  general v{general_version} promoted to @Champion.")

    nokids_version = promote_to_champion(catalog, "nokids")
    print(f"  nokids v{nokids_version} promoted to @Champion.")

    print("Promotion complete.")
    dbutils.notebook.exit("PROMOTED")
else:
    raise RuntimeError(
        "One or both models failed validation — promotion skipped. Downstream tasks (batch scoring) will not run."
    )
