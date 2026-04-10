# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 10 — Monitoring & Drift Detection
# MAGIC
# MAGIC Runs feature and prediction drift checks after daily batch scoring.
# MAGIC If drift is detected, sets a task value so downstream tasks can react.

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Unity Catalog catalog name")

# COMMAND ----------

from bricks_ml3.monitoring.drift import run_drift_check
from bricks_ml3.utils.spark_helpers import get_catalog

catalog = get_catalog(dbutils)
print(f"Running drift check for catalog: {catalog}")

# COMMAND ----------

summary = run_drift_check(spark, catalog)
print(f"Drift check summary: {summary}")

# COMMAND ----------

if summary["drift_detected"]:
    print("DRIFT DETECTED — setting task value drift_detected=true")
    dbutils.jobs.taskValues.set(key="drift_detected", value="true")
else:
    print("No significant drift detected.")
    dbutils.jobs.taskValues.set(key="drift_detected", value="false")

# COMMAND ----------

print("Monitoring complete.")
