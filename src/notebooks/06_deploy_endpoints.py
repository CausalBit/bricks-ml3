# Databricks notebook source

# COMMAND ----------

# MAGIC %md
# MAGIC # 06 — Deploy Serving Endpoints
# MAGIC
# MAGIC Promotes `@Challenger` to `@Champion` and creates (or updates) the
# MAGIC Model Serving endpoints for both model variants (general and nokids).

# COMMAND ----------

dbutils.widgets.text("catalog", "", "Unity Catalog catalog name")

# COMMAND ----------

from bricks_ml3.deployment.deploy_code import (
    create_or_update_endpoint,
    promote_to_champion,
)
from bricks_ml3.utils.spark_helpers import get_catalog

catalog = get_catalog(dbutils)
print(f"Deploying serving endpoints for catalog: {catalog}")

# COMMAND ----------

general_version = promote_to_champion(catalog, "general")
print(f"Promoted general model to @Champion: v{general_version}")

create_or_update_endpoint(catalog, "general", general_version)
print("Endpoint 'genre-propensity-general' ready.")

# COMMAND ----------

nokids_version = promote_to_champion(catalog, "nokids")
print(f"Promoted nokids model to @Champion: v{nokids_version}")

create_or_update_endpoint(catalog, "nokids", nokids_version)
print("Endpoint 'genre-propensity-nokids' ready.")

# COMMAND ----------

print("All serving endpoints deployed successfully.")
