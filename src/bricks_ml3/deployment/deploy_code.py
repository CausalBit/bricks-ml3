"""Pattern A -- Deploy Code: Champion promotion and serving endpoint management.

Promotes a validated ``@Challenger`` model to ``@Champion`` and creates or
updates Model Serving endpoints using the Databricks SDK.  Called by the
``06_deploy_endpoints.py`` and ``09_promote_or_reject.py`` notebooks.
"""

from __future__ import annotations

import logging

from bricks_ml3.config.settings import (
    ENDPOINT_GENERAL,
    ENDPOINT_NOKIDS,
    MODEL_GENERAL,
    MODEL_NOKIDS,
    SCHEMA_ML,
)
from bricks_ml3.utils.spark_helpers import table_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def promote_to_champion(catalog: str, model_variant: str) -> str:
    """Promote the ``@Challenger`` alias to ``@Champion`` for a model.

    Reads the version currently tagged ``@Challenger`` and reassigns the
    ``@Champion`` alias to that same version.

    Args:
        catalog: Unity Catalog catalog name.
        model_variant: ``"general"`` or ``"nokids"``.

    Returns:
        The model version string that was promoted.
    """
    import mlflow
    from mlflow import MlflowClient

    model_name_short = MODEL_GENERAL if model_variant == "general" else MODEL_NOKIDS
    registered_model_name = table_name(catalog, SCHEMA_ML, model_name_short)

    mlflow.set_registry_uri("databricks-uc")
    mlflow_client = MlflowClient()
    challenger_mv = mlflow_client.get_model_version_by_alias(name=registered_model_name, alias="Challenger")
    champion_version = challenger_mv.version

    mlflow_client.set_registered_model_alias(
        name=registered_model_name,
        alias="Champion",
        version=champion_version,
    )

    logger.info(
        "Promoted %s v%s from @Challenger to @Champion",
        registered_model_name,
        champion_version,
    )
    return str(champion_version)


def create_or_update_endpoint(
    catalog: str,
    model_variant: str,
    champion_version: str,
) -> None:
    """Create or update a Model Serving endpoint for the given model variant.

    If the endpoint already exists its config is updated; otherwise a new
    endpoint is created with ``create_and_wait``.

    Args:
        catalog: Unity Catalog catalog name.
        model_variant: ``"general"`` or ``"nokids"``.
        champion_version: The model version to serve.
    """
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput,
        ServedEntityInput,
    )

    model_name_short = MODEL_GENERAL if model_variant == "general" else MODEL_NOKIDS
    registered_model_name = table_name(catalog, SCHEMA_ML, model_name_short)
    endpoint_name = ENDPOINT_GENERAL if model_variant == "general" else ENDPOINT_NOKIDS

    w = WorkspaceClient()

    config = EndpointCoreConfigInput(
        served_entities=[
            ServedEntityInput(
                entity_name=registered_model_name,
                entity_version=champion_version,
                workload_size="Small",
                scale_to_zero_enabled=True,
            )
        ],
    )

    try:
        w.serving_endpoints.get(name=endpoint_name)
        w.serving_endpoints.update_config_and_wait(
            name=endpoint_name,
            served_entities=config.served_entities,
        )
        logger.info("Updated endpoint '%s' to v%s", endpoint_name, champion_version)
    except Exception:
        w.serving_endpoints.create_and_wait(
            name=endpoint_name,
            config=config,
        )
        logger.info("Created endpoint '%s' with v%s", endpoint_name, champion_version)
