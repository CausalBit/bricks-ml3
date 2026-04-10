"""Pattern B -- Deploy Model: cross-catalog model copy.

Copies a validated ``@Champion`` model version from a source catalog
(typically ``dev``) to a destination catalog (typically ``prod``) and
assigns the ``@Champion`` alias in the destination.

This is the alternative to the deploy-code pattern; both are implemented
as separate DABs jobs (see ``project_design.md`` section 7).
"""

from __future__ import annotations

import logging

from bricks_ml3.config.settings import (
    MODEL_GENERAL,
    MODEL_NOKIDS,
    SCHEMA_ML,
)
from bricks_ml3.utils.spark_helpers import table_name

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def copy_model_to_prod(
    src_catalog: str,
    dst_catalog: str,
    model_variant: str,
) -> str:
    """Copy a model version across catalogs and assign ``@Champion``.

    Uses ``MlflowClient.copy_model_version`` to copy the ``@Champion``
    version from the source catalog to the destination catalog, then sets
    the ``@Champion`` alias on the newly created version.

    Args:
        src_catalog: Source catalog name (e.g. ``"dev"``).
        dst_catalog: Destination catalog name (e.g. ``"prod"``).
        model_variant: ``"general"`` or ``"nokids"``.

    Returns:
        The new model version string in the destination catalog.
    """
    from mlflow import MlflowClient

    model_name_short = MODEL_GENERAL if model_variant == "general" else MODEL_NOKIDS
    src_model_name = table_name(src_catalog, SCHEMA_ML, model_name_short)
    dst_model_name = table_name(dst_catalog, SCHEMA_ML, model_name_short)

    src_model_uri = f"models:/{src_model_name}@Champion"

    client = MlflowClient()
    copied = client.copy_model_version(
        src_model_uri=src_model_uri,
        dst_name=dst_model_name,
    )
    client.set_registered_model_alias(
        name=dst_model_name,
        alias="Champion",
        version=copied.version,
    )

    logger.info(
        "Copied %s @Champion -> %s v%s and set @Champion",
        src_model_name,
        dst_model_name,
        copied.version,
    )
    return str(copied.version)
