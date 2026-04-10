"""Pre-deploy environment setup for the bricks-ml3 project.

Ensures the target environment is ready before `databricks bundle deploy`:
  1. Creates the Unity Catalog catalog if it doesn't exist
  2. Grants the service principal required permissions
  3. Creates schemas (bronze, silver, gold, ml, inference) if they don't exist
  4. Creates the landing volume in bronze if it doesn't exist
  5. Creates the MLflow experiment workspace directory

All operations are idempotent — safe to run on every deploy.
Schemas and volumes are managed here (not in the DABs bundle) so that
`databricks bundle deploy` never fails with "Schema already exists" errors.

Usage:
    python scripts/setup_catalog.py --target dev
    python scripts/setup_catalog.py --target prod --sp-name my_sp

Authentication uses the standard Databricks environment variables:
    DATABRICKS_HOST, DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET
"""

import argparse
import sys
import yaml
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, PermissionDenied, ResourceAlreadyExists
from databricks.sdk.service.catalog import VolumeType


PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Resolve config from databricks.yml
# ---------------------------------------------------------------------------

def load_bundle_config() -> dict:
    """Load databricks.yml from the project root."""
    with open(PROJECT_ROOT / "databricks.yml") as f:
        return yaml.safe_load(f)


def resolve_target(config: dict, target: str) -> dict:
    """Return the variables dict for a given target."""
    targets = config.get("targets", {})
    if target not in targets:
        print(f"ERROR: target '{target}' not found in databricks.yml. "
              f"Available: {list(targets.keys())}")
        sys.exit(1)
    return targets[target].get("variables", {})


# ---------------------------------------------------------------------------
# Catalog setup
# ---------------------------------------------------------------------------

def ensure_catalog(w: WorkspaceClient, catalog_name: str) -> bool:
    """Create the catalog if it doesn't exist. Returns True if created.

    Also handles PermissionDenied (catalog exists but caller lacks USE CATALOG)
    by deferring to the grant step rather than crashing.
    """
    try:
        w.catalogs.get(catalog_name)
        print(f"  Catalog '{catalog_name}' already exists.")
        return False
    except NotFound:
        print(f"  Creating catalog '{catalog_name}'...")
        w.catalogs.create(name=catalog_name, comment=f"Movies ML - {catalog_name}")
        print(f"  Catalog '{catalog_name}' created.")
        return True
    except PermissionDenied:
        print(f"  Catalog '{catalog_name}' exists but the current principal lacks USE CATALOG.")
        print(f"  Will attempt to grant permissions in the next step.")
        return False


# ---------------------------------------------------------------------------
# Grants
# ---------------------------------------------------------------------------

def _grant(w: WorkspaceClient, securable_type: str, full_name: str,
           principals: list[str], privileges: list[str]) -> bool:
    """Issue a Unity Catalog grants PATCH via the REST API directly.

    Uses the raw API client to avoid a known SDK serialization issue where
    SecurableType enum values are sent as 'SECURABLETYPE.CATALOG' instead of
    'catalog', causing the server to reject the request.

    Issues one request per principal so a missing principal (e.g. the service
    principal not yet provisioned in this metastore) doesn't block grants for
    other principals.
    """
    any_success = False
    for principal in principals:
        try:
            w.api_client.do(
                "PATCH",
                f"/api/2.1/unity-catalog/permissions/{securable_type}/{full_name}",
                body={"changes": [{"principal": principal, "add": privileges}]},
            )
            any_success = True
        except Exception as e:
            print(f"  WARNING: grant({securable_type}, {full_name}, {principal}) failed: {e}")
    return any_success


def grant_catalog_permissions(w: WorkspaceClient, catalog_name: str, sp_name: str):
    """Grant the service principal and the current user permissions on the catalog.

    Granting both covers two distinct execution contexts:
      - CI/CD: runs as the service principal (sp_name) — gets the SP grant.
      - Local / manual: runs as a human user — gets the current-user grant so
        that job clusters spun up under that user's identity can access the schemas.
    When the current user IS the service principal both grants are identical and
    the API deduplicates them automatically.

    Attempts catalog-level grants first. If those fail (e.g., caller is not the
    catalog owner), falls back to schema-level grants which are applied per-schema
    in ensure_schemas() below.
    """
    catalog_privileges = ["USE_CATALOG", "CREATE_SCHEMA"]
    schema_privileges = [
        "USE_SCHEMA", "CREATE_TABLE", "CREATE_VOLUME",
        "CREATE_MODEL", "CREATE_FUNCTION", "EXECUTE",
        "SELECT", "MODIFY",
    ]

    current_user = w.current_user.me().user_name or ""
    principals = list({sp_name, current_user} - {""})

    print(f"  Granting catalog-level privileges to: {principals}...")
    if _grant(w, "catalog", catalog_name, principals, catalog_privileges):
        print(f"  Catalog-level permissions granted.")
    else:
        print(f"  Catalog-level grant failed — schema-level grants will be applied per schema.")

    # Verify we can now access the catalog. If we still can't, the remaining
    # steps (create schemas, volumes) will all fail, so bail out early with a
    # clear remediation message.
    try:
        w.catalogs.get(catalog_name)
    except PermissionDenied:
        print()
        print("ERROR: Still cannot access catalog after grant attempt.")
        print("A metastore admin must run the following SQL in a notebook or")
        print("SQL editor on this workspace:")
        print()
        for p in principals:
            print(f"  GRANT USE CATALOG, CREATE SCHEMA ON CATALOG `{catalog_name}` TO `{p}`;")
        print()
        print("Then re-run this script.")
        sys.exit(1)

    # Store principals so ensure_schemas() can grant at schema level.
    # Stored as a module-level variable to avoid changing function signatures.
    grant_catalog_permissions._principals = principals
    grant_catalog_permissions._schema_privileges = schema_privileges


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

# Schema definitions — mirrors what was previously in resources/uc_setup.yml.
# Managed here so creation is idempotent (no Terraform state dependency).
SCHEMAS = {
    "bronze":    "Raw ingested tables and landing volume",
    "silver":    "Cleaned, exploded, enriched tables",
    "gold":      "Feature Store tables",
    "ml":        "Registered models and MLflow experiment artifacts",
    "inference": "Batch scoring output tables",
}


def ensure_schemas(w: WorkspaceClient, catalog_name: str):
    """Create each schema if it doesn't already exist, then grant schema-level privileges."""
    principals = getattr(grant_catalog_permissions, "_principals", [])
    schema_privileges = getattr(grant_catalog_permissions, "_schema_privileges", [])

    for schema_name, comment in SCHEMAS.items():
        full_name = f"{catalog_name}.{schema_name}"
        try:
            w.schemas.get(full_name)
            print(f"  Schema '{full_name}' already exists.")
        except NotFound:
            print(f"  Creating schema '{full_name}'...")
            w.schemas.create(
                catalog_name=catalog_name,
                name=schema_name,
                comment=comment,
            )
            print(f"  Schema '{full_name}' created.")

        if principals and schema_privileges:
            if _grant(w, "schema", full_name, principals, schema_privileges):
                print(f"  Schema-level permissions granted on '{full_name}'.")
            else:
                print(f"  WARNING: Could not grant schema-level permissions on '{full_name}'. Grant manually if needed.")


# ---------------------------------------------------------------------------
# Volumes
# ---------------------------------------------------------------------------

def ensure_landing_volume(w: WorkspaceClient, catalog_name: str):
    """Create the landing volume in bronze if it doesn't exist."""
    full_name = f"{catalog_name}.bronze.landing"
    try:
        w.volumes.read(full_name)
        print(f"  Volume '{full_name}' already exists.")
    except NotFound:
        print(f"  Creating volume '{full_name}'...")
        w.volumes.create(
            catalog_name=catalog_name,
            schema_name="bronze",
            name="landing",
            volume_type=VolumeType.MANAGED,
            comment="Upload destination for raw ML-25M CSV files",
        )
        print(f"  Volume '{full_name}' created.")


# ---------------------------------------------------------------------------
# Experiment directory
# ---------------------------------------------------------------------------

def ensure_experiment_directory(w: WorkspaceClient, catalog_name: str):
    """Create the /Shared/genre_propensity/{catalog} workspace directory."""
    dir_path = f"/Shared/genre_propensity/{catalog_name}"
    try:
        w.workspace.mkdirs(dir_path)
        print(f"  Experiment directory '{dir_path}' ready.")
    except Exception as e:
        print(f"  WARNING: Could not create experiment directory: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pre-deploy environment setup")
    parser.add_argument("--target", required=True, choices=["pre-dev", "dev", "staging", "prod"],
                        help="DABs target environment")
    parser.add_argument("--sp-name", default=None,
                        help="Service principal name (overrides databricks.yml)")
    args = parser.parse_args()

    config = load_bundle_config()
    variables = resolve_target(config, args.target)

    catalog_name = variables.get("catalog")
    if not catalog_name:
        print(f"ERROR: no 'catalog' variable defined for target '{args.target}'")
        sys.exit(1)

    sp_name = args.sp_name or config.get("variables", {}).get("service_principal_name", {}).get("default", "mlops_sp")

    print(f"--- Pre-deploy setup for target '{args.target}' ---")
    print(f"  Catalog:           {catalog_name}")
    print(f"  Service Principal: {sp_name}")
    print(f"  (Current user will also receive grants if different from SP)")
    print()

    w = WorkspaceClient()

    ensure_catalog(w, catalog_name)
    grant_catalog_permissions(w, catalog_name, sp_name)
    ensure_schemas(w, catalog_name)
    ensure_landing_volume(w, catalog_name)
    ensure_experiment_directory(w, catalog_name)

    print()
    print("Pre-deploy setup complete.")


if __name__ == "__main__":
    main()
