"""Roll back @Champion model aliases to a previous version.

Lists recent model versions and reverts @Champion to a specified version
(or the version before the current @Champion if none is specified).

Usage:
    python scripts/rollback_model.py --target prod
    python scripts/rollback_model.py --target prod --version 3

Exit codes:
    0 — rollback succeeded
    1 — rollback failed (no previous version, target not found, etc.)
"""

import argparse
import sys
import yaml
from pathlib import Path

import mlflow
from mlflow import MlflowClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_VARIANTS = {
    "general": "genre_propensity_general",
    "nokids": "genre_propensity_nokids",
}
SCHEMA_ML = "ml"


def load_catalog(target: str) -> str:
    with open(PROJECT_ROOT / "databricks.yml") as f:
        config = yaml.safe_load(f)
    targets = config.get("targets", {})
    if target not in targets:
        print(f"ERROR: target '{target}' not found in databricks.yml.")
        sys.exit(1)
    return targets[target].get("variables", {}).get("catalog", "")


def full_model_name(catalog: str, model_short: str) -> str:
    return f"{catalog}.{SCHEMA_ML}.{model_short}"


def get_champion_version(client: MlflowClient, model_name: str) -> str | None:
    try:
        mv = client.get_model_version_by_alias(name=model_name, alias="Champion")
        return mv.version
    except Exception:
        return None


def find_previous_version(
    client: MlflowClient, model_name: str, current_version: str
) -> str | None:
    """Find the most recent version before current_version."""
    versions = client.search_model_versions(
        f"name='{model_name}'", order_by=["version_number DESC"]
    )
    for mv in versions:
        if int(mv.version) < int(current_version):
            return mv.version
    return None


def rollback_variant(
    client: MlflowClient,
    model_name: str,
    variant: str,
    target_version: str | None,
) -> bool:
    current = get_champion_version(client, model_name)
    if current is None:
        print(f"  {variant}: no @Champion alias found — nothing to roll back.")
        return False

    if target_version is None:
        target_version = find_previous_version(client, model_name, current)
        if target_version is None:
            print(f"  {variant}: @Champion is v{current} but no earlier version exists.")
            return False

    if target_version == current:
        print(f"  {variant}: @Champion is already v{current} — no change needed.")
        return True

    client.set_registered_model_alias(
        name=model_name, alias="Champion", version=target_version
    )
    print(f"  {variant}: rolled back @Champion from v{current} to v{target_version}.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Roll back @Champion model aliases")
    parser.add_argument(
        "--target", required=True, choices=["dev", "staging", "prod"]
    )
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help="Target version number to roll back to (default: version before current @Champion)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be rolled back without making changes",
    )
    args = parser.parse_args()

    catalog = load_catalog(args.target)
    if not catalog:
        print(f"ERROR: no 'catalog' variable for target '{args.target}'")
        sys.exit(1)

    mlflow.set_registry_uri("databricks-uc")
    client = MlflowClient()

    print(f"Rolling back @Champion aliases in {catalog} ({args.target}):")

    if args.dry_run:
        for variant, model_short in MODEL_VARIANTS.items():
            model_name = full_model_name(catalog, model_short)
            current = get_champion_version(client, model_name)
            if current is None:
                print(f"  {variant}: no @Champion alias found.")
                continue
            target = args.version or find_previous_version(client, model_name, current)
            if target is None:
                print(f"  {variant}: @Champion v{current}, no earlier version available.")
            elif target == current:
                print(f"  {variant}: @Champion v{current}, already at target.")
            else:
                print(f"  {variant}: would roll back @Champion from v{current} to v{target}.")
        print("\nDry run — no changes made.")
        sys.exit(0)

    success = True
    for variant, model_short in MODEL_VARIANTS.items():
        model_name = full_model_name(catalog, model_short)
        if not rollback_variant(client, model_name, variant, args.version):
            success = False

    if success:
        print("Rollback complete.")
        sys.exit(0)
    else:
        print("Rollback failed for one or more variants.")
        sys.exit(1)


if __name__ == "__main__":
    main()
