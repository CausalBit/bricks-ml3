"""Check if a target environment has been bootstrapped.

Verifies that the minimum data prerequisites exist before running
operational jobs (weekly_retraining, daily_scoring, etc.). Used by
the reusable deploy workflow to skip jobs on uninitialized environments
instead of failing with confusing table-not-found errors.

Usage:
    python scripts/check_readiness.py --target dev

Exit codes:
    0 — environment is ready (silver.ratings exists)
    1 — environment needs bootstrapping
"""

import argparse
import sys
import yaml
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_bundle_config() -> dict:
    with open(PROJECT_ROOT / "databricks.yml") as f:
        return yaml.safe_load(f)


def resolve_catalog(config: dict, target: str) -> str:
    targets = config.get("targets", {})
    if target not in targets:
        print(f"ERROR: target '{target}' not found in databricks.yml.")
        sys.exit(1)
    return targets[target].get("variables", {}).get("catalog", "")


def main():
    parser = argparse.ArgumentParser(description="Check environment readiness")
    parser.add_argument("--target", required=True, choices=["dev", "staging", "prod"])
    args = parser.parse_args()

    config = load_bundle_config()
    catalog = resolve_catalog(config, args.target)

    if not catalog:
        print(f"ERROR: no 'catalog' variable for target '{args.target}'")
        sys.exit(1)

    w = WorkspaceClient()
    table_name = f"{catalog}.silver.ratings"

    try:
        w.tables.get(table_name)
        print(f"Environment '{args.target}' ({catalog}) is ready.")
        sys.exit(0)
    except NotFound:
        print(f"Environment '{args.target}' ({catalog}) has NOT been bootstrapped.")
        print("Run the Bootstrap Environment workflow to seed data first.")
        sys.exit(1)
    except Exception as e:
        print(f"WARNING: Could not verify readiness — {e}")
        print("Skipping job run as a precaution.")
        sys.exit(1)


if __name__ == "__main__":
    main()
