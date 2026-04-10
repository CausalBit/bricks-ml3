#!/usr/bin/env bash
# detect_changes.sh — Determine which tests to run and which Databricks job
# to trigger based on the files that changed between two git refs.
#
# Usage:
#   scripts/detect_changes.sh <base_sha> <head_sha>
#   scripts/detect_changes.sh HEAD~1 HEAD
#
# Outputs (written to $GITHUB_OUTPUT when available, otherwise stdout):
#   tests         — space-separated pytest paths (or "all", or "none")
#   job           — Databricks job name (or "" for deploy-only)
#   skip_deploy   — "true" if only docs/non-code files changed
#
# Job priority (highest wins when multiple domains match):
#   weekly_retraining > feature_backfill > daily_scoring > endpoint_deploy > ""

set -euo pipefail

BASE_SHA="${1:?Usage: detect_changes.sh <base_sha> <head_sha>}"
HEAD_SHA="${2:?Usage: detect_changes.sh <base_sha> <head_sha>}"

CHANGED=$(git diff --name-only "$BASE_SHA" "$HEAD_SHA" 2>/dev/null || echo "")
if [ -z "$CHANGED" ]; then
  echo "No changed files detected."
  TESTS="none"; JOB=""; SKIP="true"
else
  TESTS=""
  JOB_PRIORITY=0   # 0=none, 1=endpoint_deploy, 2=daily_scoring, 3=feature_backfill, 4=weekly_retraining
  JOB=""
  SKIP="true"

  set_job() {
    local priority=$1 name=$2
    if [ "$priority" -gt "$JOB_PRIORITY" ]; then
      JOB_PRIORITY=$priority
      JOB=$name
    fi
  }

  add_tests() {
    [ "$TESTS" = "all" ] && return
    for t in "$@"; do
      case " $TESTS " in
        *" $t "*) ;;
        *) TESTS="$TESTS $t" ;;
      esac
    done
  }

  while IFS= read -r file; do
    case "$file" in
      # --- Docs / non-code (never deploy) ---
      README.md|docs/*|project_specs/*|.cursor/*|.claude/*|*.md)
        ;;

      # --- Silver / ingestion pipeline ---
      src/bricks_ml3/ingestion/*|src/bricks_ml3/transformations/silver.py|\
      src/notebooks/00_*|src/notebooks/01_*|src/notebooks/02_*)
        SKIP="false"
        add_tests "tests/unit/test_silver.py"
        ;;

      # --- Gold / feature engineering ---
      src/bricks_ml3/transformations/gold.py|\
      src/notebooks/03_*|src/notebooks/08_*)
        SKIP="false"
        add_tests "tests/unit/test_gold.py"
        set_job 3 "feature_backfill"
        ;;

      # --- Training / validation ---
      src/bricks_ml3/training/*|src/bricks_ml3/validation/*|\
      src/notebooks/04_*|src/notebooks/05_*|src/notebooks/09_*)
        SKIP="false"
        add_tests "tests/unit/test_train.py" "tests/unit/test_validate.py"
        set_job 4 "weekly_retraining"
        ;;

      # --- Inference / scoring ---
      src/bricks_ml3/inference/*|src/notebooks/07_*)
        SKIP="false"
        set_job 2 "daily_scoring"
        ;;

      # --- Monitoring ---
      src/bricks_ml3/monitoring/*|src/notebooks/10_*)
        SKIP="false"
        set_job 2 "daily_scoring"
        ;;

      # --- Deployment / serving ---
      src/bricks_ml3/deployment/*|src/notebooks/06_*|src/notebooks/11_*)
        SKIP="false"
        set_job 1 "endpoint_deploy"
        ;;

      # --- Shared library code (conservative — run everything) ---
      src/bricks_ml3/config/*|src/bricks_ml3/utils/*|src/bricks_ml3/__init__.py|\
      setup.py|requirements.txt)
        SKIP="false"
        TESTS="all"
        set_job 4 "weekly_retraining"
        ;;

      # --- Infrastructure / bundle config (deploy only, no job) ---
      databricks.yml|resources/*|scripts/*|pytest.ini|ruff.toml)
        SKIP="false"
        ;;

      # --- Workflow files (deploy only) ---
      .github/*)
        SKIP="false"
        ;;

      # --- Anything else under src/ is code that needs deploying ---
      src/*)
        SKIP="false"
        ;;
    esac
  done <<< "$CHANGED"

  TESTS=$(echo "$TESTS" | sed 's/^ *//;s/ *$//')
  [ -z "$TESTS" ] && TESTS="none"
fi

# Write outputs
if [ -n "${GITHUB_OUTPUT:-}" ]; then
  echo "tests=$TESTS" >> "$GITHUB_OUTPUT"
  echo "job=$JOB" >> "$GITHUB_OUTPUT"
  echo "skip_deploy=$SKIP" >> "$GITHUB_OUTPUT"
else
  echo "tests=$TESTS"
  echo "job=$JOB"
  echo "skip_deploy=$SKIP"
fi
