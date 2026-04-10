#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_task.sh — Submit a single initial_setup task to an existing cluster.
#
# Builds a timestamped dev wheel so each invocation forces a fresh install
# on the existing cluster (avoids "already installed" skips).
#
# Usage:
#   ./scripts/run_task.sh <task_name> [cluster_id]
#
# task_name: validate_general | validate_nokids |
#            train_nokids | create_serving_endpoints |
#            batch_score_general | batch_score_nokids
#
# cluster_id: defaults to DEV_CLUSTER_ID env var or the hardcoded default.
#
# Examples:
#   ./scripts/run_task.sh validate_general
#   ./scripts/run_task.sh train_nokids 0314-010537-ubfrotya
# ---------------------------------------------------------------------------

set -euo pipefail

TASK="${1:-}"
CLUSTER_ID="${2:-${DEV_CLUSTER_ID:-YOUR_CLUSTER_ID}}"
CATALOG="${CATALOG:-dsml_dev}"
SAMPLE_FRACTION="0.2"

BUNDLE_BASE="/Workspace/Users/${DATABRICKS_USER:?Set DATABRICKS_USER}/.bundle/bricks-ml3/dev"
NOTEBOOK_BASE="${BUNDLE_BASE}/files/src/notebooks"
ARTIFACTS_DIR="${BUNDLE_BASE}/artifacts/.internal"

if [[ -z "$TASK" ]]; then
  echo "Usage: $0 <task_name> [cluster_id]"
  echo ""
  echo "Available tasks:"
  echo "  validate_general"
  echo "  train_general"
  echo "  train_nokids"
  echo "  validate_nokids"
  echo "  create_serving_endpoints"
  echo "  batch_score_general"
  echo "  batch_score_nokids"
  echo "  simulate_new_data"
  echo "  feature_engineering"
  echo "  monitor"
  echo "  promote_or_reject"
  exit 1
fi

# ---------------------------------------------------------------------------
# Build a timestamped dev wheel so pip always reinstalls (avoids version cache)
# ---------------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TIMESTAMP=$(date +%s)
DEV_VERSION="0.1.0.dev${TIMESTAMP}"

echo "Building wheel v${DEV_VERSION}..."
# Temporarily patch setup.py version
sed -i.bak "s/version=\"[^\"]*\"/version=\"${DEV_VERSION}\"/" "${PROJECT_ROOT}/setup.py" 2>/dev/null || \
  sed -i.bak "s/version='[^']*'/version='${DEV_VERSION}'/" "${PROJECT_ROOT}/setup.py"

(cd "${PROJECT_ROOT}" && pip wheel . -w dist/ -q --no-deps 2>&1 | grep -v "^$" || true)

# Restore setup.py
mv "${PROJECT_ROOT}/setup.py.bak" "${PROJECT_ROOT}/setup.py"

WHL_FILE=$(ls "${PROJECT_ROOT}/dist/bricks_ml3-${DEV_VERSION}-"*.whl 2>/dev/null | head -1)
if [[ -z "$WHL_FILE" ]]; then
  echo "ERROR: Could not find built wheel for version ${DEV_VERSION}"
  ls "${PROJECT_ROOT}/dist/" 2>/dev/null
  exit 1
fi
WHL_NAME=$(basename "$WHL_FILE")
WHL_REMOTE="${ARTIFACTS_DIR}/${WHL_NAME}"

echo "Uploading ${WHL_NAME}..."
databricks workspace import --overwrite --format AUTO --file "${WHL_FILE}" "${WHL_REMOTE}"

# ---------------------------------------------------------------------------
# Build per-task notebook path and parameters
# ---------------------------------------------------------------------------
case "$TASK" in
  validate_general)
    NOTEBOOK="${NOTEBOOK_BASE}/05_validate"
    # Look up latest registered model version
    MODEL_INFO=$(databricks model-versions list dsml_dev.ml.genre_propensity_general -t dev 2>/dev/null | \
      python3 -c "
import json, sys
mvs = json.load(sys.stdin)
if mvs:
    latest = sorted(mvs, key=lambda x: int(x.get('version','0')))[-1]
    print(latest.get('model_name',''), latest.get('version',''))
" 2>/dev/null || echo "")
    MODEL_NAME=$(echo "$MODEL_INFO" | awk '{print $1}')
    MODEL_VER=$(echo "$MODEL_INFO" | awk '{print $2}')
    PARAMS="{\"catalog\": \"${CATALOG}\", \"model_variant\": \"general\", \"model_name\": \"dsml_dev.ml.${MODEL_NAME}\", \"model_version\": \"${MODEL_VER}\"}"
    ;;
  train_general)
    NOTEBOOK="${NOTEBOOK_BASE}/04_train"
    PARAMS="{\"catalog\": \"${CATALOG}\", \"model_variant\": \"general\", \"sample_fraction\": \"${SAMPLE_FRACTION}\"}"
    ;;
  train_nokids)
    NOTEBOOK="${NOTEBOOK_BASE}/04_train"
    PARAMS="{\"catalog\": \"${CATALOG}\", \"model_variant\": \"nokids\", \"sample_fraction\": \"${SAMPLE_FRACTION}\"}"
    ;;
  validate_nokids)
    NOTEBOOK="${NOTEBOOK_BASE}/05_validate"
    MODEL_INFO=$(databricks model-versions list dsml_dev.ml.genre_propensity_nokids -t dev 2>/dev/null | \
      python3 -c "
import json, sys
mvs = json.load(sys.stdin)
if mvs:
    latest = sorted(mvs, key=lambda x: int(x.get('version','0')))[-1]
    print(latest.get('model_name',''), latest.get('version',''))
" 2>/dev/null || echo "")
    MODEL_NAME=$(echo "$MODEL_INFO" | awk '{print $1}')
    MODEL_VER=$(echo "$MODEL_INFO" | awk '{print $2}')
    PARAMS="{\"catalog\": \"${CATALOG}\", \"model_variant\": \"nokids\", \"model_name\": \"dsml_dev.ml.${MODEL_NAME}\", \"model_version\": \"${MODEL_VER}\"}"
    ;;
  create_serving_endpoints)
    NOTEBOOK="${NOTEBOOK_BASE}/06_deploy_endpoints"
    PARAMS="{\"catalog\": \"${CATALOG}\"}"
    ;;
  batch_score_general)
    NOTEBOOK="${NOTEBOOK_BASE}/07_batch_score"
    PARAMS="{\"catalog\": \"${CATALOG}\", \"model_variant\": \"general\"}"
    ;;
  batch_score_nokids)
    NOTEBOOK="${NOTEBOOK_BASE}/07_batch_score"
    PARAMS="{\"catalog\": \"${CATALOG}\", \"model_variant\": \"nokids\"}"
    ;;
  simulate_new_data)
    NOTEBOOK="${NOTEBOOK_BASE}/08_simulate_new_data"
    PARAMS="{\"catalog\": \"${CATALOG}\", \"days_window\": \"1\", \"sample_fraction\": \"${SAMPLE_FRACTION}\"}"
    ;;
  feature_engineering)
    NOTEBOOK="${NOTEBOOK_BASE}/03_feature_engineering"
    PARAMS="{\"catalog\": \"${CATALOG}\", \"sample_fraction\": \"${SAMPLE_FRACTION}\"}"
    ;;
  monitor)
    NOTEBOOK="${NOTEBOOK_BASE}/10_monitor"
    PARAMS="{\"catalog\": \"${CATALOG}\"}"
    ;;
  promote_or_reject)
    NOTEBOOK="${NOTEBOOK_BASE}/09_promote_or_reject"
    PARAMS="{\"catalog\": \"${CATALOG}\"}"
    ;;
  *)
    echo "Unknown task: $TASK"
    exit 1
    ;;
esac

echo ""
echo "Submitting: $TASK"
echo "  Cluster : $CLUSTER_ID"
echo "  Notebook: $NOTEBOOK"
echo "  Params  : $PARAMS"
echo "  Wheel   : $WHL_REMOTE"
echo ""

RUN_OUTPUT=$(databricks jobs submit -t dev --no-wait --json "{
  \"run_name\": \"debug_${TASK}\",
  \"tasks\": [{
    \"task_key\": \"${TASK}\",
    \"existing_cluster_id\": \"${CLUSTER_ID}\",
    \"libraries\": [{\"whl\": \"${WHL_REMOTE}\"}],
    \"notebook_task\": {
      \"notebook_path\": \"${NOTEBOOK}\",
      \"base_parameters\": ${PARAMS}
    }
  }]
}")

RUN_ID=$(echo "$RUN_OUTPUT" | python3 -c "import json,sys; print(json.load(sys.stdin)['run_id'])")
echo "Run submitted: run_id=${RUN_ID}"
echo "  UI: ${DATABRICKS_HOST}#job/0/run/${RUN_ID}"
echo ""

# Poll until done
echo "Polling..."
while true; do
  STATE=$(databricks jobs get-run "$RUN_ID" -t dev --output json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
s = d.get('state', {})
print(s.get('life_cycle_state',''), s.get('result_state',''))
")
  echo "  $(date '+%H:%M:%S') $STATE"
  if echo "$STATE" | grep -qE "TERMINATED|INTERNAL_ERROR|SKIPPED"; then
    break
  fi
  sleep 20
done

# Get task-level run_id for output
TASK_RUN_ID=$(databricks jobs get-run "$RUN_ID" -t dev --output json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
for t in d.get('tasks', []):
    print(t.get('run_id',''))
" | head -1)

RS=$(echo "$STATE" | awk '{print $2}')
if [[ "$RS" == "SUCCESS" ]]; then
  echo ""
  echo "PASSED: $TASK"
else
  echo ""
  echo "FAILED: $TASK — fetching error..."
  databricks jobs get-run-output "$TASK_RUN_ID" -t dev --output json 2>/dev/null | python3 -c "
import json, sys
d = json.load(sys.stdin)
err = d.get('error', '')
trace = d.get('error_trace', '')
print('ERROR:', err)
print(trace[-3000:] if trace else '')
"
fi
