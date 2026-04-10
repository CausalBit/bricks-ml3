# Deployment Resource Inventory

> **Audience:** Platform / workspace admins deploying this project into a new organization  
> **Assumption:** Single Databricks workspace, three logical environments (dev, staging, prod) separated by Unity Catalog catalogs

---

## Quick Summary

| Category | Count | Details |
|----------|-------|---------|
| Unity Catalog catalogs | 3 | `dsml_dev`, `dsml_staging`, `dsml_prod` |
| Schemas per catalog | 5 | `bronze`, `silver`, `gold`, `ml`, `inference` |
| Volumes | 1 per catalog | `bronze.landing` (managed) |
| Registered ML models | 2 per catalog | `ml.genre_propensity_general`, `ml.genre_propensity_nokids` |
| MLflow experiments | 2 per catalog | Training + validation |
| Databricks jobs | 6 per target | bootstrap, initial_training, daily_scoring, weekly_retraining, feature_backfill, promote_model |
| Service principals | 1 | `mlops_sp` (CI/CD identity) |
| GitHub repository | 1 | With 3 protected environments |
| GitHub Actions runners | 1+ | GitHub-hosted or self-hosted |

---

## 1. Databricks Workspace

### 1.1 Workspace Requirements

| Requirement | Detail |
|-------------|--------|
| Cloud provider | Azure (VM types are `Standard_D8ds_v5` / `Standard_E8ds_v5`; adjust for AWS/GCP) |
| Unity Catalog | Enabled with a metastore attached to the workspace |
| Databricks Runtime | `15.4.x-cpu-ml-scala2.12` (ML Runtime) |
| Feature Engineering | `databricks-feature-engineering` client support (included in ML Runtime) |
| Cluster log delivery | Optional — `dbfs:/cluster-logs` is configured for log collection but not required for pipeline execution |

### 1.2 Compute Sizing

| Cluster profile | Node type | Workers | Purpose |
|----------------|-----------|---------|---------|
| `etl_cluster` (dev) | `Standard_D8ds_v5` | 0 (single node) | Bronze/silver ETL, feature engineering |
| `etl_cluster` (staging/prod) | `Standard_D8ds_v5` | 2 | Same, at full data scale |
| `training_cluster` | `Standard_D8ds_v5` (dev/prod) / `Standard_E8ds_v5` (staging) | 0 (single node) | Model training, validation, batch scoring |

> All clusters use `SINGLE_USER` data security mode and ephemeral job clusters (no interactive clusters required).

---

## 2. Unity Catalog Resources

### 2.1 Catalogs (3 total)

| Catalog | Environment | Created by |
|---------|-------------|------------|
| `dsml_dev` | Development | `setup_catalog.py` (automated) |
| `dsml_staging` | Staging | `setup_catalog.py` (automated) |
| `dsml_prod` | Production | `setup_catalog.py` (automated) |

**Ticket may be required:** If your organization restricts catalog creation to metastore admins, request creation of all three catalogs before the first deployment.

### 2.2 Schemas (5 per catalog = 15 total)

| Schema | Purpose |
|--------|---------|
| `bronze` | Raw ingested tables and landing volume |
| `silver` | Cleaned, exploded, enriched tables |
| `gold` | Feature Store tables |
| `ml` | Registered models and MLflow experiment artifacts |
| `inference` | Batch scoring output tables |

### 2.3 Volumes (1 per catalog = 3 total)

| Volume | Type | Purpose |
|--------|------|---------|
| `{catalog}.bronze.landing` | Managed | Upload destination for raw MovieLens CSV files |

### 2.4 Tables Created by the Pipeline

**Bronze (6 tables):** `ratings`, `movies`, `tags`, `genome_scores`, `genome_tags`, `links`

**Silver (5 tables):** `ratings_clean`, `movies_genre_exploded`, `movies_genre_exploded_nokids`, `genome_genre_agg`, `ratings_holdout`

**Gold (2 tables):** `user_genre_features`, `user_profile_features` (Feature Store-managed)

**ML (1 table):** `split_metadata`

**Inference (2 tables):** `genre_propensity_scores_daily`, `genre_propensity_scores_daily_nokids`

**Monitoring (2 tables):** `monitoring_log`, `feature_baseline`

### 2.5 Registered Models (2 per catalog)

| Model name | Description |
|------------|-------------|
| `{catalog}.ml.genre_propensity_general` | All 18 genres |
| `{catalog}.ml.genre_propensity_nokids` | 15 genres (excludes Children, Animation, Fantasy) |

Both use MLflow with Unity Catalog model registry (`mlflow.set_registry_uri("databricks-uc")`).

### 2.6 Required UC Permissions

The `setup_catalog.py` script grants these automatically, but a metastore admin may need to intervene if the deploying principal lacks sufficient privileges:

**Catalog-level:** `USE_CATALOG`, `CREATE_SCHEMA`

**Schema-level:** `USE_SCHEMA`, `CREATE_TABLE`, `CREATE_VOLUME`, `CREATE_MODEL`, `CREATE_FUNCTION`, `EXECUTE`, `SELECT`, `MODIFY`

---

## 3. MLflow Experiments

| Experiment path | Purpose |
|-----------------|---------|
| `/Shared/genre_propensity/{catalog}/training` | Training run tracking |
| `/Shared/genre_propensity/{catalog}/validation` | Validation run tracking |

The workspace directory `/Shared/genre_propensity/{catalog}` is created automatically by `setup_catalog.py`.

---

## 4. Service Principal

| Name | Purpose | Permissions needed |
|------|---------|-------------------|
| `mlops_sp` | CI/CD automation identity | `USE_CATALOG`, `CREATE_SCHEMA` on all three catalogs; schema-level R/W on all schemas; ability to create/run jobs |

**Ticket required:** A workspace admin must create the service principal and generate OAuth credentials (`client_id` + `client_secret`) for CI/CD. The SP must be:
1. Added to the Databricks workspace
2. Added to the Unity Catalog metastore
3. Granted `CAN_MANAGE` or `CAN_RESTART` on job clusters (or be a workspace admin)

---

## 5. GitHub Repository

### 5.1 Repository Setup

| Item | Detail |
|------|--------|
| Branches | `dev` (development), `main` (staging + prod) |
| Branch protection | PRs required to `dev` and `main` |

### 5.2 GitHub Environments (3)

Configure these under **Settings > Environments**:

| Environment | Protection rules | Purpose |
|-------------|-----------------|---------|
| `dev` | None | Auto-deploy on push to `dev` |
| `staging` | None | Auto-deploy on push to `main` |
| `prod` | Manual approval recommended | Manual dispatch from `main` only |

### 5.3 GitHub Secrets (2 required)

| Secret | Scope | Description |
|--------|-------|-------------|
| `DATABRICKS_CLIENT_ID` | Repository | Service principal OAuth client ID |
| `DATABRICKS_CLIENT_SECRET` | Repository | Service principal OAuth client secret |

> `GITHUB_TOKEN` is provided automatically by GitHub Actions.

### 5.4 GitHub Actions Workflows (9 total)

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | PR to `dev` or `main` | Lint, format check, bundle validate, conditional unit tests |
| `_reusable-deploy.yml` | Called by other workflows | Deploy bundle + optional job run |
| `deploy-dev.yml` | Push to `dev` | Deploy to dev, run integration tests |
| `deploy-staging.yml` | Push to `main` | Deploy to staging |
| `deploy-prod.yml` | Manual dispatch | Guard checks + deploy to prod + tag release |
| `promote.yml` | After dev deploy | Auto-create PR from `dev` to `main` |
| `rollback.yml` | Manual dispatch | Roll back model or redeploy previous tag |
| `run-job.yml` | Manual dispatch | Run any job on any target on-demand |
| `setup-environment.yml` | Manual dispatch | Bootstrap a new environment end-to-end |

### 5.5 GitHub Actions Runner Requirements

The runner (GitHub-hosted or self-hosted) must have:

| Requirement | Detail |
|-------------|--------|
| Python | 3.10+ |
| Databricks CLI | `databricks` CLI installed and on `PATH` |
| GitHub CLI | `gh` CLI (used by `promote.yml`, `deploy-prod.yml`) |
| Network | Outbound HTTPS to Databricks workspace and `files.grouplens.org` |

> The workflow files currently use `runs-on: self-hosted`. Update to `ubuntu-latest` (or your org's runner label) and adjust the `PYTHON` env variable path accordingly.

---

## 6. External Data Dependencies

| Resource | URL | Purpose |
|----------|-----|---------|
| MovieLens 25M dataset | `https://files.grouplens.org/datasets/movielens/ml-25m.zip` | Source data (~250 MB zip) |

The `bootstrap` job downloads this automatically. **Network egress** from the cluster must allow HTTPS to `files.grouplens.org`. Alternatively, pre-upload the CSV files to the `bronze.landing` volume.

---

## 7. Configuration Changes Required

Before deploying, update these hardcoded values:

| File | Value to change | Current value |
|------|----------------|---------------|
| `.github/workflows/ci.yml` | `DATABRICKS_HOST` | `https://YOUR_WORKSPACE.azuredatabricks.net/` |
| `.github/workflows/_reusable-deploy.yml` | `DATABRICKS_HOST` | Same as above |
| `.github/workflows/ci.yml` | `PYTHON` path and `runs-on` | Adjust to match your runner environment |
| `resources/jobs.yml` | Email notifications | `your-email@example.com` |
| `databricks.yml` | `service_principal_name` default | `mlops_sp` |
| `databricks.yml` | VM node types | `Standard_D8ds_v5` / `Standard_E8ds_v5` (Azure-specific) |
| `databricks.yml` | Catalog names (if desired) | `dsml_dev`, `dsml_staging`, `dsml_prod` |

---

## 8. Resources Likely Requiring a Ticket

The following resources typically require admin action or a ticket in most organizations:

| # | Resource | Who to contact | Why |
|---|----------|---------------|-----|
| 1 | **Unity Catalog catalogs** (x3) | Metastore admin | Catalog creation is often restricted |
| 2 | **Service principal** (`mlops_sp`) | Workspace admin / identity team | SP creation + OAuth credential generation |
| 3 | **Service principal metastore grant** | Metastore admin | SP needs `USE_CATALOG` + `CREATE_SCHEMA` |
| 4 | **GitHub Actions runner** | DevOps / platform team | Provision runner with Databricks CLI + Python 3.10+ |
| 5 | **Network egress rule** | Network / security team | Allow cluster outbound to `files.grouplens.org` |
| 6 | **GitHub environments + secrets** | Repo admin | Create `dev`/`staging`/`prod` environments and store SP credentials |

---

## 9. Deployment Order

Once all resources are provisioned, deploy in this order:

```
1. Configure ~/.databrickscfg with DEFAULT profile (or set env vars)
2. Set GitHub secrets (DATABRICKS_CLIENT_ID, DATABRICKS_CLIENT_SECRET)
3. Update DATABRICKS_HOST in workflow files
4. Run: python scripts/setup_catalog.py --target dev
5. Run: databricks bundle deploy -t dev
6. Run: databricks bundle run -t dev bootstrap
7. Run: databricks bundle run -t dev initial_training
8. Repeat steps 4-7 for staging and prod (prod uses promote_model instead of initial_training)
```

---

## 10. Python Dependencies

Listed in `requirements.txt`:

| Package | Purpose |
|---------|---------|
| `databricks-connect` | Remote Spark session |
| `databricks-sdk` | Workspace API client |
| `databricks-feature-engineering` | Feature Store client |
| `mlflow` | Experiment tracking + model registry |
| `lightgbm` | Model algorithm |
| `scikit-learn` | Train/test split, metrics |
| `pandas` | Local data manipulation |
| `numpy` | Numerical operations |
| `shap` | Model explainability |
| `pytest` | Unit testing |
| `pytest-cov` | Coverage reporting |
| `ruff` | Linting and formatting |

> Cluster jobs rely on ML Runtime 15.4 which includes most ML libraries. The project wheel is uploaded as a job library.
