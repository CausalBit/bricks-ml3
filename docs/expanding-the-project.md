# Expanding the Project

This guide explains how to grow the bricks-ml3 project when you
want to add new features, new model variants, or entirely new training
frameworks. Each section builds on the previous one, from the simplest change
to the most involved.

The CI/CD pipeline (`databricks bundle deploy`) deploys your updated code (the
wheel) and your updated infrastructure (`jobs.yml`) atomically in a single
step. This means you can make coordinated changes across library code, notebooks,
and job definitions in the same commit, and everything takes effect together.

---

## 1. Adding New Features to an Existing Feature Table

**When to use:** you want to enrich the model with more signals, and the data
describes the same entity the table already tracks — per-user for
`user_profile_features`, or per-(userId, genre) for `user_genre_features`.

**Example:** adding a "days since last rating" feature to the user profile
table, or a "genre fatigue score" to the user-genre table.

### Files to touch

| File | What to do |
|------|------------|
| `src/bricks_ml3/transformations/gold.py` | Add the computation inside the existing `_compute_user_genre_features_transform()` or `_compute_user_profile_features_transform()`. Add the new column to the final `.select()`. |
| `src/bricks_ml3/config/settings.py` | Add any new constants (thresholds, decay rates). |
| `tests/unit/test_gold.py` | Add a test for the new feature column. |

### Files you do NOT touch

- **`resources/jobs.yml`** — the same `feature_engineering` task runs the same
  notebook; it automatically picks up the new column.
- **`src/bricks_ml3/training/train.py`** — Feature Store lookups pull all columns
  from the table. The new feature flows into training automatically.
- **`src/bricks_ml3/inference/batch_score.py`** — scoring uses the model's
  signature, which includes whatever features were present at training time.
- **Notebooks** — no changes needed.

### How the pipeline picks it up

Push your branch. The `weekly_retraining` job rebuilds features with the new
column, trains a model that uses it, validates, promotes, and batch scores.
Query the inference table to see the effect.

For faster iteration, select `feature_backfill` from the manual dropdown (or
use the Run Job workflow) to rebuild just the feature tables without retraining.
Once you are happy with the feature shape, push again and let the full pipeline
run.

---

## 2. Adding a New Feature Table or a New Model Variant

**When to use:** you want features at a granularity the existing tables do not
cover, or you want to train a model for a different audience segment.

### Adding a new feature table

**Example:** you want `movie_popularity_features` keyed by `(movieId)` with
columns like total ratings, average rating, and trending score.

| File | What to do |
|------|------------|
| `src/bricks_ml3/config/settings.py` | Add `TABLE_MOVIE_POPULARITY_FEATURES` (or whatever the table name is). |
| `src/bricks_ml3/transformations/gold.py` | Add a new public function, e.g. `build_movie_popularity_features()`. Follow the same pattern as the existing functions: read silver tables, compute features, register with Feature Store. |
| `src/notebooks/03_feature_engineering.py` | Add a call to your new function after the existing ones. |
| `src/bricks_ml3/training/train.py` | Add a `FeatureLookup` in `create_training_set()` pointing to the new table with the appropriate `lookup_key`. |
| `tests/` | Add unit tests for the new feature function. |

**Important:** if the new table has a different primary key than `(userId, genre)`,
you need to aggregate up to the level the model expects before the Feature Store
lookup can join it. For example, compute "average movie popularity of movies this
user watched in this genre" so the table joins on `(userId, genre)`.

### Adding a new model variant

**Example:** you want a "horror\_fans" model that only predicts propensity for
horror-related genres, similar to how "nokids" excludes children's genres.

| File | What to do |
|------|------------|
| `src/bricks_ml3/config/settings.py` | Add `MODEL_HORROR`, `TABLE_SCORES_DAILY_HORROR`, and genre list constants. |
| `src/bricks_ml3/training/train.py` | Add logic to `build_training_labels()` and `train_model()` so they handle `model_variant="horror_fans"` (similar to how "nokids" filters genres). |
| `src/bricks_ml3/inference/batch_score.py` | Add the new variant to the model-name and inference-table dispatch. |
| `src/notebooks/09_promote_or_reject.py` | Add the new variant to the `variants` dict so promotion covers it. |
| `resources/jobs.yml` | Add `train_horror`, `validate_horror`, and `batch_score_horror` tasks to `weekly_retraining` (and `initial_training` if you want it in setup). These tasks reuse the same notebooks (`04_train.py`, `05_validate.py`, `07_batch_score.py`) with `model_variant: horror_fans` as a parameter. |
| `scripts/setup_catalog.py` | If the new variant writes to a new inference table, add it to the schema setup. |

### How the pipeline picks it up

All changes go in the same commit. When you push, `databricks bundle deploy`
deploys the updated wheel (with your new code) AND the updated `jobs.yml` (with
new tasks) atomically. The `weekly_retraining` job now includes your new tasks.

---

## 3. Adding a Model That Uses a Different Training Framework

**When to use:** you want to train a model using PyTorch, XGBoost, a Hugging
Face transformer, or any framework other than LightGBM — while keeping the
existing LightGBM models running alongside it.

**Example:** a deep learning genre propensity model using PyTorch that consumes
the same user-genre features.

This builds on Section 2 (new model variant) but adds framework-specific
considerations.

### New files to create

| File | Purpose |
|------|---------|
| `src/bricks_ml3/training/train_pytorch.py` | New training module. Follow the same contract as `train.py`: accept `(spark, catalog, model_variant, hyperparams, sample_fraction)`, return `(registered_model_name, model_version)`. Use `mlflow.pytorch.log_model()` instead of `mlflow.sklearn.log_model()`. You can import the shared building blocks from `train.py`: `build_training_labels()`, `create_training_set()`, `_pivot_to_multi_output()` — these are framework-agnostic. |
| `src/notebooks/04_train_pytorch.py` (optional) | A new training notebook that imports from `train_pytorch.py`. Alternatively, extend the existing `04_train.py` with a `framework` widget parameter and dispatch to the right module. |

### Files to modify

| File | What to do |
|------|------------|
| `src/bricks_ml3/config/settings.py` | Add model name and hyperparameter constants for the new framework. |
| `resources/jobs.yml` | Add the new training, validation, and scoring tasks. If the framework needs GPU, define a second `job_cluster_key`: |

```yaml
job_clusters:
  - job_cluster_key: ml_cluster        # Existing: CPU for LightGBM
    new_cluster:
      spark_version: "15.4.x-cpu-ml-scala2.12"
      # ...
  - job_cluster_key: gpu_cluster       # New: GPU for PyTorch
    new_cluster:
      spark_version: "15.4.x-gpu-ml-scala2.12"
      node_type_id: Standard_NC6s_v3
      # ...
```

Each task can reference a different cluster — this is how DABs supports
heterogeneous ML workloads in a single job.

### Files you do NOT need to change

- **`batch_score.py`** — it uses `mlflow.pyfunc.load_model()`, which loads any
  MLflow model flavor. Your PyTorch model is scored the same way as the LightGBM
  model.
- **`gold.py`** and the feature engineering notebook — features are
  framework-agnostic.
- **`deploy_code.py`** — alias promotion (`@Challenger` to `@Champion`) works
  the same regardless of model flavor.
- **GitHub workflows** — the CI/CD pipeline deploys everything atomically; it
  does not care what framework the model uses.

### Dependencies

If the new framework needs Python packages not in ML Runtime, add them to
`requirements.txt` and to the `libraries` section of the DABs task (as a PyPI
dependency), or include them in the wheel's `install_requires` in `setup.py`.

### Combining with new features

If the new model needs completely different features (different granularity,
different data), combine this section with Section 2. You would create both a
new feature table and a new training module.

---

## Quick Reference

| Scenario | Files to touch |
|----------|---------------|
| Add a feature column | `gold.py`, `settings.py` |
| Add a feature table | `gold.py`, `settings.py`, `03_feature_engineering.py`, `train.py` (FeatureLookup) |
| Add a model variant (same framework) | `settings.py`, `train.py`, `batch_score.py`, `09_promote_or_reject.py`, `jobs.yml` |
| Add a model with a different framework | Everything above, plus a new `training/train_*.py` module and possibly a GPU cluster in `jobs.yml` |
