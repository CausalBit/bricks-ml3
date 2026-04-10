"""Central configuration for the ML3 project.

Every constant, threshold, genre list, and hyperparameter default lives here.
No magic numbers should appear anywhere else in the codebase.
"""

from typing import Dict, List

# ---------------------------------------------------------------------------
# Genre definitions (18 genres from dataset_info.md lines 89-107)
# ---------------------------------------------------------------------------
GENRES: List[str] = [
    "Action",
    "Adventure",
    "Animation",
    "Children",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
]

NOKIDS_EXCLUDE_GENRES: List[str] = ["Children", "Animation", "Fantasy"]

NOKIDS_GENRES: List[str] = [g for g in GENRES if g not in NOKIDS_EXCLUDE_GENRES]

# ---------------------------------------------------------------------------
# Feature engineering parameters
# ---------------------------------------------------------------------------
DECAY_LAMBDA: float = 0.001  # per-day exponential decay for recency score
HOLDOUT_PERCENTILE: float = 0.8  # 80th-percentile timestamp split for holdout
TRAIN_VAL_PERCENTILE: float = 0.6  # boundary between train and validation subsets
TRAIN_TEST_SPLIT_PERCENTILE: float = 0.8  # boundary between validation and test/holdout

# ---------------------------------------------------------------------------
# Validation metric thresholds
# ---------------------------------------------------------------------------
RMSE_THRESHOLD: float = 1.5
R2_THRESHOLD: float = 0.05
PER_GENRE_RMSE_THRESHOLD: float = 2.0
SLICE_R2_THRESHOLD: float = -0.05

# ---------------------------------------------------------------------------
# Activity-level slice boundaries (number of ratings)
# ---------------------------------------------------------------------------
LOW_ACTIVITY_MAX: int = 50
MEDIUM_ACTIVITY_MAX: int = 200

# ---------------------------------------------------------------------------
# Hyperparameter defaults
# ---------------------------------------------------------------------------
HYPERPARAMS_DEV: Dict[str, object] = {
    "n_estimators": 200,
    "learning_rate": 0.1,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
}

HYPERPARAMS_PROD: Dict[str, object] = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 20,
}

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
SAMPLE_FRACTION_DEV: float = 0.2
SAMPLE_FRACTION_PROD: float = 1.0

# ---------------------------------------------------------------------------
# New-data simulation
# ---------------------------------------------------------------------------
SIMULATION_DAYS_WINDOW: int = 30

# ---------------------------------------------------------------------------
# Schema names (used with catalog variable to build full table paths)
# ---------------------------------------------------------------------------
SCHEMA_BRONZE: str = "bronze"
SCHEMA_SILVER: str = "silver"
SCHEMA_GOLD: str = "gold"
SCHEMA_ML: str = "ml"
SCHEMA_INFERENCE: str = "inference"

# ---------------------------------------------------------------------------
# Table names (bare, without catalog prefix)
# ---------------------------------------------------------------------------
TABLE_RATINGS: str = "ratings"
TABLE_MOVIES: str = "movies"
TABLE_TAGS: str = "tags"
TABLE_GENOME_SCORES: str = "genome_scores"
TABLE_GENOME_TAGS: str = "genome_tags"
TABLE_LINKS: str = "links"

TABLE_RATINGS_CLEAN: str = "ratings_clean"
TABLE_MOVIES_GENRE_EXPLODED: str = "movies_genre_exploded"
TABLE_MOVIES_GENRE_EXPLODED_NOKIDS: str = "movies_genre_exploded_nokids"
TABLE_GENOME_GENRE_AGG: str = "genome_genre_agg"
TABLE_RATINGS_HOLDOUT: str = "ratings_holdout"

TABLE_USER_GENRE_FEATURES: str = "user_genre_features"
TABLE_USER_PROFILE_FEATURES: str = "user_profile_features"

TABLE_SPLIT_METADATA: str = "split_metadata"

TABLE_SCORES_DAILY: str = "genre_propensity_scores_daily"
TABLE_SCORES_DAILY_NOKIDS: str = "genre_propensity_scores_daily_nokids"

# ---------------------------------------------------------------------------
# Model names (bare, without catalog prefix)
# ---------------------------------------------------------------------------
MODEL_GENERAL: str = "genre_propensity_general"
MODEL_NOKIDS: str = "genre_propensity_nokids"

# ---------------------------------------------------------------------------
# Volume
# ---------------------------------------------------------------------------
VOLUME_LANDING: str = "landing"

# ---------------------------------------------------------------------------
# CSV file names (ML-25M dataset)
# ---------------------------------------------------------------------------
CSV_FILES: Dict[str, str] = {
    "ratings": "ratings.csv",
    "movies": "movies.csv",
    "tags": "tags.csv",
    "genome_scores": "genome-scores.csv",
    "genome_tags": "genome-tags.csv",
    "links": "links.csv",
}

# ---------------------------------------------------------------------------
# Serving endpoint names
# ---------------------------------------------------------------------------
ENDPOINT_GENERAL: str = "genre-propensity-general"
ENDPOINT_NOKIDS: str = "genre-propensity-nokids"

# ---------------------------------------------------------------------------
# Monitoring
# ---------------------------------------------------------------------------
PSI_DRIFT_THRESHOLD: float = 0.2
TABLE_MONITORING_LOG: str = "monitoring_log"
TABLE_FEATURE_BASELINE: str = "feature_baseline"
