"""Unit tests for model validation checks.

All tests run on a local SparkSession -- no Databricks cluster required.
Uses mocks for ``WorkspaceClient``, ``dbutils``, and ``mlflow`` to isolate
the validation logic from platform dependencies.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import r2_score

from bricks_ml3.config.settings import (
    PER_GENRE_RMSE_THRESHOLD,
    R2_THRESHOLD,
    RMSE_THRESHOLD,
    SLICE_R2_THRESHOLD,
)
from bricks_ml3.validation.validate import (
    _check_activity_slices,
    _check_artifacts,
    _check_champion_comparison,
    _check_description,
    _check_metrics,
    _check_per_genre_rmse,
    _check_r2_threshold,
    _check_rmse_threshold,
    _compute_per_genre_metrics,
    _set_governance_tags,
    _smoke_test,
)

# -- helpers -----------------------------------------------------------------


def _make_good_predictions(n_users: int, n_genres: int, noise: float = 0.1):
    """Generate synthetic true/predicted values that pass all thresholds."""
    rng = np.random.RandomState(42)
    genres = sorted([f"Genre_{i}" for i in range(n_genres)])
    y_true = pd.DataFrame(
        rng.rand(n_users, n_genres) * 4.0 + 0.5,
        columns=genres,
    )
    y_pred = y_true.values + rng.randn(n_users, n_genres) * noise
    return y_true, y_pred, genres


# -- test_all_checks_pass ----------------------------------------------------


@pytest.mark.unit
@patch("bricks_ml3.validation.validate.mlflow")
def test_all_checks_pass(mock_mlflow) -> None:
    """Model that passes all checks gets @Challenger alias and PASSED status."""
    n_users, n_genres = 50, 3
    y_true, y_pred, genres = _make_good_predictions(n_users, n_genres, noise=0.05)

    mock_model = MagicMock()
    mock_model.predict.return_value = y_pred

    mock_mlflow.pyfunc.load_model.return_value = mock_model

    mock_model_info = MagicMock()
    mock_model_info.signature = MagicMock()
    mock_mlflow.models.get_model_info.return_value = mock_model_info

    mock_mv = MagicMock()
    mock_mv.description = "Test model description"
    mock_mlflow.MlflowClient.return_value.get_model_version.return_value = mock_mv
    mock_mlflow.MlflowClient.return_value.get_model_version_by_alias.side_effect = Exception("No Champion")

    if hasattr(_check_artifacts, "__wrapped__"):
        _check_artifacts.__wrapped__("models:/test/1")

    loaded_model = mock_mlflow.pyfunc.load_model("models:/test/1")
    assert loaded_model is not None

    mock_mlflow.MlflowClient.return_value.get_model_version.return_value = mock_mv

    metrics = _check_metrics(y_true, y_pred)
    assert metrics["rmse_overall"] < RMSE_THRESHOLD
    assert metrics["r2_overall"] > R2_THRESHOLD

    _check_rmse_threshold(metrics)
    _check_r2_threshold(metrics)

    per_genre = _compute_per_genre_metrics(y_true, y_pred, genres)
    _check_per_genre_rmse(per_genre)

    sample_X = pd.DataFrame(np.random.rand(5, 10), columns=[f"f_{i}" for i in range(10)])
    _smoke_test(mock_model, sample_X)

    w = MagicMock()
    _set_governance_tags(w, "test.ml.model", "1")

    w.model_registry.set_registered_model_alias.assert_not_called()

    w.model_registry.set_registered_model_alias(name="test.ml.model", alias="Challenger", version="1")
    w.model_registry.set_registered_model_alias.assert_called_once_with(
        name="test.ml.model", alias="Challenger", version="1"
    )


# -- test_rmse_fails ---------------------------------------------------------


@pytest.mark.unit
def test_rmse_fails() -> None:
    """Model with RMSE above threshold fails check 5 — no alias assigned."""
    rng = np.random.RandomState(99)
    n_users, n_genres = 50, 3
    genres = sorted([f"Genre_{i}" for i in range(n_genres)])
    y_true = pd.DataFrame(
        rng.rand(n_users, n_genres) * 4.0 + 0.5,
        columns=genres,
    )
    y_pred = y_true.values + rng.randn(n_users, n_genres) * 2.0

    metrics = _check_metrics(y_true, y_pred)
    assert metrics["rmse_overall"] >= RMSE_THRESHOLD

    with pytest.raises(ValueError, match="Check 5 FAILED"):
        _check_rmse_threshold(metrics)


# -- test_no_champion_skips_comparison ---------------------------------------


@pytest.mark.unit
@patch("bricks_ml3.validation.validate.mlflow")
def test_no_champion_skips_comparison(mock_mlflow) -> None:
    """When no @Champion exists, check 7 is skipped (not failed)."""
    mock_mlflow.MlflowClient.return_value.get_model_version_by_alias.side_effect = Exception("RESOURCE_DOES_NOT_EXIST")

    _check_champion_comparison("test.ml.model", 0.5, pd.DataFrame(), pd.DataFrame())


# -- test_per_genre_threshold_exceeded ---------------------------------------


@pytest.mark.unit
def test_per_genre_threshold_exceeded() -> None:
    """One genre with RMSE above threshold fails check 8."""
    per_genre_rmse = {
        "Action": 0.5,
        "Comedy": 0.6,
        "Drama": PER_GENRE_RMSE_THRESHOLD + 0.1,
    }

    with pytest.raises(ValueError, match="Check 8 FAILED"):
        _check_per_genre_rmse(per_genre_rmse)


# -- test_activity_slice_below_r2 -------------------------------------------


@pytest.mark.unit
def test_activity_slice_below_r2() -> None:
    """Low-activity slice with R2 below threshold flags at check 9."""
    rng = np.random.RandomState(42)
    n_users = 30
    genres = sorted(["Action", "Comedy", "Drama"])

    y_true_good = pd.DataFrame(rng.rand(n_users, 3) * 4.0 + 0.5, columns=genres)
    y_pred_good = y_true_good.values + rng.randn(n_users, 3) * 0.1

    y_true_bad = pd.DataFrame(rng.rand(10, 3) * 4.0 + 0.5, columns=genres)
    y_pred_bad = rng.rand(10, 3) * 4.0 + 0.5

    y_true_by_slice = {"low": y_true_bad, "medium": y_true_good, "high": y_true_good}
    y_pred_by_slice = {"low": y_pred_bad, "medium": y_pred_good, "high": y_pred_good}

    low_r2 = float(r2_score(y_true_bad, y_pred_bad))
    assert low_r2 < SLICE_R2_THRESHOLD

    with pytest.raises(ValueError, match="Check 9 FAILED"):
        _check_activity_slices(y_true_by_slice, y_pred_by_slice)


# -- test_missing_description_fails ------------------------------------------


@pytest.mark.unit
@patch("bricks_ml3.validation.validate.mlflow")
def test_missing_description_fails(mock_mlflow) -> None:
    """Model with empty description fails at check 2."""
    mock_mv = MagicMock()
    mock_mv.description = ""
    mock_mlflow.MlflowClient.return_value.get_model_version.return_value = mock_mv

    with pytest.raises(ValueError, match="Check 2 FAILED"):
        _check_description("test.ml.model", "1")


# -- test_governance_tags_set_on_pass ----------------------------------------


@pytest.mark.unit
def test_governance_tags_set_on_pass() -> None:
    """All 4 governance tags are set via WorkspaceClient on check 10."""
    w = MagicMock()

    _set_governance_tags(w, "test.ml.model", "1")

    assert w.set_model_version_tag.call_count == 4

    tag_calls = w.set_model_version_tag.call_args_list
    keys_set = {call.kwargs["key"] for call in tag_calls}
    expected_keys = {
        "model_validation_status",
        "model_owner",
        "use_case",
        "data_classification",
    }
    assert keys_set == expected_keys

    for call in tag_calls:
        assert call.kwargs["name"] == "test.ml.model"
        assert call.kwargs["version"] == "1"
