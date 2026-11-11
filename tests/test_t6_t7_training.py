"""Unit tests for T6/T7 remote SageMaker training packaging and evaluation metrics."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np

from waymark.sagemaker_entry import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    expected_calibration_error,
    pinball_loss,
)
from waymark.training import SKLEARN_CONTAINER_URI, create_sourcedir_tarball

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_sourcedir_tarball_packaging() -> None:
    """Verify in-memory tarball packaging contains entrypoint and requirements."""
    tar_bytes = create_sourcedir_tarball(REPO_ROOT)
    assert len(tar_bytes) > 0, "Tarball must not be empty"

    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        names = tar.getnames()
        assert "train_entry.py" in names, "train_entry.py must be present"
        assert "requirements.txt" in names, "requirements.txt must be present"

        req_file = tar.extractfile("requirements.txt")
        assert req_file is not None
        req_text = req_file.read().decode("utf-8")
        assert "lightgbm" in req_text
        assert "catboost" in req_text
        assert "xgboost" in req_text
        assert "shap" in req_text


def test_feature_list_anti_leakage_invariants() -> None:
    """Verify remote feature schema strictly excludes post-approval leakage."""
    all_features = set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)
    forbidden = {
        "delivery_duration_days",
        "is_late_delivered",
        "order_delivered_customer_date",
        "order_delivered_carrier_date",
        "review_score",
        "review_answer_timestamp",
        "y_days",
        "y_late",
    }
    leakage = all_features.intersection(forbidden)
    assert not leakage, f"Feature list contains forbidden post-approval columns: {leakage}"


def test_pinball_loss_and_ece_metrics() -> None:
    """Verify pinball loss and ECE metrics on synthetic values."""
    y_true = np.array([10.0, 15.0, 20.0])
    y_pred = np.array([12.0, 15.0, 18.0])

    p50_loss = pinball_loss(y_true, y_pred, alpha=0.50)
    expected_p50 = 0.5 * np.mean(np.abs(y_true - y_pred))
    assert abs(p50_loss - expected_p50) < 1e-6

    # Perfectly calibrated probabilities
    y_binary = np.array([0, 1, 0, 1])
    y_perfect_prob = np.array([0.05, 0.95, 0.05, 0.95])
    ece_good = expected_calibration_error(y_binary, y_perfect_prob, n_bins=5)
    assert ece_good < 0.10

    # Container URI in ap-southeast-2
    assert "ap-southeast-2" in SKLEARN_CONTAINER_URI
