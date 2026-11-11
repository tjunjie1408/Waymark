"""Exercise orchestration/artifacts with explicitly fake estimators, never learned-model fits."""

from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from waymark.feature_schema import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERICAL_FEATURES
from waymark.sagemaker_entry import main
from waymark.training import parse_training_artifacts


class FakeEstimator:
    """Deterministic fake; fit records no state and executes no optimization."""

    def __init__(self, **kwargs):
        self.alpha = kwargs.get("alpha", 0.5)

    def fit(self, *args, **kwargs):
        return self

    def predict(self, frame):
        return np.full(len(frame), 0.5 + 4 * self.alpha)

    def predict_proba(self, frame):
        p = np.where(frame.index.to_numpy() % 2, 0.8, 0.2)
        return np.column_stack([1 - p, p])

    def save_model(self, path):
        Path(path).write_bytes(b"synthetic model placeholder, no fit")

    def get_feature_importance(self, data, **kwargs):
        return np.zeros((len(data.frame), len(MODEL_FEATURES) + 1))


class FakeCalibrator(FakeEstimator):
    def __init__(self, *args, **kwargs):
        super().__init__()


class FakePool:
    def __init__(self, frame, **kwargs):
        self.frame = frame


@pytest.mark.parametrize("allow_holdout", [False, True])
def test_mocked_remote_evaluation_writes_complete_identity_bound_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow_holdout: bool,
) -> None:
    for module_name, symbols in {
        "catboost": {"CatBoostClassifier": FakeEstimator, "Pool": FakePool},
        "lightgbm": {"LGBMRegressor": FakeEstimator},
        "xgboost": {"XGBClassifier": FakeEstimator},
        "sklearn": {},
        "sklearn.calibration": {"CalibratedClassifierCV": FakeCalibrator},
    }.items():
        module = types.ModuleType(module_name)
        module.__dict__.update(symbols)
        monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setenv("SM_TRAINING_ENV", "explicit mock, no real managed job")
    monkeypatch.setenv("SM_CURRENT_HOST", "fake-estimator-test")
    monkeypatch.setenv("SM_CHECKPOINT_DIR", str(tmp_path / "checkpoints"))
    monkeypatch.setattr("waymark.sagemaker_entry.verify_packaged_source", lambda *_: "mock-source-inventory")
    monkeypatch.setattr(
        "waymark.sagemaker_entry.capture_environment",
        lambda *_: {"status": "PASS", "test_evidence": "FAKE_IMPORT_RECEIPT_NOT_REAL_ENVIRONMENT_VALIDATION"},
    )
    rows = []
    for month, role in [(1, "train"), (2, "val_calibration"), (3, "holdout_test")]:
        for day in range(1, 13):
            approved = pd.Timestamp(2018, month, day)
            row = {column: 1.0 for column in NUMERICAL_FEATURES}
            row.update({column: "category" for column in CATEGORICAL_FEATURES})
            row.update(
                order_id=f"{role}-{day}",
                split_role=role,
                is_eligible_mature=1,
                y_days=2.0,
                y_late=float((day - 1) % 2),
                order_approved_at=approved,
                order_delivered_customer_date=approved + pd.Timedelta(days=2),
                bootstrap_cluster_id=f"seller-W{day // 3}",
                state_corridor="A->B",
            )
            rows.append(row)
    training = tmp_path / "training"
    training.mkdir()
    path = training / "accepted.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    args = [
        "--model-dir",
        str(tmp_path / "model"),
        "--training",
        str(training),
        "--feature-sha256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
        "--source-sha256",
        "a" * 64,
        "--snapshot-id",
        "b" * 64,
        "--image-uri",
        "mock",
        "--allow-holdout",
        str(allow_holdout).lower(),
    ]
    main(args)
    model = tmp_path / "model"
    summary = json.loads((model / "evaluation_summary.json").read_text())
    assert summary["gate_wm3_passed"] is None
    assert summary["gate"] == ("WM3_REVIEW_REQUIRED" if allow_holdout else "WM3_INCOMPLETE")
    assert "pr_auc" in summary["risk"]
    assert summary["alert_budget"]["alert_count"] == 1  # floor(12 * 10%)
    assert len(summary["uncertainty"]) == 3
    scored = pd.read_parquet(model / "predictions.parquet")
    expected_role = "holdout_test" if allow_holdout else "val_calibration"
    assert scored.order_id.str.startswith(expected_role).all()
    frozen = json.loads((model / "development_freeze.json").read_text())
    assert frozen["holdout_authorized"] is allow_holdout
    archive_buffer = io.BytesIO()
    with tarfile.open(fileobj=archive_buffer, mode="w:gz") as archive:
        for output in model.iterdir():
            archive.add(output, arcname=output.name)
    parsed = parse_training_artifacts(archive_buffer.getvalue(), {"source_sha256": "a" * 64})
    assert parsed["evaluation_scope"] == expected_role
