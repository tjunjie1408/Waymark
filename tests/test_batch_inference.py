"""Prediction-only serving and packaging oracle, with estimators that cannot fit."""

from __future__ import annotations

import io
import json
import pickle
import tarfile

import numpy as np
import pandas as pd
import pytest
from batch_fixtures import Client, fixture

from waymark.batch_contract import FEATURE_SCHEMA_SHA256, canonical_bytes, digest
from waymark.batch_delivery import (
    finalize_batch,
    read_committed_batch,
    register_committed_table,
    submit_batch,
)
from waymark.batch_inference import input_fn, model_fn, output_fn, predict_fn
from waymark.batch_preparation import load_bundle, prepare_batch
from waymark.feature_schema import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERICAL_FEATURES
from waymark.sagemaker_entry import prepare_features


class Quantile:
    def __init__(self, value):
        self.value = value

    def predict(self, frame):
        return np.full(len(frame), self.value)

    def fit(self, *args, **kwargs):
        raise AssertionError("Fitting is forbidden")


class Risk(Quantile):
    def predict_proba(self, frame):
        return np.tile([0.3, 0.7], (len(frame), 1))


def feature_frame():
    return pd.DataFrame(
        [
            {
                "order_id": identity,
                **dict.fromkeys(NUMERICAL_FEATURES, 1.0),
                **dict.fromkeys(CATEGORICAL_FEATURES, "BR"),
            }
            for identity in ["001", "002"]
        ]
    )


def model_archive():
    _, manifest = fixture()
    environment = manifest["model"]["environment"]
    freeze = {
        "model_features": MODEL_FEATURES,
        "environment": environment,
        "categorical_levels_from_train": dict.fromkeys(CATEGORICAL_FEATURES, ["BR"]),
        "conformal_c": 0.5,
        "threshold_tau": 0.5,
    }
    files = {
        "development_freeze.json": canonical_bytes(freeze),
        **{f"m1_lgbm_p{p}.pkl": pickle.dumps(Quantile(v)) for p, v in [(10, 1), (50, 2), (90, 3)]},
        "m2_calibrated_catboost.pkl": pickle.dumps(Risk(0.7)),
        "m2_catboost.cbm": b"test-only",
        "m2_xgboost.json": b"{}",
        **{
            name: b"synthetic-evaluation-evidence"
            for name in ["predictions.parquet", "shap_values.parquet", "shap_background.parquet"]
        },
    }
    summary = {
        "schema": "waymark.training_evaluation/v2",
        "status": "OK",
        "gate": "WM3_REVIEW_REQUIRED",
        "identities": {"source_sha256": "a" * 64},
        "uncertainty": {"synthetic": True},
        "strata": {"synthetic": True},
        "model_artifacts": [{"file": k, "sha256": digest(v)} for k, v in files.items()],
        "eta": dict.fromkeys(
            [
                "pinball_loss_p10",
                "pinball_loss_p50",
                "pinball_loss_p90",
                "interval_coverage_80",
                "interval_mean_width_days",
            ],
            1,
        ),
        "risk": dict.fromkeys(["roc_auc", "pr_auc", "brier_score", "expected_calibration_error_10bin"], 1),
    }
    files["evaluation_summary.json"] = canonical_bytes(summary)
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w:gz") as archive:
        for name, body in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(body)
            archive.addfile(info, io.BytesIO(body))
    return stream.getvalue(), files, environment


def test_inference_reuses_training_levels_and_frozen_calibration(tmp_path):
    _, files, environment = model_archive()
    for name, body in files.items():
        (tmp_path / name).write_bytes(body)
    model = model_fn(tmp_path, environment)
    frame = feature_frame()
    frame.loc[1, CATEGORICAL_FEATURES[0]] = "UNSEEN"
    body = frame[["order_id", *MODEL_FEATURES]].to_csv(header=False, index=False).encode()
    parsed = input_fn(body, "text/csv")
    encoded, _ = prepare_features(frame.iloc[:1], parsed)
    assert encoded[CATEGORICAL_FEATURES[0]].isna().tolist() == [False, True]
    output = [
        json.loads(line)
        for line in output_fn(predict_fn(parsed, model), "application/jsonlines").splitlines()
    ]
    assert [row["order_id"] for row in output] == ["001", "002"]
    assert output[0] == {
        "order_id": "001",
        "eta_p10_days": 0.5,
        "eta_p50_days": 2.0,
        "eta_p90_days": 3.5,
        "prob_late": 0.7,
        "risk_alert_flag": 1,
    }
    (tmp_path / "m1_lgbm_p10.pkl").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="identity"):
        model_fn(tmp_path, environment)


def test_prepare_serve_finalize_read_and_catalog_use_one_committed_batch(tmp_path):
    archive, files, environment = model_archive()
    acceptance = canonical_bytes(
        {
            "schema": "waymark.model_acceptance/v1",
            "decision": "accepted",
            "review_reference": "synthetic-user-review-only",
            "model_sha256": digest(archive),
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "image_uri": environment["image_uri"],
        }
    )
    directory = tmp_path / "bundle"
    bucket = fixture()[1]["bucket"]
    prepare_batch(
        features=feature_frame(),
        model_archive=archive,
        acceptance=acceptance,
        environment=environment,
        model_name="accepted-v1",
        bucket=bucket,
        scoring_cutoff_date="2018-08-29",
        execution_role_arn="arn:aws:iam::149270211548:role/test",
        output=directory,
        rows_per_part=1,
    )
    manifest, payloads, request = load_bundle(directory)
    client = Client()
    client.objects.update(payloads)
    client.model = request
    started = submit_batch(client, manifest, attempt_seq=1)
    client.job = {**client.requests[0], "TransformJobStatus": "Completed"}
    model_directory = tmp_path / "models"
    model_directory.mkdir()
    for name, body in files.items():
        (model_directory / name).write_bytes(body)
    model = model_fn(model_directory, environment)
    for part in manifest["inputs"]:
        data = input_fn(client.objects[part["key"]], "text/csv")
        key = started["staging_s3_uri"].split("/", 3)[3] + part["key"] + ".out"
        client.objects[key] = output_fn(predict_fn(data, model), "application/jsonlines")
    assert finalize_batch(client, manifest, attempt_seq=1)["status"] == "PUBLISHED"
    assert finalize_batch(client, manifest, attempt_seq=2)["status"] == "DUPLICATE_SKIPPED"
    client.objects["gold/predictions/orphan.parquet"] = b"orphan"
    frame, marker = read_committed_batch(client, manifest)
    assert frame.order_id.tolist() == ["001", "002"]
    assert frame.eta_p90_days.tolist() == [3.5, 3.5]

    class Glue:
        class exceptions:
            class AlreadyExistsException(Exception):
                pass

        def create_table(self, **kwargs):
            if hasattr(self, "table"):
                raise self.exceptions.AlreadyExistsException()
            self.table = kwargs["TableInput"]

        def get_table(self, **kwargs):
            table = json.loads(json.dumps(self.table))
            table["StorageDescriptor"]["Compressed"] = False
            table["StorageDescriptor"]["SerdeInfo"]["Parameters"] = {}
            table["Parameters"]["EXTERNAL"] = "TRUE"
            return {"Table": table}

    glue = Glue()
    table = register_committed_table(client, glue, manifest, "test")
    location = glue.table["StorageDescriptor"]["Location"]
    assert marker["sha256"] in location
    assert "objects/" not in location
    assert client.list_keys(bucket, location.split("/", 3)[3]) == [
        location.split("/", 3)[3] + "predictions.parquet"
    ]
    assert table.startswith("predictions_")
    assert register_committed_table(client, glue, manifest, "test") == table
    client.objects[location.split("/", 3)[3] + "orphan.parquet"] = b"orphan"
    with pytest.raises(ValueError, match="extra objects"):
        register_committed_table(client, glue, manifest, "test")


def test_preparation_requires_acceptance(tmp_path):
    archive, _, environment = model_archive()
    acceptance = {
        "schema": "waymark.model_acceptance/v1",
        "decision": "REVIEW_REQUIRED",
        "review_reference": "synthetic",
        "model_sha256": digest(archive),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "image_uri": environment["image_uri"],
    }
    kwargs = dict(
        features=feature_frame(),
        model_archive=archive,
        acceptance=canonical_bytes(acceptance),
        environment=environment,
        model_name="accepted-v1",
        bucket=fixture()[1]["bucket"],
        scoring_cutoff_date="2018-08-29",
        execution_role_arn="arn:aws:iam::149270211548:role/test",
        output=tmp_path / "bundle",
    )
    with pytest.raises(ValueError, match="acceptance"):
        prepare_batch(**kwargs)
    assert not kwargs["output"].exists()


def test_prepare_cli_drops_training_labels_and_never_creates_aws_session(tmp_path, monkeypatch, capsys):
    from waymark.cli import main

    archive, _, environment = model_archive()
    acceptance = {
        "schema": "waymark.model_acceptance/v1",
        "decision": "accepted",
        "review_reference": "synthetic-user-review-only",
        "model_sha256": digest(archive),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "image_uri": environment["image_uri"],
    }
    features = feature_frame().assign(y_days=100.0, y_late=1)
    features.to_parquet(tmp_path / "features.parquet")
    (tmp_path / "model.tar.gz").write_bytes(archive)
    (tmp_path / "acceptance.json").write_bytes(canonical_bytes(acceptance))
    (tmp_path / "environment.json").write_bytes(canonical_bytes(environment))

    def forbidden(*args, **kwargs):
        raise AssertionError("Local preparation must not access AWS")

    monkeypatch.setattr("boto3.Session", forbidden)
    assert (
        main(
            [
                "prepare-batch",
                "--features",
                str(tmp_path / "features.parquet"),
                "--model-archive",
                str(tmp_path / "model.tar.gz"),
                "--acceptance",
                str(tmp_path / "acceptance.json"),
                "--environment-receipt",
                str(tmp_path / "environment.json"),
                "--model-name",
                "accepted-v1",
                "--scoring-cutoff-date",
                "2018-08-29",
                "--execution-role-arn",
                "arn:aws:iam::149270211548:role/test",
                "--output",
                str(tmp_path / "bundle"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "PREPARED_LOCALLY_NOT_UPLOADED"
    manifest, payloads, _ = load_bundle(tmp_path / "bundle")
    parsed = input_fn(payloads[manifest["inputs"][0]["key"]], "text/csv")
    assert list(parsed.columns) == ["order_id", *MODEL_FEATURES]
    assert parsed.order_id.tolist() == ["001", "002"]
