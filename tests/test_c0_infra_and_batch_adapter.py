"""Tests for C0 unapplied Terraform definitions and O8 Batch Transform idempotency adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from waymark.batch_adapter import (
    REQUIRED_PREDICTION_COLUMNS,
    LogicalBatchSpec,
    gold_predictions_key,
    lambda_handler,
    staging_output_prefix,
    success_marker_key,
    trigger_batch_transform_idempotent,
    validate_and_publish_staging_predictions,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
TF_DIR = REPO_ROOT / "infra" / "terraform"


class FakeBatchClient:
    """In-memory mock of S3 and SageMaker Batch Transform for deterministic C0/O8 tests."""

    def __init__(self) -> None:
        self.objects: dict[tuple[str, str], Any] = {}
        self.active_jobs: list[str] = []
        self.created_requests: list[dict[str, Any]] = []

    def object_exists(self, bucket: str, key: str) -> bool:
        return (bucket, key) in self.objects

    def read_json(self, bucket: str, key: str) -> dict[str, Any]:
        return dict(self.objects[(bucket, key)])

    def write_json(self, bucket: str, key: str, payload: dict[str, Any]) -> None:
        self.objects[(bucket, key)] = dict(payload)

    def list_active_transform_jobs(self, name_prefix: str) -> list[str]:
        return [j for j in self.active_jobs if j.startswith(name_prefix)]

    def create_transform_job(self, request: dict[str, Any]) -> str:
        self.created_requests.append(request)
        job_name = request["TransformJobName"]
        self.active_jobs.append(job_name)
        return f"arn:aws:sagemaker:ap-southeast-2:149270211548:transform-job/{job_name}"

    def copy_staging_to_gold(self, bucket: str, staging_key: str, gold_key: str) -> None:
        self.objects[(bucket, gold_key)] = self.objects[(bucket, staging_key)]

    def read_bytes(self, bucket: str, key: str) -> bytes:
        return self.objects[(bucket, key)]

    def put_bytes_immutable(self, bucket: str, key: str, body: bytes) -> None:
        if (bucket, key) in self.objects and self.objects[(bucket, key)] != body:
            raise ValueError("Immutable conflict")
        self.objects[(bucket, key)] = body

    def write_json_if_absent(self, bucket: str, key: str, payload: dict[str, Any]) -> bool:
        if (bucket, key) in self.objects:
            return False
        self.objects[(bucket, key)] = dict(payload)
        return True


def test_c0_terraform_files_and_o7_o8_invariants() -> None:
    """Verify C0 Terraform definitions exist and encode frozen O7/O8 parameters."""
    expected_files = ["versions.tf", "variables.tf", "main.tf", "outputs.tf"]
    combined_tf = ""
    for fname in expected_files:
        fpath = TF_DIR / fname
        assert fpath.exists(), f"Missing Terraform file: {fpath}"
        text = fpath.read_text(encoding="utf-8")
        assert "\u2014" not in text, f"Em-dash found in {fname}"
        assert "\u2013" not in text, f"En-dash found in {fname}"
        combined_tf += "\n" + text

    # O7 account, region, and service sizing checks
    assert '"149270211548"' in combined_tf
    assert '"ap-southeast-2"' in combined_tf
    assert '"waymark"' in combined_tf
    assert "0.0625" in combined_tf
    assert '"pythonshell"' in combined_tf
    assert '"Athena engine version 3"' in combined_tf
    assert "1073741824" in combined_tf
    assert '"ml.m5.large"' in combined_tf

    # O8 Lambda + EventBridge Scheduler checks (DISABLED by default)
    assert '"python3.12"' in combined_tf
    assert '"arm64"' in combined_tf
    assert '"batch_adapter.lambda_handler"' in combined_tf
    assert "enable_batch_schedule" in combined_tf
    assert '"DISABLED"' in combined_tf


def test_o8_logical_batch_id_and_idempotent_trigger() -> None:
    """Verify deterministic batch IDs, active job suppression, and _SUCCESS marker idempotency."""
    spec = LogicalBatchSpec(
        model_artifact_hash="sha256_model_v1",
        input_snapshot_hash="ed79c125992d7f63",
        scoring_cutoff_date="2018-08-29",
        config_version="wm0_v1",
    )
    batch_id = spec.compute_logical_batch_id()
    assert len(batch_id) == 16
    assert spec.format_transform_job_name(1) == f"wm-bt-{batch_id}-01"
    assert spec.format_transform_job_name(2) == f"wm-bt-{batch_id}-02"

    client = FakeBatchClient()
    bucket = "waymark-lake-pilot-149270211548-ap-southeast-2"

    # 1. First trigger starts a new TransformJob
    res1 = trigger_batch_transform_idempotent(
        client=client,
        bucket=bucket,
        spec=spec,
        attempt_seq=1,
        model_name="waymark-m1-m2-v1",
        input_s3_key="gold/batch_inputs/test.csv",
    )
    assert res1["status"] == "STARTED"
    assert res1["transform_job_name"] == f"wm-bt-{batch_id}-01"
    assert len(client.created_requests) == 1

    # 2. Duplicate trigger while job is still InProgress is skipped without creating a second job
    res2 = trigger_batch_transform_idempotent(
        client=client,
        bucket=bucket,
        spec=spec,
        attempt_seq=1,
        model_name="waymark-m1-m2-v1",
        input_s3_key="gold/batch_inputs/test.csv",
    )
    assert res2["status"] == "DUPLICATE_SKIPPED"
    assert res2["reason"] == "ACTIVE_TRANSFORM_JOB_IN_PROGRESS"
    assert len(client.created_requests) == 1

    # 3. Simulate first attempt failing (no longer active, no _SUCCESS marker) -> attempt_seq=2 starts
    client.active_jobs.clear()
    res3 = trigger_batch_transform_idempotent(
        client=client,
        bucket=bucket,
        spec=spec,
        attempt_seq=2,
        model_name="waymark-m1-m2-v1",
        input_s3_key="gold/batch_inputs/test.csv",
    )
    assert res3["status"] == "STARTED"
    assert res3["transform_job_name"] == f"wm-bt-{batch_id}-02"
    assert len(client.created_requests) == 2

    # 4. Once _SUCCESS marker is committed, subsequent triggers are skipped immediately
    client.active_jobs.clear()
    client.write_json(bucket, success_marker_key(batch_id), {"status": "COMMITTED"})
    res4 = trigger_batch_transform_idempotent(
        client=client,
        bucket=bucket,
        spec=spec,
        attempt_seq=3,
        model_name="waymark-m1-m2-v1",
        input_s3_key="gold/batch_inputs/test.csv",
    )
    assert res4["status"] == "DUPLICATE_SKIPPED"
    assert res4["reason"] == "GOLD_SUCCESS_MARKER_EXISTS"
    assert len(client.created_requests) == 2


def test_o8_staging_validation_rejects_partial_or_invalid_outputs() -> None:
    """Verify partial row counts, missing columns, or quantile crossings never publish to Gold."""
    spec = LogicalBatchSpec(
        model_artifact_hash="sha256_model_v1",
        input_snapshot_hash="ed79c125992d7f63",
        scoring_cutoff_date="2018-08-29",
    )
    batch_id = spec.compute_logical_batch_id()
    client = FakeBatchClient()
    bucket = "waymark-lake-pilot-149270211548-ap-southeast-2"
    staging_key = f"{staging_output_prefix(batch_id, 1)}predictions.parquet"

    # Case A: Partial row count (e.g. 14,000 instead of 14,460)
    partial_res = validate_and_publish_staging_predictions(
        client=client,
        bucket=bucket,
        spec=spec,
        attempt_seq=1,
        staging_key=staging_key,
        expected_row_count=14460,
        observed_row_count=14000,
        observed_columns=REQUIRED_PREDICTION_COLUMNS,
    )
    assert partial_res["status"] == "PUBLICATION_REJECTED"
    assert not client.object_exists(bucket, success_marker_key(batch_id))
    assert not client.object_exists(bucket, gold_predictions_key(batch_id))

    # Case B: Unrepaired quantile crossing
    crossing_res = validate_and_publish_staging_predictions(
        client=client,
        bucket=bucket,
        spec=spec,
        attempt_seq=1,
        staging_key=staging_key,
        expected_row_count=14460,
        observed_row_count=14460,
        observed_columns=REQUIRED_PREDICTION_COLUMNS,
        quantile_crossing_count=5,
    )
    assert crossing_res["status"] == "PUBLICATION_REJECTED"
    assert not client.object_exists(bucket, success_marker_key(batch_id))

    # Case C: Valid complete batch publishes to Gold and writes _SUCCESS marker
    client.objects[(bucket, staging_key)] = pd.DataFrame(
        {
            "order_id": [f"synthetic-{i}" for i in range(14460)],
            "eta_p10_days": 1.0,
            "eta_p50_days": 2.0,
            "eta_p90_days": 3.0,
            "prob_late": 0.2,
            "risk_alert_flag": 0,
        }
    ).to_parquet(index=False)
    valid_res = validate_and_publish_staging_predictions(
        client=client,
        bucket=bucket,
        spec=spec,
        attempt_seq=1,
        staging_key=staging_key,
        expected_row_count=14460,
        observed_row_count=14460,
        observed_columns=REQUIRED_PREDICTION_COLUMNS,
    )
    assert valid_res["status"] == "PUBLISHED"
    assert client.object_exists(bucket, success_marker_key(batch_id))
    assert client.object_exists(bucket, valid_res["committed_gold_uri"].removeprefix(f"s3://{bucket}/"))
    manifest = client.read_json(bucket, success_marker_key(batch_id))
    assert manifest["expected_row_count"] == 14460
    assert manifest["logical_batch_id"] == batch_id


def test_lambda_handler_invokes_with_mocked_boto3(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify lambda_handler parses EventBridge Scheduler payload and delegates to client."""
    fake_client = FakeBatchClient()
    monkeypatch.setattr("waymark.batch_adapter.Boto3BatchClient", lambda: fake_client)

    event = {
        "bucket": "waymark-lake-pilot-149270211548-ap-southeast-2",
        "model_name": "waymark-pilot-m1-m2-model",
        "model_artifact_hash": "abc123model",
        "input_snapshot_hash": "ed79c125992d7f63",
        "scoring_cutoff_date": "2018-08-29",
        "config_version": "wm0_v1",
        "attempt_seq": 1,
    }
    response = lambda_handler(event, None)
    assert response["status"] == "STARTED"
    assert response["transform_job_name"].startswith("wm-bt-")
