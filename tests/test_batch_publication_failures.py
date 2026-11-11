"""No-fit batch publication race/retry counterexamples."""

from __future__ import annotations

import json
from dataclasses import asdict

import pandas as pd
import pytest

from waymark.batch_adapter import (
    REQUIRED_PREDICTION_COLUMNS,
    LogicalBatchSpec,
    staging_output_prefix,
    success_marker_key,
    validate_and_publish_staging_predictions,
)


class PublicationClient:
    def __init__(self):
        self.objects = {}
        self.fail_marker = False
        self.competing_marker = None

    def object_exists(self, bucket, key):
        return key in self.objects

    def read_json(self, bucket, key):
        return json.loads(self.objects[key])

    def read_bytes(self, bucket, key):
        return self.objects[key]

    def put_bytes_immutable(self, bucket, key, body):
        if key in self.objects and self.objects[key] != body:
            raise ValueError("Object conflict")
        self.objects[key] = body

    def write_json_if_absent(self, bucket, key, payload):
        if self.fail_marker:
            raise OSError("Injected commit failure")
        if self.competing_marker:
            self.objects[key] = json.dumps(self.competing_marker).encode()
            return False
        if key in self.objects:
            return False
        self.objects[key] = json.dumps(payload).encode()
        return True


def predictions(value=2.0):
    return pd.DataFrame(
        {
            "order_id": ["a"],
            "eta_p10_days": [1.0],
            "eta_p50_days": [value],
            "eta_p90_days": [3.0],
            "prob_late": [0.2],
            "risk_alert_flag": [0],
        }
    ).to_parquet(index=False)


def test_second_publication_cannot_change_committed_payload():
    client = PublicationClient()
    spec = LogicalBatchSpec("model", "input", "2018-01-01")
    identity = spec.compute_logical_batch_id()
    key1 = staging_output_prefix(identity, 1) + "predictions.parquet"
    key2 = staging_output_prefix(identity, 2) + "predictions.parquet"
    client.objects[key1], client.objects[key2] = predictions(), predictions(2.5)
    args = dict(
        client=client,
        bucket="fake",
        spec=spec,
        expected_row_count=1,
        observed_row_count=1,
        observed_columns=REQUIRED_PREDICTION_COLUMNS,
    )
    first = validate_and_publish_staging_predictions(attempt_seq=1, staging_key=key1, **args)
    before = dict(client.objects)
    client.fail_marker = True
    second = validate_and_publish_staging_predictions(attempt_seq=2, staging_key=key2, **args)
    assert second["status"] == "DUPLICATE_SKIPPED"
    assert second["committed_gold_uri"] == first["committed_gold_uri"]
    assert client.objects == before


def test_failed_commit_leaves_no_visible_success_and_retry_can_commit():
    client = PublicationClient()
    spec = LogicalBatchSpec("model", "input", "2018-01-01")
    identity = spec.compute_logical_batch_id()
    key = staging_output_prefix(identity, 1) + "predictions.parquet"
    client.objects[key] = predictions()
    args = dict(
        client=client,
        bucket="fake",
        spec=spec,
        attempt_seq=1,
        staging_key=key,
        expected_row_count=1,
        observed_row_count=1,
        observed_columns=REQUIRED_PREDICTION_COLUMNS,
    )
    client.fail_marker = True
    with pytest.raises(OSError):
        validate_and_publish_staging_predictions(**args)
    assert success_marker_key(identity) not in client.objects
    client.fail_marker = False
    assert validate_and_publish_staging_predictions(**args)["status"] == "PUBLISHED"


def test_competing_publisher_keeps_first_committed_object():
    client = PublicationClient()
    spec = LogicalBatchSpec("model", "input", "2018-01-01")
    identity = spec.compute_logical_batch_id()
    key = staging_output_prefix(identity, 1) + "predictions.parquet"
    client.objects[key] = predictions()
    client.competing_marker = {"spec": asdict(spec), "gold_s3_uri": "s3://fake/first-winner"}
    result = validate_and_publish_staging_predictions(
        client=client,
        bucket="fake",
        spec=spec,
        attempt_seq=1,
        staging_key=key,
        expected_row_count=1,
        observed_row_count=1,
        observed_columns=REQUIRED_PREDICTION_COLUMNS,
    )
    assert result["status"] == "DUPLICATE_SKIPPED"
    assert result["committed_gold_uri"] == "s3://fake/first-winner"


def test_missing_staging_never_fabricates_payload():
    client = PublicationClient()
    spec = LogicalBatchSpec("model", "input", "2018-01-01")
    identity = spec.compute_logical_batch_id()
    with pytest.raises(KeyError):
        validate_and_publish_staging_predictions(
            client=client,
            bucket="fake",
            spec=spec,
            attempt_seq=1,
            staging_key=staging_output_prefix(identity, 1) + "missing.parquet",
            expected_row_count=1,
            observed_row_count=1,
            observed_columns=REQUIRED_PREDICTION_COLUMNS,
        )
    assert success_marker_key(identity) not in client.objects
