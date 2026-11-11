"""O8 Batch Transform trigger adapter, idempotency guard, and staging publisher.

This module runs both as the AWS Lambda handler invoked by EventBridge Scheduler
(Python 3.12 arm64, 128 MB) and as a locally testable module for C0/T8 contract
verification without cloud resource allocation.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any, Protocol


class BatchStorageAndTransformClient(Protocol):
    """Protocol abstracting S3 and SageMaker operations for deterministic testing."""

    def object_exists(self, bucket: str, key: str) -> bool:
        """Return True if s3://{bucket}/{key} exists."""

    def read_json(self, bucket: str, key: str) -> dict[str, Any]:
        """Read and parse a JSON object from S3."""

    def write_json(self, bucket: str, key: str, payload: dict[str, Any]) -> None:
        """Serialize and write a JSON object to S3."""

    def list_active_transform_jobs(self, name_prefix: str) -> list[str]:
        """Return active (InProgress) TransformJobNames matching name_prefix."""

    def create_transform_job(self, request: dict[str, Any]) -> str:
        """Start a SageMaker Batch Transform job and return its TransformJobArn."""

    def copy_staging_to_gold(
        self,
        bucket: str,
        staging_key: str,
        gold_key: str,
    ) -> None:
        """Copy validated staging predictions to the final Gold partition."""


@dataclass(frozen=True)
class LogicalBatchSpec:
    """Inputs defining a deterministic logical batch execution identity (O8)."""

    model_artifact_hash: str
    input_snapshot_hash: str
    scoring_cutoff_date: str
    config_version: str = "wm0_v1"

    def compute_logical_batch_id(self) -> str:
        """Compute SHA256(model_hash:input_hash:cutoff:config)[:16] per O8 contract."""
        if not self.model_artifact_hash or not self.input_snapshot_hash or not self.scoring_cutoff_date:
            raise ValueError("AllLogicalBatchSpec fields must be non-empty strings.")
        raw = (
            f"{self.model_artifact_hash}:{self.input_snapshot_hash}:"
            f"{self.scoring_cutoff_date}:{self.config_version}"
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def format_transform_job_name(self, attempt_seq: int = 1) -> str:
        """Format unique SageMaker TransformJobName: wm-bt-{logical_batch_id}-{attempt_seq}."""
        if attempt_seq < 1:
            raise ValueError(f"attempt_seq must be >= 1, got {attempt_seq}")
        batch_id = self.compute_logical_batch_id()
        return f"wm-bt-{batch_id}-{attempt_seq:02d}"


def success_marker_key(logical_batch_id: str) -> str:
    """Return the S3 key for the Gold batch publication commit marker."""
    return f"gold/prediction_manifests/_SUCCESS_{logical_batch_id}.json"


def staging_output_prefix(logical_batch_id: str, attempt_seq: int) -> str:
    """Return the S3 prefix where SageMaker Batch Transform writes raw staging output."""
    return f"staging/batch_raw/{logical_batch_id}/attempt={attempt_seq:02d}/"


def gold_predictions_key(logical_batch_id: str) -> str:
    """Return the S3 key for verified Gold Parquet predictions."""
    return f"gold/predictions/batch_id={logical_batch_id}/predictions.parquet"


def build_create_transform_job_request(
    *,
    spec: LogicalBatchSpec,
    attempt_seq: int,
    model_name: str,
    bucket: str,
    input_s3_key: str,
    instance_type: str = "ml.m5.large",
    instance_count: int = 1,
    max_concurrent_transforms: int = 2,
) -> dict[str, Any]:
    """Build a SageMaker CreateTransformJob request dict adhering to O7/O8 sizing."""
    logical_batch_id = spec.compute_logical_batch_id()
    job_name = spec.format_transform_job_name(attempt_seq)
    staging_prefix = staging_output_prefix(logical_batch_id, attempt_seq)

    return {
        "TransformJobName": job_name,
        "ModelName": model_name,
        "MaxConcurrentTransforms": max_concurrent_transforms,
        "MaxPayloadInMB": 6,
        "BatchStrategy": "MultiRecord",
        "TransformInput": {
            "DataSource": {
                "S3DataSource": {
                    "S3DataType": "S3Prefix",
                    "S3Uri": f"s3://{bucket}/{input_s3_key}",
                }
            },
            "ContentType": "text/csv",
            "SplitType": "Line",
        },
        "TransformOutput": {
            "S3OutputPath": f"s3://{bucket}/{staging_prefix}",
            "AssembleWith": "Line",
            "Accept": "application/jsonlines",
        },
        "TransformResources": {
            "InstanceType": instance_type,
            "InstanceCount": instance_count,
        },
        "Tags": [
            {"Key": "Project", "Value": "waymark"},
            {"Key": "LogicalBatchId", "Value": logical_batch_id},
            {"Key": "ScoringCutoffDate", "Value": spec.scoring_cutoff_date},
        ],
    }


def trigger_batch_transform_idempotent(
    *,
    client: BatchStorageAndTransformClient,
    bucket: str,
    spec: LogicalBatchSpec,
    attempt_seq: int,
    model_name: str,
    input_s3_key: str,
    instance_type: str = "ml.m5.large",
) -> dict[str, Any]:
    """Execute the O8 idempotent trigger check and start Batch Transform only if needed."""
    logical_batch_id = spec.compute_logical_batch_id()
    marker_key = success_marker_key(logical_batch_id)

    # 1. Check if Gold commit marker already exists
    if client.object_exists(bucket, marker_key):
        return {
            "status": "DUPLICATE_SKIPPED",
            "reason": "GOLD_SUCCESS_MARKER_EXISTS",
            "logical_batch_id": logical_batch_id,
            "marker_s3_uri": f"s3://{bucket}/{marker_key}",
            "transform_job_name": None,
        }

    # 2. Check if an active TransformJob for this logical batch is already running
    prefix = f"wm-bt-{logical_batch_id}-"
    active_jobs = client.list_active_transform_jobs(prefix)
    if active_jobs:
        return {
            "status": "DUPLICATE_SKIPPED",
            "reason": "ACTIVE_TRANSFORM_JOB_IN_PROGRESS",
            "logical_batch_id": logical_batch_id,
            "active_jobs": active_jobs,
            "transform_job_name": active_jobs[0],
        }

    # 3. Build and submit CreateTransformJob
    request = build_create_transform_job_request(
        spec=spec,
        attempt_seq=attempt_seq,
        model_name=model_name,
        bucket=bucket,
        input_s3_key=input_s3_key,
        instance_type=instance_type,
    )
    job_arn = client.create_transform_job(request)
    return {
        "status": "STARTED",
        "reason": "NEW_TRANSFORM_JOB_CREATED",
        "logical_batch_id": logical_batch_id,
        "transform_job_name": request["TransformJobName"],
        "transform_job_arn": job_arn,
        "staging_s3_uri": request["TransformOutput"]["S3OutputPath"],
    }


REQUIRED_PREDICTION_COLUMNS: tuple[str, ...] = (
    "order_id",
    "eta_p10_days",
    "eta_p50_days",
    "eta_p90_days",
    "prob_late",
    "risk_alert_flag",
)


def validate_and_publish_staging_predictions(
    *,
    client: BatchStorageAndTransformClient,
    bucket: str,
    spec: LogicalBatchSpec,
    attempt_seq: int,
    staging_key: str,
    expected_row_count: int,
    observed_row_count: int,
    observed_columns: list[str] | tuple[str, ...],
    null_order_id_count: int = 0,
    quantile_crossing_count: int = 0,
) -> dict[str, Any]:
    """Validate staging Batch Transform output and commit Gold marker only on full pass (O8).

    Partial, mismatched-schema, or quantile-crossed outputs are rejected and left isolated
    in staging/ so readers of vw_latest_batch_predictions never see incomplete batches.
    """
    logical_batch_id = spec.compute_logical_batch_id()
    missing_cols = [c for c in REQUIRED_PREDICTION_COLUMNS if c not in observed_columns]

    errors: list[str] = []
    if expected_row_count <= 0:
        errors.append(f"expected_row_count must be > 0, got {expected_row_count}")
    if observed_row_count != expected_row_count:
        errors.append(
            f"Row count mismatch: observed {observed_row_count} != expected {expected_row_count}"
        )
    if missing_cols:
        errors.append(f"Missing required prediction columns: {missing_cols}")
    if null_order_id_count > 0:
        errors.append(f"Found {null_order_id_count} null order_id values in predictions")
    if quantile_crossing_count > 0:
        errors.append(
            f"Found {quantile_crossing_count} unrepaired quantile crossings (p10 > p50 or p50 > p90)"
        )

    if errors:
        return {
            "status": "PUBLICATION_REJECTED",
            "logical_batch_id": logical_batch_id,
            "attempt_seq": attempt_seq,
            "errors": errors,
            "committed_gold_uri": None,
        }

    gold_key = gold_predictions_key(logical_batch_id)
    marker_key = success_marker_key(logical_batch_id)

    client.copy_staging_to_gold(bucket=bucket, staging_key=staging_key, gold_key=gold_key)

    manifest_payload = {
        "logical_batch_id": logical_batch_id,
        "attempt_seq": attempt_seq,
        "spec": asdict(spec),
        "expected_row_count": expected_row_count,
        "observed_row_count": observed_row_count,
        "columns": list(observed_columns),
        "gold_s3_uri": f"s3://{bucket}/{gold_key}",
        "committed_at_utc": datetime.now(UTC).isoformat(),
    }
    client.write_json(bucket=bucket, key=marker_key, payload=manifest_payload)

    return {
        "status": "PUBLISHED",
        "logical_batch_id": logical_batch_id,
        "attempt_seq": attempt_seq,
        "errors": [],
        "committed_gold_uri": f"s3://{bucket}/{gold_key}",
        "marker_s3_uri": f"s3://{bucket}/{marker_key}",
    }


class Boto3BatchClient:
    """Production AWS boto3 implementation of BatchStorageAndTransformClient for Lambda."""

    def __init__(self, region_name: str | None = None) -> None:
        import boto3

        region = region_name or os.environ.get("AWS_REGION", "ap-southeast-2")
        self._s3 = boto3.client("s3", region_name=region)
        self._sm = boto3.client("sagemaker", region_name=region)

    def object_exists(self, bucket: str, key: str) -> bool:
        from botocore.exceptions import ClientError

        try:
            self._s3.head_object(Bucket=bucket, Key=key)
            return True
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def read_json(self, bucket: str, key: str) -> dict[str, Any]:
        resp = self._s3.get_object(Bucket=bucket, Key=key)
        return json.loads(resp["Body"].read().decode("utf-8"))

    def write_json(self, bucket: str, key: str, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
        self._s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=body,
            ContentType="application/json",
        )

    def list_active_transform_jobs(self, name_prefix: str) -> list[str]:
        resp = self._sm.list_transform_jobs(
            NameContains=name_prefix,
            StatusEquals="InProgress",
            MaxResults=20,
        )
        return [
            job["TransformJobName"]
            for job in resp.get("TransformJobSummaries", [])
            if job["TransformJobName"].startswith(name_prefix)
        ]

    def create_transform_job(self, request: dict[str, Any]) -> str:
        resp = self._sm.create_transform_job(**request)
        return str(resp["TransformJobArn"])

    def copy_staging_to_gold(self, bucket: str, staging_key: str, gold_key: str) -> None:
        self._s3.copy_object(
            Bucket=bucket,
            CopySource={"Bucket": bucket, "Key": staging_key},
            Key=gold_key,
        )


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """AWS Lambda entrypoint invoked by EventBridge Scheduler (O8)."""
    del context
    bucket = event.get("bucket") or os.environ["WAYMARK_LAKE_BUCKET"]
    model_name = event.get("model_name") or os.environ.get("WAYMARK_MODEL_NAME", "waymark-m1-m2-prod")
    input_s3_key = event.get("input_s3_key") or os.environ.get(
        "WAYMARK_BATCH_INPUT_KEY", "gold/batch_inputs/current_scoring_input.csv"
    )
    instance_type = event.get("instance_type") or os.environ.get("WAYMARK_INSTANCE_TYPE", "ml.m5.large")
    attempt_seq = int(event.get("attempt_seq", 1))

    spec = LogicalBatchSpec(
        model_artifact_hash=str(event["model_artifact_hash"]),
        input_snapshot_hash=str(event["input_snapshot_hash"]),
        scoring_cutoff_date=str(event["scoring_cutoff_date"]),
        config_version=str(event.get("config_version", "wm0_v1")),
    )
    client = Boto3BatchClient()
    return trigger_batch_transform_idempotent(
        client=client,
        bucket=bucket,
        spec=spec,
        attempt_seq=attempt_seq,
        model_name=model_name,
        input_s3_key=input_s3_key,
        instance_type=instance_type,
    )
