"""Thin verified submission and explicit Python 3.12 post-job completion/consumption.

Only finalization/reading imports pandas. The Lambda submit path is standard library
plus boto3; it never waits for a job or attempts Parquet processing.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import asdict
from typing import Any

from waymark.batch_adapter import (
    REQUIRED_PREDICTION_COLUMNS,
    LogicalBatchSpec,
    build_create_transform_job_request,
    staging_output_prefix,
    success_marker_key,
    validate_and_publish_staging_predictions,
)
from waymark.batch_contract import (
    canonical_bytes,
    digest,
    manifest_key,
    transform_manifest,
    transform_manifest_key,
    validate_acceptance,
    validate_manifest,
)


def load_batch_manifest(client: Any, bucket: str, key: str) -> dict[str, Any]:
    if (
        bucket != "waymark-lake-pilot-149270211548-ap-southeast-2"
        or not isinstance(key, str)
        or not re.fullmatch(r"batch/manifests/[0-9a-f]{64}\.json", key)
    ):
        raise ValueError("Use a content-addressed manifest in the dedicated bucket")
    body = client.read_bytes(bucket, key)
    manifest = json.loads(body)
    validate_manifest(manifest)
    if key != manifest_key(manifest) or manifest["bucket"] != bucket:
        raise ValueError("Batch manifest content identity mismatch")
    return manifest


def batch_spec(manifest: dict[str, Any]) -> LogicalBatchSpec:
    validate_manifest(manifest)
    return LogicalBatchSpec(
        manifest["model"]["sha256"],
        manifest["input_snapshot_hash"],
        manifest["scoring_cutoff_date"],
        manifest["config_version"],
    )


def verify_bindings(client: Any, manifest: dict[str, Any]) -> None:
    validate_manifest(manifest)
    bucket, model = manifest["bucket"], manifest["model"]
    validate_acceptance(
        client.read_bytes(bucket, f"batch/acceptances/{model['acceptance_sha256']}.json"), model
    )
    for item in [model, *manifest["inputs"]]:
        if client.object_sha256(bucket, item["key"]) != item["sha256"]:
            raise ValueError("Remote model/input hash mismatch")
    if client.read_bytes(bucket, transform_manifest_key(manifest)) != transform_manifest(manifest):
        raise ValueError("Transform input manifest identity mismatch")
    bound = client.describe_model(model["name"])
    container = bound.get("PrimaryContainer", {})
    if (
        bound.get("ModelName") != model["name"]
        or bound.get("Containers")
        or container.get("Image") != model["image_uri"]
        or container.get("ModelDataUrl") != f"s3://{bucket}/{model['key']}"
        or container.get("Environment", {}).get("WAYMARK_SERVING_IMAGE_URI") != model["image_uri"]
    ):
        raise ValueError("SageMaker model binding identity mismatch")


def batch_request(manifest: dict[str, Any], attempt_seq: int) -> dict[str, Any]:
    if type(attempt_seq) is not int or attempt_seq < 1:
        raise ValueError("Attempt must be a positive integer")
    request = build_create_transform_job_request(
        spec=batch_spec(manifest),
        attempt_seq=attempt_seq,
        model_name=manifest["model"]["name"],
        bucket=manifest["bucket"],
        input_s3_key=transform_manifest_key(manifest),
    )
    request["TransformInput"]["DataSource"]["S3DataSource"]["S3DataType"] = "ManifestFile"
    return request


def submit_batch(client: Any, manifest: dict[str, Any], *, attempt_seq: int) -> dict[str, Any]:
    request = batch_request(manifest, attempt_seq)
    bucket, spec = manifest["bucket"], batch_spec(manifest)
    if client.object_exists(bucket, success_marker_key(spec.compute_logical_batch_id())):
        marker = client.read_json(bucket, success_marker_key(spec.compute_logical_batch_id()))
        verify_commit_identity(marker, manifest)
        return {"status": "DUPLICATE_SKIPPED", "committed_gold_uri": marker["gold_s3_uri"]}
    verify_bindings(client, manifest)
    if client.list_active_transform_jobs(f"wm-bt-{spec.compute_logical_batch_id()}-"):
        return {"status": "DUPLICATE_SKIPPED", "reason": "ACTIVE_TRANSFORM_JOB_IN_PROGRESS"}
    arn = client.create_transform_job(request)
    return {
        "status": "STARTED",
        "transform_job_arn": arn,
        "transform_job_name": request["TransformJobName"],
        "staging_s3_uri": request["TransformOutput"]["S3OutputPath"],
    }


def verify_commit_identity(marker: dict[str, Any], manifest: dict[str, Any]) -> None:
    spec = batch_spec(manifest)
    if (
        marker.get("schema") != "waymark.prediction_publication/v2"
        or marker.get("spec") != asdict(spec)
        or marker.get("batch_manifest_sha256") != digest(canonical_bytes(manifest))
        or marker.get("logical_batch_id") != spec.compute_logical_batch_id()
    ):
        raise ValueError("Committed batch identity mismatch")
    sha = marker.get("sha256", "")
    expected = f"s3://{manifest['bucket']}/gold/predictions/batch_id={spec.compute_logical_batch_id()}/objects/{sha}.parquet"
    if not re.fullmatch(r"[0-9a-f]{64}", sha) or marker.get("gold_s3_uri") != expected:
        raise ValueError("Committed object location mismatch")


def finalize_batch(client: Any, manifest: dict[str, Any], *, attempt_seq: int) -> dict[str, Any]:
    """Finalize one successful attempt; never poll, silently retry or publish partial files."""
    request = batch_request(manifest, attempt_seq)
    bucket, spec = manifest["bucket"], batch_spec(manifest)
    marker_key = success_marker_key(spec.compute_logical_batch_id())
    if client.object_exists(bucket, marker_key):
        _, marker = read_committed_batch(client, manifest)
        return {"status": "DUPLICATE_SKIPPED", "committed_gold_uri": marker["gold_s3_uri"]}
    verify_bindings(client, manifest)
    job = client.describe_transform_job(request["TransformJobName"])
    if job.get("TransformJobStatus") != "Completed":
        raise ValueError("Only a Completed transform job can publish")
    for field in ["TransformJobName", "ModelName", "TransformInput", "TransformOutput", "TransformResources"]:
        if not matches_request(job.get(field), request[field]):
            raise ValueError(f"Completed job identity mismatch: {field}")
    prefix = staging_output_prefix(spec.compute_logical_batch_id(), attempt_seq)
    expected_keys = [prefix + part["key"] + ".out" for part in manifest["inputs"]]
    if sorted(client.list_keys(bucket, prefix)) != sorted(expected_keys):
        raise ValueError("Completed output inventory differs from input inventory")
    bodies = []
    for part, key in zip(manifest["inputs"], expected_keys, strict=True):
        body = client.read_bytes(bucket, key)
        records = [json.loads(line) for line in body.decode().strip().splitlines()]
        ids = [record.get("order_id") for record in records]
        if len(ids) != len(part["order_ids"]) or set(ids) != set(part["order_ids"]):
            raise ValueError("Output order identities differ from the input part")
        bodies.append(body.strip())
    expected_ids = [value for part in manifest["inputs"] for value in part["order_ids"]]
    return validate_and_publish_staging_predictions(
        client=client,
        bucket=bucket,
        spec=spec,
        attempt_seq=attempt_seq,
        staging_key=expected_keys[0],
        staging_body=b"\n".join(bodies),
        wire_format="jsonlines",
        expected_order_ids=expected_ids,
        expected_row_count=len(expected_ids),
        observed_row_count=len(expected_ids),
        observed_columns=REQUIRED_PREDICTION_COLUMNS,
        batch_manifest_sha256=digest(canonical_bytes(manifest)),
    )


def matches_request(actual: Any, expected: Any) -> bool:
    """Describe responses can add service defaults; every requested field must still match."""
    if isinstance(expected, dict):
        return isinstance(actual, dict) and all(
            key in actual and matches_request(actual[key], value) for key, value in expected.items()
        )
    return actual == expected


def read_committed_batch(client: Any, manifest: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Fetch only the marker-bound object; candidate prefixes are never scanned."""
    import pandas as pd

    spec, bucket = batch_spec(manifest), manifest["bucket"]
    marker = client.read_json(bucket, success_marker_key(spec.compute_logical_batch_id()))
    verify_commit_identity(marker, manifest)
    key = marker["gold_s3_uri"].removeprefix(f"s3://{bucket}/")
    body = client.read_bytes(bucket, key)
    if digest(body) != marker["sha256"]:
        raise ValueError("Committed prediction hash mismatch")
    frame = pd.read_parquet(io.BytesIO(body))
    expected = [value for part in manifest["inputs"] for value in part["order_ids"]]
    if (
        set(frame.columns) != set(REQUIRED_PREDICTION_COLUMNS)
        or len(frame) != len(expected)
        or frame.order_id.duplicated().any()
        or set(frame.order_id) != set(expected)
        or marker["observed_row_count"] != len(expected)
    ):
        raise ValueError("Committed prediction schema/order identity mismatch")
    return frame, marker


def register_committed_table(client: Any, glue: Any, manifest: dict[str, Any], database: str) -> str:
    """Expose one verified commit through an isolated, immutable Parquet location."""
    frame, marker = read_committed_batch(client, manifest)
    bucket, batch_id, sha = manifest["bucket"], marker["logical_batch_id"], marker["sha256"]
    key = f"athena/committed/{batch_id}/{sha}/predictions.parquet"
    source_key = marker["gold_s3_uri"].removeprefix(f"s3://{bucket}/")
    body = client.read_bytes(bucket, source_key)
    if digest(body) != sha:
        raise ValueError("Committed source hash changed before exposure")
    client.put_bytes_immutable(bucket, key, body)
    prefix = key.removesuffix("predictions.parquet")
    if client.list_keys(bucket, prefix) != [key]:
        raise ValueError("Query location contains uncommitted extra objects")
    table = f"predictions_{batch_id}_{sha[:16]}"
    definition = {
        "Name": table,
        "TableType": "EXTERNAL_TABLE",
        "Parameters": {
            "classification": "parquet",
            "waymark.commit_sha256": sha,
            "waymark.manifest_sha256": marker["batch_manifest_sha256"],
        },
        "StorageDescriptor": {
            "Columns": [
                {
                    "Name": name,
                    "Type": "string"
                    if name == "order_id"
                    else ("bigint" if name == "risk_alert_flag" else "double"),
                }
                for name in frame.columns
            ],
            "Location": f"s3://{bucket}/{prefix}",
            "InputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetInputFormat",
            "OutputFormat": "org.apache.hadoop.hive.ql.io.parquet.MapredParquetOutputFormat",
            "SerdeInfo": {
                "SerializationLibrary": "org.apache.hadoop.hive.ql.io.parquet.serde.ParquetHiveSerDe"
            },
        },
    }
    try:
        glue.create_table(DatabaseName=database, TableInput=definition)
    except glue.exceptions.AlreadyExistsException:
        existing = glue.get_table(DatabaseName=database, Name=table)["Table"]
        storage, expected = existing["StorageDescriptor"], definition["StorageDescriptor"]
        if (
            any(
                storage.get(field) != expected[field]
                for field in ["Columns", "Location", "InputFormat", "OutputFormat"]
            )
            or storage.get("SerdeInfo", {}).get("SerializationLibrary")
            != expected["SerdeInfo"]["SerializationLibrary"]
            or any(
                existing.get("Parameters", {}).get(key) != value
                for key, value in definition["Parameters"].items()
            )
        ):
            raise ValueError("Existing committed table identity conflict") from None
    return table
