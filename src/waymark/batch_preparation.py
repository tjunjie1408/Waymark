"""Prepare immutable batch inputs/model requests locally without fitting or uploading."""

from __future__ import annotations

import io
import json
import re
import tarfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from waymark.artifacts import write_immutable
from waymark.batch_contract import (
    FEATURE_SCHEMA_SHA256,
    canonical_bytes,
    digest,
    manifest_key,
    transform_manifest,
    transform_manifest_key,
    validate_acceptance,
    validate_manifest,
)
from waymark.feature_schema import CATEGORICAL_FEATURES, MODEL_FEATURES
from waymark.training import parse_training_artifacts


def prepare_batch(
    *,
    features: pd.DataFrame,
    model_archive: bytes,
    acceptance: bytes,
    environment: dict[str, Any],
    model_name: str,
    bucket: str,
    scoring_cutoff_date: str,
    execution_role_arn: str,
    output: Path,
    rows_per_part: int = 2000,
) -> dict[str, Any]:
    """Bind accepted model/schema/environment and explicit scoring order inventory."""
    if output.exists():
        raise FileExistsError("Select a new batch preparation directory")
    if rows_per_part < 1 or type(rows_per_part) is not int:
        raise ValueError("rows_per_part must be positive")
    if (
        features.empty
        or not set(["order_id", *MODEL_FEATURES]).issubset(features.columns)
        or features.order_id.isna().any()
        or features.order_id.duplicated().any()
        or any(not isinstance(value, str) or not value for value in features.order_id)
    ):
        raise ValueError("Invalid scoring feature schema/order identities")
    if not re.fullmatch(r"arn:aws:iam::149270211548:role/[A-Za-z0-9+=,.@_/-]+", execution_role_arn):
        raise ValueError("Use the dedicated account's SageMaker execution role")
    with tarfile.open(fileobj=io.BytesIO(model_archive), mode="r:gz") as archive:
        for member in archive.getmembers():
            if (
                not member.isfile()
                or Path(member.name).name != member.name
                or "\\" in member.name
                or member.name in {".", ".."}
            ):
                raise ValueError("Serving model archive must contain regular root files only")
        summary = json.loads(archive.extractfile("evaluation_summary.json").read())
        freeze = json.loads(archive.extractfile("development_freeze.json").read())
    parse_training_artifacts(model_archive, summary["identities"])
    if (
        freeze["model_features"] != MODEL_FEATURES
        or freeze["environment"]["versions"] != environment["versions"]
    ):
        raise ValueError("Model preprocessing/environment identity mismatch")
    model = {
        "name": model_name,
        "sha256": digest(model_archive),
        "key": f"batch/models/{digest(model_archive)}/model.tar.gz",
        "acceptance_sha256": digest(acceptance),
        "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
        "image_uri": environment["image_uri"],
        "environment": environment,
    }
    validate_acceptance(acceptance, model)
    frame = features[["order_id", *MODEL_FEATURES]].copy()
    for column in MODEL_FEATURES:
        if column in CATEGORICAL_FEATURES:
            frame[column] = frame[column].fillna("__MISSING__").astype(str)
        else:
            frame[column] = pd.to_numeric(frame[column], errors="raise").astype(float)
            if np.isinf(frame[column]).any():
                raise ValueError("Infinite scoring features")
    if any(
        "\n" in value or "\r" in value
        for column in ["order_id", *CATEGORICAL_FEATURES]
        for value in frame[column]
    ):
        raise ValueError("Batch CSV cannot contain embedded newline characters")
    payloads: dict[str, bytes] = {
        model["key"]: model_archive,
        f"batch/acceptances/{digest(acceptance)}.json": acceptance,
    }
    inputs = []
    for index, offset in enumerate(range(0, len(frame), rows_per_part)):
        part = frame.iloc[offset : offset + rows_per_part]
        body = part.to_csv(header=False, index=False, lineterminator="\n").encode()
        if any(len(line) > 6 * 1024 * 1024 for line in body.splitlines()):
            raise ValueError("A scoring record exceeds the transform payload limit")
        key = f"batch/inputs/{digest(body)}/part-{index:05d}.csv"
        inputs.append({"key": key, "sha256": digest(body), "order_ids": part.order_id.tolist()})
        payloads[key] = body
    manifest = {
        "schema": "waymark.batch/v1",
        "bucket": bucket,
        "model": model,
        "inputs": inputs,
        "input_snapshot_hash": digest(canonical_bytes(inputs)),
        "scoring_cutoff_date": scoring_cutoff_date,
        "config_version": "wm0_v1",
    }
    validate_manifest(manifest)
    payloads[transform_manifest_key(manifest)] = transform_manifest(manifest)
    payloads[manifest_key(manifest)] = canonical_bytes(manifest)
    request = {
        "ModelName": model_name,
        "ExecutionRoleArn": execution_role_arn,
        "EnableNetworkIsolation": True,
        "PrimaryContainer": {
            "Image": model["image_uri"],
            "ModelDataUrl": f"s3://{bucket}/{model['key']}",
            "Environment": {"WAYMARK_SERVING_IMAGE_URI": model["image_uri"]},
        },
        "Tags": [{"Key": "Project", "Value": "waymark"}],
    }
    for key, body in payloads.items():
        write_immutable(output / "objects" / key, body)
    write_immutable(output / "batch.json", canonical_bytes(manifest))
    write_immutable(output / "create_model.json", canonical_bytes(request))
    write_immutable(
        output / "bundle.json",
        canonical_bytes(
            {
                "schema": "waymark.batch_bundle/v1",
                "manifest_key": manifest_key(manifest),
                "objects": {key: digest(body) for key, body in payloads.items()},
                "request_sha256": digest(canonical_bytes(request)),
            }
        ),
    )
    return {
        "status": "PREPARED_LOCALLY_NOT_UPLOADED",
        "batch_manifest_key": manifest_key(manifest),
        "output": str(output),
        "objects": len(payloads),
    }


def load_bundle(directory: Path) -> tuple[dict[str, Any], dict[str, bytes], dict[str, Any]]:
    """Verify the local preparation inventory before any subsequent AWS mutation."""
    bundle = json.loads((directory / "bundle.json").read_bytes())
    manifest = json.loads((directory / "batch.json").read_bytes())
    validate_manifest(manifest)
    request = json.loads((directory / "create_model.json").read_bytes())
    if (
        bundle.get("schema") != "waymark.batch_bundle/v1"
        or bundle["manifest_key"] != manifest_key(manifest)
        or bundle["request_sha256"] != digest(canonical_bytes(request))
    ):
        raise ValueError("Batch bundle identity mismatch")
    expected = {
        manifest_key(manifest),
        transform_manifest_key(manifest),
        manifest["model"]["key"],
        f"batch/acceptances/{manifest['model']['acceptance_sha256']}.json",
        *[part["key"] for part in manifest["inputs"]],
    }
    if set(bundle["objects"]) != expected:
        raise ValueError("Batch bundle inventory mismatch")
    payloads = {key: (directory / "objects" / key).read_bytes() for key in expected}
    if any(digest(body) != bundle["objects"][key] for key, body in payloads.items()):
        raise ValueError("Batch bundle object hash mismatch")
    if (
        payloads[manifest_key(manifest)] != canonical_bytes(manifest)
        or payloads[transform_manifest_key(manifest)] != transform_manifest(manifest)
        or digest(payloads[manifest["model"]["key"]]) != manifest["model"]["sha256"]
        or any(digest(payloads[part["key"]]) != part["sha256"] for part in manifest["inputs"])
    ):
        raise ValueError("Bundle payload/manifest mismatch")
    model = manifest["model"]
    validate_acceptance(payloads[f"batch/acceptances/{model['acceptance_sha256']}.json"], model)
    if (
        not re.fullmatch(
            r"arn:aws:iam::149270211548:role/[A-Za-z0-9+=,.@_/-]+", request.get("ExecutionRoleArn", "")
        )
        or request.get("EnableNetworkIsolation") is not True
    ):
        raise ValueError("Prepared model role/network configuration mismatch")
    if request.get("ModelName") != model["name"] or request.get("PrimaryContainer") != {
        "Image": model["image_uri"],
        "ModelDataUrl": f"s3://{manifest['bucket']}/{model['key']}",
        "Environment": {"WAYMARK_SERVING_IMAGE_URI": model["image_uri"]},
    }:
        raise ValueError("Prepared CreateModel request binding mismatch")
    return manifest, payloads, request
