"""Standard-library-only batch identities shared by Lambda and local completion."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any

from waymark.feature_schema import CATEGORICAL_FEATURES, MODEL_FEATURES


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


FEATURE_SCHEMA_SHA256 = digest(
    canonical_bytes({"features": MODEL_FEATURES, "categorical": CATEGORICAL_FEATURES})
)


def require_sha(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Expected a full SHA-256 identity")


def manifest_key(manifest: dict[str, Any]) -> str:
    return f"batch/manifests/{digest(canonical_bytes(manifest))}.json"


def transform_manifest(manifest: dict[str, Any]) -> bytes:
    return canonical_bytes(
        [{"prefix": f"s3://{manifest['bucket']}/"}, *[p["key"] for p in manifest["inputs"]]]
    )


def transform_manifest_key(manifest: dict[str, Any]) -> str:
    return f"batch/input-manifests/{digest(transform_manifest(manifest))}.json"


def validate_manifest(manifest: dict[str, Any]) -> None:
    """Reject unbound identities before a job or publication can be requested."""
    if manifest.get("schema") != "waymark.batch/v1":
        raise ValueError("Invalid batch manifest schema")
    if manifest.get("bucket") != "waymark-lake-pilot-149270211548-ap-southeast-2":
        raise ValueError("Batch must use the dedicated O7 lake bucket")
    date.fromisoformat(manifest["scoring_cutoff_date"])
    if manifest.get("config_version") != "wm0_v1":
        raise ValueError("Unexpected batch configuration")
    model = manifest["model"]
    require_sha(model["sha256"])
    require_sha(model["acceptance_sha256"])
    if model.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256:
        raise ValueError("Incompatible model feature schema")
    if model.get("key") != f"batch/models/{model['sha256']}/model.tar.gz":
        raise ValueError("Model must be content addressed")
    if not re.fullmatch(r"[a-zA-Z0-9](-*[a-zA-Z0-9]){0,62}", model["name"]):
        raise ValueError("Invalid SageMaker model name")
    if not re.fullmatch(
        r"149270211548\.dkr\.ecr\.ap-southeast-2\.amazonaws\.com/[a-z0-9._/-]+@sha256:[0-9a-f]{64}",
        model["image_uri"],
    ):
        raise ValueError("Use a digest-pinned serving image in the dedicated account/Region")
    from waymark.training_environment import validate_environment

    if (
        validate_environment(model["environment"], Path(__file__).with_name("remote_requirements.txt"))
        != model["image_uri"]
    ):
        raise ValueError("Serving environment receipt does not bind the selected image")
    inputs = manifest["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("Batch input inventory is empty")
    keys, identities = [], []
    for index, part in enumerate(inputs):
        require_sha(part["sha256"])
        if part["key"] != f"batch/inputs/{part['sha256']}/part-{index:05d}.csv":
            raise ValueError("Input must be content addressed with a unique part name")
        ids = part["order_ids"]
        if not ids or any(not isinstance(value, str) or not value for value in ids):
            raise ValueError("Missing input order identities")
        keys.append(part["key"])
        identities.extend(ids)
    if len(set(keys)) != len(keys) or len(set(identities)) != len(identities):
        raise ValueError("Duplicate input key/order identity")
    if manifest.get("input_snapshot_hash") != digest(canonical_bytes(inputs)):
        raise ValueError("Input inventory identity mismatch")


def validate_acceptance(body: bytes, model: dict[str, Any]) -> None:
    receipt = json.loads(body)
    if (
        digest(body) != model["acceptance_sha256"]
        or receipt.get("schema") != "waymark.model_acceptance/v1"
        or receipt.get("decision") != "accepted"
        or not receipt.get("review_reference")
        or receipt.get("model_sha256") != model["sha256"]
        or receipt.get("feature_schema_sha256") != FEATURE_SCHEMA_SHA256
        or receipt.get("image_uri") != model["image_uri"]
    ):
        raise ValueError("Missing or incompatible explicit model acceptance")
