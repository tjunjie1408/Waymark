from __future__ import annotations

import json
from pathlib import Path

from waymark.batch_adapter import REQUIRED_PREDICTION_COLUMNS
from waymark.batch_contract import (
    FEATURE_SCHEMA_SHA256,
    canonical_bytes,
    digest,
    manifest_key,
    transform_manifest,
    transform_manifest_key,
)
from waymark.training_environment import requirement_versions


class Storage:
    def __init__(self):
        self.objects = {}

    def object_exists(self, bucket, key):
        return key in self.objects

    def read_bytes(self, bucket, key):
        return self.objects[key]

    def read_json(self, bucket, key):
        return json.loads(self.read_bytes(bucket, key))

    def put_bytes_immutable(self, bucket, key, body):
        if key in self.objects and self.objects[key] != body:
            raise ValueError("Immutable conflict")
        self.objects[key] = body

    def write_json_if_absent(self, bucket, key, value):
        if key in self.objects:
            return False
        self.objects[key] = json.dumps(value).encode()
        return True


def rows():
    return [dict(zip(REQUIRED_PREDICTION_COLUMNS, ["001", 1.0, 2.0, 3.0, 0.25, 0], strict=True))]


BUCKET = "waymark-lake-pilot-149270211548-ap-southeast-2"


class Client(Storage):
    def __init__(self):
        super().__init__()
        self.requests = []
        self.job = {}
        self.model = {}

    def object_sha256(self, bucket, key):
        return digest(self.read_bytes(bucket, key))

    def describe_model(self, name):
        return self.model

    def describe_transform_job(self, name):
        return self.job

    def list_active_transform_jobs(self, prefix):
        return []

    def list_keys(self, bucket, prefix):
        return sorted(key for key in self.objects if key.startswith(prefix))

    def create_transform_job(self, request):
        self.requests.append(request)
        return "arn:test:job"


def fixture():
    client = Client()
    model = b"accepted-own-model"
    data = b"001,0,BR\n"
    part = {"sha256": digest(data), "order_ids": ["001"]}
    part["key"] = f"batch/inputs/{part['sha256']}/part-00000.csv"
    manifest = {
        "schema": "waymark.batch/v1",
        "bucket": BUCKET,
        "scoring_cutoff_date": "2018-08-29",
        "config_version": "wm0_v1",
        "inputs": [part],
        "input_snapshot_hash": digest(canonical_bytes([part])),
        "model": {
            "name": "accepted-v1",
            "sha256": digest(model),
            "key": f"batch/models/{digest(model)}/model.tar.gz",
            "acceptance_sha256": "a" * 64,
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "image_uri": "149270211548.dkr.ecr.ap-southeast-2.amazonaws.com/waymark@sha256:" + "b" * 64,
        },
    }
    receipt = {
        "schema": "waymark.environment/v1",
        "status": "PASS",
        "python": "3.12.9",
        "image_uri": manifest["model"]["image_uri"],
        "versions": requirement_versions(Path("src/waymark/remote_requirements.txt")),
        "requirements_sha256": digest(Path("src/waymark/remote_requirements.txt").read_bytes()),
        "check": "installation_and_imports_only_no_fit",
    }
    manifest["model"]["environment"] = receipt
    acceptance = canonical_bytes(
        {
            "schema": "waymark.model_acceptance/v1",
            "decision": "accepted",
            "review_reference": "synthetic-test-only",
            "model_sha256": digest(model),
            "feature_schema_sha256": FEATURE_SCHEMA_SHA256,
            "image_uri": manifest["model"]["image_uri"],
        }
    )
    manifest["model"]["acceptance_sha256"] = digest(acceptance)
    client.objects[f"batch/acceptances/{digest(acceptance)}.json"] = acceptance
    client.objects[part["key"]] = data
    client.objects[manifest["model"]["key"]] = model
    client.objects[transform_manifest_key(manifest)] = transform_manifest(manifest)
    client.objects[manifest_key(manifest)] = canonical_bytes(manifest)
    client.model = {
        "ModelName": "accepted-v1",
        "PrimaryContainer": {
            "Image": manifest["model"]["image_uri"],
            "ModelDataUrl": f"s3://{BUCKET}/{manifest['model']['key']}",
            "Environment": {"WAYMARK_SERVING_IMAGE_URI": manifest["model"]["image_uri"]},
        },
    }
    return client, manifest
