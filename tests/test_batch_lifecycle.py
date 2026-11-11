"""Bounded submit -> completed JSONLines -> committed reader journey, without fits."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from batch_fixtures import fixture, rows

from waymark.batch_delivery import finalize_batch, read_committed_batch, submit_batch


def complete(client, manifest):
    result = submit_batch(client, manifest, attempt_seq=1)
    client.job = {"TransformJobStatus": "Completed", **client.requests[-1]}
    key = result["staging_s3_uri"].split("/", 3)[3] + manifest["inputs"][0]["key"] + ".out"
    client.objects[key] = (json.dumps(rows()[0]) + "\n").encode()
    return result


def test_wire_journey_reads_only_the_committed_object():
    client, manifest = fixture()
    complete(client, manifest)
    result = finalize_batch(client, manifest, attempt_seq=1)
    client.objects["gold/predictions/orphan.parquet"] = b"not committed"
    frame, marker = read_committed_batch(client, manifest)
    assert result["status"] == "PUBLISHED"
    assert frame.order_id.tolist() == ["001"]
    assert marker["spec"]["input_snapshot_hash"] == manifest["input_snapshot_hash"]
    assert client.requests[0]["TransformInput"]["DataSource"]["S3DataSource"]["S3DataType"] == "ManifestFile"


def test_completed_describe_response_may_include_service_defaults():
    client, manifest = fixture()
    complete(client, manifest)
    client.job["TransformInput"]["CompressionType"] = "None"
    client.job["TransformOutput"]["KmsKeyId"] = "service-default"
    assert finalize_batch(client, manifest, attempt_seq=1)["status"] == "PUBLISHED"


def test_invalid_manifest_location_is_rejected_before_storage_access():
    from waymark.batch_delivery import load_batch_manifest

    class NoReads:
        def read_bytes(self, *args):
            raise AssertionError("An invalid location must not be read")

    with pytest.raises(ValueError):
        load_batch_manifest(NoReads(), "other-bucket", "batch/manifests/" + "a" * 64 + ".json")
    with pytest.raises(ValueError):
        load_batch_manifest(NoReads(), fixture()[1]["bucket"], "mutable/latest.json")


@pytest.mark.parametrize("status", ["InProgress", "Failed", "Stopped", "Stopping", "Unknown"])
def test_unsuccessful_or_uncertain_job_never_commits(status):
    client, manifest = fixture()
    complete(client, manifest)
    client.job["TransformJobStatus"] = status
    with pytest.raises(ValueError, match="Completed"):
        finalize_batch(client, manifest, attempt_seq=1)
    assert not any(key.startswith("gold/") for key in client.objects)


def test_input_and_model_tamper_prevent_submission():
    for kind in ["input", "model", "binding"]:
        client, manifest = fixture()
        if kind == "input":
            client.objects[manifest["inputs"][0]["key"]] += b"other\n"
        elif kind == "model":
            client.objects[manifest["model"]["key"]] = b"other model"
        else:
            client.model["PrimaryContainer"]["ModelDataUrl"] = "s3://other/model"
        with pytest.raises(ValueError):
            submit_batch(client, manifest, attempt_seq=1)
        assert not client.requests


def test_missing_or_extra_output_object_never_commits():
    for extra in [False, True]:
        client, manifest = fixture()
        result = complete(client, manifest)
        key = result["staging_s3_uri"].split("/", 3)[3] + manifest["inputs"][0]["key"] + ".out"
        if extra:
            client.objects[key + "-unknown"] = b"{}\n"
        else:
            del client.objects[key]
        with pytest.raises(ValueError, match="inventory"):
            finalize_batch(client, manifest, attempt_seq=1)


def test_job_with_other_input_and_corrupt_commit_are_rejected():
    client, manifest = fixture()
    complete(client, manifest)
    original = deepcopy(client.job)
    client.job["ModelName"] = "other"
    with pytest.raises(ValueError, match="identity"):
        finalize_batch(client, manifest, attempt_seq=1)
    client.job = original
    finalize_batch(client, manifest, attempt_seq=1)
    marker_key = next(key for key in client.objects if key.startswith("gold/prediction_manifests/"))
    marker = json.loads(client.objects[marker_key])
    client.objects[marker["gold_s3_uri"].split("/", 3)[3]] = b"corrupted"
    with pytest.raises(ValueError, match="hash"):
        read_committed_batch(client, manifest)
