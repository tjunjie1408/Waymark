"""Independent hash/inventory verification for a prepared lake snapshot (Glue Python 3.9)."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import PurePosixPath
from uuid import uuid4

import boto3


def verify_snapshot(s3, bucket: str, key: str, expected_sha256: str) -> dict:
    body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
    if hashlib.sha256(body).hexdigest() != expected_sha256:
        raise ValueError("Snapshot manifest hash mismatch")
    manifest = json.loads(body)
    identity = manifest.pop("snapshot_id")
    canonical = (json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    if hashlib.sha256(canonical).hexdigest() != identity or key != f"snapshots/{identity}/manifest.json":
        raise ValueError("Snapshot identity/path mismatch")
    objects = manifest["objects"]
    paths = [obj["path"] for obj in objects]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate snapshot paths")
    for layer, count in [("bronze", 9), ("silver", 9), ("gold", 6), ("manifests", 5), ("scripts", 1)]:
        if sum(path.startswith(layer + "/") for path in paths) != count:
            raise ValueError(f"Missing required {layer} inventory")
    if len(paths) != 30:
        raise ValueError("Unexpected snapshot inventory")
    for obj in objects:
        path = PurePosixPath(obj["path"])
        if path.is_absolute() or ".." in path.parts or chr(92) in str(path):
            raise ValueError("Unsafe snapshot path")
        response = s3.get_object(Bucket=bucket, Key=f"snapshots/{identity}/{path}")
        digest = hashlib.sha256()
        length = 0
        while True:
            chunk = response["Body"].read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            length += len(chunk)
        if length != obj["size_bytes"] or digest.hexdigest() != obj["sha256"]:
            raise ValueError(f"Snapshot object mismatch: {path}")
    return {"snapshot_id": identity, "verified_objects": len(paths)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--JOB_NAME", default="waymark-pilot-medallion-pythonshell")
    parser.add_argument("--WAYMARK_LAKE_BUCKET", required=True)
    parser.add_argument("--WAYMARK_GLUE_DB", required=True)
    parser.add_argument("--WAYMARK_SNAPSHOT_KEY", required=True)
    parser.add_argument("--WAYMARK_SNAPSHOT_SHA256", required=True)
    args, _ = parser.parse_known_args()
    s3 = boto3.client("s3")
    boto3.client("glue").get_database(Name=args.WAYMARK_GLUE_DB)
    result = verify_snapshot(
        s3, args.WAYMARK_LAKE_BUCKET, args.WAYMARK_SNAPSHOT_KEY, args.WAYMARK_SNAPSHOT_SHA256
    )
    now = datetime.now(timezone.utc).isoformat()  # noqa: UP017
    result.update(status="SUCCEEDED", job_name=args.JOB_NAME, execution_timestamp_utc=now)
    audit_key = f"snapshots/{result['snapshot_id']}/glue_audits/{uuid4().hex}.json"
    s3.put_object(
        Bucket=args.WAYMARK_LAKE_BUCKET,
        Key=audit_key,
        Body=json.dumps(result, indent=2).encode(),
        ContentType="application/json",
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
