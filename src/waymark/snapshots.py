"""Content-addressed lake inventories and independently verified S3 transport."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
from botocore.exceptions import ClientError

from waymark.artifacts import canonical_json, sha256_file
from waymark.contracts import WM0Contract, load_wm0_contract

SILVER_TABLES = (
    "silver_orders",
    "silver_order_items",
    "silver_order_payments",
    "silver_order_reviews",
    "silver_products",
    "silver_sellers",
    "silver_customers",
    "silver_geolocation_zip",
    "silver_category_translations",
)
GOLD_TABLES = (
    "fact_order_fulfillment",
    "dim_seller",
    "dim_customer",
    "dim_product",
    "dim_geolocation_zip",
    "order_features_and_splits",
)
MANIFESTS = (
    "wm0_contract.yaml",
    "t1_discriminating_fixture.json",
    "t3_source_manifest.json",
    "t4_gold_reconciliation.json",
    "t5_wm2_split_manifest.json",
)


def verify_account(sts_client: Any, region: str, contract: WM0Contract) -> str:
    expected = contract.o7_aws_and_cost
    account = sts_client.get_caller_identity()["Account"]
    if account != expected.account_id or region != expected.region:
        raise ValueError(
            f"AWS account/Region mismatch: {account}/{region}; expected "
            f"{expected.account_id}/{expected.region}"
        )
    return account


def build_lake_snapshot(repo_root: Path) -> dict[str, Any]:
    contract = load_wm0_contract(repo_root / "configs/wm0_contract.yaml")
    paths = [
        (f"bronze/{name}/{name}.parquet", f"data/bronze/{name}.parquet")
        for name in contract.o1_source["expected_tables"]
    ]
    for layer, tables in [("silver", SILVER_TABLES), ("gold", GOLD_TABLES)]:
        paths.extend((f"{layer}/{name}/{name}.parquet", f"data/{layer}/{name}.parquet") for name in tables)
    paths.extend((f"manifests/{name}", f"configs/{name}") for name in MANIFESTS)
    paths.append(("scripts/glue_etl_entry.py", "scripts/glue_etl_entry.py"))
    missing = [local for _, local in paths if not (repo_root / local).is_file()]
    if missing:
        raise ValueError(f"Incomplete lake snapshot, missing required inputs: {missing}")
    t3 = json.loads((repo_root / "configs/t3_source_manifest.json").read_text(encoding="utf-8"))
    for spec in t3["bronze_manifest"]["tables"].values():
        raw = repo_root / "data/raw/olist" / spec["filename"]
        if not raw.is_file() or sha256_file(raw) != spec["sha256"]:
            raise ValueError(f"Source hash mismatch or missing frozen source: {raw}")
    t5 = json.loads((repo_root / "configs/t5_wm2_split_manifest.json").read_text(encoding="utf-8"))
    if sha256_file(repo_root / "data/gold/order_features_and_splits.parquet") != t5["feature_table_sha256"]:
        raise ValueError("Feature table hash does not match the frozen T5 manifest")
    objects = []
    for relative, local in sorted(paths):
        path = repo_root / local
        item = {
            "path": relative,
            "local_path": local,
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        if path.suffix == ".parquet":
            item["row_count"] = pq.read_metadata(path).num_rows
        objects.append(item)
    inventory = {
        "schema": "waymark.lake_snapshot/v1",
        "contract_id": contract.contract_id,
        "objects": objects,
    }
    return {**inventory, "snapshot_id": hashlib.sha256(canonical_json(inventory)).hexdigest()}


def put_s3_immutable(client: Any, bucket: str, key: str, body: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(body).hexdigest()
    try:
        response = client.put_object(
            Bucket=bucket, Key=key, Body=body, IfNoneMatch="*", Metadata={"sha256": digest}
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"PreconditionFailed", "412"}:
            raise
        response = client.get_object(Bucket=bucket, Key=key)
        if hashlib.sha256(response["Body"].read()).hexdigest() != digest:
            raise ValueError(f"Immutable S3 object conflict: s3://{bucket}/{key}") from exc
    return {"key": key, "sha256": digest, "version_id": response.get("VersionId")}


def upload_snapshot(client: Any, bucket: str, root: Path, snapshot: dict[str, Any]) -> dict[str, Any]:
    prefix = f"snapshots/{snapshot['snapshot_id']}/"
    receipts = []
    for obj in snapshot["objects"]:
        body = (root / obj["local_path"]).read_bytes()
        if hashlib.sha256(body).hexdigest() != obj["sha256"]:
            raise ValueError(f"Snapshot input changed before upload: {obj['local_path']}")
        receipts.append(put_s3_immutable(client, bucket, prefix + obj["path"], body))
    receipt = put_s3_immutable(client, bucket, prefix + "manifest.json", canonical_json(snapshot))
    return {"prefix": prefix, "manifest": receipt, "objects": receipts}


def verify_remote_snapshot(client: Any, bucket: str, snapshot: dict[str, Any]) -> None:
    prefix = f"snapshots/{snapshot['snapshot_id']}/"
    remote = client.get_object(Bucket=bucket, Key=prefix + "manifest.json")["Body"].read()
    if remote != canonical_json(snapshot):
        raise ValueError("Remote snapshot manifest mismatch")
    for obj in snapshot["objects"]:
        body = client.get_object(Bucket=bucket, Key=prefix + obj["path"])["Body"].read()
        if len(body) != obj["size_bytes"] or hashlib.sha256(body).hexdigest() != obj["sha256"]:
            raise ValueError(f"Remote snapshot object mismatch: {obj['path']}")
