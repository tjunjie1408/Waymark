"""Unit tests for C1 cloud data slice, Athena DDL, and Gate WM1 reconciliation."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from waymark.cloud_slice import pyarrow_schema_to_athena_ddl

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_type_map_and_ddl_generation(tmp_path: Path) -> None:
    """Verify pyarrow schema converter emits compliant Athena DDL without backticks."""
    silver_orders_path = tmp_path / "silver_orders.parquet"
    pq.write_table(
        pa.table({"order_id": ["synthetic"], "is_eligible_mature": [1], "delivery_duration_days": [2.0]}),
        silver_orders_path,
    )

    ddl = pyarrow_schema_to_athena_ddl(
        table_name="silver_orders",
        parquet_path=silver_orders_path,
        s3_location="s3://test-bucket/silver/silver_orders/",
    )

    assert "CREATE EXTERNAL TABLE IF NOT EXISTS silver_orders (" in ddl
    assert "STORED AS PARQUET" in ddl
    assert "LOCATION 's3://test-bucket/silver/silver_orders/'" in ddl
    assert "`" not in ddl, "Trino engine requires unquoted or double-quoted identifiers, zero backticks"
    assert "order_id STRING" in ddl
    assert "is_eligible_mature BIGINT" in ddl
    assert "delivery_duration_days DOUBLE" in ddl


def test_c1_cloud_reconciliation_manifest_and_gate_wm1() -> None:
    """Verify C1 cloud reconciliation manifest matches local T4 and T5 ground truth."""
    c1_manifest_path = REPO_ROOT / "configs" / "c1_cloud_reconciliation.json"
    assert c1_manifest_path.exists(), "configs/c1_cloud_reconciliation.json must exist"

    c1 = json.loads(c1_manifest_path.read_text(encoding="utf-8"))
    assert c1["status"] == "OK"
    assert c1["stage"] == "C1"
    assert c1["gate"] == "WM1_VERIFIED"
    assert c1["aws_account_id"] == "149270211548"
    assert c1["aws_region"] == "ap-southeast-2"
    assert c1["glue_job_execution"]["status"] == "SUCCEEDED"

    recon = c1["athena_reconciliation"]
    assert recon["gate_wm1_status"] == "PASSED"

    # Row count invariants
    counts = recon["table_row_counts"]
    assert counts["silver_orders"] == 99441
    assert counts["fact_order_fulfillment"] == 99441
    assert counts["dim_customer"] == 99441
    assert counts["dim_seller"] == 3095
    assert counts["dim_product"] == 32951
    assert counts["dim_geolocation_zip"] == 19010
    assert counts["order_features_and_splits"] == 99441

    # Monetary and items reconciliation against local T4
    money = recon["athena_monetary_summary"]
    assert money["total_orders"] == 99441
    assert money["sum_items"] == 112650
    assert abs(money["sum_price_brl"] - 13591643.70) < 0.1
    assert abs(money["sum_freight_brl"] - 2251909.54) < 0.1
    assert abs(money["sum_payment_brl"] - 16008872.12) < 0.1

    # Splits reconciliation against local T5 (Gate WM2)
    splits = recon["athena_splits_summary"]
    assert splits["train"] == 64404
    assert splits["val_calibration"] == 12551
    assert splits["holdout_test"] == 14460
    assert splits["EXCL_IMMATURE_AT_TRAIN_BOUNDARY"] == 3072
    assert splits["EXCL_IMMATURE_AT_VAL_BOUNDARY"] == 1908
    assert splits["EXCLUDED"] == 3046

    # Verify Athena cost protection (< 10 MB scanned, < 1 GB limit)
    assert recon["total_athena_bytes_scanned"] < 10 * 1024 * 1024
    assert recon["estimated_athena_cost_usd"] < 0.001
