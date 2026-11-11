"""Cloud data slice orchestration and Athena reconciliation for Gate WM1.

Handles S3 lake layout sync, Glue Python Shell execution, Athena DDL registration,
analytical query execution, and reconciliation against local T3-T5 manifests.
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Any

import boto3
import pyarrow.parquet as pq

from waymark.artifacts import canonical_json, write_immutable
from waymark.contracts import load_wm0_contract
from waymark.costs import athena_billable_bytes, usage_estimate
from waymark.snapshots import build_lake_snapshot, upload_snapshot, verify_account, verify_remote_snapshot

logger = logging.getLogger("waymark.cloud_slice")

TYPE_MAP = {
    "bool": "BOOLEAN",
    "timestamp[us]": "TIMESTAMP",
    "int64": "BIGINT",
    "string": "STRING",
    "large_string": "STRING",
    "double": "DOUBLE",
}


def pyarrow_schema_to_athena_ddl(table_name: str, parquet_path: Path, s3_location: str) -> str:
    """Generate Athena CREATE EXTERNAL TABLE DDL from Parquet schema."""
    if not re.fullmatch(r"[a-z_][a-z0-9_]*", table_name):
        raise ValueError("Unsafe table identifier")
    if not s3_location.startswith("s3://") or "'" in s3_location:
        raise ValueError("Invalid S3 location")
    schema = pq.read_schema(parquet_path)
    col_defs = []
    for field in schema:
        t_str = str(field.type)
        if t_str not in TYPE_MAP or not re.fullmatch(r"[a-z_][a-z0-9_]*", field.name):
            raise ValueError(f"Unsupported Athena field: {field}")
        sql_type = TYPE_MAP[t_str]
        col_defs.append(f"  {field.name} {sql_type}")
    cols_joined = ",\n".join(col_defs)
    return (
        f"CREATE EXTERNAL TABLE IF NOT EXISTS {table_name} (\n"
        f"{cols_joined}\n"
        f")\n"
        f"STORED AS PARQUET\n"
        f"LOCATION '{s3_location}'\n"
        f"TBLPROPERTIES ('parquet.compress'='SNAPPY')"
    )


class AthenaQueryRunner:
    """Helper for executing Athena queries synchronously and retrieving results."""

    def __init__(self, athena_client: Any, workgroup: str, database: str) -> None:
        self.client = athena_client
        self.workgroup = workgroup
        self.database = database
        self.total_bytes_scanned = 0
        self.query_statistics: list[dict[str, Any]] = []
        self.timeout_seconds = 300.0

    def run_query(self, query: str) -> tuple[list[dict[str, Any]], int]:
        """Execute query, wait for completion, and return rows as dictionaries."""
        response = self.client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": self.database},
            WorkGroup=self.workgroup,
        )
        query_id = response["QueryExecutionId"]

        deadline = time.monotonic() + self.timeout_seconds
        while True:
            status_resp = self.client.get_query_execution(QueryExecutionId=query_id)
            state = status_resp["QueryExecution"]["Status"]["State"]
            if state == "SUCCEEDED":
                stats = status_resp["QueryExecution"]["Statistics"]
                bytes_scanned = stats.get("DataScannedInBytes", 0)
                self.total_bytes_scanned += bytes_scanned
                statement_type = status_resp["QueryExecution"].get("StatementType", "DML")
                self.query_statistics.append(
                    {
                        "query_id": query_id,
                        "statement_type": statement_type,
                        "status": state,
                        "scanned_bytes": bytes_scanned,
                        "billable_bytes": athena_billable_bytes(bytes_scanned, statement_type),
                    }
                )
                break
            elif state in ("FAILED", "CANCELLED"):
                reason = status_resp["QueryExecution"]["Status"].get("StateChangeReason", "Unknown error")
                msg = f"Athena query {query_id} failed with state {state}: {reason}\nQuery: {query}"
                raise RuntimeError(msg)
            if state not in {"QUEUED", "RUNNING"}:
                raise RuntimeError(f"Unknown Athena query state {state}: {query_id}")
            if time.monotonic() >= deadline:
                self.client.stop_query_execution(QueryExecutionId=query_id)
                raise TimeoutError(f"Athena polling deadline exceeded: {query_id}")
            time.sleep(1)

        # Fetch results
        results_paginator = self.client.get_paginator("get_query_results")
        rows: list[dict[str, Any]] = []
        column_names: list[str] = []

        for page in results_paginator.paginate(QueryExecutionId=query_id):
            result_set = page["ResultSet"]
            rows_data = result_set["Rows"]
            if not rows_data:
                continue

            start_idx = 0
            if not column_names:
                column_names = [col.get("VarCharValue", "") for col in rows_data[0]["Data"]]
                start_idx = 1

            for row in rows_data[start_idx:]:
                row_vals = [col.get("VarCharValue") for col in row["Data"]]
                row_dict = dict(zip(column_names, row_vals, strict=False))
                rows.append(row_dict)

        return rows, bytes_scanned


def sync_local_lake_to_s3(s3_client: Any, bucket_name: str, repo_root: Path) -> dict[str, Any]:
    """Transfer an exact content-addressed snapshot, never replace the legacy lake."""
    snapshot = build_lake_snapshot(repo_root)
    transport = upload_snapshot(s3_client, bucket_name, repo_root, snapshot)
    verify_remote_snapshot(s3_client, bucket_name, snapshot)
    return {"snapshot": snapshot, "transport": transport}


def run_glue_job_verification(
    glue_client: Any,
    job_name: str,
    *,
    arguments: dict[str, str] | None = None,
    timeout_seconds: float = 1000.0,
    dpu_hourly_rate: float | None = None,
    rate_source: str = "",
) -> dict[str, Any]:
    """Run the snapshot verifier, bounded by a client deadline and remote job timeout."""
    response = glue_client.start_job_run(JobName=job_name, Arguments=arguments or {})
    run_id = response["JobRunId"]
    deadline = time.monotonic() + timeout_seconds
    while True:
        run = glue_client.get_job_run(JobName=job_name, RunId=run_id)["JobRun"]
        state = run["JobRunState"]
        if state == "SUCCEEDED":
            duration = run.get("ExecutionTime")
            capacity = run.get("MaxCapacity")
            if duration is None or capacity is None or duration < 0 or capacity <= 0:
                raise ValueError(f"Missing/invalid Glue usage for {run_id}")
            cost = usage_estimate(
                max(60, duration) * capacity / 3600.0,
                "DPU_hours_60_second_minimum",
                dpu_hourly_rate,
                rate_source,
            )
            return {
                "job_name": job_name,
                "job_run_id": run_id,
                "status": state,
                "execution_time_seconds": duration,
                "allocated_dpu": capacity,
                "cost": cost,
                "estimated_cost_usd": cost["estimated_cost_usd"],
            }
        if state in {"FAILED", "STOPPED", "TIMEOUT", "ERROR", "EXPIRED"}:
            raise RuntimeError(f"Glue {job_name}/{run_id}: {state}: {run.get('ErrorMessage', '')}")
        if state not in {"STARTING", "RUNNING", "STOPPING", "WAITING"}:
            raise RuntimeError(f"Unknown Glue state {state}: {run_id}")
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Glue polling deadline exceeded; inspect remote run {job_name}/{run_id}")
        time.sleep(5)


def register_catalog_tables(
    runner: AthenaQueryRunner,
    bucket_name: str,
    repo_root: Path,
    snapshot_prefix: str,
) -> list[str]:
    """Register all Silver and Gold external Parquet tables in Athena."""
    tables_registered: list[str] = []

    # Drop and recreate tables to ensure idempotent fresh metadata
    silver_dir = repo_root / "data" / "silver"
    for p in sorted(silver_dir.glob("*.parquet")):
        table_name = p.stem
        s3_loc = f"s3://{bucket_name}/{snapshot_prefix}silver/{table_name}/"
        ddl = pyarrow_schema_to_athena_ddl(table_name, p, s3_loc)
        runner.run_query(ddl)
        tables_registered.append(table_name)

    gold_dir = repo_root / "data" / "gold"
    for p in sorted(gold_dir.glob("*.parquet")):
        table_name = p.stem
        s3_loc = f"s3://{bucket_name}/{snapshot_prefix}gold/{table_name}/"
        ddl = pyarrow_schema_to_athena_ddl(table_name, p, s3_loc)
        runner.run_query(ddl)
        tables_registered.append(table_name)

    return tables_registered


def execute_athena_reconciliation(
    runner: AthenaQueryRunner,
    t4_summary: dict[str, Any],
    t5_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Execute analytical Athena queries and reconcile against local T4/T5 outputs."""
    reconciliation_results: dict[str, Any] = {}

    # 1. Table row counts
    count_checks = {}
    for table_name in [
        "silver_orders",
        "fact_order_fulfillment",
        "dim_customer",
        "dim_seller",
        "dim_product",
        "dim_geolocation_zip",
        "order_features_and_splits",
    ]:
        rows, _ = runner.run_query(f"SELECT count(*) as row_count FROM {table_name}")
        count_checks[table_name] = int(rows[0]["row_count"])
    reconciliation_results["table_row_counts"] = count_checks

    # 2. Monetary and items reconciliation on fact_order_fulfillment
    money_query = """
    SELECT
      count(*) as total_orders,
      sum(item_count) as sum_items,
      round(sum(total_item_price_brl), 2) as sum_price,
      round(sum(total_freight_brl), 2) as sum_freight,
      round(sum(total_payment_value_brl), 2) as sum_payment
    FROM fact_order_fulfillment
    """
    money_rows, _ = runner.run_query(money_query)
    money_res = money_rows[0]
    reconciliation_results["athena_monetary_summary"] = {
        "total_orders": int(money_res["total_orders"]),
        "sum_items": int(money_res["sum_items"]),
        "sum_price_brl": float(money_res["sum_price"]),
        "sum_freight_brl": float(money_res["sum_freight"]),
        "sum_payment_brl": float(money_res["sum_payment"]),
    }

    # 3. Splits verification on order_features_and_splits
    splits_query = """
    SELECT
      split_role,
      count(*) as split_orders
    FROM order_features_and_splits
    GROUP BY split_role
    ORDER BY split_role
    """
    split_rows, _ = runner.run_query(splits_query)
    split_summary = {r["split_role"]: int(r["split_orders"]) for r in split_rows}
    reconciliation_results["athena_splits_summary"] = split_summary

    # 4. Top 5 corridors analytical parity query
    corridor_query = """
    SELECT
      state_corridor,
      count(*) as eligible_orders,
      round(avg(y_days), 4) as mean_eta_days,
      round(avg(primary_seller_haversine_km), 2) as mean_distance_km
    FROM fact_order_fulfillment
    WHERE is_eligible_mature = 1
    GROUP BY state_corridor
    ORDER BY eligible_orders DESC
    LIMIT 5
    """
    corridor_rows, _ = runner.run_query(corridor_query)
    reconciliation_results["athena_top_5_corridors"] = [
        {
            "state_corridor": r["state_corridor"],
            "eligible_orders": int(r["eligible_orders"]),
            "mean_eta_days": float(r["mean_eta_days"]),
            "mean_distance_km": float(r["mean_distance_km"]),
        }
        for r in corridor_rows
    ]

    gold = t4_summary["gold_reconciliation"]
    expected_counts = {
        "silver_orders": gold["grain_check"]["silver_orders_count"],
        "fact_order_fulfillment": gold["grain_check"]["gold_fact_rows"],
        "order_features_and_splits": t5_manifest["split_summary"]["counts"]["total_orders"],
        **{name.removesuffix("_rows"): count for name, count in gold["dimensions"].items()},
    }
    for name, expected in expected_counts.items():
        if count_checks[name] != expected:
            raise ValueError(f"row count mismatch for {name}: {count_checks[name]} != {expected}")
    expected_money = gold["monetary_and_count_reconciliation"]
    observed_money = reconciliation_results["athena_monetary_summary"]
    for observed, expected in [
        ("sum_price_brl", "gold_sum_item_price_brl"),
        ("sum_freight_brl", "gold_sum_freight_brl"),
        ("sum_payment_brl", "gold_sum_payment_value_brl"),
        ("sum_items", "gold_sum_item_count"),
    ]:
        if not math.isclose(observed_money[observed], expected_money[expected], rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"{observed} reconciliation mismatch")
    if observed_money["total_orders"] != expected_counts["fact_order_fulfillment"]:
        raise ValueError("Monetary query order count mismatch")
    counts = t5_manifest["split_summary"]["counts"]
    expected_splits = {
        "train": counts["train_mature_retained"],
        "val_calibration": counts["val_calibration_mature_retained"],
        "holdout_test": counts["holdout_test_orders"],
        "EXCL_IMMATURE_AT_TRAIN_BOUNDARY": counts["train_purged_immature_at_boundary"],
        "EXCL_IMMATURE_AT_VAL_BOUNDARY": counts["val_purged_immature_at_boundary"],
        "EXCLUDED": counts["excluded_global_orders"],
    }
    if split_summary != {k: v for k, v in expected_splits.items() if v}:
        raise ValueError("Split population mismatch")
    reference = t4_summary["duckdb_sql_parity"]["top_5_corridors_reference_query"]
    expected_corridors = {c["state_corridor"]: c for c in reference}
    if {c["state_corridor"] for c in corridor_rows} != set(expected_corridors):
        raise ValueError("Corridor inventory mismatch")
    for observed in reconciliation_results["athena_top_5_corridors"]:
        expected = expected_corridors[observed["state_corridor"]]
        for key in ("eligible_orders", "mean_eta_days", "mean_distance_km"):
            if not math.isclose(observed[key], expected[key], rel_tol=0.0, abs_tol=1e-6):
                raise ValueError(f"Corridor {key} mismatch")
    reconciliation_results["gate_wm1_status"] = "AGGREGATES_RECONCILED"
    reconciliation_results["total_athena_bytes_scanned"] = runner.total_bytes_scanned
    reconciliation_results["query_statistics"] = runner.query_statistics
    reconciliation_results["athena_cost"] = usage_estimate(
        sum(q["billable_bytes"] for q in runner.query_statistics) / 1e12, "billable_TB", None, ""
    )

    return reconciliation_results


def run_c1_orchestration(
    profile_name: str = "waymark",
    region_name: str = "ap-southeast-2",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Verify one bound snapshot; preserve historical catalogs and acceptance receipts."""
    root = repo_root or Path(__file__).resolve().parents[2]
    snapshot = build_lake_snapshot(root)
    contract = load_wm0_contract(root / "configs/wm0_contract.yaml")
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    account = verify_account(session.client("sts"), region_name, contract)
    bucket = f"waymark-lake-pilot-{account}-{region_name}"
    prefix = f"snapshots/{snapshot['snapshot_id']}/"
    glue = session.client("glue")
    job_name = "waymark-pilot-medallion-pythonshell"
    expected_script = f"s3://{bucket}/{prefix}scripts/glue_etl_entry.py"
    if glue.get_job(JobName=job_name)["Job"]["Command"]["ScriptLocation"] != expected_script:
        raise ValueError("Glue job must be configured to the prepared snapshot script URI before C1")
    s3 = session.client("s3")
    transport = upload_snapshot(s3, bucket, root, snapshot)
    verify_remote_snapshot(s3, bucket, snapshot)
    glue_summary = run_glue_job_verification(
        glue,
        job_name,
        arguments={
            "--WAYMARK_SNAPSHOT_KEY": prefix + "manifest.json",
            "--WAYMARK_SNAPSHOT_SHA256": transport["manifest"]["sha256"],
        },
    )
    database = f"waymark_pilot_{snapshot['snapshot_id'][:12]}"
    runner = AthenaQueryRunner(session.client("athena"), "waymark_pilot_wg", "default")
    runner.run_query(f"CREATE DATABASE IF NOT EXISTS {database}")
    runner.database = database
    tables = register_catalog_tables(runner, bucket, root, prefix)
    if len(tables) != 15:
        raise ValueError("Incomplete catalog registration")
    import pandas as pd

    keys_by_table = {
        "silver_orders": ["order_id"],
        "silver_order_items": ["order_id", "order_item_id"],
        "silver_order_payments": ["order_id", "payment_sequential"],
        "silver_order_reviews": ["review_id", "order_id"],
        "silver_products": ["product_id"],
        "silver_sellers": ["seller_id"],
        "silver_customers": ["customer_id"],
        "silver_geolocation_zip": ["geolocation_zip_code_prefix"],
        "silver_category_translations": ["product_category_name"],
        "fact_order_fulfillment": ["order_id"],
        "order_features_and_splits": ["order_id"],
        "dim_seller": ["seller_id"],
        "dim_customer": ["customer_id"],
        "dim_product": ["product_id"],
        "dim_geolocation_zip": ["geolocation_zip_code_prefix"],
    }
    for table in tables:
        layer = "silver" if table.startswith("silver_") else "gold"
        expected_location = f"s3://{bucket}/{prefix}{layer}/{table}/"
        metadata = glue.get_table(DatabaseName=database, Name=table)["Table"]
        if metadata["StorageDescriptor"]["Location"] != expected_location:
            raise ValueError(f"Catalog snapshot location mismatch: {table}")
        keys = keys_by_table[table]
        local_keys = pd.read_parquet(root / f"data/{layer}/{table}.parquet", columns=keys)
        remote_keys, _ = runner.run_query(f"SELECT {', '.join(keys)} FROM {table}")
        observed = [tuple(row.get(key) for key in keys) for row in remote_keys]
        expected = list(local_keys.astype(str).itertuples(index=False, name=None))
        if any(None in key for key in observed) or len(set(observed)) != len(observed):
            raise ValueError(f"Catalog null/duplicate keys: {table}")
        if set(observed) != set(expected) or len(observed) != len(expected):
            raise ValueError(f"Catalog key population mismatch: {table}")
    t4 = json.loads((root / "configs/t4_gold_reconciliation.json").read_text(encoding="utf-8"))
    t5 = json.loads((root / "configs/t5_wm2_split_manifest.json").read_text(encoding="utf-8"))
    reconciliation = execute_athena_reconciliation(runner, t4, t5)
    # Compare complete key/feature rows, not only aggregate totals.
    local = pd.read_parquet(root / "data/gold/order_features_and_splits.parquet")
    columns = ["order_id", "split_role", *t5["feature_columns"], "y_days", "y_late"]
    cloud_rows, _ = runner.run_query("SELECT " + ", ".join(columns) + " FROM order_features_and_splits")
    verify_feature_rows(local, cloud_rows, columns)
    verify_remote_snapshot(s3, bucket, snapshot)
    reconciliation["total_athena_bytes_scanned"] = runner.total_bytes_scanned
    reconciliation["athena_cost"] = usage_estimate(
        sum(q["billable_bytes"] for q in runner.query_statistics) / 1e12, "billable_TB", None, ""
    )
    report = {
        "schema": "waymark.c1_reconciliation/v2",
        "status": "OK",
        "gate": "WM1_VERIFIED",
        "snapshot": snapshot,
        "transport": transport,
        "glue_job_execution": glue_summary,
        "glue_database": database,
        "tables_registered": tables,
        "athena_reconciliation": reconciliation,
    }
    from waymark.snapshots import put_s3_immutable

    put_s3_immutable(
        s3, bucket, prefix + f"acceptance/{glue_summary['job_run_id']}.json", canonical_json(report)
    )
    write_immutable(
        root / "data/cloud_runs" / snapshot["snapshot_id"] / f"{glue_summary['job_run_id']}.json",
        canonical_json(report),
    )
    return report


def verify_feature_rows(local: Any, rows: list[dict[str, Any]], columns: list[str]) -> None:
    """Compare key sets and all model inputs using the declared absolute/relative tolerance."""
    import pandas as pd

    if len(rows) != len(local) or any(row.get("order_id") is None for row in rows):
        raise ValueError("Feature/key row count mismatch")
    remote = pd.DataFrame(rows)
    if remote["order_id"].duplicated().any() or set(remote["order_id"]) != set(local["order_id"]):
        raise ValueError("Feature order key mismatch")
    lhs = local.set_index("order_id").sort_index()
    rhs = remote.set_index("order_id").sort_index()
    for column in columns:
        if column == "order_id":
            continue
        a, b = lhs[column], rhs[column]
        if not (a.isna().to_numpy() == b.isna().to_numpy()).all():
            raise ValueError(f"Feature null mismatch: {column}")
        valid = a.notna()
        if pd.api.types.is_numeric_dtype(a) or pd.api.types.is_bool_dtype(a):

            def number(value: Any) -> float:
                if str(value).lower() in {"true", "false"}:
                    return float(str(value).lower() == "true")
                return float(value)

            if any(
                not math.isclose(float(x), number(y), rel_tol=1e-6, abs_tol=1e-6)
                for x, y in zip(a[valid], b[valid], strict=True)
            ):
                raise ValueError(f"Feature numeric mismatch: {column}")
        elif not a[valid].astype(str).equals(b[valid].astype(str)):
            raise ValueError(f"Feature category mismatch: {column}")
