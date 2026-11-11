"""Cloud data slice orchestration and Athena reconciliation for Gate WM1.

Handles S3 lake layout sync, Glue Python Shell execution, Athena DDL registration,
analytical query execution, and reconciliation against local T3-T5 manifests.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

import boto3
import pyarrow.parquet as pq

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
    schema = pq.read_schema(parquet_path)
    col_defs = []
    for field in schema:
        t_str = str(field.type)
        sql_type = TYPE_MAP.get(t_str, "STRING")
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

    def run_query(self, query: str) -> tuple[list[dict[str, Any]], int]:
        """Execute query, wait for completion, and return rows as dictionaries."""
        response = self.client.start_query_execution(
            QueryString=query,
            QueryExecutionContext={"Database": self.database},
            WorkGroup=self.workgroup,
        )
        query_id = response["QueryExecutionId"]

        while True:
            status_resp = self.client.get_query_execution(QueryExecutionId=query_id)
            state = status_resp["QueryExecution"]["Status"]["State"]
            if state == "SUCCEEDED":
                stats = status_resp["QueryExecution"]["Statistics"]
                bytes_scanned = stats.get("DataScannedInBytes", 0)
                self.total_bytes_scanned += bytes_scanned
                break
            elif state in ("FAILED", "CANCELLED"):
                reason = status_resp["QueryExecution"]["Status"].get(
                    "StateChangeReason", "Unknown error"
                )
                msg = f"Athena query {query_id} failed with state {state}: {reason}\nQuery: {query}"
                raise RuntimeError(msg)
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


def sync_local_lake_to_s3(
    s3_client: Any,
    bucket_name: str,
    repo_root: Path,
) -> dict[str, int]:
    """Upload Bronze, Silver, Gold parquet files and manifests to S3 with partitioned prefixes."""
    uploaded_counts: dict[str, int] = {"scripts": 0, "silver": 0, "gold": 0, "manifests": 0}

    # 1. Upload Glue script
    glue_script = repo_root / "scripts" / "glue_etl_entry.py"
    if glue_script.exists():
        s3_client.upload_file(str(glue_script), bucket_name, "scripts/glue_etl_entry.py")
        uploaded_counts["scripts"] += 1

    # 2. Upload Silver tables
    silver_dir = repo_root / "data" / "silver"
    for p in silver_dir.glob("*.parquet"):
        table_name = p.stem
        s3_key = f"silver/{table_name}/{p.name}"
        s3_client.upload_file(str(p), bucket_name, s3_key)
        uploaded_counts["silver"] += 1

    # 3. Upload Gold tables
    gold_dir = repo_root / "data" / "gold"
    for p in gold_dir.glob("*.parquet"):
        table_name = p.stem
        s3_key = f"gold/{table_name}/{p.name}"
        s3_client.upload_file(str(p), bucket_name, s3_key)
        uploaded_counts["gold"] += 1

    # 4. Upload configs / manifests
    configs_dir = repo_root / "configs"
    for manifest_name in [
        "wm0_contract.yaml",
        "t1_discriminating_fixture.json",
        "t3_source_manifest.json",
        "t4_gold_reconciliation.json",
        "t5_wm2_split_manifest.json",
    ]:
        m_path = configs_dir / manifest_name
        if m_path.exists():
            s3_client.upload_file(str(m_path), bucket_name, f"manifests/{manifest_name}")
            uploaded_counts["manifests"] += 1

    return uploaded_counts


def run_glue_job_verification(glue_client: Any, job_name: str) -> dict[str, Any]:
    """Trigger the Glue Python Shell job and await SUCCEEDED status."""
    run_resp = glue_client.start_job_run(JobName=job_name)
    job_run_id = run_resp["JobRunId"]

    start_time = time.time()
    while True:
        status_resp = glue_client.get_job_run(JobName=job_name, RunId=job_run_id)
        job_run = status_resp["JobRun"]
        state = job_run["JobRunState"]
        if state == "SUCCEEDED":
            duration = job_run.get("ExecutionTime", int(time.time() - start_time))
            dpu = job_run.get("AllocatedCapacity", 0.0625)
            # Glue Python Shell pricing: $0.0275 per DPU-hour (0.0625 DPU)
            cost_usd = (duration / 3600.0) * dpu * 0.44
            return {
                "job_name": job_name,
                "job_run_id": job_run_id,
                "status": "SUCCEEDED",
                "execution_time_seconds": duration,
                "allocated_dpu": dpu,
                "estimated_cost_usd": round(cost_usd, 6),
            }
        elif state in ("FAILED", "STOPPED", "TIMEOUT"):
            error_msg = job_run.get("ErrorMessage", "Unknown Glue execution error")
            msg = f"Glue job {job_name} run {job_run_id} failed with status {state}: {error_msg}"
            raise RuntimeError(msg)
        time.sleep(5)


def register_catalog_tables(
    runner: AthenaQueryRunner,
    bucket_name: str,
    repo_root: Path,
) -> list[str]:
    """Register all Silver and Gold external Parquet tables in Athena."""
    tables_registered: list[str] = []

    # Drop and recreate tables to ensure idempotent fresh metadata
    silver_dir = repo_root / "data" / "silver"
    for p in sorted(silver_dir.glob("*.parquet")):
        table_name = p.stem
        runner.run_query(f"DROP TABLE IF EXISTS {table_name}")
        s3_loc = f"s3://{bucket_name}/silver/{table_name}/"
        ddl = pyarrow_schema_to_athena_ddl(table_name, p, s3_loc)
        runner.run_query(ddl)
        tables_registered.append(table_name)

    gold_dir = repo_root / "data" / "gold"
    for p in sorted(gold_dir.glob("*.parquet")):
        table_name = p.stem
        runner.run_query(f"DROP TABLE IF EXISTS {table_name}")
        s3_loc = f"s3://{bucket_name}/gold/{table_name}/"
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

    # Validate parity against T4 and T5
    t5_counts = t5_manifest["split_summary"]["counts"]

    # Verify counts
    assert count_checks["silver_orders"] == 99441
    assert count_checks["fact_order_fulfillment"] == 99441
    assert count_checks["dim_customer"] == 99441
    assert count_checks["dim_seller"] == 3095
    assert count_checks["dim_product"] == 32951
    assert count_checks["dim_geolocation_zip"] == 19010
    assert count_checks["order_features_and_splits"] == 99441

    # Verify monetary sums match local T4 values
    assert abs(reconciliation_results["athena_monetary_summary"]["sum_price_brl"] - 13591643.70) < 0.1
    assert abs(reconciliation_results["athena_monetary_summary"]["sum_freight_brl"] - 2251909.54) < 0.1
    assert abs(reconciliation_results["athena_monetary_summary"]["sum_payment_brl"] - 16008872.12) < 0.1
    assert reconciliation_results["athena_monetary_summary"]["sum_items"] == 112650

    # Verify split counts match T5
    assert split_summary["train"] == t5_counts["train_mature_retained"]  # 64,404
    purged_train = t5_counts["train_purged_immature_at_boundary"]
    assert split_summary["EXCL_IMMATURE_AT_TRAIN_BOUNDARY"] == purged_train  # 3,072
    assert split_summary["val_calibration"] == t5_counts["val_calibration_mature_retained"]  # 12,551
    purged_val = t5_counts["val_purged_immature_at_boundary"]
    assert split_summary["EXCL_IMMATURE_AT_VAL_BOUNDARY"] == purged_val  # 1,908
    assert split_summary["holdout_test"] == t5_counts["holdout_test_orders"]  # 14,460

    # Verify top corridors match local T4 DuckDB parity
    t4_corr_list = t4_summary["duckdb_sql_parity"]["top_5_corridors_reference_query"]
    t4_corridors = {c["state_corridor"]: c for c in t4_corr_list}
    for c in reconciliation_results["athena_top_5_corridors"]:
        sc = c["state_corridor"]
        expected = t4_corridors[sc]
        assert c["eligible_orders"] == expected["eligible_orders"]
        assert abs(c["mean_eta_days"] - expected["mean_eta_days"]) < 0.01
        assert abs(c["mean_distance_km"] - expected["mean_distance_km"]) < 0.1

    reconciliation_results["gate_wm1_status"] = "PASSED"
    reconciliation_results["total_athena_bytes_scanned"] = runner.total_bytes_scanned
    reconciliation_results["estimated_athena_cost_usd"] = round(
        (runner.total_bytes_scanned / (1024**4)) * 5.0, 6
    )

    return reconciliation_results


def run_c1_orchestration(
    profile_name: str = "waymark",
    region_name: str = "ap-southeast-2",
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Execute complete C1 cloud slice workflow: sync, Glue job, Athena DDL, query reconciliation."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    s3_client = session.client("s3")
    glue_client = session.client("glue")
    athena_client = session.client("athena")
    sts_client = session.client("sts")

    account_id = sts_client.get_caller_identity()["Account"]
    bucket_name = f"waymark-lake-pilot-{account_id}-{region_name}"
    glue_db = "waymark_pilot_db"
    athena_wg = "waymark_pilot_wg"
    glue_job_name = "waymark-pilot-medallion-pythonshell"

    logger.info("Starting C1 upload to %s...", bucket_name)
    upload_summary = sync_local_lake_to_s3(s3_client, bucket_name, repo_root)

    logger.info("Executing Glue Python Shell verification job %s...", glue_job_name)
    glue_summary = run_glue_job_verification(glue_client, glue_job_name)

    logger.info("Registering Athena tables in %s...", glue_db)
    runner = AthenaQueryRunner(athena_client, athena_wg, glue_db)
    tables_registered = register_catalog_tables(runner, bucket_name, repo_root)

    # Load local T4 and T5 summaries for comparison
    t4_path = repo_root / "configs" / "t4_gold_reconciliation.json"
    t5_path = repo_root / "configs" / "t5_wm2_split_manifest.json"
    t4_summary = json.loads(t4_path.read_text(encoding="utf-8"))
    t5_manifest = json.loads(t5_path.read_text(encoding="utf-8"))

    logger.info("Running Athena analytical reconciliation queries...")
    recon_summary = execute_athena_reconciliation(runner, t4_summary, t5_manifest)

    c1_final_manifest = {
        "status": "OK",
        "stage": "C1",
        "gate": "WM1_VERIFIED",
        "contract_id": "wm0_v1",
        "contract_version": "1.0.0",
        "aws_account_id": account_id,
        "aws_region": region_name,
        "lake_bucket": bucket_name,
        "glue_database": glue_db,
        "athena_workgroup": athena_wg,
        "tables_registered": tables_registered,
        "upload_summary": upload_summary,
        "glue_job_execution": glue_summary,
        "athena_reconciliation": recon_summary,
    }

    output_path = repo_root / "configs" / "c1_cloud_reconciliation.json"
    output_path.write_text(json.dumps(c1_final_manifest, indent=2) + "\n", encoding="utf-8")

    # Also upload reconciliation output to S3
    s3_client.upload_file(str(output_path), bucket_name, "manifests/c1_cloud_reconciliation.json")

    return c1_final_manifest
