"""Command-line entrypoint for Waymark contract, data reconstruction, and environment checks."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from waymark.artifacts import canonical_json, write_immutable
from waymark.bronze_silver import build_bronze, build_silver
from waymark.cloud_slice import run_c1_orchestration
from waymark.contracts import load_discriminating_fixture, load_wm0_contract
from waymark.features_splits import build_t5_features_and_splits
from waymark.gold_model import build_gold_analytical_model, verify_duckdb_sql_equivalence
from waymark.snapshots import build_lake_snapshot
from waymark.training import create_sourcedir_tarball, run_remote_training


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="waymark", description="Waymark fulfillment DE/DS CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    verify_parser = subparsers.add_parser("verify-wm0", help="Validate WM0 contract and synthetic fixture")
    verify_parser.add_argument("--contract", default="configs/wm0_contract.yaml")
    verify_parser.add_argument("--fixture", default="configs/t1_discriminating_fixture.json")

    t3_parser = subparsers.add_parser(
        "build-t3", help="Execute T3 local Bronze and Silver reconstruction and accounting"
    )
    t3_parser.add_argument("--contract", default="configs/wm0_contract.yaml")
    t3_parser.add_argument("--raw-dir", default="data/raw/olist")
    t3_parser.add_argument("--bronze-dir", default="data/bronze")
    t3_parser.add_argument("--silver-dir", default="data/silver")
    t3_parser.add_argument(
        "--export-summary",
        default="configs/t3_source_manifest.json",
        help="Path to write shareable source hash and Silver accounting summary",
    )

    t4_parser = subparsers.add_parser(
        "build-t4",
        help="Execute T4 Gold analytical model construction and DuckDB SQL parity verification",
    )
    t4_parser.add_argument("--contract", default="configs/wm0_contract.yaml")
    t4_parser.add_argument("--silver-dir", default="data/silver")
    t4_parser.add_argument("--gold-dir", default="data/gold")
    t4_parser.add_argument(
        "--export-summary",
        default="configs/t4_gold_reconciliation.json",
        help="Path to write shareable Gold reconciliation and DuckDB SQL parity summary",
    )

    t5_parser = subparsers.add_parser(
        "build-t5",
        help="Execute T5 point-in-time feature builder, maturity-purged splits, and WM2 verification",
    )
    t5_parser.add_argument("--contract", default="configs/wm0_contract.yaml")
    t5_parser.add_argument("--gold-dir", default="data/gold")
    t5_parser.add_argument("--features-dir", default="data/gold")
    t5_parser.add_argument(
        "--export-summary",
        default="configs/t5_wm2_split_manifest.json",
        help="Path to write shareable T5 / WM2 feature and split manifest",
    )

    c1_parser = subparsers.add_parser(
        "run-c1",
        help="Execute C1 cloud slice: S3 sync, Glue verification, Athena DDL, and WM1 reconciliation",
    )
    c1_parser.add_argument("--profile", default="waymark")
    c1_parser.add_argument("--region", default="ap-southeast-2")
    c1_parser.add_argument("--aws-cli", action="store_true", help="Use AWS CLI for all C1 cloud calls")
    c1_parser.add_argument(
        "--resume-glue-run", help="Resume reconciliation using an exact snapshot-bound Glue run"
    )

    train_parser = subparsers.add_parser(
        "train-remote",
        help="Execute T6/T7 remote SageMaker training on ml.m5.large and Gate WM3 evaluation",
    )
    train_parser.add_argument("--profile", default="waymark")
    train_parser.add_argument("--region", default="ap-southeast-2")
    train_parser.add_argument("--no-spot", action="store_true", help="Disable Managed Spot training")
    train_parser.add_argument("--cloud-receipt", type=Path, required=True)
    train_parser.add_argument("--environment-receipt", type=Path, required=True)
    train_parser.add_argument("--on-demand-hourly-rate", type=float)
    train_parser.add_argument(
        "--rate-source", default="", help="Dated regional price source, not a Spot rate"
    )
    train_parser.add_argument(
        "--allow-holdout",
        action="store_true",
        help="Explicitly authorize one final holdout evaluation after development freeze",
    )
    prepare_parser = subparsers.add_parser(
        "prepare-cloud-snapshot", help="Prepare a local immutable inventory only"
    )
    prepare_parser.add_argument("--output", type=Path, required=True)
    source_parser = subparsers.add_parser(
        "prepare-training-source", help="Package remote source without fitting or AWS calls"
    )
    source_parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args(argv)
    if args.command == "prepare-training-source":
        import hashlib

        payload = create_sourcedir_tarball(Path(__file__).resolve().parents[2])
        write_immutable(args.output, payload)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "status": "PREPARED_LOCALLY_NOT_UPLOADED",
                }
            )
        )
        return 0
    if args.command == "prepare-cloud-snapshot":
        root = Path(__file__).resolve().parents[2]
        snapshot = build_lake_snapshot(root)
        write_immutable(args.output, canonical_json(snapshot))
        contract = load_wm0_contract(root / "configs/wm0_contract.yaml")
        o7 = contract.o7_aws_and_cost
        bucket = f"waymark-lake-pilot-{o7.account_id}-{o7.region}"
        print(
            json.dumps(
                {
                    "snapshot_id": snapshot["snapshot_id"],
                    "output": str(args.output),
                    "glue_snapshot_script_uri": f"s3://{bucket}/snapshots/{snapshot['snapshot_id']}/scripts/glue_etl_entry.py",
                    "status": "PREPARED_LOCALLY_NOT_UPLOADED",
                },
                indent=2,
            )
        )
        return 0
    if args.command == "verify-wm0":
        contract = load_wm0_contract(args.contract)
        fixture = load_discriminating_fixture(args.fixture)
        summary = {
            "status": "OK",
            "contract_id": contract.contract_id,
            "contract_version": contract.contract_version,
            "fixture_id": fixture.get("fixture_id"),
            "account_id": contract.o7_aws_and_cost.account_id,
            "cli_profile": contract.o7_aws_and_cost.cli_profile,
            "region": contract.o7_aws_and_cost.region,
            "budget_cap_usd": contract.o7_aws_and_cost.total_budget_cap_usd,
            "planned_ceiling_usd": contract.o7_aws_and_cost.stage_cost_ceiling_usd.total_planned_ceiling_usd,
        }
        print(json.dumps(summary, indent=2))
        return 0

    if args.command == "build-t3":
        contract = load_wm0_contract(args.contract)
        bronze_manifest = build_bronze(
            raw_dir=Path(args.raw_dir),
            bronze_dir=Path(args.bronze_dir),
            contract=contract,
        )
        silver_accounting = build_silver(
            bronze_dir=Path(args.bronze_dir),
            silver_dir=Path(args.silver_dir),
            contract=contract,
        )
        combined = {
            "status": "OK",
            "stage": "T3",
            "contract_id": contract.contract_id,
            "contract_version": contract.contract_version,
            "bronze_manifest": bronze_manifest,
            "silver_accounting": silver_accounting,
        }
        if args.export_summary:
            summary_path = Path(args.export_summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            write_immutable(summary_path, (json.dumps(combined, indent=2) + "\n").encode())
        print(json.dumps(combined, indent=2))
        return 0

    if args.command == "build-t4":
        contract = load_wm0_contract(args.contract)
        gold_recon = build_gold_analytical_model(
            silver_dir=Path(args.silver_dir),
            gold_dir=Path(args.gold_dir),
            contract=contract,
        )
        sql_parity = verify_duckdb_sql_equivalence(
            silver_dir=Path(args.silver_dir),
            gold_dir=Path(args.gold_dir),
            contract=contract,
        )
        combined = {
            "status": "OK",
            "stage": "T4",
            "contract_id": contract.contract_id,
            "contract_version": contract.contract_version,
            "gold_reconciliation": gold_recon,
            "duckdb_sql_parity": sql_parity,
        }
        if args.export_summary:
            summary_path = Path(args.export_summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            write_immutable(summary_path, (json.dumps(combined, indent=2) + "\n").encode())
        print(json.dumps(combined, indent=2))
        return 0

    if args.command == "build-t5":
        contract = load_wm0_contract(args.contract)
        manifest = build_t5_features_and_splits(
            gold_dir=Path(args.gold_dir),
            features_dir=Path(args.features_dir),
            contract=contract,
        )
        if args.export_summary:
            summary_path = Path(args.export_summary)
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            write_immutable(summary_path, (json.dumps(manifest, indent=2) + "\n").encode())
        print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "run-c1":
        manifest = run_c1_orchestration(
            profile_name=args.profile,
            region_name=args.region,
            use_aws_cli=args.aws_cli,
            resume_glue_run_id=args.resume_glue_run,
        )
        print(json.dumps(manifest, indent=2))
        return 0

    if args.command == "train-remote":
        report = run_remote_training(
            profile_name=args.profile,
            region_name=args.region,
            use_spot=not args.no_spot,
            cloud_receipt_path=args.cloud_receipt,
            environment_receipt_path=args.environment_receipt,
            allow_holdout=args.allow_holdout,
            rate_usd_per_hour=args.on_demand_hourly_rate,
            rate_source=args.rate_source,
        )
        print(json.dumps(report, indent=2))
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
