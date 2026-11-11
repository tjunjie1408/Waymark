"""Explicit preparation, submission, completion and committed-only reading commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from uuid import uuid4

from waymark.artifacts import canonical_json, write_immutable
from waymark.batch_adapter import Boto3BatchClient
from waymark.batch_contract import canonical_bytes, digest, manifest_key
from waymark.batch_delivery import (
    finalize_batch,
    load_batch_manifest,
    read_committed_batch,
    register_committed_table,
    submit_batch,
)
from waymark.batch_preparation import load_bundle, prepare_batch
from waymark.contracts import load_wm0_contract
from waymark.snapshots import verify_account

BATCH_COMMANDS = {
    "prepare-batch",
    "upload-batch",
    "register-batch-model",
    "submit-batch",
    "finalize-batch",
    "read-batch",
    "register-batch-table",
}


def add_batch_commands(subparsers) -> None:
    prepare = subparsers.add_parser("prepare-batch", help="Prepare accepted model and scoring inputs locally")
    for name in ["features", "model-archive", "acceptance", "environment-receipt", "output"]:
        prepare.add_argument(f"--{name}", type=Path, required=True)
    for name in ["model-name", "scoring-cutoff-date", "execution-role-arn"]:
        prepare.add_argument(f"--{name}", required=True)
    prepare.add_argument("--rows-per-part", type=int, default=2000)
    for name in sorted(BATCH_COMMANDS - {"prepare-batch"}):
        parser = subparsers.add_parser(name, help=f"Explicit {name} operation; no fitting")
        parser.add_argument("--profile", default="waymark")
        parser.add_argument("--region", default="ap-southeast-2")
        if name in {"upload-batch", "register-batch-model"}:
            parser.add_argument("--bundle", type=Path, required=True)
        else:
            parser.add_argument("--manifest-key", required=True)
        if name in {"submit-batch", "finalize-batch"}:
            parser.add_argument("--attempt", type=int, default=1)
        if name == "read-batch":
            parser.add_argument("--output", type=Path, required=True)
        if name == "register-batch-table":
            parser.add_argument("--database", required=True)


def run_batch_command(args: argparse.Namespace) -> int:
    root = Path(__file__).resolve().parents[2]
    contract = load_wm0_contract(root / "configs/wm0_contract.yaml")
    bucket = f"waymark-lake-pilot-{contract.o7_aws_and_cost.account_id}-{contract.o7_aws_and_cost.region}"
    if args.command == "prepare-batch":
        import pandas as pd

        result = prepare_batch(
            features=pd.read_parquet(args.features),
            model_archive=args.model_archive.read_bytes(),
            acceptance=args.acceptance.read_bytes(),
            environment=json.loads(args.environment_receipt.read_bytes()),
            model_name=args.model_name,
            bucket=bucket,
            scoring_cutoff_date=args.scoring_cutoff_date,
            execution_role_arn=args.execution_role_arn,
            output=args.output,
            rows_per_part=args.rows_per_part,
        )
        print(json.dumps(result, indent=2))
        return 0
    import boto3

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    verify_account(session.client("sts"), args.region, contract)
    client = Boto3BatchClient(args.region, session=session)
    if args.command in {"upload-batch", "register-batch-model"}:
        manifest, payloads, request = load_bundle(args.bundle)
    else:
        manifest = load_batch_manifest(client, bucket, args.manifest_key)
    ledger = root / "data/batch_runs" / digest(canonical_bytes(manifest)) / f"{args.command}-{uuid4().hex}"
    write_immutable(
        ledger / "request.json",
        canonical_json(
            {
                "command": args.command,
                "manifest_key": manifest_key(manifest),
                "attempt": getattr(args, "attempt", None),
            }
        ),
    )
    try:
        if args.command == "upload-batch":
            for key, body in payloads.items():
                client.put_bytes_immutable(bucket, key, body)
            result = {"status": "UPLOADED", "manifest_key": manifest_key(manifest)}
        elif args.command == "register-batch-model":
            for key, body in payloads.items():
                if client.object_sha256(bucket, key) != digest(body):
                    raise ValueError("Upload/verify prepared assets before registering the model")
            # Existing/uncertain registrations are inspected by the operator, never renamed/retried here.
            response = session.client("sagemaker").create_model(**request)
            result = {"status": "MODEL_REGISTERED", "model_arn": response["ModelArn"]}
        elif args.command == "submit-batch":
            result = submit_batch(client, manifest, attempt_seq=args.attempt)
        elif args.command == "finalize-batch":
            result = finalize_batch(client, manifest, attempt_seq=args.attempt)
        elif args.command == "read-batch":
            frame, marker = read_committed_batch(client, manifest)
            body = frame.to_parquet(index=False)
            write_immutable(args.output, body)
            result = {
                "status": "READ_VERIFIED",
                "rows": len(frame),
                "source_s3_uri": marker["gold_s3_uri"],
                "source_sha256": marker["sha256"],
                "local_sha256": digest(body),
                "output": str(args.output),
            }
        else:
            table = register_committed_table(client, session.client("glue"), manifest, args.database)
            result = {"status": "TABLE_REGISTERED", "database": args.database, "table": table}
        write_immutable(ledger / "result.json", canonical_json(result))
    except Exception as exc:
        write_immutable(
            ledger / "failure.json", canonical_json({"status": "FAILED_OR_UNCERTAIN", "error": str(exc)})
        )
        raise
    print(json.dumps(result, indent=2))
    return 0
