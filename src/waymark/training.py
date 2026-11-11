"""Managed SageMaker jobs bound to verified input, source and environment identities."""

from __future__ import annotations

import hashlib
import io
import json
import logging
import tarfile
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import boto3

from waymark.artifacts import canonical_json, write_immutable
from waymark.contracts import load_wm0_contract
from waymark.costs import training_cost
from waymark.snapshots import put_s3_immutable, verify_account, verify_remote_snapshot
from waymark.training_environment import validate_environment

logger = logging.getLogger(__name__)
SOURCE_FILES = (
    "__init__.py",
    "artifacts.py",
    "evaluation.py",
    "feature_schema.py",
    "training_environment.py",
    "sagemaker_entry.py",
    "remote_requirements.txt",
)
REQUIRED_MODELS = {
    "m1_lgbm_p10.pkl",
    "m1_lgbm_p50.pkl",
    "m1_lgbm_p90.pkl",
    "m2_catboost.cbm",
    "m2_xgboost.json",
    "m2_calibrated_catboost.pkl",
    "development_freeze.json",
}


def create_sourcedir_tarball(repo_root: Path) -> bytes:
    package = repo_root / "src/waymark"
    contents = {f"waymark/{name}": (package / name).read_bytes() for name in SOURCE_FILES}
    contents["requirements.txt"] = (package / "remote_requirements.txt").read_bytes()
    contents["train_entry.py"] = (
        b"from waymark.sagemaker_entry import main\nif __name__ == '__main__':\n    main()\n"
    )
    contents["source_manifest.json"] = canonical_json(
        {name: hashlib.sha256(body).hexdigest() for name, body in contents.items()}
    )
    buffer = io.BytesIO()
    import gzip

    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, body in sorted(contents.items()):
                info = tarfile.TarInfo(name)
                info.size = len(body)
                info.mtime = 0
                archive.addfile(info, io.BytesIO(body))
    return buffer.getvalue()


def parse_training_artifacts(payload: bytes, expected: dict[str, str]) -> dict[str, Any]:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(set(names)) != len(names) or not REQUIRED_MODELS.issubset(names):
            raise ValueError("Missing/duplicate required model artifacts")
        if names.count("evaluation_summary.json") != 1:
            raise ValueError("Exactly one root evaluation_summary.json is required")
        member = archive.getmember("evaluation_summary.json")
        if not member.isfile() or member.size > 20_000_000:
            raise ValueError("Invalid evaluation summary member")
        summary = json.loads(archive.extractfile(member).read())
        if summary.get("schema") != "waymark.training_evaluation/v2" or summary.get("status") != "OK":
            raise ValueError("Invalid evaluation summary schema/status")
        if summary.get("gate_wm3_passed") is not None or summary.get("gate") not in {
            "WM3_REVIEW_REQUIRED",
            "WM3_INCOMPLETE",
        }:
            raise ValueError("Scientific acceptance requires explicit user review")
        identities = summary.get("identities", {})
        if any(identities.get(key) != value for key, value in expected.items()):
            raise ValueError("Training artifact identity mismatch")
        required_outputs = {"predictions.parquet", "shap_values.parquet", "shap_background.parquet"}
        if (
            not required_outputs.issubset(names)
            or not summary.get("uncertainty")
            or not summary.get("strata")
        ):
            raise ValueError("Incomplete required evaluation evidence")
        inventory = summary.get("model_artifacts", [])
        if not (REQUIRED_MODELS | required_outputs).issubset({item["file"] for item in inventory}):
            raise ValueError("Incomplete artifact hash inventory")
        for item in inventory:
            if item["file"] not in names:
                raise ValueError("Missing inventoried model artifact")
            info = archive.getmember(item["file"])
            if not info.isfile() or info.size > 500_000_000:
                raise ValueError("Invalid model artifact member")
            if hashlib.sha256(archive.extractfile(info).read()).hexdigest() != item["sha256"]:
                raise ValueError(f"Model artifact hash mismatch: {item['file']}")
        for section, keys in [
            (
                "eta",
                {
                    "pinball_loss_p10",
                    "pinball_loss_p50",
                    "pinball_loss_p90",
                    "interval_coverage_80",
                    "interval_mean_width_days",
                },
            ),
            ("risk", {"roc_auc", "pr_auc", "brier_score", "expected_calibration_error_10bin"}),
        ]:
            if not keys.issubset(summary.get(section, {})):
                raise ValueError(f"Missing {section} metrics")
        return summary


def launch_sagemaker_training_job(
    sm_client: Any,
    s3_client: Any,
    bucket_name: str,
    role_arn: str,
    repo_root: Path,
    use_spot: bool = True,
    *,
    cloud_receipt: dict[str, Any] | None = None,
    environment_receipt: dict[str, Any] | None = None,
    allow_holdout: bool = False,
    timeout_seconds: float = 4200.0,
    rate_usd_per_hour: float | None = None,
    rate_source: str = "",
) -> dict[str, Any]:
    if cloud_receipt is None or environment_receipt is None:
        raise ValueError("Verified cloud snapshot and no-fit environment receipts are required")
    image = validate_environment(environment_receipt, repo_root / "src/waymark/remote_requirements.txt")
    if (
        cloud_receipt.get("schema") != "waymark.c1_reconciliation/v2"
        or cloud_receipt.get("gate") != "WM1_VERIFIED"
    ):
        raise ValueError("C1 v2 acceptance receipt is required; legacy aggregates are insufficient")
    snapshot = cloud_receipt["snapshot"]
    verify_remote_snapshot(s3_client, bucket_name, snapshot)
    snapshot_id = snapshot["snapshot_id"]
    prefix = f"snapshots/{snapshot_id}/"
    receipt_key = prefix + f"acceptance/{cloud_receipt['glue_job_execution']['job_run_id']}.json"
    remote_receipt = s3_client.get_object(Bucket=bucket_name, Key=receipt_key)["Body"].read()
    if remote_receipt != canonical_json(cloud_receipt):
        raise ValueError("Remote cloud acceptance receipt mismatch")
    feature = next(
        obj
        for obj in snapshot["objects"]
        if obj["path"] == "gold/order_features_and_splits/order_features_and_splits.parquet"
    )
    source = create_sourcedir_tarball(repo_root)
    source_hash = hashlib.sha256(source).hexdigest()
    code_key = f"code/{source_hash}/sourcedir.tar.gz"
    put_s3_immutable(s3_client, bucket_name, code_key, source)
    job_name = f"waymark-train-{uuid4().hex[:24]}"
    identities = {
        "source_sha256": source_hash,
        "feature_sha256": feature["sha256"],
        "snapshot_id": snapshot_id,
    }
    root = repo_root / "data/training_runs" / job_name
    root.mkdir(parents=True, exist_ok=False)
    settings = load_wm0_contract(repo_root / "configs/wm0_contract.yaml").o7_aws_and_cost
    sizing = settings.model_extra["service_sizing_decisions"]["sagemaker_training"]
    hyperparameters = {
        "sagemaker_program": "train_entry.py",
        "sagemaker_submit_directory": f"s3://{bucket_name}/{code_key}",
        "sagemaker_region": settings.region,
        "image-uri": image,
        "source-sha256": source_hash,
        "feature-sha256": feature["sha256"],
        "snapshot-id": snapshot_id,
        "random-seed": "42",
    }
    if allow_holdout:
        hyperparameters["allow-holdout"] = "true"
    stopping = {"MaxRuntimeInSeconds": sizing["max_run_seconds_per_job"]}
    if use_spot:
        stopping["MaxWaitTimeInSeconds"] = sizing["max_wait_seconds_per_job"]
    request = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {"TrainingImage": image, "TrainingInputMode": "File"},
        "RoleArn": role_arn,
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": f"s3://{bucket_name}/{prefix}{feature['path']}",
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
            }
        ],
        "OutputDataConfig": {"S3OutputPath": f"s3://{bucket_name}/models/{job_name}/"},
        "ResourceConfig": {"InstanceType": sizing["instance_type"], "InstanceCount": 1, "VolumeSizeInGB": 30},
        "CheckpointConfig": {
            "S3Uri": f"s3://{bucket_name}/checkpoints/{job_name}/",
            "LocalPath": "/opt/ml/checkpoints",
        },
        "StoppingCondition": stopping,
        "EnableManagedSpotTraining": use_spot,
        "HyperParameters": hyperparameters,
        "Tags": [{"Key": "Project", "Value": "waymark"}, {"Key": "ContractId", "Value": "wm0_v1"}],
    }
    write_immutable(root / "request.json", canonical_json(request))
    deadline = time.monotonic() + timeout_seconds
    write_immutable(root / "attempt.json", canonical_json({"status": "SUBMITTING", **identities}))
    try:
        sm_client.create_training_job(**request)
        while True:
            response = sm_client.describe_training_job(TrainingJobName=job_name)
            state = response["TrainingJobStatus"]
            if state == "Completed":
                break
            if state in {"Failed", "Stopped"}:
                raise RuntimeError(f"SageMaker {job_name}: {state}: {response.get('FailureReason', '')}")
            if state not in {"InProgress", "Stopping"}:
                raise RuntimeError(f"Unknown SageMaker state {state}: {job_name}")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Polling deadline exceeded; inspect SageMaker job {job_name}")
            time.sleep(15)
        uri = response["ModelArtifacts"]["S3ModelArtifacts"]
        if not uri.startswith(f"s3://{bucket_name}/models/{job_name}/"):
            raise ValueError("Unexpected model artifact location")
        archive = s3_client.get_object(Bucket=bucket_name, Key=uri.removeprefix(f"s3://{bucket_name}/"))[
            "Body"
        ].read()
        summary = parse_training_artifacts(archive, identities)
        write_immutable(root / "model.tar.gz", archive)
        cost = training_cost(response["BillableTimeInSeconds"], 1, rate_usd_per_hour, rate_source)
        report = {
            "schema": "waymark.training_run/v2",
            "status": "OK",
            "training_job_name": job_name,
            "gate": summary["gate"],
            "evaluation_summary": summary,
            "model_artifact_uri": uri,
            "model_artifact_sha256": hashlib.sha256(archive).hexdigest(),
            "billable_time_seconds": response["BillableTimeInSeconds"],
            "training_time_seconds": response["TrainingTimeInSeconds"],
            "cost": cost,
        }
        write_immutable(root / "result.json", canonical_json(report))
        return report
    except Exception as exc:
        write_immutable(
            root / "failure.json",
            canonical_json(
                {
                    "status": "FAILED_OR_UNCERTAIN",
                    "job_name": job_name,
                    "error": str(exc),
                    "identities": identities,
                    "cost": "UNKNOWN_UNTIL_REMOTE_USAGE_READBACK",
                }
            ),
        )
        raise


def run_remote_training(
    profile_name: str = "waymark",
    region_name: str = "ap-southeast-2",
    repo_root: Path | None = None,
    use_spot: bool = True,
    *,
    cloud_receipt_path: Path | None = None,
    environment_receipt_path: Path | None = None,
    allow_holdout: bool = False,
    rate_usd_per_hour: float | None = None,
    rate_source: str = "",
) -> dict[str, Any]:
    root = repo_root or Path(__file__).resolve().parents[2]
    contract = load_wm0_contract(root / "configs/wm0_contract.yaml")
    if cloud_receipt_path is None or environment_receipt_path is None:
        raise ValueError("Supply the accepted cloud snapshot and no-fit environment receipt paths")
    cloud = json.loads(cloud_receipt_path.read_text(encoding="utf-8"))
    environment = json.loads(environment_receipt_path.read_text(encoding="utf-8"))
    validate_environment(environment, root / "src/waymark/remote_requirements.txt")
    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    account = verify_account(session.client("sts"), region_name, contract)
    return launch_sagemaker_training_job(
        session.client("sagemaker"),
        session.client("s3"),
        f"waymark-lake-pilot-{account}-{region_name}",
        f"arn:aws:iam::{account}:role/waymark-pilot-sagemaker-exec-role",
        root,
        use_spot,
        cloud_receipt=cloud,
        environment_receipt=environment,
        allow_holdout=allow_holdout,
        rate_usd_per_hour=rate_usd_per_hour,
        rate_source=rate_source,
    )
