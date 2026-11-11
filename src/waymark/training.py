"""SageMaker remote training launcher and Gate WM3 evaluation orchestrator.

Packages source code, triggers remote ml.m5.large Managed Spot training in
ap-southeast-2, monitors CloudWatch logs, and downloads frozen model artifacts.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3

logger = logging.getLogger("waymark.training")

# Official SageMaker Scikit-Learn training container for ap-southeast-2
SKLEARN_CONTAINER_URI = (
    "783357654285.dkr.ecr.ap-southeast-2.amazonaws.com/"
    "sagemaker-scikit-learn:1.2-1-cpu-py3"
)


def create_sourcedir_tarball(repo_root: Path) -> bytes:
    """Package train_entry.py and requirements.txt into a tar.gz payload."""
    entry_path = repo_root / "src" / "waymark" / "sagemaker_entry.py"
    req_path = repo_root / "src" / "waymark" / "remote_requirements.txt"

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        # Add train_entry.py as root entry
        entry_bytes = entry_path.read_bytes()
        ti_entry = tarfile.TarInfo(name="train_entry.py")
        ti_entry.size = len(entry_bytes)
        ti_entry.mtime = int(time.time())
        tar.addfile(ti_entry, io.BytesIO(entry_bytes))

        # Add requirements.txt
        req_bytes = req_path.read_bytes()
        ti_req = tarfile.TarInfo(name="requirements.txt")
        ti_req.size = len(req_bytes)
        ti_req.mtime = int(time.time())
        tar.addfile(ti_req, io.BytesIO(req_bytes))

    buf.seek(0)
    return buf.getvalue()


def launch_sagemaker_training_job(
    sm_client: Any,
    s3_client: Any,
    bucket_name: str,
    role_arn: str,
    repo_root: Path,
    use_spot: bool = True,
) -> dict[str, Any]:
    """Launch remote training job on ml.m5.large and await completion."""
    now_ts = int(datetime.now(UTC).timestamp())
    job_name = f"waymark-train-{now_ts}"

    # 1. Package and upload source tarball
    tar_bytes = create_sourcedir_tarball(repo_root)
    code_s3_key = f"code/sourcedir-{job_name}.tar.gz"
    s3_client.put_object(
        Bucket=bucket_name,
        Key=code_s3_key,
        Body=tar_bytes,
        ContentType="application/x-tar",
    )
    code_s3_uri = f"s3://{bucket_name}/{code_s3_key}"
    logger.info("Uploaded training code to %s (%d bytes)", code_s3_uri, len(tar_bytes))

    # 2. Configure training request
    input_s3_uri = f"s3://{bucket_name}/gold/order_features_and_splits/"
    output_s3_uri = f"s3://{bucket_name}/models/{job_name}/"

    hyperparameters = {
        "sagemaker_program": "train_entry.py",
        "sagemaker_submit_directory": code_s3_uri,
        "sagemaker_region": "ap-southeast-2",
        "sagemaker_container_log_level": "20",
    }

    stopping_condition = {"MaxRuntimeInSeconds": 1800}
    if use_spot:
        stopping_condition["MaxWaitTimeInSeconds"] = 3600

    training_request = {
        "TrainingJobName": job_name,
        "AlgorithmSpecification": {
            "TrainingImage": SKLEARN_CONTAINER_URI,
            "TrainingInputMode": "File",
        },
        "RoleArn": role_arn,
        "InputDataConfig": [
            {
                "ChannelName": "training",
                "DataSource": {
                    "S3DataSource": {
                        "S3DataType": "S3Prefix",
                        "S3Uri": input_s3_uri,
                        "S3DataDistributionType": "FullyReplicated",
                    }
                },
                "CompressionType": "None",
                "RecordWrapperType": "None",
            }
        ],
        "OutputDataConfig": {"S3OutputPath": output_s3_uri},
        "ResourceConfig": {
            "InstanceType": "ml.m5.large",
            "InstanceCount": 1,
            "VolumeSizeInGB": 30,
        },
        "StoppingCondition": stopping_condition,
        "EnableManagedSpotTraining": use_spot,
        "HyperParameters": hyperparameters,
        "Tags": [
            {"Key": "Project", "Value": "waymark"},
            {"Key": "ContractId", "Value": "wm0_v1"},
            {"Key": "Stage", "Value": "T6_T7"},
        ],
    }

    logger.info("Submitting SageMaker training job %s...", job_name)
    sm_client.create_training_job(**training_request)

    # 3. Poll for completion
    logger.info("Monitoring SageMaker training job %s...", job_name)
    start_poll = time.time()
    while True:
        resp = sm_client.describe_training_job(TrainingJobName=job_name)
        status = resp["TrainingJobStatus"]
        secondary_status = resp.get("SecondaryStatus", "")
        elapsed = int(time.time() - start_poll)

        if status == "Completed":
            billable_secs = resp.get("BillableTimeInSeconds", elapsed)
            training_secs = resp.get("TrainingTimeInSeconds", elapsed)
            logger.info(
                "Training job %s completed in %ds (billable: %ds).",
                job_name,
                training_secs,
                billable_secs,
            )
            break
        elif status in ("Failed", "Stopped"):
            failure_reason = resp.get("FailureReason", "Unknown SageMaker failure")
            raise RuntimeError(f"SageMaker training job {job_name} failed: {failure_reason}")

        logger.info("Status: %s (%s), elapsed: %ds", status, secondary_status, elapsed)
        time.sleep(15)

    # 4. Download and parse evaluation summary
    model_artifact_uri = resp["ModelArtifacts"]["S3ModelArtifacts"]
    logger.info("Model artifacts produced at: %s", model_artifact_uri)

    # Extract evaluation_summary.json from model.tar.gz in memory
    model_key = model_artifact_uri.replace(f"s3://{bucket_name}/", "")
    model_tar_obj = s3_client.get_object(Bucket=bucket_name, Key=model_key)
    model_tar_bytes = model_tar_obj["Body"].read()

    summary_dict = {}
    with tarfile.open(fileobj=io.BytesIO(model_tar_bytes), mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("evaluation_summary.json"):
                extracted = tar.extractfile(member)
                if extracted:
                    summary_dict = json.loads(extracted.read().decode("utf-8"))

    # Estimate cloud cost on ml.m5.large spot
    # Spot rate ~$0.045/hr in ap-southeast-2 (vs $0.144 on-demand)
    hourly_rate = 0.045 if use_spot else 0.144
    estimated_cost = round((billable_secs / 3600.0) * hourly_rate, 4)

    full_report = {
        "status": "OK",
        "stage": "T6_T7",
        "gate": "WM3_REVIEW",
        "training_job_name": job_name,
        "instance_type": "ml.m5.large",
        "use_spot_instances": use_spot,
        "training_time_seconds": training_secs,
        "billable_time_seconds": billable_secs,
        "estimated_cost_usd": estimated_cost,
        "model_artifact_uri": model_artifact_uri,
        "evaluation_summary": summary_dict,
    }

    # Save to local configs/
    summary_path = repo_root / "configs" / "t6_t7_evaluation_summary.json"
    summary_path.write_text(json.dumps(full_report, indent=2) + "\n", encoding="utf-8")
    logger.info("Saved local summary to %s", summary_path)

    return full_report


def run_remote_training(
    profile_name: str = "waymark",
    region_name: str = "ap-southeast-2",
    repo_root: Path | None = None,
    use_spot: bool = True,
) -> dict[str, Any]:
    """Top-level entrypoint for executing remote SageMaker training."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[2]

    session = boto3.Session(profile_name=profile_name, region_name=region_name)
    sm_client = session.client("sagemaker")
    s3_client = session.client("s3")
    sts_client = session.client("sts")

    account_id = sts_client.get_caller_identity()["Account"]
    bucket_name = f"waymark-lake-pilot-{account_id}-{region_name}"
    role_arn = f"arn:aws:iam::{account_id}:role/waymark-pilot-sagemaker-exec-role"

    return launch_sagemaker_training_job(
        sm_client=sm_client,
        s3_client=s3_client,
        bucket_name=bucket_name,
        role_arn=role_arn,
        repo_root=repo_root,
        use_spot=use_spot,
    )
