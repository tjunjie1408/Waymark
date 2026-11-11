"""Snapshot integrity, pure evaluation and cloud-boundary rejection tests; never fit models."""

from __future__ import annotations

import hashlib
import io
import runpy
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from botocore.exceptions import ClientError

from waymark.artifacts import canonical_json, write_immutable
from waymark.bronze_silver import build_bronze, build_silver, write_fixture_to_raw_csvs
from waymark.cloud_slice import run_c1_orchestration, verify_feature_rows
from waymark.contracts import load_wm0_contract
from waymark.costs import athena_billable_bytes, training_cost, usage_estimate
from waymark.evaluation import (
    baseline_predictions,
    binary_metrics,
    eta_metrics,
    expected_calibration_error,
    paired_cluster_interval,
    ranking_metrics,
    top_k_alerts,
)
from waymark.features_splits import build_t5_features_and_splits
from waymark.gold_model import build_gold_analytical_model, verify_duckdb_sql_equivalence
from waymark.snapshots import build_lake_snapshot, put_s3_immutable, upload_snapshot, verify_account
from waymark.training import (
    create_sourcedir_tarball,
    launch_sagemaker_training_job,
    parse_training_artifacts,
    run_remote_training,
)

ROOT = Path(__file__).resolve().parents[1]


class MemoryS3:
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    def put_object(self, *, Bucket, Key, Body, **kwargs):
        if kwargs.get("IfNoneMatch") == "*" and Key in self.objects:
            raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
        self.objects[Key] = Body
        return {"VersionId": "synthetic-version"}

    def get_object(self, *, Bucket, Key):
        return {"Body": io.BytesIO(self.objects[Key])}


@pytest.fixture
def synthetic_lake(tmp_path: Path) -> Path:
    (tmp_path / "configs").mkdir()
    (tmp_path / "scripts").mkdir()
    for name in ["wm0_contract.yaml", "t1_discriminating_fixture.json"]:
        shutil.copyfile(ROOT / "configs" / name, tmp_path / "configs" / name)
    shutil.copyfile(ROOT / "scripts/glue_etl_entry.py", tmp_path / "scripts/glue_etl_entry.py")
    contract = load_wm0_contract()
    raw, bronze, silver, gold = [
        tmp_path / "data" / name for name in ["raw/olist", "bronze", "silver", "gold"]
    ]
    write_fixture_to_raw_csvs(ROOT / "configs/t1_discriminating_fixture.json", raw, contract)
    b = build_bronze(raw, bronze, contract)
    s = build_silver(bronze, silver, contract)
    g = build_gold_analytical_model(silver, gold, contract)
    sql = verify_duckdb_sql_equivalence(silver, gold, contract)
    t5 = build_t5_features_and_splits(gold, gold, contract, fixed_global_prior_late_rate=0.25)
    for name, value in [
        ("t3_source_manifest.json", {"bronze_manifest": b, "silver_accounting": s}),
        ("t4_gold_reconciliation.json", {"gold_reconciliation": g, "duckdb_sql_parity": sql}),
        ("t5_wm2_split_manifest.json", t5),
    ]:
        (tmp_path / "configs" / name).write_bytes(canonical_json(value))
    return tmp_path


def test_snapshot_upload_and_independent_glue_verifier(synthetic_lake: Path) -> None:
    snapshot = build_lake_snapshot(synthetic_lake)
    s3 = MemoryS3()
    receipt = upload_snapshot(s3, "synthetic", synthetic_lake, snapshot)
    assert len(receipt["objects"]) == 30
    assert sum("/bronze/" in key for key in s3.objects) == 9
    verifier = runpy.run_path(str(ROOT / "scripts/glue_etl_entry.py"))["verify_snapshot"]
    result = verifier(s3, "synthetic", receipt["manifest"]["key"], receipt["manifest"]["sha256"])
    assert result["verified_objects"] == 30
    key = receipt["objects"][0]["key"]
    s3.objects[key] = b"corrupt"
    with pytest.raises(ValueError, match="object mismatch"):
        verifier(s3, "synthetic", receipt["manifest"]["key"], receipt["manifest"]["sha256"])


def test_missing_local_lake_rejected_before_aws(tmp_path: Path) -> None:
    (tmp_path / "configs").mkdir()
    shutil.copyfile(ROOT / "configs/wm0_contract.yaml", tmp_path / "configs/wm0_contract.yaml")
    with patch("waymark.cloud_slice.boto3.Session") as session:
        with pytest.raises(ValueError, match="missing required inputs"):
            run_c1_orchestration(repo_root=tmp_path)
        session.assert_not_called()


def test_changed_feature_snapshot_rejected(synthetic_lake: Path) -> None:
    path = synthetic_lake / "data/gold/order_features_and_splits.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "item_count"] += 1
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="Feature table hash"):
        build_lake_snapshot(synthetic_lake)


def test_immutable_writes_and_conditional_remote_conflict(tmp_path: Path) -> None:
    path = tmp_path / "accepted.json"
    write_immutable(path, b"old")
    write_immutable(path, b"old")
    with pytest.raises(FileExistsError):
        write_immutable(path, b"new")
    assert path.read_bytes() == b"old"
    s3 = MemoryS3()
    put_s3_immutable(s3, "b", "frozen", b"old")
    put_s3_immutable(s3, "b", "frozen", b"old")
    with pytest.raises(ValueError, match="conflict"):
        put_s3_immutable(s3, "b", "frozen", b"new")
    assert s3.objects["frozen"] == b"old"


def test_account_and_region_are_blocking() -> None:
    sts = Mock()
    sts.get_caller_identity.return_value = {"Account": "000000000000"}
    with pytest.raises(ValueError, match="mismatch"):
        verify_account(sts, "ap-southeast-2", load_wm0_contract())
    sts.get_caller_identity.return_value = {"Account": "149270211548"}
    with pytest.raises(ValueError, match="mismatch"):
        verify_account(sts, "us-east-1", load_wm0_contract())


def test_zero_money_orphan_rejected(synthetic_lake: Path) -> None:
    path = synthetic_lake / "data/bronze/olist_order_items_dataset.parquet"
    items = pd.read_parquet(path)
    orphan = items.iloc[[0]].assign(order_id="orphan", price=0.0, freight_value=0.0)
    pd.concat([items, orphan], ignore_index=True).to_parquet(path, index=False)
    with pytest.raises(ValueError, match="foreign key"):
        build_silver(synthetic_lake / "data/bronze", synthetic_lake / "new_silver", load_wm0_contract())


def test_existing_bronze_is_not_replaced(synthetic_lake: Path) -> None:
    path = synthetic_lake / "data/bronze/olist_orders_dataset.parquet"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError, match="protected"):
        build_bronze(synthetic_lake / "data/raw/olist", synthetic_lake / "data/bronze", load_wm0_contract())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_feature_key_and_value_parity() -> None:
    local = pd.DataFrame({"order_id": ["a", "b"], "feature": [1.0, np.nan]})
    verify_feature_rows(
        local,
        [{"order_id": "b", "feature": None}, {"order_id": "a", "feature": "1.0"}],
        ["order_id", "feature"],
    )
    with pytest.raises(ValueError, match="numeric mismatch"):
        verify_feature_rows(
            local,
            [{"order_id": "a", "feature": "9"}, {"order_id": "b", "feature": None}],
            ["order_id", "feature"],
        )
    with pytest.raises(ValueError, match="key mismatch"):
        verify_feature_rows(
            local,
            [{"order_id": "a", "feature": "1"}, {"order_id": "a", "feature": None}],
            ["order_id", "feature"],
        )


def test_top_k_ties_are_bounded_and_order_independent() -> None:
    ids = np.array([f"o{i:02d}" for i in range(20)])
    selected = top_k_alerts(np.full(20, 0.2), ids)
    assert selected.sum() == 2
    assert set(ids[selected]) == {"o00", "o01"}
    reversed_selection = top_k_alerts(np.full(20, 0.2), ids[::-1])
    assert set(ids[::-1][reversed_selection]) == {"o00", "o01"}


def test_baselines_obey_frozen_fallback_and_smoothing() -> None:
    train = pd.DataFrame(
        {
            "state_corridor": ["sparse"] + ["common"] * 24,
            "y_days": [100.0] + [2.0] * 24,
            "y_late": [1] + [0] * 24,
        }
    )
    target = pd.DataFrame({"state_corridor": ["sparse", "unseen"], "promised_days": [4.0, 4.0]})
    result = baseline_predictions(train, target)
    assert result.loc[0, "b1_p50"] == 2.0
    assert result.loc[0, "b0_late_prob"] == pytest.approx((1 + 5 * 0.04) / 6)
    assert result.loc[1, "b0_late_prob"] == 0.04


def test_rank_metrics_ties_and_known_example() -> None:
    y = np.array([0, 1, 0, 1])
    result = ranking_metrics(y, np.array([0.1, 0.4, 0.35, 0.8]))
    assert result == {"roc_auc": 1.0, "pr_auc": 1.0}
    tied = ranking_metrics(y, np.full(4, 0.5))
    assert tied == {"roc_auc": 0.5, "pr_auc": 0.5}
    assert binary_metrics(y, np.array([0.1, 0.4, 0.35, 0.8]), np.array([0, 0, 0, 1]))["recall_at_tau"] == 0.5


@pytest.mark.parametrize("probability", [np.array([-0.1]), np.array([1.1]), np.array([np.nan])])
def test_invalid_probabilities_fail(probability: np.ndarray) -> None:
    with pytest.raises(ValueError):
        expected_calibration_error(np.array([1]), probability)


def test_interval_validation_and_paired_cluster_units() -> None:
    with pytest.raises(ValueError, match="crossed"):
        eta_metrics(np.array([2.0]), np.array([[3.0, 2.0, 1.0]]))
    result = paired_cluster_interval(np.full(4, -2.0), np.array(["a", "a", "b", "b"]), replicates=20)
    assert result["ci95"] == [-2.0, -2.0]
    assert result["cluster_count"] == 2


def test_usage_pricing_uses_per_query_minimum_and_no_second_spot_discount() -> None:
    assert athena_billable_bytes(1) == 10_000_000
    assert athena_billable_bytes(10_000_001) == 11_000_000
    assert athena_billable_bytes(0, "DDL") == 0
    assert training_cost(100, 1, 0.144, "synthetic dated regional rate")[
        "estimated_cost_usd"
    ] == pytest.approx(0.004)
    assert usage_estimate(1, "DPU_hours", None, "")["estimated_cost_usd"] is None


def test_training_source_is_deterministic_and_legacy_inputs_are_rejected() -> None:
    assert create_sourcedir_tarball(ROOT) == create_sourcedir_tarball(ROOT)
    with patch("waymark.training.boto3.Session") as session:
        with pytest.raises(ValueError, match="receipt"):
            run_remote_training(repo_root=ROOT)
        session.assert_not_called()
    import tarfile

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz"):
        pass
    with pytest.raises(ValueError, match="artifact"):
        parse_training_artifacts(buffer.getvalue(), {})


@pytest.mark.parametrize("tamper_feature", [False, True])
def test_complete_mock_cloud_journey_requires_current_object_and_feature_identity(
    synthetic_lake: Path,
    tamper_feature: bool,
) -> None:
    import duckdb

    from waymark.costs import athena_billable_bytes

    snapshot = build_lake_snapshot(synthetic_lake)
    bucket = "waymark-lake-pilot-149270211548-ap-southeast-2"
    prefix = f"snapshots/{snapshot['snapshot_id']}/"
    s3 = MemoryS3()
    verifier = runpy.run_path(str(ROOT / "scripts/glue_etl_entry.py"))["verify_snapshot"]

    class Glue:
        def get_job(self, **kwargs):
            return {
                "Job": {"Command": {"ScriptLocation": f"s3://{bucket}/{prefix}scripts/glue_etl_entry.py"}}
            }

        def start_job_run(self, **kwargs):
            args = kwargs["Arguments"]
            verifier(s3, bucket, args["--WAYMARK_SNAPSHOT_KEY"], args["--WAYMARK_SNAPSHOT_SHA256"])
            return {"JobRunId": "mock-verified-run"}

        def get_job_run(self, **kwargs):
            return {"JobRun": {"JobRunState": "SUCCEEDED", "ExecutionTime": 1, "MaxCapacity": 0.0625}}

        def get_table(self, *, Name, **kwargs):
            layer = "silver" if Name.startswith("silver_") else "gold"
            return {"Table": {"StorageDescriptor": {"Location": f"s3://{bucket}/{prefix}{layer}/{Name}/"}}}

    class QueryRunner:
        def __init__(self, *args):
            self.database = "default"
            self.total_bytes_scanned = 0
            self.query_statistics = []
            self.connection = duckdb.connect(":memory:")
            for item in snapshot["objects"]:
                if item["path"].startswith(("silver/", "gold/")):
                    frame = pd.read_parquet(io.BytesIO(s3.objects[prefix + item["path"]]))
                    self.connection.register(Path(item["path"]).stem, frame)

        def run_query(self, query):
            if query.startswith("CREATE "):
                self.query_statistics.append({"billable_bytes": 0})
                return [], 0
            self.total_bytes_scanned += 100
            self.query_statistics.append({"billable_bytes": athena_billable_bytes(100)})
            frame = self.connection.execute(query).df()
            integer_keys = {"row_count", "sum_items", "total_orders", "eligible_orders", "split_orders"}
            rows = [
                {
                    key: None if pd.isna(value) else str(int(value)) if key in integer_keys else str(value)
                    for key, value in row.items()
                }
                for row in frame.to_dict("records")
            ]
            if tamper_feature and query.startswith("SELECT order_id, split_role,"):
                rows[0]["item_count"] = "9999"
            return rows, 100

    sts = Mock()
    sts.get_caller_identity.return_value = {"Account": "149270211548"}
    session = Mock()
    session.client.side_effect = lambda name: {"sts": sts, "s3": s3, "glue": Glue(), "athena": Mock()}[name]
    with (
        patch("waymark.cloud_slice.boto3.Session", return_value=session),
        patch("waymark.cloud_slice.AthenaQueryRunner", QueryRunner),
    ):
        if tamper_feature:
            with pytest.raises(ValueError, match="numeric mismatch"):
                run_c1_orchestration(repo_root=synthetic_lake)
            assert not any("/acceptance/" in key for key in s3.objects)
        else:
            report = run_c1_orchestration(repo_root=synthetic_lake)
            assert report["gate"] == "WM1_VERIFIED"
            assert len(report["tables_registered"]) == 15
            assert report["athena_reconciliation"]["athena_cost"]["estimated_cost_usd"] is None
            assert report["glue_job_execution"]["cost"]["quantity"] == pytest.approx(60 * 0.0625 / 3600)
            assert s3.objects[prefix + "acceptance/mock-verified-run.json"] == canonical_json(report)


@pytest.mark.parametrize("terminal", ["Failed", "CompletedWithoutArtifacts"])
def test_mocked_launcher_validates_request_and_preserves_failed_attempt(
    synthetic_lake: Path,
    terminal: str,
) -> None:
    from botocore.session import get_session
    from botocore.validate import ParamValidator

    from waymark.artifacts import sha256_file
    from waymark.training_environment import requirement_versions

    shutil.copytree(ROOT / "src", synthetic_lake / "src", ignore=shutil.ignore_patterns("__pycache__"))
    snapshot = build_lake_snapshot(synthetic_lake)
    s3 = MemoryS3()
    upload_snapshot(s3, "bucket", synthetic_lake, snapshot)
    prefix = f"snapshots/{snapshot['snapshot_id']}/"
    cloud = {
        "schema": "waymark.c1_reconciliation/v2",
        "gate": "WM1_VERIFIED",
        "snapshot": snapshot,
        "glue_job_execution": {"job_run_id": "accepted"},
    }
    s3.objects[prefix + "acceptance/accepted.json"] = canonical_json(cloud)
    requirements = synthetic_lake / "src/waymark/remote_requirements.txt"
    environment = {
        "schema": "waymark.environment/v1",
        "status": "PASS",
        "python": "3.12.13",
        "image_uri": "149270211548.dkr.ecr.ap-southeast-2.amazonaws.com/test@sha256:" + "a" * 64,
        "versions": requirement_versions(requirements),
        "requirements_sha256": sha256_file(requirements),
        "check": "installation_and_imports_only_no_fit",
    }
    sm = Mock()

    def describe(**kwargs):
        job = kwargs["TrainingJobName"]
        if terminal == "Failed":
            return {"TrainingJobStatus": "Failed", "FailureReason": "injected failure"}
        buffer = io.BytesIO()
        import tarfile

        with tarfile.open(fileobj=buffer, mode="w:gz"):
            pass
        key = f"models/{job}/output/model.tar.gz"
        s3.objects[key] = buffer.getvalue()
        return {
            "TrainingJobStatus": "Completed",
            "ModelArtifacts": {"S3ModelArtifacts": f"s3://bucket/{key}"},
        }

    sm.describe_training_job.side_effect = describe
    with pytest.raises((ValueError, RuntimeError)):
        launch_sagemaker_training_job(
            sm,
            s3,
            "bucket",
            "arn:aws:iam::149270211548:role/test",
            synthetic_lake,
            cloud_receipt=cloud,
            environment_receipt=environment,
        )
    request = sm.create_training_job.call_args.kwargs
    shape = get_session().get_service_model("sagemaker").operation_model("CreateTrainingJob").input_shape
    errors = ParamValidator().validate(request, shape)
    assert not errors.has_errors(), errors.generate_report()
    assert "CheckpointConfig" in request
    assert "allow-holdout" not in request["HyperParameters"]
    assert "@sha256:" in request["AlgorithmSpecification"]["TrainingImage"]
    attempt = synthetic_lake / "data/training_runs" / request["TrainingJobName"]
    assert (attempt / "failure.json").is_file()
    assert not (attempt / "result.json").exists()
