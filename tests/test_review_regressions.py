"""No-fit counterexamples from the 2026-11-11 review."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from waymark.cloud_slice import execute_athena_reconciliation, run_glue_job_verification
from waymark.features_splits import compute_seller_rolling_30d_features
from waymark.sagemaker_entry import expected_calibration_error

ROOT = Path(__file__).resolve().parents[1]


class ReceiptRunner:
    """Return recorded query rows, with deliberate independently specified corruption."""

    total_bytes_scanned = 0
    query_statistics: list = []

    def __init__(self, wrong_counts: bool = False) -> None:
        self.receipt = json.loads((ROOT / "configs/c1_cloud_reconciliation.json").read_text())[
            "athena_reconciliation"
        ]
        self.wrong_counts = wrong_counts

    def run_query(self, query: str) -> tuple[list[dict], int]:
        q = " ".join(query.split())
        r = self.receipt
        if q.startswith("SELECT count(*) as row_count FROM "):
            value = 0 if self.wrong_counts else r["table_row_counts"][q.split()[-1]]
            return [{"row_count": value}], 0
        if "sum(item_count)" in q:
            m = r["athena_monetary_summary"]
            return [
                {
                    "total_orders": m["total_orders"],
                    "sum_items": m["sum_items"],
                    "sum_price": m["sum_price_brl"],
                    "sum_freight": m["sum_freight_brl"],
                    "sum_payment": m["sum_payment_brl"],
                }
            ], 0
        if "GROUP BY split_role" in q:
            return [{"split_role": k, "split_orders": v} for k, v in r["athena_splits_summary"].items()], 0
        if "GROUP BY state_corridor" in q:
            return r["athena_top_5_corridors"], 0
        raise AssertionError(f"Unexpected query: {q}")


def test_ece_includes_certain_wrong_predictions() -> None:
    assert expected_calibration_error(np.array([0, 0]), np.array([1.0, 1.0])) == 1.0


@pytest.mark.parametrize("state", ["ERROR", "EXPIRED"])
def test_glue_terminal_errors_do_not_poll_again(state: str) -> None:
    client = Mock()
    client.start_job_run.return_value = {"JobRunId": "review-error"}
    client.get_job_run.return_value = {"JobRun": {"JobRunState": state}}
    with patch("waymark.cloud_slice.time.sleep", side_effect=AssertionError("extra poll")):
        with pytest.raises(RuntimeError, match=state):
            run_glue_job_verification(client, "review-job")


def test_glue_uses_fractional_capacity() -> None:
    client = Mock()
    client.start_job_run.return_value = {"JobRunId": "review-capacity"}
    client.get_job_run.return_value = {
        "JobRun": {
            "JobRunState": "SUCCEEDED",
            "ExecutionTime": 16,
            "AllocatedCapacity": 0,
            "MaxCapacity": 0.0625,
        }
    }
    result = run_glue_job_verification(client, "review-job")
    assert result["allocated_dpu"] == 0.0625


def test_parity_uses_supplied_local_money_reference() -> None:
    t4 = json.loads((ROOT / "configs/t4_gold_reconciliation.json").read_text())
    t5 = json.loads((ROOT / "configs/t5_wm2_split_manifest.json").read_text())
    t4["gold_reconciliation"]["monetary_and_count_reconciliation"]["gold_sum_item_price_brl"] = -1
    with pytest.raises(ValueError, match="price"):
        execute_athena_reconciliation(ReceiptRunner(), t4, t5)


def test_gate_validation_survives_python_optimization() -> None:
    namespace = {"__name__": "waymark.cloud_slice", "__package__": "waymark"}
    source = (ROOT / "src/waymark/cloud_slice.py").read_text()
    exec(compile(source, "cloud_slice.py", "exec", optimize=2), namespace)
    t4 = json.loads((ROOT / "configs/t4_gold_reconciliation.json").read_text())
    t5 = json.loads((ROOT / "configs/t5_wm2_split_manifest.json").read_text())
    with pytest.raises(ValueError, match="row count"):
        namespace["execute_athena_reconciliation"](ReceiptRunner(wrong_counts=True), t4, t5)


def test_rolling_history_preserves_nonpositional_index() -> None:
    frame = pd.DataFrame(
        {
            "order_id": ["a", "b"],
            "primary_seller_id": ["s", "s"],
            "order_approved_at": pd.to_datetime(["2018-01-01", "2018-01-03"]),
            "order_delivered_customer_date": pd.to_datetime(["2018-01-02", "2018-01-04"]),
            "is_eligible_mature": [1, 1],
            "y_days": [1.0, 1.0],
            "y_late": [0.0, 1.0],
        },
        index=[4, 9],
    )
    original = copy.deepcopy(frame)
    actual = compute_seller_rolling_30d_features(frame, fixed_global_prior_late_rate=0.25)
    assert list(actual.index) == [4, 9]
    assert actual.loc[9, "seller_delivered_count_30d"] == 1
    pd.testing.assert_frame_equal(frame, original)
