"""Wire-format and committed-read counterexamples; no fitting or AWS calls."""

from __future__ import annotations

import json

import pandas as pd
import pytest
from batch_fixtures import Storage, rows

from waymark.batch_adapter import (
    REQUIRED_PREDICTION_COLUMNS,
    LogicalBatchSpec,
    staging_output_prefix,
    success_marker_key,
    validate_and_publish_staging_predictions,
)


def publish(client, spec, body, **kwargs):
    key = staging_output_prefix(spec.compute_logical_batch_id(), 1) + "part.csv.out"
    client.objects[key] = body
    return validate_and_publish_staging_predictions(
        client=client,
        bucket="test",
        spec=spec,
        attempt_seq=1,
        staging_key=key,
        expected_row_count=1,
        observed_row_count=1,
        observed_columns=REQUIRED_PREDICTION_COLUMNS,
        expected_order_ids=["001"],
        **kwargs,
    )


def test_real_jsonlines_are_normalized_to_committed_parquet():
    client, spec = Storage(), LogicalBatchSpec("model", "input", "2018-08-29")
    result = publish(client, spec, (json.dumps(rows()[0]) + "\n").encode(), wire_format="jsonlines")
    import io

    frame = pd.read_parquet(io.BytesIO(client.objects[result["committed_gold_uri"].split("/", 3)[3]]))
    assert frame.order_id.tolist() == ["001"]
    assert frame.eta_p50_days.tolist() == [2.0]


def test_same_count_wrong_order_cannot_publish():
    client, spec = Storage(), LogicalBatchSpec("model", "input", "2018-08-29")
    data = rows()
    data[0]["order_id"] = "other"
    with pytest.raises(ValueError, match="order"):
        publish(client, spec, pd.DataFrame(data).to_parquet(index=False))
    assert success_marker_key(spec.compute_logical_batch_id()) not in client.objects


def test_record_order_can_change_without_losing_explicit_order_mapping():
    import io

    client, spec = Storage(), LogicalBatchSpec("model", "input", "2018-08-29")
    first, second = rows()[0], rows()[0]
    second.update(order_id="002", eta_p10_days=7.0, eta_p50_days=8.0, eta_p90_days=9.0)
    key = staging_output_prefix(spec.compute_logical_batch_id(), 1) + "part.csv.out"
    client.objects[key] = (json.dumps(second) + "\n" + json.dumps(first)).encode()
    result = validate_and_publish_staging_predictions(
        client=client,
        bucket="test",
        spec=spec,
        attempt_seq=1,
        staging_key=key,
        expected_row_count=2,
        observed_row_count=2,
        observed_columns=REQUIRED_PREDICTION_COLUMNS,
        expected_order_ids=["001", "002"],
        wire_format="jsonlines",
    )
    frame = pd.read_parquet(io.BytesIO(client.objects[result["committed_gold_uri"].split("/", 3)[3]]))
    assert frame.set_index("order_id").eta_p50_days.to_dict() == {"001": 2.0, "002": 8.0}


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, "0.2"])
def test_invalid_json_numeric_types_cannot_publish(value):
    client, spec = Storage(), LogicalBatchSpec("model", "input", "2018-08-29")
    data = rows()[0]
    data["prob_late"] = value
    with pytest.raises(ValueError):
        publish(client, spec, json.dumps(data).encode(), wire_format="jsonlines")
    assert success_marker_key(spec.compute_logical_batch_id()) not in client.objects
