"""Unit tests for the frozen WM0 contract and synthetic relational oracle fixture."""

from __future__ import annotations

import math

from waymark.cli import main
from waymark.contracts import haversine_km, load_discriminating_fixture, load_wm0_contract


def test_wm0_contract_validates() -> None:
    contract = load_wm0_contract("configs/wm0_contract.yaml")
    assert contract.contract_id == "wm0_v1"
    assert contract.o3_prediction_and_targets["prediction_cutoff_t0"] == "order_approved_at"
    assert contract.o7_aws_and_cost.account_id == "149270211548"
    assert contract.o7_aws_and_cost.cli_profile == "waymark"
    assert contract.o7_aws_and_cost.region == "ap-southeast-2"
    assert contract.o7_aws_and_cost.total_budget_cap_usd == 30.0
    assert contract.o7_aws_and_cost.stage_cost_ceiling_usd.total_planned_ceiling_usd == 8.5


def test_discriminating_fixture_oracle_assertions() -> None:
    fixture = load_discriminating_fixture("configs/t1_discriminating_fixture.json")
    assert fixture["fixture_id"] == "t1_discriminating_fixture_v1"
    tables = fixture["tables"]
    assert len(tables) == 9
    assert len(tables["olist_orders_dataset"]) == 8

    reconcil = fixture["expected_reconciliation"]
    items = tables["olist_order_items_dataset"]
    payments = tables["olist_order_payments_dataset"]
    assert math.isclose(sum(r["price"] for r in items), reconcil["sum_item_price_brl"], abs_tol=1e-6)
    assert math.isclose(
        sum(r["freight_value"] for r in items), reconcil["sum_freight_value_brl"], abs_tol=1e-6
    )
    assert math.isclose(
        sum(r["payment_value"] for r in payments), reconcil["sum_payment_value_brl"], abs_tol=1e-6
    )

    ord1 = fixture["expected_oracle_by_order"]["O_TARGET_1"]
    dist = haversine_km(
        reconcil["zip_01001_median_lat"],
        reconcil["zip_01001_median_lng"],
        -22.9068,
        -43.1729,
    )
    assert math.isclose(
        dist,
        ord1["primary_seller_haversine_km"],
        abs_tol=ord1["primary_seller_haversine_tol_km"],
    )


def test_cli_verify_wm0(capsys) -> None:
    rc = main(["verify-wm0"])
    assert rc == 0
    captured = capsys.readouterr()
    assert '"status": "OK"' in captured.out
