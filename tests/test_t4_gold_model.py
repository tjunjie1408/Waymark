"""Tests for T4 Gold analytical model and DuckDB SQL parity (`src/waymark/gold_model.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from waymark.bronze_silver import (
    build_bronze,
    build_silver,
    write_fixture_to_raw_csvs,
)
from waymark.contracts import load_discriminating_fixture, load_wm0_contract
from waymark.gold_model import build_gold_analytical_model, verify_duckdb_sql_equivalence


def test_t4_gold_model_and_duckdb_parity_on_synthetic_fixture(tmp_path: Path) -> None:
    """Verify Gold fact_order_fulfillment and DuckDB SQL parity against the T1 8-order oracle."""
    contract = load_wm0_contract("configs/wm0_contract.yaml")
    fixture_path = Path("configs/t1_discriminating_fixture.json")
    fixture = load_discriminating_fixture(fixture_path)
    recon = fixture["expected_reconciliation"]
    oracle_by_order = fixture["expected_oracle_by_order"]

    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"

    write_fixture_to_raw_csvs(fixture_path, raw_dir, contract=contract)
    build_bronze(raw_dir, bronze_dir, contract)
    build_silver(bronze_dir, silver_dir, contract)

    gold_recon = build_gold_analytical_model(silver_dir, gold_dir, contract)
    assert gold_recon["grain_check"]["one_row_per_order_id"] is True
    assert gold_recon["grain_check"]["gold_fact_rows"] == recon["fact_orders_count"]
    assert gold_recon["monetary_and_count_reconciliation"][
        "gold_sum_item_price_brl"
    ] == pytest.approx(recon["sum_item_price_brl"])
    assert gold_recon["monetary_and_count_reconciliation"][
        "gold_sum_freight_brl"
    ] == pytest.approx(recon["sum_freight_value_brl"])
    assert gold_recon["monetary_and_count_reconciliation"][
        "gold_sum_payment_value_brl"
    ] == pytest.approx(recon["sum_payment_value_brl"])

    # Verify independent DuckDB SQL query parity over the fixture
    sql_parity = verify_duckdb_sql_equivalence(silver_dir, gold_dir, contract)
    assert sql_parity["status"] == "OK"
    assert sql_parity["compared_rows"] == 8

    # Check exact order-level oracle values for O_TARGET_1 and O_MULTI_2
    fact = pd.read_parquet(gold_dir / "fact_order_fulfillment.parquet").set_index(
        "order_id"
    )

    t1 = oracle_by_order["O_TARGET_1"]
    row_t1 = fact.loc["O_TARGET_1"]
    assert row_t1["eligibility_code"] == t1["eligibility_code"]
    assert float(row_t1["y_days"]) == pytest.approx(t1["y_days"])
    assert float(row_t1["promised_days"]) == pytest.approx(t1["promised_days"])
    assert int(row_t1["y_late"]) == t1["y_late"]
    assert int(row_t1["item_count"]) == t1["item_count"]
    assert int(row_t1["distinct_seller_count"]) == t1["distinct_seller_count"]
    assert bool(row_t1["is_multi_seller"]) is t1["is_multi_seller"]
    assert row_t1["primary_seller_id"] == t1["primary_seller_id"]
    assert float(row_t1["total_item_price_brl"]) == pytest.approx(
        t1["total_item_price_brl"]
    )
    assert float(row_t1["total_freight_brl"]) == pytest.approx(t1["total_freight_brl"])
    assert float(row_t1["freight_to_price_ratio"]) == pytest.approx(
        t1["freight_to_price_ratio"]
    )
    assert float(row_t1["total_weight_g"]) == pytest.approx(t1["total_weight_g"])
    assert float(row_t1["total_volume_cm3"]) == pytest.approx(t1["total_volume_cm3"])
    assert float(row_t1["volumetric_density_g_per_cm3"]) == pytest.approx(
        t1["volumetric_density_g_per_cm3"]
    )
    assert int(row_t1["dim_missing_flag"]) == t1["dim_missing_flag"]
    assert int(row_t1["geo_missing_flag"]) == t1["geo_missing_flag"]
    assert int(row_t1["is_interstate"]) == t1["is_interstate"]
    assert float(row_t1["primary_seller_haversine_km"]) == pytest.approx(
        t1["primary_seller_haversine_km"], abs=t1["primary_seller_haversine_tol_km"]
    )
    assert float(row_t1["approval_lag_hours"]) == pytest.approx(
        t1["approval_lag_hours"]
    )
    assert float(row_t1["shipping_limit_days_from_approval"]) == pytest.approx(
        t1["shipping_limit_days_from_approval"]
    )

    m2 = oracle_by_order["O_MULTI_2"]
    row_m2 = fact.loc["O_MULTI_2"]
    assert row_m2["eligibility_code"] == m2["eligibility_code"]
    assert float(row_m2["y_days"]) == pytest.approx(m2["y_days"])
    assert float(row_m2["promised_days"]) == pytest.approx(m2["promised_days"])
    assert int(row_m2["y_late"]) == m2["y_late"]
    assert int(row_m2["item_count"]) == m2["item_count"]
    assert int(row_m2["distinct_seller_count"]) == m2["distinct_seller_count"]
    assert bool(row_m2["is_multi_seller"]) is m2["is_multi_seller"]
    assert row_m2["primary_seller_id"] == m2["primary_seller_id"]
    assert float(row_m2["total_item_price_brl"]) == pytest.approx(
        m2["total_item_price_brl"]
    )
    assert float(row_m2["total_freight_brl"]) == pytest.approx(m2["total_freight_brl"])
    assert float(row_m2["freight_to_price_ratio"]) == pytest.approx(
        m2["freight_to_price_ratio"]
    )
    assert float(row_m2["total_weight_g"]) == pytest.approx(m2["total_weight_g"])
    assert np.isnan(row_m2["total_volume_cm3"])
    assert np.isnan(row_m2["volumetric_density_g_per_cm3"])
    assert int(row_m2["dim_missing_flag"]) == m2["dim_missing_flag"]
    assert int(row_m2["geo_missing_flag"]) == m2["geo_missing_flag"]
    assert row_m2["primary_payment_type"] == m2["primary_payment_type"]
    assert int(row_m2["max_payment_installments"]) == m2["max_payment_installments"]
    assert float(row_m2["total_payment_value_brl"]) == pytest.approx(
        m2["total_payment_value_brl"]
    )


def test_t4_exported_reconciliation_summary() -> None:
    """Verify the exported T4 Gold reconciliation and DuckDB SQL parity summary file."""
    path = Path("configs/t4_gold_reconciliation.json")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["status"] == "OK"
    assert data["stage"] == "T4"
    assert data["gold_reconciliation"]["grain_check"]["one_row_per_order_id"] is True
    assert data["gold_reconciliation"]["grain_check"]["gold_fact_rows"] == 99441
    assert data["duckdb_sql_parity"]["status"] == "OK"
    assert data["duckdb_sql_parity"]["compared_rows"] == 99441
