"""Tests for T5 point-in-time feature builder, maturity-purged splits, and WM2 leakage gates."""

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
from waymark.features_splits import (
    FEATURE_COLUMNS,
    assert_approval_time_feature_safety,
    assign_chronological_splits_with_maturity_purge,
    compute_seller_rolling_30d_features,
)
from waymark.gold_model import build_gold_analytical_model


def test_t5_point_in_time_seller_features_on_synthetic_fixture(tmp_path: Path) -> None:
    """Verify rolling 30d seller features against the T1 discriminating fixture oracle."""
    contract = load_wm0_contract("configs/wm0_contract.yaml")
    fixture_path = Path("configs/t1_discriminating_fixture.json")
    fixture = load_discriminating_fixture(fixture_path)
    oracle_by_order = fixture["expected_oracle_by_order"]
    prior_rate = float(fixture["global_prior_late_rate_at_fixture"])

    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"

    write_fixture_to_raw_csvs(fixture_path, raw_dir, contract=contract)
    build_bronze(raw_dir, bronze_dir, contract)
    build_silver(bronze_dir, silver_dir, contract)
    build_gold_analytical_model(silver_dir, gold_dir, contract)

    fact = pd.read_parquet(gold_dir / "fact_order_fulfillment.parquet")
    feat = compute_seller_rolling_30d_features(
        fact_df=fact,
        window_days=30,
        prior_weight_m=float(fixture["prior_smoothing_weight_m"]),
        min_mature_orders=3,
        fixed_global_prior_late_rate=prior_rate,
    ).set_index("order_id")

    # O_TARGET_1: S_VET has 4 prior approved orders in 30d (including O_HIST_IMMATURE),
    # but only 3 mature delivered orders by 2018-01-15T12:00:00 (O_HIST_IMMATURE delivered 2018-01-25)
    t1_oracle = oracle_by_order["O_TARGET_1"]
    r_t1 = feat.loc["O_TARGET_1"]
    assert int(r_t1["congestion_order_count_30d"]) == t1_oracle["congestion_order_count_30d"]
    assert int(r_t1["seller_delivered_count_30d"]) == t1_oracle["seller_delivered_count_30d"]
    assert int(r_t1["seller_late_count_30d"]) == t1_oracle["seller_late_count_30d"]
    assert int(r_t1["seller_cold_start_flag"]) == t1_oracle["seller_cold_start_flag"]
    assert float(r_t1["seller_late_rate_30d_smoothed"]) == pytest.approx(
        t1_oracle["seller_late_rate_30d_smoothed"]
    )
    assert float(r_t1["seller_median_eta_days_30d"]) == pytest.approx(
        t1_oracle["seller_median_eta_days_30d"]
    )

    # O_MULTI_2: primary seller S_COLD has 0 prior orders -> cold-start fallback to global prior (0.25)
    m2_oracle = oracle_by_order["O_MULTI_2"]
    r_m2 = feat.loc["O_MULTI_2"]
    assert int(r_m2["congestion_order_count_30d"]) == m2_oracle["congestion_order_count_30d"]
    assert int(r_m2["seller_delivered_count_30d"]) == m2_oracle["seller_delivered_count_30d"]
    assert int(r_m2["seller_cold_start_flag"]) == m2_oracle["seller_cold_start_flag"]
    assert float(r_m2["seller_late_rate_30d_smoothed"]) == pytest.approx(
        m2_oracle["seller_late_rate_30d_smoothed"]
    )
    assert np.isnan(r_m2["seller_median_eta_days_30d"])


def test_t5_leakage_gate_rejects_forbidden_columns_and_immature_labels(
    tmp_path: Path,
) -> None:
    """Verify assert_approval_time_feature_safety rejects post-cutoff columns and immature split rows."""
    contract = load_wm0_contract("configs/wm0_contract.yaml")
    fixture_path = Path("configs/t1_discriminating_fixture.json")

    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    gold_dir = tmp_path / "gold"

    write_fixture_to_raw_csvs(fixture_path, raw_dir, contract=contract)
    build_bronze(raw_dir, bronze_dir, contract)
    build_silver(bronze_dir, silver_dir, contract)
    build_gold_analytical_model(silver_dir, gold_dir, contract)

    fact = pd.read_parquet(gold_dir / "fact_order_fulfillment.parquet")
    feat = compute_seller_rolling_30d_features(fact, fixed_global_prior_late_rate=0.25)
    split_df, split_summary = assign_chronological_splits_with_maturity_purge(
        feat, contract
    )

    # Clean feature matrix and maturity-purged splits pass
    assert_approval_time_feature_safety(split_df, FEATURE_COLUMNS, split_summary)

    # 1. Injecting forbidden post-cutoff column (order_delivered_carrier_date) must fail
    with pytest.raises(ValueError, match="Leakage gate violation: forbidden post-cutoff"):
        assert_approval_time_feature_safety(
            split_df,
            [*FEATURE_COLUMNS, "order_delivered_carrier_date"],
            split_summary,
        )

    # 2. Injecting an immature delivery timestamp into a 'train' row must fail
    tampered = split_df.copy()
    train_idx = tampered.index[tampered["split_role"] == "train"][0]
    tampered.loc[train_idx, "order_delivered_customer_date"] = pd.Timestamp(
        "2019-01-01T00:00:00"
    )
    with pytest.raises(
        ValueError, match="train split contains orders delivered after T_train_end"
    ):
        assert_approval_time_feature_safety(tampered, FEATURE_COLUMNS, split_summary)


def test_t5_exported_wm2_split_manifest() -> None:
    """Verify the exported T5 / WM2 feature and split manifest on the full Olist dataset."""
    path = Path("configs/t5_wm2_split_manifest.json")
    assert path.exists()
    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["status"] == "OK"
    assert data["stage"] == "T5"
    assert data["gate"] == "WM2_VERIFIED_LOCALLY"
    assert data["feature_column_count"] == 34

    counts = data["split_summary"]["counts"]
    assert counts["total_orders"] == 99441
    assert counts["eligible_mature_orders"] == 96395
    assert (
        counts["train_window_raw"]
        + counts["val_calibration_window_raw"]
        + counts["holdout_test_orders"]
        == 96395
    )
    assert (
        counts["train_mature_retained"] + counts["train_purged_immature_at_boundary"]
        == counts["train_window_raw"]
    )
    assert (
        counts["val_calibration_mature_retained"]
        + counts["val_purged_immature_at_boundary"]
        == counts["val_calibration_window_raw"]
    )
