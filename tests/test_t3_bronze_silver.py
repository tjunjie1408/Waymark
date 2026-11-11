"""Unit and integration tests for T3 Bronze and Silver reconstruction (`src/waymark/bronze_silver.py`)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from waymark.bronze_silver import (
    build_bronze,
    build_silver,
    write_fixture_to_raw_csvs,
)
from waymark.contracts import load_discriminating_fixture, load_wm0_contract


def test_t3_bronze_and_silver_on_synthetic_fixture(tmp_path: Path) -> None:
    """Reconstruct Bronze and Silver from the T1 8-order synthetic fixture and verify oracle invariants."""
    contract = load_wm0_contract("configs/wm0_contract.yaml")
    fixture_path = Path("configs/t1_discriminating_fixture.json")
    fixture = load_discriminating_fixture(fixture_path)
    recon = fixture["expected_reconciliation"]
    oracle_by_order = fixture["expected_oracle_by_order"]

    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"

    write_fixture_to_raw_csvs(fixture_path, raw_dir, contract=contract)
    bronze_manifest = build_bronze(raw_dir, bronze_dir, contract)
    assert bronze_manifest["table_count"] == 9
    assert (bronze_dir / "bronze_manifest.json").exists()

    silver_acc = build_silver(bronze_dir, silver_dir, contract)
    assert (silver_dir / "silver_accounting.json").exists()

    # Check geolocation deduplication and Brazil bounding-box filtering against T1 oracle
    assert silver_acc["geolocation"]["raw_row_count"] == recon["raw_geolocation_rows"]
    assert (
        silver_acc["geolocation"]["deduped_row_count"]
        == recon["silver_geolocation_after_exact_dedup"]
    )
    assert (
        silver_acc["geolocation"]["in_bounds_row_count"]
        == recon["silver_geolocation_in_bounds_rows"]
    )
    assert (
        silver_acc["geolocation"]["distinct_in_bounds_zip_prefix_count"]
        == recon["gold_distinct_valid_zip_prefixes"]
    )

    geo_zip = pd.read_parquet(silver_dir / "silver_geolocation_zip.parquet").set_index(
        "geolocation_zip_code_prefix"
    )
    assert float(geo_zip.loc["01001", "geolocation_lat"]) == pytest.approx(
        recon["zip_01001_median_lat"]
    )
    assert float(geo_zip.loc["01001", "geolocation_lng"]) == pytest.approx(
        recon["zip_01001_median_lng"]
    )

    # Check order eligibility codes against T1 oracle
    orders_silver = pd.read_parquet(silver_dir / "silver_orders.parquet").set_index(
        "order_id"
    )
    for order_id, order_oracle in oracle_by_order.items():
        assert (
            orders_silver.loc[order_id, "eligibility_code"]
            == order_oracle["eligibility_code"]
        )

    # Check product spelling normalization and zero-dimension flag on P_BADDIM
    products_silver = pd.read_parquet(silver_dir / "silver_products.parquet").set_index(
        "product_id"
    )
    assert "product_name_length" in products_silver.columns
    assert "product_description_length" in products_silver.columns
    assert int(products_silver.loc["P_BADDIM", "dim_missing_flag"]) == 1
    assert float(products_silver.loc["P_BADDIM", "product_volume_cm3"]) == 0.0
    assert int(products_silver.loc["P_STD", "dim_missing_flag"]) == 0
    assert float(products_silver.loc["P_STD", "product_volume_cm3"]) == 2000.0


def test_t3_rejects_corrupted_columns_and_duplicate_keys(tmp_path: Path) -> None:
    """Verify Bronze schema gate rejects missing columns and Silver gate rejects duplicate keys."""
    contract = load_wm0_contract("configs/wm0_contract.yaml")
    fixture_path = Path("configs/t1_discriminating_fixture.json")

    raw_dir = tmp_path / "raw"
    bronze_dir = tmp_path / "bronze"
    silver_dir = tmp_path / "silver"
    write_fixture_to_raw_csvs(fixture_path, raw_dir, contract=contract)

    # Corrupt a column header in olist_sellers_dataset.csv
    sellers_csv = raw_dir / "olist_sellers_dataset.csv"
    df_sellers = pd.read_csv(sellers_csv)
    df_sellers.rename(columns={"seller_state": "corrupted_col"}).to_csv(
        sellers_csv, index=False
    )
    with pytest.raises(ValueError, match="Column mismatch"):
        build_bronze(raw_dir, bronze_dir, contract)

    # Restore valid CSVs, build Bronze, then inject a duplicate order_id in Bronze orders
    write_fixture_to_raw_csvs(fixture_path, raw_dir, contract=contract)
    build_bronze(raw_dir, bronze_dir, contract)
    orders_pq = bronze_dir / "olist_orders_dataset.parquet"
    df_orders = pd.read_parquet(orders_pq)
    df_orders_dup = pd.concat([df_orders, df_orders.iloc[[0]]], ignore_index=True)
    df_orders_dup.to_parquet(orders_pq, index=False)

    with pytest.raises(ValueError, match="duplicate or null order_id"):
        build_silver(bronze_dir, silver_dir, contract)


def test_t3_source_manifest_reconciliation() -> None:
    """Verify the exported T3 source manifest and Silver accounting invariants."""
    manifest_path = Path("configs/t3_source_manifest.json")
    assert manifest_path.exists()
    data = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert data["status"] == "OK"
    assert data["stage"] == "T3"
    assert data["bronze_manifest"]["table_count"] == 9
    assert data["bronze_manifest"]["total_rows"] == 1550922

    orders_acc = data["silver_accounting"]["orders"]
    assert orders_acc["raw_row_count"] == 99441
    assert (
        orders_acc["eligible_mature_count"] + orders_acc["excluded_count"]
        == orders_acc["raw_row_count"]
    )
    assert sum(orders_acc["eligibility_breakdown"].values()) == 99441

    geo_acc = data["silver_accounting"]["geolocation"]
    assert (
        geo_acc["raw_row_count"] - geo_acc["exact_duplicate_rows_dropped"]
        == geo_acc["deduped_row_count"]
    )
    assert (
        geo_acc["deduped_row_count"] - geo_acc["out_of_bounds_rows_dropped"]
        == geo_acc["in_bounds_row_count"]
    )
