"""T3 Bronze and Silver deterministic reconstruction and accounting for Waymark.

Supports both:
1. Full raw Olist CSV directory (`data/raw/olist/*.csv`)
2. Synthetic discriminating fixture (`configs/t1_discriminating_fixture.json`)
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from waymark.contracts import WM0Contract, load_discriminating_fixture, load_wm0_contract

TIMESTAMP_COLUMNS: dict[str, list[str]] = {
    "olist_orders_dataset": [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ],
    "olist_order_items_dataset": [
        "shipping_limit_date",
    ],
    "olist_order_reviews_dataset": [
        "review_creation_date",
        "review_answer_timestamp",
    ],
}


def compute_sha256(path: Path) -> str:
    """Compute hex SHA-256 digest of a file."""
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_fixture_to_raw_csvs(
    fixture_path: Path,
    raw_dir: Path,
    contract: WM0Contract | None = None,
) -> dict[str, Path]:
    """Materialize the T1 synthetic JSON fixture as 9 canonical CSV files for T3 testing."""
    if contract is None:
        contract = load_wm0_contract()
    fixture = load_discriminating_fixture(fixture_path)
    raw_dir.mkdir(parents=True, exist_ok=True)
    expected_tables = contract.o1_source["expected_tables"]
    written: dict[str, Path] = {}
    for table_key, spec in expected_tables.items():
        rows = fixture["tables"][table_key]
        csv_path = raw_dir / spec["filename"]
        df = pd.DataFrame(rows, columns=spec["columns"])
        df.to_csv(csv_path, index=False)
        written[table_key] = csv_path
    return written


def build_bronze(
    raw_dir: Path,
    bronze_dir: Path,
    contract: WM0Contract,
) -> dict[str, Any]:
    """Validate raw CSVs against O1 contract, write Bronze Parquet tables, and emit manifest."""
    bronze_dir.mkdir(parents=True, exist_ok=True)
    tables_manifest: dict[str, Any] = {}
    total_rows = 0
    total_bytes = 0

    expected_tables = contract.o1_source["expected_tables"]
    for table_key, spec in expected_tables.items():
        filename = spec["filename"]
        expected_cols = spec["columns"]
        csv_path = raw_dir / filename
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing required Olist table '{table_key}' at {csv_path}"
            )

        sha256_hex = compute_sha256(csv_path)
        size_bytes = csv_path.stat().st_size
        df = pd.read_csv(csv_path, low_memory=False)

        actual_cols = list(df.columns)
        if actual_cols != expected_cols:
            raise ValueError(
                f"Column mismatch in {filename}: expected {expected_cols}, got {actual_cols}"
            )

        parquet_path = bronze_dir / f"{table_key}.parquet"
        df.to_parquet(parquet_path, index=False, engine="pyarrow")

        row_count = int(len(df))
        total_rows += row_count
        total_bytes += size_bytes

        tables_manifest[table_key] = {
            "filename": filename,
            "parquet_file": parquet_path.name,
            "sha256": sha256_hex,
            "size_bytes": size_bytes,
            "row_count": row_count,
            "columns": actual_cols,
        }

    manifest = {
        "contract_id": contract.contract_id,
        "contract_version": contract.contract_version,
        "dataset_slug": contract.o1_source["canonical_slug"],
        "table_count": len(tables_manifest),
        "total_rows": total_rows,
        "total_size_bytes": total_bytes,
        "tables": tables_manifest,
    }
    manifest_path = bronze_dir / "bronze_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _classify_order_eligibility(
    orders_df: pd.DataFrame,
    order_ids_with_items: set[str],
) -> pd.Series:
    """Assign deterministic O3 eligibility / exclusion reason codes to each order."""
    status = orders_df["order_status"].fillna("").astype(str)
    has_approval = orders_df["order_approved_at"].notna()
    has_items = orders_df["order_id"].isin(order_ids_with_items)
    has_delivery = orders_df["order_delivered_customer_date"].notna()
    duration_days = (
        orders_df["order_delivered_customer_date"] - orders_df["order_approved_at"]
    ).dt.total_seconds() / 86400.0

    codes = pd.Series("ELIGIBLE_MATURE", index=orders_df.index, dtype="object")

    # Apply O3 exclusion precedence deterministically:
    # 1. Missing order_approved_at -> EXCL_MISSING_APPROVAL
    # 2. Canceled or unavailable status -> EXCL_CANCELED_OR_UNAVAILABLE
    # 3. Zero items in order_items -> EXCL_NO_ITEMS
    # 4. Non-delivered status or missing delivery timestamp -> EXCL_UNDELIVERED_OR_MISSING_DELIVERY_TS
    # 5. Non-positive actual delivery duration -> EXCL_NONPOSITIVE_DURATION
    nonpositive_mask = (
        has_approval
        & status.eq("delivered")
        & has_items
        & has_delivery
        & (duration_days <= 0.0)
    )
    codes = codes.mask(nonpositive_mask, "EXCL_NONPOSITIVE_DURATION")

    undelivered_mask = (
        has_approval
        & ~status.isin(["canceled", "unavailable"])
        & has_items
        & (~status.eq("delivered") | ~has_delivery)
    )
    codes = codes.mask(undelivered_mask, "EXCL_UNDELIVERED_OR_MISSING_DELIVERY_TS")

    no_items_mask = has_approval & ~status.isin(["canceled", "unavailable"]) & ~has_items
    codes = codes.mask(no_items_mask, "EXCL_NO_ITEMS")

    canceled_mask = has_approval & status.isin(["canceled", "unavailable"])
    codes = codes.mask(canceled_mask, "EXCL_CANCELED_OR_UNAVAILABLE")

    missing_approval_mask = ~has_approval
    codes = codes.mask(missing_approval_mask, "EXCL_MISSING_APPROVAL")

    return codes


def build_silver(
    bronze_dir: Path,
    silver_dir: Path,
    contract: WM0Contract,
) -> dict[str, Any]:
    """Transform Bronze Parquet tables into typed, cleaned Silver Parquet tables with full accounting."""
    silver_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Bronze tables
    orders = pd.read_parquet(bronze_dir / "olist_orders_dataset.parquet")
    items = pd.read_parquet(bronze_dir / "olist_order_items_dataset.parquet")
    payments = pd.read_parquet(bronze_dir / "olist_order_payments_dataset.parquet")
    reviews = pd.read_parquet(bronze_dir / "olist_order_reviews_dataset.parquet")
    products = pd.read_parquet(bronze_dir / "olist_products_dataset.parquet")
    sellers = pd.read_parquet(bronze_dir / "olist_sellers_dataset.parquet")
    customers = pd.read_parquet(bronze_dir / "olist_customers_dataset.parquet")
    geolocation = pd.read_parquet(bronze_dir / "olist_geolocation_dataset.parquet")
    translations = pd.read_parquet(
        bronze_dir / "product_category_name_translation.parquet"
    )

    # 2. Parse timestamps across orders, items, reviews
    for col in TIMESTAMP_COLUMNS["olist_orders_dataset"]:
        orders[col] = pd.to_datetime(orders[col], errors="coerce")
    for col in TIMESTAMP_COLUMNS["olist_order_items_dataset"]:
        items[col] = pd.to_datetime(items[col], errors="coerce")
    for col in TIMESTAMP_COLUMNS["olist_order_reviews_dataset"]:
        reviews[col] = pd.to_datetime(reviews[col], errors="coerce")

    # 3. Validate primary/logical key uniqueness
    if orders["order_id"].isna().any() or orders["order_id"].duplicated().any():
        raise ValueError("Silver gate failed: duplicate or null order_id in orders")

    if (
        items[["order_id", "order_item_id"]].isna().any().any()
        or items.duplicated(subset=["order_id", "order_item_id"]).any()
    ):
        raise ValueError(
            "Silver gate failed: duplicate or null (order_id, order_item_id) in order_items"
        )

    if (
        payments[["order_id", "payment_sequential"]].isna().any().any()
        or payments.duplicated(subset=["order_id", "payment_sequential"]).any()
    ):
        raise ValueError(
            "Silver gate failed: duplicate or null (order_id, payment_sequential) in order_payments"
        )

    if products["product_id"].isna().any() or products["product_id"].duplicated().any():
        raise ValueError("Silver gate failed: duplicate or null product_id in products")

    if sellers["seller_id"].isna().any() or sellers["seller_id"].duplicated().any():
        raise ValueError("Silver gate failed: duplicate or null seller_id in sellers")

    if (
        customers["customer_id"].isna().any()
        or customers["customer_id"].duplicated().any()
    ):
        raise ValueError(
            "Silver gate failed: duplicate or null customer_id in customers"
        )

    # 4. Clean and normalize Products (fix *_lenght spelling, flag missing/zero dimensions)
    products = products.rename(
        columns={
            "product_name_lenght": "product_name_length",
            "product_description_lenght": "product_description_length",
        }
    )
    for num_col in [
        "product_weight_g",
        "product_length_cm",
        "product_height_cm",
        "product_width_cm",
    ]:
        products[num_col] = pd.to_numeric(products[num_col], errors="coerce").astype(
            "float64"
        )

    dim_invalid_mask = (
        products["product_weight_g"].isna()
        | (products["product_weight_g"] <= 0.0)
        | products["product_length_cm"].isna()
        | (products["product_length_cm"] <= 0.0)
        | products["product_height_cm"].isna()
        | (products["product_height_cm"] <= 0.0)
        | products["product_width_cm"].isna()
        | (products["product_width_cm"] <= 0.0)
    )
    products["dim_missing_flag"] = dim_invalid_mask.astype("int64")
    products["product_volume_cm3"] = np.where(
        dim_invalid_mask,
        0.0,
        products["product_length_cm"]
        * products["product_height_cm"]
        * products["product_width_cm"],
    ).astype("float64")

    # Join category translation onto products
    translations["product_category_name"] = (
        translations["product_category_name"].astype(str).str.strip()
    )
    translations["product_category_name_english"] = (
        translations["product_category_name_english"].astype(str).str.strip()
    )
    products = products.merge(
        translations[["product_category_name", "product_category_name_english"]],
        on="product_category_name",
        how="left",
    )
    missing_cat_mask = products["product_category_name"].isna()
    untranslated_mask = (
        ~missing_cat_mask & products["product_category_name_english"].isna()
    )
    products["product_category_name_english"] = np.where(
        missing_cat_mask,
        "unknown",
        np.where(
            untranslated_mask,
            products["product_category_name"].astype(str),
            products["product_category_name_english"],
        ),
    )

    # 5. Clean Geolocation: exact 5-column dedup, Brazil bounding box filter, zip median aggregation
    bbox = contract.o2_grain_and_cleaning["geolocation_cleaning"]["brazil_bounding_box"]
    geo_raw_rows = int(len(geolocation))
    geolocation["geolocation_zip_code_prefix"] = (
        pd.to_numeric(geolocation["geolocation_zip_code_prefix"], errors="raise")
        .astype("int64")
        .astype(str)
        .str.zfill(5)
    )
    geolocation["geolocation_lat"] = pd.to_numeric(
        geolocation["geolocation_lat"], errors="coerce"
    ).astype("float64")
    geolocation["geolocation_lng"] = pd.to_numeric(
        geolocation["geolocation_lng"], errors="coerce"
    ).astype("float64")

    geo_deduped = geolocation.drop_duplicates().copy()
    geo_deduped_rows = int(len(geo_deduped))
    geo_exact_dups_dropped = geo_raw_rows - geo_deduped_rows

    in_bounds_mask = (
        geo_deduped["geolocation_lat"].between(
            bbox["lat_min"], bbox["lat_max"], inclusive="both"
        )
        & geo_deduped["geolocation_lng"].between(
            bbox["lng_min"], bbox["lng_max"], inclusive="both"
        )
    )
    geo_in_bounds = geo_deduped.loc[in_bounds_mask].copy()
    geo_in_bounds_rows = int(len(geo_in_bounds))
    geo_out_of_bounds_dropped = geo_deduped_rows - geo_in_bounds_rows

    geo_zip = (
        geo_in_bounds.groupby("geolocation_zip_code_prefix", as_index=False)
        .agg(
            geolocation_lat=("geolocation_lat", "median"),
            geolocation_lng=("geolocation_lng", "median"),
            in_bounds_point_count=("geolocation_lat", "count"),
            geolocation_state=("geolocation_state", "first"),
            geolocation_city=("geolocation_city", "first"),
        )
        .sort_values("geolocation_zip_code_prefix")
        .reset_index(drop=True)
    )

    # 6. Normalize Seller & Customer zip prefixes and attach geo coordinates
    sellers["seller_zip_code_prefix"] = (
        pd.to_numeric(sellers["seller_zip_code_prefix"], errors="raise")
        .astype("int64")
        .astype(str)
        .str.zfill(5)
    )
    sellers = sellers.merge(
        geo_zip[["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]],
        left_on="seller_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).drop(columns=["geolocation_zip_code_prefix"])
    sellers = sellers.rename(
        columns={
            "geolocation_lat": "seller_lat",
            "geolocation_lng": "seller_lng",
        }
    )
    sellers["seller_geo_missing_flag"] = (
        sellers["seller_lat"].isna().astype("int64")
    )

    customers["customer_zip_code_prefix"] = (
        pd.to_numeric(customers["customer_zip_code_prefix"], errors="raise")
        .astype("int64")
        .astype(str)
        .str.zfill(5)
    )
    customers = customers.merge(
        geo_zip[["geolocation_zip_code_prefix", "geolocation_lat", "geolocation_lng"]],
        left_on="customer_zip_code_prefix",
        right_on="geolocation_zip_code_prefix",
        how="left",
    ).drop(columns=["geolocation_zip_code_prefix"])
    customers = customers.rename(
        columns={
            "geolocation_lat": "customer_lat",
            "geolocation_lng": "customer_lng",
        }
    )
    customers["customer_geo_missing_flag"] = (
        customers["customer_lat"].isna().astype("int64")
    )

    # 7. Type-cast Order Items, Payments, Reviews
    items["order_item_id"] = items["order_item_id"].astype("int64")
    items["price"] = pd.to_numeric(items["price"], errors="raise").astype("float64")
    items["freight_value"] = pd.to_numeric(
        items["freight_value"], errors="raise"
    ).astype("float64")

    payments["payment_sequential"] = payments["payment_sequential"].astype("int64")
    payments["payment_installments"] = payments["payment_installments"].astype("int64")
    payments["payment_value"] = pd.to_numeric(
        payments["payment_value"], errors="raise"
    ).astype("float64")

    reviews_raw_rows = int(len(reviews))
    reviews_deduped = reviews.drop_duplicates(subset=["review_id", "order_id"]).copy()
    reviews_deduped_rows = int(len(reviews_deduped))
    reviews_deduped["review_score"] = pd.to_numeric(
        reviews_deduped["review_score"], errors="coerce"
    ).astype("float64")

    # 8. Order eligibility and target derivation (O3)
    order_ids_with_items = set(items["order_id"].unique())
    orders["eligibility_code"] = _classify_order_eligibility(
        orders, order_ids_with_items
    )
    orders["is_eligible_mature"] = (
        orders["eligibility_code"].eq("ELIGIBLE_MATURE").astype("int64")
    )
    orders["delivery_duration_days"] = (
        orders["order_delivered_customer_date"] - orders["order_approved_at"]
    ).dt.total_seconds() / 86400.0
    orders["promised_duration_days"] = (
        orders["order_estimated_delivery_date"] - orders["order_approved_at"]
    ).dt.total_seconds() / 86400.0
    orders["is_late_delivered"] = np.where(
        orders["is_eligible_mature"].eq(1),
        (
            orders["order_delivered_customer_date"]
            > orders["order_estimated_delivery_date"]
        ).astype("float64"),
        np.nan,
    )

    # 9. Write Silver Parquet outputs
    orders.to_parquet(silver_dir / "silver_orders.parquet", index=False)
    items.to_parquet(silver_dir / "silver_order_items.parquet", index=False)
    payments.to_parquet(silver_dir / "silver_order_payments.parquet", index=False)
    reviews_deduped.to_parquet(silver_dir / "silver_order_reviews.parquet", index=False)
    products.to_parquet(silver_dir / "silver_products.parquet", index=False)
    sellers.to_parquet(silver_dir / "silver_sellers.parquet", index=False)
    customers.to_parquet(silver_dir / "silver_customers.parquet", index=False)
    geo_zip.to_parquet(silver_dir / "silver_geolocation_zip.parquet", index=False)
    translations.to_parquet(
        silver_dir / "silver_category_translations.parquet", index=False
    )

    eligibility_counts = {
        str(k): int(v)
        for k, v in orders["eligibility_code"].value_counts().sort_index().items()
    }
    order_status_counts = {
        str(k): int(v)
        for k, v in orders["order_status"].value_counts().sort_index().items()
    }

    accounting = {
        "contract_id": contract.contract_id,
        "contract_version": contract.contract_version,
        "orders": {
            "raw_row_count": int(len(orders)),
            "distinct_order_id_count": int(orders["order_id"].nunique()),
            "eligible_mature_count": int(orders["is_eligible_mature"].sum()),
            "excluded_count": int((orders["is_eligible_mature"] == 0).sum()),
            "eligibility_breakdown": eligibility_counts,
            "order_status_breakdown": order_status_counts,
        },
        "order_items": {
            "raw_row_count": int(len(items)),
            "distinct_order_id_count": int(items["order_id"].nunique()),
            "total_price_brl": round(float(items["price"].sum()), 2),
            "total_freight_brl": round(float(items["freight_value"].sum()), 2),
        },
        "order_payments": {
            "raw_row_count": int(len(payments)),
            "distinct_order_id_count": int(payments["order_id"].nunique()),
            "total_payment_value_brl": round(float(payments["payment_value"].sum()), 2),
        },
        "order_reviews": {
            "raw_row_count": reviews_raw_rows,
            "deduped_logical_key_row_count": reviews_deduped_rows,
            "duplicate_logical_key_rows_dropped": reviews_raw_rows - reviews_deduped_rows,
            "distinct_review_id_count": int(reviews_deduped["review_id"].nunique()),
            "distinct_order_id_count": int(reviews_deduped["order_id"].nunique()),
        },
        "products": {
            "raw_row_count": int(len(products)),
            "dim_missing_or_zero_count": int(products["dim_missing_flag"].sum()),
            "missing_category_name_count": int(missing_cat_mask.sum()),
            "untranslated_category_fallback_count": int(untranslated_mask.sum()),
        },
        "geolocation": {
            "raw_row_count": geo_raw_rows,
            "exact_duplicate_rows_dropped": geo_exact_dups_dropped,
            "deduped_row_count": geo_deduped_rows,
            "out_of_bounds_rows_dropped": geo_out_of_bounds_dropped,
            "in_bounds_row_count": geo_in_bounds_rows,
            "distinct_in_bounds_zip_prefix_count": int(len(geo_zip)),
        },
        "sellers": {
            "raw_row_count": int(len(sellers)),
            "geo_missing_count": int(sellers["seller_geo_missing_flag"].sum()),
        },
        "customers": {
            "raw_row_count": int(len(customers)),
            "distinct_customer_unique_id_count": int(
                customers["customer_unique_id"].nunique()
            ),
            "geo_missing_count": int(customers["customer_geo_missing_flag"].sum()),
        },
    }

    accounting_path = silver_dir / "silver_accounting.json"
    accounting_path.write_text(
        json.dumps(accounting, indent=2) + "\n", encoding="utf-8"
    )
    return accounting
