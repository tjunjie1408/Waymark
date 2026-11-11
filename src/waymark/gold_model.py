"""T4 Gold analytical model (`fact_order_fulfillment` + dimensions) and DuckDB SQL reference equivalence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from waymark.artifacts import require_new_outputs, write_immutable, write_parquet_immutable
from waymark.contracts import WM0Contract


def _vectorized_haversine_km(
    lat1: pd.Series | np.ndarray,
    lng1: pd.Series | np.ndarray,
    lat2: pd.Series | np.ndarray,
    lng2: pd.Series | np.ndarray,
    radius_km: float = 6371.0088,
) -> np.ndarray:
    """Vectorized great-circle Haversine distance in kilometers (`float64`)."""
    lat1_arr = np.asarray(lat1, dtype="float64")
    lng1_arr = np.asarray(lng1, dtype="float64")
    lat2_arr = np.asarray(lat2, dtype="float64")
    lng2_arr = np.asarray(lng2, dtype="float64")

    valid = ~np.isnan(lat1_arr) & ~np.isnan(lng1_arr) & ~np.isnan(lat2_arr) & ~np.isnan(lng2_arr)
    out = np.full(lat1_arr.shape, np.nan, dtype="float64")
    if not np.any(valid):
        return out

    phi1 = np.radians(lat1_arr[valid])
    phi2 = np.radians(lat2_arr[valid])
    dphi = np.radians(lat2_arr[valid] - lat1_arr[valid])
    dlambda = np.radians(lng2_arr[valid] - lng1_arr[valid])

    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2.0) ** 2
    a = np.clip(a, 0.0, 1.0)
    out[valid] = 2.0 * radius_km * np.arcsin(np.sqrt(a))
    return out


def build_gold_analytical_model(
    silver_dir: Path,
    gold_dir: Path,
    contract: WM0Contract,
) -> dict[str, Any]:
    """Build Gold `fact_order_fulfillment` and dimensions from Silver Parquet, enforcing O2 invariants."""
    gold_dir.mkdir(parents=True, exist_ok=True)
    require_new_outputs(
        gold_dir,
        [
            "fact_order_fulfillment.parquet",
            "dim_seller.parquet",
            "dim_customer.parquet",
            "dim_product.parquet",
            "dim_geolocation_zip.parquet",
            "gold_reconciliation.json",
        ],
    )
    earth_radius_km = float(contract.o5_features_and_numerics["earth_radius_km"])

    orders = pd.read_parquet(silver_dir / "silver_orders.parquet")
    items = pd.read_parquet(silver_dir / "silver_order_items.parquet")
    payments = pd.read_parquet(silver_dir / "silver_order_payments.parquet")
    reviews = pd.read_parquet(silver_dir / "silver_order_reviews.parquet")
    products = pd.read_parquet(silver_dir / "silver_products.parquet")
    sellers = pd.read_parquet(silver_dir / "silver_sellers.parquet")
    customers = pd.read_parquet(silver_dir / "silver_customers.parquet")
    geo_zip = pd.read_parquet(silver_dir / "silver_geolocation_zip.parquet")

    # 1. Enrich order_items with product physical attributes and seller coordinates
    items_enriched = items.merge(
        products[
            [
                "product_id",
                "product_category_name_english",
                "product_weight_g",
                "product_volume_cm3",
                "dim_missing_flag",
            ]
        ],
        on="product_id",
        how="left",
    ).merge(
        sellers[
            [
                "seller_id",
                "seller_zip_code_prefix",
                "seller_city",
                "seller_state",
                "seller_lat",
                "seller_lng",
                "seller_geo_missing_flag",
            ]
        ],
        on="seller_id",
        how="left",
    )
    items_enriched["item_total_brl"] = items_enriched["price"] + items_enriched["freight_value"]
    items_enriched["weight_missing_flag"] = (
        items_enriched["product_weight_g"].isna() | (items_enriched["product_weight_g"] <= 0.0)
    ).astype("int64")
    items_enriched["dim_missing_flag"] = items_enriched["dim_missing_flag"].fillna(1).astype("int64")

    # Attach customer coordinates to items_enriched to compute max/mean seller haversine per order
    order_cust_coords = orders[["order_id", "customer_id"]].merge(
        customers[["customer_id", "customer_lat", "customer_lng"]],
        on="customer_id",
        how="left",
    )
    items_enriched = items_enriched.merge(
        order_cust_coords[["order_id", "customer_lat", "customer_lng"]],
        on="order_id",
        how="left",
    )
    items_enriched["item_seller_haversine_km"] = _vectorized_haversine_km(
        items_enriched["seller_lat"],
        items_enriched["seller_lng"],
        items_enriched["customer_lat"],
        items_enriched["customer_lng"],
        radius_km=earth_radius_km,
    )

    # 2. Aggregate item metrics to 1 row per order_id
    item_agg = items_enriched.groupby("order_id", as_index=False).agg(
        item_count=("order_item_id", "count"),
        distinct_seller_count=("seller_id", "nunique"),
        distinct_product_count=("product_id", "nunique"),
        total_item_price_brl=("price", "sum"),
        total_freight_brl=("freight_value", "sum"),
        raw_total_weight_g=("product_weight_g", "sum"),
        raw_total_volume_cm3=("product_volume_cm3", "sum"),
        any_weight_missing=("weight_missing_flag", "max"),
        dim_missing_flag=("dim_missing_flag", "max"),
        earliest_shipping_limit_date=("shipping_limit_date", "min"),
        max_seller_haversine_km=("item_seller_haversine_km", "max"),
        mean_seller_haversine_km=("item_seller_haversine_km", "mean"),
    )

    # Select primary_seller_id: MAX(SUM(price + freight_value)) per (order_id, seller_id),
    # tie-break MIN(seller_id) lexicographically
    seller_order_totals = (
        items_enriched.groupby(["order_id", "seller_id"], as_index=False)
        .agg(seller_order_brl=("item_total_brl", "sum"))
        .sort_values(
            ["order_id", "seller_order_brl", "seller_id"],
            ascending=[True, False, True],
        )
    )
    primary_sellers = seller_order_totals.drop_duplicates(subset=["order_id"], keep="first")[
        ["order_id", "seller_id"]
    ].rename(columns={"seller_id": "primary_seller_id"})

    # Select primary_product_id and primary_product_category_name_english:
    # MAX(item_total_brl), tie-break MIN(order_item_id)
    primary_products = (
        items_enriched.sort_values(
            ["order_id", "item_total_brl", "order_item_id"],
            ascending=[True, False, True],
        )
        .drop_duplicates(subset=["order_id"], keep="first")[
            ["order_id", "product_id", "product_category_name_english"]
        ]
        .rename(
            columns={
                "product_id": "primary_product_id",
                "product_category_name_english": "primary_product_category_name_english",
            }
        )
    )

    item_agg = item_agg.merge(primary_sellers, on="order_id", how="left").merge(
        primary_products, on="order_id", how="left"
    )

    # 3. Aggregate payments to 1 row per order_id
    payment_agg = payments.groupby("order_id", as_index=False).agg(
        payment_row_count=("payment_sequential", "count"),
        total_payment_value_brl=("payment_value", "sum"),
        max_payment_installments=("payment_installments", "max"),
    )
    payment_type_totals = (
        payments.groupby(["order_id", "payment_type"], as_index=False)
        .agg(type_value_brl=("payment_value", "sum"))
        .sort_values(
            ["order_id", "type_value_brl", "payment_type"],
            ascending=[True, False, True],
        )
    )
    primary_payment_types = payment_type_totals.drop_duplicates(subset=["order_id"], keep="first")[
        ["order_id", "payment_type"]
    ].rename(columns={"payment_type": "primary_payment_type"})
    payment_agg = payment_agg.merge(primary_payment_types, on="order_id", how="left")

    # 4. Aggregate reviews (diagnostic/post-hoc only, never feature/target)
    review_agg = reviews.groupby("order_id", as_index=False).agg(
        review_count=("review_id", "count"),
        min_review_score=("review_score", "min"),
        latest_review_answer_timestamp=("review_answer_timestamp", "max"),
    )

    # 5. Assemble 1-row-per-order fact_order_fulfillment
    fact = (
        orders.merge(
            customers[
                [
                    "customer_id",
                    "customer_unique_id",
                    "customer_zip_code_prefix",
                    "customer_city",
                    "customer_state",
                    "customer_lat",
                    "customer_lng",
                    "customer_geo_missing_flag",
                ]
            ],
            on="customer_id",
            how="left",
        )
        .merge(item_agg, on="order_id", how="left")
        .merge(payment_agg, on="order_id", how="left")
        .merge(review_agg, on="order_id", how="left")
    )

    # Attach primary seller geography
    primary_seller_geo = sellers.rename(
        columns={
            "seller_id": "primary_seller_id",
            "seller_zip_code_prefix": "primary_seller_zip_code_prefix",
            "seller_city": "primary_seller_city",
            "seller_state": "primary_seller_state",
            "seller_lat": "primary_seller_lat",
            "seller_lng": "primary_seller_lng",
            "seller_geo_missing_flag": "primary_seller_geo_missing_flag",
        }
    )
    fact = fact.merge(primary_seller_geo, on="primary_seller_id", how="left")

    # Fill zero-counts for orders without items, payments, or reviews
    fact["item_count"] = fact["item_count"].fillna(0).astype("int64")
    fact["distinct_seller_count"] = fact["distinct_seller_count"].fillna(0).astype("int64")
    fact["distinct_product_count"] = fact["distinct_product_count"].fillna(0).astype("int64")
    fact["is_multi_seller"] = fact["distinct_seller_count"] > 1
    fact["total_item_price_brl"] = fact["total_item_price_brl"].fillna(0.0).astype("float64")
    fact["total_freight_brl"] = fact["total_freight_brl"].fillna(0.0).astype("float64")
    fact["freight_to_price_ratio"] = np.where(
        fact["total_item_price_brl"] > 0.0,
        fact["total_freight_brl"] / fact["total_item_price_brl"],
        np.nan,
    ).astype("float64")

    # Apply O2 invalid_dimension_policy: set volume and density to NULL when dim_missing_flag == 1
    fact["dim_missing_flag"] = np.where(
        fact["item_count"] == 0, 1, fact["dim_missing_flag"].fillna(1)
    ).astype("int64")
    fact["total_weight_g"] = np.where(
        (fact["item_count"] == 0) | (fact["any_weight_missing"].fillna(1) == 1),
        np.nan,
        fact["raw_total_weight_g"],
    ).astype("float64")
    fact["total_volume_cm3"] = np.where(
        fact["dim_missing_flag"] == 1,
        np.nan,
        fact["raw_total_volume_cm3"],
    ).astype("float64")
    fact["volumetric_density_g_per_cm3"] = np.where(
        (fact["dim_missing_flag"] == 1)
        | fact["total_volume_cm3"].isna()
        | (fact["total_volume_cm3"] <= 0.0)
        | fact["total_weight_g"].isna(),
        np.nan,
        fact["total_weight_g"] / fact["total_volume_cm3"],
    ).astype("float64")
    fact = fact.drop(columns=["raw_total_weight_g", "raw_total_volume_cm3", "any_weight_missing"])

    fact["payment_row_count"] = fact["payment_row_count"].fillna(0).astype("int64")
    fact["total_payment_value_brl"] = fact["total_payment_value_brl"].fillna(0.0).astype("float64")
    fact["max_payment_installments"] = fact["max_payment_installments"].fillna(0).astype("int64")
    fact["review_count"] = fact["review_count"].fillna(0).astype("int64")

    # Geography & Haversine distance for primary seller -> customer
    fact["primary_seller_geo_missing_flag"] = fact["primary_seller_lat"].isna().astype("int64")
    fact["geo_missing_flag"] = (
        (fact["customer_geo_missing_flag"] == 1) | (fact["primary_seller_geo_missing_flag"] == 1)
    ).astype("int64")
    fact["is_interstate"] = np.where(
        fact["primary_seller_state"].notna() & fact["customer_state"].notna(),
        (fact["primary_seller_state"] != fact["customer_state"]).astype("int64"),
        0,
    ).astype("int64")
    fact["state_corridor"] = np.where(
        fact["primary_seller_state"].notna() & fact["customer_state"].notna(),
        fact["primary_seller_state"].astype(str) + "->" + fact["customer_state"].astype(str),
        "UNKNOWN",
    )
    fact["primary_seller_haversine_km"] = _vectorized_haversine_km(
        fact["primary_seller_lat"],
        fact["primary_seller_lng"],
        fact["customer_lat"],
        fact["customer_lng"],
        radius_km=earth_radius_km,
    )

    # Approval-time timing intervals and targets (O3)
    fact["approval_lag_hours"] = (
        fact["order_approved_at"] - fact["order_purchase_timestamp"]
    ).dt.total_seconds() / 3600.0
    fact["shipping_limit_days_from_approval"] = (
        fact["earliest_shipping_limit_date"] - fact["order_approved_at"]
    ).dt.total_seconds() / 86400.0
    fact["y_days"] = fact["delivery_duration_days"].astype("float64")
    fact["promised_days"] = fact["promised_duration_days"].astype("float64")
    fact["y_late"] = fact["is_late_delivered"].astype("float64")

    fact = fact.sort_values("order_id").reset_index(drop=True)

    # 6. Verify O2 reconciliation invariants strictly before writing Gold outputs
    silver_order_count = int(len(orders))
    fact_order_count = int(len(fact))
    fact_distinct_orders = int(fact["order_id"].nunique())
    if fact_order_count != silver_order_count or fact_distinct_orders != silver_order_count:
        raise ValueError(
            f"O2 grain violation: Silver={silver_order_count}, "
            f"Gold={fact_order_count}, distinct={fact_distinct_orders}"
        )

    silver_price_sum = float(items["price"].sum())
    gold_price_sum = float(fact["total_item_price_brl"].sum())
    if abs(silver_price_sum - gold_price_sum) > 1e-6:
        raise ValueError(
            f"O2 item price reconciliation failed: Silver={silver_price_sum}, Gold={gold_price_sum}"
        )

    silver_freight_sum = float(items["freight_value"].sum())
    gold_freight_sum = float(fact["total_freight_brl"].sum())
    if abs(silver_freight_sum - gold_freight_sum) > 1e-6:
        raise ValueError(
            f"O2 freight reconciliation failed: Silver={silver_freight_sum}, Gold={gold_freight_sum}"
        )

    silver_payment_sum = float(payments["payment_value"].sum())
    gold_payment_sum = float(fact["total_payment_value_brl"].sum())
    if abs(silver_payment_sum - gold_payment_sum) > 1e-6:
        raise ValueError(
            f"O2 payment reconciliation failed: Silver={silver_payment_sum}, Gold={gold_payment_sum}"
        )

    if int(fact["item_count"].sum()) != len(items):
        raise ValueError("O2 item count reconciliation failed: orphan or duplicated items")
    if int(fact["payment_row_count"].sum()) != len(payments):
        raise ValueError("O2 payment count reconciliation failed: orphan or duplicated payments")

    # 7. Write Gold fact and dimension Parquet tables
    write_parquet_immutable(fact, gold_dir / "fact_order_fulfillment.parquet")
    write_parquet_immutable(sellers, gold_dir / "dim_seller.parquet")
    write_parquet_immutable(customers, gold_dir / "dim_customer.parquet")
    write_parquet_immutable(products, gold_dir / "dim_product.parquet")
    write_parquet_immutable(geo_zip, gold_dir / "dim_geolocation_zip.parquet")

    reconciliation = {
        "contract_id": contract.contract_id,
        "contract_version": contract.contract_version,
        "fact_table": "fact_order_fulfillment.parquet",
        "grain_check": {
            "silver_orders_count": silver_order_count,
            "gold_fact_rows": fact_order_count,
            "gold_distinct_order_ids": fact_distinct_orders,
            "one_row_per_order_id": fact_order_count == fact_distinct_orders == silver_order_count,
        },
        "monetary_and_count_reconciliation": {
            "silver_items_row_count": int(len(items)),
            "gold_sum_item_count": int(fact["item_count"].sum()),
            "silver_sum_item_price_brl": round(silver_price_sum, 2),
            "gold_sum_item_price_brl": round(gold_price_sum, 2),
            "price_abs_diff": abs(silver_price_sum - gold_price_sum),
            "silver_sum_freight_brl": round(silver_freight_sum, 2),
            "gold_sum_freight_brl": round(gold_freight_sum, 2),
            "freight_abs_diff": abs(silver_freight_sum - gold_freight_sum),
            "silver_payments_row_count": int(len(payments)),
            "gold_sum_payment_row_count": int(fact["payment_row_count"].sum()),
            "silver_sum_payment_value_brl": round(silver_payment_sum, 2),
            "gold_sum_payment_value_brl": round(gold_payment_sum, 2),
            "payment_abs_diff": abs(silver_payment_sum - gold_payment_sum),
        },
        "multiplicity_summary": {
            "orders_with_zero_items": int((fact["item_count"] == 0).sum()),
            "single_item_orders": int((fact["item_count"] == 1).sum()),
            "multi_item_orders": int((fact["item_count"] > 1).sum()),
            "multi_seller_orders": int(fact["is_multi_seller"].sum()),
            "orders_with_dim_missing_flag": int((fact["dim_missing_flag"] == 1).sum()),
            "orders_with_geo_missing_flag": int((fact["geo_missing_flag"] == 1).sum()),
            "interstate_orders": int((fact["is_interstate"] == 1).sum()),
        },
        "dimensions": {
            "dim_seller_rows": int(len(sellers)),
            "dim_customer_rows": int(len(customers)),
            "dim_product_rows": int(len(products)),
            "dim_geolocation_zip_rows": int(len(geo_zip)),
        },
    }

    write_immutable(
        gold_dir / "gold_reconciliation.json", (json.dumps(reconciliation, indent=2) + "\n").encode()
    )
    return reconciliation


def verify_duckdb_sql_equivalence(
    silver_dir: Path,
    gold_dir: Path,
    contract: WM0Contract,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Run independent DuckDB SQL reference queries over Silver Parquet and assert parity with Gold."""
    earth_radius_km = float(contract.o5_features_and_numerics["earth_radius_km"])
    con = duckdb.connect(database=":memory:")

    orders_pq = (silver_dir / "silver_orders.parquet").as_posix()
    items_pq = (silver_dir / "silver_order_items.parquet").as_posix()
    payments_pq = (silver_dir / "silver_order_payments.parquet").as_posix()
    products_pq = (silver_dir / "silver_products.parquet").as_posix()
    sellers_pq = (silver_dir / "silver_sellers.parquet").as_posix()
    customers_pq = (silver_dir / "silver_customers.parquet").as_posix()

    sql_query = f"""
    WITH item_product AS (
        SELECT
            i.order_id,
            i.order_item_id,
            i.product_id,
            i.seller_id,
            i.shipping_limit_date,
            i.price,
            i.freight_value,
            (i.price + i.freight_value) AS item_total_brl,
            p.product_weight_g,
            p.product_volume_cm3,
            COALESCE(p.dim_missing_flag, 1) AS dim_missing_flag,
            CASE
                WHEN p.product_weight_g IS NULL OR p.product_weight_g <= 0 THEN 1
                ELSE 0
            END AS weight_missing_flag
        FROM read_parquet('{items_pq}') i
        LEFT JOIN read_parquet('{products_pq}') p
            ON i.product_id = p.product_id
    ),
    item_agg AS (
        SELECT
            order_id,
            COUNT(order_item_id) AS item_count,
            COUNT(DISTINCT seller_id) AS distinct_seller_count,
            COUNT(DISTINCT product_id) AS distinct_product_count,
            SUM(price) AS total_item_price_brl,
            SUM(freight_value) AS total_freight_brl,
            SUM(product_weight_g) AS raw_total_weight_g,
            SUM(product_volume_cm3) AS raw_total_volume_cm3,
            MAX(weight_missing_flag) AS any_weight_missing,
            MAX(dim_missing_flag) AS dim_missing_flag,
            MIN(shipping_limit_date) AS earliest_shipping_limit_date
        FROM item_product
        GROUP BY order_id
    ),
    seller_totals AS (
        SELECT
            order_id,
            seller_id,
            SUM(item_total_brl) AS seller_order_brl,
            ROW_NUMBER() OVER (
                PARTITION BY order_id
                ORDER BY SUM(item_total_brl) DESC, seller_id ASC
            ) AS rn
        FROM item_product
        GROUP BY order_id, seller_id
    ),
    primary_seller AS (
        SELECT order_id, seller_id AS primary_seller_id
        FROM seller_totals
        WHERE rn = 1
    ),
    payment_agg AS (
        SELECT
            order_id,
            COUNT(payment_sequential) AS payment_row_count,
            SUM(payment_value) AS total_payment_value_brl,
            MAX(payment_installments) AS max_payment_installments
        FROM read_parquet('{payments_pq}')
        GROUP BY order_id
    ),
    payment_types AS (
        SELECT
            order_id,
            payment_type AS primary_payment_type,
            ROW_NUMBER() OVER (
                PARTITION BY order_id
                ORDER BY SUM(payment_value) DESC, payment_type ASC
            ) AS rn
        FROM read_parquet('{payments_pq}')
        GROUP BY order_id, payment_type
    )
    SELECT
        o.order_id,
        o.eligibility_code,
        o.is_eligible_mature,
        COALESCE(ia.item_count, 0) AS item_count,
        COALESCE(ia.distinct_seller_count, 0) AS distinct_seller_count,
        COALESCE(ia.total_item_price_brl, 0.0) AS total_item_price_brl,
        COALESCE(ia.total_freight_brl, 0.0) AS total_freight_brl,
        CASE
            WHEN COALESCE(ia.item_count, 0) = 0 OR COALESCE(ia.any_weight_missing, 1) = 1 THEN NULL
            ELSE ia.raw_total_weight_g
        END AS total_weight_g,
        CASE
            WHEN COALESCE(ia.item_count, 0) = 0 OR COALESCE(ia.dim_missing_flag, 1) = 1 THEN NULL
            ELSE ia.raw_total_volume_cm3
        END AS total_volume_cm3,
        CASE
            WHEN COALESCE(ia.item_count, 0) = 0 THEN 1
            ELSE COALESCE(ia.dim_missing_flag, 1)
        END AS dim_missing_flag,
        ps.primary_seller_id,
        COALESCE(pa.payment_row_count, 0) AS payment_row_count,
        COALESCE(pa.total_payment_value_brl, 0.0) AS total_payment_value_brl,
        COALESCE(pa.max_payment_installments, 0) AS max_payment_installments,
        pt.primary_payment_type,
        CASE
            WHEN c.customer_lat IS NULL OR s.seller_lat IS NULL THEN 1
            ELSE 0
        END AS geo_missing_flag,
        CASE
            WHEN c.customer_lat IS NULL OR s.seller_lat IS NULL THEN NULL
            ELSE 2.0 * {earth_radius_km} * ASIN(SQRT(
                POWER(SIN(RADIANS(c.customer_lat - s.seller_lat) / 2.0), 2) +
                COS(RADIANS(s.seller_lat)) * COS(RADIANS(c.customer_lat)) *
                POWER(SIN(RADIANS(c.customer_lng - s.seller_lng) / 2.0), 2)
            ))
        END AS primary_seller_haversine_km
    FROM read_parquet('{orders_pq}') o
    LEFT JOIN read_parquet('{customers_pq}') c ON o.customer_id = c.customer_id
    LEFT JOIN item_agg ia ON o.order_id = ia.order_id
    LEFT JOIN primary_seller ps ON o.order_id = ps.order_id
    LEFT JOIN read_parquet('{sellers_pq}') s ON ps.primary_seller_id = s.seller_id
    LEFT JOIN payment_agg pa ON o.order_id = pa.order_id
    LEFT JOIN payment_types pt ON o.order_id = pt.order_id AND pt.rn = 1
    ORDER BY o.order_id
    """

    sql_df = con.execute(sql_query).df()
    gold_df = (
        pd.read_parquet(gold_dir / "fact_order_fulfillment.parquet")
        .sort_values("order_id")
        .reset_index(drop=True)
    )

    if len(sql_df) != len(gold_df):
        raise ValueError(f"Row count mismatch: SQL={len(sql_df)}, Gold={len(gold_df)}")
    if not (sql_df["order_id"].values == gold_df["order_id"].values).all():
        raise ValueError("order_id ordering mismatch between SQL and Gold")

    # Compare exact string/integer columns
    exact_cols = [
        "eligibility_code",
        "is_eligible_mature",
        "item_count",
        "distinct_seller_count",
        "dim_missing_flag",
        "primary_seller_id",
        "payment_row_count",
        "max_payment_installments",
        "primary_payment_type",
        "geo_missing_flag",
    ]
    for col in exact_cols:
        s_val = sql_df[col].fillna("__NULL__")
        g_val = gold_df[col].fillna("__NULL__")
        mismatches = int((s_val != g_val).sum())
        if mismatches > 0:
            raise ValueError(f"DuckDB vs Gold mismatch on column '{col}': {mismatches} rows differ")

    # Compare floating-point columns within tolerance (1e-6)
    float_cols = [
        "total_item_price_brl",
        "total_freight_brl",
        "total_weight_g",
        "total_volume_cm3",
        "total_payment_value_brl",
        "primary_seller_haversine_km",
    ]
    max_abs_diffs: dict[str, float] = {}
    for col in float_cols:
        s_num = sql_df[col].to_numpy(dtype="float64")
        g_num = gold_df[col].to_numpy(dtype="float64")
        s_na = np.isnan(s_num)
        g_na = np.isnan(g_num)
        if not np.array_equal(s_na, g_na):
            raise ValueError(f"Null mask mismatch on float column '{col}'")
        valid = ~s_na
        max_diff = float(np.max(np.abs(s_num[valid] - g_num[valid]))) if np.any(valid) else 0.0
        if max_diff > tolerance:
            raise ValueError(f"Float tolerance exceeded on '{col}': max_diff={max_diff} > {tolerance}")
        max_abs_diffs[col] = max_diff

    # Also run an analytical reference query over eligible orders by top 5 state corridors
    gold_pq = (gold_dir / "fact_order_fulfillment.parquet").as_posix()
    corridor_sql = f"""
    SELECT
        state_corridor,
        COUNT(*) AS eligible_orders,
        ROUND(AVG(y_days), 4) AS mean_eta_days,
        ROUND(MEDIAN(y_days), 4) AS median_eta_days,
        ROUND(AVG(y_late), 4) AS late_rate,
        ROUND(AVG(primary_seller_haversine_km), 2) AS mean_distance_km
    FROM read_parquet('{gold_pq}')
    WHERE is_eligible_mature = 1
    GROUP BY state_corridor
    ORDER BY eligible_orders DESC, state_corridor ASC
    LIMIT 5
    """
    top_corridors = con.execute(corridor_sql).df().to_dict(orient="records")
    con.close()

    result = {
        "status": "OK",
        "compared_rows": int(len(gold_df)),
        "exact_columns_verified": exact_cols,
        "float_columns_max_abs_diff": max_abs_diffs,
        "tolerance": tolerance,
        "top_5_corridors_reference_query": top_corridors,
    }
    write_immutable(gold_dir / "duckdb_sql_parity.json", (json.dumps(result, indent=2) + "\n").encode())
    return result
