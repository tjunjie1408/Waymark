"""Versioned feature roles shared by preparation and remote evaluation."""

from __future__ import annotations

FEATURE_COLUMNS: list[str] = [
    "primary_seller_haversine_km",
    "max_seller_haversine_km",
    "mean_seller_haversine_km",
    "is_interstate",
    "geo_missing_flag",
    "item_count",
    "distinct_seller_count",
    "distinct_product_count",
    "is_multi_seller",
    "total_item_price_brl",
    "total_freight_brl",
    "freight_to_price_ratio",
    "total_weight_g",
    "total_volume_cm3",
    "volumetric_density_g_per_cm3",
    "dim_missing_flag",
    "promised_days",
    "approval_lag_hours",
    "shipping_limit_days_from_approval",
    "approval_dow",
    "approval_month",
    "primary_payment_type",
    "max_payment_installments",
    "primary_seller_id",
    "primary_seller_state",
    "customer_state",
    "state_corridor",
    "primary_product_category_name_english",
    "congestion_order_count_30d",
    "seller_delivered_count_30d",
    "seller_late_count_30d",
    "seller_cold_start_flag",
    "seller_late_rate_30d_smoothed",
    "seller_median_eta_days_30d",
]

CATEGORICAL_FEATURES = [
    "primary_payment_type",
    "primary_seller_state",
    "customer_state",
    "state_corridor",
    "primary_product_category_name_english",
]
IDENTITY_COLUMNS = ["primary_seller_id"]
NUMERICAL_FEATURES = [name for name in FEATURE_COLUMNS if name not in CATEGORICAL_FEATURES + IDENTITY_COLUMNS]
MODEL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES


def encode_with_levels(frame, levels):
    """Apply frozen train-only categorical levels without learning from scoring rows."""
    import numpy as np
    import pandas as pd

    if np.isinf(frame[NUMERICAL_FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("Infinite model feature values are invalid; missing values remain explicit")
    encoded = frame[MODEL_FEATURES].copy()
    native = frame[MODEL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        values = frame[column].fillna("__MISSING__").astype(str)
        encoded[column] = pd.Categorical(values.where(values.isin(levels[column])), categories=levels[column])
        native[column] = values
    return encoded, native
