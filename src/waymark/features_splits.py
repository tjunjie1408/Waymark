"""T5 Point-in-time feature builder, label-maturity temporal splits, and leakage gates (WM2)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from waymark.contracts import WM0Contract

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

FORBIDDEN_FEATURE_COLUMNS: set[str] = {
    "order_delivered_carrier_date",
    "order_delivered_customer_date",
    "order_status",
    "review_count",
    "min_review_score",
    "latest_review_answer_timestamp",
    "review_score",
    "review_creation_date",
    "review_answer_timestamp",
    "delivery_duration_days",
    "is_late_delivered",
    "y_days",
    "y_late",
}


def compute_seller_rolling_30d_features(
    fact_df: pd.DataFrame,
    window_days: int = 30,
    prior_weight_m: float = 5.0,
    min_mature_orders: int = 3,
    fixed_global_prior_late_rate: float | None = None,
) -> pd.DataFrame:
    """Compute strictly point-in-time 30-day rolling seller features at each order's approval time t0.

    For each order i of `primary_seller_id` approved at t0:
    - `congestion_order_count_30d`: count of prior orders o' of the same seller with
      (t0 - window_days) <= o'.order_approved_at < t0.
    - `seller_delivered_count_30d`: count of prior mature delivered orders o' of the same seller with
      (t0 - window_days) <= o'.order_approved_at < t0 AND o'.order_delivered_customer_date <= t0
      AND o'.is_eligible_mature == 1.
    - `seller_late_count_30d`: sum of o'.y_late over the mature delivered set.
    - `seller_cold_start_flag`: 1 if seller_delivered_count_30d < min_mature_orders else 0.
    - `seller_late_rate_30d_smoothed`: (seller_late_count_30d + m * prior) / (seller_delivered_count_30d + m).
    - `seller_median_eta_days_30d`: median(o'.y_days) over the mature delivered set if
      count >= min_mature_orders, else NaN.
    """
    df = fact_df.copy()
    n_rows = len(df)

    congestion_counts = np.zeros(n_rows, dtype="int64")
    delivered_counts = np.zeros(n_rows, dtype="int64")
    late_counts = np.zeros(n_rows, dtype="int64")
    cold_start_flags = np.ones(n_rows, dtype="int64")
    smoothed_late_rates = np.full(n_rows, np.nan, dtype="float64")
    median_etas = np.full(n_rows, np.nan, dtype="float64")

    window_ns = np.int64(window_days) * np.int64(86400) * np.int64(1_000_000_000)

    # Determine expanding or fixed global prior late rate at t0
    if fixed_global_prior_late_rate is not None:
        global_priors = np.full(n_rows, float(fixed_global_prior_late_rate), dtype="float64")
    else:
        # Compute point-in-time cumulative mature late rate across all orders delivered <= t0.
        # Fallback to 0.08 during initial bootstrap when fewer than 20 mature orders exist.
        mature_mask = (
            df["is_eligible_mature"].eq(1)
            & df["order_delivered_customer_date"].notna()
            & df["y_late"].notna()
        )
        deliv_times = (
            df.loc[mature_mask, "order_delivered_customer_date"]
            .astype("datetime64[ns]")
            .to_numpy(dtype="int64")
        )
        deliv_late = df.loc[mature_mask, "y_late"].to_numpy(dtype="float64")
        deliv_order = np.argsort(deliv_times, kind="mergesort")
        sorted_deliv_times = deliv_times[deliv_order]
        cum_deliv_late = np.cumsum(deliv_late[deliv_order])

        app_times_all = (
            df["order_approved_at"].astype("datetime64[ns]").to_numpy(dtype="int64")
        )
        has_app_all = df["order_approved_at"].notna().to_numpy()
        global_priors = np.full(n_rows, 0.08, dtype="float64")
        if len(sorted_deliv_times) > 0:
            idx_right = np.searchsorted(
                sorted_deliv_times, app_times_all[has_app_all], side="right"
            )
            priors_valid = np.where(
                idx_right >= 20,
                cum_deliv_late[np.maximum(idx_right - 1, 0)] / np.maximum(idx_right, 1),
                0.08,
            )
            global_priors[has_app_all] = priors_valid

    # Group by primary_seller_id for seller-level rolling 30d history
    valid_seller_app = df["primary_seller_id"].notna() & df["order_approved_at"].notna()
    valid_indices = np.flatnonzero(valid_seller_app.to_numpy())

    app_ns = df["order_approved_at"].astype("datetime64[ns]").to_numpy(dtype="int64")
    deliv_ns = (
        df["order_delivered_customer_date"]
        .astype("datetime64[ns]")
        .to_numpy(dtype="int64")
    )
    has_deliv = (
        df["is_eligible_mature"].eq(1) & df["order_delivered_customer_date"].notna()
    ).to_numpy()
    y_days_arr = df["y_days"].to_numpy(dtype="float64")
    y_late_arr = df["y_late"].fillna(0.0).to_numpy(dtype="int64")

    sub = df.iloc[valid_indices]
    for _, group in sub.groupby("primary_seller_id", sort=False):
        idx = group.index.to_numpy()
        g_app = app_ns[idx]
        # Sort chronologically by approval timestamp
        order = np.argsort(g_app, kind="mergesort")
        s_idx = idx[order]
        s_app = g_app[order]
        s_deliv = deliv_ns[s_idx]
        s_has_deliv = has_deliv[s_idx]
        s_y_days = y_days_arr[s_idx]
        s_y_late = y_late_arr[s_idx]

        k = len(s_idx)
        for pos in range(k):
            t0 = s_app[pos]
            t_start = t0 - window_ns
            # Strictly prior approvals: t_start <= app < t0
            left = int(np.searchsorted(s_app, t_start, side="left"))
            right = int(np.searchsorted(s_app, t0, side="left"))

            row_i = s_idx[pos]
            prior_p = global_priors[row_i]

            if right <= left:
                congestion_counts[row_i] = 0
                delivered_counts[row_i] = 0
                late_counts[row_i] = 0
                cold_start_flags[row_i] = 1
                smoothed_late_rates[row_i] = prior_p
                median_etas[row_i] = np.nan
                continue

            congestion_counts[row_i] = right - left

            # Filter window [left:right] for mature outcomes observed by t0 (delivered <= t0)
            w_has_deliv = s_has_deliv[left:right]
            w_deliv = s_deliv[left:right]
            mature_in_win = w_has_deliv & (w_deliv <= t0)

            n_mature = int(np.sum(mature_in_win))
            delivered_counts[row_i] = n_mature

            if n_mature == 0:
                late_counts[row_i] = 0
                cold_start_flags[row_i] = 1
                smoothed_late_rates[row_i] = prior_p
                median_etas[row_i] = np.nan
            else:
                n_late = int(np.sum(s_y_late[left:right][mature_in_win]))
                late_counts[row_i] = n_late
                is_cold = 1 if n_mature < min_mature_orders else 0
                cold_start_flags[row_i] = is_cold
                smoothed_late_rates[row_i] = (
                    float(n_late) + prior_weight_m * prior_p
                ) / (float(n_mature) + prior_weight_m)
                if n_mature >= min_mature_orders:
                    median_etas[row_i] = float(
                        np.median(s_y_days[left:right][mature_in_win])
                    )
                else:
                    median_etas[row_i] = np.nan

    df["congestion_order_count_30d"] = congestion_counts
    df["seller_delivered_count_30d"] = delivered_counts
    df["seller_late_count_30d"] = late_counts
    df["seller_cold_start_flag"] = cold_start_flags
    df["seller_late_rate_30d_smoothed"] = smoothed_late_rates
    df["seller_median_eta_days_30d"] = median_etas
    df["approval_dow"] = (
        df["order_approved_at"].dt.day_of_week.fillna(-1).astype("int64")
    )
    df["approval_month"] = (
        df["order_approved_at"].dt.month.fillna(-1).astype("int64")
    )
    return df


def assign_chronological_splits_with_maturity_purge(
    df: pd.DataFrame,
    contract: WM0Contract,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assign 70/15/15 chronological splits and purge immature labels at split boundaries."""
    out = df.copy()
    props = contract.o4_splits_and_history["proportions"]
    train_prop = float(props["train"])
    val_prop = float(props["val_calibration"])

    eligible_mask = out["eligibility_code"].eq("ELIGIBLE_MATURE")
    eligible = out.loc[eligible_mask].sort_values(
        ["order_approved_at", "order_id"], ascending=[True, True]
    )
    n_elig = len(eligible)
    if n_elig == 0:
        raise ValueError("Cannot construct chronological splits: 0 ELIGIBLE_MATURE orders")

    n_train_raw = max(1, int(round(n_elig * train_prop)))
    n_val_raw = max(1, int(round(n_elig * val_prop)))
    if n_train_raw + n_val_raw >= n_elig:
        n_train_raw = max(1, n_elig - 2)
        n_val_raw = 1

    train_idx = eligible.index[:n_train_raw]
    val_idx = eligible.index[n_train_raw : n_train_raw + n_val_raw]
    test_idx = eligible.index[n_train_raw + n_val_raw :]

    t_train_end = eligible.loc[train_idx, "order_approved_at"].max()
    t_val_end = eligible.loc[val_idx, "order_approved_at"].max()
    t_test_end = eligible.loc[test_idx, "order_approved_at"].max()

    out["split_window"] = "EXCLUDED_GLOBAL"
    out.loc[train_idx, "split_window"] = "train_window"
    out.loc[val_idx, "split_window"] = "val_calibration_window"
    out.loc[test_idx, "split_window"] = "holdout_test_window"

    # Apply O4 label-maturity rule at split boundaries:
    # A split ending at T_split may only consume labels where order_delivered_customer_date <= T_split.
    out["split_role"] = "EXCLUDED"
    train_mature = train_idx[
        out.loc[train_idx, "order_delivered_customer_date"] <= t_train_end
    ]
    train_immature = train_idx[
        out.loc[train_idx, "order_delivered_customer_date"] > t_train_end
    ]

    val_mature = val_idx[out.loc[val_idx, "order_delivered_customer_date"] <= t_val_end]
    val_immature = val_idx[
        out.loc[val_idx, "order_delivered_customer_date"] > t_val_end
    ]

    out.loc[train_mature, "split_role"] = "train"
    out.loc[train_immature, "split_role"] = "EXCL_IMMATURE_AT_TRAIN_BOUNDARY"
    out.loc[val_mature, "split_role"] = "val_calibration"
    out.loc[val_immature, "split_role"] = "EXCL_IMMATURE_AT_VAL_BOUNDARY"
    out.loc[test_idx, "split_role"] = "holdout_test"

    # Block bootstrap cluster key: (primary_seller_id, approval_iso_week)
    iso_cal = out["order_approved_at"].dt.isocalendar()
    iso_week_str = (
        iso_cal["year"].fillna(0).astype("int64").astype(str)
        + "-W"
        + iso_cal["week"].fillna(0).astype("int64").astype(str).str.zfill(2)
    )
    out["bootstrap_cluster_id"] = (
        out["primary_seller_id"].fillna("NO_SELLER").astype(str) + "__" + iso_week_str
    )

    # Also compute O3 end-of-day (+23:59:59) late-label sensitivity diagnostic on eligible orders
    eod_deadline = out["order_estimated_delivery_date"] + pd.Timedelta(
        hours=23, minutes=59, seconds=59
    )
    late_eod = (out["order_delivered_customer_date"] > eod_deadline).astype("float64")

    split_summary = {
        "contract_id": contract.contract_id,
        "contract_version": contract.contract_version,
        "ordering_timestamp": "order_approved_at",
        "boundaries": {
            "train_min_approved_at": str(
                eligible.loc[train_idx, "order_approved_at"].min()
            ),
            "T_train_end": str(t_train_end),
            "T_val_end": str(t_val_end),
            "T_test_end": str(t_test_end),
        },
        "counts": {
            "total_orders": int(len(out)),
            "eligible_mature_orders": int(n_elig),
            "excluded_global_orders": int((~eligible_mask).sum()),
            "train_window_raw": int(len(train_idx)),
            "train_mature_retained": int(len(train_mature)),
            "train_purged_immature_at_boundary": int(len(train_immature)),
            "val_calibration_window_raw": int(len(val_idx)),
            "val_calibration_mature_retained": int(len(val_mature)),
            "val_purged_immature_at_boundary": int(len(val_immature)),
            "holdout_test_orders": int(len(test_idx)),
        },
        "target_rates_by_split": {
            "train_mean_y_days": round(
                float(out.loc[train_mature, "y_days"].mean()), 4
            ),
            "train_late_rate": round(float(out.loc[train_mature, "y_late"].mean()), 4),
            "val_calibration_mean_y_days": round(
                float(out.loc[val_mature, "y_days"].mean()), 4
            ),
            "val_calibration_late_rate": round(
                float(out.loc[val_mature, "y_late"].mean()), 4
            ),
            "holdout_test_mean_y_days": round(
                float(out.loc[test_idx, "y_days"].mean()), 4
            ),
            "holdout_test_late_rate": round(
                float(out.loc[test_idx, "y_late"].mean()), 4
            ),
        },
        "deadline_comparison_sensitivity": {
            "exact_timestamp_late_rate_eligible": round(
                float(out.loc[eligible_mask, "y_late"].mean()), 4
            ),
            "end_of_day_235959_late_rate_eligible": round(
                float(late_eod.loc[eligible_mask].mean()), 4
            ),
        },
    }
    return out, split_summary


def assert_approval_time_feature_safety(
    df: pd.DataFrame,
    feature_cols: list[str],
    split_summary: dict[str, Any],
) -> None:
    """Enforce T5 anti-leakage gate: reject forbidden post-cutoff columns and immature split labels."""
    leaked_cols = set(feature_cols).intersection(FORBIDDEN_FEATURE_COLUMNS)
    if leaked_cols:
        raise ValueError(
            f"Leakage gate violation: forbidden post-cutoff columns in feature matrix: {sorted(leaked_cols)}"
        )

    t_train_end = pd.Timestamp(split_summary["boundaries"]["T_train_end"])
    t_val_end = pd.Timestamp(split_summary["boundaries"]["T_val_end"])

    train_rows = df.loc[df["split_role"] == "train"]
    if (train_rows["order_delivered_customer_date"] > t_train_end).any():
        raise ValueError(
            "Leakage gate violation: train split contains orders delivered after T_train_end"
        )

    val_rows = df.loc[df["split_role"] == "val_calibration"]
    if (val_rows["order_delivered_customer_date"] > t_val_end).any():
        raise ValueError(
            "Leakage gate violation: val_calibration split contains orders delivered after T_val_end"
        )

    if (train_rows["order_approved_at"] > val_rows["order_approved_at"].min()).any():
        raise ValueError("Chronological ordering violation between train and val_calibration")

    test_rows = df.loc[df["split_role"] == "holdout_test"]
    if (val_rows["order_approved_at"] > test_rows["order_approved_at"].min()).any():
        raise ValueError(
            "Chronological ordering violation between val_calibration and holdout_test"
        )


def evaluate_non_learned_baselines(df: pd.DataFrame) -> dict[str, Any]:
    """Compute O6 non-learned arithmetic/empirical baselines (`B0`, `B1`) strictly from `train` statistics."""
    train = df.loc[df["split_role"] == "train"].copy()
    val = df.loc[df["split_role"] == "val_calibration"].copy()

    global_p10 = float(np.percentile(train["y_days"], 10))
    global_p50 = float(np.percentile(train["y_days"], 50))
    global_p90 = float(np.percentile(train["y_days"], 90))
    global_late_rate = float(train["y_late"].mean())

    # Corridor empirical quantiles and smoothed late rates from train only (fallback when group < 20)
    corridor_stats = (
        train.groupby("state_corridor", as_index=False)
        .agg(
            n_train=("y_days", "count"),
            p10=("y_days", lambda s: float(np.percentile(s, 10))),
            p50=("y_days", lambda s: float(np.percentile(s, 50))),
            p90=("y_days", lambda s: float(np.percentile(s, 90))),
            late_sum=("y_late", "sum"),
        )
    )
    corridor_stats["b1_p10"] = np.where(
        corridor_stats["n_train"] >= 20, corridor_stats["p10"], global_p10
    )
    corridor_stats["b1_p50"] = np.where(
        corridor_stats["n_train"] >= 20, corridor_stats["p50"], global_p50
    )
    corridor_stats["b1_p90"] = np.where(
        corridor_stats["n_train"] >= 20, corridor_stats["p90"], global_p90
    )
    corridor_stats["b0_corridor_late_rate"] = (
        corridor_stats["late_sum"] + 5.0 * global_late_rate
    ) / (corridor_stats["n_train"] + 5.0)

    val_eval = val.merge(
        corridor_stats[
            ["state_corridor", "b1_p10", "b1_p50", "b1_p90", "b0_corridor_late_rate"]
        ],
        on="state_corridor",
        how="left",
    )
    val_eval["b1_p10"] = val_eval["b1_p10"].fillna(global_p10)
    val_eval["b1_p50"] = val_eval["b1_p50"].fillna(global_p50)
    val_eval["b1_p90"] = val_eval["b1_p90"].fillna(global_p90)
    val_eval["b0_corridor_late_rate"] = val_eval["b0_corridor_late_rate"].fillna(
        global_late_rate
    )

    y_val = val_eval["y_days"].to_numpy(dtype="float64")
    b0_mae = float(np.mean(np.abs(y_val - val_eval["promised_days"].to_numpy(dtype="float64"))))
    b1_mae = float(np.mean(np.abs(y_val - val_eval["b1_p50"].to_numpy(dtype="float64"))))
    b1_cov80 = float(
        np.mean(
            (y_val >= val_eval["b1_p10"].to_numpy(dtype="float64"))
            & (y_val <= val_eval["b1_p90"].to_numpy(dtype="float64"))
        )
    )
    b1_width80 = float(
        np.mean(
            val_eval["b1_p90"].to_numpy(dtype="float64")
            - val_eval["b1_p10"].to_numpy(dtype="float64")
        )
    )
    b0_risk_brier = float(
        np.mean(
            (
                val_eval["y_late"].to_numpy(dtype="float64")
                - val_eval["b0_corridor_late_rate"].to_numpy(dtype="float64")
            )
            ** 2
        )
    )

    return {
        "evaluated_split": "val_calibration",
        "note": (
            "Non-learned arithmetic/empirical baselines fitted strictly on train; "
            "holdout_test untouched before WM3."
        ),
        "B0_platform_promised_eta": {
            "val_mae_days": round(b0_mae, 4),
        },
        "B1_corridor_empirical_quantiles_eta": {
            "val_mae_p50_days": round(b1_mae, 4),
            "val_interval_coverage_80": round(b1_cov80, 4),
            "val_interval_mean_width_days": round(b1_width80, 4),
        },
        "B0_corridor_historical_late_rate": {
            "val_brier_score": round(b0_risk_brier, 4),
        },
    }


def build_t5_features_and_splits(
    gold_dir: Path,
    features_dir: Path,
    contract: WM0Contract,
    fixed_global_prior_late_rate: float | None = None,
) -> dict[str, Any]:
    """Execute T5 feature generation, leakage verification, and split manifest export (closing WM2)."""
    features_dir.mkdir(parents=True, exist_ok=True)
    fact = pd.read_parquet(gold_dir / "fact_order_fulfillment.parquet")

    seller_cfg = contract.o5_features_and_numerics["seller_rolling_history"]
    window_days = int(seller_cfg["window_days"])
    prior_m = float(seller_cfg["seller_late_rate_30d_smoothed"]["prior_weight_m"])
    min_mature = int(seller_cfg["cold_start_policy"]["min_mature_orders"])

    feat_df = compute_seller_rolling_30d_features(
        fact_df=fact,
        window_days=window_days,
        prior_weight_m=prior_m,
        min_mature_orders=min_mature,
        fixed_global_prior_late_rate=fixed_global_prior_late_rate,
    )

    split_df, split_summary = assign_chronological_splits_with_maturity_purge(
        feat_df, contract
    )
    assert_approval_time_feature_safety(split_df, FEATURE_COLUMNS, split_summary)

    # Export full feature+label+split table and clean model-ready development/holdout Parquet files
    full_pq = features_dir / "order_features_and_splits.parquet"
    split_df.to_parquet(full_pq, index=False)

    hasher = hashlib.sha256()
    hasher.update(full_pq.read_bytes())
    parquet_sha256 = hasher.hexdigest()

    baselines_val = evaluate_non_learned_baselines(split_df)

    manifest = {
        "status": "OK",
        "stage": "T5",
        "gate": "WM2_VERIFIED_LOCALLY",
        "contract_id": contract.contract_id,
        "contract_version": contract.contract_version,
        "feature_table": full_pq.name,
        "feature_table_sha256": parquet_sha256,
        "feature_column_count": len(FEATURE_COLUMNS),
        "feature_columns": FEATURE_COLUMNS,
        "forbidden_columns_excluded": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "split_summary": split_summary,
        "non_learned_baselines_val_preview": baselines_val,
    }
    (features_dir / "t5_split_and_feature_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
