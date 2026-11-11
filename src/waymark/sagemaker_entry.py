"""Remote SageMaker training entrypoint for M1 (ETA quantiles) and M2 (delay risk).

Executes inside SageMaker ml.m5.large training container in ap-southeast-2.
Enforces zero local model fitting (PRD F2) and logs complete evaluation metrics.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sagemaker_entry")

NUMERICAL_FEATURES = [
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
    "max_payment_installments",
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


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    """Calculate quantile pinball loss."""
    diff = y_true - y_pred
    return float(np.mean(np.maximum(alpha * diff, (alpha - 1.0) * diff)))


def expected_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Calculate 10-bin expected calibration error (ECE)."""
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(y_true[in_bin])
            avg_prob_in_bin = np.mean(y_prob[in_bin])
            ece += np.abs(avg_prob_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def main() -> None:
    from catboost import CatBoostClassifier
    from lightgbm import LGBMRegressor
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.metrics import (
        brier_score_loss,
        f1_score,
        mean_absolute_error,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from xgboost import XGBClassifier

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument(
        "--training",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"),
    )
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    training_dir = Path(args.training)
    output_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    logger.info("Loading training parquet from %s...", training_dir)
    parquet_files = list(training_dir.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet file found in {training_dir}")
    df = pd.read_parquet(parquet_files[0])
    logger.info("Loaded dataframe with %d rows, %d columns.", len(df), len(df.columns))

    # Strict split isolation per WM0 contract
    eligible_df = df[df["is_eligible_mature"] == 1].copy()
    train_df = eligible_df[eligible_df["split_role"] == "train"].copy()
    val_df = eligible_df[eligible_df["split_role"] == "val_calibration"].copy()
    test_df = eligible_df[eligible_df["split_role"] == "holdout_test"].copy()

    logger.info(
        "Split sizes: train=%d, val=%d, test=%d",
        len(train_df),
        len(val_df),
        len(test_df),
    )

    # Encode categorical features for LightGBM / XGBoost
    cat_maps: dict[str, dict[str, int]] = {}
    for col in CATEGORICAL_FEATURES:
        unique_vals = sorted(train_df[col].dropna().unique().tolist())
        mapping = {v: idx for idx, v in enumerate(unique_vals)}
        cat_maps[col] = mapping

    def prepare_x(split_df: pd.DataFrame) -> pd.DataFrame:
        x_df = split_df[NUMERICAL_FEATURES].copy()
        for c in CATEGORICAL_FEATURES:
            mapping = cat_maps[c]
            # Unseen categories mapped to -1
            x_df[c] = split_df[c].map(mapping).fillna(-1).astype(int)
        return x_df

    x_train = prepare_x(train_df)
    x_val = prepare_x(val_df)
    x_test = prepare_x(test_df)

    y_eta_train = train_df["y_days"].to_numpy(dtype=float)
    y_eta_val = val_df["y_days"].to_numpy(dtype=float)
    y_eta_test = test_df["y_days"].to_numpy(dtype=float)

    y_late_train = train_df["y_late"].to_numpy(dtype=int)
    y_late_val = val_df["y_late"].to_numpy(dtype=int)
    y_late_test = test_df["y_late"].to_numpy(dtype=int)

    # -------------------------------------------------------------------------
    # Task 1 (T6): M1 LightGBM Quantile Regression (P10, P50, P90)
    # -------------------------------------------------------------------------
    logger.info("Training M1 LightGBM quantile regression models...")
    m1_models: dict[str, LGBMRegressor] = {}
    m1_val_preds: dict[str, np.ndarray] = {}
    m1_test_preds: dict[str, np.ndarray] = {}

    for alpha in [0.10, 0.50, 0.90]:
        model = LGBMRegressor(
            objective="quantile",
            alpha=alpha,
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            random_state=args.random_seed,
            n_jobs=-1,
        )
        model.fit(x_train, y_eta_train)
        alpha_key = f"p{int(alpha * 100)}"
        m1_models[alpha_key] = model
        m1_val_preds[alpha_key] = model.predict(x_val)
        m1_test_preds[alpha_key] = model.predict(x_test)
        with open(model_dir / f"m1_lgbm_{alpha_key}.pkl", "wb") as f:
            pickle.dump(model, f)

    # Raw quantile crossing check
    raw_crossings = np.sum(
        (m1_val_preds["p10"] > m1_val_preds["p50"])
        | (m1_val_preds["p50"] > m1_val_preds["p90"])
    )
    raw_crossing_rate = float(raw_crossings / len(val_df))

    # Post-hoc sorting repair for crossings (development protocol)
    val_sorted_stack = np.sort(
        np.column_stack([m1_val_preds["p10"], m1_val_preds["p50"], m1_val_preds["p90"]]),
        axis=1,
    )
    val_p10_rep, val_p50_rep, val_p90_rep = (
        val_sorted_stack[:, 0],
        val_sorted_stack[:, 1],
        val_sorted_stack[:, 2],
    )

    test_sorted_stack = np.sort(
        np.column_stack([m1_test_preds["p10"], m1_test_preds["p50"], m1_test_preds["p90"]]),
        axis=1,
    )
    test_p10_rep, test_p50_rep, test_p90_rep = (
        test_sorted_stack[:, 0],
        test_sorted_stack[:, 1],
        test_sorted_stack[:, 2],
    )

    val_coverage_raw = float(np.mean((y_eta_val >= val_p10_rep) & (y_eta_val <= val_p90_rep)))
    val_mean_width_raw = float(np.mean(val_p90_rep - val_p10_rep))
    val_mae_p50 = float(mean_absolute_error(y_eta_val, val_p50_rep))

    # Split-conformal calibration on val_calibration
    conformal_c = 0.0
    if abs(val_coverage_raw - 0.80) > 0.03:
        conformity_scores = np.maximum(val_p10_rep - y_eta_val, y_eta_val - val_p90_rep)
        conformal_c = float(np.quantile(conformity_scores, 0.80))
        logger.info("Applied split conformal calibration adjustment: c = %.4f", conformal_c)

    # Test set evaluation for M1
    test_cov = float(
        np.mean(
            (y_eta_test >= (test_p10_rep - conformal_c))
            & (y_eta_test <= (test_p90_rep + conformal_c))
        )
    )
    test_width = float(np.mean((test_p90_rep + conformal_c) - (test_p10_rep - conformal_c)))
    test_mae_p50 = float(mean_absolute_error(y_eta_test, test_p50_rep))

    # -------------------------------------------------------------------------
    # Task 2 (T7): M2 Delay Risk Classification (CatBoost vs XGBoost)
    # -------------------------------------------------------------------------
    logger.info("Training M2 CatBoost and XGBoost classifiers...")
    # Candidate A: CatBoost
    catboost_model = CatBoostClassifier(
        iterations=200,
        learning_rate=0.08,
        depth=6,
        random_seed=args.random_seed,
        verbose=False,
    )
    catboost_model.fit(x_train, y_late_train)
    catboost_val_raw = catboost_model.predict_proba(x_val)[:, 1]

    # Candidate B: XGBoost
    xgboost_model = XGBClassifier(
        n_estimators=150,
        learning_rate=0.08,
        max_depth=5,
        random_state=args.random_seed,
        eval_metric="logloss",
    )
    xgboost_model.fit(x_train, y_late_train)
    xgboost_val_raw = xgboost_model.predict_proba(x_val)[:, 1]

    # Model evaluation and comparison on validation set
    cb_val_auc = float(roc_auc_score(y_late_val, catboost_val_raw))
    cb_val_brier = float(brier_score_loss(y_late_val, catboost_val_raw))
    cb_val_ece = float(expected_calibration_error(y_late_val, catboost_val_raw))

    xgb_val_auc = float(roc_auc_score(y_late_val, xgboost_val_raw))
    xgb_val_brier = float(brier_score_loss(y_late_val, xgboost_val_raw))
    xgb_val_ece = float(expected_calibration_error(y_late_val, xgboost_val_raw))

    logger.info("CatBoost Val: ROC-AUC=%.4f, Brier=%.4f, ECE=%.4f", cb_val_auc, cb_val_brier, cb_val_ece)
    logger.info("XGBoost Val: ROC-AUC=%.4f, Brier=%.4f, ECE=%.4f", xgb_val_auc, xgb_val_brier, xgb_val_ece)

    # Probability calibration on val_calibration (Platt scaling & Isotonic)
    calibrated_cb = CalibratedClassifierCV(catboost_model, cv="prefit", method="isotonic")
    calibrated_cb.fit(x_val, y_late_val)
    cb_val_calibrated = calibrated_cb.predict_proba(x_val)[:, 1]
    cb_calibrated_brier = float(brier_score_loss(y_late_val, cb_val_calibrated))
    cb_calibrated_ece = float(expected_calibration_error(y_late_val, cb_val_calibrated))

    # Threshold selection policy on val_calibration: maximize F1
    best_tau = 0.5
    best_f1 = 0.0
    for tau in np.linspace(0.1, 0.9, 81):
        f1 = f1_score(y_late_val, (cb_val_calibrated >= tau).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_tau = float(tau)

    # Test set evaluation for M2
    cb_test_prob = calibrated_cb.predict_proba(x_test)[:, 1]
    test_auc = float(roc_auc_score(y_late_test, cb_test_prob))
    test_brier = float(brier_score_loss(y_late_test, cb_test_prob))
    test_ece = float(expected_calibration_error(y_late_test, cb_test_prob))
    test_pred_tau = (cb_test_prob >= best_tau).astype(int)
    test_precision = float(precision_score(y_late_test, test_pred_tau, zero_division=0))
    test_recall = float(recall_score(y_late_test, test_pred_tau, zero_division=0))
    test_f1 = float(f1_score(y_late_test, test_pred_tau, zero_division=0))

    # Top-10% alert volume budget evaluation
    top10_cutoff = float(np.quantile(cb_test_prob, 0.90))
    test_pred_top10 = (cb_test_prob >= top10_cutoff).astype(int)
    test_precision_top10 = float(precision_score(y_late_test, test_pred_top10, zero_division=0))
    test_recall_top10 = float(recall_score(y_late_test, test_pred_top10, zero_division=0))

    # Save models
    catboost_model.save_model(str(model_dir / "m2_catboost.cbm"))
    xgboost_model.save_model(str(model_dir / "m2_xgboost.json"))
    with open(model_dir / "m2_calibrated_catboost.pkl", "wb") as f:
        pickle.dump(calibrated_cb, f)
    with open(model_dir / "cat_mappings.json", "w", encoding="utf-8") as f:
        json.dump(cat_maps, f, indent=2)

    # -------------------------------------------------------------------------
    # Baseline Comparisons on holdout_test
    # -------------------------------------------------------------------------
    # B0 Platform Promised ETA on test:
    b0_test_mae = float(mean_absolute_error(y_eta_test, test_df["promised_days"]))

    # B1 Corridor Median ETA on test:
    train_corridor_p50 = train_df.groupby("state_corridor")["y_days"].median().to_dict()
    global_p50 = float(train_df["y_days"].median())
    b1_test_pred = test_df["state_corridor"].map(train_corridor_p50).fillna(global_p50).to_numpy()
    b1_test_mae = float(mean_absolute_error(y_eta_test, b1_test_pred))

    # B0 Corridor Historical Late Rate on test:
    train_corridor_late = train_df.groupby("state_corridor")["y_late"].mean().to_dict()
    global_late = float(train_df["y_late"].mean())
    b0_test_late_prob = test_df["state_corridor"].map(train_corridor_late).fillna(global_late).to_numpy()
    b0_test_brier = float(brier_score_loss(y_late_test, b0_test_late_prob))

    # Gate WM3 Verification
    gate_wm3_passed = bool(
        (test_mae_p50 < b0_test_mae)
        and (test_mae_p50 < b1_test_mae)
        and (test_auc > 0.70)
        and (test_brier < b0_test_brier)
    )

    training_time_seconds = time.time() - start_time
    summary: dict[str, Any] = {
        "status": "OK",
        "stage": "T6_T7",
        "gate": "WM3_REVIEW",
        "gate_wm3_passed": gate_wm3_passed,
        "runtime_seconds": round(training_time_seconds, 2),
        "m1_eta_quantile_metrics": {
            "val_mae_p50_days": round(val_mae_p50, 4),
            "val_coverage_80_raw": round(val_coverage_raw, 4),
            "val_mean_width_days": round(val_mean_width_raw, 4),
            "raw_quantile_crossing_rate": round(raw_crossing_rate, 4),
            "conformal_adjustment_c": round(conformal_c, 4),
            "test_mae_p50_days": round(test_mae_p50, 4),
            "test_interval_coverage_80": round(test_cov, 4),
            "test_interval_mean_width_days": round(test_width, 4),
            "b0_platform_test_mae_days": round(b0_test_mae, 4),
            "b1_corridor_median_test_mae_days": round(b1_test_mae, 4),
            "m1_vs_b0_mae_reduction_pct": round((b0_test_mae - test_mae_p50) / b0_test_mae * 100.0, 2),
            "m1_vs_b1_mae_reduction_pct": round((b1_test_mae - test_mae_p50) / b1_test_mae * 100.0, 2),
        },
        "m2_delay_risk_metrics": {
            "catboost_val_roc_auc": round(cb_val_auc, 4),
            "catboost_val_brier": round(cb_val_brier, 4),
            "catboost_val_ece": round(cb_val_ece, 4),
            "xgboost_val_roc_auc": round(xgb_val_auc, 4),
            "xgboost_val_brier": round(xgb_val_brier, 4),
            "xgboost_val_ece": round(xgb_val_ece, 4),
            "calibrated_val_brier": round(cb_calibrated_brier, 4),
            "calibrated_val_ece": round(cb_calibrated_ece, 4),
            "optimal_threshold_tau_star": round(best_tau, 4),
            "val_max_f1": round(best_f1, 4),
            "test_roc_auc": round(test_auc, 4),
            "test_brier_score": round(test_brier, 4),
            "test_expected_calibration_error": round(test_ece, 4),
            "test_precision_at_tau": round(test_precision, 4),
            "test_recall_at_tau": round(test_recall, 4),
            "test_f1_at_tau": round(test_f1, 4),
            "test_precision_at_top10pct": round(test_precision_top10, 4),
            "test_recall_at_top10pct": round(test_recall_top10, 4),
            "b0_test_brier_score": round(b0_test_brier, 4),
            "brier_improvement_vs_b0_pct": round(
                (b0_test_brier - test_brier) / b0_test_brier * 100.0, 2
            ),
        },
    }

    # Write summary to output dir and model dir
    summary_json = json.dumps(summary, indent=2) + "\n"
    (output_dir / "evaluation_summary.json").write_text(summary_json, encoding="utf-8")
    (model_dir / "evaluation_summary.json").write_text(summary_json, encoding="utf-8")
    logger.info("Saved evaluation summary to %s", output_dir / "evaluation_summary.json")
    print(summary_json)


if __name__ == "__main__":
    main()
