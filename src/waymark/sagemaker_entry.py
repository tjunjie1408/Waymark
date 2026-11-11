"""Remote-only M1/M2 evaluation with shared references and explicit holdout authorization."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from waymark.artifacts import canonical_json, require_new_outputs, sha256_file, write_immutable
from waymark.evaluation import (
    baseline_predictions,  # noqa: F401
    binary_metrics,
    eta_metrics,
    paired_cluster_interval,  # noqa: F401
    ranking_metrics,
    top_k_alerts,
)
from waymark.evaluation import expected_calibration_error as expected_calibration_error
from waymark.evaluation import pinball_loss as pinball_loss
from waymark.feature_schema import CATEGORICAL_FEATURES, MODEL_FEATURES, NUMERICAL_FEATURES  # noqa: F401
from waymark.training_environment import capture_environment, verify_packaged_source

METHOD_ID = "waymark.training/v2"


def prepare_features(train: pd.DataFrame, frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if np.isinf(frame[NUMERICAL_FEATURES].to_numpy(dtype=float)).any():
        raise ValueError("Infinite model feature values are invalid; missing values remain explicit")
    encoded = frame[MODEL_FEATURES].copy()
    native = frame[MODEL_FEATURES].copy()
    for column in CATEGORICAL_FEATURES:
        categories = sorted(train[column].fillna("__MISSING__").astype(str).unique().tolist())
        encoded[column] = pd.Categorical(
            frame[column].fillna("__MISSING__").astype(str), categories=categories
        )
        native[column] = frame[column].fillna("__MISSING__").astype(str)
    return encoded, native


def validate_training_frame(frame: pd.DataFrame) -> None:
    required = set(
        MODEL_FEATURES
        + [
            "order_id",
            "split_role",
            "is_eligible_mature",
            "y_days",
            "y_late",
            "bootstrap_cluster_id",
            "order_approved_at",
            "order_delivered_customer_date",
        ]
    )
    if (
        not required.issubset(frame.columns)
        or frame["order_id"].isna().any()
        or frame["order_id"].duplicated().any()
    ):
        raise ValueError("Invalid training schema/order identities")
    for role in ["train", "val_calibration", "holdout_test"]:
        rows = frame.loc[frame["split_role"].eq(role)]
        if rows.empty or not rows["is_eligible_mature"].eq(1).all():
            raise ValueError(f"Missing or ineligible {role} population")
        if not np.isfinite(rows[["y_days", "y_late"]].to_numpy(dtype=float)).all():
            raise ValueError(f"Invalid labels in {role}")
        if not rows["y_late"].isin([0, 1]).all() or not rows["y_days"].gt(0).all():
            raise ValueError(f"Invalid target definition in {role}")
    train = frame.loc[frame.split_role.eq("train")]
    val = frame.loc[frame.split_role.eq("val_calibration")]
    test = frame.loc[frame.split_role.eq("holdout_test")]
    if (
        train.order_delivered_customer_date.max() > val.order_approved_at.min()
        or val.order_delivered_customer_date.max() > test.order_approved_at.min()
    ):
        raise ValueError("Immature labels cross temporal boundaries")


def save_frame(path: Path, frame: pd.DataFrame) -> dict[str, Any]:
    frame.to_parquet(path, index=False)
    return {"file": path.name, "sha256": sha256_file(path), "rows": len(frame)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument(
        "--training", default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--allow-holdout", choices=["true", "false"], default="false")
    parser.add_argument("--feature-sha256")
    parser.add_argument("--source-sha256")
    parser.add_argument("--snapshot-id")
    parser.add_argument("--image-uri", required=True)
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--preflight-output")
    args = parser.parse_args(argv)
    requirements = Path(__file__).with_name("remote_requirements.txt")
    if not requirements.is_file():
        requirements = Path(__file__).with_name("requirements.txt")
    source_inventory = verify_packaged_source(Path(__file__).resolve().parents[1])
    environment = capture_environment(args.image_uri, requirements)
    environment["source_inventory_sha256"] = source_inventory
    if args.preflight_only:
        if args.preflight_output:
            write_immutable(Path(args.preflight_output), canonical_json(environment))
        print(json.dumps(environment))
        return
    if not os.environ.get("SM_TRAINING_ENV") or not os.environ.get("SM_CURRENT_HOST"):
        raise RuntimeError("Model fitting is permitted only in a user-launched managed SageMaker job")
    if any(not value for value in [args.feature_sha256, args.source_sha256, args.snapshot_id]):
        raise ValueError("Training requires frozen source, input and snapshot identities")
    allow_holdout = args.allow_holdout == "true"
    start = time.monotonic()
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    require_new_outputs(model_dir, ["evaluation_summary.json", "development_freeze.json"])
    files = list(Path(args.training).glob("*.parquet"))
    if len(files) != 1 or sha256_file(files[0]) != args.feature_sha256:
        raise ValueError("Training input must be exactly the accepted feature Parquet")
    frame = pd.read_parquet(files[0])
    validate_training_frame(frame)
    train = frame.loc[frame.split_role.eq("train")].copy()
    val = frame.loc[frame.split_role.eq("val_calibration")].copy()
    x_train, cb_train = prepare_features(train, train)
    x_val, cb_val = prepare_features(train, val)
    identities = {
        "method_id": METHOD_ID,
        "feature_sha256": args.feature_sha256,
        "source_sha256": args.source_sha256,
        "snapshot_id": args.snapshot_id,
        "environment": environment,
        "random_seed": args.random_seed,
    }
    checkpoint_dir = Path(os.environ.get("SM_CHECKPOINT_DIR", "/opt/ml/checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_immutable(checkpoint_dir / "identity.json", canonical_json(identities))
    ledger = checkpoint_dir / "stages.jsonl"

    def fit_stage(name: str, model: Any, x: pd.DataFrame, y: Any, **kwargs: Any) -> Any:
        checkpoint = checkpoint_dir / f"{name}.pkl"
        if checkpoint.is_file():
            with checkpoint.open("rb") as stream:
                return pickle.load(stream)  # Own identity-bound completed-stage checkpoint only.
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": name, "status": "STARTED"}) + "\n")
        try:
            model.fit(x, y, **kwargs)
        except Exception:
            with ledger.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({"stage": name, "status": "FAILED"}) + "\n")
            raise
        temporary = checkpoint.with_suffix(".tmp")
        temporary.write_bytes(pickle.dumps(model))
        temporary.replace(checkpoint)
        with ledger.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"stage": name, "status": "COMPLETED"}) + "\n")
        return model

    from catboost import CatBoostClassifier, Pool
    from lightgbm import LGBMRegressor
    from sklearn.calibration import CalibratedClassifierCV
    from xgboost import XGBClassifier

    quantile_models = {}
    raw_val = []
    for alpha in [0.10, 0.50, 0.90]:
        name = f"m1_lgbm_p{int(100 * alpha)}"
        model = fit_stage(
            name,
            LGBMRegressor(
                objective="quantile",
                alpha=alpha,
                n_estimators=150,
                learning_rate=0.05,
                num_leaves=31,
                random_state=args.random_seed,
                n_jobs=-1,
            ),
            x_train,
            train.y_days,
            categorical_feature=CATEGORICAL_FEATURES,
        )
        quantile_models[name] = model
        raw_val.append(model.predict(x_val))
        (model_dir / f"{name}.pkl").write_bytes(pickle.dumps(model))
    raw_val_q = np.column_stack(raw_val)
    val_q = np.sort(raw_val_q, axis=1)
    raw_eta = eta_metrics(val.y_days.to_numpy(), val_q, raw_val_q)
    conformal_c = 0.0
    if abs(raw_eta["interval_coverage_80"] - 0.80) > 0.03:
        scores = np.maximum(val_q[:, 0] - val.y_days.to_numpy(), val.y_days.to_numpy() - val_q[:, 2])
        level = min(1.0, np.ceil((len(scores) + 1) * 0.80) / len(scores))
        conformal_c = max(0.0, float(np.quantile(scores, level, method="higher")))
    val_q[:, 0] -= conformal_c
    val_q[:, 2] += conformal_c

    cb = fit_stage(
        "m2_catboost",
        CatBoostClassifier(
            iterations=200, learning_rate=0.08, depth=6, random_seed=args.random_seed, verbose=False
        ),
        cb_train,
        train.y_late.astype(int),
        cat_features=CATEGORICAL_FEATURES,
    )
    xgb = fit_stage(
        "m2_xgboost",
        XGBClassifier(
            n_estimators=150,
            learning_rate=0.08,
            max_depth=5,
            random_state=args.random_seed,
            eval_metric="logloss",
            enable_categorical=True,
            tree_method="hist",
        ),
        x_train,
        train.y_late.astype(int),
    )
    raw_cb = cb.predict_proba(cb_val)[:, 1]
    raw_xgb = xgb.predict_proba(x_val)[:, 1]
    calibrated = CalibratedClassifierCV(cb, cv="prefit", method="isotonic")
    calibrated.fit(cb_val, val.y_late.astype(int))
    probabilities = calibrated.predict_proba(cb_val)[:, 1]
    threshold = max(
        np.linspace(0.1, 0.9, 81),
        key=lambda t: binary_metrics(val.y_late.to_numpy(), probabilities, probabilities >= t)["f1_at_tau"],
    )
    cb.save_model(str(model_dir / "m2_catboost.cbm"))
    xgb.save_model(str(model_dir / "m2_xgboost.json"))
    (model_dir / "m2_calibrated_catboost.pkl").write_bytes(pickle.dumps(calibrated))
    categories = {c: list(x_train[c].cat.categories) for c in CATEGORICAL_FEATURES}
    freeze = {
        **identities,
        "categorical_levels_from_train": categories,
        "model_features": MODEL_FEATURES,
        "excluded_identity_feature": "primary_seller_id",
        "conformal_c": conformal_c,
        "threshold_tau": float(threshold),
        "selected_classifier": "CatBoost predeclared; XGBoost development comparator",
        "crossing_repair": "row_sort_then_nonnegative_conformal_expansion",
        "bootstrap_replicates": 1000,
        "holdout_authorized": allow_holdout,
    }
    write_immutable(model_dir / "development_freeze.json", canonical_json(freeze))
    scope = "holdout_test" if allow_holdout else "val_calibration"
    scored = frame.loc[frame.split_role.eq(scope)].copy()
    encoded, native = prepare_features(train, scored)
    raw_q = np.column_stack([model.predict(encoded) for model in quantile_models.values()])
    q = np.sort(raw_q, axis=1)
    q[:, 0] -= conformal_c
    q[:, 2] += conformal_c
    prob = calibrated.predict_proba(native)[:, 1]
    y_eta, y_late = scored.y_days.to_numpy(), scored.y_late.to_numpy()
    alerts = top_k_alerts(prob, scored.order_id.to_numpy())
    references = baseline_predictions(train, scored)
    eta = eta_metrics(y_eta, q, raw_q)
    risk = binary_metrics(y_late, prob, prob >= threshold)
    budget_metrics = binary_metrics(y_late, prob, alerts)
    predictions = scored[["order_id", "bootstrap_cluster_id", "y_days", "y_late", "state_corridor"]].copy()
    for i, label in enumerate(["p10", "p50", "p90"]):
        predictions[f"raw_eta_{label}_days"] = raw_q[:, i]
        predictions[f"eta_{label}_days"] = q[:, i]
    predictions["prob_late"] = prob
    predictions["risk_alert_flag"] = (prob >= threshold).astype(int)
    predictions["top10_alert_flag"] = alerts.astype(int)
    for column in references:
        predictions[column] = references[column]
    inventory = [save_frame(model_dir / "predictions.parquet", predictions)]
    background = cb_train.sample(n=min(1000, len(train)), random_state=args.random_seed)
    explain = native.sample(n=min(1000, len(native)), random_state=args.random_seed)
    values = cb.get_feature_importance(
        data=Pool(explain, cat_features=CATEGORICAL_FEATURES),
        reference_data=Pool(background, cat_features=CATEGORICAL_FEATURES),
        type="ShapValues",
    )
    attribution = pd.DataFrame(values, columns=[*MODEL_FEATURES, "base_value"])
    attribution.insert(0, "order_id", scored.loc[explain.index, "order_id"].to_numpy())
    inventory.append(save_frame(model_dir / "shap_values.parquet", attribution))
    background = background.copy()
    background.insert(0, "order_id", train.loc[background.index, "order_id"].to_numpy())
    inventory.append(save_frame(model_dir / "shap_background.parquet", background))
    b0_error = abs(y_eta - references.b0_promised.to_numpy())
    b1_error = abs(y_eta - references.b1_p50.to_numpy())
    risk_baseline = binary_metrics(
        y_late, references.b0_late_prob.to_numpy(), references.b0_late_prob.to_numpy() >= threshold
    )
    intervals = {
        "M1_minus_B0_MAE": paired_cluster_interval(
            abs(y_eta - q[:, 1]) - b0_error, scored.bootstrap_cluster_id.to_numpy(), seed=args.random_seed
        ),
        "M1_minus_B1_MAE": paired_cluster_interval(
            abs(y_eta - q[:, 1]) - b1_error, scored.bootstrap_cluster_id.to_numpy(), seed=args.random_seed
        ),
        "M2_minus_B0_Brier": paired_cluster_interval(
            (prob - y_late) ** 2 - (references.b0_late_prob.to_numpy() - y_late) ** 2,
            scored.bootstrap_cluster_id.to_numpy(),
            seed=args.random_seed,
        ),
    }
    strata = []
    for corridor, indices in scored.reset_index(drop=True).groupby("state_corridor").groups.items():
        idx = np.asarray(indices)
        strata.append(
            {
                "state_corridor": corridor,
                "orders": len(idx),
                "eta": eta_metrics(y_eta[idx], q[idx], raw_q[idx]),
                "risk": binary_metrics(y_late[idx], prob[idx], prob[idx] >= threshold),
            }
        )
    summary = {
        "schema": "waymark.training_evaluation/v2",
        "status": "OK",
        "stage": "T6_T7",
        "gate": "WM3_REVIEW_REQUIRED" if allow_holdout else "WM3_INCOMPLETE",
        "gate_wm3_passed": None,
        "evaluation_scope": scope,
        "acceptance_owner": "user",
        "identities": identities,
        "runtime_seconds": time.monotonic() - start,
        "eta": eta,
        "risk": risk,
        "alert_budget": budget_metrics,
        "development": {
            "raw_eta": raw_eta,
            "catboost_raw": binary_metrics(val.y_late.to_numpy(), raw_cb, raw_cb >= threshold),
            "xgboost_raw": binary_metrics(val.y_late.to_numpy(), raw_xgb, raw_xgb >= threshold),
            "calibration_evaluation_is_in_sample": True,
        },
        "baselines": {
            "B0_promised_mae": float(b0_error.mean()),
            "B1_eta": eta_metrics(y_eta, references[["b1_p10", "b1_p50", "b1_p90"]].to_numpy()),
            "B0_risk": risk_baseline,
            "B1_eta_margin": ranking_metrics(y_late, references.b1_eta_margin.to_numpy()),
        },
        "uncertainty": intervals,
        "strata": strata,
        "artifacts": inventory,
        "checkpoint_semantics": "completed fit stages restored; interrupted fits restart, overhead unknown",
        "shap_interpretation": (
            "associations only; raw CatBoost log-odds, not calibrated probabilities or causes"
        ),
    }
    summary["model_artifacts"] = [
        {"file": path.name, "sha256": sha256_file(path)}
        for path in sorted(model_dir.iterdir())
        if path.is_file()
    ]
    write_immutable(model_dir / "evaluation_summary.json", canonical_json(summary))
    print(json.dumps(summary, allow_nan=False))


if __name__ == "__main__":
    main()
