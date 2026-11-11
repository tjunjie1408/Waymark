"""Shared, no-fit reference predictions and evaluation calculations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _vectors(y_true: np.ndarray, prediction: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y, p = np.asarray(y_true, dtype=float), np.asarray(prediction, dtype=float)
    if (
        y.ndim != 1
        or y.shape != p.shape
        or not len(y)
        or not np.isfinite(y).all()
        or not np.isfinite(p).all()
    ):
        raise ValueError("Expected nonempty, matching finite prediction vectors")
    return y, p


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, alpha: float) -> float:
    y, p = _vectors(y_true, y_pred)
    if not 0 < alpha < 1:
        raise ValueError("Quantile level must be inside (0, 1)")
    difference = y - p
    return float(np.mean(np.maximum(alpha * difference, (alpha - 1) * difference)))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    y, p = _vectors(y_true, y_prob)
    if n_bins < 1 or not np.isin(y, [0, 1]).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid binary labels, probabilities or bin count")
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    return float(
        sum(
            np.mean(bins == i) * abs(np.mean(p[bins == i]) - np.mean(y[bins == i]))
            for i in range(n_bins)
            if np.any(bins == i)
        )
    )


def top_k_alerts(probabilities: np.ndarray, order_ids: np.ndarray, fraction: float = 0.10) -> np.ndarray:
    p = np.asarray(probabilities, dtype=float)
    ids = np.asarray(order_ids, dtype=str)
    if p.ndim != 1 or p.shape != ids.shape or not np.isfinite(p).all() or not 0 <= fraction <= 1:
        raise ValueError("Invalid alert inputs")
    if len(set(ids)) != len(ids) or ((p < 0) | (p > 1)).any():
        raise ValueError("Alert inputs require unique order IDs and valid probabilities")
    selected = np.zeros(len(p), dtype=bool)
    selected[np.lexsort((ids, -p))[: int(np.floor(len(p) * fraction))]] = True
    return selected


def baseline_predictions(train: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """Frozen O6: corridor quantiles with n<20 global fallback; risk prior weight 5."""
    if train.empty or train[["y_days", "y_late"]].isna().any().any():
        raise ValueError("Reference baselines require mature train labels")
    global_quantiles = np.quantile(train["y_days"], [0.10, 0.50, 0.90])
    global_late = float(train["y_late"].mean())
    groups = train.groupby("state_corridor")["y_days"]
    counts = groups.count()
    out = pd.DataFrame(index=target.index)
    out["b0_promised"] = target["promised_days"]
    for level, global_value in zip([10, 50, 90], global_quantiles, strict=True):
        values = groups.quantile(level / 100).where(counts >= 20, global_value)
        out[f"b1_p{level}"] = target["state_corridor"].map(values).fillna(global_value).astype(float)
    late_sums = train.groupby("state_corridor")["y_late"].sum()
    smoothed = (late_sums + 5.0 * global_late) / (counts + 5.0)
    out["b0_late_prob"] = target["state_corridor"].map(smoothed).fillna(global_late).astype(float)
    out["b1_eta_margin"] = out["b1_p90"] - target["promised_days"]
    return out


def eta_metrics(
    y: np.ndarray, quantiles: np.ndarray, raw_quantiles: np.ndarray | None = None
) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    q = np.asarray(quantiles, dtype=float)
    if q.shape != (len(y), 3) or not len(y) or not np.isfinite(q).all() or not np.isfinite(y).all():
        raise ValueError("Invalid quantile predictions")
    if ((q[:, 0] > q[:, 1]) | (q[:, 1] > q[:, 2])).any():
        raise ValueError("Invalid/crossed prediction interval")
    raw = q if raw_quantiles is None else np.asarray(raw_quantiles, dtype=float)
    if raw.shape != q.shape or not np.isfinite(raw).all():
        raise ValueError("Invalid raw quantile predictions")
    losses = [pinball_loss(y, q[:, i], alpha) for i, alpha in enumerate([0.10, 0.50, 0.90])]
    return {
        "mae_p50_days": float(np.mean(abs(y - q[:, 1]))),
        **{f"pinball_loss_p{level}": loss for level, loss in zip([10, 50, 90], losses, strict=True)},
        "mean_pinball_loss": float(np.mean(losses)),
        "interval_coverage_80": float(np.mean((q[:, 0] <= y) & (y <= q[:, 2]))),
        "interval_mean_width_days": float(np.mean(q[:, 2] - q[:, 0])),
        "quantile_crossing_rate": float(np.mean((raw[:, 0] > raw[:, 1]) | (raw[:, 1] > raw[:, 2]))),
    }


def ranking_metrics(y: np.ndarray, scores: np.ndarray) -> dict[str, float | None]:
    """ROC AUC with average ranks; PR AUC means stepwise average precision, including ties."""
    y, p = _vectors(y, scores)
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Expected binary labels")
    positives = int(y.sum())
    negatives = len(y) - positives
    ranks = pd.Series(p).rank(method="average").to_numpy()
    auc = (
        None
        if not positives or not negatives
        else float((ranks[y == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives))
    )
    grouped = pd.DataFrame({"score": p, "y": y}).groupby("score")["y"].agg(["sum", "count"])
    grouped = grouped.sort_index(ascending=False)
    precision = grouped["sum"].cumsum() / grouped["count"].cumsum()
    ap = None if not positives else float((precision * grouped["sum"] / positives).sum())
    return {"roc_auc": auc, "pr_auc": ap}


def binary_metrics(y: np.ndarray, probability: np.ndarray, alerts: np.ndarray) -> dict[str, Any]:
    y, p = _vectors(y, probability)
    if ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be in [0, 1]")
    alerts = np.asarray(alerts, dtype=bool)
    if alerts.shape != y.shape:
        raise ValueError("Alert shape mismatch")
    tp = int(np.sum((y == 1) & alerts))
    precision = tp / int(alerts.sum()) if alerts.any() else 0.0
    recall = tp / int(y.sum()) if y.sum() else 0.0
    return {
        **ranking_metrics(y, p),
        "brier_score": float(np.mean((p - y) ** 2)),
        "expected_calibration_error_10bin": expected_calibration_error(y, p),
        "precision_at_tau": precision,
        "recall_at_tau": recall,
        "f1_at_tau": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "alert_count": int(alerts.sum()),
        "alert_fraction": float(np.mean(alerts)),
    }


def paired_cluster_interval(
    differences: np.ndarray, clusters: np.ndarray, replicates: int = 1000, seed: int = 42
) -> dict[str, Any]:
    d, c = np.asarray(differences, dtype=float), np.asarray(clusters, dtype=str)
    if d.ndim != 1 or d.shape != c.shape or not len(d) or not np.isfinite(d).all() or replicates < 2:
        raise ValueError("Invalid paired bootstrap inputs")
    labels, inverse = np.unique(c, return_inverse=True)
    sums = np.bincount(inverse, weights=d)
    sizes = np.bincount(inverse)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(replicates):
        sample = rng.integers(0, len(labels), size=len(labels))
        means.append(float(sums[sample].sum() / sizes[sample].sum()))
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_loss_difference_candidate_minus_baseline": float(d.mean()),
        "ci95": [float(low), float(high)],
        "replicates": replicates,
        "cluster_count": len(labels),
        "seed": seed,
        "unit": "primary_seller_id_and_approval_iso_week",
    }
