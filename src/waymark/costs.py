"""Usage-based estimates with explicit prices; these are never invoiced costs."""

from __future__ import annotations

import math
from typing import Any


def athena_billable_bytes(scanned: int, statement_type: str = "DML") -> int:
    if scanned < 0:
        raise ValueError("Scanned bytes cannot be negative")
    if statement_type == "DDL":
        return 0
    if statement_type not in {"DML", "UTILITY"}:
        raise ValueError(f"Unknown Athena statement type: {statement_type}")
    return max(10_000_000, math.ceil(scanned / 1_000_000) * 1_000_000)


def usage_estimate(quantity: float, unit: str, rate: float | None, rate_source: str) -> dict[str, Any]:
    if not math.isfinite(quantity) or quantity < 0:
        raise ValueError("Usage quantity must be finite and nonnegative")
    if rate is not None and (not math.isfinite(rate) or rate <= 0 or not rate_source.strip()):
        raise ValueError("A positive rate requires its dated regional source")
    return {
        "evidence": "Inferred" if rate is not None else "Unknown",
        "quantity": quantity,
        "unit": unit,
        "rate_usd": rate,
        "rate_source": rate_source or None,
        "estimated_cost_usd": None if rate is None else quantity * rate,
        "billing_status": "ESTIMATE_NOT_INVOICE" if rate is not None else "UNPRICED_USAGE",
    }


def training_cost(
    billable_seconds: int, instances: int, on_demand_hourly_rate: float | None, rate_source: str = ""
) -> dict[str, Any]:
    if instances < 1:
        raise ValueError("Instance count must be positive")
    return usage_estimate(
        billable_seconds * instances / 3600.0, "billable_instance_hours", on_demand_hourly_rate, rate_source
    )
