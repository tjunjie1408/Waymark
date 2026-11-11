"""Executable contract loader and synthetic oracle verifier for Waymark (WM0)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field


class StageCostCeiling(BaseModel):
    model_config = ConfigDict(extra="allow")

    C1_s3_glue_athena_slice: float
    T6_T7_remote_sagemaker_training: float
    T8_batch_transform_and_trigger: float
    T9_logs_state_requests_and_buffer: float
    total_planned_ceiling_usd: float


class O7AwsAndCost(BaseModel):
    model_config = ConfigDict(extra="allow")

    account_id: str = Field(pattern=r"^[0-9]{12}$")
    cli_profile: str
    region: str
    total_budget_cap_usd: float = Field(le=30.0)
    pilot_target_burn_usd: float
    safety_margin_usd: float
    stage_cost_ceiling_usd: StageCostCeiling


class WM0Contract(BaseModel):
    """Top-level schema for configs/wm0_contract.yaml."""

    model_config = ConfigDict(extra="forbid")

    contract_id: str
    contract_version: str
    frozen_date: str
    timezone_display: str
    o1_source: dict[str, Any]
    o2_grain_and_cleaning: dict[str, Any]
    o3_prediction_and_targets: dict[str, Any]
    o4_splits_and_history: dict[str, Any]
    o5_features_and_numerics: dict[str, Any]
    o6_modeling_and_evaluation: dict[str, Any]
    o7_aws_and_cost: O7AwsAndCost
    o8_batch_delivery: dict[str, Any]


def load_wm0_contract(path: str | Path = "configs/wm0_contract.yaml") -> WM0Contract:
    """Load and validate the frozen WM0 contract YAML."""
    contract_path = Path(path)
    with contract_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    contract = WM0Contract.model_validate(raw)
    ceilings = contract.o7_aws_and_cost.stage_cost_ceiling_usd
    stage_sum = round(
        ceilings.C1_s3_glue_athena_slice
        + ceilings.T6_T7_remote_sagemaker_training
        + ceilings.T8_batch_transform_and_trigger
        + ceilings.T9_logs_state_requests_and_buffer,
        4,
    )
    if abs(stage_sum - ceilings.total_planned_ceiling_usd) > 1e-6:
        raise ValueError(
            f"Stage ceiling sum {stage_sum} != total_planned_ceiling_usd {ceilings.total_planned_ceiling_usd}"
        )
    if ceilings.total_planned_ceiling_usd > contract.o7_aws_and_cost.total_budget_cap_usd:
        raise ValueError("Planned ceiling exceeds total budget cap")
    return contract


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float, radius_km: float = 6371.0088) -> float:
    """Compute great-circle distance in kilometers using the WGS84 mean earth radius."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * math.asin(math.sqrt(a))


def load_discriminating_fixture(
    path: str | Path = "configs/t1_discriminating_fixture.json",
) -> dict[str, Any]:
    """Load the synthetic relational discriminating fixture JSON."""
    fixture_path = Path(path)
    with fixture_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
