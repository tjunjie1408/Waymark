"""Prediction-only SageMaker BYOC server; never fits or selects a model."""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import pickle
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from waymark.batch_contract import canonical_bytes, digest
from waymark.feature_schema import CATEGORICAL_FEATURES, MODEL_FEATURES, encode_with_levels
from waymark.training_environment import capture_environment


def input_fn(body: bytes, content_type: str) -> pd.DataFrame:
    if content_type.split(";", 1)[0] != "text/csv":
        raise ValueError("Inference requires headerless text/csv")
    rows = list(csv.reader(io.StringIO(body.decode("utf-8"))))
    if not rows or any(len(row) != len(MODEL_FEATURES) + 1 for row in rows):
        raise ValueError("Invalid inference CSV schema")
    frame = pd.DataFrame(rows, columns=["order_id", *MODEL_FEATURES])
    if frame.order_id.eq("").any() or frame.order_id.duplicated().any():
        raise ValueError("Invalid inference order identities")
    for column in MODEL_FEATURES:
        if column not in CATEGORICAL_FEATURES:
            frame[column] = [float(value) if value else np.nan for value in frame[column]]
    return frame


def predict_fn(frame: pd.DataFrame, model: dict[str, Any]) -> list[dict[str, Any]]:
    freeze = model["freeze"]
    encoded, native = encode_with_levels(frame, freeze["categorical_levels_from_train"])
    raw = np.column_stack([value.predict(encoded) for value in model["quantiles"]]).astype(float)
    quantiles = np.sort(raw, axis=1)
    correction, threshold = freeze["conformal_c"], freeze["threshold_tau"]
    if not math.isfinite(correction) or correction < 0 or not 0 <= threshold <= 1:
        raise ValueError("Invalid frozen calibration values")
    quantiles[:, 0] -= correction
    quantiles[:, 2] += correction
    probabilities = model["risk"].predict_proba(native)[:, 1]
    if (
        not np.isfinite(quantiles).all()
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities > 1).any()
    ):
        raise ValueError("Invalid model prediction values")
    result = frame[["order_id"]].copy()
    for index, name in enumerate(["p10", "p50", "p90"]):
        result[f"eta_{name}_days"] = quantiles[:, index]
    result["prob_late"] = probabilities
    result["risk_alert_flag"] = (probabilities >= threshold).astype(int)
    return result.to_dict("records")


def output_fn(records: list[dict[str, Any]], accept: str) -> bytes:
    if accept != "application/jsonlines":
        raise ValueError("Inference output requires application/jsonlines")
    return b"\n".join(canonical_bytes(record) for record in records) + b"\n"


def model_fn(directory: Path, environment: dict[str, Any]) -> dict[str, Any]:
    summary = json.loads((directory / "evaluation_summary.json").read_text())
    inventory = {item["file"]: item["sha256"] for item in summary["model_artifacts"]}
    names = [
        "development_freeze.json",
        "m1_lgbm_p10.pkl",
        "m1_lgbm_p50.pkl",
        "m1_lgbm_p90.pkl",
        "m2_calibrated_catboost.pkl",
    ]
    for name in names:
        if digest((directory / name).read_bytes()) != inventory.get(name):
            raise ValueError(f"Serving model file identity mismatch: {name}")
    freeze = json.loads((directory / "development_freeze.json").read_text())
    if (
        freeze["model_features"] != MODEL_FEATURES
        or freeze["environment"]["versions"] != environment["versions"]
        or set(freeze["categorical_levels_from_train"]) != set(CATEGORICAL_FEATURES)
    ):
        raise ValueError("Serving model schema/environment mismatch")
    # Only the operator's accepted, hash-verified artifact is ever deserialized.
    return {
        "freeze": freeze,
        "quantiles": [pickle.loads((directory / name).read_bytes()) for name in names[1:4]],
        "risk": pickle.loads((directory / names[4]).read_bytes()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", nargs="?", choices=["serve"], default="serve")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--image-uri", default=os.environ.get("WAYMARK_SERVING_IMAGE_URI"))
    args = parser.parse_args()
    environment = capture_environment(
        args.image_uri or "", Path(__file__).with_name("remote_requirements.txt")
    )
    if args.preflight_only:
        print(json.dumps(environment))
        return
    model = model_fn(Path("/opt/ml/model"), environment)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200 if self.path == "/ping" else 404)
            self.end_headers()

        def do_POST(self):  # noqa: N802
            if self.path != "/invocations":
                self.send_error(404)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 6 * 1024 * 1024:
                    raise ValueError("Invalid inference request length")
                body = self.rfile.read(size)
                output = output_fn(
                    predict_fn(input_fn(body, self.headers.get("Content-Type", "")), model),
                    self.headers.get("Accept", ""),
                )
            except (ValueError, KeyError, UnicodeError):
                self.send_error(400, "Invalid inference payload")
                return
            self.send_response(200)
            self.send_header("Content-Type", "application/jsonlines")
            self.send_header("Content-Length", str(len(output)))
            self.end_headers()
            self.wfile.write(output)

        def log_message(self, format, *args):
            return  # Do not log order records or request bodies.

    HTTPServer(("0.0.0.0", 8080), Handler).serve_forever()


if __name__ == "__main__":
    main()
