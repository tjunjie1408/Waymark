"""No-fit environment evidence bound to an immutable training image and requirements."""

from __future__ import annotations

import importlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path
from typing import Any

from waymark.artifacts import sha256_file

IMAGE_PATTERN = r"[0-9]{12}\.dkr\.ecr\.ap-southeast-2\.amazonaws\.com/[a-z0-9_./-]+@sha256:[0-9a-f]{64}"


def verify_packaged_source(root: Path) -> str:
    manifest = root / "source_manifest.json"
    entries = json.loads(manifest.read_text(encoding="utf-8"))
    if not entries or "train_entry.py" not in entries or "requirements.txt" not in entries:
        raise ValueError("Invalid packaged source inventory")
    for name, digest in entries.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root.resolve()) or not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"Packaged source mismatch: {name}")
    return sha256_file(manifest)


def requirement_versions(path: Path) -> dict[str, str]:
    expected = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if not re.fullmatch(r"[a-zA-Z0-9_-]+==[a-zA-Z0-9_.]+", line):
            raise ValueError(f"Remote requirement must be pinned: {line}")
        name, version = line.split("==")
        expected[name] = version
    return expected


def capture_environment(image_uri: str, requirements: Path) -> dict[str, Any]:
    if not re.fullmatch(IMAGE_PATTERN, image_uri):
        raise ValueError("Use a reviewed regional ECR image pinned by sha256 digest")
    expected = requirement_versions(requirements)
    if sys.version_info[:2] != (3, 12):
        raise ValueError("Training environment must run Python 3.12")
    observed = {name: importlib.metadata.version(name) for name in expected}
    if observed != expected:
        raise ValueError(f"Training package versions differ: expected {expected}, got {observed}")
    for name in expected:
        importlib.import_module({"scikit-learn": "sklearn"}.get(name, name.replace("-", "_")))
    return {
        "schema": "waymark.environment/v1",
        "status": "PASS",
        "image_uri": image_uri,
        "python": platform.python_version(),
        "versions": observed,
        "requirements_sha256": sha256_file(requirements),
        "check": "installation_and_imports_only_no_fit",
    }


def validate_environment(receipt: dict[str, Any], requirements: Path) -> str:
    image = receipt.get("image_uri", "")
    if (
        receipt.get("schema") != "waymark.environment/v1"
        or receipt.get("status") != "PASS"
        or not re.fullmatch(IMAGE_PATTERN, image)
        or not str(receipt.get("python", "")).startswith("3.12.")
        or receipt.get("versions") != requirement_versions(requirements)
        or receipt.get("requirements_sha256") != sha256_file(requirements)
        or receipt.get("check") != "installation_and_imports_only_no_fit"
    ):
        raise ValueError("Missing/incompatible no-fit environment receipt; do not launch training")
    return image
