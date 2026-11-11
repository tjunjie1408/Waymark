"""No-fit environment receipts and packaged source rejection checks."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from waymark.artifacts import sha256_file
from waymark.training import create_sourcedir_tarball
from waymark.training_environment import (
    capture_environment,
    requirement_versions,
    validate_environment,
    verify_packaged_source,
)

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "149270211548.dkr.ecr.ap-southeast-2.amazonaws.com/test@sha256:" + "a" * 64
REQUIREMENTS = ROOT / "src/waymark/remote_requirements.txt"


def test_environment_receipt_requires_real_interpreter_versions_and_imports() -> None:
    expected = requirement_versions(REQUIREMENTS)
    # Explicitly fake metadata/import responses, not an actual container compatibility claim.
    with (
        patch("waymark.training_environment.importlib.metadata.version", side_effect=expected.__getitem__),
        patch("waymark.training_environment.importlib.import_module") as imported,
    ):
        receipt = capture_environment(IMAGE, REQUIREMENTS)
    assert imported.call_count == len(expected)
    assert validate_environment(receipt, REQUIREMENTS) == IMAGE
    receipt["versions"]["numpy"] = "incompatible"
    with pytest.raises(ValueError, match="incompatible"):
        validate_environment(receipt, REQUIREMENTS)


def test_old_python_and_mutable_image_tags_cannot_obtain_receipts() -> None:
    with pytest.raises(ValueError, match="digest"):
        capture_environment(IMAGE.split("@")[0] + ":latest", REQUIREMENTS)
    with patch("waymark.training_environment.sys", SimpleNamespace(version_info=(3, 8))):
        with pytest.raises(ValueError, match="Python 3.12"):
            capture_environment(IMAGE, REQUIREMENTS)


def test_source_inventory_verifies_packaged_files_and_rejects_changes(tmp_path: Path) -> None:
    with tarfile.open(fileobj=io.BytesIO(create_sourcedir_tarball(ROOT)), mode="r:gz") as archive:
        for member in archive.getmembers():
            path = (tmp_path / member.name).resolve()
            assert path.is_relative_to(tmp_path.resolve())
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(archive.extractfile(member).read())
    assert verify_packaged_source(tmp_path) == sha256_file(tmp_path / "source_manifest.json")
    (tmp_path / "train_entry.py").write_text("corruption", encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch"):
        verify_packaged_source(tmp_path)
