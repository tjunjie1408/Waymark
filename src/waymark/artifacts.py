"""Immutable local artifacts and canonical content identities."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def write_immutable(path: Path, body: bytes) -> None:
    """Create once; identical retries are harmless, incompatible replacements fail."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".pending-", delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            if os.name == "nt":
                os.rename(temporary, path)  # Windows rename refuses an existing destination.
            else:
                os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != body:
                raise FileExistsError(f"Refusing to overwrite accepted artifact: {path}") from None
    finally:
        temporary.unlink(missing_ok=True)


def write_parquet_immutable(frame: Any, path: Path) -> None:
    write_immutable(path, frame.to_parquet(index=False, engine="pyarrow"))


def require_new_outputs(directory: Path, filenames: list[str]) -> None:
    existing = [str(directory / name) for name in filenames if (directory / name).exists()]
    if existing:
        raise FileExistsError(f"Use a new output directory; existing artifacts are protected: {existing}")
