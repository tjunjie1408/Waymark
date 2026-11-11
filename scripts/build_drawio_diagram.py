"""Check or export the canonical editable architecture, without generating XML."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[1] / "docs/images/waymark_aws_architecture.drawio"


def check_diagram() -> None:
    """Check page identities, connector references and orthogonal routing."""
    pages = ET.parse(SOURCE).findall("diagram")
    if not pages:
        raise ValueError("Expected an uncompressed draw.io document with at least one page.")
    for page in pages:
        cells = page.findall("./mxGraphModel/root/mxCell")
        identifiers = [cell.get("id") for cell in cells]
        if not cells or None in identifiers or len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Missing or duplicate cell identities: {page.get('name')}.")
        for cell in cells:
            for reference in ("parent", "source", "target"):
                target = cell.get(reference)
                if target is not None and target not in identifiers:
                    raise ValueError(f"Unknown {reference} {target!r} on cell {cell.get('id')}.")
            if cell.get("edge") != "1":
                continue
            style = dict(item.split("=", 1) for item in cell.get("style", "").split(";") if "=" in item)
            if style.get("edgeStyle") != "orthogonalEdgeStyle" or style.get("rounded") != "0":
                raise ValueError(f"Non-orthogonal connector: {cell.get('id')}.")
            if any(port not in style for port in ("exitX", "exitY", "entryX", "entryY")):
                raise ValueError(f"Missing connector ports: {cell.get('id')}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check the source without rendering.")
    parser.add_argument("--format", choices=("png", "svg"), default="png")
    parser.add_argument("--drawio", help="Path to an installed draw.io desktop executable.")
    args = parser.parse_args()
    check_diagram()
    if args.check:
        print(f"Validated: {SOURCE}")
        return
    executable = args.drawio or shutil.which("drawio")
    if executable is None:
        parser.error("Use --drawio PATH or export the source through the draw.io MCP/editor.")
    output = SOURCE.with_suffix(f".{args.format}")
    command = [executable, "--export", "--format", args.format, "--output", str(output)]
    if args.format == "png":
        command.extend(["--scale", "2", "--border", "20"])
    subprocess.run([*command, str(SOURCE)], check=True)
    print(f"Rendered: {output}")


if __name__ == "__main__":
    main()
