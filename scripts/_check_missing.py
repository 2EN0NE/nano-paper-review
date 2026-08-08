#!/usr/bin/env python3
"""
Helper: compare downloaded wheel directory against pyproject.toml dependencies.

Used by offline_pack.sh to detect which dependencies failed binary-only download.

Usage:
    python scripts/_check_missing.py <wheel_dir> <pyproject_toml_path>
    python scripts/_check_missing.py --list-missing <wheel_dir> <pyproject_toml_path>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # Python <3.11


def parse_dep_name(spec: str) -> str:
    """Extract the package name from a PEP 508 dependency string."""
    # Remove extras, version specifiers, markers
    name = re.split(r"[>=<!~[;]", spec)[0].strip()
    return name.lower()


def get_installed_wheel_names(wheel_dir: str) -> set[str]:
    """Get set of normalized package names from .whl files in a directory."""
    names: set[str] = set()
    for f in Path(wheel_dir).glob("*.whl"):
        # Wheel filename format: {distribution}-{version}(-{build tag})?-{python tag}-{abi tag}-{platform tag}.whl
        parts = f.name.split("-")
        if parts:
            names.add(parts[0].lower().replace("_", "-"))
    return names


def get_tar_gz_names(wheel_dir: str) -> set[str]:
    """Also check .tar.gz source archives as fallback."""
    names: set[str] = set()
    for f in Path(wheel_dir).glob("*.tar.gz"):
        parts = f.name.split("-")
        if parts:
            names.add(parts[0].lower().replace("_", "-"))
    return names


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("wheel_dir", help="Directory containing downloaded wheels")
    parser.add_argument("pyproject", help="Path to pyproject.toml")
    parser.add_argument(
        "--list-missing",
        action="store_true",
        help="Print missing package specs (one per line) instead of exit code",
    )
    args = parser.parse_args()

    with open(args.pyproject, "rb") as f:
        data = tomllib.load(f)

    deps: list[str] = data["project"]["dependencies"]
    installed = get_installed_wheel_names(args.wheel_dir) | get_tar_gz_names(args.wheel_dir)

    missing: list[str] = []
    for dep in deps:
        pkg_name = parse_dep_name(dep)
        if pkg_name and pkg_name not in installed:
            missing.append(dep)

    if args.list_missing:
        for dep in missing:
            print(dep)
        return

    if missing:
        print(f"Missing packages: {', '.join(missing)}")
        sys.exit(1)
    else:
        print("All dependencies accounted for.")


if __name__ == "__main__":
    main()
