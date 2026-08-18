"""
E2E: CLI smoke test — verify the paper-review entry point is installed and callable.

These tests run against the *installed* package (not via PYTHONPATH), so they
catch problems like missing ``[project.scripts]``, broken entry points, or
import errors that unit tests (``PYTHONPATH=src``) would miss.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e

# ── helpers ─────────────────────────────────────────────────────────────────


def _paper_review_bin() -> str:
    """Return the path to the installed ``paper-review`` executable."""
    # Prefer the same Python environment's bin dir
    bindir = Path(sys.executable).parent
    candidate = bindir / "paper-review"
    if candidate.exists():
        return str(candidate)
    # Fall back to PATH lookup
    which = subprocess.run(["which", "paper-review"], capture_output=True, text=True, check=False)
    if which.returncode == 0:
        return which.stdout.strip()
    # Last resort: module invocation (works regardless of entry-point registration)
    return f"{sys.executable} -m paper_review"


def _run(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        [_paper_review_bin(), *args],
        capture_output=True,
        text=True,
        check=check,
    )


# ── tests ────────────────────────────────────────────────────────────────────


class TestSmoke:
    """Smoke-level verification that the CLI is alive."""

    def test_help_exits_successfully(self):
        """``paper-review --help`` returns exit code 0."""
        result = _run("--help")
        assert result.returncode == 0

    def test_help_contains_expected_commands(self):
        """Help output lists core subcommands."""
        result = _run("--help")
        assert result.returncode == 0
        for cmd in ("index", "search", "status", "tags", "serve", "review"):
            assert cmd in result.stdout, f"Expected '{cmd}' in help output"

    def test_no_args_shows_help(self):
        """Running ``paper-review`` with no args prints help (not crash)."""
        result = _run(check=False)
        assert result.returncode == 0

    def test_module_invocation(self):
        """``python -m paper_review --help`` also works."""
        result = subprocess.run(
            [sys.executable, "-m", "paper_review", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "index" in result.stdout
