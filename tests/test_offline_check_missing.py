"""
Tests for scripts/_check_missing.py — wheel name parsing and dependency matching.

These tests cover the offline_pack.sh dependency: when pip download fails
binary-only, _check_missing.py detects which packages are missing so the
pack script can retry with source tarballs.
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

# Import under test
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _check_missing import get_installed_wheel_names, get_tar_gz_names, parse_dep_name


class TestParseDepName:
    """PEP 508 dependency string → normalized package name."""

    def test_simple_name(self):
        assert parse_dep_name("pymupdf") == "pymupdf"

    def test_with_version(self):
        assert parse_dep_name("pymupdf>=1.23.0") == "pymupdf"

    def test_with_extras(self):
        assert parse_dep_name("pydantic[email]>=2.0.0") == "pydantic"

    def test_with_markers(self):
        assert parse_dep_name('typer>=0.9.0; python_version>="3.10"') == "typer"

    def test_with_semicolon(self):
        assert parse_dep_name("flask>=2.3.0; sys_platform!='win32'") == "flask"

    def test_case_insensitive(self):
        assert parse_dep_name("PyMuPDF>=1.23.0") == "pymupdf"

    def test_whitespace_stripped(self):
        assert parse_dep_name("  pymupdf >= 1.23.0  ") == "pymupdf"


class TestGetInstalledWheelNames:
    """Wheel filename → normalized package name extraction."""

    def test_single_wheel(self, tmp_path: Path):
        (tmp_path / "pymupdf-1.23.0-cp310-cp310-manylinux_2_17_x86_64.whl").write_text("")
        names = get_installed_wheel_names(str(tmp_path))
        assert names == {"pymupdf"}

    def test_underscore_to_hyphen(self, tmp_path: Path):
        """Package names with underscores are normalized to hyphens."""
        (tmp_path / "rank_bm25-0.2.0-py3-none-any.whl").write_text("")
        names = get_installed_wheel_names(str(tmp_path))
        assert "rank-bm25" in names

    def test_multiple_wheels(self, tmp_path: Path):
        (tmp_path / "pymupdf-1.23.0-cp310-cp310-manylinux.whl").write_text("")
        (tmp_path / "typer-0.9.0-py3-none-any.whl").write_text("")
        names = get_installed_wheel_names(str(tmp_path))
        assert names == {"pymupdf", "typer"}

    def test_ignores_non_wheel_files(self, tmp_path: Path):
        (tmp_path / "pymupdf-1.23.0.tar.gz").write_text("")
        (tmp_path / "some_file.txt").write_text("")
        names = get_installed_wheel_names(str(tmp_path))
        assert names == set()


class TestGetTarGzNames:
    """Source tarball filename → normalized package name extraction."""

    def test_single_tarball(self, tmp_path: Path):
        (tmp_path / "pymupdf-1.23.0.tar.gz").write_text("")
        names = get_tar_gz_names(str(tmp_path))
        assert names == {"pymupdf"}

    def test_ignores_whl(self, tmp_path: Path):
        (tmp_path / "pymupdf-1.23.0-cp310-cp310-manylinux.whl").write_text("")
        names = get_tar_gz_names(str(tmp_path))
        assert names == set()


class TestMainIntegration:
    """End-to-end: _check_missing.py main() with a real pyproject.toml."""

    def test_all_present(self, tmp_path: Path, monkeypatch):
        """main() exits 0 when all deps have wheels."""
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        (wheel_dir / "pymupdf-1.23.0-cp310-cp310-manylinux.whl").write_text("")
        (wheel_dir / "typer-0.9.0-py3-none-any.whl").write_text("")

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            textwrap.dedent("""\
            [project]
            name = "test"
            version = "0.1.0"
            dependencies = [
              "pymupdf>=1.23.0",
              "typer>=0.9.0",
            ]
        """)
        )

        monkeypatch.setattr("sys.argv", ["_check_missing.py", str(wheel_dir), str(pyproject)])
        # Should not raise SystemExit(1)
        from _check_missing import main

        main()

    def test_missing_detected(self, tmp_path: Path, monkeypatch):
        """main() exits 1 when a dependency is missing."""
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        (wheel_dir / "pymupdf-1.23.0-cp310-cp310-manylinux.whl").write_text("")
        # typer wheel is missing

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            textwrap.dedent("""\
            [project]
            name = "test"
            version = "0.1.0"
            dependencies = [
              "pymupdf>=1.23.0",
              "typer>=0.9.0",
            ]
        """)
        )

        monkeypatch.setattr("sys.argv", ["_check_missing.py", str(wheel_dir), str(pyproject)])
        from _check_missing import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 1

    def test_list_missing_output(self, tmp_path: Path, monkeypatch, capsys):
        """--list-missing prints missing deps on stdout."""
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        (wheel_dir / "pymupdf-1.23.0-cp310-cp310-manylinux.whl").write_text("")
        # typer and flask are missing

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            textwrap.dedent("""\
            [project]
            name = "test"
            version = "0.1.0"
            dependencies = [
              "pymupdf>=1.23.0",
              "typer>=0.9.0",
              "flask>=2.3.0",
            ]
        """)
        )

        monkeypatch.setattr(
            "sys.argv",
            ["_check_missing.py", "--list-missing", str(wheel_dir), str(pyproject)],
        )
        from _check_missing import main

        main()
        captured = capsys.readouterr()
        assert "typer" in captured.out
        assert "flask" in captured.out
        assert "pymupdf" not in captured.out

    def test_tar_gz_as_fallback(self, tmp_path: Path, monkeypatch):
        """main() exits 0 when dep is covered by .tar.gz fallback."""
        wheel_dir = tmp_path / "wheels"
        wheel_dir.mkdir()
        (wheel_dir / "pymupdf-1.23.0.tar.gz").write_text("")

        pyproject = tmp_path / "pyproject.toml"
        pyproject.write_text(
            textwrap.dedent("""\
            [project]
            name = "test"
            version = "0.1.0"
            dependencies = [
              "pymupdf>=1.23.0",
            ]
        """)
        )

        monkeypatch.setattr("sys.argv", ["_check_missing.py", str(wheel_dir), str(pyproject)])
        from _check_missing import main

        main()  # Should not raise
