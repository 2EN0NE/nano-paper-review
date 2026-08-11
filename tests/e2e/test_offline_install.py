"""
E2E: 离线安装 — 验证 install.sh --offline 和 offline_pack.sh

Seam 1 (快, pre-commit): install.sh --offline 脚本逻辑（轻量 fixture）
Seam 2 (慢, pre-push/CI): 完整 offline_pack.sh → install.sh --offline 链路
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


# ============================================================================
# 辅助函数
# ============================================================================


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _install_script() -> Path:
    return _project_root() / "scripts" / "install.sh"


def _offline_pack_script() -> Path:
    return _project_root() / "scripts" / "offline_pack.sh"


def _make_minimal_pyproject(path: Path, name: str = "test_pkg") -> None:
    """创建最小 pyproject.toml（零依赖，src-layout 避免误发现 offline_packages）。

    name 必须用下划线而非连字符：entry point 引用要求有效的 Python 模块名。
    """
    (path / "pyproject.toml").write_text(
        textwrap.dedent(f"""\
        [build-system]
        requires = ["setuptools>=68", "wheel"]
        build-backend = "setuptools.build_meta"

        [project]
        name = "{name}"
        version = "0.1.0"
        requires-python = ">=3.10"
        dependencies = []

        [project.scripts]
        {name} = "{name}:main"

        [tool.setuptools.packages.find]
        where = ["src"]
    """)
    )


def _make_minimal_src(path: Path, name: str = "test_pkg") -> None:
    """创建 src-layout 包结构（避免 setuptools 误发现其他目录）。"""
    src_dir = path / "src" / name
    src_dir.mkdir(parents=True)
    (src_dir / "__init__.py").write_text(
        textwrap.dedent(f"""\
        def main():
            print("hello-from-{name}")
    """)
    )


def _env_without_venv() -> dict:
    """返回不包含 VIRTUAL_ENV 的环境变量（让脚本走自动创建逻辑）。"""
    return {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}


def _build_fixture_wheel(cwd: Path, name: str = "test_pkg") -> None:
    """构造最小 wheel 到 offline_packages/，并附 setuptools + wheel 构建依赖。

    不依赖 uv venv 中的 pip（uv 默认不装 pip）。直接用系统 Python 下载。
    """
    import zipfile

    wheel_dir = cwd / "offline_packages"
    wheel_dir.mkdir(exist_ok=True)

    # 寻找带 pip 的 Python（优先 python3.10+）
    # 坑：CI 中 `uv run pytest` 会把项目根 .venv/bin 前置到 PATH，shutil.which 可能命中
    # uv venv 里的 python —— 而 uv 默认不装 pip，`-m pip download` 会直接失败。
    # 因此：1) 跳过 venv 内解释器；2) 必须验证 `-m pip --version` 可用才采用。
    pip_python = None
    for candidate in ("python3.10", "python3.11", "python3.12", "python3.13", "python3"):
        candidate_path = shutil.which(candidate)
        if not candidate_path or "/.venv/" in candidate_path:
            continue
        ver = subprocess.run(
            [
                candidate_path,
                "-c",
                "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
            ],
            capture_output=True,
            text=True,
        )
        if ver.returncode != 0:
            continue
        major, minor = ver.stdout.strip().split(".")
        if int(major) < 3 or int(minor) < 10:
            continue
        pip_ok = subprocess.run(
            [candidate_path, "-m", "pip", "--version"],
            capture_output=True,
            text=True,
        )
        if pip_ok.returncode != 0:
            continue
        pip_python = candidate_path
        break

    if pip_python:
        # 下载 setuptools + wheel（用旧版避免传递依赖 packaging>=24.0）
        subprocess.run(
            [
                pip_python,
                "-m",
                "pip",
                "download",
                "--no-deps",
                "setuptools==68.0.0",
                "wheel==0.38.4",
                "--dest",
                str(wheel_dir),
            ],
            capture_output=True,
            text=True,
            cwd=cwd,
            check=True,
        )

    dist_info = f"{name}-0.1.0.dist-info"
    wheel_name = f"{name}-0.1.0-py3-none-any.whl"
    wheel_path = wheel_dir / wheel_name

    # 构造 WHEEL 元数据
    wheel_meta = textwrap.dedent("""\
        Wheel-Version: 1.0
        Generator: test-fixture
        Root-Is-Purelib: true
        Tag: py3-none-any
    """)

    # 构造 METADATA
    metadata = textwrap.dedent(f"""\
        Metadata-Version: 2.1
        Name: {name}
        Version: 0.1.0
        Summary: test fixture
        Requires-Python: >=3.10
    """)

    # 构造 entry_points.txt
    entry_points = textwrap.dedent(f"""\
        [console_scripts]
        {name} = {name}:main
    """)

    # 构造 RECORD（不含 hash）
    record_lines = [
        f"{name}/__init__.py,,",
        f"{name}-0.1.0.dist-info/METADATA,,",
        f"{name}-0.1.0.dist-info/WHEEL,,",
        f"{name}-0.1.0.dist-info/entry_points.txt,,",
        f"{name}-0.1.0.dist-info/RECORD,,",
    ]

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as zf:
        # 包代码
        zf.writestr(f"{name}/__init__.py", f"def main(): print('hello-from-{name}')\n")
        # dist-info
        zf.writestr(f"{dist_info}/METADATA", metadata)
        zf.writestr(f"{dist_info}/WHEEL", wheel_meta)
        zf.writestr(f"{dist_info}/entry_points.txt", entry_points)
        zf.writestr(f"{dist_info}/RECORD", "\n".join(record_lines) + "\n")


# ============================================================================
# Seam 1: install.sh --offline 脚本逻辑（快 — 不需网络、无模型下载）
# ============================================================================


class TestInstallOfflineFast:
    """快速测试：install.sh --offline 行为（轻量 fixture，无网络）。"""

    def test_missing_offline_packages_fails(self, tmp_path: Path):
        """缺少 offline_packages/ 目录时应清晰报错退出。"""
        _make_minimal_pyproject(tmp_path)
        _make_minimal_src(tmp_path)
        shutil.copytree(_project_root() / "scripts", tmp_path / "scripts")

        result = subprocess.run(
            ["bash", str(tmp_path / "scripts" / "install.sh"), "--offline"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=_env_without_venv(),
        )
        assert result.returncode != 0
        combined = result.stdout + result.stderr
        assert "offline_packages" in combined

    def test_offline_install_minimal_package(self, tmp_path: Path):
        """install.sh --offline 安装一个无依赖的最小包。"""
        _make_minimal_pyproject(tmp_path)
        _make_minimal_src(tmp_path)

        # 构建 wheel
        _build_fixture_wheel(tmp_path)

        # 复制 scripts/
        shutil.copytree(_project_root() / "scripts", tmp_path / "scripts")

        # 运行 install.sh --offline（无预置 venv，让脚本自动创建）
        venv_dir = tmp_path / ".venv"
        result = subprocess.run(
            ["bash", str(tmp_path / "scripts" / "install.sh"), "--offline"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=_env_without_venv(),
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}\nstdout:\n{result.stdout}"
        assert venv_dir.is_dir(), ".venv 应该被自动创建"

        # 验证包已被安装
        python = str(venv_dir / "bin" / "python")
        result = subprocess.run(
            [python, "-c", "import test_pkg; test_pkg.main()"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
        )
        assert "hello-from-test_pkg" in result.stdout

    def test_offline_install_reuses_existing_venv(self, tmp_path: Path):
        """已有 .venv 时 install.sh --offline 应复用而非重建。"""
        _make_minimal_pyproject(tmp_path)
        _make_minimal_src(tmp_path)

        # 预建 venv
        venv_dir = tmp_path / ".venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            check=True,
        )

        # 构建 wheel
        _build_fixture_wheel(tmp_path)

        shutil.copytree(_project_root() / "scripts", tmp_path / "scripts")

        result = subprocess.run(
            ["bash", str(tmp_path / "scripts" / "install.sh"), "--offline"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=_env_without_venv(),
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
        # 应看到"已有"字样，不是"创建"
        assert "已有虚拟环境" in result.stdout

    def test_offline_install_respects_active_venv(self, tmp_path: Path):
        """已在 venv 中时 install.sh --offline 应直接使用当前环境。"""
        _make_minimal_pyproject(tmp_path)
        _make_minimal_src(tmp_path)

        # 预建 venv 并激活
        venv_dir = tmp_path / ".venv"
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            capture_output=True,
            check=True,
        )

        # 构建 wheel
        _build_fixture_wheel(tmp_path)

        shutil.copytree(_project_root() / "scripts", tmp_path / "scripts")

        # 传入 VIRTUAL_ENV 模拟已在 venv 中
        result = subprocess.run(
            ["bash", str(tmp_path / "scripts" / "install.sh"), "--offline"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**os.environ, "VIRTUAL_ENV": str(venv_dir)},
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"
        assert "已在虚拟环境中" in result.stdout

    def test_offline_install_copies_models(self, tmp_path: Path):
        """models/ 目录存在时 install.sh --offline 应拷贝到缓存路径。"""
        _make_minimal_pyproject(tmp_path)
        _make_minimal_src(tmp_path)

        # 构建 wheel
        _build_fixture_wheel(tmp_path)

        # 创建虚拟 models/ 目录
        model_name = "test-model"
        models_dir = tmp_path / "models" / model_name
        models_dir.mkdir(parents=True)
        (models_dir / "model.onnx").write_text("dummy-onnx-content")
        (models_dir / "config.json").write_text("{}")
        (models_dir / "tokenizer.json").write_text("{}")

        shutil.copytree(_project_root() / "scripts", tmp_path / "scripts")

        # 覆盖 HOME 以隔离缓存
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()
        expected_cache = fake_home / ".cache" / "paper-review" / "models"

        result = subprocess.run(
            ["bash", str(tmp_path / "scripts" / "install.sh"), "--offline"],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env={**os.environ, "HOME": str(fake_home)},
        )
        assert result.returncode == 0, f"stderr:\n{result.stderr}"

        # 验证模型文件被拷贝
        assert (expected_cache / model_name / "model.onnx").is_file()
        assert (expected_cache / model_name / "config.json").is_file()
        assert (expected_cache / model_name / "tokenizer.json").is_file()
        assert (expected_cache / model_name / "model.onnx").read_text() == "dummy-onnx-content"


# ============================================================================
# Seam 2: 完整 offline_pack.sh → install.sh --offline 链路（慢 — 需网络）
# ============================================================================


@pytest.mark.e2e_slow
class TestOfflineFullChain:
    """完整端到端：offline_pack.sh → tar → install.sh --offline → 验证。"""

    def test_full_offline_pack_and_install(self, tmp_path: Path):
        """运行 offline_pack.sh，解压，install --offline，验证 CLI。"""
        # wheels 按 manylinux x86_64 打包（offline_pack.sh 硬编码平台标签）：
        # 非 Linux x86_64 上 pip 找不到匹配轮子，必然失败 —— 直接跳过
        if not sys.platform.startswith("linux") or platform.machine() != "x86_64":
            pytest.skip("离线打包只支持 Linux x86_64（manylinux wheels）")
        # wheels 按 cp312 打包；venv 必须用 python3.12 创建，否则 pip 装不上
        py312 = shutil.which("python3.12")
        if py312 is None:
            pytest.skip("需要 python3.12（离线包 wheels 按 cp312 打包）")

        project_root = _project_root()

        # Step 1: 打包
        output_dir = tmp_path / "output"
        output_dir.mkdir()
        result = subprocess.run(
            ["bash", str(_offline_pack_script()), "--output-dir", str(output_dir)],
            capture_output=True,
            text=True,
            cwd=project_root,
            timeout=1200,  # 20 分钟：下载 wheel + 模型
        )
        assert result.returncode == 0, (
            f"offline_pack.sh 失败:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )

        # 找到输出 tarball（脚本放在 OUTPUT_DIR 的上级，与默认 dist/ 对齐）
        tarballs = list(output_dir.parent.glob("paper-review-offline-*.tar.gz"))
        assert len(tarballs) == 1, f"期望 1 个 tarball，得到: {tarballs}"
        tarball = tarballs[0]

        # Step 2: 解压 → 验证目录结构
        extract_dir = tmp_path / "extract"
        extract_dir.mkdir()
        subprocess.run(["tar", "xzf", str(tarball), "-C", str(extract_dir)], check=True)

        pack_dir = extract_dir / "paper-review-offline"
        assert pack_dir.is_dir()
        assert (pack_dir / "offline_packages").is_dir()
        assert (pack_dir / "scripts" / "install.sh").is_file()
        assert (pack_dir / "models").is_dir()
        assert (pack_dir / "src").is_dir()
        assert (pack_dir / "pyproject.toml").is_file()

        # Step 3: 离线安装（隔离 HOME，避免污染真实 ~/.cache）
        fake_home = tmp_path / "fake-home"
        fake_home.mkdir()
        venv_dir = tmp_path / "venv"
        subprocess.run(
            [py312, "-m", "venv", str(venv_dir)],
            capture_output=True,
            check=True,
        )

        result = subprocess.run(
            ["bash", str(pack_dir / "scripts" / "install.sh"), "--offline"],
            capture_output=True,
            text=True,
            cwd=pack_dir,
            env={**os.environ, "VIRTUAL_ENV": str(venv_dir), "HOME": str(fake_home)},
            timeout=600,
        )
        assert result.returncode == 0, (
            f"install.sh --offline 失败:\nSTDERR:\n{result.stderr}\nSTDOUT:\n{result.stdout}"
        )

        # Step 4: 验证 paper-review 可 import
        python = str(venv_dir / "bin" / "python")
        result = subprocess.run(
            [python, "-c", "import paper_review; print('import-ok')"],
            capture_output=True,
            text=True,
            cwd=pack_dir,
        )
        assert "import-ok" in result.stdout, f"stderr:\n{result.stderr}"

        # Step 5: 验证 paper-review --help 可用
        paper_review_bin = str(venv_dir / "bin" / "paper-review")
        result = subprocess.run(
            [paper_review_bin, "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "index" in result.stdout

        # Step 6: 验证模型已被拷贝到缓存（HOME 隔离后的路径）
        model_cache = fake_home / ".cache" / "paper-review" / "models"
        assert model_cache.is_dir(), f"模型缓存不存在: {model_cache}"
        model_dirs = [d for d in model_cache.iterdir() if d.is_dir()]
        assert len(model_dirs) >= 2, f"模型缓存应至少包含 2 个模型，实际: {model_dirs}"

        # 验证每个模型目录都有 INT8 权重（model_quantized.onnx 优先，
        # 名称随仓库而异 → 从源码动态导入候选列表，不硬编码）
        from paper_review.model_discovery import RUNTIME_MODEL_FILE_NAMES

        for d in model_dirs:
            onnx_file = next(
                (d / name for name in RUNTIME_MODEL_FILE_NAMES if (d / name).is_file()),
                None,
            )
            assert onnx_file is not None, f"{d.name} 缺少 ONNX 权重（{RUNTIME_MODEL_FILE_NAMES}）"
            assert not onnx_file.is_symlink(), (
                f"{d.name} {onnx_file.name} 是 symlink，copy_mode=True 应产生实体文件以支持跨机器部署"
            )

        # Step 7: 验证 install.sh 把 models-manifest.json 的模型名写入 config.yaml
        # （否则运行时按默认模型名查找 → 模型静默失效 —— 本次变更的核心接线）
        import json

        manifest = json.loads((pack_dir / "models-manifest.json").read_text(encoding="utf-8"))
        assert "embedding_model" in manifest and "reranker_model" in manifest
        for cfg_path in (
            fake_home / ".paper-review" / "config.yaml",
            pack_dir / ".paper-review" / "config.yaml",
        ):
            assert cfg_path.is_file(), f"install.sh 未写入 {cfg_path}"
            cfg_text = cfg_path.read_text(encoding="utf-8")
            for key, value in manifest.items():
                assert f"{key}: {value}" in cfg_text, (
                    f"{cfg_path} 缺少 {key}: {value}（当前内容:\n{cfg_text}）"
                )
