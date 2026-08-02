"""
E2E: paper-review review — 验证管线可通过 CLI 从头执行到底。

测试策略：
- 创建临时管线目录（含 pipeline.yaml + .py 步骤）
- 创建合成 PDF
- 用 subprocess 调用 paper-review review <path>
- 验证 intermediates/output.json 产出物
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def _paper_review_bin() -> str:
    """返回 paper-review 可执行文件路径。"""
    bindir = Path(sys.executable).parent
    candidate = bindir / "paper-review"
    if candidate.exists():
        return str(candidate)
    which = subprocess.run(["which", "paper-review"], capture_output=True, text=True, check=False)
    if which.returncode == 0:
        return which.stdout.strip()
    return f"{sys.executable} -m paper_review"


def _find_task_dir(output_dir: Path) -> Path:
    """在 output/result/ 下找到唯一的 task 目录并返回。"""
    result_dir = output_dir / "result"
    assert result_dir.is_dir(), f"result dir not found: {result_dir}"
    subdirs = list(result_dir.iterdir())
    assert len(subdirs) == 1, f"expected 1 task dir, got {len(subdirs)}: {subdirs}"
    return subdirs[0]


def _make_pipeline_dir(base: Path, data_dir: Path) -> tuple[Path, Path]:
    """在 base 下创建完整管线目录，返回 (pipeline_dir, output_dir)。"""
    # 初始化数据目录（创建 index/ 和 output/ 以避免首次运行交互提示）
    (data_dir / "index").mkdir(parents=True, exist_ok=True)
    output_dir = data_dir / "output"

    pipeline_dir = base / "pipeline"
    pipeline_dir.mkdir(parents=True)

    (pipeline_dir / "pipeline.yaml").write_text(
        f"name: e2e-review\n"
        f"output_dir: {output_dir}\n"
        f"pre:\n"
        f"  directory: pre-review/\n"
        f"review:\n"
        f"  directory: review-pipeline/\n"
        f"post:\n"
        f"  directory: post-review/\n"
    )

    # Pre 步骤
    pre_dir = pipeline_dir / "pre-review"
    pre_dir.mkdir()
    (pre_dir / "01-pre.py").write_text(
        "import json, os;"
        'd=os.environ["PIPELINE_STEP_DIR"];'
        "os.makedirs(d, exist_ok=True);"
        'json.dump({"step":"01-pre","status":"ok","data":{"msg":"pre-done"}},'
        'open(os.path.join(d,"output.json"),"w"))'
    )

    # Review 步骤
    review_dir = pipeline_dir / "review-pipeline"
    review_dir.mkdir()
    (review_dir / "01-search.py").write_text(
        "import json, os;"
        'd=os.environ["PIPELINE_STEP_DIR"];'
        "os.makedirs(d, exist_ok=True);"
        'json.dump({"step":"01-search","status":"ok","data":{}},'
        'open(os.path.join(d,"output.json"),"w"))'
    )

    # Post 步骤
    post_dir = pipeline_dir / "post-review"
    post_dir.mkdir()
    (post_dir / "01-archive.py").write_text(
        "import json, os;"
        'd=os.environ["PIPELINE_STEP_DIR"];'
        "os.makedirs(d, exist_ok=True);"
        'json.dump({"step":"01-archive","status":"ok","data":{}},'
        'open(os.path.join(d,"output.json"),"w"))'
    )

    # 合成 PDF
    pdf_path = pipeline_dir / "paper-001.pdf"
    pdf_path.write_text("dummy pdf content")

    return pipeline_dir, output_dir


class TestReviewE2E:
    """paper-review review E2E"""

    def test_review_full_pipeline(self, tmp_path: Path):
        """完整管线执行：pre → review → post，验证三步 intermediates。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir(parents=True)
        pipeline_dir, output_dir = _make_pipeline_dir(tmp_path, data_dir)

        result = subprocess.run(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                "--skip-warnings",
                str(pipeline_dir),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # 管线不应该 crash
        assert result.returncode == 0, (
            f"review failed:\nstdout:{result.stdout[:500]}\nstderr:{result.stderr[:500]}"
        )

        # 验证 pre intermediates
        task_dir = _find_task_dir(output_dir)
        pre_out = task_dir / "intermediates" / "pre" / "01-pre" / "output.json"
        assert pre_out.exists(), f"Missing pre output: {pre_out}"

        # 验证 review intermediates
        review_out = task_dir / "intermediates" / "paper-001" / "01-search" / "output.json"
        assert review_out.exists(), f"Missing review output: {review_out}"

        # 验证 post intermediates
        post_out = task_dir / "intermediates" / "post" / "01-archive" / "output.json"
        assert post_out.exists(), f"Missing post output: {post_out}"

    def test_review_without_index_returns_ok(self, tmp_path: Path):
        """索引不存在时 review 不应崩溃——01-search 应优雅降级返回空引用。"""
        data_dir = tmp_path / "data-no-index"
        # ⚠ 关键：不创建 index/ 目录，模拟 index 尚未建立的场景
        output_dir = data_dir / "output"

        pipeline_dir = tmp_path / "noindex-review"
        pipeline_dir.mkdir(parents=True)

        steps_dir = pipeline_dir / "review-pipeline"
        steps_dir.mkdir()
        # 模拟真实 01-search.py 的行为：使用 PIPELINE_DATA_DIR 定位 index
        (steps_dir / "01-search.py").write_text(
            "import json, os, sys;"
            'sys.path.insert(0, os.environ.get("PIPELINE_PIPELINE_DIR", "."));'
            "from paper_review.search.store import Store;"
            'dd = os.environ.get("PIPELINE_DATA_DIR", "");'
            'db_path = os.path.join(dd, "index", "index.sqlite") if dd else ":memory:";'
            "store = Store(db_path=db_path);"
            "store.load_all();"
            'd = os.environ["PIPELINE_STEP_DIR"];'
            "os.makedirs(d, exist_ok=True);"
            'json.dump({"step":"01-search","status":"ok","data":{}},'
            '          open(os.path.join(d,"output.json"),"w"))'
        )

        pdf = pipeline_dir / "paper.pdf"
        pdf.write_text("dummy")

        result = subprocess.run(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                "--skip-warnings",
                str(pdf),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        # 不应 crash——Store 在空目录能自动创建 db 文件
        assert result.returncode == 0, (
            f"review crashed on empty index:\n"
            f"stdout:{result.stdout[:500]}\nstderr:{result.stderr[:500]}"
        )
        assert "✅ 01-search" in result.stdout, f"01-search should succeed:\n{result.stdout}"

    def test_review_no_pipeline_dir_fails(self, tmp_path: Path):
        """不存在的路径应返回非零退出码。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        result = subprocess.run(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                str(tmp_path / "nonexistent"),
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert result.returncode != 0

    def test_review_single_pdf(self, tmp_path: Path):
        """直接传入 PDF 文件路径（免配置模式），使用 --data-dir 避免交互提示。"""
        data_dir = tmp_path / "data"
        (data_dir / "index").mkdir(parents=True)
        output_dir = data_dir / "output"

        pipeline_dir = tmp_path / "simple-review"
        pipeline_dir.mkdir(parents=True)

        steps_dir = pipeline_dir / "review-pipeline"
        steps_dir.mkdir()
        (steps_dir / "01-check.py").write_text(
            "import json, os;"
            'd=os.environ["PIPELINE_STEP_DIR"];'
            "os.makedirs(d, exist_ok=True);"
            'json.dump({"step":"01-check","status":"ok","data":{}},'
            'open(os.path.join(d,"output.json"),"w"))'
        )

        pdf_path = pipeline_dir / "paper.pdf"
        pdf_path.write_text("dummy")

        result = subprocess.run(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                "--skip-warnings",
                str(pdf_path),
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )

        assert result.returncode == 0, (
            f"review failed:\nstdout:{result.stdout[:500]}\nstderr:{result.stderr[:500]}"
        )
        # 检查 output 产物
        task_dir = _find_task_dir(output_dir)
        out = task_dir / "intermediates" / "paper" / "01-check" / "output.json"
        assert out.exists(), f"Missing output: {out}"
