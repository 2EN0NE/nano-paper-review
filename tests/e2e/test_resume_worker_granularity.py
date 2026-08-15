"""
E2E: Resume（断点续做）+ Worker 粒度（granularity）测试

覆盖场景：
  1. 完整 review 后 task.json status=done
  2. 中断（kill 子进程）→ 再次 review 检测到未完成任务 → 交互选择 [1] 续做
     → task_id 复用、已完成步骤跳过、最终 done
  3. 中断 → 选择 [2] 重新一批 → 新 task_id + 旧任务 abandoned
  4. granularity: step 配置端到端跑通（barrier 调度产物齐全）
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


def _paper_review_bin() -> str:
    bindir = Path(sys.executable).parent
    candidate = bindir / "paper-review"
    if candidate.exists():
        return str(candidate)
    which = subprocess.run(["which", "paper-review"], capture_output=True, text=True, check=False)
    if which.returncode == 0:
        return which.stdout.strip()
    return f"{sys.executable} -m paper_review"


def _make_pdf(path: Path, content: str = "paper content") -> Path:
    text_obj = f"BT /F1 12 Tf 100 700 Td ({content}) Tj ET"
    pdf_content = (
        b"%PDF-1.4\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        b"/Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\n"
        + f"4 0 obj << /Length {len(text_obj) + 2} >> stream\n{text_obj}\nendstream endobj\n".encode()
        + b"5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
    )
    offsets: list[int] = []
    pos = 0
    for line in pdf_content.split(b"\n"):
        if b" obj" in line:
            offsets.append(pos)
        pos += len(line) + 1
    xref_start = pos
    xref_table = f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f \n"
    for i, off in enumerate(offsets, 1):
        xref_table += f"{off:010d} 00000 n \n"
    trailer = (
        f"trailer << /Size {len(offsets) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n"
    )
    path.write_bytes(pdf_content + xref_table.encode() + trailer.encode())
    return path


def _make_review_step(pipeline_dir: Path, name: str, script: str) -> None:
    """创建 review step 脚本。"""
    step_dir = pipeline_dir / "review-pipeline"
    step_dir.mkdir(parents=True, exist_ok=True)
    (step_dir / name).write_text(script)


def _setup_pipeline(
    data_dir: Path, *, granularity: str = "subject", workers: int = 2, slow: bool = False
) -> None:
    """创建含 pre/review/post 的管线定义。

    slow=True 时 review 第一步 sleep 2s（供中断测试制造窗口）。
    """
    pipelines_dir = data_dir / "pipelines" / "e2e-resume"
    pipelines_dir.mkdir(parents=True, exist_ok=True)

    review_steps = []
    if slow:
        review_steps.append(
            "01-slow.py",
        )
    review_steps.extend(["02-quick.py"])

    pool_block = f"""
    pool:
      workers: {workers}
      timeout: 120
      ordered: true
      granularity: {granularity}
"""

    (pipelines_dir / "pipeline.yaml").write_text(
        'name: "e2e-resume"\n'
        "phases:\n"
        "  - name: pre\n"
        "    mode: batch\n"
        "    directory: pre-review/\n"
        "    manifest_step: 01-convert\n"
        "    duplicate_policy: skip\n"
        "    retry:\n"
        "      max_attempts: 1\n"
        "      on_failure: skip\n"
        "  - name: review\n"
        "    mode: per_subject\n"
        "    directory: review-pipeline/\n"
        "    duplicate_policy: skip\n"
        "    retry:\n"
        "      max_attempts: 1\n"
        "      on_failure: skip\n"
        f"{pool_block}"
        "  - name: post\n"
        "    mode: batch\n"
        "    directory: post-review/\n"
        "    duplicate_policy: skip\n"
        "    retry:\n"
        "      max_attempts: 1\n"
        "      on_failure: skip\n"
    )

    # pre-review/01-convert.py —— 直接拷贝真实模板
    src_convert = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "paper_review"
        / "templates"
        / "pre-review"
        / "01-convert.py"
    )
    pre_dir = pipelines_dir / "pre-review"
    pre_dir.mkdir(parents=True, exist_ok=True)
    if src_convert.exists():
        shutil.copy(src_convert, pre_dir / "01-convert.py")
    else:  # pragma: no cover
        (pre_dir / "01-convert.py").write_text(
            "import json, os, sys\n"
            "from pathlib import Path\n"
            "inp = Path(os.environ['PIPELINE_INPUT_PATH'])\n"
            "files = [inp] if inp.is_file() else sorted(inp.glob('*.pdf'))\n"
            "subs = [{'name': f.stem, 'pdf_path': str(f), 'original_path': str(f)} for f in files]\n"
            "step_dir = Path(os.environ['PIPELINE_STEP_DIR'])\n"
            "step_dir.mkdir(parents=True, exist_ok=True)\n"
            "out = {'step': '01-convert', 'status': 'ok', 'error': None, 'data': {'subjects': subs}}\n"
            "(step_dir / 'output.json').write_text(json.dumps(out, ensure_ascii=False))\n"
            "Path(os.environ['PIPELINE_OUTPUT_DIR']) / 'subject-manifest.json'\n"
            "import pathlib; pathlib.Path(os.environ['PIPELINE_OUTPUT_DIR']).mkdir(parents=True, exist_ok=True)\n"
            "pathlib.Path(os.environ['PIPELINE_OUTPUT_DIR'], 'subject-manifest.json').write_text(\n"
            "    json.dumps({'subjects': subs, 'converted': 0, 'skipped': [], 'duplicates': 0}, ensure_ascii=False))\n"
        )

    # review steps
    _make_review_step(
        pipelines_dir,
        "01-slow.py",
        "import json, os, time\n"
        "from pathlib import Path\n"
        "time.sleep(2)\n"
        "d = Path(os.environ['PIPELINE_STEP_DIR'])\n"
        "d.mkdir(parents=True, exist_ok=True)\n"
        "(d / 'output.json').write_text(json.dumps(\n"
        "    {'step': '01-slow', 'status': 'ok', 'error': None,\n"
        "     'data': {'subject': os.environ.get('PIPELINE_SUBJECT', '')}}))\n",
    )
    _make_review_step(
        pipelines_dir,
        "02-quick.py",
        "import json, os\n"
        "from pathlib import Path\n"
        "d = Path(os.environ['PIPELINE_STEP_DIR'])\n"
        "d.mkdir(parents=True, exist_ok=True)\n"
        "(d / 'output.json').write_text(json.dumps(\n"
        "    {'step': '02-quick', 'status': 'ok', 'error': None,\n"
        "     'data': {'subject': os.environ.get('PIPELINE_SUBJECT', '')}}))\n",
    )

    # post-review/01-archive.py
    post_dir = pipelines_dir / "post-review"
    post_dir.mkdir(parents=True, exist_ok=True)
    (post_dir / "01-archive.py").write_text(
        "import json, os\n"
        "from pathlib import Path\n"
        "d = Path(os.environ['PIPELINE_STEP_DIR'])\n"
        "d.mkdir(parents=True, exist_ok=True)\n"
        "(d / 'output.json').write_text(json.dumps(\n"
        "    {'step': '01-archive', 'status': 'ok', 'error': None, 'data': {}}))\n",
    )


def _setup_input(data_dir: Path, *pdf_names: str) -> Path:
    input_dir = data_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    for name in pdf_names:
        _make_pdf(input_dir / f"{name}.pdf", f"content of {name}")
    return input_dir


def _find_task_dirs(output_dir: Path) -> list[Path]:
    result_root = output_dir / "result"
    if not result_root.is_dir():
        return []
    return sorted(result_root.iterdir(), reverse=True)


def _read_manifest(task_dir: Path) -> dict:
    return json.loads((task_dir / "task.json").read_text(encoding="utf-8"))


# ============================================================================
# 测试
# ============================================================================


class TestResumeWorkflow:
    def test_full_run_marks_task_done(self, tmp_path: Path):
        """完整 review 后 task.json status=done。"""
        data_dir = tmp_path / "data"
        (data_dir / "index").mkdir(parents=True)
        (data_dir / ".first-use-hint-shown").touch()
        _setup_pipeline(data_dir)
        input_dir = _setup_input(data_dir, "alpha", "beta")

        env = os.environ.copy()
        env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"

        result = subprocess.run(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                "--skip-warnings",
                str(input_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert result.returncode == 0, (
            f"Pipeline failed:\nSTDOUT:{result.stdout[:800]}\nSTDERR:{result.stderr[:500]}"
        )

        tasks = _find_task_dirs(data_dir / "output")
        assert len(tasks) == 1, f"Expected 1 task, got {tasks}"
        manifest = _read_manifest(tasks[0])
        assert manifest["status"] == "done"
        assert manifest["success"]
        assert set(manifest["subjects"]) == {"alpha", "beta"}
        # 产物齐全
        for subj in ("alpha", "beta"):
            assert (tasks[0] / "intermediates" / subj / "01-slow" / "output.json").exists()
            assert (tasks[0] / "intermediates" / subj / "02-quick" / "output.json").exists()

    def test_interrupt_then_resume(self, tmp_path: Path):
        """中断（kill）→ 再次 review 检测到未完成 → [1] 续做 → task_id 复用 + done。"""
        data_dir = tmp_path / "data"
        (data_dir / "index").mkdir(parents=True)
        (data_dir / ".first-use-hint-shown").touch()
        _setup_pipeline(data_dir, slow=True)
        input_dir = _setup_input(data_dir, "alpha", "beta")

        env = os.environ.copy()
        env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"

        # ── 第一次运行：慢步骤执行中 kill 掉 ──
        proc = subprocess.Popen(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                "--skip-warnings",
                str(input_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        time.sleep(3.0)  # 等慢步骤开始（sleep 2s 的 review step 窗口）
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)

        tasks = _find_task_dirs(data_dir / "output")
        assert len(tasks) == 1, f"Expected 1 interrupted task, got {tasks}"
        first_task = tasks[0]
        manifest = _read_manifest(first_task)
        assert manifest["status"] == "running", (
            f"kill 后应为 running（无 SIGINT handler 兜底）: {manifest}"
        )

        # ── 再次 review：检测到未完成 → stdin 喂 [1] 续做 ──
        result = subprocess.run(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                str(input_dir),
            ],
            input="Y\n1\n",  # 先确认空索引警告，再选择 [1] 续做
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert result.returncode == 0, (
            f"Resume failed:\nSTDOUT:{result.stdout[:800]}\nSTDERR:{result.stderr[:500]}"
        )
        assert "未完成" in result.stdout or "续做" in result.stdout

        tasks = _find_task_dirs(data_dir / "output")
        assert len(tasks) == 1, f"续做应复用原 task，不应新建: {tasks}"
        final_task = tasks[0]
        assert final_task.name == first_task.name, "Resume 应复用原 task_id"
        final_manifest = _read_manifest(final_task)
        assert final_manifest["status"] == "done", f"续做后应为 done: {final_manifest}"
        # 产物齐全
        for subj in ("alpha", "beta"):
            assert (final_task / "intermediates" / subj / "01-slow" / "output.json").exists()
            assert (final_task / "intermediates" / subj / "02-quick" / "output.json").exists()

    def test_interrupt_sigint_marks_interrupted_then_resume(self, tmp_path: Path):
        """SIGINT 优雅中断 → task.json status=interrupted + 续做 → done。

        SIGKILL 验证“running 无 done”状态推断兜底；SIGINT 验证 handler 路径
        （写 interrupted + interrupted_at_step）——两条中断路径都应有 E2E 覆盖。
        """
        data_dir = tmp_path / "data"
        (data_dir / "index").mkdir(parents=True)
        (data_dir / ".first-use-hint-shown").touch()
        _setup_pipeline(data_dir, slow=True)
        input_dir = _setup_input(data_dir, "alpha", "beta")

        env = os.environ.copy()
        env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"

        # ── 第一次运行：慢步骤执行中 SIGINT（优雅中断）──
        proc = subprocess.Popen(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                "--skip-warnings",
                str(input_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        time.sleep(3.0)  # 等慢步骤开始（sleep 2s 的 review step 窗口）
        proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
        assert proc.returncode != 0, "SIGINT 应以非零码退出"

        tasks = _find_task_dirs(data_dir / "output")
        assert len(tasks) == 1, f"Expected 1 interrupted task, got {tasks}"
        first_task = tasks[0]
        manifest = _read_manifest(first_task)
        assert manifest["status"] == "interrupted", f"SIGINT 后应为 interrupted: {manifest}"

        # ── 再次 review：检测到未完成 → stdin 喂 [1] 续做 ──
        result = subprocess.run(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                str(input_dir),
            ],
            input="Y\n1\n",  # 先确认空索引警告，再选择 [1] 续做
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert result.returncode == 0, (
            f"Resume failed:\nSTDOUT:{result.stdout[:800]}\nSTDERR:{result.stderr[:500]}"
        )

        tasks = _find_task_dirs(data_dir / "output")
        assert len(tasks) == 1, f"续做应复用原 task，不应新建: {tasks}"
        final_manifest = _read_manifest(tasks[0])
        assert final_manifest["status"] == "done", f"续做后应为 done: {final_manifest}"
        for subj in ("alpha", "beta"):
            assert (tasks[0] / "intermediates" / subj / "01-slow" / "output.json").exists()
            assert (tasks[0] / "intermediates" / subj / "02-quick" / "output.json").exists()

    def test_restart_new_batch_abandons_old(self, tmp_path: Path):
        """中断 → 选择 [2] 重新一批 → 新 task_id + 旧任务 abandoned。"""
        data_dir = tmp_path / "data"
        (data_dir / "index").mkdir(parents=True)
        (data_dir / ".first-use-hint-shown").touch()
        _setup_pipeline(data_dir, slow=True)
        input_dir = _setup_input(data_dir, "alpha", "beta")

        env = os.environ.copy()
        env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"

        # 第一次运行 → kill
        proc = subprocess.Popen(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                "--skip-warnings",
                str(input_dir),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        time.sleep(3.0)
        proc.send_signal(signal.SIGKILL)
        proc.wait(timeout=10)
        first_task = _find_task_dirs(data_dir / "output")[0]

        # 再次 review → [2] 重新一批
        result = subprocess.run(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                str(input_dir),
            ],
            input="Y\n2\n",  # 先确认空索引警告，再选择 [2] 重新一批
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert result.returncode == 0
        assert "重新" in result.stdout

        tasks = _find_task_dirs(data_dir / "output")
        assert len(tasks) == 2, f"应新建 task（旧任务保留）: {tasks}"
        old = _read_manifest(first_task)
        assert old["status"] == "abandoned", f"旧任务应标记 abandoned: {old}"
        new_task = tasks[0]
        assert new_task.name != first_task.name
        assert _read_manifest(new_task)["status"] == "done"

    def test_step_granularity_full_run(self, tmp_path: Path):
        """granularity: step 配置端到端跑通（barrier 调度产物齐全）。"""
        data_dir = tmp_path / "data"
        (data_dir / "index").mkdir(parents=True)
        (data_dir / ".first-use-hint-shown").touch()
        _setup_pipeline(data_dir, granularity="step", workers=2)
        input_dir = _setup_input(data_dir, "alpha", "beta", "gamma")

        env = os.environ.copy()
        env["PATH"] = str(tmp_path) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"

        result = subprocess.run(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                "--skip-warnings",
                str(input_dir),
            ],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )
        assert result.returncode == 0, (
            f"step-granularity pipeline failed:\nSTDOUT:{result.stdout[:800]}\nSTDERR:{result.stderr[:500]}"
        )
        # 进度条会静音 stderr 的 INFO 日志（终端保护），改从文件日志验证 step 级调度生效
        log_file = data_dir / "logs" / "paper-review.log"
        log_text = log_file.read_text(encoding="utf-8") if log_file.exists() else ""
        assert "Granularity=step" in log_text, f"step 级调度未生效，日志尾部: {log_text[-2000:]}"

        tasks = _find_task_dirs(data_dir / "output")
        assert len(tasks) == 1
        manifest = _read_manifest(tasks[0])
        assert manifest["status"] == "done"
        assert set(manifest["subjects"]) == {"alpha", "beta", "gamma"}
        for subj in ("alpha", "beta", "gamma"):
            assert (tasks[0] / "intermediates" / subj / "01-slow" / "output.json").exists()
            assert (tasks[0] / "intermediates" / subj / "02-quick" / "output.json").exists()
