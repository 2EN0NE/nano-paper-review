"""
E2E: Pre 大步骤内部进度上报 + Resume 断点续做（T1-T6）

覆盖：
- T2: batch 步骤（post）收到 PIPELINE_BATCH_PROGRESS_FILE 注入
- T3: Resume 步骤级续做——已完成 Pre 步骤跳过（skipped），未完成步骤重跑
- T4: 04-extract-features / 05-batch-search 内部增量续做（SKIP_EXISTING 跳过已有篇）
- T5: 02-auto-index 增量续做（per-subject 产物 + paper_id 映射恢复）

用真实模板 01-05（faiss 已装；无 ONNX 模型时 embedding 哈希降级），
review/post 用确定性 .py 步骤避免依赖 pi 输出格式。
Resume 通过 CLI 交互 [1] 触发（detect_unfinished_tasks 检测未完成任务）。
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

from tests.e2e.test_pipeline_integration import (
    _make_mock_pandoc,
    _make_pdf,
    _paper_review_bin,
)

pytestmark = pytest.mark.e2e

_TEMPLATES_PRE = (
    Path(__file__).resolve().parent.parent.parent
    / "src"
    / "paper_review"
    / "templates"
    / "pre-review"
)

_PRE_STEPS = (
    "01-convert.py",
    "02-auto-index.py",
    "03-generate-query.py",
    "04-extract-features.py",
    "05-batch-search.py",
)


def _make_mock_pi(bindir: Path, counter: Path) -> Path:
    """mock pi 二进制：每次调用追加计数 + 输出 LLM 特征 JSON（04 解析用）。"""
    mock = bindir / "pi"
    script = (
        "#!/usr/bin/env python3\n"
        "import sys, os\n"
        "c = os.environ.get('MOCK_PI_COUNTER', '')\n"
        "if c:\n"
        "    with open(c, 'a', encoding='utf-8') as f:\n"
        "        f.write('call\\n')\n"
        'print(\'["深度学习", "Transformer", "对比学习"]\')\n'
    )
    mock.write_text(script)
    mock.chmod(mock.stat().st_mode | stat.S_IEXEC)
    return mock


def _setup_pipeline(pipelines_dir: Path) -> Path:
    """真实模板 pre（01-05）+ 确定性 review/post 步骤。"""
    pipeline_dir = pipelines_dir / "pre-progress"
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    (pipeline_dir / "pipeline.yaml").write_text(
        """\
name: "pre-progress-resume"
version: "2.0"
index:
  store_dir: ""
  reference_dir: ""
  auto_index: true
  copy_subjects: true
phases:
  - name: pre
    mode: batch
    directory: pre-review/
    manifest_step: "01-convert"
    duplicate_policy: skip
    retry:
      max_attempts: 1
      on_failure: skip
  - name: review
    mode: per_subject
    directory: review-pipeline/
    subject_source:
      type: manifest
      path: "{{ output_dir }}/subject-manifest.json"
    duplicate_policy: skip
    retry:
      max_attempts: 1
      on_failure: skip
    subject_order:
      sort_by: name
      direction: asc
    pool:
      workers: 1
      timeout: 120
  - name: post
    mode: batch
    directory: post-review/
    duplicate_policy: skip
    retry:
      max_attempts: 1
      on_failure: skip
"""
    )

    # pre-review/：复制真实模板 01-05
    pre_dir = pipeline_dir / "pre-review"
    pre_dir.mkdir()
    for stem in _PRE_STEPS:
        shutil.copy(_TEMPLATES_PRE / stem, pre_dir / stem)

    # review-pipeline/：确定性 .py（不调 pi）
    review_dir = pipeline_dir / "review-pipeline"
    review_dir.mkdir()
    (review_dir / "06-direct-scoring.py").write_text(
        "import json, os\n"
        'd = os.environ["PIPELINE_STEP_DIR"]\n'
        "os.makedirs(d, exist_ok=True)\n"
        'json.dump({"step":"06-direct-scoring","status":"ok","data":{}},'
        'open(os.path.join(d,"output.json"),"w"))\n'
    )

    # post-review/：确定性 .py + 记录 batch 进度文件 env（T2 验证）
    post_dir = pipeline_dir / "post-review"
    post_dir.mkdir()
    (post_dir / "09-archive.py").write_text(
        "import json, os\n"
        'd = os.environ["PIPELINE_STEP_DIR"]\n'
        "os.makedirs(d, exist_ok=True)\n"
        'progress_file = os.environ.get("PIPELINE_BATCH_PROGRESS_FILE", "")\n'
        'json.dump({"step":"09-archive","status":"ok",'
        '"data":{"progress_file": progress_file}},'
        'open(os.path.join(d,"output.json"),"w"))\n'
    )
    return pipeline_dir


def _make_env(tmp_path: Path, data_dir: Path, counter: Path) -> dict:
    mock_bin = tmp_path / "mock-bin"
    mock_bin.mkdir(exist_ok=True)
    _make_mock_pandoc(mock_bin)
    _make_mock_pi(mock_bin, counter)
    env = os.environ.copy()
    env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
    env["PIPELINE_PI_BINARY"] = "pi"
    env["MOCK_PI_COUNTER"] = str(counter)
    return env


def _setup_data(tmp_path: Path, subjects: tuple[str, ...]) -> tuple[Path, Path, dict]:
    """搭建隔离 data-dir + 真实模板管线 + 输入 PDF，返回 (data_dir, input_dir, env)。"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "index").mkdir()
    (data_dir / ".first-use-hint-shown").touch()
    _setup_pipeline(data_dir / "pipelines")

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    for name in subjects:
        _make_pdf(input_dir / f"{name}.pdf", f"{name} paper about deep learning and retrieval")

    counter = tmp_path / "pi-calls.txt"
    env = _make_env(tmp_path, data_dir, counter)
    return data_dir, input_dir, env


def _run_first(data_dir: Path, input_dir: Path, env: dict) -> subprocess.CompletedProcess:
    """首次完整运行（无人值守，不交互）。"""
    return subprocess.run(
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
        timeout=180,
        env=env,
    )


def _run_resume(data_dir: Path, input_dir: Path, env: dict) -> subprocess.CompletedProcess:
    """续做：检测未完成任务 → stdin [1] 继续最近一批。"""
    return subprocess.run(
        [
            _paper_review_bin(),
            "--data-dir",
            str(data_dir),
            "review",
            str(input_dir),
        ],
        input="1\n",
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
    )


def _find_task_dir(output_dir: Path) -> Path:
    result_dirs = list((output_dir / "result").iterdir())
    assert len(result_dirs) == 1, f"Expected 1 result dir, got {result_dirs}"
    return result_dirs[0]


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _pi_calls(counter: Path) -> int:
    return counter.read_text().count("call") if counter.exists() else 0


class TestPreStepResume:
    """T3+T4: Resume 步骤级续做 + 04/05 内部增量（真实模板）。"""

    def test_resume_04_reruns_only_missing_subjects(self, tmp_path: Path):
        """中断在 04：已完成步骤跳过，04 只重跑被删篇（mock pi 调用只 +1）。"""
        data_dir, input_dir, env = _setup_data(tmp_path, ("alpha", "beta", "gamma"))

        r1 = _run_first(data_dir, input_dir, env)
        assert r1.returncode == 0, f"首次失败:\n{r1.stdout[-600:]}\n{r1.stderr[-400:]}"
        calls1 = _pi_calls(tmp_path / "pi-calls.txt")
        assert calls1 == 3, f"首次 04 应每篇调一次 pi: {calls1}"

        task_dir = _find_task_dir(data_dir / "output")
        # 模拟中断在 04：删 gamma 的 04 产物 + 04 全局产物
        (task_dir / "intermediates" / "gamma" / "04-extract-features" / "output.json").unlink()
        (task_dir / "intermediates" / "pre" / "04-extract-features" / "output.json").unlink()
        _read_json(task_dir / "task.json")  # 确认 task.json 可读
        from paper_review.orchestrator import write_task_manifest

        write_task_manifest(task_dir, status="running")

        r2 = _run_resume(data_dir, input_dir, env)
        assert r2.returncode == 0, f"续做失败:\n{r2.stdout[-600:]}\n{r2.stderr[-400:]}"
        assert "续做" in r2.stdout or "未完成" in r2.stdout

        # 04 只重跑 gamma（pi 调用 +1）
        calls2 = _pi_calls(tmp_path / "pi-calls.txt")
        assert calls2 == calls1 + 1, f"04 应只重跑被删篇（+1）: {calls1} → {calls2}"

        # 04 全局产物重建（reused 计数=2：alpha/beta 复用）
        pre_out = _read_json(
            task_dir / "intermediates" / "pre" / "04-extract-features" / "output.json"
        )
        assert pre_out["status"] == "ok"
        assert pre_out["data"]["reused_count"] == 2, pre_out["data"]
        # gamma 的 04 产物重建
        assert (
            task_dir / "intermediates" / "gamma" / "04-extract-features" / "output.json"
        ).exists()
        # 任务完成
        assert _read_json(task_dir / "task.json")["status"] == "done"

    def test_resume_05_reruns_only_missing_subjects(self, tmp_path: Path):
        """中断在 05：05 只重跑被删篇（05 无 LLM，用 reused_count 断言）。"""
        data_dir, input_dir, env = _setup_data(tmp_path, ("alpha", "beta", "gamma"))

        r1 = _run_first(data_dir, input_dir, env)
        assert r1.returncode == 0, f"首次失败:\n{r1.stdout[-600:]}"

        task_dir = _find_task_dir(data_dir / "output")
        # 模拟中断在 05：删 beta 的 05 产物 + 05 全局产物
        (task_dir / "intermediates" / "beta" / "05-batch-search" / "output.json").unlink()
        (task_dir / "intermediates" / "pre" / "05-batch-search" / "output.json").unlink()
        from paper_review.orchestrator import write_task_manifest

        write_task_manifest(task_dir, status="running")

        r2 = _run_resume(data_dir, input_dir, env)
        assert r2.returncode == 0, f"续做失败:\n{r2.stdout[-600:]}"

        pre_out = _read_json(task_dir / "intermediates" / "pre" / "05-batch-search" / "output.json")
        assert pre_out["status"] == "ok"
        assert pre_out["data"]["reused_count"] == 2, pre_out["data"]
        # beta 的 05 产物重建且含检索结果结构
        beta_out = _read_json(
            task_dir / "intermediates" / "beta" / "05-batch-search" / "output.json"
        )
        assert beta_out["status"] == "ok"
        assert "history" in beta_out["data"] and "pending" in beta_out["data"]

    def test_resume_02_reruns_only_missing_subjects(self, tmp_path: Path):
        """中断在 02：02 只重跑被删篇，subject_paper_ids 映射从产物恢复保持完整。"""
        data_dir, input_dir, env = _setup_data(tmp_path, ("alpha", "beta", "gamma"))

        r1 = _run_first(data_dir, input_dir, env)
        assert r1.returncode == 0, f"首次失败:\n{r1.stdout[-600:]}"

        task_dir = _find_task_dir(data_dir / "output")
        # 模拟中断在 02：删 gamma 的 02 产物 + 02 全局产物
        (task_dir / "intermediates" / "gamma" / "02-auto-index" / "output.json").unlink()
        (task_dir / "intermediates" / "pre" / "02-auto-index" / "output.json").unlink()
        from paper_review.orchestrator import write_task_manifest

        write_task_manifest(task_dir, status="running")

        r2 = _run_resume(data_dir, input_dir, env)
        assert r2.returncode == 0, f"续做失败:\n{r2.stdout[-600:]}"

        pre_out = _read_json(task_dir / "intermediates" / "pre" / "02-auto-index" / "output.json")
        assert pre_out["status"] == "ok"
        assert pre_out["data"]["reused_count"] == 2, pre_out["data"]
        # paper_id 映射完整（3 篇，跳过篇从产物恢复）
        ids = pre_out["data"]["subject_paper_ids"]
        assert set(ids) == {"alpha", "beta", "gamma"}, ids


class TestPreBatchProgressEnv:
    """T2: batch 阶段步骤收到 PIPELINE_BATCH_PROGRESS_FILE 注入。"""

    def test_post_step_receives_progress_file_env(self, tmp_path: Path):
        """post（batch）步骤收到进度文件路径 env（orchestrator 注入）。"""
        data_dir, input_dir, env = _setup_data(tmp_path, ("alpha",))

        r1 = _run_first(data_dir, input_dir, env)
        assert r1.returncode == 0, f"首次失败:\n{r1.stdout[-600:]}"

        task_dir = _find_task_dir(data_dir / "output")
        post_out = _read_json(task_dir / "intermediates" / "post" / "09-archive" / "output.json")
        progress_file = post_out["data"].get("progress_file", "")
        assert progress_file, f"post 步骤应收到 PIPELINE_BATCH_PROGRESS_FILE: {post_out}"
        assert progress_file.endswith("post-batch-step.json"), progress_file


class TestResume01And03:
    """T6: 01-convert 复用已转换 PDF / 03-generate-query 增量（resume E2E）。"""

    def _setup_env_with_counting_pandoc(self, tmp_path: Path, counter: Path) -> dict:
        """mock pandoc：每次调用追加计数（01 复用验证用）。"""
        # 先建基础 env（_make_mock_pandoc 会写入非计数版 pandoc），再覆盖为计数版
        env = _make_env(tmp_path, tmp_path / "data", tmp_path / "pi-calls.txt")
        mock_bin = tmp_path / "mock-bin"
        pandoc = mock_bin / "pandoc"
        pandoc.write_text(
            "#!/usr/bin/env python3\n"
            "import sys, os\n"
            "c = os.environ.get('MOCK_PANDOC_COUNTER', '')\n"
            "if c:\n"
            "    with open(c, 'a', encoding='utf-8') as f:\n"
            "        f.write('call\\n')\n"
            "try:\n"
            "    out_idx = sys.argv.index('-o')\n"
            "    out_path = sys.argv[out_idx + 1]\n"
            "    content = 'converted docx content for testing'\n"
            "    text_obj = f'BT /F1 12 Tf 100 700 Td ({content}) Tj ET'\n"
            "    pdf = b'%PDF-1.4\\n1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\\n2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\\n3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >> endobj\\n' + f'4 0 obj << /Length {len(text_obj) + 2} >> stream\\n{text_obj}\\nendstream endobj\\n'.encode() + b'5 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\\ntrailer << /Size 6 /Root 1 0 R >>\\n%%EOF\\n'\n"
            "    with open(out_path, 'wb') as f:\n"
            "        f.write(pdf)\n"
            "except (ValueError, IndexError):\n"
            "    sys.exit(1)\n"
        )
        pandoc.chmod(pandoc.stat().st_mode | stat.S_IEXEC)
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
        env["MOCK_PANDOC_COUNTER"] = str(counter)
        return env

    def test_resume_03_reruns_missing_subjects(self, tmp_path: Path):
        """中断在 03：03 只重跑被删篇，query 映射保持完整。"""
        data_dir, input_dir, env = _setup_data(tmp_path, ("alpha", "beta", "gamma"))

        r1 = _run_first(data_dir, input_dir, env)
        assert r1.returncode == 0, f"首次失败:\n{r1.stdout[-600:]}"

        task_dir = _find_task_dir(data_dir / "output")
        # 模拟中断在 03：删 beta 的 03 产物 + 03 全局产物
        (task_dir / "intermediates" / "beta" / "03-generate-query" / "output.json").unlink()
        (task_dir / "intermediates" / "pre" / "03-generate-query" / "output.json").unlink()
        from paper_review.orchestrator import write_task_manifest

        write_task_manifest(task_dir, status="running")

        r2 = _run_resume(data_dir, input_dir, env)
        assert r2.returncode == 0, f"续做失败:\n{r2.stdout[-600:]}"

        pre_out = _read_json(
            task_dir / "intermediates" / "pre" / "03-generate-query" / "output.json"
        )
        assert pre_out["status"] == "ok"
        assert pre_out["data"]["reused_count"] == 2, pre_out["data"]
        # query 映射完整（3 篇）
        assert set(pre_out["data"]["queries"]) == {"alpha", "beta", "gamma"}

    def test_resume_01_reuses_converted_pdf(self, tmp_path: Path):
        """中断在 01（docx 转换中途）：resume 复用已转换 PDF，不重复调 pandoc。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "index").mkdir()
        (data_dir / ".first-use-hint-shown").touch()
        _setup_pipeline(data_dir / "pipelines")

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        # 2 个 docx（走 pandoc 转换）+ 1 个 pdf
        from tests.e2e.test_pipeline_integration import _make_docx

        _make_docx(input_dir / "alpha.docx", "alpha content")
        _make_docx(input_dir / "beta.docx", "beta content")
        _make_pdf(input_dir / "gamma.pdf", "gamma content")

        counter = tmp_path / "pandoc-calls.txt"
        env = self._setup_env_with_counting_pandoc(tmp_path, counter)

        r1 = _run_first(data_dir, input_dir, env)
        assert r1.returncode == 0, f"首次失败:\n{r1.stdout[-600:]}"
        calls1 = counter.read_text().count("call")
        assert calls1 == 2, f"首次应转换 2 个 docx: {calls1}"

        task_dir = _find_task_dir(data_dir / "output")
        # 模拟中断在 01：删 01 全局产物（manifest 已存在，转换产物 pdf/ 下已有）
        (task_dir / "intermediates" / "pre" / "01-convert" / "output.json").unlink()
        from paper_review.orchestrator import write_task_manifest

        write_task_manifest(task_dir, status="running")

        r2 = _run_resume(data_dir, input_dir, env)
        assert r2.returncode == 0, f"续做失败:\n{r2.stdout[-600:]}"

        # 01 重跑但复用已转换 PDF（pandoc 不重复调用）
        calls2 = counter.read_text().count("call")
        assert calls2 == calls1, f"01 应复用已转换 PDF（pandoc 不重复调用）: {calls1} → {calls2}"
        pre_out = _read_json(task_dir / "intermediates" / "pre" / "01-convert" / "output.json")
        assert pre_out["status"] == "ok"
        assert pre_out["data"]["reused_count"] == 2, pre_out["data"]
        # manifest 覆盖全部 3 篇
        manifest = pre_out["data"]["manifest"]
        assert {s["name"] for s in manifest["subjects"]} == {"alpha", "beta", "gamma"}
