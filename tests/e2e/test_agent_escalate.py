"""E2E: Agent 升级链（escalate）的 provider/model 参数端到端传递。

问题背景：用户 escalate 配置写裸 ``--model deepseek-v4-flash``（缺 --provider），
pi 在多个 provider 名下都有同名模型（如 cli-proxy-api 与 deepseek 各有
deepseek-v4-flash）时无法消歧，0.4~1s 内以 402/歧义秒败。修复是显式写
``--provider deepseek``（或 ``--model deepseek/deepseek-v4-flash`` 合并形式）。

本测试用 mock pi 二进制记录收到的 argv，跑完整 review（.md 步骤走升级链），
断言框架把 escalate 链条目的 provider/model flag **原样**传给 pi——不吞参、
不重排，确保「消歧」依赖的显式 provider 标识完整到达 pi。

不 mock 内部函数；唯一 mock 是外部工具 pi（mock 二进制记录 argv）。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.e2e.test_resume_worker_granularity import (
    _paper_review_bin,
    _setup_input,
)

pytestmark = pytest.mark.e2e


def _make_mock_pi(bindir: Path, record_file: Path) -> Path:
    """创建 mock pi 脚本：把 argv 追加到记录文件，stdout 输出合法 JSON。"""
    mock = bindir / "pi"
    mock.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        f"rec = {str(record_file)!r}\n"
        "with open(rec, 'a', encoding='utf-8') as f:\n"
        "    f.write(json.dumps(sys.argv, ensure_ascii=False) + '\\n')\n"
        "print(json.dumps({'status': 'ok', 'data': {'mocked': True}}, ensure_ascii=False))\n",
        encoding="utf-8",
    )
    mock.chmod(0o755)
    return mock


def _setup_escalate_pipeline(data_dir: Path) -> None:
    """创建含 .md review 步骤 + 显式 provider 升级链的管线定义。"""
    pipelines_dir = data_dir / "pipelines" / "e2e-escalate"
    pipelines_dir.mkdir(parents=True, exist_ok=True)

    (pipelines_dir / "pipeline.yaml").write_text(
        'name: "e2e-escalate"\n'
        "phases:\n"
        "  - name: pre\n"
        "    mode: batch\n"
        "    directory: pre-review/\n"
        '    manifest_step: "01-convert"\n'
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
        "    agent:\n"
        "      escalate:\n"
        '        - "pi -ne --provider deepseek --model deepseek-v4-flash"\n'
        '        - "pi -ne --provider deepseek --model deepseek-v4-pro"\n'
        "    pool:\n"
        "      workers: 1\n"
        "      timeout: 120\n"
        "      ordered: true\n"
        "  - name: post\n"
        "    mode: batch\n"
        "    directory: post-review/\n"
        "    duplicate_policy: skip\n"
        "    retry:\n"
        "      max_attempts: 1\n"
        "      on_failure: skip\n",
        encoding="utf-8",
    )

    # pre-review/01-convert.py —— 复用真实模板
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
    shutil.copy(src_convert, pre_dir / "01-convert.py")

    # review-pipeline/01-score.md —— .md 步骤（走升级链调用 pi）
    review_dir = pipelines_dir / "review-pipeline"
    review_dir.mkdir(parents=True, exist_ok=True)
    (review_dir / "01-score.md").write_text(
        "# 评分\n\n对 {subject.name} 打分并输出 JSON。\n", encoding="utf-8"
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
        encoding="utf-8",
    )


def _read_records(record_file: Path) -> list[list[str]]:
    lines = [line for line in record_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [json.loads(line) for line in lines]


class TestAgentEscalateProvider:
    """Layer 3 E2E：升级链 provider/model flag 完整传递到 pi 子进程。"""

    def test_escalate_provider_flag_reaches_pi(self, tmp_path: Path):
        data_dir = tmp_path / "data"
        (data_dir / "index").mkdir(parents=True)
        (data_dir / ".first-use-hint-shown").touch()
        _setup_escalate_pipeline(data_dir)
        input_dir = _setup_input(data_dir, "alpha", "beta")

        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        record_file = tmp_path / "pi-argv.jsonl"
        _make_mock_pi(mock_bin, record_file)

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")

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
            timeout=90,
            env=env,
        )
        assert result.returncode == 0, f"review 失败:\n{result.stdout[:800]}\n{result.stderr[:500]}"

        records = _read_records(record_file)
        assert records, "mock pi 未被调用（升级链未生效？）"
        # 2 个 subject，每个 attempt 1 调一次 pi
        assert len(records) == 2, f"预期 2 次 pi 调用，实际 {len(records)}"

        for argv in records:
            assert "--provider" in argv, f"缺 --provider: {argv}"
            provider_idx = argv.index("--provider")
            assert argv[provider_idx + 1] == "deepseek", f"provider 值错误: {argv}"
            assert "--model" in argv, f"缺 --model: {argv}"
            model_idx = argv.index("--model")
            assert argv[model_idx + 1] == "deepseek-v4-flash", f"model 值错误: {argv}"
            # 框架兜底追加的批处理必需参数
            assert "--no-session" in argv, f"缺 --no-session: {argv}"
            assert "-p" in argv and any(a.startswith("@") for a in argv), f"缺 prompt 文件: {argv}"
