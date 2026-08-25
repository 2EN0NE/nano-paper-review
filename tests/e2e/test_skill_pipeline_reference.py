"""
E2E: 锁定 skills/user/paper-review-pipeline/references/steps.md 的 .py 骨架约定。

references/steps.md 是给 agent 的「默认写法」，本测试锁定它与框架实际行为一致：
- 环境变量（PIPELINE_STEP_DIR / PIPELINE_SUBJECT）注入正确
- Path.mkdir + write_text 写法可运行（区别于 test_cli_review 的 os.makedirs + json.dump）
- output.json 符合最小 schema（step/status/error/data 四字段）
"""

from __future__ import annotations

import json
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


# 与 references/steps.md 里「.py 步骤最小骨架」一致的写法（锁定契约，不漂移）。
_REFERENCE_PY_SKELETON = """\
import json
import os
from pathlib import Path

def main():
    step_dir = os.environ["PIPELINE_STEP_DIR"]
    subject = os.environ.get("PIPELINE_SUBJECT", "_batch_")

    result = {
        "step": "01-skeleton",
        "status": "ok",
        "error": None,
        "data": {"subject": subject},
    }
    Path(step_dir).mkdir(parents=True, exist_ok=True)
    (Path(step_dir) / "output.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

if __name__ == "__main__":
    main()
"""


class TestSkillPipelineReference:
    def test_reference_py_skeleton_runs(self, tmp_path: Path):
        """references/steps.md 的 .py 骨架可运行，output.json 符合 schema 且 subject 已注入。"""
        data_dir = tmp_path / "data"
        (data_dir / "index").mkdir(parents=True)

        pipeline_dir = data_dir / "pipelines" / "standard"
        pipeline_dir.mkdir(parents=True)
        (pipeline_dir / "pipeline.yaml").write_text(
            "name: skeleton\nphases:\n"
            "  - name: review\n    mode: per_subject\n    directory: review-pipeline\n"
        )

        review_dir = pipeline_dir / "review-pipeline"
        review_dir.mkdir()
        (review_dir / "01-skeleton.py").write_text(_REFERENCE_PY_SKELETON)

        pdf = tmp_path / "paper.pdf"
        pdf.write_text("dummy pdf")

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

        assert result.returncode == 0, (
            f"review failed:\nstdout:{result.stdout[:500]}\nstderr:{result.stderr[:500]}"
        )

        outs = list((data_dir / "output").rglob("01-skeleton/output.json"))
        assert outs, "未找到 01-skeleton 的 output.json"
        payload = json.loads(outs[0].read_text(encoding="utf-8"))

        # 最小 schema（references/steps.md 约定的四字段）
        assert payload["step"] == "01-skeleton"
        assert payload["status"] == "ok"
        assert payload["error"] is None
        assert payload["data"]["subject"], "PIPELINE_SUBJECT 应已注入（Review 模式）"
