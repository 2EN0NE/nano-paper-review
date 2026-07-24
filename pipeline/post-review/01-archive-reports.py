"""
01-archive-reports.py — 归档评审报告
将评审中间产物汇总为最终报告写入 reports/ 目录
"""

from __future__ import annotations

import json
import os
from pathlib import Path


def main():
    output_dir = os.environ.get("PIPELINE_OUTPUT_DIR", ".")
    step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")

    output_root = Path(output_dir)
    intermediates_dir = output_root / "intermediates"
    reports_dir = output_root / "reports"

    # 收集所有 Subject 的 review 中间产物
    if not intermediates_dir.exists():
        output = {
            "step": "01-archive-reports",
            "status": "ok",
            "error": None,
            "data": {"archived_subjects": [], "total": 0},
        }
    else:
        archived = []
        for subj_dir in intermediates_dir.iterdir():
            if not subj_dir.is_dir():
                continue
            # 读取所有步骤的 output.json
            step_outputs = {}
            for step_dir_entry in sorted(subj_dir.iterdir()):
                if not step_dir_entry.is_dir():
                    continue
                output_file = step_dir_entry / "output.json"
                if output_file.exists():
                    try:
                        with open(output_file) as f:
                            step_outputs[step_dir_entry.name] = json.load(f)
                    except (json.JSONDecodeError, OSError):
                        step_outputs[step_dir_entry.name] = {"status": "error"}

            # 汇总为报告
            report = {
                "subject": subj_dir.name,
                "steps": step_outputs,
                "all_ok": all(s.get("status") == "ok" for s in step_outputs.values()),
            }
            archived.append(subj_dir.name)

            # 写入 reports/
            subject_report_dir = reports_dir / subj_dir.name
            subject_report_dir.mkdir(parents=True, exist_ok=True)
            with open(subject_report_dir / "report.json", "w") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        output = {
            "step": "01-archive-reports",
            "status": "ok",
            "error": None,
            "data": {"archived_subjects": archived, "total": len(archived)},
        }

    try:
        os.makedirs(step_dir, exist_ok=True)
        with open(os.path.join(step_dir, "output.json"), "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Failed to write archive output: {e}", file=__import__("sys").stderr)

    print(f"Archived {len(output['data']['archived_subjects'])} subject report(s)")


if __name__ == "__main__":
    main()
