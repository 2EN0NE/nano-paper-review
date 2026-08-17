"""
09-archive-reports.py — 归档评审报告 + 索引写回（标签 + Pool Promotion）

将评审中间产物汇总为最终报告写入 reports/ 目录；并把本批已索引的 Subject
从 Pending Pool 提升到 History Pool（Pool Promotion，ADR 0016），使审过的
论文成为后续 review 的潜在 Reference。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main():
    output_dir = os.environ.get("PIPELINE_OUTPUT_DIR", ".")
    step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")

    output_root = Path(output_dir)
    # 修复：intermediates 实际位于 result/{task_id}/intermediates（orchestrator 注入的
    # PIPELINE_INTERMEDIATES），而非 output_root/intermediates。缺失环境变量时回退旧
    # 路径（runpy 直调/单测兼容）。
    intermediates_dir = Path(
        os.environ.get("PIPELINE_INTERMEDIATES", str(output_root / "intermediates"))
    )
    reports_dir = output_root / "reports"

    # 收集所有 Subject 的 review 中间产物
    if not intermediates_dir.exists():
        output = {
            "step": "09-archive-reports",
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
            try:
                with open(subject_report_dir / "report.json", "w") as f:
                    json.dump(report, f, ensure_ascii=False, indent=2)
            except OSError as e:
                print(
                    f"Failed to write report for {subj_dir.name}: {e}",
                    file=__import__("sys").stderr,
                )

        output = {
            "step": "09-archive-reports",
            "status": "ok",
            "error": None,
            "data": {"archived_subjects": archived, "total": len(archived)},
        }

    # ── 索引写回：标签 + Pool Promotion（ADR 0016）──
    tags_written = 0
    promoted = 0
    promote_error = None
    pending_before = 0
    store_dir = Path(os.environ.get("PIPELINE_INDEX_STORE_DIR", "./index"))
    auto_index_output = intermediates_dir / "pre" / "02-auto-index" / "output.json"
    subject_paper_ids: dict[str, str] = {}
    if auto_index_output.exists():
        try:
            auto_index_data = json.loads(auto_index_output.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠ 无法读取 02-auto-index 产物，标签写回跳过: {e}", file=sys.stderr)
        else:
            subject_paper_ids = (auto_index_data.get("data") or {}).get("subject_paper_ids", {})

    if subject_paper_ids:
        try:
            from paper_review.search.store import Store

            store = Store(str(store_dir / "index.sqlite"))
        except Exception as e:
            print(f"  ⚠ 打开索引失败，标签写回跳过: {e}", file=sys.stderr)
        else:
            try:
                for subject_name, paper_id in subject_paper_ids.items():
                    scoring_output = (
                        intermediates_dir / subject_name / "06-direct-scoring" / "output.json"
                    )
                    if not scoring_output.exists():
                        continue
                    try:
                        scoring_data = json.loads(scoring_output.read_text(encoding="utf-8"))
                        tags = (scoring_data.get("data") or {}).get("tags") or []
                    except (json.JSONDecodeError, OSError):
                        continue
                    if not isinstance(tags, list):
                        continue
                    tags = [str(t) for t in tags if isinstance(t, str) and t.strip()]
                    if not tags:
                        continue
                    try:
                        updated = store.update_tags(paper_id, tags)
                    except Exception as e:
                        print(f"  ⚠ 标签写回失败 {subject_name}: {e}", file=sys.stderr)
                        continue
                    if updated:
                        tags_written += 1
                        print(f"  ✓ 标签写回: {subject_name} → {tags}")

                # ── Pool Promotion：本批已索引 Subject → history ──
                # 先查 pending 数：整批重跑时全部已是 history（pending=0）属正常，
                # 不触发哨兵告警；真的提升失败则记录 promote_error 供哨兵区分。
                try:
                    pending_before = store.count_pending(list(subject_paper_ids.values()))
                    promoted = store.promote_to_history(list(subject_paper_ids.values()))
                    print(f"  ✓ 池提升: {promoted} 篇 pending → history")
                except Exception as e:
                    promote_error = f"{type(e).__name__}: {e}"
                    print(f"  ⚠ 池提升失败: {promote_error}", file=sys.stderr)
            finally:
                store.close()

    output["data"]["tags_written"] = tags_written
    output["data"]["promoted"] = promoted
    output["data"]["promote_error"] = promote_error
    output["data"]["pending_before"] = pending_before

    try:
        os.makedirs(step_dir, exist_ok=True)
        with open(os.path.join(step_dir, "output.json"), "w") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
    except OSError as e:
        print(f"Failed to write archive output: {e}", file=__import__("sys").stderr)

    print(
        f"Archived {len(output['data']['archived_subjects'])} subject report(s), "
        f"tags_written={tags_written}, promoted={promoted}"
    )


if __name__ == "__main__":
    main()
