"""
01-convert.py — 格式归一化：doc/docx → PDF + 产 Subject Manifest

将输入目录中的 doc/docx 文件转换为 PDF，产 manifest 供下游步骤消费。
仅 PDF 输入时：扫描目录写 manifest，不做转换。

依赖：
  - pandoc + weasyprint（docx → PDF）
  - libreoffice（.doc 降级尝试，可选）
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

# ============================================================================
# 工具函数
# ============================================================================


def _find_pandoc() -> str | None:
    return shutil.which("pandoc")


def _find_libreoffice() -> str | None:
    return shutil.which("libreoffice")


def _extract_pdf_text(pdf_path: Path) -> str:
    """从 PDF 提取文本用于 manifest 中的预览。

    PyMuPDF 未安装时静默返回空字符串（可选依赖）。
    运行时异常记录 warning 后返回空字符串。
    """
    import logging

    _logger = logging.getLogger(__name__)
    try:
        import fitz  # PyMuPDF

        doc = fitz.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text()
            if len(text) > 500:
                break
        doc.close()
        return text[:500]
    except ImportError:
        return ""
    except Exception as e:
        _logger.warning("Failed to extract PDF text from %s: %s", pdf_path, e)
        return ""


def _convert_docx_to_pdf(docx_path: Path, pdf_path: Path) -> tuple[bool, str]:
    """用 pandoc + weasyprint 将 docx 转为 PDF。

    Returns:
        (success, error_reason)
    """
    pandoc = _find_pandoc()
    if not pandoc:
        return False, "pandoc not found in PATH"

    try:
        proc = subprocess.run(
            [
                pandoc,
                str(docx_path),
                "-o",
                str(pdf_path),
                "--pdf-engine=weasyprint",
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            error_tail = (
                proc.stderr.strip()[-300:]
                if proc.stderr.strip()
                else f"exit code {proc.returncode}"
            )
            return False, f"pandoc failed: {error_tail}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "pandoc timed out"
    except Exception as e:
        return False, str(e)


def _convert_doc_to_docx_via_libreoffice(doc_path: Path, docx_path: Path) -> tuple[bool, str]:
    """用 libreoffice 将 .doc 转为 .docx（中间步骤，再走 pandoc）。

    Returns:
        (success, error_reason)
    """
    lo = _find_libreoffice()
    if not lo:
        return False, "libreoffice not found in PATH — cannot convert .doc files"

    output_dir = docx_path.parent
    try:
        proc = subprocess.run(
            [
                lo,
                "--headless",
                "--convert-to",
                "docx",
                "--outdir",
                str(output_dir),
                str(doc_path),
            ],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            error_tail = (
                proc.stderr.strip()[-200:]
                if proc.stderr.strip()
                else f"exit code {proc.returncode}"
            )
            return False, f"libreoffice failed: {error_tail}"
        # libreoffice 输出文件名与输入同名但扩展名 .docx
        expected = output_dir / (doc_path.stem + ".docx")
        if not expected.exists():
            return False, "libreoffice completed but output file not found"
        # 重命名为目标路径
        if expected != docx_path:
            expected.rename(docx_path)
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "libreoffice timed out"
    except Exception as e:
        return False, str(e)


# ============================================================================
# 主逻辑
# ============================================================================


def main():
    step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")
    output_dir = os.environ.get("PIPELINE_OUTPUT_DIR", ".")
    input_path_str = os.environ.get("PIPELINE_INPUT_PATH", "")

    if not input_path_str:
        # fallback: 没有注入时用 cwd
        input_path_str = os.environ.get("PIPELINE_SUBJECT", "") or str(Path.cwd())

    input_path = Path(input_path_str)

    # 如果是单文件，解析父目录
    if input_path.is_file():
        source_dir = input_path.parent
        single_file_mode = True
        single_file = input_path
    else:
        source_dir = input_path
        single_file_mode = False
        single_file = None

    manifest_path = Path(output_dir) / "subject-manifest.json"
    pdf_base_dir = source_dir / "pdf"

    results: list[dict] = []  # converted/subject entries
    skipped: list[dict] = []  # failed entries

    # ── 收集需要处理的文件 ──
    files_to_process: list[Path] = []
    if single_file_mode and single_file:
        files_to_process = [single_file]
    else:
        files_to_process = sorted(
            [f for f in source_dir.iterdir() if f.is_file() and not f.name.startswith(".")],
            key=lambda f: (f.suffix.lower() != ".pdf", f.name),  # PDF 优先，确保去重以原始 PDF 为准
        )

    # ── 处理每个文件 ──
    for f in files_to_process:
        suffix = f.suffix.lower()
        stem = f.stem

        if suffix == ".pdf":
            # 原始 PDF：直接列入 manifest
            results.append(
                {
                    "name": stem,
                    "pdf_path": str(f.absolute()),
                    "original_path": str(f.absolute()),
                    "source_type": "original_pdf",
                }
            )

        elif suffix == ".docx":
            # docx → PDF
            pdf_base_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = pdf_base_dir / (stem + ".pdf")
            success, error = _convert_docx_to_pdf(f, pdf_path)
            if success and pdf_path.exists():
                results.append(
                    {
                        "name": stem,
                        "pdf_path": str(pdf_path.absolute()),
                        "original_path": str(f.absolute()),
                        "source_type": "converted_docx",
                    }
                )
                # 检查 pdf/ 下是否有同名原始 PDF（之前已加入）
                dup_original = [
                    r for r in results if r["name"] == stem and r["source_type"] == "original_pdf"
                ]
                if dup_original:
                    # 转换产物和原始 PDF 同名，保留先出现的原始 PDF，跳过此转换产物
                    results[:] = [
                        r
                        for r in results
                        if not (r["name"] == stem and r["source_type"] == "converted_docx")
                    ]
                    skipped.append(
                        {
                            "name": stem,
                            "reason": f"duplicate: original PDF '{stem}.pdf' already exists, converted docx skipped",
                            "original_path": str(f.absolute()),
                        }
                    )
            else:
                skipped.append(
                    {
                        "name": stem,
                        "reason": error or "unknown error",
                        "original_path": str(f.absolute()),
                    }
                )

        elif suffix == ".doc":
            # .doc → docx → PDF（两跳）
            pdf_base_dir.mkdir(parents=True, exist_ok=True)
            docx_temp = pdf_base_dir / (stem + ".docx")
            docx_ok, docx_error = _convert_doc_to_docx_via_libreoffice(f, docx_temp)
            if not docx_ok:
                skipped.append(
                    {
                        "name": stem,
                        "reason": f"libreoffice .doc→.docx conversion failed: {docx_error}",
                        "original_path": str(f.absolute()),
                    }
                )
                continue

            pdf_path = pdf_base_dir / (stem + ".pdf")
            pdf_ok, pdf_error = _convert_docx_to_pdf(docx_temp, pdf_path)

            # 清理临时 docx
            if docx_temp.exists():
                docx_temp.unlink()

            if pdf_ok and pdf_path.exists():
                results.append(
                    {
                        "name": stem,
                        "pdf_path": str(pdf_path.absolute()),
                        "original_path": str(f.absolute()),
                        "source_type": "converted_doc",
                    }
                )
            else:
                skipped.append(
                    {
                        "name": stem,
                        "reason": f"pandoc .docx→.pdf failed: {pdf_error}",
                        "original_path": str(f.absolute()),
                    }
                )
        else:
            skipped.append(
                {
                    "name": stem,
                    "reason": f"unsupported format: {suffix}",
                    "original_path": str(f.absolute()),
                }
            )

    # ── 写 manifest ──
    manifest = {
        "source": "01-convert",
        "total_input": len(files_to_process),
        "converted": len(results),
        "skipped": len(skipped),
        "subjects": results,
        "skipped_entries": skipped,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    # ── 写本步骤的 output.json ──
    output = {
        "step": "01-convert",
        "status": "ok",
        "error": None,
        "data": {
            "converted_count": len(results),
            "skipped_count": len(skipped),
            "manifest_path": str(manifest_path),
            "manifest": manifest,
        },
    }

    os.makedirs(step_dir, exist_ok=True)
    with open(os.path.join(step_dir, "output.json"), "w") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"01-convert: {len(results)} converted, {len(skipped)} skipped")
    for s in skipped:
        print(f"  SKIP {s['name']}: {s['reason']}")


if __name__ == "__main__":
    main()
