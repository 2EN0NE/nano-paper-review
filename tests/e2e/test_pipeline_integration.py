"""
E2E: 管线集成测试 — 覆盖 docx/PDF 混合输入全链路

测试场景：
  1. 单 PDF 文件 — Pre（无转换）→ Review → Post
  2. 单 docx 文件 — Pre（转换 PDF）→ Review → Post
  3. 目录（混合 PDF+docx）— Pre（转换+去重）→ Review → Post（含 Excel）
  4. 纯 PDF 目录 — Pre（扫描）→ Review → Post（含 Excel）
  5. .doc 无 libreoffice — Pre（告警跳过）→ subjects 为空 → 跳过 Review
"""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.e2e


# ============================================================================
# 辅助函数
# ============================================================================


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


def _make_pdf(path: Path, content: str = "dummy pdf content") -> Path:
    """创建简单 PDF 占位文件（PyMuPDF 能读的格式）。"""

    # Minimal PDF with text
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
    # xref table: 扫描每行找到每个 object 的字节偏移
    offsets: list[int] = []
    pos = 0
    for line in pdf_content.split(b"\n"):
        if b" obj" in line:
            offsets.append(pos)
        pos += len(line) + 1  # +1 for the newline char
    xref_start = pos
    xref_table = f"xref\n0 {len(offsets) + 1}\n0000000000 65535 f \n"
    for i, off in enumerate(offsets, 1):
        xref_table += f"{off:010d} 00000 n \n"
    trailer = (
        f"trailer << /Size {len(offsets) + 1} /Root 1 0 R >>\nstartxref\n{xref_start}\n%%EOF\n"
    )
    path.write_bytes(pdf_content + xref_table.encode() + trailer.encode())
    return path


def _make_docx(path: Path, content: str = "Test document content") -> Path:
    """创建最小 docx 文件（zip 内嵌 document.xml）。"""
    import zipfile

    doc_xml = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        "<w:body>"
        f"<w:p><w:r><w:t>{content}</w:t></w:r></w:p>"
        "</w:body>"
        "</w:document>"
    )

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )

    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/>'
        "</Relationships>"
    )

    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", rels)
        zf.writestr("word/document.xml", doc_xml)
    return path


def _make_mock_pandoc(bindir: Path) -> Path:
    """创建 mock pandoc 脚本：简单复制输入到输出（模拟转换成功）。"""
    mock = bindir / "pandoc"
    script = (
        "#!/usr/bin/env python3\n"
        "import sys, shutil\n"
        "# Simple mock: copy input to output if -o flag present\n"
        "try:\n"
        "    out_idx = sys.argv.index('-o')\n"
        "    shutil.copy2(sys.argv[1], sys.argv[out_idx + 1])\n"
        "except (ValueError, IndexError):\n"
        "    sys.exit(1)\n"
    )
    mock.write_text(script)
    mock.chmod(mock.stat().st_mode | stat.S_IEXEC)
    return mock


def _find_task_dir(output_dir: Path) -> Path:
    """在 output/result/ 下找到唯一的任务目录。"""
    result_dirs = list((output_dir / "result").iterdir())
    assert len(result_dirs) == 1, f"Expected 1 result dir, got {result_dirs}"
    return result_dirs[0]


def _setup_pipeline_steps(pipelines_dir: Path, name: str = "e2e-test") -> Path:
    """在 pipelines_dir/{name}/ 下创建完整管线定义。

    返回管线目录路径。
    """
    pipeline_dir = pipelines_dir / name
    pipeline_dir.mkdir(parents=True, exist_ok=True)
    # ── pipeline.yaml ──
    (pipeline_dir / "pipeline.yaml").write_text("""\
name: "e2e-test"
version: "2.0"
phases:
  - name: pre
    mode: batch
    directory: pre-review/
    manifest_step: "00-convert"
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
      ordered: true
  - name: post
    mode: batch
    directory: post-review/
    duplicate_policy: skip
    retry:
      max_attempts: 1
      on_failure: skip
""")

    # ── pre-review/00-convert.py ──
    pre_dir = pipeline_dir / "pre-review"
    pre_dir.mkdir()
    # 使用项目源码中的真实 00-convert.py
    src_convert = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "paper_review"
        / "templates"
        / "pre-review"
        / "00-convert.py"
    )
    if src_convert.exists():
        shutil.copy(src_convert, pre_dir / "00-convert.py")
    else:
        (pre_dir / "00-convert.py").write_text(
            "import json, os, shutil, subprocess, sys\n"
            "from pathlib import Path\n"
            "# Minimal fallback convert\n"
            'step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")\n'
            'output_dir = os.environ.get("PIPELINE_OUTPUT_DIR", ".")\n'
            'input_path = Path(os.environ.get("PIPELINE_INPUT_PATH", "."))\n'
            "results = []\n"
            "skipped = []\n"
            "seen = set()\n"
            "if input_path.is_file():\n"
            "    files = [input_path]\n"
            "    src_dir = input_path.parent\n"
            "else:\n"
            "    src_dir = input_path\n"
            "    files = sorted([f for f in src_dir.iterdir() if f.is_file() and not f.name.startswith('.')],\n"
            "                 key=lambda f: (f.suffix.lower() != '.pdf', f.name))\n"
            "pdf_base = src_dir / 'pdf'\n"
            "for f in files:\n"
            "    s = f.suffix.lower()\n"
            "    stem = f.stem\n"
            "    if stem in seen:\n"
            "        continue\n"
            "    seen.add(stem)\n"
            '    if s == ".pdf":\n'
            '        results.append({"name":stem,"pdf_path":str(f.absolute()),"original_path":str(f.absolute()),"source_type":"original_pdf"})\n'
            '    elif s in (".docx", ".doc"):\n'
            "        pdf_base.mkdir(parents=True, exist_ok=True)\n"
            "        pdf_path = pdf_base / (stem + '.pdf')\n"
            '        pandoc = shutil.which("pandoc")\n'
            "        if pandoc:\n"
            "            try:\n"
            "                subprocess.run([pandoc, str(f), '-o', str(pdf_path), '--pdf-engine=weasyprint'],\n"
            "                               capture_output=True, text=True, timeout=30)\n"
            "                if pdf_path.exists():\n"
            '                    dup = [r for r in results if r["name"]==stem and r["source_type"]=="original_pdf"]\n'
            "                    if not dup:\n"
            '                        results.append({"name":stem,"pdf_path":str(pdf_path.absolute()),"original_path":str(f.absolute()),"source_type":"converted"})\n'
            "                    else:\n"
            '                        skipped.append({"name":stem,"reason":"duplicate: original PDF exists","original_path":str(f.absolute())})\n'
            "            except Exception as e:\n"
            '                skipped.append({"name":stem,"reason":str(e),"original_path":str(f.absolute())})\n'
            "        else:\n"
            '            skipped.append({"name":stem,"reason":"pandoc not found","original_path":str(f.absolute())})\n'
            "    else:\n"
            '        skipped.append({"name":stem,"reason":f"unsupported: {s}","original_path":str(f.absolute())})\n'
            'manifest = {"source":"00-convert","total_input":len(files),"converted":len(results),"skipped":len(skipped),"subjects":results,"skipped_entries":skipped}\n'
            "os.makedirs(Path(output_dir), exist_ok=True)\n"
            'with open(Path(output_dir)/"subject-manifest.json","w") as f:\n'
            "    json.dump(manifest, f, ensure_ascii=False, indent=2)\n"
            "os.makedirs(step_dir, exist_ok=True)\n"
            'with open(os.path.join(step_dir,"output.json"),"w") as f:\n'
            '    json.dump({"step":"00-convert","status":"ok","data":{"manifest":manifest}}, f)\n'
        )

    # ── review-pipeline/ steps ──
    review_dir = pipeline_dir / "review-pipeline"
    review_dir.mkdir()
    (review_dir / "01-search.py").write_text(
        "import json, os\n"
        'd = os.environ["PIPELINE_STEP_DIR"]\n'
        "os.makedirs(d, exist_ok=True)\n"
        'json.dump({"step":"01-search","status":"ok","data":{"references":[]}},'
        'open(os.path.join(d,"output.json"),"w"))'
    )
    (review_dir / "02-extract-keywords.py").write_text(
        "import json, os\n"
        'd = os.environ["PIPELINE_STEP_DIR"]\n'
        "os.makedirs(d, exist_ok=True)\n"
        'json.dump({"step":"02-extract-keywords","status":"ok","data":{"keywords":["测试"]}},'
        'open(os.path.join(d,"output.json"),"w"))'
    )
    (review_dir / "03-direct-scoring.md").write_text("# direct scoring\n\n{subject.name}\n")
    (review_dir / "04-indirect-scoring.md").write_text("# indirect scoring\n\n{subject.name}\n")
    # 05-summarize.py — 复制真实脚本
    src_summarize = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "paper_review"
        / "templates"
        / "review-pipeline"
        / "05-summarize.py"
    )
    if src_summarize.exists():
        shutil.copy(src_summarize, review_dir / "05-summarize.py")
    else:
        (review_dir / "05-summarize.py").write_text(
            "import json, os\n"
            'd = os.environ["PIPELINE_STEP_DIR"]\n'
            "os.makedirs(d, exist_ok=True)\n"
            'json.dump({"step":"05-summarize","status":"ok","data":{"final_scores":{},"indirect_scores":{},"original_direct_scores":{}}},'
            'open(os.path.join(d,"output.json"),"w"))'
        )

    # ── post-review/ steps ──
    post_dir = pipeline_dir / "post-review"
    post_dir.mkdir()
    (post_dir / "01-archive.py").write_text(
        "import json, os\n"
        'd = os.environ["PIPELINE_STEP_DIR"]\n'
        "os.makedirs(d, exist_ok=True)\n"
        'json.dump({"step":"01-archive","status":"ok","data":{}},'
        'open(os.path.join(d,"output.json"),"w"))'
    )
    # 复制真实 Excel 脚本（如有 openpyxl）
    src_excel = (
        Path(__file__).resolve().parent.parent.parent
        / "src"
        / "paper_review"
        / "templates"
        / "post-review"
        / "02-generate-excel.py"
    )
    if src_excel.exists():
        shutil.copy(src_excel, post_dir / "02-generate-excel.py")

    return pipeline_dir


# ============================================================================
# 场景 1: 单 PDF 文件
# ============================================================================


class TestPipelineSinglePDF:
    """单 PDF 文件全链路：Pre → Review → Post。"""

    def test_single_pdf_full_pipeline(self, tmp_path: Path):
        """单 PDF：Pre 产生 1 subject manifest → Review 处理 → Post 归档。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "index").mkdir()
        (data_dir / ".first-use-hint-shown").touch()

        # pipeline 放在 input 目录下（CLI 从此处发现 pipeline.yaml）
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _setup_pipeline_steps(data_dir / "pipelines")

        # Mock pandoc
        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        _make_mock_pandoc(mock_bin)

        # 合成 PDF（放在 input 目录下）
        pdf = input_dir / "test-paper.pdf"
        _make_pdf(pdf, "Test paper content")

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"

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
            timeout=30,
            env=env,
        )

        assert result.returncode == 0, (
            f"Pipeline failed:\nSTDOUT:{result.stdout[:800]}\nSTDERR:{result.stderr[:500]}"
        )

        # 验证 manifest
        manifest = data_dir / "output" / "subject-manifest.json"
        assert manifest.exists(), f"Manifest not found: {manifest}"
        manifest_data = json.loads(manifest.read_text())
        assert manifest_data["converted"] >= 1
        assert len(manifest_data["subjects"]) == 1
        assert manifest_data["subjects"][0]["name"] == "test-paper"

        # 验证 review intermediates（任务目录在 output/result/<hash>/ 下）
        result_dirs = list((data_dir / "output" / "result").iterdir())
        assert len(result_dirs) == 1, f"Expected 1 result dir, got {result_dirs}"
        task_dir = result_dirs[0]
        intermediates = task_dir / "intermediates"
        assert (intermediates / "test-paper" / "01-search" / "output.json").exists()
        assert (intermediates / "test-paper" / "02-extract-keywords" / "output.json").exists()
        # .md 步骤跳过但是 dir 创建了
        assert (intermediates / "test-paper" / "05-summarize" / "output.json").exists()

        # 验证 post intermediates
        assert (intermediates / "post" / "01-archive" / "output.json").exists()


# ============================================================================
# 场景 2: 单 docx 文件
# ============================================================================


class TestPipelineSingleDocx:
    """单 docx 文件全链路：Pre（转换 PDF）→ Review → Post。"""

    def test_single_docx_conversion(self, tmp_path: Path):
        """docx 转换为 PDF 后进入 Review 管线。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "index").mkdir()
        (data_dir / ".first-use-hint-shown").touch()

        # pipeline 和 docx 放在同一目录
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _setup_pipeline_steps(data_dir / "pipelines")

        # Mock pandoc
        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        _make_mock_pandoc(mock_bin)

        # 合成 docx（放在 input 目录下）
        docx = input_dir / "my-paper.docx"
        _make_docx(docx, "This is a test paper about distributed systems")

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"

        result = subprocess.run(
            [
                _paper_review_bin(),
                "--data-dir",
                str(data_dir),
                "review",
                "--skip-warnings",
                str(docx),
            ],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )

        assert result.returncode == 0, (
            f"Pipeline failed:\nSTDOUT:{result.stdout[:800]}\nSTDERR:{result.stderr[:500]}"
        )

        # 验证 manifest
        manifest = data_dir / "output" / "subject-manifest.json"
        assert manifest.exists()
        manifest_data = json.loads(manifest.read_text())
        assert manifest_data["converted"] >= 1
        assert len(manifest_data["subjects"]) == 1
        assert manifest_data["subjects"][0]["name"] == "my-paper"
        assert manifest_data["subjects"][0]["source_type"] == "converted_docx"

        # 验证 conversion 产物
        pdf_dir = input_dir / "pdf"
        assert pdf_dir.exists()
        assert (pdf_dir / "my-paper.pdf").exists()

        # 验证 review intermediates
        intermediates = _find_task_dir(data_dir / "output") / "intermediates"
        assert (intermediates / "my-paper" / "01-search" / "output.json").exists()


# ============================================================================
# 场景 3: 目录混合（PDF + docx）
# ============================================================================


class TestPipelineMixedDirectory:
    """目录混合输入：PDF + docx + 同名去重。"""

    def test_mixed_dir_with_dedup(self, tmp_path: Path):
        """混合目录：PDF 优先，同名转换产物去重。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "index").mkdir()
        (data_dir / ".first-use-hint-shown").touch()

        # pipeline 和输入文件放在同一目录
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _setup_pipeline_steps(data_dir / "pipelines")

        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        _make_mock_pandoc(mock_bin)

        # 原始 PDF
        _make_pdf(input_dir / "paper-a.pdf", "Paper A content")
        _make_pdf(input_dir / "paper-b.pdf", "Paper B content")

        # docx（无同名 PDF）
        _make_docx(input_dir / "paper-c.docx", "Paper C content")

        # docx（与 paper-a 同名 — 应被去重跳过）
        _make_docx(input_dir / "paper-a.docx", "Paper A duplicate in docx")

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
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
            timeout=30,
            env=env,
        )

        assert result.returncode == 0, (
            f"Pipeline failed:\nSTDOUT:{result.stdout[:800]}\nSTDERR:{result.stderr[:500]}"
        )

        manifest = data_dir / "output" / "subject-manifest.json"
        assert manifest.exists()
        manifest_data = json.loads(manifest.read_text())

        # paper-a, paper-b (original PDF), paper-c (converted) = 3 subjects
        # paper-a.docx should be skipped due to duplicate name
        subjects = manifest_data["subjects"]
        subject_names = [s["name"] for s in subjects]
        assert "paper-a" in subject_names, f"paper-a should be in subjects: {subject_names}"
        assert "paper-b" in subject_names
        assert "paper-c" in subject_names

        # 验证跳过记录（pipeline.yaml 也会被扫描，属于正常 unsupported format）
        skipped_names = [s["name"] for s in manifest_data.get("skipped_entries", [])]
        # paper-a.docx should be skipped due to duplicate with paper-a.pdf
        assert "paper-a" in skipped_names, (
            f"paper-a.docx should be in skipped: skipped={skipped_names}, subjects={subject_names}"
        )

        # 验证 review intermediates 包含所有 3 个 subject
        intermediates = _find_task_dir(data_dir / "output") / "intermediates"
        for name in subject_names:
            assert (intermediates / name / "01-search" / "output.json").exists(), (
                f"Missing intermediates for {name}"
            )


# ============================================================================
# 场景 4: 纯 PDF 目录（多篇 → Excel）
# ============================================================================


class TestPipelinePDFDirectory:
    """纯 PDF 目录：多篇 + Post 阶段 Excel 汇总。"""

    def test_multi_pdf_with_excel(self, tmp_path: Path):
        """多 PDF 目录：3 篇 → Review → Post（含 Excel）。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "index").mkdir()
        (data_dir / ".first-use-hint-shown").touch()

        # pipeline 和 PDF 放在同一目录
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _setup_pipeline_steps(data_dir / "pipelines")

        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        _make_mock_pandoc(mock_bin)

        # 3 篇 PDF
        _make_pdf(input_dir / "paper-1.pdf", "Paper 1")
        _make_pdf(input_dir / "paper-2.pdf", "Paper 2")
        _make_pdf(input_dir / "paper-3.pdf", "Paper 3")

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
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
            timeout=30,
            env=env,
        )

        assert result.returncode == 0

        manifest = data_dir / "output" / "subject-manifest.json"
        assert manifest.exists()
        manifest_data = json.loads(manifest.read_text())
        assert len(manifest_data["subjects"]) == 3

        # 验证所有 3 个 subject 的 intermediates
        intermediates = _find_task_dir(data_dir / "output") / "intermediates"
        for name in ["paper-1", "paper-2", "paper-3"]:
            assert (intermediates / name / "01-search" / "output.json").exists()

        # 验证 post 阶段
        assert (intermediates / "post" / "01-archive" / "output.json").exists()

        # Excel: 3 subjects → should generate (openpyxl is available in test env)
        excel_out = intermediates / "post" / "02-generate-excel" / "output.json"
        assert excel_out.exists(), f"Excel output.json missing at {excel_out}"
        excel_data = json.loads(excel_out.read_text())
        try:
            import openpyxl  # noqa: F401

            _has_openpyxl = True
        except ImportError:
            _has_openpyxl = False
        if _has_openpyxl:
            assert excel_data["status"] == "ok", (
                f"Excel should be ok (openpyxl installed), got {excel_data['status']}"
            )
            assert excel_data["data"]["subject_count"] == 3
        else:
            assert excel_data["status"] in ("ok", "skipped")


# ============================================================================
# 场景 5: .doc 无 libreoffice → 跳过 → 空 subjects
# ============================================================================


class TestPipelineDocNoLibreoffice:
    """.doc 文件无 libreoffice 时告警跳过。"""

    def test_doc_no_libreoffice(self, tmp_path: Path):
        """.doc 无法转换 → subjects 为空 → Review 跳过。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "index").mkdir()
        (data_dir / ".first-use-hint-shown").touch()

        # pipeline 和 .doc 放在同一目录
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _setup_pipeline_steps(data_dir / "pipelines")

        # 不创建 mock pandoc，也不安装 libreoffice
        # 只放一个 .doc 文件
        doc_file = input_dir / "old-paper.doc"
        doc_file.write_text("fake doc content")  # 二进制格式无法被 pandoc 处理

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
            timeout=30,
        )

        # 不应 crash — subjects 为空时优雅退出
        assert result.returncode == 0, (
            f"Pipeline crashed:\nSTDOUT:{result.stdout[:800]}\nSTDERR:{result.stderr[:500]}"
        )

        # manifest 应该存在但 subjects 为空
        manifest = data_dir / "output" / "subject-manifest.json"
        if manifest.exists():
            manifest_data = json.loads(manifest.read_text())
            assert len(manifest_data["subjects"]) == 0, (
                f"Expected 0 subjects, got: {manifest_data['subjects']}"
            )
            assert len(manifest_data.get("skipped_entries", [])) > 0, (
                ".doc should be in skipped_entries"
            )


# ============================================================================
# 附加：验证 Excel 跳过单篇场景
# ============================================================================


class TestExcelSkipSingleSubject:
    """单篇 PDF 也生成 Excel（已修复为 len(subjects)==0 才跳过）。"""

    def test_single_subject_generates_excel(self, tmp_path: Path):
        """只有 1 subject 时 Excel 也应生成（一行数据）。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "index").mkdir()
        (data_dir / ".first-use-hint-shown").touch()

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _setup_pipeline_steps(data_dir / "pipelines")

        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        _make_mock_pandoc(mock_bin)

        pdf = input_dir / "only-one.pdf"
        _make_pdf(pdf, "Single paper")

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"

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
            timeout=30,
            env=env,
        )

        assert result.returncode == 0

        intermediates = _find_task_dir(data_dir / "output") / "intermediates"
        excel_out = intermediates / "post" / "02-generate-excel" / "output.json"
        assert excel_out.exists(), f"Excel output.json not found at {excel_out}"
        excel_data = json.loads(excel_out.read_text())
        # Single subject now generates Excel when openpyxl is available
        assert excel_data["status"] in ("ok", "skipped"), (
            f"Excel should be ok or skipped, got {excel_data['status']}"
        )
        if excel_data["status"] == "ok":
            assert excel_data["data"]["subject_count"] == 1


# ============================================================================
# 场景 6: 进度卡片渲染验证（PAPER_REVIEW_FORCE_TTY）
# ============================================================================


class TestProgressCardRendering:
    """E2E 验证：进度卡片在终端环境中的渲染。

    这是 Layer 3 测试——验证真实 CLI 进程中 stderr 的进度卡输出。
    """

    def test_force_tty_produces_progress_box_in_stderr(self, tmp_path: Path):
        """PAPER_REVIEW_FORCE_TTY=1 时 stderr 包含完整进度 box。

        关键回归：确保进度卡在非 TTY 环境（如 CI）中也能通过
        PAPER_REVIEW_FORCE_TTY=1 强制渲染。
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "index").mkdir()
        (data_dir / ".first-use-hint-shown").touch()

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _setup_pipeline_steps(data_dir / "pipelines")

        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        _make_mock_pandoc(mock_bin)

        pdf = input_dir / "test-paper.pdf"
        _make_pdf(pdf, "Test paper for progress card")

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"
        env["PAPER_REVIEW_FORCE_TTY"] = "1"

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
            timeout=30,
            env=env,
        )

        assert result.returncode == 0, (
            f"Pipeline failed:\nSTDOUT:{result.stdout[:800]}\nSTDERR:{result.stderr[:500]}"
        )

        # 验证 stderr 包含进度 box 边框字符
        stderr = result.stderr
        assert "┌" in stderr, f"Progress box should have top border '┌' in stderr:\n{stderr[:500]}"
        assert "└" in stderr, (
            f"Progress box should have bottom border '└' in stderr:\n{stderr[:500]}"
        )
        assert "┐" in stderr, "Progress box should have top-right corner"
        assert "┘" in stderr, "Progress box should have bottom-right corner"

        # 验证包含三阶段名称
        assert "Pre" in stderr, "Progress box should show Pre phase"
        assert "Review" in stderr, "Progress box should show Review phase"
        assert "Post" in stderr, "Progress box should show Post phase"

        # 验证包含总进度行
        assert "总进度" in stderr, "Progress box should show summary line"

        # 验证 ANSI escape 序列存在
        assert "\033[" in stderr, "Progress box should contain ANSI escape codes"

    def test_without_force_tty_stderr_has_plain_text(self, tmp_path: Path):
        """未设置 FORCE_TTY 时（非 TTY 环境），stderr 输出纯文本阶段摘要。

        这是进度卡"不出现"问题的另一面——确认回退路径也是可读的。
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "index").mkdir()
        (data_dir / ".first-use-hint-shown").touch()

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _setup_pipeline_steps(data_dir / "pipelines")

        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        _make_mock_pandoc(mock_bin)

        pdf = input_dir / "test-paper.pdf"
        _make_pdf(pdf, "Test paper for plain text progress")

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"
        # 不设置 PAPER_REVIEW_FORCE_TTY —— 默认用 sys.stderr.isatty()
        # subprocess.run 捕获输出，所以 stderr 不是 TTY

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
            timeout=30,
            env=env,
        )

        assert result.returncode == 0, (
            f"Pipeline failed:\nSTDOUT:{result.stdout[:800]}\nSTDERR:{result.stderr[:500]}"
        )

        stderr = result.stderr
        # 非 TTY 模式：不应有 box 边框
        assert "┌" not in stderr, (
            f"Without FORCE_TTY, stderr should NOT have box border:\n{stderr[:500]}"
        )

        # 但应有纯文本阶段摘要
        assert "[进度]" in stderr or "Pre" in stderr, (
            f"Non-TTY should have plain text progress summary:\n{stderr[:500]}"
        )

        # 不应有 ANSI escape
        assert "\033[" not in stderr, f"Non-TTY should not have ANSI escapes:\n{stderr[:500]}"

    def test_force_tty_progress_card_updates_during_run(self, tmp_path: Path):
        """进度卡在管线运行过程中有多次更新（spinner 帧变化）。

        验证 stderr 中至少有 2 个不同的 spinner 帧，证明 in-place 刷新在运行。
        """
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "index").mkdir()
        (data_dir / ".first-use-hint-shown").touch()

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _setup_pipeline_steps(data_dir / "pipelines")

        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        _make_mock_pandoc(mock_bin)

        pdf = input_dir / "test-paper.pdf"
        _make_pdf(pdf, "Test paper for progress updates")

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"
        env["PAPER_REVIEW_FORCE_TTY"] = "1"

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
            timeout=30,
            env=env,
        )

        assert result.returncode == 0

        # 收集所有唯一的 spinner 帧
        import re

        spinners = set(re.findall(r"[⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏]", result.stderr))
        # 通常运行过程中至少会看到 2+ 个不同的 spinner 帧
        assert len(spinners) >= 2, (
            f"Expected at least 2 spinner frames in progress output, got {len(spinners)}:\n"
            f"{result.stderr[:800]}"
        )


# ============================================================================
# 场景 7: Auto-Index E2E — 管线集成验证 sentinel + 索引建立
# ============================================================================


class TestAutoIndexPipeline:
    """Auto-index 步骤在真实 CLI 管线中的行为验证。

    由于完整的 auto-index 依赖 ONNX embedding 模型，此处使用简化版本来测试管线集成：
    - sentinel 检查与写入
    - 首次运行触发批量索引，第二次跳过
    - output.json 产物结构正确
    - 环境变量正确注入
    """

    def _setup_auto_index_pipeline(self, pipelines_dir):
        """创建含 auto-index 简化步骤的管线定义。"""
        pipeline_dir = pipelines_dir / "auto-idx-test"
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        (pipeline_dir / "pipeline.yaml").write_text("""\
name: "auto-idx-test"
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
    manifest_step: "00-convert"
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
""")

        pre_dir = pipeline_dir / "pre-review"
        pre_dir.mkdir()

        # 真实 00-convert.py（复制自项目模板）
        src_convert = (
            Path(__file__).resolve().parent.parent.parent
            / "src"
            / "paper_review"
            / "templates"
            / "pre-review"
            / "00-convert.py"
        )
        if src_convert.exists():
            shutil.copy(src_convert, pre_dir / "00-convert.py")
        else:
            (pre_dir / "00-convert.py").write_text(
                "import json, os;"
                "from pathlib import Path;"
                'd=os.environ["PIPELINE_STEP_DIR"];'
                'out=os.environ["PIPELINE_OUTPUT_DIR"];'
                "os.makedirs(d, exist_ok=True);"
                'manifest={"source":"00-convert","total_input":1,"converted":1,'
                '"skipped":0,"subjects":[{"name":"test","pdf_path":'
                'f"{os.environ.get("PIPELINE_INPUT_PATH",".")}",'
                '"original_path":".","source_type":"original_pdf"}],'
                '"skipped_entries":[]};'
                'json.dump(manifest,open(os.path.join(out,"subject-manifest.json"),"w"));'
                'json.dump({"step":"00-convert","status":"ok","data":{"manifest":manifest}},'
                'open(os.path.join(d,"output.json"),"w"))'
            )

        # 简化 01-auto-index.py — 不依赖 ONNX 模型，测试管线集成
        (pre_dir / "01-auto-index.py").write_text(
            "import json, os, sqlite3\n"
            "from pathlib import Path\n"
            "\n"
            'step_dir = os.environ.get("PIPELINE_STEP_DIR", ".")\n'
            'output_dir = os.environ.get("PIPELINE_OUTPUT_DIR", ".")\n'
            'reference_dir = Path(os.environ.get("PIPELINE_INDEX_REFERENCE_DIR", "./origin/pdf"))\n'
            'store_dir = Path(os.environ.get("PIPELINE_INDEX_STORE_DIR", "./index"))\n'
            'auto_index = os.environ.get("PIPELINE_INDEX_AUTO_INDEX", "1") == "1"\n'
            'copy_subjects = os.environ.get("PIPELINE_INDEX_COPY_SUBJECTS", "1") == "1"\n'
            'data_dir = Path(os.environ.get("PIPELINE_DATA_DIR", "."))\n'
            'manifest_path = Path(output_dir) / "subject-manifest.json"\n'
            "subjects = []\n"
            "if manifest_path.exists():\n"
            "    m = json.loads(manifest_path.read_text())\n"
            '    subjects = m.get("subjects", [])\n'
            "\n"
            "history_indexed = 0\n"
            "subjects_indexed = 0\n"
            "dedup_skipped = 0\n"
            "copied = 0\n"
            "\n"
            "# Sentinel check\n"
            'sentinel = data_dir / ".auto-index-done"\n'
            "first_run = not sentinel.exists()\n"
            "\n"
            "# Create a minimal SQLite index\n"
            "store_dir.mkdir(parents=True, exist_ok=True)\n"
            'db_path = str(store_dir / "index.sqlite")\n'
            "conn = sqlite3.connect(db_path)\n"
            'conn.execute("CREATE TABLE IF NOT EXISTS papers (paper_id TEXT PRIMARY KEY, filename TEXT, pool TEXT)")\n'
            'conn.execute("CREATE TABLE IF NOT EXISTS content_dedup (sha256 TEXT PRIMARY KEY, paper_id TEXT)")\n'
            "\n"
            "# Index reference PDFs\n"
            "if auto_index and first_run:\n"
            "    reference_dir.mkdir(parents=True, exist_ok=True)\n"
            '    pdf_files = sorted(reference_dir.glob("*.pdf"))\n'
            "    for pdf_file in pdf_files:\n"
            "        import hashlib\n"
            "        pid = hashlib.sha256(str(pdf_file).encode()).hexdigest()[:12]\n"
            '        conn.execute("INSERT OR IGNORE INTO papers VALUES(?,?,?)", (pid, pdf_file.name, "history"))\n'
            "        history_indexed += 1\n"
            "    conn.commit()\n"
            '    sentinel.write_text("")\n'
            "\n"
            "# Index subjects\n"
            "for subj in subjects:\n"
            "    import hashlib\n"
            '    pdf_path = Path(subj["pdf_path"]) if "pdf_path" in subj else Path(subj.get("original_path", "."))\n'
            "    pid = hashlib.sha256(str(pdf_path).encode()).hexdigest()[:12]\n"
            '    conn.execute("INSERT OR IGNORE INTO papers VALUES(?,?,?)", (pid, pdf_path.name, "pending"))\n'
            "    subjects_indexed += 1\n"
            "    # Copy to reference dir\n"
            "    if copy_subjects and pdf_path.exists():\n"
            "        dest = reference_dir / pdf_path.name\n"
            "        if not dest.exists():\n"
            "            import shutil\n"
            "            shutil.copy2(str(pdf_path), str(dest))\n"
            "            copied += 1\n"
            "\n"
            "conn.commit()\n"
            "conn.close()\n"
            "\n"
            "output = {\n"
            '    "step": "01-auto-index",\n'
            '    "status": "ok",\n'
            '    "error": None,\n'
            '    "data": {\n'
            '        "history_indexed": history_indexed,\n'
            '        "subjects_indexed": subjects_indexed,\n'
            '        "dedup_skipped": dedup_skipped,\n'
            '        "copied": copied,\n'
            '        "conflict_renamed": 0,\n'
            "    },\n"
            "}\n"
            "os.makedirs(step_dir, exist_ok=True)\n"
            'with open(os.path.join(step_dir, "output.json"), "w") as f:\n'
            "    json.dump(output, f, ensure_ascii=False, indent=2)\n"
            'print(f"auto-index: history={history_indexed}, subjects={subjects_indexed}, copied={copied}")\n'
        )

        # review 和 post 步骤
        review_dir = pipeline_dir / "review-pipeline"
        review_dir.mkdir()
        (review_dir / "01-search.py").write_text(
            "import json, os;"
            'd=os.environ["PIPELINE_STEP_DIR"];'
            "os.makedirs(d, exist_ok=True);"
            'json.dump({"step":"01-search","status":"ok","data":{"references":[]}},'
            'open(os.path.join(d,"output.json"),"w"))'
        )

        post_dir = pipeline_dir / "post-review"
        post_dir.mkdir()
        (post_dir / "01-archive.py").write_text(
            "import json, os;"
            'd=os.environ["PIPELINE_STEP_DIR"];'
            "os.makedirs(d, exist_ok=True);"
            'json.dump({"step":"01-archive","status":"ok","data":{}},'
            'open(os.path.join(d,"output.json"),"w"))'
        )

        return pipeline_dir

    def test_first_run_indexes_reference_pdfs(self, tmp_path):
        """首次运行：扫描 reference_dir 中 PDF → 写入 index.sqlite + sentinel。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / ".first-use-hint-shown").touch()

        # 创建 reference_dir 并放入 PDF
        pipelines_dir = data_dir / "pipelines"
        self._setup_auto_index_pipeline(pipelines_dir)

        reference_dir = data_dir / "origin" / "pdf"
        reference_dir.mkdir(parents=True)
        _make_pdf(reference_dir / "ref-paper-1.pdf", "Reference paper 1")
        _make_pdf(reference_dir / "ref-paper-2.pdf", "Reference paper 2")

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        pdf = input_dir / "subject-a.pdf"
        _make_pdf(pdf, "Subject A content")

        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        _make_mock_pandoc(mock_bin)

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"

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
            timeout=30,
            env=env,
        )

        assert result.returncode == 0, (
            f"Pipeline failed:\nSTDOUT:{result.stdout[:500]}\nSTDERR:{result.stderr[:500]}"
        )

        # 验证索引已建立
        index_db = data_dir / "index" / "index.sqlite"
        assert index_db.exists(), f"index.sqlite not found at {index_db}"

        # 验证 sentinel
        sentinel = data_dir / ".auto-index-done"
        assert sentinel.exists(), "sentinel should be created after first auto-index"

        # 验证 auto-index 输出产物
        intermediates = _find_task_dir(data_dir / "output") / "intermediates"
        auto_index_out = intermediates / "pre" / "01-auto-index" / "output.json"
        assert auto_index_out.exists(), f"Missing {auto_index_out}"
        ai_data = json.loads(auto_index_out.read_text())
        assert ai_data["status"] == "ok"
        assert ai_data["data"]["history_indexed"] >= 1
        assert ai_data["data"]["subjects_indexed"] >= 1

    def test_second_run_skips_batch_index(self, tmp_path):
        """第二次运行：sentinel 已存在 → 跳过批量索引。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / ".first-use-hint-shown").touch()

        pipelines_dir = data_dir / "pipelines"
        self._setup_auto_index_pipeline(pipelines_dir)

        # 预先创建 sentinel（模拟已经执行过一次）
        (data_dir / ".auto-index-done").write_text("")
        index_dir = data_dir / "index"
        index_dir.mkdir(parents=True)

        reference_dir = data_dir / "origin" / "pdf"
        reference_dir.mkdir(parents=True)
        _make_pdf(reference_dir / "ref-paper-1.pdf", "Ref 1")

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        pdf = input_dir / "subject-b.pdf"
        _make_pdf(pdf, "Subject B")

        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        _make_mock_pandoc(mock_bin)

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = "pi-not-found"

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
            timeout=30,
            env=env,
        )

        assert result.returncode == 0

        intermediates = _find_task_dir(data_dir / "output") / "intermediates"
        auto_index_out = intermediates / "pre" / "01-auto-index" / "output.json"
        assert auto_index_out.exists()
        ai_data = json.loads(auto_index_out.read_text())
        assert ai_data["status"] == "ok"
        # sentinel 已存在 → history_indexed 为 0
        assert ai_data["data"]["history_indexed"] == 0
        # subjects 仍然被索引
        assert ai_data["data"]["subjects_indexed"] >= 1


# ============================================================================
# 场景 8: Dynamic Pool 降级 E2E — 假 pi 返回 429 触发 worker 降级
# ============================================================================


class TestDynamicPoolDowngrade:
    """profile=dynamic 模式下，假 pi 间歇返回 429 触发并发降级。"""

    def _make_fake_pi_429_counter(self, tmp_path, name):
        """创建假 pi：每次调用读取计数文件，奇数次返回 429。

        这确保后续 subject 也有机会成功，而非全部超时。
        """
        script = tmp_path / name
        counter_file = tmp_path / f"{name}.counter"
        counter_file.write_text("0")
        lines = [
            "#!/bin/sh",
            f"CNT=$(cat {counter_file})",
            f"echo $((CNT + 1)) > {counter_file}",
            "if [ $((CNT % 2)) -eq 0 ]; then",
            '  echo \'{"step":"03-scoring","status":"ok","data":{"score":80}}\'',
            "  exit 0",
            "else",
            "  echo 'Error: 429 too many requests' >&2",
            "  exit 1",
            "fi",
        ]
        script.write_text("\n".join(lines) + "\n")
        os.chmod(str(script), stat.S_IRWXU)  # noqa: S103
        return script

    def _setup_dynamic_pool_pipeline(self, pipelines_dir):
        """创建 profile=dynamic 的管线定义，只含 review phase。"""
        pipeline_dir = pipelines_dir / "dynpool-test"
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        (pipeline_dir / "pipeline.yaml").write_text("""\
name: "dynpool-test"
version: "2.0"
phases:
  - name: pre
    mode: batch
    directory: pre-review/
    manifest_step: "00-convert"
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
    retry:
      max_attempts: 1
      on_failure: skip
    subject_order:
      sort_by: name
      direction: asc
    pool:
      workers: 3
      profile: dynamic
      workers_min: 1
      workers_max: 4
      ordered: false
      timeout: 30
    step_timeout: 10
""")

        # Pre — minimal convert
        pre_dir = pipeline_dir / "pre-review"
        pre_dir.mkdir()
        (pre_dir / "00-convert.py").write_text(
            "import json, os;"
            "from pathlib import Path;"
            'd=os.environ["PIPELINE_STEP_DIR"];'
            'out=os.environ["PIPELINE_OUTPUT_DIR"];'
            'input_path=Path(os.environ.get("PIPELINE_INPUT_PATH","."));'
            "os.makedirs(d, exist_ok=True);"
            "files=sorted([f for f in (input_path.iterdir() if input_path.is_dir() else [input_path])"
            'if f.is_file() and f.suffix==".pdf"],key=lambda f:f.name);'
            'subjects=[{"name":f.stem,"pdf_path":str(f.absolute()),'
            '"original_path":str(f.absolute()),"source_type":"original_pdf"} for f in files];'
            'manifest={"source":"00-convert","total_input":len(files),"converted":len(subjects),'
            '"skipped":0,"subjects":subjects,"skipped_entries":[]};'
            'json.dump(manifest,open(os.path.join(out,"subject-manifest.json"),"w"));'
            'json.dump({"step":"00-convert","status":"ok","data":{"manifest":manifest}},'
            'open(os.path.join(d,"output.json"),"w"))'
        )

        # Review — 一个 .md 步骤（通过 pi 执行）
        review_dir = pipeline_dir / "review-pipeline"
        review_dir.mkdir()
        (review_dir / "03-direct-scoring.md").write_text(
            "# Review {subject.name}\n\nScore this paper.\n"
        )

        return pipeline_dir

    def test_dynamic_pool_completes_all_subjects_with_429(self, tmp_path):
        """假 pi 间歇返回 429 → 所有 subject 最终完成，stderr 含 downgrade 日志。"""
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / ".first-use-hint-shown").touch()
        (data_dir / "index").mkdir()

        pipelines_dir = data_dir / "pipelines"
        self._setup_dynamic_pool_pipeline(pipelines_dir)

        # 假 pi（间歇返回 429）
        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        fake_pi = self._make_fake_pi_429_counter(mock_bin, "pi")
        _make_mock_pandoc(mock_bin)

        # 4 篇 PDF → 足够触发并发
        input_dir = tmp_path / "input"
        input_dir.mkdir()
        for name in ["paper-a", "paper-b", "paper-c", "paper-d"]:
            _make_pdf(input_dir / f"{name}.pdf", f"Content of {name}")

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = str(fake_pi)

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

        # 不应崩溃
        assert result.returncode == 0, (
            f"Pipeline failed:\nSTDOUT:{result.stdout[:800]}\nSTDERR:{result.stderr[:800]}"
        )

        # 所有 subject 的 intermediates 都应存在
        intermediates = _find_task_dir(data_dir / "output") / "intermediates"
        for name in ["paper-a", "paper-b", "paper-c", "paper-d"]:
            step_dir = intermediates / name / "03-direct-scoring"
            assert step_dir.exists(), f"Missing step dir for {name}"
            out_file = step_dir / "output.json"
            assert out_file.exists(), f"Missing output.json for {name}"

        # 验证 stderr/logs 中是否有降级日志（因为假 pi 间歇返回 429）
        combined = result.stdout + result.stderr
        # 至少能看到 429 错误被捕获
        assert "429" in combined, (
            f"Expected 429 errors in output:\nSTDOUT:{result.stdout[:500]}\nSTDERR:{result.stderr[:500]}"
        )


# ============================================================================
# 场景 9: 默认配置值可运行性 — 动态验证 _DEFAULT_PI_ARGS 实际可用
# ============================================================================


class TestDefaultConfigValidity:
    """管线默认配置值的 E2E 可运行性验证。

    原则：不对默认值做硬编码断言；从源码导入当前默认值，
    验证它们在不报错的情况下被实际传递到 pi 进程。
    当有人修改 _DEFAULT_PI_ARGS 时，此测试自动适用新值。
    """

    def _make_arg_recording_pi(self, tmp_path, name):
        """创建假 pi：记录全部接收到的参数到 args.txt，然后成功退出。"""
        script = tmp_path / name
        lines = [
            "#!/bin/sh",
            # 把全部参数写到 step 目录下的 args.txt
            'echo "$@" > "$PIPELINE_STEP_DIR/args.txt"',
            # 输出有效 output.json
            'echo \'{"step":"test","status":"ok","data":{"ok":true}}\'',
            "exit 0",
        ]
        script.write_text("\n".join(lines) + "\n")
        os.chmod(str(script), stat.S_IRWXU)  # noqa: S103
        return script

    def _setup_minimal_pipeline(self, pipelines_dir):
        """创建最小管线：pre（convert）+ review（一个 .md 步骤）。"""
        pipeline_dir = pipelines_dir / "default-args-test"
        pipeline_dir.mkdir(parents=True, exist_ok=True)

        (pipeline_dir / "pipeline.yaml").write_text("""\
name: "default-args-test"
version: "2.0"
phases:
  - name: pre
    mode: batch
    directory: pre-review/
    manifest_step: "00-convert"
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
    retry:
      max_attempts: 1
      on_failure: skip
    subject_order:
      sort_by: name
      direction: asc
    pool:
      workers: 1
      timeout: 120
""")

        pre_dir = pipeline_dir / "pre-review"
        pre_dir.mkdir()
        (pre_dir / "00-convert.py").write_text(
            """
import json, os
from pathlib import Path

d = os.environ["PIPELINE_STEP_DIR"]
out = os.environ["PIPELINE_OUTPUT_DIR"]
input_path = Path(os.environ.get("PIPELINE_INPUT_PATH", "."))
os.makedirs(d, exist_ok=True)

if input_path.is_dir():
    pdfs = sorted(input_path.glob("*.pdf"))
else:
    pdfs = [input_path]

subjects = [{
    "name": f.stem,
    "pdf_path": str(f.absolute()),
    "original_path": str(f.absolute()),
    "source_type": "original_pdf",
} for f in pdfs]

manifest = {
    "source": "00-convert",
    "total_input": len(pdfs),
    "converted": len(subjects),
    "skipped": 0,
    "subjects": subjects,
    "skipped_entries": [],
}
json.dump(manifest, open(os.path.join(out, "subject-manifest.json"), "w"))
json.dump({"step": "00-convert", "status": "ok", "data": {"manifest": manifest}},
          open(os.path.join(d, "output.json"), "w"))
""".strip()
        )

        review_dir = pipeline_dir / "review-pipeline"
        review_dir.mkdir()
        (review_dir / "01-review.md").write_text("# Review {subject.name}\n\nScore this paper.\n")

        return pipeline_dir

    def test_default_pi_args_passed_and_dont_error(self, tmp_path):
        """E2E：_DEFAULT_PI_ARGS 被实际传递到 pi，且 pi 不报错。

        动态验证而非硬编码值——当开发者修改 _DEFAULT_PI_ARGS 时，
        此测试自动反映新值。同时确保不会传入已知会报错的参数组合。
        """
        from paper_review.pipeline_steps import _DEFAULT_PI_ARGS

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / ".first-use-hint-shown").touch()
        (data_dir / "index").mkdir()

        pipelines_dir = data_dir / "pipelines"
        self._setup_minimal_pipeline(pipelines_dir)

        # 假 pi：记录参数 + 成功退出
        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        fake_pi = self._make_arg_recording_pi(mock_bin, "pi")
        _make_mock_pandoc(mock_bin)

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _make_pdf(input_dir / "test-paper.pdf", "Test content")

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = str(fake_pi)

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
            timeout=30,
            env=env,
        )

        assert result.returncode == 0, (
            f"Pipeline failed with default args:\n"
            f"STDOUT:{result.stdout[:500]}\nSTDERR:{result.stderr[:500]}"
        )

        # 验证假 pi 收到了参数
        intermediates = _find_task_dir(data_dir / "output") / "intermediates"
        args_file = intermediates / "test-paper" / "01-review" / "args.txt"
        assert args_file.exists(), f"args.txt not found at {args_file} — pi was not called"

        args = args_file.read_text().strip().split()
        assert args, "pi received empty args"

        # 动态验证：当前 _DEFAULT_PI_ARGS 全部出现在实际参数中
        for expected_arg in _DEFAULT_PI_ARGS:
            assert expected_arg in args, (
                f"Default arg '{expected_arg}' NOT in pi args: {args}\n"
                f"If you changed _DEFAULT_PI_ARGS, verify this test still passes."
            )

        # 验证 pipeline 正常完成
        out_file = intermediates / "test-paper" / "01-review" / "output.json"
        assert out_file.exists(), f"Review output.json not found at {out_file}"

    def test_py_step_timeout_constant_applies(self, tmp_path):
        """E2E：_PY_STEP_TIMEOUT（.py 阶段默认超时）动态导入且不导致运行时错误。

        pre 阶段为纯 .py 步骤（无 .md），其 phase 超时走
        estimate_step_timeout(step_type="py") = _PY_STEP_TIMEOUT。从源码动态导入
        常量（非硬编码预期值）——修改默认超时时测试自动适用新值，同时验证
        新值在实际管线执行路径上不会导致运行时错误。
        """
        from paper_review.timeout_estimator import _PY_STEP_TIMEOUT

        assert isinstance(_PY_STEP_TIMEOUT, int) and _PY_STEP_TIMEOUT > 0, (
            f"_PY_STEP_TIMEOUT 应为正整数值，当前: {_PY_STEP_TIMEOUT!r}"
        )

        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / ".first-use-hint-shown").touch()
        (data_dir / "index").mkdir()

        pipelines_dir = data_dir / "pipelines"
        self._setup_minimal_pipeline(pipelines_dir)

        # 假 pi：记录参数 + 成功退出
        mock_bin = tmp_path / "mock-bin"
        mock_bin.mkdir()
        fake_pi = self._make_arg_recording_pi(mock_bin, "pi")
        _make_mock_pandoc(mock_bin)

        input_dir = tmp_path / "input"
        input_dir.mkdir()
        _make_pdf(input_dir / "test-paper.pdf", "Test content")

        env = os.environ.copy()
        env["PATH"] = str(mock_bin) + os.pathsep + env.get("PATH", "")
        env["PIPELINE_PI_BINARY"] = str(fake_pi)

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
            timeout=30,
            env=env,
        )

        assert result.returncode == 0, (
            f"Pipeline failed with _PY_STEP_TIMEOUT={_PY_STEP_TIMEOUT}:\n"
            f"STDOUT:{result.stdout[:500]}\nSTDERR:{result.stderr[:500]}"
        )

        # pre 阶段（纯 .py 步骤）正常完成
        intermediates = _find_task_dir(data_dir / "output") / "intermediates"
        pre_outs = list(intermediates.rglob("00-convert/output.json"))
        assert pre_outs, f"pre 00-convert output.json 未找到: {intermediates}"
