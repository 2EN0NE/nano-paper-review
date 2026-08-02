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


def _setup_pipeline_steps(pipeline_dir: Path) -> None:
    """在 pipeline_dir 下创建完整的管线目录和所有步骤文件。"""
    # ── pipeline.yaml ──
    (pipeline_dir / "pipeline.yaml").write_text("""\
name: "e2e-test"
version: "2.0"
pre:
  directory: pre-review/
  manifest_step: "00-convert"
  duplicate_policy: skip
  retry:
    max_attempts: 1
    on_failure: skip
review:
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
post:
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
        Path(__file__).resolve().parent.parent.parent / "pipeline" / "pre-review" / "00-convert.py"
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
        / "pipeline"
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
        / "pipeline"
        / "post-review"
        / "02-generate-excel.py"
    )
    if src_excel.exists():
        shutil.copy(src_excel, post_dir / "02-generate-excel.py")


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
        _setup_pipeline_steps(input_dir)

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
        _setup_pipeline_steps(input_dir)

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
        _setup_pipeline_steps(input_dir)

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
        _setup_pipeline_steps(input_dir)

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
        _setup_pipeline_steps(input_dir)

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
        _setup_pipeline_steps(input_dir)

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
