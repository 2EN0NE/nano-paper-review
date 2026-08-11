"""
Tests for model_discovery — scan, validate, download with symlinks.

Coverage:
  - _validate_model_dir: missing files → False, complete → True
  - _infer_model_type: embedding vs reranker from config.json
  - _dir_size_mb: real files and symlinks
  - scan_model_cache: finds complete models, skips incomplete
  - scan_huggingface_cache: finds models in HF snapshot structure
  - get_known_download_options: returns 3-tier lists
  - download_model: HF cache → symlink flow
"""

from __future__ import annotations

import json
from pathlib import Path

# Import under test
from paper_review.model_discovery import (
    _dir_size_mb,
    _infer_model_type,
    _model_dir_name,
    _validate_model_dir,
    find_model_file,
    get_known_download_options,
    scan_huggingface_cache,
    scan_model_cache,
)

# ── helpers ──


def _make_minimal_onnx_model(path: Path, model_type: str = "embedding") -> None:
    """Create a minimal ONNX model directory with required files."""
    path.mkdir(parents=True, exist_ok=True)

    # model.onnx — minimal valid bytes
    (path / "model.onnx").write_bytes(b"\x08\x01\x12\x05model" + b"\x00" * 100)

    # tokenizer.json
    (path / "tokenizer.json").write_text('{"version":"1.0","model":{}}')

    # config.json with architecture hint
    arch = "BertModel" if model_type == "embedding" else "BertForSequenceClassification"
    (path / "config.json").write_text(json.dumps({"architectures": [arch]}))


# ── _model_dir_name ──


def test_model_dir_name_slash():
    assert _model_dir_name("BAAI/bge-small-zh-v1.5") == "BAAI--bge-small-zh-v1.5"


def test_model_dir_name_no_slash():
    assert _model_dir_name("org/repo") == "org--repo"


# ── _validate_model_dir ──


def test_validate_complete_embedding(tmp_path):
    _make_minimal_onnx_model(tmp_path, "embedding")
    assert _validate_model_dir(tmp_path, "embedding")


def test_validate_missing_tokenizer(tmp_path):
    _make_minimal_onnx_model(tmp_path, "embedding")
    (tmp_path / "tokenizer.json").unlink()
    assert not _validate_model_dir(tmp_path, "embedding")


def test_validate_empty_onnx(tmp_path):
    _make_minimal_onnx_model(tmp_path, "embedding")
    (tmp_path / "model.onnx").write_bytes(b"")  # zero-length
    assert not _validate_model_dir(tmp_path, "embedding")


def test_validate_missing_config(tmp_path):
    _make_minimal_onnx_model(tmp_path, "embedding")
    (tmp_path / "config.json").unlink()
    assert not _validate_model_dir(tmp_path, "embedding")


# ── _infer_model_type ──


def test_infer_embedding():
    p = Path("/tmp/_test_config.json")
    p.write_text(json.dumps({"architectures": ["BertModel"]}))
    assert _infer_model_type(p) == "embedding"


def test_infer_reranker():
    p = Path("/tmp/_test_config.json")
    p.write_text(json.dumps({"architectures": ["BertForSequenceClassification"]}))
    assert _infer_model_type(p) == "reranker"


def test_infer_reranker_ranking_arch():
    """JinaForRanking（jina-reranker-v3，架构名无 rerank/cross）识别为 reranker。"""
    p = Path("/tmp/_test_config.json")
    p.write_text(json.dumps({"architectures": ["JinaForRanking"]}))
    assert _infer_model_type(p) == "reranker"


def test_infer_reranker_qwen_ranking_arch():
    """Qwen3ForRanking 同样走 "ranking" 分支。"""
    p = Path("/tmp/_test_config.json")
    p.write_text(json.dumps({"architectures": ["Qwen3ForRanking"]}))
    assert _infer_model_type(p) == "reranker"


def test_infer_unknown():
    p = Path("/tmp/_test_config.json")
    p.write_text(json.dumps({"architectures": ["UnknownModel"]}))
    assert _infer_model_type(p) is None


def test_infer_empty_architectures():
    p = Path("/tmp/_test_config.json")
    p.write_text(json.dumps({"architectures": []}))
    assert _infer_model_type(p) is None


def test_infer_broken_json():
    p = Path("/tmp/_test_config.json")
    p.write_text("not json")
    assert _infer_model_type(p) is None


# ── _dir_size_mb ──


def test_dir_size_real_files(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"x" * 1024)
    size = _dir_size_mb(tmp_path)
    assert 0.0005 < size < 0.002  # 1024 bytes ≈ 0.001 MB


def test_dir_size_symlink(tmp_path):
    real = tmp_path / "real.bin"
    real.write_bytes(b"x" * 1024 * 1024)  # 1MB
    link = tmp_path / "link.bin"
    link.symlink_to(real)
    size = _dir_size_mb(tmp_path)
    assert 1.9 < size < 2.1  # real file (1MB) + symlink dereferences to same (1MB) = ~2MB


# ── scan_model_cache ──


def test_scan_empty_cache(tmp_path):
    assert scan_model_cache(tmp_path) == []


def test_scan_finds_complete_model(tmp_path):
    _make_minimal_onnx_model(tmp_path / "BAAI--bge-small-zh-v1.5", "embedding")
    results = scan_model_cache(tmp_path)
    assert len(results) == 1
    assert results[0].display_name == "BAAI/bge-small-zh-v1.5"
    assert results[0].model_type == "embedding"


def test_scan_skips_incomplete_model(tmp_path):
    d = tmp_path / "BAAI--broken"
    d.mkdir()
    (d / "model.onnx").write_bytes(b"\x00")
    # missing tokenizer.json and config.json
    assert scan_model_cache(tmp_path) == []


def test_scan_skips_non_model_dirs(tmp_path):
    (tmp_path / "not-a-model").mkdir()
    assert scan_model_cache(tmp_path) == []


# ── scan_huggingface_cache ──


def test_scan_hf_cache_finds_model(tmp_path, monkeypatch):
    """Simulate HF hub cache structure and verify discovery."""

    # Override the home/.cache/huggingface/hub path
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    hub = tmp_path / ".cache" / "huggingface" / "hub"
    model_dir = hub / "models--onnx-community--bge-small-zh-v1.5-ONNX"
    snapshot_dir = model_dir / "snapshots" / "abc123"
    onnx_sub = snapshot_dir / "onnx"
    onnx_sub.mkdir(parents=True)
    _make_minimal_onnx_model(onnx_sub, "embedding")

    # refs/main → commit hash
    refs_dir = model_dir / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("abc123")

    results = scan_huggingface_cache()
    assert len(results) == 1
    assert results[0].display_name == "onnx-community/bge-small-zh-v1.5-ONNX"


def test_scan_hf_cache_skips_without_snapshots(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    hub = tmp_path / ".cache" / "huggingface" / "hub"
    model_dir = hub / "models--some--model"
    refs_dir = model_dir / "refs"
    refs_dir.mkdir(parents=True)
    (refs_dir / "main").write_text("deadbeef")
    # No snapshots directory
    assert scan_huggingface_cache() == []


# ── get_known_download_options ──


def test_get_embedding_options():
    opts = get_known_download_options("embedding")
    tiers = {o.get("tier") for o in opts}
    assert "small" in tiers
    assert "balanced" in tiers
    assert "best" in tiers
    assert len(opts) == 3


def test_get_reranker_options():
    opts = get_known_download_options("reranker")
    tiers = {o.get("tier") for o in opts}
    assert "small" in tiers
    assert "balanced" in tiers
    assert "best" in tiers
    assert len(opts) == 3
    # Verify the specific models
    names = {o["display_name"] for o in opts}
    assert "BAAI/bge-reranker-v2-m3" in names
    assert "jinaai/jina-reranker-v3" in names
    assert "Qwen/Qwen3-Reranker-0.6B" in names


def test_get_unknown_type():
    assert get_known_download_options("unknown") == []


# ── find_model_file ──


def test_find_model_file_prefers_int8(tmp_path):
    """存在 model_quantized.onnx（INT8）时应优先于 model.onnx（fp32）。"""
    _make_minimal_onnx_model(tmp_path, "embedding")
    (tmp_path / "model_quantized.onnx").write_bytes(b"\x08\x01INT8" + b"\x00" * 50)
    f = find_model_file(tmp_path)
    assert f is not None
    assert f.name == "model_quantized.onnx"


def test_find_model_file_falls_back_to_plain(tmp_path):
    """只有 model.onnx（如 jina-reranker-v3 仓库）时返回它。"""
    _make_minimal_onnx_model(tmp_path, "embedding")
    f = find_model_file(tmp_path)
    assert f is not None
    assert f.name == "model.onnx"


def test_find_model_file_skips_empty(tmp_path):
    """空文件不算可用权重。"""
    _make_minimal_onnx_model(tmp_path, "embedding")
    (tmp_path / "model_quantized.onnx").write_bytes(b"")
    f = find_model_file(tmp_path)
    assert f is not None
    assert f.name == "model.onnx"


# ── download_model ──


def _make_repo_files(repo_dir: Path, files: dict[str, str | bytes]) -> None:
    """构造模拟的 HF 仓库文件（relpath -> content）。"""
    for rel, content in files.items():
        p = repo_dir / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            p.write_bytes(content)
        else:
            p.write_text(content)


def _patch_hf(monkeypatch, repo_dir: Path) -> None:
    """Mock huggingface_hub：list_repo_files + hf_hub_download 指向本地模拟仓库。"""
    files = {str(p.relative_to(repo_dir)): p for p in repo_dir.rglob("*") if p.is_file()}

    class _FakeHfApi:
        def list_repo_files(self, repo_id=None, **kwargs):
            return list(files.keys())

    def _fake_hub_download(repo_id, filename, **kwargs):
        return str(files[filename])

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_hub_download)


_COMMON_AUX = {
    "config.json": json.dumps({"architectures": ["BertModel"]}),
    "tokenizer.json": '{"version":"1.0","model":{}}',
}


def test_download_only_single_quantization(tmp_path, monkeypatch):
    """关键行为：整仓含 fp32/fp16/int8 等多个变体时，只下载单个 INT8 版本。"""
    repo = tmp_path / "repo"
    _make_repo_files(
        repo,
        {
            **_COMMON_AUX,
            "onnx/model.onnx": b"\x08\x01fp32" + b"\x00" * 2000,  # 不应被下载
            "onnx/model_fp16.onnx": b"\x08\x01fp16" + b"\x00" * 1000,  # 不应被下载
            "onnx/model_quantized.onnx": b"\x08\x01int8" + b"\x00" * 100,  # 应被选中
            "onnx/model_q4.onnx": b"\x08\x01q4" + b"\x00" * 50,  # 不应被下载
        },
    )
    _patch_hf(monkeypatch, repo)

    target = tmp_path / "cache" / "BAAI--bge-small-zh-v1.5"
    import paper_review.model_discovery as md

    ok = md.download_model("onnx-community/bge-small-zh-v1.5-ONNX", target)
    assert ok

    # 只有 INT8 变体 + tokenizer/config 落地，其余量化版本一律不下载
    placed = {p.name for p in target.iterdir()}
    assert "model_quantized.onnx" in placed
    assert "model.onnx" not in placed
    assert "model_fp16.onnx" not in placed
    assert "model_q4.onnx" not in placed
    assert "tokenizer.json" in placed
    assert "config.json" in placed
    assert (target / "model_quantized.onnx").read_bytes().startswith(b"\x08\x01int8")


def test_download_onnx_subdir_layout_symlinks(tmp_path, monkeypatch):
    """onnx-community 布局：权重在 onnx/ 子目录、config/tokenizer 在根——
    目标目录应同时拿到（默认 symlink 模式）。"""
    repo = tmp_path / "repo"
    _make_repo_files(
        repo,
        {
            **_COMMON_AUX,
            "onnx/model_quantized.onnx": b"\x08\x01int8" + b"\x00" * 100,
            "onnx/model_quantized.onnx_data": b"data-bytes",
        },
    )
    _patch_hf(monkeypatch, repo)

    target = tmp_path / "cache" / "BAAI--bge-reranker-v2-m3"
    import paper_review.model_discovery as md

    ok = md.download_model("onnx-community/bge-reranker-v2-m3-ONNX", target)
    assert ok

    assert (target / "model_quantized.onnx").is_symlink()
    assert (target / "model_quantized.onnx").exists()
    # 外部数据文件保留原名（改名会破坏 onnx 内部的 location 引用）
    assert (target / "model_quantized.onnx_data").exists()
    assert (target / "config.json").is_symlink()
    assert (target / "tokenizer.json").is_symlink()


def test_download_root_model_fallback(tmp_path, monkeypatch):
    """仓库只有根级 model.onnx（s-lorin/jina-reranker-v3 布局）时也能下载。"""
    repo = tmp_path / "repo"
    _make_repo_files(
        repo,
        {**_COMMON_AUX, "model.onnx": b"\x08\x01onnx" + b"\x00" * 100},
    )
    _patch_hf(monkeypatch, repo)

    target = tmp_path / "cache" / "jinaai--jina-reranker-v3"
    import paper_review.model_discovery as md

    ok = md.download_model("s-lorin/jina-reranker-v3-onnx", target)
    assert ok
    assert (target / "model.onnx").exists()
    assert (target / "tokenizer.json").exists()


def test_download_skips_if_exists(tmp_path, monkeypatch):
    """目标目录已有可用权重时不再发起下载（不调用 HfApi）。"""
    target = tmp_path / "existing"
    _make_minimal_onnx_model(target, "embedding")
    (target / "model_quantized.onnx").write_bytes(b"\x08\x01int8" + b"\x00" * 50)

    called = []

    class _FakeHfApi:
        def list_repo_files(self, repo_id=None, **kwargs):
            called.append(True)
            return []

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)

    import paper_review.model_discovery as md

    ok = md.download_model("some/repo", target)
    assert ok
    assert not called, "已有模型时不应访问网络"


def test_download_copy_mode_creates_real_files(tmp_path, monkeypatch):
    """download_model(copy_mode=True) 应拷贝真实文件（离线打包用），而非 symlink。"""
    repo = tmp_path / "repo"
    _make_repo_files(
        repo,
        {
            **_COMMON_AUX,
            "onnx/model_quantized.onnx": b"\x08\x01int8" + b"\x00" * 100,
        },
    )
    _patch_hf(monkeypatch, repo)

    target = tmp_path / "cache" / "BAAI--bge-small-zh-v1.5"
    import paper_review.model_discovery as md

    ok = md.download_model("onnx-community/bge-small-zh-v1.5-ONNX", target, copy_mode=True)
    assert ok

    for fname in ("model_quantized.onnx", "config.json", "tokenizer.json"):
        p = target / fname
        assert p.exists(), f"{fname} should exist"
        assert p.is_file() and not p.is_symlink(), f"{fname} should be a regular file"


def test_download_copy_mode_still_skips_if_exists(tmp_path, monkeypatch):
    """目标已存在时 copy_mode 也不重复下载、不覆盖。"""
    target = tmp_path / "existing"
    _make_minimal_onnx_model(target, "embedding")
    original_content = (target / "model.onnx").read_bytes()

    class _FakeHfApi:
        def list_repo_files(self, repo_id=None, **kwargs):
            raise AssertionError("should not be called")

    monkeypatch.setattr("huggingface_hub.HfApi", _FakeHfApi)

    import paper_review.model_discovery as md

    ok = md.download_model("some/repo", target, copy_mode=True)
    assert ok
    assert (target / "model.onnx").read_bytes() == original_content


def test_download_no_recognized_weight(tmp_path, monkeypatch):
    """仓库没有任何候选权重文件时返回 False。"""
    repo = tmp_path / "repo"
    _make_repo_files(repo, {"README.md": "nothing here"})
    _patch_hf(monkeypatch, repo)

    import paper_review.model_discovery as md

    assert not md.download_model("some/repo", tmp_path / "cache" / "x")


# ── update_config_models（模型选择 → 写入 config.yaml） ──


def test_update_config_models_preserves_comments(tmp_path, monkeypatch):
    """逐行替换模型键，文件其余内容（注释）保持不变。"""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "# paper-review 配置\n"
        "chunk_size: 512\n"
        "# ── 模型 ──\n"
        'embedding_model: "BAAI/bge-small-zh-v1.5"\n'
        'reranker_model: "BAAI/bge-reranker-v2-m3"\n'
        "vector_dim: 512\n"
    )

    from paper_review.model_discovery import update_config_models

    written = update_config_models(
        embedding_model="jinaai/jina-reranker-v3",
        reranker_model="jinaai/jina-reranker-v3",
        vector_dim=768,
        data_dir=str(tmp_path),
    )
    assert written == cfg

    text = cfg.read_text()
    assert "chunk_size: 512" in text
    assert "# paper-review 配置" in text
    assert "embedding_model: jinaai/jina-reranker-v3" in text
    assert "reranker_model: jinaai/jina-reranker-v3" in text
    assert "vector_dim: 768" in text


def test_update_config_models_uncomments_and_appends(tmp_path):
    """被注释的键应被反注释；不存在的键追加到末尾。"""
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "# paper-review 配置\n# embedding_model: BAAI/bge-small-zh-v1.5\nchunk_size: 256\n"
    )

    from paper_review.model_discovery import update_config_models

    update_config_models(
        embedding_model="BAAI/bge-base-zh-v1.5",
        vector_dim=768,
        data_dir=str(tmp_path),
    )

    text = cfg.read_text()
    assert "embedding_model: BAAI/bge-base-zh-v1.5" in text
    assert "# embedding_model" not in text
    assert "vector_dim: 768" in text
    assert "chunk_size: 256" in text


def test_update_config_models_prefers_active_line_over_commented(tmp_path):
    """注释行在生效行之前时，应替换生效行而非注释行——否则产生重复键，
    YAML 解析（last-wins）会静默保留旧值，更新失效。"""
    import yaml

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        '# embedding_model: old-commented\nembedding_model: "active-value"\nchunk_size: 512\n'
    )

    from paper_review.model_discovery import update_config_models

    update_config_models(
        embedding_model="new-value",
        data_dir=str(tmp_path),
    )

    text = cfg.read_text()
    # 注释行保持注释（历史参考），生效行被更新
    assert "# embedding_model: old-commented" in text
    assert "embedding_model: new-value" in text
    # 关键：YAML 解析后生效的是新值（无重复键，last-wins 不再吞掉更新）
    loaded = yaml.safe_load(text)
    assert loaded["embedding_model"] == "new-value"
    assert loaded["chunk_size"] == 512


def test_update_config_models_creates_when_missing(tmp_path):
    """没有任何 config.yaml 时在 data_dir 下新建。"""
    from paper_review.model_discovery import update_config_models

    target = update_config_models(
        reranker_model="jinaai/jina-reranker-v3", data_dir=str(tmp_path / "dd")
    )
    assert target is not None
    assert target.exists()
    assert "reranker_model: jinaai/jina-reranker-v3" in target.read_text()


def test_scan_detects_model_with_root_config(tmp_path):
    """scan_model_cache should detect model when config.json was symlinked
    from root level of an onnx/ layout snapshot."""
    model_dir = tmp_path / "BAAI--bge-reranker-v2-m3"
    _make_minimal_onnx_model(model_dir, "reranker")
    # model.onnx and tokenizer.json are fine via _make_minimal_onnx_model
    # Remove and re-create config.json — scan_model_cache uses it first
    (model_dir / "config.json").unlink()
    (model_dir / "config.json").write_text('{"architectures":["BertForSequenceClassification"]}')

    models = scan_model_cache(tmp_path)
    assert len(models) == 1
    assert models[0].display_name == "BAAI/bge-reranker-v2-m3"
    assert models[0].model_type == "reranker"


def test_pick_or_download_skips_default_when_models_exist(monkeypatch):
    """_pick_or_download_model should default to 's' (skip) when local models exist."""

    from pathlib import Path as _Path

    from paper_review.cli import _pick_or_download_model
    from paper_review.model_discovery import DiscoveredModel

    local = [
        DiscoveredModel(
            path=_Path("/fake/cache/BAAI--bge-reranker-v2-m3"),
            display_name="BAAI/bge-reranker-v2-m3",
            model_type="reranker",
            size_mb=200.0,
        )
    ]

    prompt_calls = []

    def _fake_prompt(msg, default=None):
        prompt_calls.append((msg, default))
        return "s"

    monkeypatch.setattr("paper_review.cli.typer.prompt", _fake_prompt)
    monkeypatch.setattr("paper_review.cli.typer.echo", lambda msg, **kw: None)

    _pick_or_download_model("reranker", local, _Path("/fake/cache"))

    # Verify prompt default was "s"
    assert prompt_calls[0][1] == "s", f"Expected default='s', got {prompt_calls[0][1]}"


def test_pick_or_download_prompts_download_when_no_models(monkeypatch):
    """_pick_or_download_model should show download options when no local models."""

    from pathlib import Path as _Path

    from paper_review.cli import _pick_or_download_model

    prompt_calls = []

    def _fake_prompt(msg, default=None):
        prompt_calls.append((msg, default))
        return "s"

    monkeypatch.setattr("paper_review.cli.typer.prompt", _fake_prompt)
    monkeypatch.setattr("paper_review.cli.typer.echo", lambda msg, **kw: None)

    _pick_or_download_model("reranker", [], _Path("/fake/cache"))

    # Should have prompted with a download selection
    assert len(prompt_calls) >= 1
    # The prompt text should mention "下载" or "download"
    assert any("选择" in msg for msg, _ in prompt_calls) or any(
        "下载" in msg for msg, _ in prompt_calls
    )
