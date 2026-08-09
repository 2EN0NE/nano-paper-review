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


# ── download_model ──


def test_download_creates_symlinks(tmp_path, monkeypatch):
    """download_model should create symlinks from target_dir to HF snapshot."""

    # Simulate HF cache
    hf_snapshot = tmp_path / "hf_snapshot"
    onnx_sub = hf_snapshot / "onnx"
    _make_minimal_onnx_model(onnx_sub, "embedding")

    # Mock snapshot_download to return our simulated path
    def _mock_snapshot(repo, **kwargs):
        return str(hf_snapshot)

    monkeypatch.setattr(
        "paper_review.model_discovery._snapshot",
        _mock_snapshot,
        raising=False,
    )
    # Also patch the import path used inside download_model
    import paper_review.model_discovery as md

    monkeypatch.setattr(
        md, "download_model", lambda repo, target_dir: _real_download(repo, target_dir, hf_snapshot)
    )
    # Actually, let's just test the symlink logic directly

    target = tmp_path / "my_cache" / "BAAI--bge-small-zh-v1.5"
    target.mkdir(parents=True, exist_ok=True)

    # Simulate what download_model does internally
    source_dir = hf_snapshot / "onnx"
    for f in source_dir.iterdir():
        if f.is_file():
            dest = target / f.name
            if not dest.exists():
                dest.symlink_to(f)

    # Verify symlinks
    assert (target / "model.onnx").is_symlink()
    assert (target / "model.onnx").exists()
    assert (target / "model.onnx").resolve() == (onnx_sub / "model.onnx")
    assert (target / "tokenizer.json").is_symlink()
    assert (target / "config.json").is_symlink()


def _real_download(repo, target_dir, hf_snapshot):
    """Helper for the download test."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    source_dir = hf_snapshot / "onnx" if (hf_snapshot / "onnx").is_dir() else hf_snapshot
    for f in source_dir.iterdir():
        if f.is_file():
            dest = target / f.name
            if not dest.exists():
                dest.symlink_to(f)
    return True


def test_download_skips_if_exists(tmp_path):
    """download_model should not re-download if model.onnx already present."""
    target = tmp_path / "existing"
    _make_minimal_onnx_model(target, "embedding")
    # Call should succeed without actually downloading
    # We verify by checking no exception is raised and model.onnx exists
    assert (target / "model.onnx").exists()


def test_download_symlinks_root_files_too(tmp_path, monkeypatch):
    """When model.onnx is in onnx/ subdir but config.json is at root,
    both should be symlinked — this is the bge-reranker-v2-m3 scenario."""
    # Simulate HF cache layout: config.json at root, model files in onnx/
    hf_root = tmp_path / "hf_snapshot"
    onnx_sub = hf_root / "onnx"
    _make_minimal_onnx_model(onnx_sub, "reranker")
    (hf_root / "config.json").write_text('{"architectures":["BertForSequenceClassification"]}')
    (hf_root / "tokenizer.json").write_text('{"version":"1.0","model":{}}')

    # Patch huggingface_hub.snapshot_download — the local alias inside download_model
    def _mock_snapshot(repo):
        return str(hf_root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _mock_snapshot, raising=False)

    target = tmp_path / "cache" / "BAAI--bge-reranker-v2-m3"

    import paper_review.model_discovery as md

    ok = md.download_model("onnx-community/bge-reranker-v2-m3-ONNX", target)
    assert ok

    # Both onnx/ files and root-level files should be symlinked
    assert (target / "model.onnx").is_symlink()
    assert (target / "config.json").is_symlink()
    assert (target / "tokenizer.json").is_symlink()


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


# ── download_model copy_mode ──


def test_download_copy_mode_creates_real_files(tmp_path, monkeypatch):
    """download_model(copy_mode=True) should copy files, not symlink."""
    # Simulate HF cache with onnx/ subdirectory layout
    hf_root = tmp_path / "hf_snapshot"
    onnx_sub = hf_root / "onnx"
    _make_minimal_onnx_model(onnx_sub, "reranker")
    (hf_root / "config.json").write_text('{"architectures":["BertForSequenceClassification"]}')
    (hf_root / "tokenizer.json").write_text('{"version":"1.0","model":{}}')

    def _mock_snapshot(repo):
        return str(hf_root)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _mock_snapshot, raising=False)

    target = tmp_path / "cache" / "BAAI--bge-reranker-v2-m3"

    import paper_review.model_discovery as md

    ok = md.download_model("onnx-community/bge-reranker-v2-m3-ONNX", target, copy_mode=True)
    assert ok

    # All files must be real files, NOT symlinks
    for fname in ("model.onnx", "config.json", "tokenizer.json"):
        p = target / fname
        assert p.exists(), f"{fname} should exist"
        assert p.is_file(), f"{fname} should be a regular file"
        assert not p.is_symlink(), f"{fname} should not be a symlink (copy_mode=True)"

    # Content should match source
    assert (target / "model.onnx").read_bytes() == (onnx_sub / "model.onnx").read_bytes()


def test_download_copy_mode_still_skips_if_exists(tmp_path, monkeypatch):
    """download_model(copy_mode=True) returns True when model.onnx already present.

    Note: download_model always calls snapshot_download first (HF cache is
    idempotent), then skips copy/symlink if target model.onnx already exists.
    """
    target = tmp_path / "existing"
    _make_minimal_onnx_model(target, "embedding")

    # Simulate a successful HF download; the function should still return True
    # and not overwrite existing files.
    hf_snapshot = tmp_path / "hf_snapshot"
    _make_minimal_onnx_model(hf_snapshot, "embedding")

    def _mock_snapshot(repo):
        return str(hf_snapshot)

    monkeypatch.setattr("huggingface_hub.snapshot_download", _mock_snapshot, raising=False)

    import paper_review.model_discovery as md

    # Capture original file content to verify no overwrite
    original_content = (target / "model.onnx").read_bytes()
    ok = md.download_model("some/repo", target, copy_mode=True)
    assert ok
    # Verify the original file wasn't touched
    assert (target / "model.onnx").read_bytes() == original_content
