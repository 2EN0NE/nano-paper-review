"""
Model discovery — scan local caches for ready-to-use ONNX models.

Supports discovery inside:
- paper-review model cache (``~/.cache/paper-review/models/``)
- HuggingFace hub cache (``~/.cache/huggingface/hub/``)

Each discovered model is returned as a :class:`DiscoveredModel` with:
- display name, path, type (embedding / reranker), vector dimension,
  and estimated file size.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Known-good ONNX models (3 tiers) ──
# 注意：size_hint 一律标注「INT8 单文件」体积 —— 下载只拉取单个量化版本
# （onnx-community 仓库同时含 fp32/fp16/q4 等 5-8 个变体，整仓下载会是提示的 3-5 倍）。
_KNOWN_EMBEDDING_MODELS = {
    "BAAI/bge-small-zh-v1.5": {
        "onnx_repo": "onnx-community/bge-small-zh-v1.5-ONNX",
        "dim": 512,
        "size_hint": "~25 MB（INT8 单文件）",
        "tier": "small",
        "description": "轻量中文嵌入（512维），快速省资源，Apache 2.0",
    },
    "BAAI/bge-base-zh-v1.5": {
        "onnx_repo": "onnx-community/bge-base-zh-v1.5-ONNX",
        "dim": 768,
        "size_hint": "~100 MB（INT8 单文件）",
        "tier": "balanced",
        "description": "均衡中文嵌入（768维），精度与速度兼顾，Apache 2.0",
    },
    "BAAI/bge-large-zh-v1.5": {
        "onnx_repo": "onnx-community/bge-large-zh-v1.5-ONNX",
        "dim": 1024,
        "size_hint": "~330 MB（INT8 单文件）",
        "tier": "best",
        "description": "最强中文嵌入（1024维），效果最佳，Apache 2.0",
    },
}

# 排序即推荐顺序：JINA 优先（用户偏好；INT8 单文件即可用）
_KNOWN_RERANKER_MODELS = {
    "jinaai/jina-reranker-v3": {
        "onnx_repo": "s-lorin/jina-reranker-v3-onnx",
        "size_hint": "~600 MB（INT8 单文件，0.6B）",
        "tier": "best",
        "description": "多语言 Reranker（0.6B，BEIR 领先），INT8 单文件开箱即用，CC-BY-NC-4.0（非商业）",
    },
    "Qwen/Qwen3-Reranker-0.6B": {
        "onnx_repo": "onnx-community/Qwen3-Reranker-0.6B-ONNX",
        "size_hint": "~570 MB（INT8 单文件，0.6B）",
        "tier": "balanced",
        "description": "中文 Reranker（0.6B，32K 上下文），Apache 2.0",
    },
    "BAAI/bge-reranker-v2-m3": {
        "onnx_repo": "onnx-community/bge-reranker-v2-m3-ONNX",
        "size_hint": "~570 MB（INT8 单文件）",
        "tier": "small",
        "description": "中文 Cross-Encoder（568M），Apache 2.0",
    },
}


@dataclass
class DiscoveredModel:
    """A locally-available ONNX model ready for use."""

    path: Path
    display_name: str  # e.g. "BAAI/bge-small-zh-v1.5"
    model_type: str  # "embedding" or "reranker"
    dim: int | None = None  # vector dimension (embedding only)
    size_mb: float = 0.0  # approximate total size


def _model_dir_name(hf_name: str) -> str:
    """HuggingFace model name → paper-review cache directory name."""
    return hf_name.replace("/", "--")


# ONNX 权重文件名候选（按优先级）：优先 INT8 量化，其次 fp32 等。
# onnx-community 仓库用 model_quantized.onnx（INT8）；s-lorin 等仓库直接叫 model.onnx。
RUNTIME_MODEL_FILE_NAMES = [
    "model_quantized.onnx",
    "model_int8.onnx",
    "model.onnx",
    "model_fp16.onnx",
    "model_q4.onnx",
]

# 下载时在仓库内查找权重文件的相对路径候选（按优先级）
_DOWNLOAD_WEIGHT_CANDIDATES = [
    "onnx/model_quantized.onnx",
    "onnx/model_int8.onnx",
    "model_quantized.onnx",
    "model_int8.onnx",
    "model.onnx",
    "onnx/model.onnx",
]

# 下载时一并拉取的配套文件（存在才拉）
_DOWNLOAD_AUX_FILES = [
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
]


def find_model_file(model_dir: str | Path) -> Path | None:
    """在模型目录中查找可用的 ONNX 权重文件（INT8 优先）。

    onnx-community 仓库的量化文件名为 ``model_quantized.onnx`` 而非
    ``model.onnx``；运行时统一通过此函数定位，避免加载 fp32 大文件。
    """
    d = Path(model_dir)
    for name in RUNTIME_MODEL_FILE_NAMES:
        p = d / name
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


def _validate_model_dir(path: Path, model_type: str) -> bool:
    """Check that *path* contains all required files for *model_type*."""
    for fname in ("tokenizer.json", "config.json"):
        if not (path / fname).exists():
            return False
    if find_model_file(path) is None:
        return False
    return True


def _infer_model_type(config_path: Path) -> str | None:
    """Infer model type from config.json → 'embedding' or 'reranker'."""
    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    architectures = cfg.get("architectures", [])
    if not architectures:
        return None
    arch = architectures[0].lower()
    # Cross-encoder / reranker models
    # "ranking" 覆盖 JinaForRanking（jina-reranker-v3 基于 Qwen3，架构名无 rerank/cross）
    if (
        "cross" in arch
        or "rerank" in arch
        or "ranking" in arch
        or "forsequenceclassification" in arch
    ):
        return "reranker"
    # Embedding / encoder models — "bert" (BertModel), "roberta", etc.
    if "bert" in arch or "roberta" in arch or "encoder" in arch or "embedding" in arch:
        return "embedding"
    return None


def _infer_dim(onnx_path: Path | None) -> int | None:
    """Extract output dimension from ONNX model metadata.

    Parses the protobuf header without importing onnxruntime.
    """
    if onnx_path is None:
        return None
    try:
        import onnxruntime as ort
    except ImportError:
        logger.debug("onnxruntime not available — cannot infer embedding dimension")
        return None

    try:
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        for output in session.get_outputs():
            shape = output.shape
            # Typical embedding output: (batch, seq_len, dim) → dim is last
            if len(shape) >= 2 and shape[-1] and shape[-1] != "batch_size":
                return int(shape[-1]) if shape[-1] != "sequence_length" else None
    except Exception:
        logger.warning(
            "Failed to infer embedding dimension from ONNX model %s", onnx_path, exc_info=True
        )
    return None


def _dir_size_mb(path: Path) -> float:
    """Approximate directory size in MB (follows symlinks)."""
    total = 0
    for f in path.rglob("*"):
        if f.is_symlink():
            try:
                total += f.resolve().stat().st_size
            except OSError:
                pass
        elif f.is_file():
            total += f.stat().st_size
    return total / (1024 * 1024)


# ── Public API ──


def scan_model_cache(cache_dir: str | Path) -> list[DiscoveredModel]:
    """Scan the paper-review model cache for complete ONNX models.

    Args:
        cache_dir: Path to the model cache (e.g. ``~/.cache/paper-review/models/``).

    Returns:
        List of :class:`DiscoveredModel` each representing a ready-to-use model.
    """
    cache = Path(cache_dir)
    if not cache.is_dir():
        return []

    results: list[DiscoveredModel] = []
    for entry in sorted(cache.iterdir()):
        if not entry.is_dir():
            continue

        # Try to find config.json to infer model type
        config_path = entry / "config.json"
        if not config_path.exists():
            continue

        model_type = _infer_model_type(config_path)
        if model_type is None:
            continue

        if not _validate_model_dir(entry, model_type):
            logger.debug("Incomplete model dir skipped: %s", entry)
            continue

        # Infer embedding dimension
        dim = None
        if model_type == "embedding":
            dim = _infer_dim(find_model_file(entry))

        # Build display name — try to parse from directory name
        dir_name = entry.name
        display_name = dir_name.replace("--", "/")

        results.append(
            DiscoveredModel(
                path=entry,
                display_name=display_name,
                model_type=model_type,
                dim=dim,
                size_mb=round(_dir_size_mb(entry), 1),
            )
        )

    return results


def scan_huggingface_cache() -> list[DiscoveredModel]:
    """Scan the HuggingFace hub cache for ONNX models compatible with paper-review.

    Scans ``~/.cache/huggingface/hub/`` for snapshots that contain the
    required ONNX files (model.onnx + tokenizer.json + config.json).
    """
    hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not hf_cache.is_dir():
        return []

    results: list[DiscoveredModel] = []

    for model_dir in sorted(hf_cache.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.startswith("models--"):
            continue

        # Find the actual snapshot directory
        refs_dir = model_dir / "refs"
        if not refs_dir.is_dir():
            continue

        # Read the first ref to get the commit hash
        ref_files = list(refs_dir.iterdir())
        if not ref_files:
            continue
        try:
            commit_hash = ref_files[0].read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue

        snapshot_path = model_dir / "snapshots" / commit_hash
        if not snapshot_path.is_dir():
            continue

        # ONNX models may be in root or in onnx/ subdirectory
        for candidate in (snapshot_path, snapshot_path / "onnx"):
            if not candidate.is_dir():
                continue
            config_file = candidate / "config.json"
            if not config_file.exists():
                continue
            model_type = _infer_model_type(config_file)
            if model_type is None:
                continue
            if not _validate_model_dir(candidate, model_type):
                continue

            display_name = model_dir.name.replace("models--", "").replace("--", "/")
            dim = None
            if model_type == "embedding":
                dim = _infer_dim(find_model_file(candidate))

            results.append(
                DiscoveredModel(
                    path=candidate,
                    display_name=display_name,
                    model_type=model_type,
                    dim=dim,
                    size_mb=round(_dir_size_mb(candidate), 1),
                )
            )
            break  # Found at this candidate path, don't check the other

    return results


def get_known_download_options(model_type: str) -> list[dict]:
    """Return recommended models available for download from HuggingFace.

    Args:
        model_type: ``"embedding"`` or ``"reranker"``.

    Returns:
        List of dicts with keys: display_name, onnx_repo, size_hint, dim (optional), description.
    """
    mapping = (
        _KNOWN_EMBEDDING_MODELS
        if model_type == "embedding"
        else _KNOWN_RERANKER_MODELS
        if model_type == "reranker"
        else {}
    )
    return [
        {
            "display_name": name,
            "onnx_repo": info["onnx_repo"],
            "size_hint": info.get("size_hint", "N/A"),
            "dim": info.get("dim"),
            "tier": info.get("tier"),
            "description": info.get("description", ""),
        }
        for name, info in mapping.items()
    ]


def download_model(onnx_repo: str, target_dir: str | Path, copy_mode: bool = False) -> bool:
    """Download an ONNX model via HuggingFace Hub — 只拉单个量化版本。

    之前用 ``snapshot_download`` 会拉下整个仓库（onnx-community 仓库含
    fp32/fp16/int8/q4/q4f16 等 5-8 个变体，体积是提示的 3-5 倍）。现在改为：

    1. 列仓库文件，按 ``_DOWNLOAD_WEIGHT_CANDIDATES`` 优先级挑选一个权重
       （优先 INT8 的 ``model_quantized.onnx``）；
    2. 逐个 ``hf_hub_download`` 下载该权重（含可能的 ``.onnx_data`` 外部数据）
       与 tokenizer/config 等配套文件，保留原文件名；
    3. 拷贝或软链到 *target_dir*（copy_mode=True 用于离线打包）。

    Args:
        onnx_repo: HuggingFace repo name (e.g. ``onnx-community/bge-small-zh-v1.5-ONNX``).
        target_dir: Local directory for model files.
        copy_mode: If True, copy files instead of symlinking (for offline packaging).

    Returns:
        True if a usable ONNX weight file exists at *target_dir* after download.
    """
    import shutil

    target = Path(target_dir)
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        logger.error("huggingface-hub not installed — cannot download model")
        return False

    # 幂等：目标目录已有完整模型（权重 + tokenizer + config）则跳过；
    # 仅有部分文件（上次下载中断）不短路——_place 会跳过已存在文件，只补缺。
    _target_onnx = find_model_file(target)
    if (
        _target_onnx is not None
        and (target / "tokenizer.json").exists()
        and (target / "config.json").exists()
    ):
        logger.info("ONNX model already exists at %s — skipping download", target)
        return True

    # 列出仓库文件（不下载），挑选单个权重 + 配套文件
    try:
        repo_files = set(HfApi().list_repo_files(repo_id=onnx_repo))
    except Exception as e:
        logger.error("Failed to list repo files for %s: %s", onnx_repo, e)
        return False

    weight = next((f for f in _DOWNLOAD_WEIGHT_CANDIDATES if f in repo_files), None)
    if weight is None:
        logger.error(
            "No recognized ONNX weight file in repo %s (files: %s)", onnx_repo, sorted(repo_files)
        )
        return False

    target.mkdir(parents=True, exist_ok=True)

    def _place(local_path: str | Path, dest_name: str) -> None:
        dest = target / dest_name
        if dest.exists():
            return
        src = Path(local_path)
        if copy_mode:
            shutil.copy2(src, dest)
        else:
            dest.symlink_to(src)

    try:
        # 权重本体（保留原文件名，外部数据引用不破坏）
        _place(hf_hub_download(onnx_repo, weight), Path(weight).name)
        # 外部数据文件（onnx 模型同名的 .onnx_data）
        data_file = f"{weight}_data"
        if data_file in repo_files:
            _place(hf_hub_download(onnx_repo, data_file), Path(data_file).name)
        # tokenizer / config 等配套文件（存在才拉）
        for aux in _DOWNLOAD_AUX_FILES:
            if aux in repo_files:
                _place(hf_hub_download(onnx_repo, aux), aux)
    except Exception as e:
        logger.error("Download failed for %s: %s", onnx_repo, e)
        return False

    return find_model_file(target) is not None


# ============================================================================
# 模型选择 → 写入 config.yaml（让选中的模型真正生效）
# ============================================================================


def _config_candidates(data_dir: str | None = None) -> list[Path]:
    """可能写入的 config.yaml 路径（高→低优先级）。

    data_dir 显式指定时只考虑该目录（安装/测试场景）；否则按
    data_dir 解析链 + 项目根目录兜底。
    """
    from paper_review.config import resolve_data_dir

    if data_dir:
        return [Path(data_dir) / "config.yaml"]

    dd = resolve_data_dir()
    candidates = [dd / "config.yaml"]
    cwd_dot = Path.cwd() / ".paper-review"
    if cwd_dot != dd:
        candidates.append(cwd_dot / "config.yaml")
    candidates.append(Path.cwd() / "config.yaml")
    return candidates


def update_config_models(
    embedding_model: str | None = None,
    reranker_model: str | None = None,
    vector_dim: int | None = None,
    data_dir: str | None = None,
) -> Path | None:
    """把选中的模型名写入 config.yaml（保留文件其余内容与注释）。

    逐行定位 ``embedding_model:`` / ``reranker_model:`` / ``vector_dim:``
    （含被注释的行），原位替换为生效值；不存在则追加到文件末尾。
    这是「下载了模型却没接上运行时」问题的关键一环：之前 config 命令只下载
    不写配置，运行时仍按默认模型名找目录 → 模型静默失效。

    Returns:
        写入的 config.yaml 路径；未写入返回 None。
    """
    updates = {
        k: v
        for k, v in {
            "embedding_model": embedding_model,
            "reranker_model": reranker_model,
            "vector_dim": vector_dim,
        }.items()
        if v is not None
    }
    if not updates:
        return None

    target: Path | None = None
    for cand in _config_candidates(data_dir):
        if cand.exists():
            target = cand
            break
    if target is None:
        # 都不存在：写入默认数据目录；data_dir 未显式指定时同时写项目级
        # （init 默认选项目级，无论用户选哪一级，模型配置都不会被模板默认值顶掉）
        from paper_review.config import resolve_data_dir

        targets = [resolve_data_dir(data_dir) / "config.yaml"]
        if data_dir is None:
            cwd_dot = Path.cwd() / ".paper-review" / "config.yaml"
            if cwd_dot != targets[0]:
                targets.append(cwd_dot)
        for t in targets:
            _write_model_config_lines(t, updates)
        logger.info("Updated config models in %s: %s", targets, updates)
        return targets[0]

    _write_model_config_lines(target, updates)
    logger.info("Updated config models in %s: %s", target, updates)
    return target


def _write_model_config_lines(target: Path, updates: dict) -> None:
    """逐行写模型键到 config.yaml（保留注释），找不到的键追加到末尾。"""
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = target.read_text(encoding="utf-8").splitlines(keepends=True) if target.exists() else []

    remaining = dict(updates)
    # 第一遍：只替换未注释的生效行。若先替换注释行而保留后面的生效行，
    # 会产生两个同名键，而 YAML 解析（last-wins）会静默保留旧值。
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        for key in list(remaining):
            if stripped.startswith(f"{key}:"):
                indent = line[: len(line) - len(stripped)]
                lines[i] = f"{indent}{key}: {remaining.pop(key)}\n"
                break
    # 第二遍：无生效行时，解除注释行并替换（仅当该 key 仍剩余）
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        for key in list(remaining):
            if stripped.startswith(f"# {key}:"):
                indent = line[: len(line) - len(stripped)]
                lines[i] = f"{indent}{key}: {remaining.pop(key)}\n"
                break

    for key, value in remaining.items():
        lines.append(f"\n{key}: {value}\n")

    target.write_text("".join(lines), encoding="utf-8")
