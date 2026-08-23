#!/usr/bin/env python3
"""Generate tiny ONNX model + tokenizer fixtures for integration tests.

CI 上不下载真实模型（embedding ~25MB / reranker ~570MB），而是用这里生成的
「极小确定性模型」覆盖 onnxruntime + tokenizers + numpy 后处理的**真实集成链路**
（session 加载、tokenizer 解析、mean-pooling / sigmoid、L2 归一化），不验证
模型语义质量（语义由真实模型在本地/单独 job 覆盖）。

只需 ``onnx`` + ``tokenizers``（**不需要 torch**）。生成产物提交进仓库：

    tests/fixtures/tiny-embedding/{model.onnx, tokenizer.json, config.json}
    tests/fixtures/tiny-reranker/{model.onnx, tokenizer.json, config.json}

模型语义：
  - embedding：Gather(确定性 embedding 表, input_ids) → [batch, seq, 4]，
    行向量非共线，不同 token 序列 mean-pool 后方向不同。
  - reranker：input_ids → Cast→float → Mul(mask) → ReduceMean(seq) → [batch, 1]
    （单 logit，_parse_logits 走 sigmoid 分支）。

Usage::

    uv pip install onnx   # 仅生成时需要；测试只依赖 onnxruntime
    python scripts/export_tiny_models.py
"""

from __future__ import annotations

from pathlib import Path

import onnx
from onnx import TensorProto, helper

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"

# 覆盖测试用例中出现的词，避免 [UNK] 塌缩导致「不同文本 → 相同向量」。
_VOCAB_WORDS = (
    "deep learning is a branch of machine natural language processing "
    "transformer changed the nlp field weather nice today suitable for walking "
    "apple contains rich vitamin c credit assessment method risk control using "
    "graph neural networks social network analysis distributed system scheduling "
    "algorithms database query optimization techniques security vulnerability "
    "detection methods text one two three four five six seven eight nine ten"
).split()

_SPECIAL = ["[UNK]", "[PAD]", "[CLS]", "[SEP]"]
_UNK = "[UNK]"


def _build_vocab() -> dict[str, int]:
    vocab: dict[str, int] = {}
    for tok in _SPECIAL:
        vocab[tok] = len(vocab)
    for w in _VOCAB_WORDS:
        if w not in vocab:
            vocab[w] = len(vocab)
    return vocab


def _make_tokenizer(path: Path) -> None:
    from tokenizers import Tokenizer, models, normalizers, pre_tokenizers

    tok = Tokenizer(models.WordLevel(_build_vocab(), unk_token=_UNK))
    tok.normalizer = normalizers.BertNormalizer(lowercase=True)
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    tok.save(str(path))


def _make_embedding_onnx(path: Path) -> None:
    """input_ids + attention_mask → last_hidden_state [batch, seq, 4]。

    Gather 从确定性 embedding 表查行向量（行间非共线），保证不同 token 序列
    经 mean-pooling 后方向不同——避免「Concat 相同标量 → L2 归一化后方向恒等
    （cosine=1.0）」的退化。
    """
    import numpy as np

    dim = 4
    vocab_size = len(_build_vocab())
    rng = np.random.default_rng(42)
    weights = rng.standard_normal((vocab_size, dim)).astype(np.float32)

    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "seq"]
    )
    last_hidden = helper.make_tensor_value_info(
        "last_hidden_state", TensorProto.FLOAT, ["batch", "seq", dim]
    )

    const = helper.make_node(
        "Constant",
        [],
        ["embedding_weight"],
        value=helper.make_tensor(
            "embedding_weight", TensorProto.FLOAT, [vocab_size, dim], weights.flatten().tolist()
        ),
    )
    gather = helper.make_node("Gather", ["embedding_weight", "input_ids"], ["gathered"], axis=0)
    cast_mask = helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT)
    unsq_mask = helper.make_node("Unsqueeze", ["mask_f"], ["mask_3d"], axes=[2])
    mul = helper.make_node("Mul", ["gathered", "mask_3d"], ["last_hidden_state"])

    graph = helper.make_graph(
        [const, gather, cast_mask, unsq_mask, mul],
        "tiny_embedding",
        [input_ids, attention_mask],
        [last_hidden],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    onnx.save(model, str(path))


def _make_reranker_onnx(path: Path) -> None:
    """input_ids + attention_mask → logits [batch, 1]（单 logit → sigmoid）。"""
    input_ids = helper.make_tensor_value_info("input_ids", TensorProto.INT64, ["batch", "seq"])
    attention_mask = helper.make_tensor_value_info(
        "attention_mask", TensorProto.INT64, ["batch", "seq"]
    )
    logits = helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["batch", 1])

    cast_ids = helper.make_node("Cast", ["input_ids"], ["ids_f"], to=TensorProto.FLOAT)
    cast_mask = helper.make_node("Cast", ["attention_mask"], ["mask_f"], to=TensorProto.FLOAT)
    masked = helper.make_node("Mul", ["ids_f", "mask_f"], ["masked"])
    reduce = helper.make_node("ReduceMean", ["masked"], ["logits"], axes=[1], keepdims=1)

    graph = helper.make_graph(
        [cast_ids, cast_mask, masked, reduce],
        "tiny_reranker",
        [input_ids, attention_mask],
        [logits],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 11)])
    model.ir_version = 8
    onnx.save(model, str(path))


def main() -> None:
    import json

    for name, maker in (
        ("tiny-embedding", _make_embedding_onnx),
        ("tiny-reranker", _make_reranker_onnx),
    ):
        out = FIXTURES_DIR / name
        out.mkdir(parents=True, exist_ok=True)
        maker(out / "model.onnx")
        _make_tokenizer(out / "tokenizer.json")
        arch = "BertModel" if name == "tiny-embedding" else "BertForSequenceClassification"
        (out / "config.json").write_text(
            json.dumps({"architectures": [arch], "hidden_size": 4}), encoding="utf-8"
        )
        print(f"generated {out.relative_to(FIXTURES_DIR.parent)}")


if __name__ == "__main__":
    main()
