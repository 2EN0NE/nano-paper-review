---
name: paper-review-testing
description: 编写和运行 paper-review 测试。当维护者想运行单元/集成/E2E 测试、遵守 E2E 红线、编写测试或进度卡 TUI 回归测试时使用。
---

# paper-review-testing

测试体系分三级，本地命令与 CI / Makefile / pre-push hook 完全一致（同一 `uv run pytest` + 同一 marker）。

## 分层

| 层级 | 目录 | 定位 |
| --- | --- | --- |
| 单元测试 | `tests/test_*.py` | 纯 Python 函数/类逻辑，允许 mock 第三方（onnxruntime 等） |
| 模型集成测试 | `tests/test_model_integration.py` | 有模型真跑 ONNX，无模型 mock 兜底 |
| E2E 测试 | `tests/e2e/` | 独立空间 CLI 全链路，**禁止 mock** |

## 运行

```bash
uv run pytest tests/ -q -m "not integration and not e2e_slow"   # 单元
uv run pytest tests/ -q -m "integration"                        # 集成
uv run pytest tests/e2e/ -v -m "e2e and not e2e_slow"           # E2E
make test-unit && make test-integration && make test-e2e        # 全量（与 CI 一致）
```

前置：`pip install -e .[dev]`。

## E2E 红线（写 E2E 测试必须遵守）

1. **CLI 独立空间执行**：每个测试在 `tmp_path` 建完整 data_dir，用 `--data-dir` 隔离。
2. **禁止 mock**：不能 mock 内部函数；只允许 mock 外部工具（pandoc、pi 的 mock 二进制）。
3. **验证产物**：检查 output.json / manifest / Excel / report 的存在与内容，不只查 returncode。
4. **覆盖关键路径**：Pre→Review→Post 全链路、边界（空输入、去重、格式不支持）、特性开关（单篇 vs 多篇 Excel）。
5. **隔离性**：测试间互不依赖。
6. **默认配置值可运行性**：影响外部工具调用的模块级默认常量必须有 E2E 测试，且从源码动态 import 常量（`from module import _CONSTANT`），不硬编码预期值。

## 进度卡 TUI 回归（ANSI 残影）

进度卡在 stderr 上做 ANSI 原地重绘，**不能从字节流判断残影**，必须重放为屏幕状态再断言：

- 真实 TTY：`pty.openpty()` 让 stdout+stderr 接同一 slave fd。
- 终端模拟器重放：极简 VT100 模拟器（`tests/e2e/test_progress_tui.py` 内联 `Term` 类）把字节流重放为行缓冲。
- 屏幕级断言：完整盒子恰好 1 个、盒高固定、盒内无步骤输出混入。
- 已知坑：勿用 `startswith("┌")` 找盒子（CLI 树形 `└── POST` 会误匹配）；PTY slave 默认 ONLCR（`\n`→`\r\n`）。

## 测试数据

- `_make_pdf()` / `_make_docx()` 生成确定性最小文档（在 `tests/` 辅助模块里）。
- 不测试 FAISS / sentence-transformers 第三方行为；HTTP 路由单独集成测试。
