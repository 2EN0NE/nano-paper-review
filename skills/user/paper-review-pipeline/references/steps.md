# 管线步骤编写参考（案例 + 调试）

本文件随 `paper-review-pipeline` skill 分发，写自定义步骤时的实操参考。
最完整的范例永远是 `init` 生成的默认步骤——`{data_dir}/pipelines/standard/` 下的
`.py` / `.md` 文件就是可运行的范例，改它们之前先读一遍。

## .md 步骤（Agent 步骤）

框架会为 `.md` 步骤自动拼接 Agent Prefix（前序步骤汇总 + 输出约束），
所以你写的 `.md` 内容**只需包含评审规则**，不需要写 output.json 格式（框架强制校验）。

完整示例（评审维度打分）：

```markdown
## 待审论文
{subject.text}

## 历史参考（已审论文）
{intermediates.05-batch-search.data.history}

## 本批次参考（同批待审论文）
{intermediates.05-batch-search.data.pending}

请按以下维度评审，每个维度 1-5 分并引用原文证据：
1. 创新性
2. 方法合理性
3. 实验充分性

> 若预检索为空，不要臆造参考来源，基于论文自身内容评分并注明「无相似论文可比对」。
```

## .py 步骤（脚本步骤）

执行方式：`runpy.run_path()` 在**主进程内**执行，通过环境变量接收上下文。
约定：

| 环境变量 | 含义 |
| --- | --- |
| `PIPELINE_STEP_DIR` | 本步骤输出目录（写 `output.json` 到这里） |
| `PIPELINE_OUTPUT_DIR` | output 根目录 |
| `PIPELINE_INTERMEDIATES` | intermediates 根目录 |
| `PIPELINE_SUBJECT` | 当前 Subject（Review 模式）；Pre/Post 为 `_batch_` |

最小骨架：

```python
import json
import os
from pathlib import Path


def main():
    step_dir = os.environ["PIPELINE_STEP_DIR"]
    subject = os.environ.get("PIPELINE_SUBJECT", "_batch_")

    # ... 你的逻辑 ...

    result = {
        "step": "NN-my-step",
        "status": "ok",  # ok | error | skipped
        "error": None,
        "data": {  # 供后续步骤经 {intermediates.NN-my-step.data.X} 读取
            "your_key": "your_value",
        },
    }
    Path(step_dir).mkdir(parents=True, exist_ok=True)
    (Path(step_dir) / "output.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
```

要点：

- **必须写 `output.json`**（框架强制校验，status 为 `ok|error|skipped`）。
- `print()` 写 stdout（TTY 进度卡期间被静音）；要进文件日志用
  `logging.getLogger("paper_review.xxx")`（`paper_review.*` 前缀才挂 FileHandler）。
- 批量步骤（Pre/Post）建议逐篇失败隔离（try/except 单篇不中断批次），并复用
  `paper_review.progress.report_batch_progress` / `load_existing_step_products` 上报进度与续做。

## 模板变量

`.md` 步骤里可用，提交 Agent 前由框架替换（未识别的变量原样保留）：

| 类别 | 变量 | 值 |
| --- | --- | --- |
| Subject | `{subject.name}` | 文件名（无扩展名） |
| Subject | `{subject.path}` | Subject 绝对路径 |
| Subject | `{subject.text}` | PDF 提取全文 |
| Subject | `{subject.meta}` | 元数据 JSON 字符串 |
| Path | `{output_dir}` | pipeline 的 output_dir |
| Path | `{intermediates_dir}` | intermediates 根目录 |
| Path | `{step_dir}` | 当前步骤 intermediates 子目录 |
| Path | `{reports_dir}` | 最终报告目录 |
| 前序步骤 | `{intermediates.05-batch-search.output}` | 该步骤整个 output.json |
| 前序步骤 | `{intermediates.05-batch-search.data.KEY}` | output.json 的 data.KEY 字段 |
| 前序步骤 | `{intermediates.05-batch-search.status}` | output.json 的 status 字段 |

## output.json 最小 Schema

```json
{
  "step": "NN-my-step",
  "status": "ok",
  "error": null,
  "data": { }
}
```

三个合法状态：`ok`（成功，data 有实质产出）、`error`（失败，error 填原因）、
`skipped`（主动跳过，error 填原因）。

## 调试与验证

改完步骤后，从快到慢：

1. **单步重跑**（最快，复用已有中间产物）：

   ```bash
   paper-review review ./一篇测试.pdf --step NN-你的步骤
   ```

2. **看该步骤产物**：

   ```bash
   cat {data_dir}/output/intermediates/{subject}/NN-你的步骤/output.json
   ```

3. **看运行时日志**（`.py` 步骤失败/耗时的逐篇细节）：

   ```bash
   tail -f {data_dir}/logs/paper-review.log
   ```

4. **看 Agent 异常占比**（`.md` 步骤超时/格式错/退出码）：

   ```bash
   paper-review agent-status
   ```

5. **跑某个阶段**（隔离 Pre/Review/Post）：

   ```bash
   paper-review review ./dir/ --phase review
   ```
