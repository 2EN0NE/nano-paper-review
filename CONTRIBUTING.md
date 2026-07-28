# 贡献指南

## 本地开发

### 环境准备

```bash
# 创建虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 安装开发依赖（含 pytest、ruff 等）
pip install -e .[dev]

# 安装 git hooks（只需一次）
make setup-hooks
```

> `[dev]` extras 定义在 `pyproject.toml` 的 `[project.optional-dependencies] dev` 中，
> 包含 `pytest`、`ruff`、`huggingface-hub` 等开发时所需的依赖。
> CI 中也使用 `pip install -e .[dev]` 方式安装。
> 如果不需要 dev 依赖，`pip install -e .` 仅安装运行时依赖。

### 运行测试

```bash
# 全部测试
make test             # PYTHONPATH=src python -m pytest tests/ -v

# 分层运行
make test-unit        # 单元测试（不含集成标注）
make test-integration # 集成测试（含集成标注）
make test-e2e         # E2E 测试（需先 pip install -e .）

# 单个文件
make test-one t=test_store
```

### 代码质量

```bash
make fmt       # ruff format + fix
make fmt-check # 仅检查（同 CI）
```

## CI 与分支保护

本项目使用 GitHub Actions 作为 CI。所有 PR 必须通过以下检查：

| Job | 检查内容 |
|---|---|
| quality | ruff format + lint |
| unit-tests | 单元测试（Python 3.10 / 3.11）|
| integration-tests | 集成测试 |
| e2e-smoke | CLI 冒烟 + E2E |

### 设置分支保护（仓库管理员）

CI 的门禁价值依赖于 **main 分支已启用分支保护**。请确保：

1. 前往 GitHub → Settings → Branches → Add rule
2. **Branch name pattern**: `main`
3. 勾选以下所有项：
   - ☑ **Require a pull request before merging**
   - ☑ **Require status checks to pass before merging**
   - ☑ 在 Status checks 中搜索并勾选：`quality`、`unit-tests`、`integration-tests`、`e2e-smoke`
   - ☑ **Require branches to be up to date**
   - ☑ **Do not allow bypassing the above settings**
4. 可选：勾选 **Include administrators**
5. 保存

### 绕过 hooks

```bash
git commit --no-verify   # 跳过 pre-commit（不推荐）
git push --no-verify     # 跳过 pre-push（不推荐）
```

滥用 `--no-verify` 是门禁失效的信号——如果 hooks 太慢请提 Issue。
