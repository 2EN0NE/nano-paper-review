# paper-review 能力封装为 Agent Skills 体系

将 paper-review 的 CLI 能力封装为一系列 Agent Skills，让用户在 agent 里以自然语言
驱动论文评审/检索，而非记忆命令行。封装形态是**纯指令文档**：skill 教 agent
「何时、如何、为什么」调用 `paper-review` CLI，不再包一层脚本重复 CLI 已实现的能力。

## Status

accepted

## Context

paper-review 已有 9 个功能完整的 CLI 命令（`init` / `config` / `review` / `index` /
`search` / `status` / `tags` / `serve` / `agent-status`）。目标是让「安装了项目的用户」
和「构建/维护项目本身的人」都能在 agent 里以自然语言驱动这些能力。参考 ask-matt
技能体系的结构：一个引导技能（router）+ 分类分层的具体技能。

## Decision

1. **纯指令形态**：每个 skill 的 `SKILL.md` 是操作指南，指导 agent 调用 CLI 并验证
   产物；确定性工作留在 CLI，skill 不重复实现（单一事实来源）。唯一例外是安装脚本
   本身——它是部署 skills 的运维动作，独立于 skill 体系。

2. **受众二分**：`skills/builder/`（给构建/维护项目的人）+ `skills/user/`（给安装
   项目的用户）。router（`skills/paper-review/`）作为共享入口放根，两类安装都含。

3. **命名**：统一 `paper-review-` 前缀（router 例外，直接用 `paper-review`），避免
   与 `search`/`review`/`test` 等裸词 skill 撞名（pi 对同名 skill 保留先发现者并告警）。

4. **粒度**：1 router + 5 user + 3 builder = 9 个 skill。
   - user：`setup`（init+config）、`index`（建库）、`search`（search+status+tags+serve）、
     `review`（核心，含 fix-warn/resume/agent-status）、`pipeline`（自定义管线）
   - builder：`testing`（三级测试体系）、`retrieval-dev`（检索子系统开发）、
     `deploy`（离线部署 + ONNX 导出）

5. **全部 model-invoked**：9 个 skill 均不设 `disable-model-invocation`（默认 agent
   自动可见）。因为目标用户不知道这套 skill 体系存在，必须靠 agent 按 description
   匹配自动触发；router 也保持自动可见以便在需求模糊时被拉起做推荐。

6. **`pipeline` 横跨 builder/user**：同一个「写管线」job，落点不同——user 写自己的
   管线到 `{data_dir}/pipelines/`（Pipelines Directory），builder 改标准模板到
   `src/paper_review/templates/`（Scaffold Template，进版本管理）。skill 归入 `user/`
   （保证外部用户可自定义评审维度），`SKILL.md` 内分两个 branch，builder 不复制。

7. **安装脚本**：`scripts/install-skills.sh`，默认拷贝（可移植，离线 tarball 软链会断）
   到项目级 `.agents/skills/`；`--global` 装到 `~/.agents/skills/`，`--all` 额外装
   builder，`--link` 用软链（开发迭代）。`.agents/` 加入 `.gitignore`（安装副本不入库）。

## Considered Options

- **skill 内嵌脚本封装 CLI**：放弃。CLI 已是唯一执行入口，再封装是 duplication。
- **router 设 `disable-model-invocation: true`（ask-matt 做法）**：放弃。ask-matt 服务
  「会主动用 skill 的工程师」，paper-review 用户画像相反——依赖 agent 自动路由。
- **裸词命名（`review`/`search`/`test`）**：放弃。撞名风险高，前缀换来归属清晰。
- **`pipeline` 归 builder 并在 user 复制一份**：放弃。duplication；放 user/ + 内分
  branch 更干净。
- **软链默认**：放弃。离线部署 tarball 中软链会断，拷贝更可移植。

## Consequences

- 新增 `skills/` 源目录 + `scripts/install-skills.sh`，不触碰现有 CLI/源码。
- **user skills 的进阶参考必须随 skill 分发**：`pyproject.toml` 的
  `[tool.setuptools.packages.find] where=["src"]` 只打包 `src/`，`docs/`、`CONTEXT.md`
  不进 wheel——skill 装到用户级 `~/.agents/skills/` 后 cwd 非仓库时，`docs/...` 相对
  路径不可达，且项目文档可能过时（如 `docs/PIPELINE.md` 的 .py 步骤命令行参数描述）。
  因此 user skills（`pipeline`、`search`）的进阶参考（步骤案例、API 契约）放进 skill
  自己的 `references/` 目录随 `cp -R` 递归分发；builder skills 假设在仓库内工作，
  可继续引用 `docs/`。
- `.agents/` 需加入 `.gitignore`，否则安装副本会被 git 跟踪。
