.PHONY: test clean install fmt fmt-check test-unit test-integration test-e2e smoke setup-hooks

install:
	if command -v uv >/dev/null 2>&1; then uv pip install -e .; else pip install -e .; fi

# ─── 质量门禁 ──────────────────────────────────────────

fmt:                                           # 自动格式化全部代码
	uv run ruff format .
	uv run ruff check --fix .

fmt-check:                                     # 仅检查格式（CI / pre-push）
	uv run ruff format --check .
	uv run ruff check .

test-unit:                                     # 单元测试（对齐 CI unit-tests job）
	uv run pytest tests/ -q -m "not integration and not e2e_slow" --ignore=tests/e2e

test-integration:                              # 集成测试（对齐 CI integration-tests job）
	uv run pytest tests/ -q -m "integration"

test:                                          # 全部测试
	uv run pytest tests/ -v

test-one:                                      # 单个测试文件: make test-one t=test_store
	uv run pytest tests/$(t) -v

test-e2e:                                      # E2E smoke 测试（对齐 CI e2e-smoke job）
	uv run pytest tests/e2e/ -v -m "e2e and not e2e_slow"

smoke: test-e2e                                # 别名

# ─── Git Hooks ────────────────────────────────────────

setup-hooks:                                   # 安装 git hooks（只需执行一次）
	git config core.hooksPath .githooks
	@echo "✅ hooks 已安装: .githooks/pre-commit + .githooks/pre-push"

# ─── 业务命令 ──────────────────────────────────────────

index:
	PYTHONPATH=src python3 -m paper_review.cli index --pdf-dir $(PDF_DIR)

search:
	PYTHONPATH=src python3 -m paper_review.cli search "$(Q)"

status:
	PYTHONPATH=src python3 -m paper_review.cli status

clean:
	rm -rf data/index/*.sqlite data/index/*.index
	rm -rf .pytest_cache __pycache__ src/**/__pycache__
