.PHONY: test clean install fmt fmt-check test-unit test-integration setup-hooks

install:
	pip install -e .

# ─── 质量门禁 ──────────────────────────────────────────

fmt:                                           # 自动格式化全部代码
	ruff format .
	ruff check --fix .

fmt-check:                                     # 仅检查格式（CI / pre-push）
	ruff format --check .
	ruff check .

test-unit:                                     # 单元测试（全量）
	PYTHONPATH=src python3 -m pytest tests/ -q -m "not integration"

test-integration:                              # 集成测试（全量）
	PYTHONPATH=src python3 -m pytest tests/ -q -m "integration"

test:                                          # 全部测试
	PYTHONPATH=src python3 -m pytest tests/ -v

test-one:                                      # 单个测试文件: make test-one t=test_store
	PYTHONPATH=src python3 -m pytest tests/$(t) -v

# ─── Git Hooks ────────────────────────────────────────

setup-hooks:                                   # 安装 git hooks（只需执行一次）
	git config core.hooksPath .githooks
	@echo "✅ hooks 已安装: .githooks/pre-commit + .githooks/pre-push"

# ─── 业务命令 ──────────────────────────────────────────

prototype:
	python3 -m prototype.tui

index:
	PYTHONPATH=src python3 -m paper_rag.cli index --pdf-dir $(PDF_DIR)

search:
	PYTHONPATH=src python3 -m paper_rag.cli search "$(Q)"

status:
	PYTHONPATH=src python3 -m paper_rag.cli status

clean:
	rm -rf data/index/*.sqlite data/index/*.index
	rm -rf .pytest_cache __pycache__ src/**/__pycache__
