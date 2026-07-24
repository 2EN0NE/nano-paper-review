.PHONY: test clean install

install:
	pip install -e .

test:
	PYTHONPATH=src python3 -m pytest tests/ -v

test-one:
	PYTHONPATH=src python3 -m pytest tests/$(t) -v

# 原型
prototype:
	python3 -m prototype.tui

# 索引
index:
	PYTHONPATH=src python3 -m paper_rag.cli index --pdf-dir $(PDF_DIR)

search:
	PYTHONPATH=src python3 -m paper_rag.cli search "$(Q)"

status:
	PYTHONPATH=src python3 -m paper_rag.cli status

clean:
	rm -rf data/index/*.sqlite data/index/*.index
	rm -rf .pytest_cache __pycache__ src/**/__pycache__
