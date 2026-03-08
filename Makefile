.PHONY: docs build

lint:
	.venv/bin/ruff check hulearn tests
	.venv/bin/ruff format --check hulearn tests

format:
	.venv/bin/ruff check --fix hulearn tests
	.venv/bin/ruff format hulearn tests

install:
	uv venv
	uv pip install -e ".[dev]"

test:
	.venv/bin/pytest --disable-warnings --cov=hulearn tests

clean:
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf hulearn.egg-info
	rm -rf .ipynb_checkpoints
	rm -rf .coverage*
	rm -rf tests/.ipynb_checkpoints

check: lint test clean

pypi: clean
	uv build
	uv publish
