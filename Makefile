.PHONY: docs build

flake:
	flake8 hulearn tests setup.py

install:
	pip install -e ".[test]"

develop:
	pip install -e ".[dev]"
	pre-commit install
	python setup.py develop

test:
	pytest --nbval-lax --disable-warnings --cov=hulearn tests

clean:
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf hulearn.egg-info
	rm -rf .ipynb_checkpoints
	rm -rf .coverage*
	rm -rf tests/.ipynb_checkpoints

black:
	black --check .

test-notebooks:
	pytest --nbval-lax docs/guide/notebooks/*.ipynb

check: black flake test clean test-notebooks

pypi: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*

build:
	npm run build
	cp -r public/* hulearn/static
