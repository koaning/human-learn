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
	pytest --disable-warnings --cov=hulearn

clean:
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf scikit_lego.egg-info
	rm -rf .ipynb_checkpoints
	rm -rf .coverage*

black:
	black --check .

check: black flake test clean

pypi: clean
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*

build:
	npm run build
	cp -r public/* hulearn/static
