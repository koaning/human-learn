.PHONY: docs build

flake:
	flake8 hulearn
	flake8 tests
	flake8 setup.py

install:
	pip install -e ".[test]"

develop:
	pip install -e ".[dev]"
	pre-commit install
	python setup.py develop

test:
	pytest --disable-warnings --cov=hulearn
	#rm -rf .coverage*

docs:
	rm -rf doc/.ipynb_checkpoints
	sphinx-build -a -E doc docs

clean:
	rm -rf .pytest_cache
	rm -rf build
	rm -rf dist
	rm -rf docs
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
