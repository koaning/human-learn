import os
from setuptools import setup, find_packages

import hulearn

base_packages = [
    "scikit-learn>=0.23.2",
    "pandas>=0.23.4",
    "clumper>=0.2.5",
    "Shapely>=1.7.1",
    "bokeh>=2.2.1",
]

docs_packages = [
    "mkdocs==1.1",
    "mkdocs-material==4.6.3",
    "mkdocstrings==0.8.0",
]

test_packages = [
    "flake8>=3.6.0",
    "nbval>=0.9.1",
    "pytest>=4.0.2",
    "pytest-xdist>=1.32.0",
    "black>=19.3b0",
    "pytest-cov>=2.6.1",
    "pytest-mock>=1.6.3",
    "pre-commit>=1.18.3",
    "nbval>=0.9.6",
    "scikit-lego>=0.6.0",
    "matplotlib>=3.0.2",
]
util_packages = [
    "jupyter>=1.0.0",
    "jupyterlab>=0.35.4",
]
dev_packages = docs_packages + test_packages + util_packages


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="human-learn",
    version=hulearn.__version__,
    description="natural intelligence benchmarks for scikit-learn",
    author="Vincent D. Warmerdam",
    packages=find_packages(exclude=["notebooks"]),
    package_data={"hulearn": ["data/*.zip", "images/*.png"]},
    install_requires=base_packages,
    extras_require={"docs": docs_packages, "dev": dev_packages, "test": test_packages},
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
