from hulearn.regression import FunctionRegressor
from hulearn.classification import FunctionClassifier
from hulearn.datasets import load_titanic

import pytest


def handle_docstring(doc):
    """
    This function will read through the docstring and grab
    the first python code block. It will try to execute it.
    If it fails, the calling test should raise a flag.
    """
    if not doc:
        return
    start = doc.find("```python\n")
    end = doc.find("```\n")
    if start != -1:
        if end != -1:
            code_part = doc[(start + 10) : end]
            code = "\n".join([c[4:] for c in code_part.split("\n")])
            print(code)
            exec(code)


@pytest.mark.parametrize("m", [load_titanic])
def test_mappers_docstrings(m):
    """
    Take the docstring of every method on the `Clumper` class.
    The test passes if the usage examples causes no errors.
    """
    handle_docstring(m.__doc__)
