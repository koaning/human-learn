import pytest
from mktestdocs import check_docstring, get_codeblock_members

from hulearn.datasets import load_titanic
from hulearn.experimental import CaseWhenRuler
from hulearn.common import flatten, df_to_dictlist

members = get_codeblock_members(CaseWhenRuler)


@pytest.mark.parametrize(
    "func", [load_titanic, flatten, df_to_dictlist], ids=lambda d: d.__name__
)
def test_docstring(func):
    check_docstring(obj=func)


@pytest.mark.parametrize("obj", members, ids=lambda d: d.__qualname__)
def test_members(obj):
    check_docstring(obj)
