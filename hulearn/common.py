import collections


def flatten(nested_iterable):
    """
    Helper function, returns an iterator of flattened values from an arbitrarily
    nested iterable.

    Usage:

    ```python
    from hulearn.common import flatten

    res1 = list(flatten([['test1', 'test2'], ['a', 'b', ['c', 'd']]]))
    res2 = list(flatten(['test1', ['test2']]))
    assert res1 == ['test1', 'test2', 'a', 'b', 'c', 'd']
    assert res2 == ['test1', 'test2']
    ```
    """
    for el in nested_iterable:
        if isinstance(el, collections.abc.Iterable) and not isinstance(
            el, (str, bytes)
        ):
            yield from flatten(el)
        else:
            yield el


def df_to_dictlist(dataf):
    """
    Helper function, takes a dataframe and turns it into a list of
    dictionaries. This might make it easier to write if else chains
    in `FunctionClassifier`.

    Usage:

    ```python
    import pandas as pd
    from hulearn.common import df_to_dictlist

    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    res = df_to_dictlist(df)
    assert res == [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
    ```
    """
    data = dataf.iterrows()
    return [dict(d) for i, d in data]
