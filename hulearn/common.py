import collections


def flatten(nested_iterable):
    """
    Helper function, returns an iterator of flattened values from an arbitrarily nested iterable

    Usage:

    ```
    list(flatten([['test1', 'test2'], ['a', 'b', ['c', 'd']]]))
    # ['test1', 'test2', 'a', 'b', 'c', 'd']
    list(flatten(['test1', ['test2']]))
    # ['test1', 'test2']
    ```
    """
    for el in nested_iterable:
        if isinstance(el, collections.abc.Iterable) and not isinstance(
            el, (str, bytes)
        ):
            yield from flatten(el)
        else:
            yield el
