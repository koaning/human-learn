import collections

from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


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
        if isinstance(el, collections.abc.Iterable) and not isinstance(el, (str, bytes)):
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


def iter_poly_data(json_desc, buffer=0.0):
    """
    Yields polygon dicts from a json_desc, applying Shapely `.buffer()` and skipping empty results.

    Arguments:
        json_desc: chart data in list-of-dicts form
        buffer: buffer to apply to each polygon (positive = grow, negative = shrink)
    """
    for chart in json_desc:
        chart_id = chart["chart_id"]
        labels = chart["polygons"].keys()
        coords = chart["polygons"].values()
        for lab, p in zip(labels, coords):
            x_lab, y_lab = p.keys()
            x_coords, y_coords = list(p.values())
            for i in range(len(x_coords)):
                poly_data = list(zip(x_coords[i], y_coords[i]))
                if len(poly_data) >= 3:
                    poly = Polygon(poly_data)
                    if buffer != 0.0:
                        poly = poly.buffer(buffer)
                    if not poly.is_empty:
                        yield {
                            "x_lab": x_lab,
                            "y_lab": y_lab,
                            "poly": poly,
                            "label": lab,
                            "chart_id": chart_id,
                        }


def count_hits(poly_data_iter, data_in, classes):
    """
    Counts polygon containments per class.

    Arguments:
        poly_data_iter: iterable of polygon dicts (from iter_poly_data)
        data_in: dict-like with column values for a single row
        classes: list of class labels
    """
    counts = {k: 0 for k in classes}
    for c in poly_data_iter:
        point = Point(data_in[c["x_lab"]], data_in[c["y_lab"]])
        if c["poly"].contains(point):
            counts[c["label"]] += 1
    return counts
