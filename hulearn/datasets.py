import os
from pkg_resources import resource_filename

import pandas as pd


def load_titanic(return_X_y: bool = False, as_frame: bool = False):
    """
    Loads in a subset of the titanic dataset. You can find the full dataset [here](https://www.kaggle.com/c/titanic/data).

    Arguments:
        return_X_y: return a tuple of (`X`, `y`) for convenience
        as_frame: return all the data as a pandas dataframe

    Usage:

    ```python
    from hulearn.datasets import load_titanic

    df = load_titanic(as_frame=True)
    X, y = load_titanic(return_X_y=True)
    ```
    """
    filepath = resource_filename("hulearn", os.path.join("data", "titanic.zip"))
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X, y = (
        df[["pclass", "name", "sex", "age", "fare", "sibsp", "parch"]].values,
        df["survived"].values,
    )
    if return_X_y:
        return X, y
    return {"data": X, "target": y}


def load_fish(return_X_y: bool = False, as_frame: bool = False):
    """
    Loads in a subset of the Fish market dataset. You can find the full dataset [here](https://www.kaggle.com/aungpyaeap/fish-market).

    Arguments:
        return_X_y: return a tuple of (`X`, `y`) for convenience
        as_frame: return all the data as a pandas dataframe

    Usage:

    ```python
    from hulearn.datasets import load_fish

    df = load_fish(as_frame=True)
    X, y = load_fish(return_X_y=True)
    ```
    """
    filepath = resource_filename("hulearn", os.path.join("data", "fish.zip"))
    df = pd.read_csv(filepath)
    if as_frame:
        return df
    X, y = (
        df[["Species", "Length1", "Length2", "Length3", "Height", "Width"]].values,
        df["Weight"].values,
    )
    if return_X_y:
        return X, y
    return {"data": X, "target": y}
