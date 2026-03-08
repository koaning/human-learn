import marimo

app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        """
        # Function-based Preprocessing

        The `PipeTransformer` lets you use any pandas `.pipe()` function as a
        scikit-learn transformer. This makes it easy to do feature engineering
        with familiar pandas idioms inside a scikit-learn pipeline.

        This notebook demonstrates:
        - Creating a `PipeTransformer` with a pandas function
        - Using it in a scikit-learn `Pipeline`
        - Grid-searching over preprocessing parameters
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline

    from hulearn.datasets import load_titanic
    from hulearn.preprocessing import PipeTransformer

    return (
        GridSearchCV,
        LogisticRegression,
        Pipeline,
        PipeTransformer,
        load_titanic,
        np,
        pd,
    )


@app.cell
def _(load_titanic):
    df = load_titanic(as_frame=True)
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(
        """
        ## A preprocessing function

        Define a function that takes a DataFrame and returns a transformed DataFrame.
        This function selects numeric columns and adds engineered features.
        """
    )
    return


@app.cell
def _(np):
    def preprocess(dataf, n_bins=10):
        return dataf.assign(
            fare_bin=lambda d: np.digitize(d["fare"], bins=np.linspace(0, 100, n_bins)),
            age_bin=lambda d: np.digitize(d["age"], bins=np.linspace(0, 80, n_bins)),
        )[["fare_bin", "age_bin", "pclass"]]

    return (preprocess,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Using PipeTransformer in a Pipeline

        Wrap the function with `PipeTransformer` and use it in a scikit-learn pipeline.
        """
    )
    return


@app.cell
def _(LogisticRegression, Pipeline, PipeTransformer, df, preprocess):
    pipe = Pipeline(
        [
            ("prep", PipeTransformer(preprocess)),
            ("model", LogisticRegression(max_iter=1000)),
        ]
    )
    X, y = df.drop(columns=["survived"]), df["survived"]
    pipe.fit(X, y)
    print(f"Train accuracy: {pipe.score(X, y):.3f}")
    return X, pipe, y


@app.cell
def _(mo):
    mo.md(
        """
        ## Grid search over preprocessing parameters

        Since `n_bins` is a parameter of our preprocessing function, we can grid-search over it.
        """
    )
    return


@app.cell
def _(GridSearchCV, X, pipe, y):
    grid = GridSearchCV(
        pipe,
        param_grid={"prep__n_bins": [5, 10, 15, 20]},
        cv=3,
        scoring="accuracy",
    )
    grid.fit(X, y)
    print(f"Best n_bins: {grid.best_params_}")
    print(f"Best CV accuracy: {grid.best_score_:.3f}")
    return (grid,)


if __name__ == "__main__":
    app.run()
