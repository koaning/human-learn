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
        # Function Classifier

        The `FunctionClassifier` lets you encode domain knowledge directly as a Python function
        and use it as a scikit-learn compatible classifier.

        This notebook demonstrates:
        - Creating a classifier from a simple function
        - Using `GridSearchCV` to tune thresholds
        - Exploring data with interactive parallel coordinates
        - Building a "women and children first" rule for the Titanic dataset
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import accuracy_score

    from hulearn.datasets import load_titanic
    from hulearn.classification import FunctionClassifier
    from hulearn.experimental.interactive import parallel_coordinates

    return (
        FunctionClassifier,
        GridSearchCV,
        accuracy_score,
        load_titanic,
        np,
        parallel_coordinates,
        pd,
        plt,
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
        ## A simple fare-based rule

        Let's start with a basic rule: passengers who paid more than a threshold fare survived.
        """
    )
    return


@app.cell
def _(FunctionClassifier, df, np):
    def fare_based(dataf, threshold=10):
        return np.array(["survived" if f > threshold else "not survived" for f in dataf["fare"]])

    clf = FunctionClassifier(fare_based)
    X, y = df.drop(columns=["survived"]), df["survived"]
    clf.fit(X, y)
    preds = clf.predict(X)
    preds[:10]
    return X, clf, fare_based, preds, y


@app.cell
def _(mo):
    mo.md(
        """
        ## Grid search over the threshold

        Since `threshold` is a parameter of our function, we can grid-search over it.
        """
    )
    return


@app.cell
def _(FunctionClassifier, GridSearchCV, X, fare_based, y):
    grid = GridSearchCV(
        FunctionClassifier(fare_based),
        param_grid={"threshold": [5, 10, 15, 20, 30, 50]},
        cv=3,
        scoring="accuracy",
    )
    grid.fit(X, y)
    grid.best_params_
    return (grid,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Exploring with parallel coordinates

        Use the interactive parallel coordinates chart below to explore the Titanic data.
        Brush across axes to filter and find patterns.
        """
    )
    return


@app.cell
def _(df, parallel_coordinates):
    widget = parallel_coordinates(df, label="survived", height=300)
    widget
    return (widget,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Women and children first

        A classic rule: women and children (under a certain age) had priority on lifeboats.
        """
    )
    return


@app.cell
def _(FunctionClassifier, X, accuracy_score, np, y):
    def women_and_children(dataf, age_threshold=15):
        return np.array(
            [
                "survived" if (row["sex"] == "female" or row["age"] < age_threshold) else "not survived"
                for _, row in dataf.iterrows()
            ]
        )

    clf_wc = FunctionClassifier(women_and_children)
    clf_wc.fit(X, y)
    preds_wc = clf_wc.predict(X)
    print(f"Accuracy: {accuracy_score(y.map({0: 'not survived', 1: 'survived'}), preds_wc):.3f}")
    return clf_wc, preds_wc, women_and_children


if __name__ == "__main__":
    app.run()
