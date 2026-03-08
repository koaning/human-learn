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
        # Model Mining with Human Rules

        This notebook demonstrates how to use human-learn for "model mining" —
        exploring data interactively and encoding discovered rules as models.

        We use `parallel_coordinates` for exploration and `FunctionClassifier`
        to encode rules, then compare with traditional ML approaches.
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklearn.metrics import accuracy_score, classification_report

    from hulearn.datasets import load_titanic
    from hulearn.classification import FunctionClassifier
    from hulearn.experimental.interactive import parallel_coordinates

    return (
        FunctionClassifier,
        accuracy_score,
        classification_report,
        load_titanic,
        np,
        parallel_coordinates,
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
        ## Explore with parallel coordinates

        Use the interactive chart below to brush and filter the data.
        Look for patterns that separate survivors from non-survivors.
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
        ## Encode discovered rules

        Based on exploration, encode rules as a Python function.
        """
    )
    return


@app.cell
def _(FunctionClassifier, accuracy_score, df, np):
    def survival_rules(dataf, fare_threshold=20, age_threshold=12):
        results = []
        for _, row in dataf.iterrows():
            if row["sex"] == "female":
                results.append(1)
            elif row["age"] < age_threshold:
                results.append(1)
            elif row["fare"] > fare_threshold and row["pclass"] == 1:
                results.append(1)
            else:
                results.append(0)
        return np.array(results)

    clf = FunctionClassifier(survival_rules)
    X, y = df.drop(columns=["survived"]), df["survived"]
    clf.fit(X, y)
    preds = clf.predict(X)
    print(f"Rule-based accuracy: {accuracy_score(y, preds):.3f}")
    return X, clf, preds, survival_rules, y


@app.cell
def _(mo):
    mo.md(
        """
        ## Compare with ML

        Compare the human rules against a traditional ML model.
        """
    )
    return


@app.cell
def _(X, accuracy_score, y):
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    X_numeric = X.select_dtypes(include=["number"])
    rf.fit(X_numeric, y)
    rf_preds = rf.predict(X_numeric)
    print(f"Random Forest accuracy: {accuracy_score(y, rf_preds):.3f}")
    return RandomForestClassifier, X_numeric, rf, rf_preds


if __name__ == "__main__":
    app.run()
