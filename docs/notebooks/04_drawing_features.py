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
        # Drawing Features

        Use `InteractivePreprocessor` to turn drawn regions into new features.
        Each class of polygons becomes a feature column counting how many
        polygons of that class contain each data point.

        This notebook demonstrates:
        - Drawing custom feature regions with `InteractiveCharts`
        - Using `InteractivePreprocessor` with scikit-learn's `transform()`
        - Using `InteractivePreprocessor` with pandas `.pipe()`
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from sklego.datasets import load_penguins
    from sklearn.pipeline import Pipeline, FeatureUnion

    from hulearn.experimental.interactive import InteractiveCharts
    from hulearn.preprocessing import InteractivePreprocessor, PipeTransformer

    return (
        FeatureUnion,
        InteractiveCharts,
        InteractivePreprocessor,
        Pipeline,
        PipeTransformer,
        load_penguins,
        np,
        pd,
    )


@app.cell
def _(load_penguins):
    df = load_penguins(as_frame=True).dropna()
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Draw feature regions

        Draw regions that you think are meaningful for classification.
        These will be turned into numeric features.
        """
    )
    return


@app.cell
def _(InteractiveCharts, df):
    charts = InteractiveCharts(df, labels=["cluster_a", "cluster_b", "cluster_c"])
    widget = charts.add_chart(x="bill_length_mm", y="bill_depth_mm")
    widget
    return charts, widget


@app.cell
def _(mo):
    mo.md(
        """
        ## scikit-learn transform()

        Use the preprocessor in a scikit-learn pipeline to generate new features.
        """
    )
    return


@app.cell
def _(InteractivePreprocessor, charts, df):
    json_data = charts.data()
    tfm = InteractivePreprocessor(json_data)
    X = df.drop(columns=["species"])
    features = tfm.fit(X).transform(X)
    print(f"New feature shape: {features.shape}")
    return X, features, json_data, tfm


@app.cell
def _(mo):
    mo.md(
        """
        ## pandas pipe()

        Alternatively, use `.pandas_pipe()` to add features directly to a DataFrame.
        """
    )
    return


@app.cell
def _(InteractivePreprocessor, df, json_data):
    tfm2 = InteractivePreprocessor(json_data)
    result = df.pipe(tfm2.pandas_pipe)
    result.head()
    return result, tfm2


if __name__ == "__main__":
    app.run()
