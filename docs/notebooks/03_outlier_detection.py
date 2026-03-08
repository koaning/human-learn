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
        # Outlier Detection with Drawings

        Draw regions around "normal" data points. Anything outside the drawn
        regions is flagged as an outlier.

        This notebook demonstrates:
        - Drawing normal regions with `InteractiveCharts`
        - Creating an `InteractiveOutlierDetector`
        - The `threshold` parameter
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklego.datasets import load_penguins

    from hulearn.experimental.interactive import InteractiveCharts
    from hulearn.outlier import InteractiveOutlierDetector

    return InteractiveCharts, InteractiveOutlierDetector, load_penguins, np, plt


@app.cell
def _(load_penguins):
    df = load_penguins(as_frame=True).dropna()
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Draw normal regions

        Draw polygons around the clusters you consider "normal". Points outside
        all polygons will be considered outliers.
        """
    )
    return


@app.cell
def _(InteractiveCharts, df):
    charts = InteractiveCharts(df, labels="species")
    widget = charts.add_chart(x="bill_length_mm", y="bill_depth_mm")
    widget
    return charts, widget


@app.cell
def _(mo):
    mo.md(
        """
        ## Detect outliers

        Create an outlier detector from the drawn data. The `threshold` parameter
        controls how many polygons a point must fall into to be considered normal.
        """
    )
    return


@app.cell
def _(InteractiveOutlierDetector, charts, df):
    json_data = charts.data()
    det = InteractiveOutlierDetector(json_data, threshold=1)
    X = df.drop(columns=["species"])
    det.fit(X)
    preds = det.predict(X)
    n_outliers = (preds == -1).sum()
    print(f"Detected {n_outliers} outliers out of {len(preds)} points")
    return X, det, json_data, n_outliers, preds


@app.cell
def _(det, plt, X, preds):
    fig, ax = plt.subplots()
    colors = ["red" if p == -1 else "blue" for p in preds]
    ax.scatter(X["bill_length_mm"], X["bill_depth_mm"], c=colors, alpha=0.5, s=10)
    ax.set_xlabel("bill_length_mm")
    ax.set_ylabel("bill_depth_mm")
    ax.set_title("Outlier Detection (red = outlier)")
    fig
    return ax, colors, fig


if __name__ == "__main__":
    app.run()
