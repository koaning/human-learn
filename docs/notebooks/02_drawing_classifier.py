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
        # Drawing a Classifier

        Use `InteractiveCharts` to draw decision regions on scatter plots of your data,
        then export those drawings as a scikit-learn compatible classifier.

        This notebook demonstrates:
        - Drawing classifications with `ChartMultiSelect`
        - Exporting drawn data to create an `InteractiveClassifier`
        - How the `buffer` hyperparameter grows/shrinks drawn regions
        """
    )
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from sklego.datasets import load_penguins

    from hulearn.experimental.interactive import InteractiveCharts
    from hulearn.classification import InteractiveClassifier

    return InteractiveCharts, InteractiveClassifier, load_penguins, np, plt


@app.cell
def _(load_penguins):
    df = load_penguins(as_frame=True).dropna()
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Draw classifications

        Use the widget below to draw regions around clusters of penguin species.
        Select a class, then use lasso to draw a polygon around the points.
        """
    )
    return


@app.cell
def _(InteractiveCharts, df):
    charts = InteractiveCharts(df, labels="species")
    widget1 = charts.add_chart(x="bill_length_mm", y="bill_depth_mm")
    widget1
    return charts, widget1


@app.cell
def _(charts):
    widget2 = charts.add_chart(x="flipper_length_mm", y="body_mass_g")
    widget2
    return (widget2,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Export and predict

        After drawing, export the data and create a classifier.
        """
    )
    return


@app.cell
def _(InteractiveClassifier, charts, df):
    json_data = charts.data()
    clf = InteractiveClassifier(json_data)
    X, y = df.drop(columns=["species"]), df["species"]
    clf.fit(X, y)
    preds = clf.predict(X)
    print(f"Unique predictions: {set(preds)}")
    return X, clf, json_data, preds, y


@app.cell
def _(mo):
    mo.md(
        """
        ## What about points outside any drawn region?

        When a data point doesn't fall inside **any** drawn polygon, it gets a hit count of
        zero for every class. The classifier applies a small `smoothing` value (default `0.001`)
        to all counts before normalizing, so these unclassified points end up with
        **equal probability across all classes**. In practice this means `predict()` will
        assign them to whichever class comes first alphabetically — effectively a random-looking
        assignment. If this matters for your use case, make sure your drawn regions cover the
        full data space, or increase the `buffer` parameter to expand your polygons.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
        ## Buffer hyperparameter

        The `buffer` parameter grows (positive) or shrinks (negative) drawn polygons using
        Shapely's `.buffer()`. Use the slider below to see how different buffer values
        change the predictions. This parameter is also grid-searchable!

        ```python
        GridSearchCV(clf, param_grid={"buffer": [-0.5, 0, 0.5, 1.0]})
        ```
        """
    )
    return


@app.cell
def _(mo):
    buffer_slider = mo.ui.slider(
        start=-2.0,
        stop=3.0,
        step=0.25,
        value=0.0,
        label="buffer",
    )
    buffer_slider
    return (buffer_slider,)


@app.cell
def _(InteractiveClassifier, X, buffer_slider, json_data, np, plt, y):
    _clf = InteractiveClassifier(json_data, buffer=buffer_slider.value)
    _clf.fit(X, y)
    _preds = _clf.predict(X)
    _accuracy = np.mean(_preds == y.values)

    fig, ax = plt.subplots(figsize=(8, 3))
    classes = _clf.classes_
    counts = [np.sum(_preds == c) for c in classes]
    colors = plt.cm.tab10(range(len(classes)))
    ax.bar(classes, counts, color=colors)
    ax.set_ylabel("Number of predictions")
    ax.set_title(f"buffer={buffer_slider.value:.2f} — accuracy={_accuracy:.2%}")
    plt.tight_layout()
    fig
    return (fig,)


@app.cell
def _(mo):
    mo.md(
        """
        ## Explore with parallel coordinates

        Before drawing, it helps to explore the data interactively.
        Use the parallel coordinates chart below to brush across axes
        and discover which feature ranges separate the penguin species.
        """
    )
    return


@app.cell
def _():
    from hulearn.experimental.interactive import parallel_coordinates

    return (parallel_coordinates,)


@app.cell
def _(df, parallel_coordinates):
    pc_widget = parallel_coordinates(df.drop(columns=["island", "sex"]), label="species", height=300)
    pc_widget
    return (pc_widget,)


@app.cell
def _(mo):
    mo.md(
        """
        Try brushing the `flipper_length_mm` axis — you'll see that Gentoo penguins
        cluster clearly at higher flipper lengths, while Adelie and Chinstrap overlap
        more on bill dimensions. These patterns can guide where you draw your
        classification regions above.
        """
    )
    return


if __name__ == "__main__":
    app.run()
