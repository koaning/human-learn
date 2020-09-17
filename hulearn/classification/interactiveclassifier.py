import json
import pathlib

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class InteractiveClassifier(BaseEstimator, ClassifierMixin):
    """
    This tool allows you to take a drawn model and use it as a classifier.

    Usage:

    ```python
    from sklego.datasets import load_penguins
    from hulearn.drawing-classifier.interactive import InteractiveClassifierCharts

    df = load_penguins(as_frame=True)
    charts = InteractiveClassifierCharts(df, labels="species")

    # Next notebook cell
    charts.add_chart(x="bill_length_mm", y="bill_depth_mm")
    # Next notebook cell
    charts.add_chart(x="flipper_length_mm", y="body_mass_g")

    # After drawing a model, export the data
    json_data = charts.data()

    # You can now use your drawn intuition as a model!
    from hulearn.drawing-classifier.interactive import HumanClassifier
    clf = HumanClassifier(clf_data)
    X, y = df.drop(columns=['species']), df['species']

    # This doesn't do anything. But scikit-learn demands it.
    clf.fit(X, y)

    # This makes predictions, based on your drawn model.
    # It can also be used in `GridSearchCV` for benchmarking!
    clf.predict(X)
    ```
    """

    def __init__(self, json_desc, smoothing=0.001):
        self.json_desc = json_desc
        self.smoothing = smoothing

    @classmethod
    def from_json(cls, path):
        """
        Load the classifier from json stored on disk.

        Arguments:
            path: path of the json file

        Usage:

        ```python
        from hulearn.drawing-classifier.interactive import HumanClassifier

        HumanClassifier.from_json("path/to/file.json")
        ```
        """
        json_desc = json.loads(pathlib.Path(path).read_text())
        return InteractiveClassifier(json_desc=json_desc)

    def _clean_poly_data(self, json_desc):
        """TODO: we need to prevent poly data with only two datapoints"""
        return json_desc

    @property
    def poly_data(self):
        for chart in self.json_desc:
            chard_id = chart["chart_id"]
            labels = chart["polygons"].keys()
            coords = chart["polygons"].values()
            for lab, p in zip(labels, coords):
                x_lab, y_lab = p.keys()
                x_coords, y_coords = list(p.values())
                for poly in [
                    Polygon(list(zip(x_coords[i], y_coords[i])))
                    for i in range(len(x_coords))
                ]:
                    yield {
                        "x_lab": x_lab,
                        "y_lab": y_lab,
                        "poly": poly,
                        "label": lab,
                        "chart_id": chard_id,
                    }

    def _count_hits(self, clf_data, data_in):
        counts = {k: 0 for k in self.classes_}
        for c in clf_data:
            point = Point(data_in[c["x_lab"]], data_in[c["y_lab"]])
            if c["poly"].contains(point):
                counts[c["label"]] += 1
        return counts

    def fit(self, X, y):
        """
        Fit the classifier. Bit of a formality, it's not doing anything specifically.
        """
        self.classes_ = list(self.json_desc[0]["polygons"].keys())
        return self

    def predict_proba(self, X):
        """
        Predicts the associated probabilities for each class.

        Usage:

        ```python
        from hulearn.drawing-classifier.interactive import HumanClassifier
        clf = HumanClassifier(clf_data)
        X, y = load_data(...)

        # This doesn't do anything. But scikit-learn demands it.
        clf.fit(X, y)

        # This makes predictions, based on your drawn model.
        clf.predict_proba(X)
        ```
        """
        if isinstance(X, pd.DataFrame):
            hits = [
                self._count_hits(self.poly_data, x[1].to_dict()) for x in X.iterrows()
            ]
        else:
            hits = [
                self._count_hits(self.poly_data, {k: v for k, v in enumerate(x)})
                for x in X
            ]
        count_arr = (
            np.array([[h[c] for c in self.classes_] for h in hits]) + self.smoothing
        )
        return count_arr / count_arr.sum(axis=1).reshape(-1, 1)

    def predict(self, X):
        """
        Predicts the class for each item in `X`.

        Usage:

        ```python
        from hulearn.drawing-classifier.interactive import HumanClassifier
        clf = HumanClassifier(clf_data)
        X, y = load_data(...)

        # This doesn't do anything. But scikit-learn demands it.
        clf.fit(X, y)

        # This makes predictions, based on your drawn model.
        clf.predict(X)
        ```
        """
        check_is_fitted(self, ["classes_"])
        return np.array(
            [self.classes_[i] for i in self.predict_proba(X).argmax(axis=1)]
        )
