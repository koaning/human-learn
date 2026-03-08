import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OutlierMixin

from hulearn.common import count_hits, iter_poly_data


class InteractiveOutlierDetector(BaseEstimator, OutlierMixin):
    """
    This tool allows you to take a drawn model and use it as an outlier detector. If a datapoint
    does not fit in any of the drawn polygons it becomes a candidate to become an outlier.

    Arguments:
        json_desc: python dictionary that contains drawn data
        threshold: the minimum number of polygons a point needs to be in to not be considered an outlier
        buffer: buffer to apply to drawn polygons (positive = grow, negative = shrink). Grid-searchable.

    Usage:

    ```python
    from sklego.datasets import load_penguins
    from hulearn.experimental.interactive import InteractiveCharts

    df = load_penguins(as_frame=True)
    charts = InteractiveCharts(df, labels="species")

    # Next notebook cell
    charts.add_chart(x="bill_length_mm", y="bill_depth_mm")
    # Next notebook cell
    charts.add_chart(x="flipper_length_mm", y="body_mass_g")

    # After drawing a model, export the data
    json_data = charts.data()

    # You can now use your drawn intuition as a model!
    from hulearn.outlier import InteractiveOutlierDetector
    clf = InteractiveOutlierDetector(clf_data)
    X, y = df.drop(columns=['species']), df['species']

    # This doesn't do anything. But scikit-learn demands it.
    clf.fit(X, y)

    # This makes predictions, based on your drawn model.
    # It can also be used in `GridSearchCV` for benchmarking!
    clf.predict(X)
    ```
    """

    def __init__(self, json_desc, threshold=1, buffer=0.0):
        self.json_desc = json_desc
        self.threshold = threshold
        self.buffer = buffer

    @classmethod
    def from_json(cls, path, threshold=1, buffer=0.0):
        """
        Load the classifier from json stored on disk.

        Arguments:
            path: path of the json file
            threshold: the minimum number of polygons a point needs to be in to not be considered an outlier
            buffer: buffer to apply to drawn polygons

        Usage:

        ```python
        from hulearn.outlier import InteractiveOutlierDetector

        InteractiveOutlierDetector.from_json("path/to/file.json")
        ```
        """
        json_desc = json.loads(pathlib.Path(path).read_text())
        return InteractiveOutlierDetector(json_desc=json_desc, threshold=threshold, buffer=buffer)

    @property
    def poly_data(self):
        return iter_poly_data(self.json_desc, buffer=self.buffer)

    def _count_hits(self, clf_data, data_in):
        return count_hits(clf_data, data_in, self.classes_)

    def fit(self, X, y=None):
        """
        Fit the classifier. Bit of a formality, it's not doing anything specifically.
        """
        self.classes_ = list(self.json_desc[0]["polygons"].keys())
        return self

    def score(self, X):
        if isinstance(X, pd.DataFrame):
            hits = [self._count_hits(self.poly_data, x[1].to_dict()) for x in X.iterrows()]
        else:
            hits = [self._count_hits(self.poly_data, {k: v for k, v in enumerate(x)}) for x in X]
        return np.array([[h[c] for c in self.classes_] for h in hits])

    def predict(self, X):
        """
        Predicts the associated probabilities for each class.

        Usage:

        ```python
        from hulearn.outlier import InteractiveOutlierDetector
        # Assuming a variable `clf_data` that contains the drawn polygons.
        clf = InteractiveOutlierDetector(clf_data)
        X, y = load_data(...)

        # This doesn't do anything. But scikit-learn demands it.
        clf.fit(X, y)

        # This makes predictions, based on your drawn model.
        clf.predict(X)
        ```
        """
        count_arr = self.score(X)
        return np.where(count_arr.sum(axis=1) < self.threshold, -1, 1)
