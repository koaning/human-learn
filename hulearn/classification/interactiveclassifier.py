import json
import pathlib

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

from hulearn.common import count_hits, iter_poly_data


class InteractiveClassifier(BaseEstimator, ClassifierMixin):
    """
    This tool allows you to take a drawn model and use it as a classifier.

    Arguments:
        json_desc: chart data in dictionary form
        smoothing: smoothing to apply to poly-counts
        refit: if `True`, you no longer need to call `.fit(X, y)` in order to `.predict(X)`
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
    from hulearn.classification.interactive import InteractiveClassifier
    clf = InteractiveClassifier(clf_data)
    X, y = df.drop(columns=['species']), df['species']

    # This doesn't do anything. But scikit-learn demands it.
    clf.fit(X, y)

    # This makes predictions, based on your drawn model.
    # It can also be used in `GridSearchCV` for benchmarking!
    clf.predict(X)
    ```
    """

    def __init__(self, json_desc, smoothing=0.001, refit=True, buffer=0.0):
        self.json_desc = json_desc
        self.smoothing = smoothing
        self.refit = refit
        self.buffer = buffer

    @classmethod
    def from_json(cls, path, smoothing=0.001, refit=True, buffer=0.0):
        """
        Load the classifier from json stored on disk.

        Arguments:
            path: path of the json file
            smoothing: smoothing to apply to poly-counts
            refit: if `True`, you no longer need to call `.fit(X, y)` in order to `.predict(X)`
            buffer: buffer to apply to drawn polygons

        Usage:

        ```python
        from hulearn.classification import InteractiveClassifier

        InteractiveClassifier.from_json("path/to/file.json")
        ```
        """
        json_desc = json.loads(pathlib.Path(path).read_text())
        return InteractiveClassifier(json_desc=json_desc, smoothing=smoothing, refit=refit, buffer=buffer)

    @property
    def poly_data(self):
        return iter_poly_data(self.json_desc, buffer=self.buffer)

    def _count_hits(self, clf_data, data_in):
        return count_hits(clf_data, data_in, self.classes_)

    def fit(self, X, y):
        """
        Fit the classifier. Bit of a formality, it's not doing anything specifically.
        """
        self.classes_ = list(self.json_desc[0]["polygons"].keys())
        self.fitted_ = True
        return self

    def predict_proba(self, X):
        """
        Predicts the associated probabilities for each class.

        Usage:

        ```python
        from hulearn.classification import InteractiveClassifier
        clf = InteractiveClassifier(clf_data)
        X, y = load_data(...)

        # This doesn't do anything. But scikit-learn demands it.
        clf.fit(X, y)

        # This makes predictions, based on your drawn model.
        clf.predict_proba(X)
        ```
        """
        if self.refit:
            if not self.fitted_:
                self.fit(X)
        check_is_fitted(self, ["classes_", "fitted_"])
        if isinstance(X, pd.DataFrame):
            hits = [self._count_hits(self.poly_data, x[1].to_dict()) for x in X.iterrows()]
        else:
            hits = [self._count_hits(self.poly_data, {k: v for k, v in enumerate(x)}) for x in X]
        count_arr = np.array([[h[c] for c in self.classes_] for h in hits]) + self.smoothing
        return count_arr / count_arr.sum(axis=1).reshape(-1, 1)

    def predict(self, X):
        """
        Predicts the class for each item in `X`.

        Usage:

        ```python
        from hulearn.classification import InteractiveClassifier
        clf = InteractiveClassifier(clf_data)
        X, y = load_data(...)

        # This doesn't do anything. But scikit-learn demands it.
        clf.fit(X, y)

        # This makes predictions, based on your drawn model.
        clf.predict(X)
        ```
        """
        check_is_fitted(self, ["classes_", "fitted_"])
        return np.array([self.classes_[i] for i in self.predict_proba(X).argmax(axis=1)])
