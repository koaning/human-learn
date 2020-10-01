import json
import pathlib

import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted


class InteractivePreprocessor(BaseEstimator):
    """
    This tool allows you to take a drawn model and use it as a featurizer.

    Arguments:
        json_desc: chart da ta in dictionary form
        refit: if `True`, you no longer need to call `.fit(X, y)` in order to `.predict(X)`
    """

    def __init__(self, json_desc, refit=True):
        self.json_desc = json_desc
        self.refit = refit

    @classmethod
    def from_json(cls, path, refit=True):
        """
        Load the classifier from json stored on disk.

        Arguments:
            path: path of the json file
            refit: if `True`, you no longer need to call `.fit(X, y)` in order to `.predict(X)`

        Usage:

        ```python
        from hulearn.classification import InteractivePreprocessor

        InteractivePreprocessor.from_json("path/to/file.json")
        ```
        """
        json_desc = json.loads(pathlib.Path(path).read_text())
        return InteractivePreprocessor(json_desc=json_desc, refit=refit)

    @property
    def poly_data(self):
        for chart in self.json_desc:
            chard_id = chart["chart_id"]
            labels = chart["polygons"].keys()
            coords = chart["polygons"].values()
            for lab, p in zip(labels, coords):
                x_lab, y_lab = p.keys()
                x_coords, y_coords = list(p.values())
                for i in range(len(x_coords)):
                    poly_data = list(zip(x_coords[i], y_coords[i]))
                    if len(poly_data) >= 3:
                        poly = Polygon(poly_data)
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

    def fit(self, X, y=None):
        """
        Fit the classifier. Bit of a formality, it's not doing anything specifically.
        """
        self.classes_ = list(self.json_desc[0]["polygons"].keys())
        self.fitted_ = True
        return self

    def transform(self, X):
        """
        Apply the counting/binning based on the drawings.

        Usage:

        ```python
        from hulearn.preprocessing import InteractivePreprocessor
        clf = InteractivePreprocessor(clf_data)
        X, y = load_data(...)

        # This doesn't do anything. But scikit-learn demands it.
        clf.fit(X, y)

        # This makes predictions, based on your drawn model.
        clf.transform(X)
        ```
        """
        # Because we're not doing anything during training, for convenience this
        # method can formally "fit" during the predict call. This is a scikit-learn
        # anti-pattern so we allow you to turn this off.
        if self.refit:
            if not self.fitted_:
                self.fit(X)
        check_is_fitted(self, ["classes_", "fitted_"])
        if isinstance(X, pd.DataFrame):
            hits = [
                self._count_hits(self.poly_data, x[1].to_dict()) for x in X.iterrows()
            ]
        else:
            hits = [
                self._count_hits(self.poly_data, {k: v for k, v in enumerate(x)})
                for x in X
            ]
        count_arr = np.array([[h[c] for c in self.classes_] for h in hits])
        return count_arr

    def pandas_pipe(self, dataf):
        """
        Use this transformer as part of a `.pipe()` method chain in pandas.

        Usage:

        ```python
        import numpy as np
        import pandas as pd

        # Load in a dataframe from somewhere
        df = load_data(...)

        # Load in drawn chart data
        from hulearn.preprocessing import InteractivePreprocessor
        tfm = InteractivePreprocessor.from_json("path/file.json")

        # This adds new columns to the dataframe
        df.pipe(pandas_pipe)
        ```
        """
        new_dataf = pd.DataFrame(
            self.fit(dataf).transform(dataf), columns=self.classes_
        )
        return pd.concat(
            [dataf.copy().reset_index(drop=True), new_dataf.reset_index(drop=True)],
            axis=1,
        )
