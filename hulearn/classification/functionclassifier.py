from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class FunctionClassifier(BaseEstimator, ClassifierMixin):
    """
    This class allows you to pass a function to make the predictions you're interested in.

    Arguments:
        func: the function that can make predictions
        kwargs: extra keyword arguments will be pass to the function, can be grid-search-able

    The functions that are passed need to be pickle-able. That means no lambda functions!

    **Usage:**

    ```python
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import GridSearchCV

    from hulearn.datasets import load_titanic
    from hulearn.classification import FunctionClassifier

    df = load_titanic(as_frame=True)
    X, y = df.drop(columns=['survived']), df['survived']

    def class_based(dataf, sex='male', pclass=1):
        predicate = (dataf['sex'] == sex) & (dataf['pclass'] == pclass)
        return np.array(predicate).astype(int)

    mod = FunctionClassifier(class_based, pclass=10)
    params = {'pclass': [1, 2, 3], 'sex': ['male', 'female']}
    grid = GridSearchCV(mod, cv=3, param_grid=params).fit(X, y)
    pd.DataFrame(grid.cv_results_)
    ```
    """

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def fit(self, X, y):
        """
        Fit the classifier. No-Op.
        """
        # Run it to confirm no error happened.
        self.fitted_ = True
        _ = self.func(X, **self.kwargs)
        return self

    def partial_fit(self, X, y, classes=None, sample_weight=None):
        """
        Fit the classifier partially. No-Op.
        """
        # Run it to confirm no error happened.
        self.fitted_ = True
        _ = self.func(X, **self.kwargs)
        return self

    def predict(self, X):
        """
        Make predictions using the passed function.
        """
        check_is_fitted(self, ["fitted_"])
        return self.func(X, **self.kwargs)

    def get_params(self, deep=True):
        """"""
        return {**self.kwargs, "func": self.func}

    def set_params(self, **params):
        """"""
        for k, v in params.items():
            if k == "func":
                self.func = v
            else:
                self.kwargs[k] = v
        return self
