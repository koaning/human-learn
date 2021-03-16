from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.utils.validation import check_is_fitted


class FunctionOutlierDetector(BaseEstimator, OutlierMixin):
    """
    This class allows you to pass a function to detect outliers you're interested in. Note that the output
    of the function needs to be an array with [-1, 1] values (-1 denotes outliers).

    Arguments:
        func: the function that return an array of True/False
        kwargs: extra keyword arguments will be pass to the function, can be grid-search-able

    The functions that are passed need to be pickle-able. That means no lambda functions!
    """

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """
        Fit the classifier. No-Op.
        """
        # Run it to confirm no error happened.
        self.fitted_ = True
        _ = self.func(X, **self.kwargs)
        return self

    def partial_fit(self, X, y=None):
        """
        Fit the classifier partially. No-Op.
        """
        # Run it to confirm no error happened.
        _ = self.func(X, **self.kwargs)
        self.fitted_ = True
        self.ncol_ = 0 if len(X.shape) == 1 else X.shape[1]
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
