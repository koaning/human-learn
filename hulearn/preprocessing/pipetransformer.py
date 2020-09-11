from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, check_array


class PipeTransformer(TransformerMixin, BaseEstimator):
    """
    This transformer allows you to define a function that will take in
    data and transform it however you like. You can specify keyword arguments
    that you can benchmark as well.
    """

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """
        Fit the classifier.

        This classifier tries to confirm if the passed function can predict appropriate values on the train set.
        """
        # Run it to confirm no error happened.
        _ = self.func(X, **self.kwargs)
        self.fitted_ = True
        self.ncol_ = 0 if len(X.shape) == 1 else X.shape[1]
        return self

    def transform(self, X):
        """
        Make predictions using the passed function.
        """
        check_is_fitted(self, ["fitted_", "ncol_"])
        ncol = 0 if len(X.shape) == 1 else X.shape[1]
        if self.ncol_ != ncol:
            raise ValueError(f"Reshape your data, there were {self.ncol_} features during training, now={ncol}.")
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
