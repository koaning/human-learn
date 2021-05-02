from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class PipeTransformer(TransformerMixin, BaseEstimator):
    """
    This transformer allows you to define a function that will take in
    data and transform it however you like. You can specify keyword arguments
    that you can benchmark as well.

    Arguments:
        func: the function that can make predictions
        kwargs: extra keyword arguments will be pass to the function, can be grid-search-able

    The functions that are passed need to be pickle-able. That means no lambda functions!

    Usage:

    ```python
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.naive_bayes import GaussianNB
    from sklearn.model_selection import GridSearchCV

    from hulearn.datasets import load_titanic
    from hulearn.preprocessing import PipeTransformer


    def preprocessing(dataf, n_char=True, gender=True):
        dataf = dataf.copy()
        # I'm not using .assign() in this pipeline because lambda functions
        # do not pickle and GridSearchCV demands that it can.
        if n_char:
            dataf['nchar'] = dataf['name'].str.len()
        if gender:
            dataf['gender'] = (dataf['sex'] == 'male').astype("float")
        return dataf.drop(columns=["name", "sex"])


    df = load_titanic(as_frame=True)
    X, y = df.drop(columns=['survived']), df['survived']

    pipe = Pipeline([
        ('prep', PipeTransformer(preprocessing, n_char=True, gender=True)),
        ('mod', GaussianNB())
    ])

    params = {
        "prep__n_char": [True, False],
        "prep__gender": [True, False]
    }

    grid = GridSearchCV(pipe, cv=3, param_grid=params).fit(X, y)
    pd.DataFrame(grid.cv_results_)[['param_prep__gender', 'param_prep__n_char', 'mean_test_score']]
    ```
    """

    def __init__(self, func, **kwargs):
        self.func = func
        self.kwargs = kwargs

    def fit(self, X, y=None):
        """
        Fit the classifier. No-Op.
        """
        # Run it to confirm no error happened.
        _ = self.func(X, **self.kwargs)
        self.fitted_ = True
        self.ncol_ = 0 if len(X.shape) == 1 else X.shape[1]
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

    def transform(self, X):
        """
        Make predictions using the passed function.
        """
        check_is_fitted(self, ["fitted_", "ncol_"])
        ncol = 0 if len(X.shape) == 1 else X.shape[1]
        if self.ncol_ != ncol:
            raise ValueError(
                f"Reshape your data, there were {self.ncol_} features during training, now={ncol}."
            )
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
