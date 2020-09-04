import numpy as np


class CaseWhen:
    """
    The CaseWhen object allows you to construct a callable tree for regression/classification.

    Many custom models created with `FunctionClassifier` and `FunctionRegressor` fall into the "case-when"-category.
    To make it easier to construct these kinds of models you can use this object to define the logic.

    Arguments:
        default: the default value that will be predicted of none of the cases are hit
        cases: a list of tuples that describe all the predictions
    """

    def __init__(self, default, cases=tuple([])):
        self.default = default
        self.cases = []

    def when(self, predicate, prediction):
        """
        Adds a case-when statement to the tree.

        Arguments:
            predicate: a callable that returns `True`/`False` per row of `X`
            prediction: a value, callable or scikit-learn estimator to make predictions on `X`
        """
        return CaseWhen(default=self.default, cases=tuple(self.cases + [(predicate, prediction)]))

    def __call__(self, X, **kwargs):
        predictions = np.array([self.default for i in range(X.shape[0])])
        for predicate, predictor in self.cases:
            true_false = np.array(predicate(X, **kwargs))
            predictions = np.where(true_false, predictor, predictions)
        return predictions