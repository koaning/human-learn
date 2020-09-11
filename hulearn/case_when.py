import numpy as np


class CaseWhen:
    """
    The CaseWhen object allows you to construct a callable tree for regression/classification.

    Many custom models created with `FunctionClassifier` and `FunctionRegressor` fall into the "case-when"-category.
    To make it easier to construct these kinds of models you can use this object to define the logic.

    Important:
        This feature is experimental and unsupported

    Arguments:
        default: the default value that will be predicted of none of the cases are hit
        cases: a list of tuples that describe all the predictions

    **Usage:**

    ```python
    import pandas as pd
    from sklearn.model_selection import GridSearchCV

    from hulearn.case_when import CaseWhen
    from hulearn.datasets import load_titanic
    from hulearn.classification import FunctionClassifier

    def women_only(d):
        return d['sex'] == 'female'

    def pclass_high(d, pclass):
        return d['pclass'] == pclass

    df = load_titanic(as_frame=True)
    X, y = df.drop(columns=['survived']), df['survived']

    func = (CaseWhen(default=0)
            .when(women_only, 1)
            .when(pclass_high, 1))

    mod = FunctionClassifier(func, pclass=3)
    grid = GridSearchCV(mod, cv=3, param_grid={'pclass': [1, 2, 3]}).fit(X, y)
    ```
    """

    def __init__(self, default, cases=tuple([])):
        self.default = default
        self.cases = cases

    def when(self, predicate, prediction):
        """
        Adds a case-when statement to the tree.

        Arguments:
            predicate: a callable that returns `True`/`False` per row of `X`
            prediction: a value, callable or scikit-learn estimator to make predictions on `X`
        """
        new_case = [(predicate, prediction)]
        return CaseWhen(default=self.default, cases=tuple(list(self.cases) + new_case))

    def __call__(self, X, **kwargs):
        predictions = np.array([self.default for i in range(X.shape[0])])
        allready_predicted = np.array([False for i in range(X.shape[0])])
        for predicate, predictor in self.cases:
            true_false = np.array(predicate(X, **kwargs))
            if callable(predictor):
                predictor = predictor(X, **kwargs)
            set_new_value = true_false & ~allready_predicted
            predictions = np.where(set_new_value, predictor, predictions)
        return predictions
