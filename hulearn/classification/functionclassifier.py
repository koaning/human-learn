from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_is_fitted, check_array, check_X_y


class FunctionClassifier(BaseEstimator, ClassifierMixin):
    """
    This class allows you to pass a function to make the predictions you're interested in.
    """

    def __init__(self, func):
        self.func = func

    def fit(self, X, y):
        """
        Fit the classifier.

        This classifier tries to confirm if the passed function can predict appropriate values on the train set.
        """
        # Run it to confirm no error happened.
        self.classes_ = unique_labels(y)
        X, y = check_X_y(X, y)
        predictions = self.func(X)
        for p in predictions:
            if p not in self.classes_:
                raise ValueError(
                    f"Function created prediction {p} which is not in y={self.classes_}"
                )
        return self

    def predict(self, X):
        """
        Make predictions using the passed function.
        """
        X = check_array(X)
        check_is_fitted(self, ["classes_"])
        return self.func(X)
