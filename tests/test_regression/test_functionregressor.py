import pytest
import numpy as np

from sklearn.model_selection import GridSearchCV
from hulearn.regression.functionregressor import FunctionRegressor
from hulearn.common import flatten

from tests.conftest import (
    select_tests,
    general_checks,
    regressor_checks,
    nonmeta_checks,
)


def predict(X):
    np.random.seed(42)
    return np.array([1 if r > 0.5 else 0 for r in np.random.normal(0, 1, len(X))])


def predict_variant(X):
    np.random.seed(42)
    return np.array([1 if r > 0.0 else 0 for r in np.random.normal(0, 1, len(X))])


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        include=flatten([general_checks, regressor_checks, nonmeta_checks]),
        exclude=[
            "check_methods_subset_invariance",
            "check_fit2d_1sample",
            "check_fit2d_1feature",
            "check_regressors_train",
            "check_fit2d_predict1d",
            "check_fit1d",
            "check_regressor_data_not_an_array",
            "check_supervised_y_2d",
            "check_supervised_y_no_nan",
            "check_dtype_object",
            "check_complex_data",
            "check_estimators_empty_data_messages",
            "check_estimators_nan_inf",
            "check_estimator_sparse_data",
            "check_estimators_partial_fit_n_features",
            "check_sample_weights_list",
            "check_sample_weights_pandas_series",
        ],
    ),
)
def test_estimator_checks(test_fn):
    clf = FunctionRegressor(func=predict)
    test_fn(FunctionRegressor.__name__ + "_fallback", clf)


def test_works_with_gridsearch(random_xy_dataset_regr):
    X, y = random_xy_dataset_regr
    clf = FunctionRegressor(func=predict)
    grid = GridSearchCV(clf, cv=5, param_grid={"func": [predict, predict_variant]})
    grid.fit(X, y).predict(X)
