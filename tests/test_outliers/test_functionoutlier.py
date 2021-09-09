import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer

from hulearn.datasets import load_titanic
from hulearn.outlier import FunctionOutlierDetector
from hulearn.common import flatten

from tests.conftest import (
    select_tests,
    general_checks,
    outlier_checks,
    nonmeta_checks,
)


def predict(X):
    np.random.seed(42)
    return np.array([1 if r > 0.5 else -1 for r in np.random.normal(0, 1, len(X))])


def predict_variant(X):
    np.random.seed(42)
    return np.array([1 if r > 0.0 else -1 for r in np.random.normal(0, 1, len(X))])


def class_based(dataf, sex="male", pclass=1):
    predicate = (dataf["sex"] == sex) & (dataf["pclass"] == pclass)
    return np.array(predicate).astype(int) * 2 - 1


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        include=flatten([general_checks, outlier_checks, nonmeta_checks]),
        exclude=[
            "check_outliers_train",
            "check_estimators_nan_inf",
            "check_estimators_empty_data_messages",
            "check_complex_data",
            "check_dtype_object",
            "check_classifier_data_not_an_array",
            "check_fit1d",
            "check_methods_subset_invariance",
            "check_fit2d_predict1d",
            "check_estimator_sparse_data",
            "check_sample_weights_list",
            "check_sample_weights_pandas_series",
        ],
    ),
)
def test_estimator_checks(test_fn):
    clf = FunctionOutlierDetector(func=predict)
    test_fn(FunctionOutlierDetector.__name__ + "_fallback", clf)


def test_works_with_gridsearch(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    clf = FunctionOutlierDetector(func=predict)
    grid = GridSearchCV(
        clf,
        cv=5,
        scoring={"acc": make_scorer(accuracy_score)},
        refit="acc",
        param_grid={"func": [predict, predict_variant]},
    )
    grid.fit(X, y).predict(X)


def test_smoke_with_pandas():
    df = load_titanic(as_frame=True)
    X, y = df.drop(columns=["survived"]), df["survived"]

    mod = FunctionOutlierDetector(class_based, pclass=10)
    params = {"pclass": [1, 2, 3], "sex": ["male", "female"]}
    grid = GridSearchCV(
        mod,
        cv=3,
        scoring={"acc": make_scorer(accuracy_score)},
        refit="acc",
        param_grid=params,
    ).fit(X, y)
    pd.DataFrame(grid.cv_results_)
