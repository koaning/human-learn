import pytest
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV

from hulearn.preprocessing import PipeTransformer
from hulearn.common import flatten
from tests.conftest import (
    select_tests,
    general_checks,
    transformer_checks,
    nonmeta_checks,
)


def double(x, factor=2):
    return x * factor


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        include=flatten([general_checks, transformer_checks, nonmeta_checks]),
        exclude=[
            "check_estimators_nan_inf",
            "check_estimators_empty_data_messages",
            "check_transformer_data_not_an_array",
            "check_dtype_object",
            "check_complex_data",
            "check_fit1d",
        ],
    ),
)
def test_estimator_checks(test_fn):
    clf = PipeTransformer(func=double)
    test_fn(PipeTransformer.__name__, clf)


@pytest.mark.parametrize("factor", [1.0, 2.0, 5.0])
def test_basic_example(factor):
    np.random.seed(42)
    X = np.random.normal(0, 1, (1000, 4))
    tfm = PipeTransformer(func=double, factor=factor)
    X_tfm = tfm.fit_transform(X)
    assert np.all(np.isclose(X * factor, X_tfm))


def test_works_with_pipeline_gridsearch(random_xy_dataset_clf):
    X, y = random_xy_dataset_clf
    pipe = Pipeline(
        [("pipe", PipeTransformer(func=double, factor=1)), ("mod", GaussianNB())]
    )
    grid = GridSearchCV(pipe, cv=2, param_grid={"pipe__factor": [1, 2, 3]})
    grid.fit(X, y).predict(X)
