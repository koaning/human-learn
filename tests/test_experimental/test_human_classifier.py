import pytest

from sklearn.model_selection import GridSearchCV
from sklego.datasets import load_penguins
from sklearn.pipeline import Pipeline

from hulearn.preprocessing import PipeTransformer
from hulearn.experimental.interactive import InteractiveClassifier
from hulearn.common import flatten

from tests.conftest import (
    select_tests,
    general_checks,
    classifier_checks,
    nonmeta_checks,
)


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        include=flatten([general_checks, classifier_checks, nonmeta_checks]),
        exclude=[
            "check_estimators_pickle",
            "check_estimator_sparse_data",
            "check_estimators_nan_inf",
            "check_pipeline_consistency",
            "check_complex_data",
            "check_fit2d_predict1d",
            "check_methods_subset_invariance",
            "check_fit1d",
            "check_dict_unchanged",
            "check_classifier_data_not_an_array",
            "check_classifiers_one_label",
            "check_classifiers_classes",
            "check_classifiers_train",
            "check_supervised_y_2d",
            "check_supervised_y_no_nan",
            "check_estimators_unfitted",
            "check_estimators_dtypes",
            "check_fit_score_takes_y",
            "check_dtype_object",
            "check_estimators_empty_data_messages",
        ],
    ),
)
def test_estimator_checks(test_fn):
    """
    We're skipping a lot of tests here mainly because this model is "bespoke"
    it is *not* general. Therefore a lot of assumptions are broken.
    """
    clf = InteractiveClassifier.from_json("tests/test_experimental/demo-data.json")
    test_fn(InteractiveClassifier, clf)


def test_base_predict_usecase():
    clf = InteractiveClassifier.from_json("tests/test_experimental/demo-data.json")
    df = load_penguins(as_frame=True).dropna()
    X, y = df.drop(columns=["species"]), df["species"]

    preds = clf.fit(X, y).predict_proba(X)

    assert preds.shape[0] == df.shape[0]
    assert preds.shape[1] == 3


def identity(x):
    return x


def test_grid_predict_usecase():
    clf = InteractiveClassifier.from_json("tests/test_experimental/demo-data.json")
    pipe = Pipeline(
        [
            ("id", PipeTransformer(identity)),
            ("mod", clf),
        ]
    )
    grid = GridSearchCV(pipe, cv=5, param_grid={})
    df = load_penguins(as_frame=True).dropna()
    X, y = df.drop(columns=["species", "island", "sex"]), df["species"]

    preds = grid.fit(X, y).predict_proba(X)

    assert preds.shape[0] == df.shape[0]
    assert preds.shape[1] == 3
