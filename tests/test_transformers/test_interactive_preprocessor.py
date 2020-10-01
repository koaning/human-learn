import pytest

from sklego.datasets import load_penguins
from sklearn.pipeline import Pipeline, FeatureUnion
from hulearn.preprocessing import InteractivePreprocessor, PipeTransformer

from hulearn.common import flatten

from tests.conftest import (
    select_tests,
    general_checks,
    nonmeta_checks,
)


@pytest.mark.parametrize(
    "test_fn",
    select_tests(
        include=flatten([general_checks, nonmeta_checks]),
        exclude=[
            "check_estimators_pickle",
            "check_estimators_nan_inf",
            "check_estimators_empty_data_messages",
            "check_complex_data",
            "check_dtype_object",
            "check_estimators_dtypes",
            "check_dict_unchanged",
            "check_fit1d",
            "check_methods_subset_invariance",
            "check_fit2d_predict1d",
        ],
    ),
)
def test_estimator_checks(test_fn):
    """
    We're skipping a lot of tests here mainly because this model is "bespoke"
    it is *not* general. Therefore a lot of assumptions are broken.
    """
    clf = InteractivePreprocessor.from_json("tests/test_classification/demo-data.json")
    test_fn(InteractivePreprocessor, clf)


def test_base_predict_usecase():
    clf = InteractivePreprocessor.from_json("tests/test_classification/demo-data.json")
    df = load_penguins(as_frame=True).dropna()
    X, y = df.drop(columns=["species"]), df["species"]

    preds = clf.fit(X, y).transform(X)

    assert preds.shape[0] == df.shape[0]
    assert preds.shape[1] == 3


def identity(x):
    return x


def test_grid_predict_usecase():
    tfm = InteractivePreprocessor.from_json("tests/test_classification/demo-data.json")
    pipe = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [("original", PipeTransformer(identity)), ("new_feats", tfm)]
                ),
            ),
        ]
    )
    df = load_penguins(as_frame=True).dropna()
    X, y = df.drop(columns=["species", "island", "sex"]), df["species"]

    preds = pipe.fit(X, y).transform(X)

    assert preds.shape[0] == df.shape[0]
    assert preds.shape[1] == X.shape[1] + 3


def test_ignore_bad_data():
    """
    There might be some "bad data" drawn. For example, when you quickly hit double-click you might
    draw a line instead of a poly. Bokeh is "okeh" with it, but our point-in-poly algorithm is not.
    """
    data = [
        {
            "chart_id": "9ec8e755-2",
            "x": "bill_length_mm",
            "y": "bill_depth_mm",
            "polygons": {
                "Adelie": {"bill_length_mm": [], "bill_depth_mm": []},
                "Gentoo": {"bill_length_mm": [], "bill_depth_mm": []},
                "Chinstrap": {"bill_length_mm": [], "bill_depth_mm": []},
            },
        },
        {
            "chart_id": "11640372-c",
            "x": "flipper_length_mm",
            "y": "body_mass_g",
            "polygons": {
                "Adelie": {
                    "flipper_length_mm": [[214.43261376806052, 256.2612913545137]],
                    "body_mass_g": [[3950.9482324534456, 3859.9137496948247]],
                },
                "Gentoo": {"flipper_length_mm": [], "body_mass_g": []},
                "Chinstrap": {"flipper_length_mm": [], "body_mass_g": []},
            },
        },
    ]

    clf = InteractivePreprocessor(json_desc=data)
    assert len(list(clf.poly_data)) == 0
