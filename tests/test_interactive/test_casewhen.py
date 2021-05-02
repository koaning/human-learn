import pandas as pd

from hulearn.datasets import load_titanic
from hulearn.classification import FunctionClassifier
from hulearn.experimental import CaseWhenRuler


def test_smoke_casewhen():
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import (
        make_scorer,
        accuracy_score,
        precision_score,
        recall_score,
    )

    def make_prediction(dataf, gender_rule=True, child_rule=True, fare_rule=True):
        ruler = CaseWhenRuler(default=0)

        if gender_rule:
            ruler.add_rule(
                when=lambda d: (d["pclass"] < 3.0) & (d["sex"] == "female"),
                then=1,
                name="gender-rule",
            )

        if child_rule:
            ruler.add_rule(
                when=lambda d: (d["pclass"] < 3.0) & (d["age"] <= 15),
                then=1,
                name="child-rule",
            )

        if fare_rule:
            ruler.add_rule(when=lambda d: (d["fare"] > 100), then=1, name="fare-rule")

        return ruler.predict(dataf)

    df = load_titanic(as_frame=True)
    X, y = df.drop(columns=["survived"]), df["survived"]

    clf = FunctionClassifier(make_prediction)

    cv = GridSearchCV(
        clf,
        cv=10,
        param_grid={
            "gender_rule": [True, False],
            "child_rule": [True, False],
            "fare_rule": [True, False],
        },
        scoring={
            "accuracy": make_scorer(accuracy_score),
            "precision": make_scorer(precision_score),
            "recall": make_scorer(recall_score),
        },
        refit="accuracy",
    )

    res = pd.DataFrame(cv.fit(X, y).cv_results_)[
        [
            "param_child_rule",
            "param_fare_rule",
            "param_gender_rule",
            "mean_test_accuracy",
            "mean_test_precision",
            "mean_test_recall",
        ]
    ]

    assert res.shape[0] == 8
