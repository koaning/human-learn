from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline


from sklego.datasets import load_penguins
from sklego.preprocessing import IdentityTransformer
from hulearn.experimental.interactive import HumanClassifier


def test_base_usecase():
    clf = HumanClassifier.from_json("tests/test_experimental/demo-data.json")
    df = load_penguins(as_frame=True).dropna()
    X, y = df.drop(columns=["species"]), df["species"]

    preds = clf.fit(X, y).predict_proba(X)

    assert preds.shape[0] == df.shape[0]
    assert preds.shape[1] == 3


def test_gridsearch_usecase():
    clf = HumanClassifier.from_json("tests/test_experimental/demo-data.json")
    pipe = Pipeline(
        [
            ("id", IdentityTransformer()),
            ("mod", clf),
        ]
    )
    grid = GridSearchCV(pipe, cv=5, param_grid={})
    df = load_penguins(as_frame=True).dropna()
    X, y = df.drop(columns=["species"]), df["species"]

    preds = grid.fit(X, y).predict_proba(X)

    assert preds.shape[0] == df.shape[0]
    assert preds.shape[1] == 3
