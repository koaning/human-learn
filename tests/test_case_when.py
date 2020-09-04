import pandas as pd
from sklearn.model_selection import GridSearchCV

from hulearn.case_when import CaseWhen
from hulearn.datasets import load_titanic
from hulearn.classification import FunctionClassifier


def women_only(d):
    return d['sex'] == 'female'


def pclass_high(d, pclass):
    return d['pclass'] == pclass


def test_case_when_func_classifier():
    df = load_titanic(as_frame=True)
    X, y = df.drop(columns=['survived']), df['survived']

    func = CaseWhen(default=0).when(women_only, 1).when(pclass_high, 1)
    mod = FunctionClassifier(func, pclass=3)
    grid = GridSearchCV(mod, cv=3, param_grid={'pclass': [1, 2, 3]}).fit(X, y)
    assert pd.DataFrame(grid.cv_results_).shape[0] == 3