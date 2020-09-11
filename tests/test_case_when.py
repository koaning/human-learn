# We don't support this yet and are unsure if we really want it.
#
# import pandas as pd
# from sklearn.model_selection import GridSearchCV
#
# from hulearn.case_when import CaseWhen
# from hulearn.datasets import load_titanic
# from hulearn.underscore import _
# from hulearn.classification import FunctionClassifier
#
#
# def women_only(d):
#     return d['sex'] == 'female'
#
#
# def pclass_high(d, pclass=3):
#     return d['pclass'] == pclass
#
#
# def case_when_func(d, pclass):
#     func = CaseWhen(default=0).when(_['sex'] == 'female', 1).when(_['pclass'] == pclass, 1)
#     return func(d)
#
#
# def test_case_when_func_classifier():
#     df = load_titanic(as_frame=True)
#     X, y = df.drop(columns=['survived']), df['survived']
#
#     mod = FunctionClassifier(case_when_func)
#     grid = GridSearchCV(mod, cv=3, param_grid={'pclass': [1, 2, 3]}).fit(X, y)
#     assert pd.DataFrame(grid.cv_results_).shape[0] == 3
#
#
# def test_case_when_func_classifier_grid():
#     df = load_titanic(as_frame=True)
#     X, y = df.drop(columns=['survived']), df['survived']
#
#     mod = FunctionClassifier(case_when_func, pclass=3)
#     grid = GridSearchCV(mod, cv=3, param_grid={}).fit(X, y)
#     assert pd.DataFrame(grid.cv_results_).shape[0] == 1
#
