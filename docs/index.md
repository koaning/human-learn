<img src="logo.png" width=225 align="right">

# Human Learn

> Machine Learning models should play by the rules, literally. 

<br>

This package contains scikit-learn compatible tools that should make it easier
to construct and benchmark rule based systems that are designed by humans.

## Features 

This library hosts a couple of models that you can play with.

### Classification Models 

- `FunctionClassifier`: define a function that can make classification predictions.

### Regression Models

- `FunctionRegressor`: define a function that can make regression predictions.

### Utility Tools 

- `hulearn.case_when.CaseWhen`: a utility to construct case-when trees for Function-models.
- `hulearn.underscore._`: a utility to construct lambda functions that you can pickle.

### Datasets 

- `load_titanic`: loads in the popular titanic survivor dataset.