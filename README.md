<img src="docs/logo.png" width=225 align="right">

# Human Learning

> Machine Learning models should play by the rules, literally.

This package contains scikit-learn compatible tools that should make it easier
to construct and benchmark rule based systems that are designed by humans.

## Install

You can install this tool via `pip`.

```python
python -m pip install human-learn
```

## Features

This library hosts a couple of models that you can play with.

### Classification Models

#### `FunctionClassifier`

This allows you to define a function that can make classification predictions. It's
constructed in such a way that you can use the arguments of the function as a parameter
that you can benchmark in a grid-search.

#### `InteractiveClassifier`

This allows you to draw decision boundaries in interactive charts to create a
model. You can create charts interactively in the notebook and export it as a
scikit-learn compatible model.

### Regression Models

#### `FunctionRegressor`

This allows you to define a function that can make regression predictions. It's
constructed in such a way that you can use the arguments of the function as a parameter
that you can benchmark in a grid-search.


### Preprocessing Models

#### `PipeTransformer`

This allows you to define a function that can make handle preprocessing. It's
constructed in such a way that you can use the arguments of the function as a parameter
that you can benchmark in a grid-search.

This is especially powerful in combination with the pandas `.pipe` method. If you're
unfamiliar with this amazing feature, you may appreciate [this tutorial](https://calmcode.io/pandas-pipe/introduction.html).

### Datasets

#### `load_titanic`

Loads in the popular titanic survivor dataset. The goal of this dataset is to predict
who might have survived the titanic disaster.
