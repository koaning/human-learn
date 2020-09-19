<img src="docs/logo.png" width=225 align="right">

# Human Learning

> Machine Learning models should play by the rules, literally.

## Project Goal

Back in the old days, it was common to write rule-based systems. Systems that do;

![](docs/examples/rules.png)

Nowadays, it's much more fashionable to use machine learning instead. Something like;

![](docs/examples/ml.png)

We started wondering if we might have lost something in this transition. Sure,
machine learning covers a lot of ground but it is also capable of making bad
decision. We've also reached a stage of hype that folks forget that many
classification problems can be handled by natural intelligence too.

This package contains scikit-learn compatible tools that should make it easier
to construct and benchmark rule based systems that are designed by humans. You
can also use it in combination with ML models.

## Install

You can install this tool via `pip`.

```python
python -m pip install human-learn
```

## Features

This library hosts a couple of models that you can play with.


### Classification Models

#### FunctionClassifier

This allows you to define a function that can make classification predictions. It's
constructed in such a way that you can use the arguments of the function as a parameter
that you can benchmark in a grid-search.

#### InteractiveClassifier

This allows you to draw decision boundaries in interactive charts to create a
model. You can create charts interactively in the notebook and export it as a
scikit-learn compatible model.

### Regression Models

#### FunctionRegressor

This allows you to define a function that can make regression predictions. It's
constructed in such a way that you can use the arguments of the function as a parameter
that you can benchmark in a grid-search.

### Outlier Detection Models

#### FunctionOutlierDetector

This allows you to define a function that can declare outliers. It's constructed in
such a way that you can use the arguments of the function as a parameter that you
can benchmark in a grid-search.

#### InteractiveOutlierDetector

This allows you to draw decision boundaries in interactive charts to create a
model. If a point falls outside of these boundaries we might be able to declare
it an outlier. There's a threshold parameter for how strict you might want to be.

### Preprocessing Models

#### PipeTransformer

This allows you to define a function that can make handle preprocessing. It's
constructed in such a way that you can use the arguments of the function as a parameter
that you can benchmark in a grid-search. This is especially powerful in combination
with the pandas `.pipe` method. If you're unfamiliar with this amazing feature, yo may appreciate
[this tutorial](https://calmcode.io/pandas-pipe/introduction.html).

### Datasets

#### Titanic

This library hosts the popular titanic survivor dataset for demo purposes. The goal of
this dataset is to predict who might have survived the titanic disaster.
