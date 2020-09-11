<img src="logo.png" width=225 align="right">

# Human Learn

> Machine Learning models should play by the rules, literally. 

<br>

This package contains scikit-learn compatible tools that should make it easier
to construct and benchmark rule based systems that are designed by humans.

## Features 

This library hosts a couple of models that you can play with.


### Classification Models 

#### `FunctionClassifier`

This allows you to define a function that can make classification predictions. It's 
constructed in such a way that you can use the arguments of the function as a parameter
that you can benchmark in a grid-search.

### Regression Models

#### `FunctionRegressor`

This allows you to define a function that can make regression predictions. It's 
constructed in such a way that you can use the arguments of the function as a parameter
that you can benchmark in a grid-search.


### Preprocessing Models

#### `PipeTransformer`

This allows you to define a function that can make handle preprocessing. It's 
constructed in such a way that you can use the arguments of the function as a parameter
that you can benchmark in a grid-search. This is especially powerful in combination
with the pandas `.pipe` method. If you're unfamiliar with this amazing feature, yo may appreciate
[this tutorial](https://calmcode.io/pandas-pipe/introduction.html). 

### Datasets 

#### `load_titanic`

Loads in the popular titanic survivor dataset. The goal of this dataset is to predict 
who might have survived the titanic disaster.