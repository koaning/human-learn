As a final remark it might be good to discuss how you might use this library
in practice. The goal of this library is to add tools to the scikit-learn
ecosystem, not to suggest that you can replace them. We'll list a few examples
of when our tools might help your machine learning pipeline.

## Increase Precision

It might be the case that for a certain subset of your population you can
just apply a heuristic that is plenty accurate. If this is the case, you
can design your `FunctionClassifier` to use this information and to relay
all the "hard" cases to a more complex machine learning model.

```python

```
