This page contains a list of short examples that demonstrate the utility of
the tools in this package. The goal for each example is to be small and consise.

This page is still under construction.

## Insurance for Future Data

If you know a certain region needs to be a specific class, you shouldn't
need data to tell you what the label is going to be.

## No Data No Problem

## Model Guarantees

Let's say that we're dealing with a financial compliance use-case.

## Precision and Subgroups

## Dealing with NA

Missing data can throw off a lot of machine learning models. We are able
however to deal with it in different ways in production.

## Comfort Zone

If you want to prevent predictions where the model is "unsure" then you
might want to construct a `FunctionClassifier` that handles the logic you
require. For example, we might draw a diagram like;

![](../guide/finding-outliers/diagram.png)

As an illustrative example we'll implement a diagram like above as a `Classifier`.

```python
import numpy as np
from hulearn.outlier import InteractiveOutlierDetector
from hulearn.classification import FunctionClassifier, InteractiveClassifier

# We're importing a classifier/outlier detector from our library
# but nothing is stopping you from using those in scikit-learn.
# Just make sure that they are trained beforehand!
outlier    = InteractiveOutlierDetector.from_json("path/to/file.json")
classifier = InteractiveClassifier.from_json("path/to/file.json")

def make_decision(dataf):
    # First we create a resulting array with all the predictions
    res = classifier.predict(dataf)

    # If we detect doubt, "classify" it as a fallback instead.
    proba = classifier.predict_proba(dataf)
    res = np.where(proba.max(axis=1) < 0.8, "doubt_fallback", res)

    # If we detect an ourier, we'll fallback too.
    res = np.where(outlier.predict(dataf) == -1, "outlier_fallback", res)

    # This `res` array contains the output of the drawn diagram.
    return res

fallback_model = FunctionClassifier(make_decision)
```

For more information on why this tactic is helpful:

- [blogpost](https://koaning.io/posts/high-on-probability-low-on-certainty/)
- [pydata talk](https://www.youtube.com/watch?v=Z8MEFI7ZJlA)

## Risk Class Translation
