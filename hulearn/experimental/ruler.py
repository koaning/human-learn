import pandas as pd


class CaseWhenRuler:
    """
    Helper class to construct "case when"-style FunctionClassifiers.

    This class allows you to write a system of rules using lambda functions.
    These functions cannot be pickled by scikit-learn however, so if you'd like
    to use this class in a GridSearch you will need to wrap it around a
    FunctionClassifier.

    Arguments:
        default: the default value to predict if no rules apply

    Usage:

    ```python
    from hulearn.datasets import load_titanic
    from hulearn.experimental import CaseWhenRuler
    from hulearn.classification import FunctionClassifier


    def make_prediction(dataf, age=15):
        ruler = CaseWhenRuler(default=0)

        (ruler
         .add_rule(lambda d: (d['pclass'] < 3.0) & (d['sex'] == "female"), 1, name="gender-rule")
         .add_rule(lambda d: (d['pclass'] < 3.0) & (d['age'] <= age), 1, name="child-rule"))

        return ruler.predict(dataf)

    clf = FunctionClassifier(make_prediction)
    ```
    """

    def __init__(self, default=None):
        self.default = default
        self.rules = []

    def add_rule(self, when, then, name=None):
        """
        Adds a rule to the system.

        Arguments:
            when: a (lambda) function that tells us when the rule applies
            then: the value to output if the rule applies
            name: an optional name for the rule
        """
        if not name:
            name = f"rule-{len(self.rules) + 1}"
        self.rules.append((when, then, name))
        return self

    def predict(self, X):
        """
        Makes a prediction based on the rules sofar.

        Usage:

        ```python
        from hulearn.classification import FunctionClassifier
        from hulearn.experimental import CaseWhenRuler

        def make_prediction(dataf, gender_rule=True, child_rule=True, fare_rule=True):
            ruler = CaseWhenRuler(default=0)

            if gender_rule:
                ruler.add_rule(when=lambda d: (d['pclass'] < 3.0) & (d['sex'] == "female"),
                               then=1,
                               name="gender-rule")

            if child_rule:
                ruler.add_rule(when=lambda d: (d['pclass'] < 3.0) & (d['age'] <= 15),
                               then=1,
                               name="child-rule")

            if fare_rule:
                ruler.add_rule(when=lambda d: (d['fare'] > 100),
                               then=1,
                               name="fare-rule")

            return ruler.transform(dataf)

        clf = FunctionClassifier(make_prediction)
        ```
        """
        results = [self.default for x in range(len(X))]
        for rule in self.rules:
            when, then, name = rule
            for idx, predicate in enumerate(when(X)):
                if predicate and (results[idx] == self.default):
                    results[idx] = then
        return results

    def transform(self, X):
        """
        Produces a dataframe that indicates the state of all rules.

        Usage:

        ```python
        from hulearn.preprocessing import PipeTransformer
        from hulearn.experimental import CaseWhenRuler

        def make_prediction(dataf, gender_rule=True, child_rule=True, fare_rule=True):
            ruler = CaseWhenRuler(default=0)

            if gender_rule:
                ruler.add_rule(when=lambda d: (d['pclass'] < 3.0) & (d['sex'] == "female"),
                               then=1,
                               name="gender-rule")

            if child_rule:
                ruler.add_rule(when=lambda d: (d['pclass'] < 3.0) & (d['age'] <= 15),
                               then=1,
                               name="child-rule")

            if fare_rule:
                ruler.add_rule(when=lambda d: (d['fare'] > 100),
                               then=1,
                               name="fare-rule")

            return ruler.transform(dataf)

        clf = PipeTransformer(make_prediction)
        ```
        """
        result = pd.DataFrame()
        for rule in self.rules:
            when, then, name = rule
            result[name] = when(X)
        return result
