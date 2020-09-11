class _underscore:
    """
    The underscore object allows you to replace lambda functions with something that you can pickle.
    It's a lovely hack; we overwrite the `_` variable with a custom class.

    Important:
        This feature is experimental and unsupported

    **Base Usage:**

    ```python
    # This shows the general use-age.
    numbers = [1, 2, 3, 4]
    list(map(_ + 1, numbers)) # [2, 3, 4, 5]
    ```

    **Library Usage:**

    This example shows how you can use this feature with the other ones in this library.

    ```python
    # This is a use-case for this library. First we do the imports.
    import pandas as pd
    from sklearn.model_selection import GridSearchCV

    from hulearn.case_when import CaseWhen
    from hulearn.datasets import load_titanic
    from hulearn.classification import FunctionClassifier


    df = load_titanic(as_frame=True)
    X, y = df.drop(columns=['survived']), df['survived']

    # This is where we are using `_` instead of lambda/python functions.
    func = CaseWhen(default=0).when(_['sex'] == 'female', 1).when(_['pclass'] == 3, 1)

    # We can move on here with the actual modelling/gridsearch.
    mod = FunctionClassifier(func, pclass=3)
    grid = GridSearchCV(mod, cv=3, param_grid={'pclass': [1, 2, 3]}).fit(X, y)
    ```
    """

    def __init__(self, str_repr="_"):
        self.str_repr = str_repr

    def clean_str(self, n):
        return f'"{n}"' if isinstance(n, str) else n

    def __call__(self, _):
        return eval(self.str_repr)

    def __add__(self, n):
        str_repr = f"({self.str_repr}) + {n}"
        return _underscore(str_repr=str_repr)

    def __sub__(self, n):
        str_repr = f"({self.str_repr}) - {n}"
        return _underscore(str_repr=str_repr)

    def __mul__(self, n):
        str_repr = f"({self.str_repr}) * {n}"
        return _underscore(str_repr=str_repr)

    def __mod__(self, n):
        str_repr = f"({self.str_repr}) % {n}"
        return _underscore(str_repr=str_repr)

    def __eq__(self, n):
        str_repr = f"({self.str_repr}) == {self.clean_str(n)}"
        return _underscore(str_repr=str_repr)

    def __ne__(self, n):
        str_repr = f"({self.str_repr}) != {self.clean_str(n)}"
        return _underscore(str_repr=str_repr)

    def __ge__(self, n):
        str_repr = f"({self.str_repr}) >= {self.clean_str(n)}"
        return _underscore(str_repr=str_repr)

    def __gt__(self, n):
        str_repr = f"({self.str_repr}) > {self.clean_str(n)}"
        return _underscore(str_repr=str_repr)

    def __le__(self, n):
        str_repr = f"({self.str_repr}) <= {self.clean_str(n)}"
        return _underscore(str_repr=str_repr)

    def __lt__(self, n):
        str_repr = f"({self.str_repr}) < {self.clean_str(n)}"
        return _underscore(str_repr=str_repr)

    def __getitem__(self, n):
        str_repr = f"({self.str_repr})[{self.clean_str(n)}]"
        return _underscore(str_repr=str_repr)

    def __truediv__(self, n):
        str_repr = f"({self.str_repr}) / {self.clean_str(n)}"
        return _underscore(str_repr=str_repr)

    def __floordiv__(self, n):
        str_repr = f"({self.str_repr}) // {self.clean_str(n)}"
        return _underscore(str_repr=str_repr)

    def __abs__(self, n):
        str_repr = f"abs({self.str_repr})"
        return _underscore(str_repr=str_repr)

    def __int__(self, n):
        str_repr = f"int({self.str_repr})"
        return _underscore(str_repr=str_repr)

    def __float__(self, n):
        str_repr = f"float({self.str_repr})"
        return _underscore(str_repr=str_repr)

    def __round__(self, n):
        str_repr = f"round({self.str_repr})"
        return _underscore(str_repr=str_repr)

    def __ceil__(self, n):
        str_repr = f"ceil({self.str_repr})"
        return _underscore(str_repr=str_repr)

    def __floor__(self, n):
        str_repr = f"floor({self.str_repr})"
        return _underscore(str_repr=str_repr)

    def __trunc__(self, n):
        str_repr = f"trunc({self.str_repr})"
        return _underscore(str_repr=str_repr)

    def __len__(self, n):
        str_repr = f"len({self.str_repr})"
        return _underscore(str_repr=str_repr)

    def __setitem__(self, k, v):
        str_repr = f"setattr({self.str_repr}, {k}, {v})"
        return _underscore(str_repr=str_repr)

    def __repr__(self):
        return f"<func: {self.str_repr}>"


_ = _underscore()

__all__ = ["_"]
