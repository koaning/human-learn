In python the most popular data analysis tool is pandas while the most popular
tool for making models is scikit-learn. We love the data wrangling tools of pandas
while we appreciate the benchmarking capability of scikit-learn.

The fact that these tools don't fully interact is slightly awkward. The data
going into the model has an big effect on the output.

So how might we more easily combine the two?

## Pipe

In pandas there's an amazing trick that you can do with the `.pipe` method. We'll
give a quick overview on how it works but if you're new to this idea you may appreciate
[this resource](https://calmcode.io/pandas-pipe/introduction.html) or
[this blogpost](https://tomaugspurger.github.io/method-chaining).

```python
from hulearn.datasets import load_titanic

df = load_titanic(as_frame=True)
X, y = df.drop(columns=['survived']), df['survived']
X.head(4)
```

The goal of the titanic dataset is to predict weather or not a passenger survived
the disaster. The `X` variable represents a dataframe with variables that we're going to use
to predict survival (stored in `y`). Here's a preview of what `X` might have.

|   pclass | name                    | sex    |   age |   fare |   sibsp |   parch |
|---------:|:------------------------|:-------|------:|-------:|--------:|--------:|
|        3 | Braund, Mr. Owen Harris | male   |    22 |   7.25 |       1 |       0 |
|        3 | Heikkinen, Miss. Laina  | female |    26 |  7.925 |       0 |       0 |
|        3 | Allen, Mr. William Henry| male   |    35 |  8.05  |       0 |       0 |
|        1 | McCarthy, Mr. Timothy J | male   |    54 | 51.8625|       0 |       0 |

Let's say we want to do some preprocessing. Maybe the length of name of somebody
says something about their status so we'd like to capture that. We could add this
feature with this line of code.

```python
X['nchar'] = X['name'].str.len()
```

This line of code has downsides though. It changes the original dataset. If we do
a lot of this then our code is going to turn into something unmaintainable rather
quickly. To prevent this, we might want to change the code into a function.

```python
def process(dataf):
    # Make a copy of the dataframe to prevent it from overwriting the original data.
    dataf = dataf.copy()
    # Make the changes
    dataf['nchar'] = dataf['name'].str.len()
    # Return the name dataframe
    return dataf
```

We now have a nice function that makes our changes and we can use it like so;

```python
X_new = process(X)
```

We can do something more powerful though.

### Paramaters

Let's make some more changes to our `process` function.

```python
def preprocessing(dataf, n_char=True, gender=True):
    dataf = dataf.copy()
    if n_char:
        dataf['nchar'] = dataf['name'].str.len()
    if gender:
        dataf['gender'] = (dataf['sex'] == 'male').astype("float")
    return dataf.drop(columns=["name", "sex"])
```

This function works slightly differently now. The most important part is that the
function now accepts arguments that change the way it behaves internally. The function
also drops the non-numeric columns at the end.

We've changed the way we've defined our function but we're also changing the way
that we're going to apply it.

```python
# This is equivalent to preprocessing(X)
X.pipe(preprocessing)
```

The benefit of this notation is that if we have more functions that handle
data processing that it would remain a clean overview.

### With `.pipe()`

```python
(df
  .pipe(set_col_types)
  .pipe(preprocessing, nchar=True, gender=False)
  .pipe(add_time_info))
```

### Without `.pipe()`

```python
add_time_info(preprocessing(set_col_types(df), nchar=True, gender=False))
```

Let's be honest, this looks messy.

## PipeTransformer

It would be great if we could use the `preprocessing`-function as part of a
scikit-learn pipeline that we can benchmark. It'd be great if we could use
a function with a pandas `.pipe`-line in general!

For that we've got another feature in our library, the `PipeTransformer`.

```python
from hulearn.preprocessing import PipeTransformer

def preprocessing(dataf, n_char=True, gender=True):
    dataf = dataf.copy()
    if n_char:
        dataf['nchar'] = dataf['name'].str.len()
    if gender:
        dataf['gender'] = (dataf['sex'] == 'male').astype("float")
    return dataf.drop(columns=["name", "sex"])

# Important, don't forget to declare `n_char` and `gender` here.
tfm = PipeTransformer(preprocessing, n_char=True, gender=True))
```

The `tfm` variable now represents a component that can be used in a scikit-learn
pipeline. We can also perform a cross-validated benchmark on the parameters our
preprocessing function.

```python
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV


pipe = Pipeline([
    ('prep', tfm),
    ('mod', GaussianNB())
])

params = {
    "prep__n_char": [True, False],
    "prep__gender": [True, False]
}

grid = GridSearchCV(pipe, cv=3, param_grid=params).fit(X, y)
```

Once trained we can fetch the `grid.cv_results_` to get a glimpse at
the results of our pipeline.

| param_prep__gender   | param_prep__n_char   |   mean_test_score |
|:---------------------|:---------------------|------------------:|
| True                 | True                 |          0.785714 |
| True                 | False                |          0.778711 |
| False                | True                 |          0.70028  |
| False                | False                |          0.67507  |

It seems that we gender of the passenger has more of an effect on their
survival than the length of their name.

## Utility

The use-case here has been a relatively simple demonstration on a toy
dataset but hopefully you can recognize that this opens up a lot of
flexibility for your machine learning pipelines. You can keep the
preprocessing interpretable but you can keep everything running by
just writing pandas code.

There's a few small caveats to be aware of.

### Don't remove data

Pandas pipelines allow you to filter away rows, scikit-learn on the
other hand assumes this does not happen. Please be mindful of this.

### Don't sort data

You need to keep the order in your dataframe the same because otherwise
it will no longer correspond to the `y` variable that you're trying to
predict.

### Don't use `lambda`

There's two ways that you can add a new column to pandas.

```python
# Method 1
dataf_new = dataf.copy() # Don't overwrite data!
dataf_new['new_column'] = dataf_new['old_column'] * 2

# Method 2
dataf_new = dataf.assign(lambda d: d['old_column'] * 2)
```

In many cases you might argue that method #2 is safer because you
do not need to worry about the `dataf.copy()` that needs to happen.
In our case however, we cannot use it. The grid-search no longer works
inside of scikit-learn if you use `lambda` functions because it cannot
pickle the code.

### Don't Cheat!

The functions that you write are supposed to be stateless in the sense
that they don't learn from the data that goes in. You could theoretically
bypass this with global variables but by doing so you're doing yourself
a disservice. If you do this you'll be cheating the statistics by leaking
information.
