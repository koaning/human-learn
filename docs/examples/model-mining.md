In this example, we will demonstrate that you can use visual data mining techniques to discover meaningful patterns in your data. These patterns can be easily translated into a machine learning model by using the tools found in this package. 

You can find a full tutorial of this technique on [calmcode](https://calmcode.io/model-mining/introduction.html) but the main video can be viewed below.

<iframe src="https://player.vimeo.com/video/553884204" width="100%" height="564" frameborder="0" allow="autoplay; fullscreen" allowfullscreen style="display: inline-block;"></iframe>

## The Task 

We're going to make a rule based model for the creditcard dataset. The main feature of the dataset is that it is suffering from a class imbalance. Instead of training a machine learning model, let's try to instead explore it with a parallel coordinates chart. If you scroll all the way to the bottom of this tutorial you'll see an example of such a chart. It shows a "train"-set.

We explored the data just like in the video and that led us to define the following model. 

```python
from hulearn.classification import FunctionClassifier
from hulearn.experimental import CaseWhenRuler

def make_prediction(dataf, age=15):
    ruler = CaseWhenRuler(default=0)

    (ruler
     .add_rule(lambda d: (d['V11'] > 4), 1)
     .add_rule(lambda d: (d['V17'] < -3), 1)
     .add_rule(lambda d: (d['V14'] < -8), 1))

    return ruler.predict(dataf)

clf = FunctionClassifier(make_prediction)
```

??? "Full Code"
    First we load the data.

    ```python
    from sklearn.datasets import fetch_openml
    from sklearn.model_selection import train_test_split

    df_credit = fetch_openml(
        data_id=1597,
        as_frame=True
    )

    credit_train, credit_test = train_test_split(df_credit, test_size=0.5, shuffle=True)
    ```

    Next, we create a hiplot in jupyter.

    ```python
    import json 
    import hiplot as hip

    samples = [credit_train.loc[lambda d: d['group'] == True], credit_train.sample(5000)]
    json_data = pd.concat(samples).to_json(orient='records')

    hip.Experiment.from_iterable(json.loads(json_data)).display()
    ```

    Given that we have our model, we can make a classification report. 

    ```python
    from sklearn.metrics import classification_report

    # Note that `fit` is a no-op here.
    preds = clf.fit(credit_train, credit_train['group']).predict(credit_test))
    print(classification_report(credit_test['group'], preds)
    ```
    

When we ran the benchmark locally, we got the following classification report.

```
              precision    recall  f1-score   support

       False       1.00      1.00      1.00    142165
        True       0.70      0.73      0.71       239

    accuracy                           1.00    142404
   macro avg       0.85      0.86      0.86    142404
weighted avg       1.00      1.00      1.00    142404
```

## Deep Learning 

It's not a perfect benchmark, but we could compare this result to the one that's demonstrated 
on the [keras blog](https://keras.io/examples/structured_data/imbalanced_classification/). The 
trained model there lists 86.67% precision but only 23.9% recall. Depending on your preferences
for false-positives, you could argue that our model is outperforming the deep learning model. 

It's not 100% a fair comparison. You can imagine that the keras blogpost is written to explain 
keras. The auther likely didn't attempt to make a state-of-the-art model. But what this demo does
show is the merit of turning an exploratory data analysis into a model. You can end up with a 
very interpretable model, you might learn something about your data along the way and the model
might certainly still perform well.

## Parallel Coordinates 

If you hover of the `group` name and right-click, you'll be able to set it for coloring and 
repeat the experience in the video. By doing that it becomes quite easy to eyeball how to separate the two classes. The `V17` column especially seems powerful here. In real life we might ask "why?" this column is so distinctive but for now we'll just play around until we find a sensible model. 

<iframe src="creditcard.html" width="100%" height=1200 frameborder="0">