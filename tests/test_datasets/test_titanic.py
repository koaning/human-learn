from hulearn.datasets import load_titanic


def test_load_titanic():
    df = load_titanic(as_frame=True)
    X, y = df.drop(columns=['survived']), df['survived']
    assert X.shape[0] == y.shape[0]
