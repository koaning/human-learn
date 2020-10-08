from hulearn.datasets import load_fish


def test_load_fish():
    df = load_fish(as_frame=True)
    X, y = df.drop(columns=["Weight"]), df["Weight"]
    assert X.shape[0] == y.shape[0]
