from hulearn.datasets import load_titanic
from hulearn.experimental.interactive import parallel_coordinates


def test_smoke_parcoords():
    df = load_titanic(as_frame=True)
    chart = parallel_coordinates(df, label="survived", height=200)
    assert "d3" in chart.data
