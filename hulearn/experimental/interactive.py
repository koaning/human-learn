import uuid
import json
import pathlib
from pkg_resources import resource_filename

import numpy as np
from clumper import Clumper
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.models import PolyDrawTool, PolyEditTool
from bokeh.layouts import row
from bokeh.models import Label
from bokeh.models.widgets import Div
from bokeh.io import output_notebook

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


def color_dot(name, color):
    dot = f"<span style='height: 15px; width: 15px; background-color: {color}; border-radius: 50%; display: inline-block;'></span>"
    return f"<p>{dot} {name}</p>"


class InteractiveClassifierCharts:
    """
    This tool allows you to interactively "draw" a model.

    Usage:

    ```python
    from sklego.datasets import load_penguins
    from hulearn.experimental.interactive import InteractiveClassifierCharts

    df = load_penguins(as_frame=True)
    charts = InteractiveClassifierCharts(df, labels="species")

    # Next notebook cell
    charts.add_chart(x="bill_length_mm", y="bill_depth_mm")
    # Next notebook cell
    charts.add_chart(x="flipper_length_mm", y="body_mass_g")

    # After drawing a model, export the data
    json_data = charts.data()

    # You can now use your drawn intuition as a model!
    from hulearn.experimental.interactive import HumanClassifier
    clf = HumanClassifier(clf_data)
    X, y = df.drop(columns=['species']), df['species']

    # This doesn't do anything. But scikit-learn demands it.
    clf.fit(X, y)

    # This makes predictions, based on your drawn model.
    # It can also be used in `GridSearchCV` for benchmarking!
    clf.predict(X)
    ```
    """

    def __init__(self, dataf, labels):
        output_notebook()
        self.dataf = dataf
        self.labels = labels
        self.charts = []

    def add_chart(self, x, y):
        """
        Generate an interactive chart to a cell.

        The supported actions include:

        - Add patch or multi-line: Double tap to add the first vertex, then use tap to add each subsequent vertex,
        to finalize the draw action double tap to insert the final vertex or press the <<esc> key.
        - Move patch or ulti-line: Tap and drag an existing patch/multi-line, the point will be dropped once you let go of the mouse button.
        - Delete patch or multi-line: Tap a patch/multi-line to select it then press <<backspace>> key while the mouse is within the plot area.
        """
        chart = InteractiveChart(dataf=self.dataf.copy(), labels=self.labels, x=x, y=y)
        self.charts.append(chart)
        chart.show()

    def data(self):
        return [c.data for c in self.charts]

    def to_json(self, path):
        return Clumper(self.data).write_json(path, indent=2)


class InteractiveChart:
    def __init__(self, dataf, labels, x, y):
        self.uuid = str(uuid.uuid4())[:10]
        self.x = x
        self.y = y
        self.plot = figure(width=400, height=400, title=f"{x} vs. {y}")
        self._colors = ["red", "blue", "green", "purple", "cyan"]

        if isinstance(labels, str):
            self.labels = list(dataf[labels].unique())
            d = {k: col for k, col in zip(self.labels, self._colors)}
            dataf = dataf.assign(color=[d[lab] for lab in dataf[labels]])
            self.source = ColumnDataSource(data=dataf)
        else:
            dataf = dataf.assign(color=["gray" for _ in range(dataf.shape[0])])
            self.source = ColumnDataSource(data=dataf)
            self.labels = labels

        if len(self.labels) > 5:
            raise ValueError("We currently only allow for 5 classes max.")
        self.plot.circle(x=x, y=y, color="color", source=self.source)

        # Create all the tools for drawing
        self.poly_patches = {}
        self.poly_draw = {}
        for k, col in zip(self.labels, self._colors):
            self.poly_patches[k] = self.plot.patches(
                [], [], fill_color=col, fill_alpha=0.4, line_alpha=0.0
            )
            icon_path = resource_filename("hulearn", f"images/{col}.png")
            self.poly_draw[k] = PolyDrawTool(
                renderers=[self.poly_patches[k]], custom_icon=icon_path
            )
        c = self.plot.circle([], [], size=5, color="black")
        edit_tool = PolyEditTool(
            renderers=list(self.poly_patches.values()), vertex_renderer=c
        )
        self.plot.add_tools(*self.poly_draw.values(), edit_tool)
        self.plot.add_layout(Label(x=70, y=70, text="here your text"))

    def app(self, doc):
        html = "<ul style='width:100px'>"
        for k, col in zip(self.labels, self._colors):
            html += f"<li>{color_dot(name=k, color=col)}</li>"
        html += "</ul>"
        doc.add_root(row(Div(text=html), self.plot))

    def show(self):
        show(self.app)

    def _replace_xy(self, data):
        new_data = {}
        new_data[self.x] = data["xs"]
        new_data[self.y] = data["ys"]
        return new_data

    @property
    def data(self):
        return {
            "chart_id": self.uuid,
            "x": self.x,
            "y": self.y,
            "polygons": {
                k: self._replace_xy(v.data_source.data)
                for k, v in self.poly_patches.items()
            },
        }


class HumanClassifier(BaseEstimator, ClassifierMixin):
    """
    This tool allows you to take a drawn model and use it as a classifier.

    Usage:

    ```python
    from sklego.datasets import load_penguins
    from hulearn.experimental.interactive import InteractiveClassifierCharts

    df = load_penguins(as_frame=True)
    charts = InteractiveClassifierCharts(df, labels="species")

    # Next notebook cell
    charts.add_chart(x="bill_length_mm", y="bill_depth_mm")
    # Next notebook cell
    charts.add_chart(x="flipper_length_mm", y="body_mass_g")

    # After drawing a model, export the data
    json_data = charts.data()

    # You can now use your drawn intuition as a model!
    from hulearn.experimental.interactive import HumanClassifier
    clf = HumanClassifier(clf_data)
    X, y = df.drop(columns=['species']), df['species']

    # This doesn't do anything. But scikit-learn demands it.
    clf.fit(X, y)

    # This makes predictions, based on your drawn model.
    # It can also be used in `GridSearchCV` for benchmarking!
    clf.predict(X)
    ```
    """

    def __init__(self, json_desc, smoothing=0.001):
        self.json_desc = json_desc
        self.smoothing = smoothing

    @classmethod
    def from_json(cls, path):
        json_desc = json.loads(pathlib.Path(path).read_text())
        return HumanClassifier(json_desc=json_desc)

    def _clean_poly_data(self, json_desc):
        """TODO: we need to prevent poly data with only two datapoints"""
        return json_desc

    @property
    def poly_data(self):
        for chart in self.json_desc:
            chard_id = chart["chart_id"]
            labels = chart["polygons"].keys()
            coords = chart["polygons"].values()
            for lab, p in zip(labels, coords):
                x_lab, y_lab = p.keys()
                x_coords, y_coords = list(p.values())
                for poly in [
                    Polygon(list(zip(x_coords[i], y_coords[i])))
                    for i in range(len(x_coords))
                ]:
                    yield {
                        "x_lab": x_lab,
                        "y_lab": y_lab,
                        "poly": poly,
                        "label": lab,
                        "chart_id": chard_id,
                    }

    def _count_hits(self, clf_data, data_in):
        counts = {k: 0 for k in self.classes_}
        for c in clf_data:
            point = Point(data_in[c["x_lab"]], data_in[c["y_lab"]])
            if c["poly"].contains(point):
                counts[c["label"]] += 1
        return counts

    def fit(self, X, y):
        self.classes_ = list(self.json_desc[0]["polygons"].keys())
        return self

    def predict_proba(self, X):
        hits = [self._count_hits(self.poly_data, x[1].to_dict()) for x in X.iterrows()]
        count_arr = (
            np.array([[h[c] for c in self.classes_] for h in hits]) + self.smoothing
        )
        return count_arr / count_arr.sum(axis=1).reshape(-1, 1)

    def predict(self, X):
        check_is_fitted(self, ["classes_"])
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
