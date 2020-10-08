import uuid
from pkg_resources import resource_filename

from clumper import Clumper
from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.models import PolyDrawTool, PolyEditTool
from bokeh.layouts import row
from bokeh.models.annotations import Label
from bokeh.models.widgets import Div
from bokeh.io import output_notebook


def color_dot(name, color):
    dot = f"<span style='height: 15px; width: 15px; background-color: {color}; border-radius: 50%; display: inline-block;'></span>"
    return f"<p>{dot} {name}</p>"


class InteractiveCharts:
    """
    This tool allows you to interactively "draw" a model.

    Usage:

    ```python
    from sklego.datasets import load_penguins
    from hulearn.classification.interactive import InteractiveCharts

    df = load_penguins(as_frame=True)
    charts = InteractiveCharts(df, labels="species")
    ```
    """

    def __init__(self, dataf, labels):
        output_notebook()
        self.dataf = dataf
        self.labels = labels
        self.charts = []

    def add_chart(self, x, y, size=5, alpha=0.5):
        """
        Generate an interactive chart to a cell.

        The supported actions include:

        - Add patch or multi-line: Double tap to add the first vertex, then use tap to add each subsequent vertex,
        to finalize the draw action double tap to insert the final vertex or press the <<esc> key.
        - Move patch or ulti-line: Tap and drag an existing patch/multi-line, the point will be dropped once you let go of the mouse button.
        - Delete patch or multi-line: Tap a patch/multi-line to select it then press <<backspace>> key while the mouse is within the plot area.

        Arguments:
            x: the column from the dataset to place on the x-axis
            y: the column from the dataset to place on the y-axis

        Usage:

        ```python
        from sklego.datasets import load_penguins
        from hulearn.drawing-classifier.interactive import InteractiveCharts

        df = load_penguins(as_frame=True)
        charts = InteractiveCharts(df, labels="species")

        # Next notebook cell
        charts.add_chart(x="bill_length_mm", y="bill_depth_mm")
        # Next notebook cell
        charts.add_chart(x="flipper_length_mm", y="body_mass_g")

        # After drawing a model, export the data
        json_data = charts.data()
        ```
        """
        chart = SingleInteractiveChart(
            dataf=self.dataf.copy(),
            labels=self.labels,
            x=x,
            y=y,
            size=size,
            alpha=alpha,
        )
        self.charts.append(chart)
        chart.show()

    def data(self):
        return [c.data for c in self.charts]

    def to_json(self, path):
        return Clumper(self.data).write_json(path, indent=2)


class SingleInteractiveChart:
    def __init__(self, dataf, labels, x, y, size=5, alpha=0.5):
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
        self.plot.circle(
            x=x, y=y, color="color", source=self.source, size=size, alpha=alpha
        )

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
