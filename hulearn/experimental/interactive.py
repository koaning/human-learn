from bokeh.models import ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.models import PolyDrawTool, PolyEditTool
from bokeh.layouts import row
from bokeh.models import Label
from bokeh.models.widgets import Div
from bokeh.io import output_notebook


def color_dot(name, color):
    dot = f"<span style='height: 15px; width: 15px; background-color: {color}; border-radius: 50%; display: inline-block;'></span>"
    return f"<p>{dot} {name}</p>"


class InteractiveClassifier:
    def __init__(self, dataf, labels):
        output_notebook()
        self.dataf = dataf
        if isinstance(labels, str):
            labels = dataf[labels].unique()
        self.labels = list(labels)
        self.charts = []

    def add_chart(self, x, y):
        chart = InteractiveChart(dataf=self.dataf.copy(), labels=self.labels, x=x, y=y)
        self.charts.append(chart)
        chart.show()

    def data(self):
        return [c.data for c in self.charts]


class InteractiveChart:
    def __init__(self, dataf, labels, x, y):
        self.plot = figure(width=400, height=400, title=f"{x} vs. {y}")
        self.source = ColumnDataSource(data=dataf)
        self._colors = ["red", "blue", "green", "purple", "cyan"]
        if isinstance(labels, str):
            self.labels = list(dataf[labels].unique())
            self.plot.circle(x=x, y=y, color="gray", source=self.source)
        else:
            self.labels = labels
            self.plot.circle(x=x, y=y, color="gray", source=self.source)
        if len(self.labels) > 5:
            raise ValueError("We currently only allow for 5 classes max.")
        self.poly_patches = {}
        self.poly_draw = {}
        for k, col in zip(self.labels, self._colors):
            self.poly_patches[k] = self.plot.patches(
                [], [], fill_color=col, fill_alpha=0.4, line_alpha=0.0
            )
            self.poly_draw[k] = PolyDrawTool(
                renderers=[self.poly_patches[k]], custom_icon=f"{col}.png"
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

    @property
    def data(self):
        return {k: v.data_source.data for k, v in self.poly_patches.items()}


# chart = InteractiveChart(df, labels="species", x="bill_length_mm", y="bill_depth_mm")
