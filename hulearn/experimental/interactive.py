import json
import pathlib

import matplotlib.pyplot as plt
from wigglystuff import ChartMultiSelect, ParallelCoordinates


class InteractiveCharts:
    """
    This tool allows you to interactively "draw" a model.

    Arguments:
        dataf: the dataframe to make a single interactive chart for
        labels: the labels to be drawn, if `str` we assume a column from the dataframe is chosen, if `list` we
        assume that the labels are not in the dataset
        color: you can manually override the color of the dots to be determined by a column in a dataframe.
          This setting is useful when you want to input a list of labels but still want to color
          the dots based on a column value.

    Usage:

    ```python
    from sklego.datasets import load_penguins
    from hulearn.experimental.interactive import InteractiveCharts

    df = load_penguins(as_frame=True)
    charts = InteractiveCharts(df, labels="species")
    ```
    """

    def __init__(self, dataf, labels, color=None):
        self.dataf = dataf
        self.labels = labels
        self.color = color
        self.charts = []
        self._widgets = []

        if isinstance(labels, str):
            self._label_list = list(dataf[labels].unique())
        else:
            self._label_list = list(labels)

    def add_chart(self, x, y, size=5, alpha=0.5, width=400, height=400, legend=True):
        """
        Generate an interactive chart to a cell.

        Arguments:
            x: the column from the dataset to place on the x-axis
            y: the column from the dataset to place on the y-axis
            size: the size of the drawn points
            alpha: the alpha (see-through-ness) of the drawn points
            width: the width of the chart
            height: the height of the chart
            legend: show a legend as well

        Usage:

        ```python
        from sklego.datasets import load_penguins
        from hulearn.experimental.interactive import InteractiveCharts

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
        color_col = self.labels if isinstance(self.labels, str) else self.color
        fig, ax = plt.subplots()
        if color_col:
            for label in self._label_list:
                subset = self.dataf[self.dataf[color_col] == label]
                ax.scatter(subset[x], subset[y], label=label, s=size, alpha=alpha)
            if legend:
                ax.legend()
        else:
            ax.scatter(self.dataf[x], self.dataf[y], s=size, alpha=alpha)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"{x} vs. {y}")
        plt.close(fig)

        widget = ChartMultiSelect(fig, n_classes=len(self._label_list), mode="lasso")
        self.charts.append({"x": x, "y": y, "widget": widget})
        self._widgets.append(widget)
        return widget

    def _convert_selections(self, chart_info):
        """Convert ChartMultiSelect.selections to json_desc polygon format."""
        x_col, y_col = chart_info["x"], chart_info["y"]
        widget = chart_info["widget"]
        selections = widget.selections if hasattr(widget, "selections") else []

        polygons = {lab: {x_col: [], y_col: []} for lab in self._label_list}

        for sel in selections:
            class_id = sel.get("class_id", 0)
            if class_id >= len(self._label_list):
                continue
            label = self._label_list[class_id]
            sel_type = sel.get("type", "lasso")

            if sel_type == "lasso":
                vertices = sel.get("vertices", [])
                if len(vertices) < 3:
                    continue
                x_coords = [v[0] for v in vertices]
                y_coords = [v[1] for v in vertices]
            elif sel_type == "box":
                x1 = sel.get("x_min", 0)
                y1 = sel.get("y_min", 0)
                x2 = sel.get("x_max", 0)
                y2 = sel.get("y_max", 0)
                x_coords = [x1, x2, x2, x1, x1]
                y_coords = [y1, y1, y2, y2, y1]
            else:
                continue

            if x_coords and y_coords:
                polygons[label][x_col].append(x_coords)
                polygons[label][y_col].append(y_coords)

        return polygons

    def data(self):
        """Returns chart data as list of dicts in json_desc format."""
        result = []
        for i, chart_info in enumerate(self.charts):
            polygons = self._convert_selections(chart_info)
            result.append(
                {
                    "chart_id": f"chart-{i}",
                    "x": chart_info["x"],
                    "y": chart_info["y"],
                    "polygons": polygons,
                }
            )
        return result

    def to_json(self, path):
        """Save chart data to a JSON file."""
        pathlib.Path(path).write_text(json.dumps(self.data(), indent=2))


def parallel_coordinates(dataf, label, height=200, width=0):
    """
    Creates an interactive parallel coordinates chart to help with classification tasks.

    Arguments:
        dataf: the dataframe to render
        label: the column that represents the label, will be used for coloring
        height: the height of the chart, in pixels
        width: the width of the chart, in pixels (0 for auto)

    Usage:

    ```python
    from hulearn.datasets import load_titanic
    from hulearn.experimental.interactive import parallel_coordinates

    df = load_titanic(as_frame=True)
    parallel_coordinates(df, label="survived", height=200)
    ```
    """
    return ParallelCoordinates(data=dataf, color_by=label, height=height, width=width)
