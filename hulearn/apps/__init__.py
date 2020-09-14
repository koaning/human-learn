from bokeh.embed import json_item
import json

import os
from random import choices
from string import ascii_letters
import streamlit.components.v1 as components

_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "streamlit_bokeh_events",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "streamlit_bokeh_events", path=build_dir
    )


def streamlit_bokeh_events(bokeh_plot=None, events="", key=None, debounce_time=1000):
    """Returns event dict

    Keyword arguments:
    bokeh_plot -- Bokeh figure object (default None)
    events -- Comma separated list of events dispatched by bokeh eg. "event1,event2,event3" (default "")
    debounce_time -- Time in ms to wait before dispatching latest event (default 1000)
    """
    div_id = "".join(choices(ascii_letters, k=16))
    fig_dict = json_item(bokeh_plot, div_id)
    json_figure = json.dumps(fig_dict)
    component_value = _component_func(
        bokeh_plot=json_figure,
        events=events,
        key=key,
        _id=div_id,
        default=None,
        debounce_time=debounce_time,
    )
    return component_value


if not _RELEASE:
    import streamlit as st
    from bokeh.models import ColumnDataSource, CustomJS
    from bokeh.plotting import figure
    import pandas as pd
    import numpy as np

    df = pd.DataFrame(
        {
            "x": np.random.rand(500),
            "y": np.random.rand(500),
            "size": np.random.rand(500) * 10,
        }
    )

    source = ColumnDataSource(df)

    st.subheader("Select Points From Map")

    plot = figure(tools="lasso_select", width=250, height=250)
    plot.circle(x="x", y="y", size="size", source=source, alpha=0.6)

    source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(source=source),
            code="""
            document.dispatchEvent(
                new CustomEvent("TestSelectEvent", {detail: {indices: cb_obj.indices}})
            )
        """,
        ),
    )

    event_result = streamlit_bokeh_events(
        events="TestSelectEvent",
        bokeh_plot=plot,
        key="foo",
        debounce_time=1000,
    )

    # some event was thrown
    if event_result is not None:
        # TestSelectEvent was thrown
        if "TestSelectEvent" in event_result:
            st.subheader("Selected Points' Pandas Stat summary")
            indices = event_result["TestSelectEvent"].get("indices", [])
            st.table(df.iloc[indices].describe())

    st.subheader("Raw Event Data")
    st.write(event_result)
