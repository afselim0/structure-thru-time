#!/usr/bin/env python3
"""
Dash app for exploring 2D particle trajectories stored in a Parquet file.

Features
- Slider over (binned) time
- Fixed equal-aspect axes
- Point colors by population (from labels+colors CSVs)
- Optional 3-letter population text labels, colored to match points
- Hover shows sample_id and population
- Buttons:
    * Fit axes (this frame)
    * Reset axes (global)

Input assumptions
- Parquet columns: time, sample_id, X1, X2
- labels CSV columns: sample_id, population
- colors CSV columns: population, color
"""

import numpy as np
import pandas as pd
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx


def load_and_pack(
    parquet_path: str,
    labels_path: str,
    colors_path: str,
    min_time: float = -np.inf,
    max_time: float = np.inf,
    time_step: float = 0.01,
    quantize_xy_to_int: bool = True,
):
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)

    time_col = "time"
    x_col = "X1"
    y_col = "X2"
    id_col = "sample_id"

    # Time filter
    df = df[(df[time_col] >= min_time) & (df[time_col] <= max_time)].copy()
    if df.empty:
        raise ValueError("No rows left after time filtering.")

    # Bin time to stable steps
    df["time_bin"] = (np.round(df[time_col].to_numpy(dtype=float) / time_step) * time_step).astype(float)

    times = np.sort(df["time_bin"].unique())
    T = len(times)

    # Stable sample order across all times
    sample_order = np.array(sorted(df[id_col].unique()))
    N = len(sample_order)

    # Map time/sample to integer indices
    t_codes = pd.Categorical(df["time_bin"], categories=times, ordered=True).codes
    s_codes = pd.Index(sample_order).get_indexer(df[id_col].to_numpy())

    # Pack coordinates into (T, N, 2)
    xy = np.full((T, N, 2), np.nan, dtype=np.float32)
    xy[t_codes, s_codes, 0] = df[x_col].to_numpy(dtype=np.float32)
    xy[t_codes, s_codes, 1] = df[y_col].to_numpy(dtype=np.float32)

    if quantize_xy_to_int:
        xy = np.rint(xy).astype(np.float32)  # keep NaNs

    # Load labels/colors
    labels_df = pd.read_csv(labels_path)   # sample_id, population
    colors_df = pd.read_csv(colors_path)   # population, color

    if id_col not in labels_df.columns or "population" not in labels_df.columns:
        raise ValueError("labels CSV must contain columns: sample_id, population")
    if "population" not in colors_df.columns or "color" not in colors_df.columns:
        raise ValueError("colors CSV must contain columns: population, color")

    # sample_id -> population
    pop_map = dict(zip(labels_df[id_col].astype(str), labels_df["population"].astype(str)))
    sample_ids = sample_order.astype(str)
    pops = np.array([pop_map.get(sid, "UNK") for sid in sample_ids], dtype=object)
    pop3 = np.array([p[:3].upper() if p else "" for p in pops], dtype=object)

    # population -> color
    pop_color_map = dict(zip(colors_df["population"].astype(str), colors_df["color"].astype(str)))
    colors = np.array([pop_color_map.get(p, "#808080") for p in pops], dtype=object)

    # Global axis limits (fixed), with padding
    xmin, xmax = np.nanmin(xy[:, :, 0]), np.nanmax(xy[:, :, 0])
    ymin, ymax = np.nanmin(xy[:, :, 1]), np.nanmax(xy[:, :, 1])
    dx, dy = float(xmax - xmin), float(ymax - ymin)
    pad = 0.03
    dx = dx if dx > 0 else 1.0
    dy = dy if dy > 0 else 1.0
    xlo, xhi = float(xmin - pad * dx), float(xmax + pad * dx)
    ylo, yhi = float(ymin - pad * dy), float(ymax + pad * dy)

    return dict(
        times=times,
        xy=xy,
        sample_ids=sample_ids,
        pop=pops,
        pop3=pop3,
        color=colors,
        xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
    )


def _to_list_none(a_1d: np.ndarray):
    # Convert NaN -> None for Plotly
    out = a_1d.tolist()
    return [None if (isinstance(v, float) and np.isnan(v)) else v for v in out]


def _fit_axes_to_frame(data: dict, frame_idx: int, pad_frac: float = 0.05):
    coords = data["xy"][frame_idx]
    x = coords[:, 0]
    y = coords[:, 1]
    mask = np.isfinite(x) & np.isfinite(y)
    if not np.any(mask):
        return {"xlo": data["xlo"], "xhi": data["xhi"], "ylo": data["ylo"], "yhi": data["yhi"]}

    xmin, xmax = float(np.min(x[mask])), float(np.max(x[mask]))
    ymin, ymax = float(np.min(y[mask])), float(np.max(y[mask]))
    dx = xmax - xmin
    dy = ymax - ymin
    dx = dx if dx > 0 else 1.0
    dy = dy if dy > 0 else 1.0
    return {
        "xlo": xmin - pad_frac * dx,
        "xhi": xmax + pad_frac * dx,
        "ylo": ymin - pad_frac * dy,
        "yhi": ymax + pad_frac * dy,
    }


def make_figure(
    frame_idx: int,
    data: dict,
    show_labels: bool = True,
    point_size: float = 8.0,
    alpha: float = 0.85,
    axis_override: dict | None = None,
):
    times = data["times"]
    xy = data["xy"]
    colors = data["color"]
    pops = data["pop"]
    pop3 = data["pop3"]
    sample_ids = data["sample_ids"]

    coords = xy[frame_idx]
    x = _to_list_none(coords[:, 0])
    y = _to_list_none(coords[:, 1])

    # Hover uses customdata
    customdata = np.column_stack([sample_ids, pops])

    # Axes
    if axis_override is None:
        xlo, xhi, ylo, yhi = data["xlo"], data["xhi"], data["ylo"], data["yhi"]
    else:
        xlo, xhi, ylo, yhi = axis_override["xlo"], axis_override["xhi"], axis_override["ylo"], axis_override["yhi"]

    t = times[frame_idx]

    traces = []

    # Markers: WebGL for speed
    traces.append(
        go.Scattergl(
            x=x,
            y=y,
            mode="markers",
            marker=dict(size=point_size, opacity=alpha, color=colors.tolist()),
            customdata=customdata,
            hovertemplate="sample_id: %{customdata[0]}<br>population: %{customdata[1]}<extra></extra>",
            showlegend=False,
        )
    )

    # Text labels: SVG Scatter (reliable per-point text colors)
    if show_labels:
        traces.append(
            go.Scatter(
                x=x,
                y=y,
                mode="text",
                text=pop3.tolist(),
                textposition="top center",
                textfont=dict(size=10, color=colors.tolist()),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    fig = go.Figure(data=traces)
    fig.update_layout(
        title=f"t = {t:.2f}",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(range=[xlo, xhi], autorange=False, constrain="range"),
        yaxis=dict(range=[ylo, yhi], autorange=False, scaleanchor="x", scaleratio=1, constrain="range"),
    )
    return fig


def make_app(data: dict):
    app = Dash(__name__)
    times = data["times"]
    T = len(times)

    app.layout = html.Div(
        style={"maxWidth": "1100px", "margin": "0 auto", "fontFamily": "system-ui"},
        children=[
            html.H3("Particles (Dash slider over time)"),

            dcc.Graph(id="graph", figure=make_figure(0, data), style={"height": "800px"}),

            dcc.Store(id="axis_store", data=None),

            html.Div(
                style={"display": "flex", "gap": "16px", "alignItems": "center"},
                children=[
                    html.Div(
                        style={"flex": "1"},
                        children=[
                            dcc.Slider(
                                id="t_idx",
                                min=0,
                                max=T - 1,
                                step=1,
                                value=0,
                                marks={0: f"{times[0]:.2f}", T - 1: f"{times[-1]:.2f}"},
                                updatemode="mouseup",
                            )
                        ],
                    ),
                    dcc.Checklist(
                        id="show_labels",
                        options=[{"label": " Show labels", "value": "yes"}],
                        value=["yes"],
                        style={"whiteSpace": "nowrap"},
                    ),
                    html.Button("Fit axes (this frame)", id="fit_axes", n_clicks=0),
                    html.Button("Reset axes (global)", id="reset_axes", n_clicks=0),
                ],
            ),
        ],
    )

    @app.callback(
        Output("axis_store", "data"),
        Input("fit_axes", "n_clicks"),
        Input("reset_axes", "n_clicks"),
        State("t_idx", "value"),
        prevent_initial_call=True,
    )
    def update_axis_store(fit_clicks, reset_clicks, t_idx):
        trig = ctx.triggered_id
        if trig == "reset_axes":
            return None
        if trig == "fit_axes":
            return _fit_axes_to_frame(data, int(t_idx))
        return None

    @app.callback(
        Output("graph", "figure"),
        Input("t_idx", "value"),
        Input("show_labels", "value"),
        Input("axis_store", "data"),
    )
    def update_fig(t_idx, show_labels_val, axis_store):
        show = ("yes" in (show_labels_val or []))
        return make_figure(int(t_idx), data, show_labels=show, axis_override=axis_store)

    return app


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("parquet", help="Path to parquet file")
    p.add_argument("--labels", required=True, help="CSV with columns: sample_id,population")
    p.add_argument("--colors", required=True, help="CSV with columns: population,color")
    p.add_argument("--min-time", type=float, default=-1e300)
    p.add_argument("--max-time", type=float, default=1e300)
    p.add_argument("--time-step", type=float, default=0.01)
    p.add_argument("--no-quantize", action="store_true", help="Do not round XY to integers")
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args()

    data = load_and_pack(
        args.parquet,
        labels_path=args.labels,
        colors_path=args.colors,
        min_time=args.min_time,
        max_time=args.max_time,
        time_step=args.time_step,
        quantize_xy_to_int=(not args.no_quantize),
    )

    app = make_app(data)
    app.run(debug=False, port=args.port)

