#!/usr/bin/env python3
"""
Dash app: linked 3D trajectory + 2D time-slice views.

Layout: 3D Plotly (trajectories + time-indicator plane) on the left;
        2D scatter of the current time slice on the right.
Single slider drives both panels. Same samples highlighted on the plane
in the 3D view and as points in the 2D view.

Features
- Slider over (binned) time, shared between 3D and 2D
- Semi-transparent horizontal plane in 3D marks current time
- Current-time sample positions shown as spheres on the plane in 3D
- Fixed equal-aspect axes in 2D
- Point colors by population (from labels+colors CSVs)
- Optional 3-letter population text labels in 2D
- Hover shows sample_id and population
- Buttons: Fit axes (this frame) / Reset axes (global)  [applies to 2D only]

Input assumptions
- Parquet columns: time, sample_id, X1, X2
- labels CSV columns: sample_id, population
- colors CSV columns: population, color

CLI flags
- --time-scale  log|linear  (default log)
- --time-stride N           subsample time points for 3D trajectories (default 5)
- --lims        float       symmetric XY axis limit (both views)
"""

import numpy as np
import pandas as pd
from pathlib import Path

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, Patch, no_update


def load_and_pack(
    parquet_path: str,
    labels_path: str,
    colors_path: str,
    min_time: float = -np.inf,
    max_time: float = np.inf,
    time_step: float = 0.01,
    quantize_xy_to_int: bool = True,
    generation_time: float | None = None,
):
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)

    time_col = "time"
    x_col = "X1"
    y_col = "X2"
    id_col = "sample_id"

    df = df[(df[time_col] >= min_time) & (df[time_col] <= max_time)].copy()
    if df.empty:
        raise ValueError("No rows left after time filtering.")

    if generation_time is not None:
        df[time_col] = df[time_col].astype(float) * float(generation_time)

    df["time_bin"] = (np.round(df[time_col].to_numpy(dtype=float) / time_step) * time_step).astype(float)

    times = np.sort(df["time_bin"].unique())
    T = len(times)

    sample_order = np.array(sorted(df[id_col].unique()))
    N = len(sample_order)

    t_codes = pd.Categorical(df["time_bin"], categories=times, ordered=True).codes
    s_codes = pd.Index(sample_order).get_indexer(df[id_col].to_numpy())

    xy = np.full((T, N, 2), np.nan, dtype=np.float32)
    xy[t_codes, s_codes, 0] = df[x_col].to_numpy(dtype=np.float32)
    xy[t_codes, s_codes, 1] = df[y_col].to_numpy(dtype=np.float32)

    if quantize_xy_to_int:
        xy = np.rint(xy).astype(np.float32)

    labels_df = pd.read_csv(labels_path)
    colors_df = pd.read_csv(colors_path)

    if id_col not in labels_df.columns or "population" not in labels_df.columns:
        raise ValueError("labels CSV must contain columns: sample_id, population")
    if "population" not in colors_df.columns or "color" not in colors_df.columns:
        raise ValueError("colors CSV must contain columns: population, color")

    pop_map = dict(zip(labels_df[id_col].astype(str), labels_df["population"].astype(str)))
    sample_ids = sample_order.astype(str)
    pops = np.array([pop_map.get(sid, "UNK") for sid in sample_ids], dtype=object)
    pop3 = np.array([p[:3].upper() if p else "" for p in pops], dtype=object)

    pop_color_map = dict(zip(colors_df["population"].astype(str), colors_df["color"].astype(str)))
    colors = np.array([pop_color_map.get(p, "#808080") for p in pops], dtype=object)

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
        generation_time=generation_time,
    )


def _to_list_none(a_1d: np.ndarray):
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


def _z_from_time(t_val: float, time_scale: str) -> float:
    if time_scale == "log":
        return float(np.log10(max(float(t_val), 1e-10)))
    return float(t_val)


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

    customdata = np.column_stack([sample_ids, pops])

    if axis_override is None:
        xlo, xhi, ylo, yhi = data["xlo"], data["xhi"], data["ylo"], data["yhi"]
    else:
        xlo, xhi, ylo, yhi = axis_override["xlo"], axis_override["xhi"], axis_override["ylo"], axis_override["yhi"]

    t = times[frame_idx]

    traces = []
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


def make_3d_base_figure(
    data: dict,
    time_scale: str = "log",
    time_stride: int = 1,
    xy_lims: float | None = None,
) -> tuple[go.Figure, int]:
    """Build the 3D trajectory figure.

    Returns (figure, n_pop_traces) so the caller knows the indices of the
    two mutable traces (highlighted points and plane) appended at the end.
    """
    times = data["times"]
    xy = data["xy"]          # (T, N, 2)
    pops = data["pop"]
    colors = data["color"]
    sample_ids = data["sample_ids"]

    t_indices = np.arange(0, len(times), max(1, time_stride))
    times_sub = times[t_indices]
    xy_sub = xy[t_indices]   # (T_sub, N, 2)

    z_vals = np.array([_z_from_time(t, time_scale) for t in times_sub])

    seen: set = set()
    unique_pops: list = []
    for p in pops:
        if p not in seen:
            unique_pops.append(p)
            seen.add(p)

    pop_color_map: dict = {}
    for p, c in zip(pops, colors):
        pk = str(p)
        if pk not in pop_color_map:
            pop_color_map[pk] = str(c)

    traces = []
    for pop in unique_pops:
        mask = pops == pop
        pop_indices = np.where(mask)[0]
        color = pop_color_map.get(str(pop), "#808080")

        xs: list = []
        ys: list = []
        zs: list = []
        for si in pop_indices:
            xs.extend(_to_list_none(xy_sub[:, si, 0]) + [None])
            ys.extend(_to_list_none(xy_sub[:, si, 1]) + [None])
            zs.extend(z_vals.tolist() + [None])

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color, width=2),
            name=str(pop),
            legendgroup=str(pop),
            opacity=0.7,
        ))

    n_pop_traces = len(traces)

    if xy_lims is not None:
        xlo, xhi = -float(xy_lims), float(xy_lims)
        ylo, yhi = -float(xy_lims), float(xy_lims)
    else:
        xlo, xhi = data["xlo"], data["xhi"]
        ylo, yhi = data["ylo"], data["yhi"]

    # Highlighted current-time points (index: n_pop_traces)
    z0 = float(z_vals[0])
    pts_x = _to_list_none(xy[0, :, 0])
    pts_y = _to_list_none(xy[0, :, 1])

    traces.append(go.Scatter3d(
        x=pts_x,
        y=pts_y,
        z=[z0] * len(pts_x),
        mode="markers",
        marker=dict(size=4, color=colors.tolist(), opacity=1.0),
        showlegend=False,
        name="current_time",
        hovertemplate="%{text}<extra></extra>",
        text=sample_ids.tolist(),
    ))

    # Semi-transparent horizontal plane (index: n_pop_traces + 1)
    traces.append(go.Mesh3d(
        x=[xlo, xhi, xhi, xlo],
        y=[ylo, ylo, yhi, yhi],
        z=[z0, z0, z0, z0],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        opacity=0.15,
        color="lightblue",
        showscale=False,
        showlegend=False,
        name="time_plane",
        hoverinfo="skip",
    ))

    z_min = float(z_vals.min())
    z_max = float(z_vals.max())
    time_units = "years" if data.get("generation_time") is not None else "generations"
    if time_scale == "log":
        z_label = f"log₁₀(Time {time_units})"
    else:
        z_label = f"Time ({time_units})"

    fig = go.Figure(data=traces)
    fig.update_layout(
        scene=dict(
            xaxis=dict(title="X1", range=[xlo, xhi]),
            yaxis=dict(title="X2", range=[ylo, yhi]),
            zaxis=dict(title=z_label, range=[z_min, z_max]),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=1.2),
        ),
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        title=f"3D Trajectories (t = {times[0]:.2f})",
    )
    return fig, n_pop_traces


def load_rates(
    rates_path: str,
    reconstructed_rates_path: str,
    N_max: float,
    labels_path: str,
    colors_path: str,
    cross_pop_color: str = "#808080",
) -> dict:
    """Load input and reconstructed rate parquets and precompute per-pair metadata."""
    input_rates = pd.read_parquet(rates_path)
    reconstructed_rates = pd.read_parquet(reconstructed_rates_path)

    input_rates.columns = input_rates.columns.astype(float)
    reconstructed_rates.columns = reconstructed_rates.columns.astype(float)
    input_rates = input_rates + 1.0 / float(N_max)

    labels_df = pd.read_csv(labels_path)
    colors_df = pd.read_csv(colors_path)
    pop_map = dict(zip(labels_df["sample_id"].astype(str), labels_df["population"].astype(str)))
    pop_color_map = dict(zip(colors_df["population"].astype(str), colors_df["color"].astype(str)))

    sid1 = input_rates.index.get_level_values(0).astype(str)
    sid2 = input_rates.index.get_level_values(1).astype(str)
    pop1 = np.array([pop_map.get(s, "UNK") for s in sid1])
    pop2 = np.array([pop_map.get(s, "UNK") for s in sid2])
    pair_colors = np.where(
        pop1 == pop2,
        np.array([pop_color_map.get(p, cross_pop_color) for p in pop1]),
        cross_pop_color,
    )
    pair_pop_label = np.array([
        f"{p1} × {p2}" if p1 != p2 else p1
        for p1, p2 in zip(pop1, pop2)
    ])
    pair_ids = np.array([f"{s1} × {s2}" for s1, s2 in zip(sid1, sid2)])

    # Per-population median groups (same-pop pairs only)
    unique_same_pops = sorted({p for p, q in zip(pop1, pop2) if p == q})
    same_pop_colors = [pop_color_map.get(p, cross_pop_color) for p in unique_same_pops]
    same_pop_indices = {
        p: np.where((pop1 == p) & (pop2 == p))[0]
        for p in unique_same_pops
    }

    all_vals = np.concatenate([input_rates.values.ravel(), reconstructed_rates.values.ravel()])
    valid = np.isfinite(all_vals) & (all_vals > 0)
    rate_min = float(np.min(all_vals[valid]))
    rate_max = float(np.max(all_vals[valid]))

    return dict(
        input=input_rates,
        reconstructed=reconstructed_rates,
        input_times=input_rates.columns.values,
        reconstructed_times=reconstructed_rates.columns.values,
        pair_colors=pair_colors,
        pair_pop_label=pair_pop_label,
        pair_ids=pair_ids,
        unique_same_pops=unique_same_pops,
        same_pop_colors=same_pop_colors,
        same_pop_indices=same_pop_indices,
        rate_min=rate_min,
        rate_max=rate_max,
    )


def _pop_medians(x_vals: np.ndarray, y_vals: np.ndarray,
                 valid: np.ndarray, rates_data: dict):
    """Return (med_x, med_y) lists aligned to rates_data['unique_same_pops']."""
    med_x, med_y = [], []
    for p in rates_data["unique_same_pops"]:
        idxs = rates_data["same_pop_indices"][p]
        ok = valid[idxs]
        if ok.sum() == 0:
            med_x.append(None)
            med_y.append(None)
        else:
            med_x.append(float(np.median(x_vals[idxs[ok]])))
            med_y.append(float(np.median(y_vals[idxs[ok]])))
    return med_x, med_y


def make_rates_figure(time_val: float, rates_data: dict) -> go.Figure:
    """Log-log scatter of input vs reconstructed rates for the nearest time slice."""
    import math
    in_col_idx  = int(np.argmin(np.abs(rates_data["input_times"]         - float(time_val))))
    rec_col_idx = int(np.argmin(np.abs(rates_data["reconstructed_times"] - float(time_val))))
    t_actual = float(rates_data["input_times"][in_col_idx])

    x_vals = rates_data["input"].iloc[:, in_col_idx].values
    y_vals = rates_data["reconstructed"].iloc[:, rec_col_idx].values
    valid = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0) & (y_vals > 0)

    rate_min = rates_data["rate_min"]
    rate_max = rates_data["rate_max"]
    log_range = [math.log10(rate_min), math.log10(rate_max)]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=[rate_min, rate_max],
        y=[rate_min, rate_max],
        mode="lines",
        line=dict(color="black", dash="dash", width=1),
        showlegend=False,
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scattergl(
        x=x_vals[valid].tolist(),
        y=y_vals[valid].tolist(),
        mode="markers",
        marker=dict(size=4, color=rates_data["pair_colors"][valid].tolist(), opacity=0.6),
        customdata=np.column_stack([
            rates_data["pair_ids"][valid],
            rates_data["pair_pop_label"][valid],
        ]),
        hovertemplate=(
            "pair: %{customdata[0]}<br>"
            "pop: %{customdata[1]}<br>"
            "input: %{x:.3e}<br>"
            "reconstructed: %{y:.3e}<extra></extra>"
        ),
        showlegend=False,
    ))
    med_x, med_y = _pop_medians(x_vals, y_vals, valid, rates_data)
    fig.add_trace(go.Scatter(
        x=med_x,
        y=med_y,
        mode="markers+text",
        marker=dict(
            size=14,
            color=rates_data["same_pop_colors"],
            symbol="diamond",
            line=dict(color="white", width=1.5),
        ),
        text=rates_data["unique_same_pops"],
        textposition="top center",
        textfont=dict(size=9),
        customdata=[[p] for p in rates_data["unique_same_pops"]],
        hovertemplate=(
            "pop: %{customdata[0]}<br>"
            "median input: %{x:.3e}<br>"
            "median reconstructed: %{y:.3e}<extra></extra>"
        ),
        showlegend=False,
    ))
    fig.update_layout(
        title=f"Rate comparison (t ≈ {t_actual:.2f})",
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis=dict(title="Input rate + 1/N_max", type="log", range=log_range,
                   exponentformat="e"),
        yaxis=dict(title="Reconstructed rate", type="log", range=log_range,
                   exponentformat="e"),
    )
    return fig


def make_app(
    data: dict,
    time_scale: str = "linear",
    time_stride: int = 1,
    xy_lims: float | None = None,
    rates_data: dict | None = None,
):
    app = Dash(__name__)
    times = data["times"]
    T = len(times)

    fig_3d, n_pop_traces = make_3d_base_figure(
        data, time_scale=time_scale, time_stride=time_stride, xy_lims=xy_lims
    )
    pts_trace_idx = n_pop_traces
    plane_trace_idx = n_pop_traces + 1

    z_for_time = np.array([_z_from_time(t, time_scale) for t in times])

    fig_rates_base = (
        make_rates_figure(float(times[0]), rates_data)
        if rates_data is not None else go.Figure()
    )

    app.layout = html.Div(
        style={"maxWidth": "1400px", "margin": "0 auto", "fontFamily": "system-ui",
               "padding": "0 8px"},
        children=[
            html.H3("Structure Through Time — 3D + 2D Linked View",
                    style={"margin": "6px 0 4px 0"}),

            html.Div(
                style={"display": "flex", "gap": "4px"},
                children=[
                    dcc.Graph(
                        id="graph3d",
                        figure=fig_3d,
                        style={"flex": "1", "height": "400px"},
                    ),
                    dcc.Graph(
                        id="graph2d",
                        figure=make_figure(0, data),
                        style={"flex": "1", "height": "400px"},
                    ),
                ],
            ),

            dcc.Store(id="axis_store", data=None),
            dcc.Interval(id="play_interval", interval=200, disabled=True, n_intervals=0),

            html.Div(
                style={"display": "flex", "gap": "16px", "alignItems": "center", "marginTop": "4px"},
                children=[
                    html.Button("▶ Play", id="play_pause_btn", n_clicks=0,
                                style={"whiteSpace": "nowrap", "fontWeight": "bold"}),
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
                    dcc.Checklist(
                        id="show_rates",
                        options=[{"label": " Rate panel", "value": "yes",
                                  "disabled": rates_data is None}],
                        value=[],
                        style={"whiteSpace": "nowrap",
                               "display": "inline" if rates_data is not None else "none"},
                    ),
                    html.Div(
                        style={"display": "flex", "flexDirection": "column", "alignItems": "center",
                               "minWidth": "140px"},
                        children=[
                            html.Label("Speed", style={"fontSize": "12px", "marginBottom": "2px"}),
                            dcc.Slider(
                                id="speed_slider",
                                min=0, max=4, step=None,
                                value=2,
                                marks={0: "0.25×", 1: "0.5×", 2: "1×", 3: "2×", 4: "20×"},
                                updatemode="mouseup",
                            ),
                        ],
                    ),
                    html.Button("Fit axes (this frame)", id="fit_axes", n_clicks=0),
                    html.Button("Reset axes (global)", id="reset_axes", n_clicks=0),
                ],
            ),

            html.Div(
                id="rates_panel",
                style={"display": "none", "marginTop": "4px"},
                children=[
                    dcc.Graph(id="graph_rates", figure=fig_rates_base, style={"height": "260px"}),
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
        Output("play_interval", "disabled"),
        Output("play_pause_btn", "children"),
        Input("play_pause_btn", "n_clicks"),
        State("play_interval", "disabled"),
        prevent_initial_call=True,
    )
    def toggle_play(_n_clicks, currently_disabled):
        if currently_disabled:
            # Start from beginning if at the end
            return False, "⏸ Pause"
        return True, "▶ Play"

    _SPEED_MS = {0: 800, 1: 400, 2: 200, 3: 100, 4: 10}

    @app.callback(
        Output("play_interval", "interval"),
        Input("speed_slider", "value"),
    )
    def set_speed(level):
        return _SPEED_MS.get(int(level), 200)

    @app.callback(
        Output("t_idx", "value"),
        Output("play_interval", "disabled", allow_duplicate=True),
        Output("play_pause_btn", "children", allow_duplicate=True),
        Input("play_interval", "n_intervals"),
        State("t_idx", "value"),
        prevent_initial_call=True,
    )
    def advance_slider(_n_intervals, t_idx):
        t_idx = int(t_idx)
        if t_idx >= T - 1:
            return 0, True, "▶ Play"  # stop and reset to start
        return t_idx + 1, no_update, no_update

    @app.callback(
        Output("graph2d", "figure"),
        Output("graph3d", "figure"),
        Input("t_idx", "value"),
        Input("show_labels", "value"),
        Input("axis_store", "data"),
    )
    def update_figs(t_idx, show_labels_val, axis_store):
        t_idx = int(t_idx)
        show = "yes" in (show_labels_val or [])
        t_val = float(times[t_idx])

        fig_2d = make_figure(t_idx, data, show_labels=show, axis_override=axis_store)

        z0 = float(z_for_time[t_idx])
        pts_x = _to_list_none(data["xy"][t_idx, :, 0])
        pts_y = _to_list_none(data["xy"][t_idx, :, 1])

        patched = Patch()
        patched["data"][pts_trace_idx]["x"] = pts_x
        patched["data"][pts_trace_idx]["y"] = pts_y
        patched["data"][pts_trace_idx]["z"] = [z0] * len(pts_x)
        patched["data"][plane_trace_idx]["z"] = [z0, z0, z0, z0]
        patched["layout"]["title"]["text"] = f"3D Trajectories (t = {t_val:.2f})"

        return fig_2d, patched

    @app.callback(
        Output("rates_panel", "style"),
        Input("show_rates", "value"),
    )
    def toggle_rates_panel(show_rates_val):
        if "yes" in (show_rates_val or []):
            return {"display": "block", "marginTop": "8px"}
        return {"display": "none", "marginTop": "8px"}

    @app.callback(
        Output("graph_rates", "figure"),
        Input("t_idx", "value"),
        Input("show_rates", "value"),
    )
    def update_rates(t_idx, show_rates_val):
        if rates_data is None or "yes" not in (show_rates_val or []):
            return no_update

        t_val = float(times[int(t_idx)])

        # Full rebuild only when the panel is first toggled on; Patch for every tick.
        if ctx.triggered_id == "show_rates":
            return make_rates_figure(t_val, rates_data)

        in_col_idx  = int(np.argmin(np.abs(rates_data["input_times"]         - t_val)))
        rec_col_idx = int(np.argmin(np.abs(rates_data["reconstructed_times"] - t_val)))
        t_actual = float(rates_data["input_times"][in_col_idx])

        x_vals = rates_data["input"].iloc[:, in_col_idx].values
        y_vals = rates_data["reconstructed"].iloc[:, rec_col_idx].values
        valid = np.isfinite(x_vals) & np.isfinite(y_vals) & (x_vals > 0) & (y_vals > 0)

        med_x, med_y = _pop_medians(x_vals, y_vals, valid, rates_data)

        patched = Patch()
        patched["data"][1]["x"] = x_vals[valid].tolist()
        patched["data"][1]["y"] = y_vals[valid].tolist()
        patched["data"][1]["marker"]["color"] = rates_data["pair_colors"][valid].tolist()
        patched["data"][1]["customdata"] = np.column_stack([
            rates_data["pair_ids"][valid],
            rates_data["pair_pop_label"][valid],
        ]).tolist()
        patched["data"][2]["x"] = med_x
        patched["data"][2]["y"] = med_y
        patched["layout"]["title"]["text"] = f"Rate comparison (t ≈ {t_actual:.2f})"
        return patched

    return app


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Linked 3D+2D trajectory Dash app")
    p.add_argument("parquet", help="Path to parquet file")
    p.add_argument("--labels", required=True, help="CSV with columns: sample_id,population")
    p.add_argument("--colors", required=True, help="CSV with columns: population,color")
    p.add_argument("--min_time", type=float, default=-1e300)
    p.add_argument("--max_time", type=float, default=1e300)
    p.add_argument("--time_step", type=float, default=0.01,
                   help="Bin width for time quantization (default 0.01)")
    p.add_argument("--no_quantize", action="store_true",
                   help="Do not round XY to integers")
    p.add_argument("--time_scale", type=str, default="linear", choices=["log", "linear"],
                   help="Z-axis scale for 3D view (default linear)")
    p.add_argument("--time_stride", type=int, default=5,
                   help="Keep every Nth time point in 3D trajectories (default 5)")
    p.add_argument("--lims", type=str, default="None",
                   help="Symmetric XY axis limit (both views), or 'None'")
    p.add_argument("--generation_time", type=float, default=None,
                   help="Generation time in years; scales time axis")
    p.add_argument("--rates", type=str, default=None,
                   help="Path to input rates parquet (enables rate panel)")
    p.add_argument("--reconstructed_rates", type=str, default=None,
                   help="Path to reconstructed rates parquet (enables rate panel)")
    p.add_argument("--N_max", type=float, default=None,
                   help="N_max used during embedding (required with --rates)")
    p.add_argument("--port", type=int, default=8050)
    args = p.parse_args()

    lims = None if args.lims == "None" else float(args.lims)

    data = load_and_pack(
        args.parquet,
        labels_path=args.labels,
        colors_path=args.colors,
        min_time=args.min_time,
        max_time=args.max_time,
        time_step=args.time_step,
        quantize_xy_to_int=(not args.no_quantize),
        generation_time=args.generation_time,
    )

    rates_data = None
    if args.rates and args.reconstructed_rates:
        if args.N_max is None:
            raise ValueError("--N_max is required when --rates and --reconstructed_rates are provided")
        rates_data = load_rates(
            rates_path=args.rates,
            reconstructed_rates_path=args.reconstructed_rates,
            N_max=args.N_max,
            labels_path=args.labels,
            colors_path=args.colors,
        )

    app = make_app(
        data,
        time_scale=args.time_scale,
        time_stride=args.time_stride,
        xy_lims=lims,
        rates_data=rates_data,
    )
    app.run(debug=False, port=args.port)
