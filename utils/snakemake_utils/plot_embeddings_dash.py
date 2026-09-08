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
- --time-stride N           subsample time points for 3D trajectories (default 1)
- --lims        float       symmetric XY axis limit (both views)

Example run: 
python plot_embeddings_dash.py \
  "../results/empirical/all.npp.2/data/embeddings.d2.Nmin1000.Nmax10000000.L1[None].minWS100.maxWS1500.V1.FS1.0.maxT4000.nT4000.parquet" \
  --labels "../data/metadata/wohns_pop_labels.csv" \
  --colors "../data/metadata/wohns_pop_colors.csv" \
  --rates "../results/empirical/all.npp.2/data/rates.minWS100.maxWS1500.V1.FS1.0.maxT4000.nT4000.parquet" \
  --reconstructed_rates "../results/empirical/all.npp.2/data/reconstructed_rates.d2.Nmin1000.Nmax10000000.L1[None].minWS100.maxWS1500.V1.FS1.0.maxT4000.nT4000.parquet" \
  --N_max 10000000
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import chi2

import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, Patch, no_update


def ellipse_points(mu: np.ndarray, cov: np.ndarray, level: float = 0.68, n_pts: int = 50):
    """Return (x, y) arrays tracing a confidence ellipse.

    Args:
        mu: (2,) centre
        cov: (2, 2) covariance matrix
        level: confidence level, e.g. 0.68 or 0.95
        n_pts: number of points on the ellipse

    Returns:
        (x, y) arrays of length n_pts
    """
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)  # guard against tiny negatives from floating point
    scale = np.sqrt(chi2.ppf(level, df=2))
    theta = np.linspace(0, 2 * np.pi, n_pts, endpoint=False)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    pts = scale * (circle * np.sqrt(eigvals)) @ eigvecs.T + mu
    return pts[:, 0], pts[:, 1]


def load_covariances(
    cov_path: str,
    times: np.ndarray,
    sample_order: np.ndarray,
    time_step: float,
) -> np.ndarray:
    """Load a covariances parquet and pack into a (T, N, 2, 2) float64 array.

    Args:
        cov_path: path to covariances parquet (columns: sample_id, time, C11, C12, C21, C22)
        times: 1-D array of binned time values (same as data['times'])
        sample_order: 1-D array of sample_id strings (same as data['sample_ids'])
        time_step: time bin width used to quantise time values

    Returns:
        np.ndarray shape (T, N, 2, 2), NaN where data is absent
    """
    df = pd.read_parquet(cov_path)
    df["time_bin"] = (
        np.round(df["time"].to_numpy(dtype=float) / time_step) * time_step
    ).astype(float)

    T = len(times)
    N = len(sample_order)
    covariances = np.full((T, N, 2, 2), np.nan, dtype=np.float64)

    t_codes = pd.Categorical(df["time_bin"], categories=times, ordered=True).codes
    s_codes = pd.Index(sample_order.astype(str)).get_indexer(df["sample_id"].astype(str).to_numpy())

    valid = (t_codes >= 0) & (s_codes >= 0)
    tc = t_codes[valid]
    sc = s_codes[valid]
    for (ri, rj), col in [((0, 0), "C11"), ((0, 1), "C12"), ((1, 0), "C21"), ((1, 1), "C22")]:
        covariances[tc, sc, ri, rj] = df[col].to_numpy()[valid]

    return covariances


def load_and_pack(
    parquet_path: str,
    labels_path: str,
    colors_path: str,
    min_time: float = -np.inf,
    max_time: float = np.inf,
    time_step: float = 0.01,
    quantize_xy_to_int: bool = True,
    generation_time: float | None = None,
    covariances_path: str | None = None,
):
    parquet_path = Path(parquet_path)
    df = pd.read_parquet(parquet_path)

    time_col = "time"
    x_col = "X1"
    y_col = "X2"
    id_col = "individual_id" if "individual_id" in df.columns else "sample_id"

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

    # Always keep full-precision copy for ellipse centres before any quantisation
    xy_float = xy.copy().astype(np.float64)

    if quantize_xy_to_int:
        xy = np.rint(xy).astype(np.float32)

    labels_df = pd.read_csv(labels_path)
    colors_df = pd.read_csv(colors_path)

    if id_col not in labels_df.columns or "population" not in labels_df.columns:
        raise ValueError(f"labels CSV must contain columns: {id_col}, population")
    if "population" not in colors_df.columns or "color" not in colors_df.columns:
        raise ValueError("colors CSV must contain columns: population, color")

    pop_map = dict(zip(labels_df[id_col].astype(str), labels_df["population"].astype(str)))
    sample_ids = sample_order.astype(str)
    pops = np.array([pop_map.get(sid, "UNK") for sid in sample_ids], dtype=object)
    pop3 = np.array([p[:3].upper() if p else "" for p in pops], dtype=object)

    pop_color_map = dict(zip(colors_df["population"].astype(str), colors_df["color"].astype(str)))
    colors = np.array([pop_color_map.get(p, "#808080") for p in pops], dtype=object)

    # Per-population index mapping (stable order, used for ellipse colouring)
    unique_pops: list = []
    pop_to_sample_indices: dict = {}
    for i, p in enumerate(pops):
        pk = str(p)
        if pk not in pop_to_sample_indices:
            unique_pops.append(p)
            pop_to_sample_indices[pk] = []
        pop_to_sample_indices[pk].append(i)

    xmin, xmax = np.nanmin(xy[:, :, 0]), np.nanmax(xy[:, :, 0])
    ymin, ymax = np.nanmin(xy[:, :, 1]), np.nanmax(xy[:, :, 1])
    dx, dy = float(xmax - xmin), float(ymax - ymin)
    pad = 0.03
    dx = dx if dx > 0 else 1.0
    dy = dy if dy > 0 else 1.0
    xlo, xhi = float(xmin - pad * dx), float(xmax + pad * dx)
    ylo, yhi = float(ymin - pad * dy), float(ymax + pad * dy)

    covariances = None
    if covariances_path is not None:
        try:
            covariances = load_covariances(covariances_path, times, sample_ids, time_step)
        except Exception as e:
            print(f"Warning: could not load covariances from {covariances_path}: {e}")

    return dict(
        times=times,
        xy=xy,
        xy_float=xy_float,
        sample_ids=sample_ids,
        pop=pops,
        pop3=pop3,
        color=colors,
        xlo=xlo, xhi=xhi, ylo=ylo, yhi=yhi,
        generation_time=generation_time,
        covariances=covariances,
        unique_pops=unique_pops,
        pop_color_map_str=pop_color_map,
        pop_to_sample_indices=pop_to_sample_indices,
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
    show_ellipses: bool = False,
    confidence_level: float = 0.68,
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

    # Trace layout (fixed indices, stable across all figure rebuilds):
    #   0 .. n_pops-1  : per-population ellipse traces
    #   n_pops         : main scatter
    #   n_pops + 1     : text labels (always present; empty when show_labels=False)
    #   n_pops + 2     : highlight ellipse (initially empty; filled by hover callback)

    unique_pops = data.get("unique_pops", [])
    pop_color_map_str = data.get("pop_color_map_str", {})
    pop_to_sample_indices = data.get("pop_to_sample_indices", {})
    has_covariances = data.get("covariances") is not None

    traces = []

    # 1. Per-population ellipse traces
    if show_ellipses and has_covariances:
        xy_f = data["xy_float"][frame_idx]
        covs = data["covariances"][frame_idx]
    else:
        xy_f = covs = None

    for pop in unique_pops:
        pop_color = pop_color_map_str.get(str(pop), "#808080")
        ex_all: list = []
        ey_all: list = []
        cd_all: list = []

        if xy_f is not None:
            for i in pop_to_sample_indices.get(str(pop), []):
                mu = xy_f[i]
                cov = covs[i]
                if np.any(np.isnan(mu)) or np.any(np.isnan(cov)):
                    continue
                ex, ey = ellipse_points(mu, cov, level=confidence_level)
                # Close the ellipse and add a None break
                ex_all.extend(ex.tolist() + [ex[0], None])
                ey_all.extend(ey.tolist() + [ey[0], None])
                cd_all.extend([sample_ids[i]] * (len(ex) + 1) + [None])

        traces.append(go.Scattergl(
            x=ex_all,
            y=ey_all,
            mode="lines",
            line=dict(width=1, color=pop_color),
            opacity=0.5,
            customdata=cd_all,
            hovertemplate="sample: %{customdata}<br>pop: " + str(pop) + "<extra></extra>",
            showlegend=False,
            name=f"ellipse_{pop}",
        ))

    # 2. Main scatter
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

    # 3. Text labels (always present; empty coords when hidden)
    traces.append(
        go.Scatter(
            x=x if show_labels else [],
            y=y if show_labels else [],
            mode="text",
            text=pop3.tolist(),
            textposition="top center",
            textfont=dict(size=10, color=colors.tolist()),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    # 4. Highlight ellipse — filled by hover callback, always present for stable indexing
    traces.append(
        go.Scattergl(
            x=[], y=[],
            mode="lines",
            line=dict(width=3, color="white"),
            opacity=1.0,
            hoverinfo="skip",
            showlegend=False,
            name="highlight_ellipse",
        )
    )

    # 5. Pair edge — filled when hovering over a pairwise rate, always present
    traces.append(
        go.Scattergl(
            x=[], y=[],
            mode="lines+markers",
            line=dict(width=2, color="rgba(255,255,255,0.8)"),
            marker=dict(size=14, color=[], line=dict(color="white", width=2)),
            hoverinfo="skip",
            showlegend=False,
            name="pair_edge",
        )
    )

    # 6. Outlier over-estimated edges (red solid) — n_pops + 4
    # go.Scatter (SVG) used instead of Scattergl so hover detects along the full line segment
    traces.append(go.Scatter(
        x=[], y=[], mode="lines+markers", customdata=[],
        line=dict(color="#d62728", width=3, dash="solid"),
        marker=dict(size=6, color="#d62728"),
        opacity=0.75, showlegend=False, name="outlier_over",
        hovertemplate="pair: %{customdata[0]}<br>pops: %{customdata[2]}<br>log ratio: %{customdata[1]:.2f}<extra></extra>",
    ))

    # 7. Outlier under-estimated edges (blue dashed) — n_pops + 5
    traces.append(go.Scatter(
        x=[], y=[], mode="lines+markers", customdata=[],
        line=dict(color="#1f77b4", width=3, dash="dash"),
        marker=dict(size=6, color="#1f77b4"),
        opacity=0.75, showlegend=False, name="outlier_under",
        hovertemplate="pair: %{customdata[0]}<br>pops: %{customdata[2]}<br>log ratio: %{customdata[1]:.2f}<extra></extra>",
    ))

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
    t_start_idx: int | None = None,
    t_stop_idx: int | None = None,
) -> tuple[go.Figure, int]:
    """Build the 3D trajectory figure.

    Returns (figure, n_pop_traces) so the caller knows the indices of the
    two mutable traces (highlighted points and plane) appended at the end.

    t_start_idx/t_stop_idx restrict the drawn time range to a [start, stop]
    index window (into ``times``); this both clips the trajectories and tightens
    the z-axis to the window. Defaults to the full range.
    """
    times = data["times"]
    xy = data["xy"]          # (T, N, 2)
    pops = data["pop"]
    colors = data["color"]
    sample_ids = data["sample_ids"]

    lo = 0 if t_start_idx is None else max(0, int(t_start_idx))
    hi = len(times) - 1 if t_stop_idx is None else min(len(times) - 1, int(t_stop_idx))
    if hi < lo:
        lo, hi = hi, lo
    t_indices = np.arange(lo, hi + 1, max(1, time_stride))
    if len(t_indices) == 0:
        t_indices = np.array([lo])
    times_sub = times[t_indices]
    xy_sub = xy[t_indices]   # (T_sub, N, 2)

    z_vals = np.array([_z_from_time(t, time_scale) for t in times_sub])

    unique_pops: list = sorted(set(pops))

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
        ts: list = []
        for si in pop_indices:
            xs.extend(_to_list_none(xy_sub[:, si, 0]) + [None])
            ys.extend(_to_list_none(xy_sub[:, si, 1]) + [None])
            zs.extend(z_vals.tolist() + [None])
            ts.extend(times_sub.tolist() + [None])

        traces.append(go.Scatter3d(
            x=xs, y=ys, z=zs,
            mode="lines",
            line=dict(color=color, width=2),
            name=str(pop),
            legendgroup=str(pop),
            opacity=0.7,
            customdata=ts,
            hovertemplate="%{fullData.name}<br>x = %{x:.2f}, y = %{y:.2f}<br>t = %{customdata:.1f}<extra></extra>",
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
        customdata=[float(times[0])] * len(pts_x),
        hovertemplate="%{text}<br>x = %{x:.2f}, y = %{y:.2f}<br>t = %{customdata:.1f}<extra></extra>",
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
        opacity=0.35,
        color="steelblue",
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
    # Map the rate-pair index to populations using whichever id the rates are
    # indexed by: per-individual rates (phased=False) use individual_id, while
    # per-haplotype rates use sample_id.
    idx_name = str(input_rates.index.names[0] or "")
    id_col = "individual_id" if ("individual" in idx_name and "individual_id" in labels_df.columns) else "sample_id"
    pop_map = dict(zip(labels_df[id_col].astype(str), labels_df["population"].astype(str)))
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

    # Pre-compute log(reconstructed/input) per pair per time — aligned to reconstructed_times
    recon_times = reconstructed_rates.columns.values.astype(float)
    in_times    = input_rates.columns.values.astype(float)
    recon_arr   = reconstructed_rates.values.astype(float)           # (n_pairs, T_recon)
    in_col_idxs = np.array([int(np.argmin(np.abs(in_times - t))) for t in recon_times])
    in_arr_aln  = input_rates.values.astype(float)[:, in_col_idxs]  # (n_pairs, T_recon)
    with np.errstate(divide='ignore', invalid='ignore'):
        log_ratio = np.where(
            np.isfinite(recon_arr) & np.isfinite(in_arr_aln) & (recon_arr > 0) & (in_arr_aln > 0),
            np.log(recon_arr / in_arr_aln),
            np.nan,
        )

    return dict(
        input=input_rates,
        reconstructed=reconstructed_rates,
        input_times=input_rates.columns.values,
        reconstructed_times=reconstructed_rates.columns.values,
        log_ratio=log_ratio,
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
    fig.add_trace(go.Scattergl(
        x=med_x,
        y=med_y,
        mode="markers+text",
        marker=dict(
            size=14,
            color=rates_data["same_pop_colors"],
            symbol="diamond",
            line=dict(color="black", width=1.5),
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
    # trace 3: highlighted pair from outlier edge hover (starts empty)
    fig.add_trace(go.Scattergl(
        x=[], y=[], mode="markers",
        marker=dict(size=14, color=[], symbol="circle",
                    line=dict(color="black", width=2)),
        hoverinfo="skip", showlegend=False, name="outlier_highlight",
    ))
    fig.update_layout(
        title=f"Rate comparison (t ≈ {t_actual:.2f})",
        margin=dict(l=50, r=20, t=30, b=40),
        xaxis=dict(title="True / Input Rate", type="log", range=log_range,
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
    confidence_level: float = 0.68,
    meta: dict | None = None,
):
    app = Dash(__name__)
    times = data["times"]
    T = len(times)

    fig_3d, n_pop_traces = make_3d_base_figure(
        data, time_scale=time_scale, time_stride=time_stride, xy_lims=xy_lims
    )
    pts_trace_idx = n_pop_traces
    plane_trace_idx = n_pop_traces + 1

    has_covariances = data.get("covariances") is not None
    win_time_units = "years" if data.get("generation_time") is not None else "generations"

    # Stable 2D trace indices (determined by number of populations in dataset)
    n_pops_2d = len(data.get("unique_pops", []))
    scatter_trace_idx_2d       = n_pops_2d
    highlight_trace_idx_2d    = n_pops_2d + 2   # labels always at n_pops_2d + 1
    pair_edge_trace_idx_2d    = n_pops_2d + 3   # pair edge for rate-panel hover
    outlier_over_trace_idx_2d = n_pops_2d + 4   # over-estimated outlier edges
    outlier_under_trace_idx_2d = n_pops_2d + 5  # under-estimated outlier edges

    fig_rates_base = (
        make_rates_figure(float(times[0]), rates_data)
        if rates_data is not None else go.Figure()
    )

    app.layout = html.Div(
        style={"maxWidth": "1400px", "margin": "0 auto", "fontFamily": "system-ui",
               "padding": "0 8px"},
        children=[
            html.H3("CoaSTT — 3D + 2D Linked View",
                    style={"margin": "6px 0 4px 0"}),

            html.Div(
                style={"display": "flex", "gap": "4px"},
                children=[
                    dcc.Graph(
                        id="graph3d",
                        figure=fig_3d,
                        style={"flex": "1", "height": "400px"},
                    ),
                    html.Div(
                        style={"flex": "1", "display": "flex", "flexDirection": "column"},
                        children=[
                            html.Div(
                                style={"display": "flex", "justifyContent": "flex-end",
                                       "alignItems": "center", "marginBottom": "2px"},
                                children=[
                                    dcc.Checklist(
                                        id="log_time_toggle",
                                        options=[{"label": " Log time", "value": "log"}],
                                        value=["log"] if time_scale == "log" else [],
                                        inline=True,
                                        style={"fontSize": "12px"},
                                    ),
                                ],
                            ),
                            dcc.Graph(
                                id="graph2d",
                                figure=make_figure(0, data, confidence_level=confidence_level),
                                style={"flex": "1", "height": "376px"},
                                clear_on_unhover=True,
                            ),
                        ],
                    ),
                ],
            ),

            dcc.Store(id="axis_store", data=None),
            dcc.Store(id="time_scale_store", data=time_scale),
            dcc.Store(id="camera3d_store", data=None),
            dcc.Store(id="key_event", data=None),
            dcc.Store(id="time_window", data=[0, T - 1]),
            dcc.Interval(id="play_interval", interval=200, disabled=True, n_intervals=0),
            dcc.Interval(id="key_poll", interval=100, disabled=False, n_intervals=0),

            html.Div(
                style={"display": "flex", "gap": "16px", "alignItems": "center", "marginTop": "4px"},
                children=[
                    html.Button("▶ Play", id="play_pause_btn", n_clicks=0,
                                style={"whiteSpace": "nowrap", "fontWeight": "bold"}),
                    html.Div(
                        style={"display": "flex", "flexDirection": "column", "minWidth": "150px"},
                        children=[
                            html.Label(f"3D time window ({win_time_units})",
                                       style={"fontSize": "12px", "marginBottom": "2px"}),
                            html.Div(
                                style={"display": "flex", "gap": "4px", "alignItems": "center"},
                                children=[
                                    dcc.Input(id="win_start", type="number", value=float(times[0]),
                                              debounce=True, style={"width": "70px", "fontSize": "12px"}),
                                    html.Span("→", style={"fontSize": "12px"}),
                                    dcc.Input(id="win_stop", type="number", value=float(times[-1]),
                                              debounce=True, style={"width": "70px", "fontSize": "12px"}),
                                ],
                            ),
                        ],
                    ),
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
                        id="show_ellipses",
                        options=[{"label": " 95% Ellipses", "value": "yes",
                                  "disabled": not has_covariances}],
                        value=[],
                        style={"whiteSpace": "nowrap",
                               "display": "inline" if has_covariances else "none"},
                    ),
                    dcc.Checklist(
                        id="show_rates",
                        options=[{"label": " Rate panel", "value": "yes",
                                  "disabled": rates_data is None}],
                        value=["yes"] if rates_data is not None else [],
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
                                min=0, max=2, step=None,
                                value=1,
                                marks={0: "0.1×", 1: "1×", 2: "100×"},
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
                style={"display": "block" if rates_data is not None else "none",
                       "marginTop": "4px"},
                children=[
                    dcc.Graph(id="graph_rates", figure=fig_rates_base,
                              style={"height": "360px"},
                              clear_on_unhover=True),
                    html.Div(
                        style={"display": "flex", "alignItems": "center",
                               "gap": "8px", "marginTop": "4px"},
                        children=[
                            html.Label("Outlier pairs:", style={"fontSize": "12px"}),
                            dcc.Dropdown(
                                id="n_outliers_dropdown",
                                options=[
                                    {"label": "Off", "value": 0},
                                    {"label": "3",   "value": 3},
                                    {"label": "5",   "value": 5},
                                    {"label": "10",  "value": 10},
                                ],
                                value=0,
                                clearable=False,
                                style={"width": "80px", "fontSize": "12px"},
                            ),
                            html.Span("Red solid = over-estimated  |  Blue dashed = under-estimated",
                                      style={"fontSize": "11px", "color": "#888"}),
                        ],
                    ),
                ],
            ),
            html.Hr(style={"marginTop": "12px", "borderColor": "#ddd"}),
            html.Div(
                style={"fontSize": "11px", "color": "#888", "lineHeight": "1.8",
                       "padding": "4px 0 8px 0"},
                children=[
                    html.Div([
                        html.Span("Available time range: ", style={"fontWeight": "bold"}),
                        f"{times[0]:.2f} – {times[-1]:.2f} {win_time_units} "
                        f"(the limits for the 3D time-window boxes above)",
                    ]),
                    html.Div(html.Span("Settings", style={"fontWeight": "bold"})),
                    *[
                        html.Div(f"{k}: {v}")
                        for k, v in (meta or {}).items()
                    ],
                ],
            ),
        ],
    )

    @app.callback(
        Output("axis_store", "data"),
        Input("fit_axes", "n_clicks"),
        Input("reset_axes", "n_clicks"),
        Input("graph2d", "relayoutData"),
        State("t_idx", "value"),
        prevent_initial_call=True,
    )
    def update_axis_store(fit_clicks, reset_clicks, relayout_data, t_idx):
        trig = ctx.triggered_id
        if trig == "reset_axes":
            return None
        if trig == "fit_axes":
            return _fit_axes_to_frame(data, int(t_idx))
        if trig == "graph2d" and relayout_data:
            if "xaxis.range[0]" in relayout_data:
                return {
                    "xlo": relayout_data["xaxis.range[0]"],
                    "xhi": relayout_data["xaxis.range[1]"],
                    "ylo": relayout_data["yaxis.range[0]"],
                    "yhi": relayout_data["yaxis.range[1]"],
                }
            if "xaxis.autorange" in relayout_data or "autosize" in relayout_data:
                return None
        return no_update

    @app.callback(
        Output("camera3d_store", "data"),
        Input("graph3d", "relayoutData"),
        prevent_initial_call=True,
    )
    def store_camera(relayout):
        if relayout and "scene.camera" in relayout:
            return relayout["scene.camera"]
        return no_update

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

    _SPEED_MS = {0: 2000, 1: 200, 2: 2}  # 0.1×/1×/100× (1× ≈ 5fps)

    @app.callback(
        Output("play_interval", "interval"),
        Input("speed_slider", "value"),
    )
    def set_speed(level):
        return _SPEED_MS.get(int(level), 200)

    @app.callback(
        Output("t_idx", "value", allow_duplicate=True),
        Output("play_interval", "disabled", allow_duplicate=True),
        Output("play_pause_btn", "children", allow_duplicate=True),
        Input("play_interval", "n_intervals"),
        State("t_idx", "value"),
        State("time_window", "data"),
        prevent_initial_call=True,
    )
    def advance_slider(_n_intervals, t_idx, window):
        t_idx = int(t_idx)
        start_idx, stop_idx = int(window[0]), int(window[1])
        # Loop continuously within [start, stop]: at/after stop, jump back to start.
        if t_idx >= stop_idx or t_idx < start_idx:
            return start_idx, no_update, no_update
        return t_idx + 1, no_update, no_update

    # Arrow-key support: set up document listener once on page load
    app.clientside_callback(
        """
        function(id) {
            window._stt_arrow = null;
            document.addEventListener('keydown', function(e) {
                var tag = (document.activeElement && document.activeElement.tagName) || '';
                if (['INPUT', 'TEXTAREA', 'SELECT'].includes(tag)) return;
                if (e.key === 'ArrowRight') { window._stt_arrow = 1;  e.preventDefault(); }
                else if (e.key === 'ArrowLeft')  { window._stt_arrow = -1; e.preventDefault(); }
            });
            return window.dash_clientside.no_update;
        }
        """,
        Output("key_event", "data"),
        Input("key_event", "id"),
        prevent_initial_call=False,
    )

    # Arrow-key support: poll global state every 100ms and advance slider
    app.clientside_callback(
        """
        function(n, t_val, t_min, t_max) {
            var dir = window._stt_arrow;
            if (!dir) return window.dash_clientside.no_update;
            window._stt_arrow = null;
            var lo = parseInt(t_min), hi = parseInt(t_max);
            var new_t = Math.max(lo, Math.min(hi, parseInt(t_val) + dir));
            return new_t;
        }
        """,
        Output("t_idx", "value", allow_duplicate=True),
        Input("key_poll", "n_intervals"),
        [State("t_idx", "value"), State("t_idx", "min"), State("t_idx", "max")],
        prevent_initial_call=True,
    )

    @app.callback(
        Output("time_scale_store", "data"),
        Input("log_time_toggle", "value"),
    )
    def update_time_scale(val):
        return "log" if "log" in (val or []) else "linear"

    @app.callback(
        Output("graph3d", "figure", allow_duplicate=True),
        Input("time_scale_store", "data"),
        State("t_idx", "value"),
        State("camera3d_store", "data"),
        prevent_initial_call=True,
    )
    def rebuild_3d_on_scale_change(ts, t_idx_val, camera):
        fig, _ = make_3d_base_figure(data, time_scale=ts, time_stride=time_stride, xy_lims=xy_lims)
        t_int = int(t_idx_val)
        z0 = float(_z_from_time(float(times[t_int]), ts))
        pts_x = _to_list_none(data["xy"][t_int, :, 0])
        pts_y = _to_list_none(data["xy"][t_int, :, 1])
        t_val_3d = float(times[t_int])
        fig.data[pts_trace_idx]["x"] = pts_x
        fig.data[pts_trace_idx]["y"] = pts_y
        fig.data[pts_trace_idx]["z"] = [z0] * len(pts_x)
        fig.data[pts_trace_idx]["customdata"] = [t_val_3d] * len(pts_x)
        fig.data[plane_trace_idx]["z"] = [z0, z0, z0, z0]
        if camera:
            fig.update_layout(scene_camera=camera)
        return fig

    @app.callback(
        Output("time_window", "data"),
        Output("t_idx", "min"),
        Output("t_idx", "max"),
        Output("t_idx", "marks"),
        Output("t_idx", "value", allow_duplicate=True),
        Output("win_start", "value"),
        Output("win_stop", "value"),
        Input("win_start", "value"),
        Input("win_stop", "value"),
        State("t_idx", "value"),
        prevent_initial_call=True,
    )
    def set_time_window(start_val, stop_val, cur_t):
        # Map entered times -> nearest indices; default blanks to the data ends.
        if start_val is None:
            start_val = float(times[0])
        if stop_val is None:
            stop_val = float(times[-1])
        start_idx = int(np.argmin(np.abs(times - float(start_val))))
        stop_idx = int(np.argmin(np.abs(times - float(stop_val))))
        if stop_idx <= start_idx:                       # force a non-empty window
            if start_idx >= T - 1:
                start_idx = max(0, T - 2)
            stop_idx = min(T - 1, start_idx + 1)
        start_idx = max(0, min(start_idx, T - 1))
        stop_idx = max(0, min(stop_idx, T - 1))
        marks = {start_idx: f"{times[start_idx]:.2f}", stop_idx: f"{times[stop_idx]:.2f}"}
        new_t = int(min(max(int(cur_t), start_idx), stop_idx))   # clamp current frame
        return ([start_idx, stop_idx], start_idx, stop_idx, marks, new_t,
                float(times[start_idx]), float(times[stop_idx]))

    @app.callback(
        Output("graph3d", "figure", allow_duplicate=True),
        Input("time_window", "data"),
        State("t_idx", "value"),
        State("time_scale_store", "data"),
        State("camera3d_store", "data"),
        prevent_initial_call=True,
    )
    def rebuild_3d_on_window(window, t_idx_val, ts, camera):
        start_idx, stop_idx = int(window[0]), int(window[1])
        fig, _ = make_3d_base_figure(data, time_scale=ts or time_scale, time_stride=time_stride,
                                     xy_lims=xy_lims, t_start_idx=start_idx, t_stop_idx=stop_idx)
        t_int = int(min(max(int(t_idx_val), start_idx), stop_idx))
        z0 = float(_z_from_time(float(times[t_int]), ts or time_scale))
        pts_x = _to_list_none(data["xy"][t_int, :, 0])
        pts_y = _to_list_none(data["xy"][t_int, :, 1])
        t_val_3d = float(times[t_int])
        fig.data[pts_trace_idx]["x"] = pts_x
        fig.data[pts_trace_idx]["y"] = pts_y
        fig.data[pts_trace_idx]["z"] = [z0] * len(pts_x)
        fig.data[pts_trace_idx]["customdata"] = [t_val_3d] * len(pts_x)
        fig.data[plane_trace_idx]["z"] = [z0, z0, z0, z0]
        fig.update_layout(title=f"3D Trajectories (t = {t_val_3d:.2f})")
        if camera:
            fig.update_layout(scene_camera=camera)
        return fig

    @app.callback(
        Output("graph2d", "figure"),
        Output("graph3d", "figure"),
        Input("t_idx", "value"),
        Input("show_labels", "value"),
        Input("axis_store", "data"),
        Input("show_ellipses", "value"),
        Input("show_rates", "value"),
        Input("n_outliers_dropdown", "value"),
        State("time_scale_store", "data"),
        State("camera3d_store", "data"),
    )
    def update_figs(t_idx, show_labels_val, axis_store, show_ellipses_val, show_rates_val, n_outliers_val, ts, camera):
        t_idx = int(t_idx)
        show = "yes" in (show_labels_val or [])
        ellipses = "yes" in (show_ellipses_val or [])
        level = 0.95
        n_outliers = int(n_outliers_val or 0)
        rates_open = "yes" in (show_rates_val or [])
        show_outliers = n_outliers > 0 and rates_open and rates_data is not None
        t_val = float(times[t_idx])

        fig_2d = make_figure(
            t_idx, data,
            show_labels=show,
            axis_override=axis_store,
            show_ellipses=ellipses,
            confidence_level=level,
        )

        # --- Outlier edges ---
        if show_outliers:
            rec_col_idx = int(np.argmin(np.abs(rates_data["reconstructed_times"] - t_val)))
            lr = rates_data["log_ratio"][:, rec_col_idx]
            valid_mask = np.isfinite(lr)
            valid_idx = np.where(valid_mask)[0]
            sorted_by_lr = valid_idx[np.argsort(lr[valid_idx])]
            under_pairs = sorted_by_lr[:n_outliers]           # most negative (under-estimated)
            over_pairs  = sorted_by_lr[-n_outliers:][::-1]    # most positive (over-estimated)

            pos     = data["xy_float"][t_idx]
            sid_arr = data["sample_ids"]

            def _edge_lists(pair_indices):
                xs, ys, cds = [], [], []
                for pi in pair_indices:
                    pair_id  = rates_data["pair_ids"][pi]
                    lr_val   = float(lr[pi])
                    pop_label = rates_data["pair_pop_label"][pi]
                    s1, s2   = pair_id.split(" × ")
                    i1 = np.where(sid_arr == s1)[0]
                    i2 = np.where(sid_arr == s2)[0]
                    if not len(i1) or not len(i2):
                        continue
                    p1, p2 = pos[int(i1[0])], pos[int(i2[0])]
                    if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
                        continue
                    xs  += [float(p1[0]), float(p2[0]), None]
                    ys  += [float(p1[1]), float(p2[1]), None]
                    cds += [[pair_id, lr_val, pop_label],
                            [pair_id, lr_val, pop_label],
                            [None, None, None]]
                return xs, ys, cds

            over_x,  over_y,  over_cd  = _edge_lists(over_pairs)
            under_x, under_y, under_cd = _edge_lists(under_pairs)
            fig_2d.data[outlier_over_trace_idx_2d].x          = over_x
            fig_2d.data[outlier_over_trace_idx_2d].y          = over_y
            fig_2d.data[outlier_over_trace_idx_2d].customdata = over_cd
            fig_2d.data[outlier_under_trace_idx_2d].x          = under_x
            fig_2d.data[outlier_under_trace_idx_2d].y          = under_y
            fig_2d.data[outlier_under_trace_idx_2d].customdata = under_cd

        z0 = float(_z_from_time(float(times[t_idx]), ts or time_scale))
        pts_x = _to_list_none(data["xy"][t_idx, :, 0])
        pts_y = _to_list_none(data["xy"][t_idx, :, 1])

        patched = Patch()
        patched["data"][pts_trace_idx]["x"] = pts_x
        patched["data"][pts_trace_idx]["y"] = pts_y
        patched["data"][pts_trace_idx]["z"] = [z0] * len(pts_x)
        patched["data"][pts_trace_idx]["customdata"] = [t_val] * len(pts_x)
        patched["data"][plane_trace_idx]["z"] = [z0, z0, z0, z0]
        patched["layout"]["title"]["text"] = f"3D Trajectories (t = {t_val:.2f})"
        if camera:
            patched["layout"]["scene"]["camera"] = camera

        return fig_2d, patched

    @app.callback(
        Output("graph2d", "figure", allow_duplicate=True),
        Input("graph2d", "hoverData"),
        State("t_idx", "value"),
        prevent_initial_call=True,
    )
    def highlight_on_hover(hover_data, t_idx_val):
        if not has_covariances:
            return no_update
        if not hover_data or not hover_data.get("points"):
            return no_update

        point = hover_data["points"][0]
        curve_num = point.get("curveNumber", -1)
        level = 0.95
        t_int = int(t_idx_val)

        patched = Patch()

        if curve_num == scatter_trace_idx_2d:
            # Hover on a scatter point → draw its ellipse in the highlight trace
            sample_idx = point.get("pointIndex")
            mu = data["xy_float"][t_int][sample_idx]
            cov = data["covariances"][t_int][sample_idx]
            if not (np.any(np.isnan(mu)) or np.any(np.isnan(cov))):
                ex, ey = ellipse_points(mu, cov, level=level, n_pts=60)
                patched["data"][highlight_trace_idx_2d]["x"] = ex.tolist() + [ex[0]]
                patched["data"][highlight_trace_idx_2d]["y"] = ey.tolist() + [ey[0]]
                patched["data"][highlight_trace_idx_2d]["line"]["color"] = str(data["color"][sample_idx])
            else:
                patched["data"][highlight_trace_idx_2d]["x"] = []
                patched["data"][highlight_trace_idx_2d]["y"] = []
        else:
            # Hover off scatter (e.g. on an ellipse or empty space) → clear highlight
            patched["data"][highlight_trace_idx_2d]["x"] = []
            patched["data"][highlight_trace_idx_2d]["y"] = []

        return patched

    @app.callback(
        Output("graph2d", "figure", allow_duplicate=True),
        Input("graph_rates", "hoverData"),
        State("t_idx", "value"),
        prevent_initial_call=True,
    )
    def highlight_pair_on_rates_hover(hover_data, t_idx_val):
        patched = Patch()
        def clear():
            patched["data"][pair_edge_trace_idx_2d]["x"] = []
            patched["data"][pair_edge_trace_idx_2d]["y"] = []
            patched["data"][pair_edge_trace_idx_2d]["marker"]["color"] = []

        if not hover_data or not hover_data.get("points"):
            clear()
            return patched

        point = hover_data["points"][0]
        cd = point.get("customdata")
        if cd is None or not isinstance(cd, (list, tuple)) or len(cd) < 1:
            clear()
            return patched

        pair_id = str(cd[0])
        parts = pair_id.split(" × ")
        if len(parts) != 2:
            clear()
            return patched

        s1, s2 = parts
        t_int = int(t_idx_val)
        sid_arr = data["sample_ids"]
        idx1_arr = np.where(sid_arr == s1)[0]
        idx2_arr = np.where(sid_arr == s2)[0]
        if not len(idx1_arr) or not len(idx2_arr):
            clear()
            return patched

        idx1, idx2 = int(idx1_arr[0]), int(idx2_arr[0])
        pos = data["xy_float"][t_int]
        p1, p2 = pos[idx1], pos[idx2]
        if np.any(np.isnan(p1)) or np.any(np.isnan(p2)):
            clear()
            return patched

        patched["data"][pair_edge_trace_idx_2d]["x"] = [float(p1[0]), float(p2[0])]
        patched["data"][pair_edge_trace_idx_2d]["y"] = [float(p1[1]), float(p2[1])]
        patched["data"][pair_edge_trace_idx_2d]["marker"]["color"] = [
            str(data["color"][idx1]),
            str(data["color"][idx2]),
        ]
        return patched

    @app.callback(
        Output("graph_rates", "figure", allow_duplicate=True),
        Input("graph2d", "hoverData"),
        State("t_idx", "value"),
        prevent_initial_call=True,
    )
    def highlight_outlier_in_rates(hover_data, t_idx_val):
        """When hovering over an outlier edge in the 2D view, highlight the pair in the rates scatter."""
        patched = Patch()
        def clear():
            patched["data"][3]["x"] = []
            patched["data"][3]["y"] = []
            patched["data"][3]["marker"]["color"] = []

        if rates_data is None or not hover_data or not hover_data.get("points"):
            clear(); return patched

        point = hover_data["points"][0]
        curve_num = point.get("curveNumber", -1)
        if curve_num not in (outlier_over_trace_idx_2d, outlier_under_trace_idx_2d):
            clear(); return patched

        cd = point.get("customdata")
        if not cd or cd[0] is None:
            clear(); return patched

        pair_id = str(cd[0])
        pair_idx_arr = np.where(rates_data["pair_ids"] == pair_id)[0]
        if not len(pair_idx_arr):
            clear(); return patched

        t_val       = float(times[int(t_idx_val)])
        in_col_idx  = int(np.argmin(np.abs(rates_data["input_times"]         - t_val)))
        rec_col_idx = int(np.argmin(np.abs(rates_data["reconstructed_times"] - t_val)))
        pi  = int(pair_idx_arr[0])
        x_v = float(rates_data["input"].iloc[pi, in_col_idx])
        y_v = float(rates_data["reconstructed"].iloc[pi, rec_col_idx])
        if not np.isfinite(x_v) or not np.isfinite(y_v) or x_v <= 0 or y_v <= 0:
            clear(); return patched

        color = "#d62728" if curve_num == outlier_over_trace_idx_2d else "#1f77b4"
        patched["data"][3]["x"] = [x_v]
        patched["data"][3]["y"] = [y_v]
        patched["data"][3]["marker"]["color"] = [color]
        return patched

    @app.callback(
        Output("rates_panel", "style"),
        Input("show_rates", "value"),
    )
    def toggle_rates_panel(show_rates_val):
        if "yes" in (show_rates_val or []):
            return {"display": "block", "marginTop": "4px"}
        return {"display": "none", "marginTop": "4px"}

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
        patched["data"][3]["x"] = []    # clear outlier highlight on time change
        patched["data"][3]["y"] = []
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
    p.add_argument("--time_stride", type=int, default=1,
                   help="Keep every Nth time point in 3D trajectories (default 1; increase for very large datasets)")
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
    p.add_argument("--covariances", type=str, default=None,
                   help="Path to covariances parquet (enables uncertainty ellipses)")
    p.add_argument("--confidence_level", type=float, default=0.68,
                   help="Confidence level for ellipses: 0.68, 0.90, or 0.95 (default 0.68)")
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
        covariances_path=args.covariances,
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

    meta = {
        "embeddings": args.parquet,
        "labels":     args.labels,
        "colors":     args.colors,
        "rates":      args.rates or "—",
        "recon. rates": args.reconstructed_rates or "—",
        "covariances": args.covariances or "—",
        "N_max":      args.N_max,
        "time_scale": args.time_scale,
        "min_time":   args.min_time,
        "max_time":   args.max_time,
    }

    app = make_app(
        data,
        time_scale=args.time_scale,
        time_stride=args.time_stride,
        xy_lims=lims,
        rates_data=rates_data,
        confidence_level=args.confidence_level,
        meta=meta,
    )
    import socket

    def find_open_port(start, max_tries=20):
        for port in range(start, start + max_tries):
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                try:
                    s.bind(("", port))
                    return port
                except OSError:
                    continue
        raise RuntimeError(f"No open port found in range {start}–{start + max_tries - 1}")

    port = find_open_port(args.port)
    if port != args.port:
        print(f"Port {args.port} in use, using {port} instead.")
    app.run(debug=False, port=port)
