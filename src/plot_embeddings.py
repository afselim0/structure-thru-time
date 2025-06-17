#!/usr/bin/env python3
"""
plot_embeddings_voxel_consensus.py — 3‑D temporal PCA visualiser with
external palette support and natural‑log Z scaling
=====================================================================

Features:
 * Uses `--palette` to color populations from `population_colors.csv`.
 * Skips or adjusts the iso‑surface when volume is empty or no surface is found.
 * `--no-surface` flag to disable the marching‑cubes mesh entirely.
 * `--median-window` to control smoothing of the population median line.
 * `--log-time` flag to use natural‑log(time) on Z axis.

Usage:
    python plot_embeddings_voxel_consensus.py \
        --input          embeddings.csv \
        --output         out_dir \
        --palette        population_colors.csv \
        [--mode surface|skeleton] [--no-surface] \
        [--log-time] [--no-individual] [--no-median] [--no-ribs] \
        [--median-window N]

Dependencies:
    numpy, pandas, scipy, scikit-image, matplotlib, plotly
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from matplotlib.path import Path as MplPath
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

# ----------------------- tunables -------------------------------------------
CENTRAL_PERCENTILE    = 0.5
GRID_RES_XY           = 256
GAUSS_SIGMA           = 1.5
TUBE_ALPHA            = 0.28
INDIVIDUAL_LINE_ALPHA = 0.20
MEDIAN_SMOOTH_WINDOW  = 5
MEDIAN_LINE_WIDTH     = 5
# ---------------------------------------------------------------------------

def configure_logging(level=logging.INFO):
    logging.basicConfig(
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%H:%M:%S', level=level
    )

# ------------------------ I/O -----------------------------------------------
def load_embeddings(path: Path) -> pd.DataFrame:
    logging.info(f"Reading embeddings from {path}")
    df = pd.read_parquet(path)
    if 'sample_time' not in df.columns:
        df['sample_time'] = 0.0
    return df

def load_palette(path: Path) -> Dict[str, str]:
    # df = pd.read_csv(path)
    df = pd.read_parquet(path)
    if not {'population','hex_colour'}.issubset(df.columns):
        raise ValueError("Palette CSV must contain 'population' and 'hex_colour' columns")
    return dict(zip(df.population, df.hex_colour))

# ------------------------ hull & raster -------------------------------------
def slice_convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) < 3:
        return points
    center = points.mean(axis=0)
    d = np.linalg.norm(points - center, axis=1)
    keep = points[d <= np.percentile(d, CENTRAL_PERCENTILE * 100)]
    if len(keep) < 3:
        return keep
    hull = ConvexHull(keep)
    return keep[hull.vertices]

def rasterise_polygon(poly: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if len(poly) < 3:
        return np.zeros((len(ys), len(xs)), dtype=bool)
    path = MplPath(poly)
    xv, yv = np.meshgrid(xs, ys)
    pts = np.vstack([xv.ravel(), yv.ravel()]).T
    return path.contains_points(pts).reshape(len(ys), len(xs))

# ---------------------- volume builder --------------------------------------
def build_volume(
    df: pd.DataFrame, pop: str, grid_res: int, log_time: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sub = df[df.population == pop]
    xs_all, ys_all = sub['X1'], sub['X2']
    pad_x = 0.05 * (xs_all.max() - xs_all.min())
    pad_y = 0.05 * (ys_all.max() - ys_all.min())
    xs = np.linspace(xs_all.min() - pad_x, xs_all.max() + pad_x, grid_res)
    ys = np.linspace(ys_all.min() - pad_y, ys_all.max() + pad_y, grid_res)
    times = np.sort(sub.time.unique())
    ts = np.log(times) if log_time else times
    vol = np.zeros((len(ts), grid_res, grid_res), dtype=bool)
    for i, t in enumerate(times):
        pts = sub[sub.time == t][['X1','X2']].values
        vol[i] = rasterise_polygon(slice_convex_hull(pts), xs, ys)
    return vol, xs, ys, ts

# -------------------- iso-surface extraction -------------------------------
def surface_from_volume(
    vol: np.ndarray, xs: np.ndarray, ys: np.ndarray, ts: np.ndarray, sigma: float
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    blurred = gaussian_filter(vol.astype(float), sigma=sigma)
    lo, hi = blurred.min(), blurred.max()
    if hi <= lo:
        logging.warning(
            "Empty volume for population; skipping iso-surface (data range %.3f-%.3f)", lo, hi
        )
        return None
    level = 0.5
    if not (lo < level < hi):
        level = lo + (hi - lo) * 0.5
        logging.warning(
            "Adjusted marching_cubes level to %.3f (data range %.3f-%.3f)", level, lo, hi
        )
    try:
        verts, faces, _, _ = marching_cubes(blurred, level=level)
    except RuntimeError as e:
        logging.warning("No surface found at iso value: %s; skipping", e)
        return None
    nx, ny, nz = len(xs)-1, len(ys)-1, len(ts)-1
    vx = xs[0] + verts[:,2]/nx * (xs[-1] - xs[0])
    vy = ys[0] + verts[:,1]/ny * (ys[-1] - ys[0])
    vz = ts[0] + verts[:,0]/nz * (ts[-1] - ts[0])
    return np.stack([vx, vy, vz], axis=1), faces

# -------------------- plotting per population ------------------------------
def median_df(df: pd.DataFrame, log_time: bool) -> pd.DataFrame:
    med = (
        df.groupby('time', as_index=False)[['X1','X2']]
          .median()
          .sort_values('time')
    )
    med['time_plot'] = np.log(med.time) if log_time else med.time
    if MEDIAN_SMOOTH_WINDOW > 1:
        med['X1'] = (
            med['X1']
               .rolling(window=MEDIAN_SMOOTH_WINDOW, center=True, min_periods=1)
               .mean()
        )
        med['X2'] = (
            med['X2']
               .rolling(window=MEDIAN_SMOOTH_WINDOW, center=True, min_periods=1)
               .mean()
        )
    return med

def plot_population(
    df: pd.DataFrame, pop: str, vol_data, fig: go.Figure, colour: str,
    mode: str, log_time: bool, show_traj: bool, show_median: bool,
    show_ribs: bool, show_surface: bool
) -> None:
    vol, xs, ys, ts = vol_data
    sub = df[df.population == pop].copy()
    sub['time_plot'] = np.log(sub.time) if log_time else sub.time

    if show_traj:
        fig.add_traces(
            px.line_3d(
                sub, x='X1', y='X2', z='time_plot',
                line_group='sample_id', hover_name='population'
            ).update_traces(
                line=dict(color=colour), opacity=INDIVIDUAL_LINE_ALPHA,
                hovertemplate='Population: %{hovertext}<br>Sample: %{customdata[0]}<extra></extra>',
                customdata=sub[['sample_id']].values
            ).data
        )

    if show_median:
        med = median_df(sub, log_time)
        fig.add_trace(
            go.Scatter3d(
                x=med['X1'], y=med['X2'],
                z=med['time_plot'], mode='lines',
                line=dict(color=colour, width=MEDIAN_LINE_WIDTH),
                showlegend=False, legendgroup=pop,
                hovertemplate='Population: '+pop+'<br>Time: %{z:.2f}<extra></extra>'
            )
        )
        # if not med.empty:
        #     z_base = ts[0] - (ts[-1] - ts[0]) * 0.05
        #     fig.add_trace(
        #         go.Scatter3d(
        #             x=[med['X1'].iloc[0]],
        #             y=[med['X2'].iloc[0]],
        #             z=[z_base],
        #             mode='text', text=[pop], textposition='bottom center',
        #             textfont=dict(size=10, color=colour),
        #             showlegend=False, hoverinfo='none'
        #         )
        #     )

    if show_ribs:
        for i, t in enumerate(ts):
            mask = vol[i]
            if mask.sum() < 3:
                continue
            coords = np.column_stack([xs[mask.any(axis=0)], ys[mask.any(axis=1)]])
            hull = slice_convex_hull(coords)
            if len(hull) < 3:
                continue
            fig.add_trace(
                go.Scatter3d(
                    x=hull[:,0], y=hull[:,1], z=np.full(len(hull), t),
                    mode='lines', line=dict(color=colour, width=1),
                    showlegend=False, legendgroup=pop
                )
            )

    if show_surface and mode != 'skeleton':
        surf = surface_from_volume(vol, xs, ys, ts, GAUSS_SIGMA)
        if surf is not None:
            verts, faces = surf
            i, j, k = faces.T
            fig.add_trace(
                go.Mesh3d(
                    x=verts[:,0], y=verts[:,1], z=verts[:,2],
                    i=i, j=j, k=k, color=colour, opacity=TUBE_ALPHA,
                    name=pop, legendgroup=pop, showlegend=True
                )
            )

# -------------------- build full figure -------------------------------------
def build_plot(
    df: pd.DataFrame, palette: Dict[str,str], mode: str, log_time: bool,
    show_traj: bool, show_median: bool, show_ribs: bool, show_surface: bool
) -> go.Figure:
    pops = sorted(np.unique(df.population))
    fig = go.Figure()
    for pop in pops:
        colour = palette.get(pop, '#888888')
        vol_data = build_volume(df, pop, GRID_RES_XY, log_time)
        plot_population(
            df, pop, vol_data, fig, colour, mode, log_time,
            show_traj, show_median, show_ribs, show_surface
        )
    fig.update_layout(
        scene=dict(
            xaxis_title='X1', yaxis_title='X2',
            zaxis_title='time (generations ago)' if log_time else 'time'
        ),
        showlegend=True, margin=dict(l=0,r=0,t=0,b=0)
    )
    return fig

# ---------------------- main --------------------------------s---------------
def main():
    global MEDIAN_SMOOTH_WINDOW
    ap = argparse.ArgumentParser()
    ap.add_argument('--input','-i',required=True,help='Path to embeddings CSV/parquet')
    ap.add_argument('--output','-o',required=True,help='Directory for HTML output')
    ap.add_argument('--palette',required=True,help='CSV with population,hex_colour')
    ap.add_argument('--mode',choices=['skeleton','surface'],default='surface',
                    help='Level of detail for iso-surface')
    ap.add_argument('--no-surface',action='store_true',
                    help='Skip marching-cubes mesh')
    ap.add_argument('--log-time',action='store_true',
                    help='Use natural-log(time) on Z axis')
    ap.add_argument('--no-individual',action='store_true',
                    help='Hide individual trajectories')
    ap.add_argument('--no-median',action='store_true',
                    help='Hide median curve')
    ap.add_argument('--no-ribs','-R',action='store_true',
                    help='Hide slice hull outlines')
    ap.add_argument('--median-window',type=int,default=MEDIAN_SMOOTH_WINDOW,
                    help='Odd number of slices for median smoothing')
    args = ap.parse_args()

    MEDIAN_SMOOTH_WINDOW = args.median_window
    configure_logging()
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_embeddings(Path(args.input))
    # df = reshape_embeddings(emb)
    logging.info(f"Populations: {', '.join(sorted(np.unique(df.population)))}")

    palette = load_palette(Path(args.palette))
    fig = build_plot(
        df, palette, args.mode, args.log_time,
        not args.no_individual, not args.no_median, not args.no_ribs,
        not args.no_surface
    )

    fig.update_layout(
    scene=dict(
        zaxis=dict(
            tickvals=[np.log(n) for n in [10,20,50,100,200,500,1000,2000,5000,10000,20000]],  # The actual values on the axis
            ticktext=["10", "20", "50", "100", "200", "500","1000","2000","5000","10000","20000"] # What gets displayed
            # type="log"  # Set axis to logarithmic scale
        )
    )
)

    out_file = outdir / f"dynamic_3d_plot_{args.mode}.html"
    pio.write_html(fig, file=str(out_file), auto_open=False)
    logging.info(f"Saved plot → {out_file}")

if __name__ == '__main__':
    main()
