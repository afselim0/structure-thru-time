#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
from matplotlib.path import Path as MplPath

# ----------------------- tunables ------------------------------------------
sample_line_width = 2
sample_line_alpha = 0.5

median_line_width = 5
median_line_alpha = 1.0

# Volume and surface parameters
central_percentile = 0.95
grid_res_xy = 256
gauss_sigma = 1.5
tube_alpha = 0.18
# ---------------------------------------------------------------------------
class PlotEmbeddings:
    def __init__(self, 
                 embeddings_df: pd.DataFrame,
                 labels_df: pd.DataFrame = None,
                 colors_df: pd.DataFrame = None,
                 generation_time: int = None
                 ):
        
        # Loading population labels
        if labels_df is None:
            embeddings_df["population"] = "unlabelled_pop"
            colors_df = None # Set colors_df to None to trigger default colors if no labels provided
        else:
            embeddings_df = embeddings_df.merge(
                labels_df[["sample_id","population"]],
                on='sample_id',
                how='left'
            )
            embeddings_df["population"] = embeddings_df["population"].fillna("unlabelled_pop")

        self.populations = embeddings_df["population"].unique()
        self.num_populations = embeddings_df["population"].nunique()

        # Loading population colors
        if colors_df is None:
            colors = px.colors.qualitative.Safe # A colorblind-friendly palette if no colors assigned
            colors_dict = {pop:colors[i%len(colors)] for i,pop in enumerate(self.populations)}
            embeddings_df["color"] = embeddings_df["population"].map(colors_dict)
        else:
            embeddings_df = embeddings_df.merge(
                colors_df[["population","color"]],
                on='population',
                how='left'
            )
            embeddings_df["color"] = embeddings_df["color"].fillna("#808080") # Default gray for missing colors
            unique_colors = embeddings_df[["population","color"]].drop_duplicates()
            colors_dict = {row["population"]:row["color"] for _,row in unique_colors.iterrows()}
            
        self.colors_dict = colors_dict
        if generation_time is not None:
            embeddings_df["time"] = embeddings_df["time"] * generation_time
        
        self.times = embeddings_df["time"].unique()
        self.generation_time = generation_time
        self.embeddings_df = embeddings_df
        self.fig = None
        

    def init_plot(self, time_scale: str = "log"):
        self.fig = go.Figure()
        
        # Setting axis labels
        x_label = "X1"
        y_label = "X2"
        z_label = "Time (generations ago)" if self.generation_time is None else "Time (years ago)"

        # Setting axis styles
        axis_style = dict(
            showline=True,
            showgrid=True,
            showbackground=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            tickcolor="black",
            gridcolor="white",
            backgroundcolor="whitesmoke",
            tickmode="array"
        )
        
        # Setting equal scaling for X and Y axes
        x_min = self.embeddings_df["X1"].min()
        x_max = self.embeddings_df["X1"].max()
        y_min = self.embeddings_df["X2"].min()
        y_max = self.embeddings_df["X2"].max()

        common_min = min(x_min, y_min)
        common_max = max(x_max, y_max)
        pad = 0.05 * (common_max - common_min)
        common_min -= pad
        common_max += pad

        x_range = [common_min, common_max]
        y_range = [common_min, common_max]

        # Setting gridlines and ticks for time axis
        min_time = self.embeddings_df["time"].min()
        min_time = 1 if min_time <= 0 else min_time
        max_time = self.embeddings_df["time"].max()
        z_min_decade = int(np.floor(np.log10(min_time)))
        z_max_decade = int(np.ceil(np.log10(max_time)))
        z_ticks = 10 ** np.arange(z_min_decade, z_max_decade + 1)

        self.fig.update_layout(
            scene=dict(
                xaxis=dict(title=x_label, range=x_range, zeroline=True, **axis_style),
                yaxis=dict(title=y_label, range=y_range, zeroline=True, **axis_style),
                zaxis=dict(title=z_label, tickvals = z_ticks, ticktext = [str(int(z)) for z in z_ticks], **axis_style, type=time_scale),
                aspectmode="manual",
                aspectratio=dict(x=1, y=1, z=1)
            )
        )

    def add_legend(self):
        for pop in self.populations:
            color = self.colors_dict.get(pop, "#808080")

            self.fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(size=10, color=color),
                    name=pop,
                    legendgroup=pop,
                    showlegend=True
                )
            )

    def save_static(self,filename, format="png", width=600, height=600, scale=2, eye=dict(x=1.8, y=1.8, z=0.3), up=dict(x=0, y=0, z=1)):
        self.fig.update_layout(scene=dict(camera=dict(eye=eye, up=up)))
        pio.write_image(fig = self.fig, file = f"{filename}.{format}",format=format,width=width,height=height,scale=scale)

    def add_sample_trajectories(self,line_width: int = sample_line_width,alpha: float = sample_line_alpha):
        samples = self.embeddings_df["sample_id"].unique()
        for sample in samples:
            sample_data = self.embeddings_df[self.embeddings_df["sample_id"] == sample]
            population = sample_data["population"].values[0]
            color = self.colors_dict.get(population,'#888888')
            self.fig.add_trace(
                go.Scatter3d(
                    x=sample_data["X1"],
                    y=sample_data["X2"],
                    z=sample_data["time"],
                    mode='lines',
                    line=dict(color=color, width=line_width),
                    opacity=alpha,
                    name=f'{population} - {sample}',
                    hovertemplate=f"{population}: {sample}<br>Time: %{{z:.2f}}<extra></extra>",
                    showlegend=False,
                    legendgroup=population
                )
            )

    def add_population_medians(self,line_width: int = median_line_width, alpha: float = median_line_alpha):
        median_df = (
            self.embeddings_df
            .groupby(["population", "time"], as_index=False)
            .agg(
                X1=("X1", "median"),
                X2=("X2", "median"),
                color=("color", "first"),
            )
        )

        for population in self.populations:
            pop_data = median_df[median_df["population"] == population]
            color = self.colors_dict.get(population, "#808080")
            self.fig.add_trace(
                go.Scatter3d(
                    x=pop_data["X1"],
                    y=pop_data["X2"],
                    z=pop_data["time"],
                    mode='lines',
                    line=dict(color=color, width=line_width),
                    name=f'{population} Median',
                    hovertemplate=f"{population} Median<br>Time: %{{z:.2f}}<extra></extra>",
                    showlegend=True,
                    legendgroup=population,
                    opacity=alpha
                )
            )

    def add_convex_hull(self, *, verbose: bool = False, debug_dir: str | None = None):
        for population in self.populations:
            color = self.colors_dict.get(population, "#808080")

            vol, xs, ys, ts = build_volume(
                self.embeddings_df,
                population,
                grid_res_xy,
                verbose=verbose,
                debug_dir=debug_dir,
            )

            surf = surface_from_volume(vol, xs, ys, ts, gauss_sigma)
            if surf is not None:
                verts, faces = surf
                i, j, k = faces.T
                self.fig.add_trace(
                    go.Mesh3d(
                        x=verts[:, 0],
                        y=verts[:, 1],
                        z=verts[:, 2],
                        i=i,
                        j=j,
                        k=k,
                        color=color,
                        opacity=tube_alpha,
                        name=population,
                        legendgroup=population,
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )
            else:
                if verbose:
                    print(f"[{population}] surface_from_volume returned None (no surface)")

def slice_convex_hull(points: np.ndarray, return_stats: bool = False):
    """
    Computes the convex hull for a given population at a given time point.

    Returns:
      - poly (Nx2 array) of hull vertices (ordered)
      - if return_stats=True, also returns stats dict
    """
    stats = {
        "n_points": int(len(points)),
        "n_keep": 0,
        "n_hull_verts": 0,
    }

    if len(points) < 3:
        if return_stats:
            return points, stats
        return points

    center = points.mean(axis=0)
    d = np.linalg.norm(points - center, axis=1)

    # central trimming (remove far outliers)
    keep = points[d <= np.percentile(d, central_percentile * 100)]
    stats["n_keep"] = int(len(keep))

    if len(keep) < 3:
        if return_stats:
            return keep, stats
        return keep

    hull = ConvexHull(keep)
    poly = keep[hull.vertices]
    stats["n_hull_verts"] = int(len(poly))

    if return_stats:
        return poly, stats
    return poly

def rasterise_polygon(poly: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    if len(poly)<3: return np.zeros((len(ys),len(xs)),dtype=bool)
    path=MplPath(poly)
    xv,yv=np.meshgrid(xs,ys)
    pts=np.vstack([xv.ravel(),yv.ravel()]).T
    return path.contains_points(pts).reshape(len(ys),len(xs))

def _safe_log10(x: np.ndarray) -> np.ndarray:
    # avoid -inf for 0/negative times; caller should ideally ensure time>0 already
    x = np.asarray(x, dtype=float)
    x = np.where(x <= 0, np.nan, x)
    return np.log10(x)

def _make_time_bins(times: np.ndarray, n_bins: int, time_scale: str):
    """
    Returns:
      bin_edges (n_bins+1), bin_centers (n_bins)
    Bins in log10(time) if time_scale == "log", else in time.
    """
    times = np.asarray(times, dtype=float)
    if time_scale == "log":
        t = _safe_log10(times)
        t = t[~np.isnan(t)]
    else:
        t = times

    if len(t) == 0:
        return None, None

    lo, hi = np.nanmin(t), np.nanmax(t)
    if lo == hi:
        # degenerate: everything at same time
        edges = np.array([lo, hi + 1e-9])
        centers = np.array([lo])
        return edges, centers

    edges = np.linspace(lo, hi, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return edges, centers

def build_volume(
    df: pd.DataFrame,
    pop: str,
    grid_res: int,
    *,
    verbose: bool = False,
    debug_top_k: int = 5,
    debug_dir: str | None = None,
):
    """
    Build a boolean volume vol[t, y, x] for one population by:
      1) slicing points at each time
      2) computing a 2D convex hull polygon (with central trimming)
      3) rasterising the polygon on an XY grid

    Diagnostics (verbose):
      - per-slice: n_points, n_keep, n_hull_verts, occ_frac
      - summary: empty slices, points/bin stats, emptiest bins

    Optional:
      - debug_dir: if set, writes CSVs for the emptiest bins (raw points).
    """
    sub = df[df.population == pop]
    if sub.empty:
        if verbose:
            print(f"[{pop}] No rows for population; returning empty volume")
        xs = np.linspace(0, 1, grid_res)
        ys = np.linspace(0, 1, grid_res)
        ts = np.array([0.0])
        vol = np.zeros((len(ts), grid_res, grid_res), dtype=bool)
        return vol, xs, ys, ts

    xs_all, ys_all = sub["X1"], sub["X2"]
    pad_x = 0.05 * (xs_all.max() - xs_all.min()) if xs_all.max() > xs_all.min() else 1.0
    pad_y = 0.05 * (ys_all.max() - ys_all.min()) if ys_all.max() > ys_all.min() else 1.0
    xs = np.linspace(xs_all.min() - pad_x, xs_all.max() + pad_x, grid_res)
    ys = np.linspace(ys_all.min() - pad_y, ys_all.max() + pad_y, grid_res)

    times = np.sort(sub.time.unique())
    ts = times
    vol = np.zeros((len(ts), grid_res, grid_res), dtype=bool)

    # ---- diagnostics accumulators ----
    slice_stats = []  # dicts per slice
    points_per_slice = []
    empty_slices = 0

    for i, t in enumerate(times):
        pts = sub[sub.time == t][["X1", "X2"]].values
        n_pts = len(pts)
        points_per_slice.append(n_pts)

        poly, poly_stats = slice_convex_hull(pts, return_stats=True)
        mask = rasterise_polygon(poly, xs, ys)
        vol[i] = mask

        occ = float(mask.mean())  # fraction of grid occupied
        is_empty = (n_pts < 3) or (mask.sum() == 0)
        if is_empty:
            empty_slices += 1

        stat = {
            "idx": i,
            "time": float(t),
            "n_points": int(n_pts),
            "n_keep": int(poly_stats.get("n_keep", 0)),
            "n_hull_verts": int(poly_stats.get("n_hull_verts", 0)),
            "occ_frac": occ,
            "mask_sum": int(mask.sum()),
        }
        slice_stats.append(stat)

        if verbose:
            # concise per-slice line (easy to scan)
            print(
                f"[{pop}] slice {i:03d}/{len(times)-1:03d} "
                f"t={float(t):.6g} n={n_pts:5d} keep={stat['n_keep']:5d} "
                f"verts={stat['n_hull_verts']:4d} occ={occ:.6f} "
                f"{'EMPTY' if is_empty else ''}"
            )

    # ---- summary diagnostics ----
    if verbose:
        arr = np.asarray(points_per_slice, dtype=int)
        empties = [s for s in slice_stats if s["n_points"] == 0]
        empties_sorted = sorted(empties, key=lambda d: d["idx"])[:debug_top_k]

        # points/bin min/median/max
        pmin = int(arr.min()) if len(arr) else 0
        pmed = int(np.median(arr)) if len(arr) else 0
        pmax = int(arr.max()) if len(arr) else 0

        # "emptiest" bins: smallest n_points, then smallest occ_frac
        emptiest = sorted(slice_stats, key=lambda d: (d["n_points"], d["occ_frac"]))[:debug_top_k]
        emptiest_str = ", ".join([f"{d['idx']}:{d['n_points']}" for d in emptiest])

        empty_frac = (empty_slices / len(times)) * 100.0 if len(times) else 0.0
        print(
            f"[{pop}] slices={len(times)}, empty={empty_slices} (empty_frac={empty_frac:.2f}%) "
            f"; points/bin min={pmin}, med={pmed}, max={pmax}"
        )
        print(f"    emptiest bins: {emptiest_str}")

        # Optional: dump CSVs for worst slices for later 2D inspection
        if debug_dir is not None:
            debug_path = Path(debug_dir)
            debug_path.mkdir(parents=True, exist_ok=True)
            for d in emptiest:
                t = d["time"]
                pts = sub[sub.time == t][["X1", "X2"]].copy()
                out_csv = debug_path / f"{pop}_slice_{d['idx']:03d}_t_{t:.6g}.csv"
                pts.to_csv(out_csv, index=False)

    return vol, xs, ys, ts

def _print_volume_summary(pop: str, counts: list[int], empty: int, n_slices: int, mode: str):
    arr = np.array(counts, dtype=int) if len(counts) else np.array([], dtype=int)
    if arr.size == 0:
        print(f"[{pop}] mode={mode}: no slices.")
        return
    print(
        f"[{pop}] mode={mode}: slices={n_slices}, empty={empty} "
        f"(empty_frac={empty/n_slices:.2%}); "
        f"points/bin min={arr.min()}, med={int(np.median(arr))}, max={arr.max()}"
    )
    # show a few emptiest bins for quick debugging
    worst = np.argsort(arr)[:5]
    print(f"    emptiest bins: " + ", ".join([f"{i}:{arr[i]}" for i in worst]))

def surface_from_volume(vol: np.ndarray, xs: np.ndarray, ys: np.ndarray, ts: np.ndarray,
                        sigma_xy: float, sigma_t: float = 0.0):
    # NOTE: vol has shape (T, Y, X) in your build_volume()
    blurred = gaussian_filter(
        vol.astype(float),
        sigma=(sigma_t, sigma_xy, sigma_xy)  # time, y, x
    )
    lo, hi = blurred.min(), blurred.max()
    if hi <= lo:
        print(f"Empty volume; skipping iso-surface ({lo:.3f}–{hi:.3f})")
        return None

    level = 0.5
    if not (lo < level < hi):
        level = lo + 0.5 * (hi - lo)

    try:
        verts, faces, _, _ = marching_cubes(blurred, level=level)
    except RuntimeError as e:
        print(f"No surface found: {e}; skipping")
        return None

    nx, ny, nz = len(xs) - 1, len(ys) - 1, len(ts) - 1
    vx = xs[0] + verts[:, 2] / nx * (xs[-1] - xs[0])
    vy = ys[0] + verts[:, 1] / ny * (ys[-1] - ys[0])
    vz = ts[0] + verts[:, 0] / nz * (ts[-1] - ts[0])
    return np.stack([vx, vy, vz], axis=1), faces