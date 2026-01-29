#!/usr/bin/env python3
import numpy as np
import pandas as pd
from pathlib import Path

import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

from scipy.spatial import ConvexHull
from scipy.ndimage import gaussian_filter
from scipy.ndimage import label
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

    def add_convex_hull(self, *, n_time_bins=None, time_bin_mode="log", verbose=False, debug_dir=None):
        for population in self.populations:
            color = self.colors_dict.get(population, "#808080")

            vol, xs, ys, ts = build_volume(
                self.embeddings_df,
                population,
                grid_res_xy,
                n_time_bins=n_time_bins,        # NEW
                time_bin_mode=time_bin_mode,    # NEW
                verbose=verbose,                # if you want
                debug_dir=debug_dir,            # if you want
            )

            surf = surface_from_volume(
                vol, xs, ys, ts,
                sigma_xy=gauss_sigma,
                sigma_t=0.0,  # we’ll make this a knob next
                keep_largest_component=True,
                min_component_voxels=50,
                verbose=verbose,
            )
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
    n_time_bins: int | None = None,
    time_bin_mode: str = "log",   # "log" or "linear"
    time_min: float | None = None,
    time_max: float | None = None,
    debug_dir: str | None = None,
    verbose: bool = False,
):
    """
    Build a boolean volume vol[t, y, x] for one population by rasterizing
    per-time (or per-time-bin) convex-hull polygons on an (X1,X2) grid.

    If n_time_bins is None: uses each unique time as its own slice (original behavior).
    If n_time_bins is set: bins time (log or linear) and uses BIN CENTERS as ts.
    """

    sub = df[df.population == pop].copy()
    if sub.empty:
        vol = np.zeros((0, grid_res, grid_res), dtype=bool)
        return vol, np.array([]), np.array([]), np.array([])

    # ---- XY grid (same as before) ----
    xs_all, ys_all = sub["X1"], sub["X2"]
    pad_x = 0.05 * (xs_all.max() - xs_all.min() if xs_all.max() > xs_all.min() else 1.0)
    pad_y = 0.05 * (ys_all.max() - ys_all.min() if ys_all.max() > ys_all.min() else 1.0)

    xs = np.linspace(xs_all.min() - pad_x, xs_all.max() + pad_x, grid_res)
    ys = np.linspace(ys_all.min() - pad_y, ys_all.max() + pad_y, grid_res)

    # ---- time binning ----
    times = sub["time"].to_numpy()
    # marching cubes + log axes dislike nonpositive time; clip/guard here
    tmin_data = float(np.nanmin(times))
    tmax_data = float(np.nanmax(times))

    tmin = float(time_min) if time_min is not None else tmin_data
    tmax = float(time_max) if time_max is not None else tmax_data

    # ensure strictly positive for log binning
    if time_bin_mode == "log":
        if tmin <= 0:
            # shift to smallest positive observed time if available, else 1.0
            pos = times[times > 0]
            tmin = float(np.min(pos)) if pos.size else 1.0
        if tmax <= tmin:
            tmax = tmin * 10.0

    if n_time_bins is None:
        # Original behavior: one slice per unique time
        unique_times = np.sort(sub.time.unique())
        ts = unique_times
        # map each unique time to itself
        bin_ids = None
        edges = None
    else:
        # Binned behavior
        if n_time_bins < 2:
            n_time_bins = 2

        if time_bin_mode == "log":
            edges = np.logspace(np.log10(tmin), np.log10(tmax), n_time_bins + 1)
        elif time_bin_mode == "linear":
            edges = np.linspace(tmin, tmax, n_time_bins + 1)
        else:
            raise ValueError(f"time_bin_mode must be 'log' or 'linear', got {time_bin_mode!r}")

        # Bin centers:
        # - if bins are in log-time, the "center" on a log axis should be the geometric mean
        # - if bins are linear, arithmetic midpoint is fine
        if time_bin_mode == "log":
            ts = np.sqrt(edges[:-1] * edges[1:])  # geometric mean
        else:
            ts = 0.5 * (edges[:-1] + edges[1:])   # arithmetic mean

        # assign each row to a bin
        # bins are [edges[i], edges[i+1]) except last includes right edge
        bin_ids = np.searchsorted(edges, sub["time"].to_numpy(), side="right") - 1
        bin_ids = np.clip(bin_ids, 0, n_time_bins - 1)

    # ---- build volume ----
    vol = np.zeros((len(ts), grid_res, grid_res), dtype=bool)

    # optional debug output
    debug_path = None
    if debug_dir is not None:
        debug_path = Path(debug_dir) / pop
        debug_path.mkdir(parents=True, exist_ok=True)

    empty_slices = 0
    points_per_slice = []

    if n_time_bins is None:
        # loop over exact times
        for i, t in enumerate(ts):
            pts = sub[sub.time == t][["X1", "X2"]].values
            points_per_slice.append(len(pts))
            poly = slice_convex_hull(pts)
            if len(poly) < 3:
                empty_slices += 1
                continue
            vol[i] = rasterise_polygon(poly, xs, ys)

            if verbose:
                occ = float(vol[i].mean())
                print(f"[{pop}] slice {i:03d}/{len(ts)-1} t={t:g} n={len(pts):5d} verts={len(poly):4d} occ={occ:.6f}")

    else:
        # loop over bins
        for i in range(len(ts)):
            pts = sub.loc[bin_ids == i, ["X1", "X2"]].values
            points_per_slice.append(len(pts))
            poly = slice_convex_hull(pts)
            if len(poly) < 3:
                empty_slices += 1
                continue
            vol[i] = rasterise_polygon(poly, xs, ys)

            if verbose:
                occ = float(vol[i].mean())
                # print bin edges too for debugging
                lo = edges[i]
                hi = edges[i + 1]
                print(
                    f"[{pop}] bin {i:03d}/{len(ts)-1} "
                    f"t~{ts[i]:g} [{lo:g},{hi:g}) n={len(pts):5d} verts={len(poly):4d} occ={occ:.6f}"
                )

    if verbose:
        pts = np.array(points_per_slice) if points_per_slice else np.array([])
        if pts.size:
            print(
                f"[{pop}] slices={len(ts)}, empty={empty_slices} (empty_frac={empty_slices/len(ts):.2%}) ; "
                f"points/bin min={pts.min()}, med={int(np.median(pts))}, max={pts.max()}"
            )

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



def surface_from_volume(
    vol: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    ts: np.ndarray,
    sigma_xy: float,
    sigma_t: float = 0.0,
    *,
    keep_largest_component: bool = True,
    min_component_voxels: int = 50,
    verbose: bool = False,
):
    """
    vol has shape (T, Y, X) as built by build_volume().
    We blur, choose an isosurface level, optionally keep only the largest
    connected component in voxel space, then run marching_cubes.
    """

    # Smooth (time, y, x)
    blurred = gaussian_filter(
        vol.astype(float),
        sigma=(sigma_t, sigma_xy, sigma_xy)
    )

    lo, hi = float(blurred.min()), float(blurred.max())
    if hi <= lo:
        if verbose:
            print(f"Empty volume; skipping iso-surface ({lo:.3f}–{hi:.3f})")
        return None

    # Isosurface level
    level = 0.5
    if not (lo < level < hi):
        level = lo + 0.5 * (hi - lo)
        if verbose:
            print(f"Adjusted iso level -> {level:.4f} (range {lo:.4f}–{hi:.4f})")

    field = blurred

    # --- Connected component filtering in voxel space ---
    # Threshold at the same level we’ll use for marching cubes.
    if keep_largest_component:
        mask = field >= level
        if mask.any():
            cc, ncc = label(mask)  # 6-connectivity in 3D by default
            if ncc > 1:
                counts = np.bincount(cc.ravel())
                counts[0] = 0  # background

                # pick the largest component
                keep_id = int(np.argmax(counts))
                keep_size = int(counts[keep_id])

                if keep_size < min_component_voxels:
                    # too tiny to be meaningful
                    if verbose:
                        print(f"All components tiny (largest={keep_size}); skipping")
                    return None

                keep_mask = (cc == keep_id)

                if verbose:
                    total_on = int(mask.sum())
                    frac = keep_size / max(total_on, 1)
                    print(f"CC filter: {ncc} components; keeping {keep_id} size={keep_size} "
                          f"({frac:.2%} of thresholded voxels)")

                # Zero out everything outside the main component so marching cubes ignores islands
                field = field.copy()
                field[~keep_mask] = lo
        else:
            if verbose:
                print("Threshold mask empty; skipping")
            return None

    # --- Marching cubes ---
    try:
        verts, faces, _, _ = marching_cubes(field, level=level)
    except RuntimeError as e:
        if verbose:
            print(f"No surface found: {e}; skipping")
        return None

    # Map voxel coords -> real coords
    nx, ny = len(xs) - 1, len(ys) - 1

    vx = xs[0] + verts[:, 2] / nx * (xs[-1] - xs[0])
    vy = ys[0] + verts[:, 1] / ny * (ys[-1] - ys[0])

    # IMPORTANT: marching_cubes returns verts[:,0] in *index space* (0..T-1),
    # but ts can be non-uniform (log bins). Use interpolation instead of assuming
    # linear spacing between ts[0] and ts[-1].
    t_idx = np.arange(len(ts), dtype=float)
    vz = np.interp(verts[:, 0], t_idx, ts.astype(float))

    return np.stack([vx, vy, vz], axis=1), faces