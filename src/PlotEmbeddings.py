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
grid_res_t = 128 # Density-field time resolution (for density_3d backend)

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

    def add_convex_hull(
        self,
        *,
        backend: str = "density_3d",   # "hull_raster" (legacy) or "density_3d" (new)
        # legacy knobs
        n_time_bins=None,
        time_bin_mode="log",           # kept for CLI friendliness; mapped to strategy
        time_bin_strategy=None,        # optional explicit override
        # common
        verbose=False,
        debug_dir=None,
        # density backend knobs
        grid_res_t_override: int | None = None,
        density_time_weighting: str = "inv_time",  # "none" or "inv_time"
        density_sigma_xy: float | None = None,
        density_sigma_t: float | None = None,
        support_mass_frac: float = 0.90,

    ):
        """
        backend:
        - "hull_raster": per-slice hull -> raster -> volume
        - "density_3d":  3D histogram in (X1,X2,log10(time)) -> smooth -> isosurface
        """

        for population in self.populations:
            color = self.colors_dict.get(population, "#808080")

            if backend == "hull_raster":
                # Map old "mode" to a strategy if caller didn't pass one
                if time_bin_strategy is None:
                    # simplest default mapping
                    time_bin_strategy = "log_quantile" if time_bin_mode == "log" else "linear_quantile"

                vol, xs, ys, ts = build_volume(
                    self.embeddings_df,
                    population,
                    grid_res_xy,
                    n_time_bins=n_time_bins,
                    time_bin_strategy=time_bin_strategy,
                    verbose=verbose,
                    debug_dir=debug_dir,
                )

                surf = surface_from_volume(
                    vol, xs, ys, ts,
                    sigma_xy=gauss_sigma,
                    sigma_t=5.0,
                    keep_largest_component=False,
                    min_component_voxels=50,
                    verbose=verbose,
                    t_mapping="time",
                )

            elif backend == "density_3d":
                U = grid_res_t_override if grid_res_t_override is not None else grid_res_t

                # map CLI / overrides
                sig_xy = float(density_sigma_xy) if density_sigma_xy is not None else gauss_sigma
                sig_u  = float(density_sigma_t)  if density_sigma_t  is not None else 1.0
                mf     = float(support_mass_frac)

                vol, xs, ys, us = build_population_support_volume(
                    self.embeddings_df,
                    population,
                    grid_res_xy=grid_res_xy,
                    grid_res_u=U,
                    time_eps=1.0,
                    mass_frac=mf,            # <<< now controlled by CLI
                    sigma_xy=sig_xy,
                    sigma_u=sig_u,
                    min_points_per_slice=5,
                    verbose=verbose,
                    debug_dir=debug_dir,
                )

                # support-volume outputs a softened mask in [0,1]
                # midpoint iso is appropriate and stable here
                surf = surface_from_volume(
                    vol, xs, ys, us,
                    sigma_xy=0.0,
                    sigma_t=0.0,
                    keep_largest_component=True,
                    min_component_voxels=50,
                    verbose=verbose,
                    t_mapping="log10_time",
                    iso_mode="midpoint",
                )

            else:
                raise ValueError(f"backend must be 'hull_raster' or 'density_3d', got {backend!r}")

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

def _finite(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=float)
    return t[np.isfinite(t)]

def _positive(times: np.ndarray) -> np.ndarray:
    t = _finite(times)
    return t[t > 0]

def make_time_edges(times: np.ndarray, n_bins: int, strategy: str, *, time_min=None, time_max=None) -> np.ndarray:
    """
    Returns edges of length n_bins+1 in *time units*.

    strategy:
      - "log_edges": log-spaced edges from tmin..tmax
      - "linear_edges": linear edges from tmin..tmax
      - "log_quantile": equal-count in log10(time) (recommended default)
      - "linear_quantile": equal-count in linear time
    """
    if n_bins < 2:
        n_bins = 2

    t = _finite(times)
    if t.size == 0:
        return np.array([0.0, 1.0])

    tmin = float(np.min(t)) if time_min is None else float(time_min)
    tmax = float(np.max(t)) if time_max is None else float(time_max)

    if strategy == "log_edges":
        pos = t[t > 0]
        if pos.size == 0:
            raise ValueError("log_edges requested but no positive times available.")
        if tmin <= 0:
            tmin = float(np.min(pos))
        if tmax <= tmin:
            tmax = tmin * 10.0
        edges = np.logspace(np.log10(tmin), np.log10(tmax), n_bins + 1)
        return edges

    if strategy == "linear_edges":
        if tmax <= tmin:
            tmax = tmin + 1e-6
        edges = np.linspace(tmin, tmax, n_bins + 1)
        return edges

    if strategy == "log_quantile":
        pos = t[t > 0]
        if pos.size == 0:
            raise ValueError("log_quantile requested but no positive times available.")
        lt = np.log10(pos)

        qs = np.linspace(0, 1, n_bins + 1)
        ledges = np.quantile(lt, qs)

        # de-duplicate (ties / discrete times)
        ledges = np.unique(ledges)
        if ledges.size < 2:
            ledges = np.array([lt.min(), lt.max() + 1e-6])

        edges = 10 ** ledges

        # optional clamp to provided time_min/max if given
        if time_min is not None:
            edges[0] = max(edges[0], float(time_min))
        if time_max is not None:
            edges[-1] = min(edges[-1], float(time_max))
        return edges

    if strategy == "linear_quantile":
        qs = np.linspace(0, 1, n_bins + 1)
        edges = np.quantile(t, qs)
        edges = np.unique(edges)
        if edges.size < 2:
            edges = np.array([t.min(), t.max() + 1e-6])
        if time_min is not None:
            edges[0] = max(edges[0], float(time_min))
        if time_max is not None:
            edges[-1] = min(edges[-1], float(time_max))
        return edges

    raise ValueError(f"Unknown time bin strategy: {strategy!r}")

def bin_counts(times: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Counts points per bin for bins [edges[i], edges[i+1]).
    """
    t = _finite(times)
    if t.size == 0:
        return np.zeros(max(len(edges) - 1, 0), dtype=int)

    ids = np.searchsorted(edges, t, side="right") - 1
    ids = np.clip(ids, 0, len(edges) - 2)
    return np.bincount(ids, minlength=len(edges) - 1)

def choose_n_time_bins_auto(
    times: np.ndarray,
    *,
    strategy: str = "log_quantile",
    target_points_per_bin: int = 200,
    min_bins: int = 30,
    max_bins: int = 200,
    min_points_per_bin: int = 10,
    max_sparse_frac: float = 0.05,
    time_min: float | None = None,
    time_max: float | None = None,
    verbose: bool = False,
) -> int:
    """
    Picks n_bins based on desired points/bin, with an occupancy-based safety check.
    Reduces n_bins until fewer than max_sparse_frac bins have < min_points_per_bin points.
    """
    t = _finite(times)
    N = int(t.size)
    if N == 0:
        return min_bins

    n = int(round(N / max(target_points_per_bin, 1)))
    n = int(np.clip(n, min_bins, max_bins))

    # Iteratively reduce if too sparse
    while True:
        edges = make_time_edges(t, n, strategy, time_min=time_min, time_max=time_max)
        counts = bin_counts(t, edges)
        sparse_frac = float((counts < min_points_per_bin).mean()) if counts.size else 1.0

        if verbose:
            print(f"[auto bins] n={n} sparse_frac(<{min_points_per_bin})={sparse_frac:.1%} "
                  f"min/med/max={counts.min() if counts.size else 'NA'}/"
                  f"{int(np.median(counts)) if counts.size else 'NA'}/"
                  f"{counts.max() if counts.size else 'NA'}")

        if sparse_frac <= max_sparse_frac or n <= min_bins:
            return int(max(n, 2))

        # shrink bins and try again
        n = int(max(min_bins, round(n * 0.8)))

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
    time_bin_strategy: str = "log_quantile",
    time_min: float | None = None,
    time_max: float | None = None,
    debug_dir: str | None = None,
    verbose: bool = False,
):
    """
    Build boolean volume vol[t, y, x] for one population by rasterizing
    per-time-bin convex-hull polygons on an (X1,X2) grid.

    If n_time_bins is None:
      - uses each unique time as its own slice (exact slicing; can be huge)
    Else:
      - bins time using `time_bin_strategy`
      - uses BIN EDGES as ts (length n_bins+1)
      - extrudes each bin mask into both adjacent edge slices (vol[i], vol[i+1])
    """

    sub = df[df.population == pop].copy()
    if sub.empty:
        vol = np.zeros((0, grid_res, grid_res), dtype=bool)
        return vol, np.array([]), np.array([]), np.array([])

    # ---- XY grid ----
    xs_all, ys_all = sub["X1"], sub["X2"]
    pad_x = 0.05 * (xs_all.max() - xs_all.min() if xs_all.max() > xs_all.min() else 1.0)
    pad_y = 0.05 * (ys_all.max() - ys_all.min() if ys_all.max() > ys_all.min() else 1.0)

    xs = np.linspace(xs_all.min() - pad_x, xs_all.max() + pad_x, grid_res)
    ys = np.linspace(ys_all.min() - pad_y, ys_all.max() + pad_y, grid_res)

    times = sub["time"].to_numpy()

    # optional debug output
    if debug_dir is not None:
        debug_path = Path(debug_dir) / pop
        debug_path.mkdir(parents=True, exist_ok=True)

    # ---- exact-time slicing ----
    if n_time_bins is None:
        unique_times = np.sort(sub.time.unique())
        ts = unique_times
        vol = np.zeros((len(ts), grid_res, grid_res), dtype=bool)

        empty_slices = 0
        points_per_slice = []

        for i, t in enumerate(ts):
            pts = sub[sub.time == t][["X1", "X2"]].values
            points_per_slice.append(len(pts))
            poly = slice_convex_hull(pts)
            if len(poly) < 3:
                empty_slices += 1
                continue
            vol[i] = rasterise_polygon(poly, xs, ys)

            if verbose and (i % max(1, len(ts)//20) == 0):
                occ = float(vol[i].mean())
                print(f"[{pop}] slice {i:03d}/{len(ts)-1} t={t:g} n={len(pts):5d} occ={occ:.6f}")

        if verbose and points_per_slice:
            pts = np.array(points_per_slice, dtype=int)
            print(
                f"[{pop}] exact slices={len(ts)}, empty={empty_slices} (empty_frac={empty_slices/len(ts):.2%}); "
                f"points/slice min={pts.min()}, med={int(np.median(pts))}, max={pts.max()}"
            )

        return vol, xs, ys, ts

    # ---- binned slicing ----
    n_bins = int(max(n_time_bins, 2))

    edges = make_time_edges(times, n_bins, time_bin_strategy, time_min=time_min, time_max=time_max)
    # ts are *edges* (caps + extrusion)
    ts = edges
    vol = np.zeros((len(ts), grid_res, grid_res), dtype=bool)

    # Assign each row to a bin id in 0..n_bins-1
    bin_ids = np.searchsorted(edges, sub["time"].to_numpy(), side="right") - 1
    bin_ids = np.clip(bin_ids, 0, len(edges) - 2)

    if verbose:
        counts = np.bincount(bin_ids, minlength=len(edges) - 1)
        sparse3 = int((counts < 3).sum())
        sparse10 = int((counts < 10).sum())
        print(
            f"[{pop}] strategy={time_bin_strategy} nbins={len(edges)-1} "
            f"counts min/med/max={counts.min()}/{int(np.median(counts))}/{counts.max()} "
            f"<3={sparse3} ({sparse3/len(counts):.1%}) <10={sparse10} ({sparse10/len(counts):.1%})"
        )

    empty_bins = 0
    points_per_bin = []

    for i in range(len(edges) - 1):
        pts = sub.loc[bin_ids == i, ["X1", "X2"]].values
        points_per_bin.append(len(pts))

        poly = slice_convex_hull(pts)
        if len(poly) < 3:
            empty_bins += 1
            continue

        mask2d = rasterise_polygon(poly, xs, ys)

        # Extrude: stamp into adjacent edge slices
        vol[i]   |= mask2d
        vol[i+1] |= mask2d

    if verbose and points_per_bin:
        pts = np.array(points_per_bin, dtype=int)
        print(
            f"[{pop}] binned slices={len(ts)} (edges), empty_bins={empty_bins} (empty_frac={empty_bins/len(pts):.2%}); "
            f"points/bin min={pts.min()}, med={int(np.median(pts))}, max={pts.max()}"
        )

    return vol, xs, ys, ts

def build_field_density_3d(
    df: pd.DataFrame,
    pop: str,
    grid_res_xy: int,
    grid_res_t: int,
    *,
    time_eps: float = 1.0,
    time_transform: str = "log10",     # only log10 supported for now
    time_weighting: str = "inv_time",  # "none" or "inv_time"
    xy_pad_frac: float = 0.05,
    t_min: float | None = None,
    t_max: float | None = None,
    verbose: bool = False,
):
    """
    Build a 3D density field for one population on a regular grid in:
      X1, X2, u = log10(time)

    Returns:
      field: (T, Y, X) float array
      xs:    (X,) bin centers
      ys:    (Y,) bin centers
      us:    (T,) bin centers in u-space (log10(time))

    Key idea:
      - If you bin in u=log10(time), and your data are ~uniform in linear time,
        raw counts will heavily favor large times. Setting time_weighting="inv_time"
        applies weights=1/max(time,eps) to balance the field across log-time.
    """

    sub = df[df.population == pop].copy()
    if sub.empty:
        field = np.zeros((0, grid_res_xy, grid_res_xy), dtype=float)
        return field, np.array([]), np.array([]), np.array([])

    # ---- XY bounds (with padding) ----
    x = sub["X1"].to_numpy(dtype=float)
    y = sub["X2"].to_numpy(dtype=float)

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

    x_pad = xy_pad_frac * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = xy_pad_frac * (y_max - y_min if y_max > y_min else 1.0)

    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    # ---- Time -> u = log10(time) ----
    t = sub["time"].to_numpy(dtype=float)
    t_pos = np.maximum(t, time_eps)

    if time_transform != "log10":
        raise ValueError(f"time_transform must be 'log10' for now, got {time_transform!r}")

    u = np.log10(t_pos)

    # optional overrides for time range (in real time)
    if t_min is not None:
        u_min = np.log10(max(float(t_min), time_eps))
    else:
        u_min = float(np.nanmin(u))

    if t_max is not None:
        u_max = np.log10(max(float(t_max), time_eps))
    else:
        u_max = float(np.nanmax(u))

    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min:
        u_min = float(np.nanmin(u))
        u_max = float(np.nanmax(u))
        if u_max <= u_min:
            u_max = u_min + 1e-6

    # ---- Bin edges ----
    x_edges = np.linspace(x_min, x_max, grid_res_xy + 1)
    y_edges = np.linspace(y_min, y_max, grid_res_xy + 1)
    u_edges = np.linspace(u_min, u_max, grid_res_t + 1)

    # ---- weights (optional) ----
    weights = None
    if time_weighting == "none":
        weights = None
    elif time_weighting == "inv_time":
        # balance density across log-time
        weights = 1.0 / t_pos
    else:
        raise ValueError(f"time_weighting must be 'none' or 'inv_time', got {time_weighting!r}")

    # ---- 3D histogram ----
    # Pass (u,y,x) to get output as (T,Y,X) = (u_bins, y_bins, x_bins)
    field, _ = np.histogramdd(
        sample=np.stack([u, y, x], axis=1),
        bins=(u_edges, y_edges, x_edges),
        weights=weights,
    )
    field = field.astype(float)

    xs = 0.5 * (x_edges[:-1] + x_edges[1:])
    ys = 0.5 * (y_edges[:-1] + y_edges[1:])
    us = 0.5 * (u_edges[:-1] + u_edges[1:])

    if verbose:
        total = float(field.sum())
        nz = int((field > 0).sum())
        print(
            f"[{pop}] density_3d: points={len(sub)} total_mass={total:.3g} nonzero_voxels={nz} "
            f"u_range=[{u_min:.3f},{u_max:.3f}] grid=(T={grid_res_t},Y={grid_res_xy},X={grid_res_xy}) "
            f"weighting={time_weighting}"
        )

    return field, xs, ys, us


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

def _resample_field_uniform_logtime(field: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Resample field[T,Y,X] defined at monotonically increasing times `ts` onto
    a new grid uniform in u=log10(ts).

    Returns:
      field_u: [T,Y,X] resampled at uniform u
      u_grid:  [T] uniform log10 times
    """
    ts = np.asarray(ts, dtype=float)
    if ts.ndim != 1 or ts.size < 2:
        return field, np.log10(np.maximum(ts, 1e-12))

    # log-safe (ts should already be >0 for log axis usage)
    ts_pos = np.maximum(ts, 1e-12)
    u = np.log10(ts_pos)

    # enforce strictly increasing u (very small ties can happen)
    # if ties exist, tiny jitter to keep interp monotone
    du = np.diff(u)
    if np.any(du <= 0):
        u = u.copy()
        for i in range(1, len(u)):
            if u[i] <= u[i-1]:
                u[i] = u[i-1] + 1e-9

    T, Y, X = field.shape
    u_grid = np.linspace(u[0], u[-1], T)

    # interpolate along axis=0 for each (y,x)
    field_u = np.empty_like(field, dtype=float)

    # vectorized per-voxel interpolation is not trivial; do a loop over Y
    # (fast enough at typical grid sizes like 64–256)
    for yy in range(Y):
        # field[:,yy,:] is [T,X]; interpolate each column in X
        # We'll do it column-wise with np.interp
        for xx in range(X):
            field_u[:, yy, xx] = np.interp(u_grid, u, field[:, yy, xx])

    return field_u, u_grid

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
    t_mapping: str = "time",          # "time" or "log10_time"
    iso_mode: str = "mass_frac",      # "mass_frac" (recommended) or "midpoint"
    mass_frac: float = 0.10,          # keep top 10% of total mass when iso_mode="mass_frac"
):
    """
    vol has shape (T, Y, X).
    We blur, choose an isosurface level, optionally keep only the largest
    connected component in voxel space, then run marching_cubes.

    t_mapping:
      - "time": ts is real time (e.g. generations ago)
      - "log10_time": ts is u=log10(time) and we exponentiate back to time for plotting

    iso_mode:
      - "mass_frac": choose level so that voxels >= level contain `mass_frac` of total mass.
                     Robust to reweighting (counts vs 1/time weights).
      - "midpoint":  choose level halfway between min/max (legacy; often fragile).
    """

    if vol.size == 0 or len(ts) < 2:
        return None

    # --- Smooth / blur the field ---
    field = gaussian_filter(
        vol.astype(float),
        sigma=(sigma_t, sigma_xy, sigma_xy),
        mode="nearest",
    )

    lo, hi = float(field.min()), float(field.max())
    if hi <= lo:
        if verbose:
            print(f"Empty volume; skipping iso-surface ({lo:.6g}–{hi:.6g})")
        return None

    # ---------------------------------------------------------------------
    # Isosurface level selection  <<< THIS IS THE BLOCK YOU'RE LOOKING FOR
    # ---------------------------------------------------------------------
    if iso_mode == "mass_frac":
        flat = field.ravel()
        total_mass = float(flat.sum())
        if total_mass <= 0:
            if verbose:
                print("Field has zero mass; skipping")
            return None

        # Find cutoff level such that the region >= level contains `mass_frac` of total mass
        order = np.argsort(flat)[::-1]     # descending
        vals = flat[order]
        csum = np.cumsum(vals)
        cutoff_mass = mass_frac * total_mass

        idx = int(np.searchsorted(csum, cutoff_mass, side="left"))
        idx = int(np.clip(idx, 0, len(vals) - 1))
        level = float(vals[idx])

        # Keep strictly inside range for marching cubes stability
        level = min(max(level, lo + 1e-12), hi - 1e-12)

        if verbose:
            print(
                f"Mass-frac iso level -> {level:.6g} "
                f"(mass_frac={mass_frac:.3f}, range {lo:.6g}–{hi:.6g})"
            )

    elif iso_mode == "midpoint":
        # Legacy heuristic (often breaks when you rescale field via weighting)
        level = lo + 0.5 * (hi - lo)
        if verbose:
            print(f"Midpoint iso level -> {level:.6g} (range {lo:.6g}–{hi:.6g})")

    else:
        raise ValueError(f"iso_mode must be 'mass_frac' or 'midpoint', got {iso_mode!r}")

    # --- Connected component filtering ---
    if keep_largest_component:
        mask = field >= level
        if not mask.any():
            if verbose:
                print("Threshold mask empty; skipping")
            return None

        cc, ncc = label(mask)
        if ncc > 1:
            counts = np.bincount(cc.ravel())
            counts[0] = 0
            keep_id = int(np.argmax(counts))
            keep_size = int(counts[keep_id])

            if keep_size < min_component_voxels:
                if verbose:
                    print(f"All components tiny (largest={keep_size}); skipping")
                return None

            keep_mask = (cc == keep_id)

            if verbose:
                total_on = int(mask.sum())
                frac = keep_size / max(total_on, 1)
                print(f"CC filter: {ncc} components; keeping {keep_id} size={keep_size} ({frac:.2%})")

            # zero-out everything except the kept component
            field = field.copy()
            field[~keep_mask] = lo

    # --- Marching cubes ---
    try:
        verts, faces, _, _ = marching_cubes(field, level=level)
    except RuntimeError as e:
        if verbose:
            print(f"No surface found: {e}; skipping")
        return None

    # --- Map voxel coords -> real coords ---
    # NOTE: xs/ys are centers in your density backend; treat them as a continuous span.
    # We map index 0..(X-1) to xs[0]..xs[-1], similarly for ys and ts.
    nx = max(len(xs) - 1, 1)
    ny = max(len(ys) - 1, 1)

    vx = xs[0] + (verts[:, 2] / nx) * (xs[-1] - xs[0])
    vy = ys[0] + (verts[:, 1] / ny) * (ys[-1] - ys[0])

    # Map voxel t -> ts (which might be time or log10(time))
    t_idx = np.arange(len(ts), dtype=float)
    t_raw = np.interp(verts[:, 0], t_idx, ts.astype(float))

    if t_mapping == "time":
        vz = t_raw
    elif t_mapping == "log10_time":
        vz = 10 ** t_raw
    else:
        raise ValueError(f"t_mapping must be 'time' or 'log10_time', got {t_mapping!r}")

    return np.stack([vx, vy, vz], axis=1), faces

def _mass_fraction_threshold(density: np.ndarray, mass_frac: float) -> float:
    """
    Given a nonnegative density field, return threshold tau such that
    voxels with density >= tau contain `mass_frac` of total mass.

    mass_frac in (0,1]. Larger -> more inclusive region (bigger mask).
    """
    mass_frac = float(mass_frac)
    if not (0 < mass_frac <= 1.0):
        raise ValueError(f"mass_frac must be in (0,1], got {mass_frac}")

    d = np.asarray(density, dtype=float).ravel()
    total = float(d.sum())
    if total <= 0:
        return np.inf  # yields empty mask

    order = np.argsort(d)[::-1]  # descending
    vals = d[order]
    csum = np.cumsum(vals)

    cutoff = mass_frac * total
    idx = int(np.searchsorted(csum, cutoff, side="left"))
    idx = int(np.clip(idx, 0, len(vals) - 1))
    return float(vals[idx])


def build_population_support_volume(
    df: pd.DataFrame,
    pop: str,
    *,
    grid_res_xy: int,
    grid_res_u: int,
    time_eps: float = 1.0,
    mass_frac: float = 0.90,
    sigma_xy: float = 1.5,          # in grid cells
    sigma_u: float = 1.0,           # in u-bins
    xy_pad_frac: float = 0.05,
    min_points_per_slice: int = 5,  # slices with fewer points -> empty
    t_min: float | None = None,     # optional clamp in real time
    t_max: float | None = None,
    verbose: bool = False,
    debug_dir: str | None = None,   # if set: writes a few slice dumps
):
    """
    Two-phase tube builder:
      Phase A: for each u=log10(time) slice, compute a smoothed 2D density in (X1,X2),
               then threshold it to a mass-fraction mask (e.g. 90% region).
      Phase B: smooth masks along u (time) and return a float volume suitable for marching cubes.

    Returns:
      vol: float array (U, Y, X) in [0,1] after time-smoothing
      xs:  (X,) bin centers
      ys:  (Y,) bin centers
      us:  (U,) bin centers in log10(time) space
    """
    sub = df[df.population == pop].copy()
    if sub.empty:
        vol = np.zeros((0, grid_res_xy, grid_res_xy), dtype=float)
        return vol, np.array([]), np.array([]), np.array([])

    # --- XY bounds (with padding) ---
    x = sub["X1"].to_numpy(dtype=float)
    y = sub["X2"].to_numpy(dtype=float)
    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

    x_pad = xy_pad_frac * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = xy_pad_frac * (y_max - y_min if y_max > y_min else 1.0)
    x_min -= x_pad; x_max += x_pad
    y_min -= y_pad; y_max += y_pad

    # --- Time -> u=log10(time) ---
    t = sub["time"].to_numpy(dtype=float)
    t_pos = np.maximum(t, time_eps)
    u = np.log10(t_pos)

    # clamp time range if requested (in real time, mapped to u)
    if t_min is not None:
        u_min = np.log10(max(float(t_min), time_eps))
    else:
        u_min = float(np.nanmin(u))
    if t_max is not None:
        u_max = np.log10(max(float(t_max), time_eps))
    else:
        u_max = float(np.nanmax(u))

    if not np.isfinite(u_min) or not np.isfinite(u_max) or u_max <= u_min:
        u_min = float(np.nanmin(u))
        u_max = float(np.nanmax(u))
        if u_max <= u_min:
            u_max = u_min + 1e-6

    # --- bin edges ---
    x_edges = np.linspace(x_min, x_max, grid_res_xy + 1)
    y_edges = np.linspace(y_min, y_max, grid_res_xy + 1)
    u_edges = np.linspace(u_min, u_max, grid_res_u + 1)

    xs = 0.5 * (x_edges[:-1] + x_edges[1:])
    ys = 0.5 * (y_edges[:-1] + y_edges[1:])
    us = 0.5 * (u_edges[:-1] + u_edges[1:])

    # assign each point to a u-slice
    u_bin = np.searchsorted(u_edges, u, side="right") - 1
    u_bin = np.clip(u_bin, 0, grid_res_u - 1)

    counts = np.bincount(u_bin, minlength=grid_res_u)
    empty = int((counts < min_points_per_slice).sum())

    if verbose:
        print(
            f"[{pop}] support_volume: grid=(U={grid_res_u},Y={grid_res_xy},X={grid_res_xy}) "
            f"u_range=[{u_min:.3f},{u_max:.3f}] "
            f"sigma_xy={sigma_xy} sigma_u={sigma_u} mass_frac={mass_frac} "
            f"slices(<{min_points_per_slice} pts)={empty}/{grid_res_u}"
        )

    # --- Phase A: per-slice mask ---
    masks = np.zeros((grid_res_u, grid_res_xy, grid_res_xy), dtype=bool)

    # optional debug: dump a few slices
    dbg_path = None
    if debug_dir is not None:
        dbg_path = Path(debug_dir) / pop
        dbg_path.mkdir(parents=True, exist_ok=True)

    # get indices per slice efficiently
    # (looping over slices is fine at U<=256)
    for ui in range(grid_res_u):
        if counts[ui] < min_points_per_slice:
            continue

        sel = (u_bin == ui)
        xsamp = x[sel]
        ysamp = y[sel]

        # 2D histogram for this slice
        H, _, _ = np.histogram2d(
            ysamp, xsamp,              # note ordering: rows=y, cols=x
            bins=(y_edges, x_edges),
        )
        H = H.astype(float)

        # smooth within-slice in XY to suppress jitter and fill gaps
        if sigma_xy and sigma_xy > 0:
            Hs = gaussian_filter(H, sigma=(sigma_xy, sigma_xy), mode="nearest")
        else:
            Hs = H

        tau = _mass_fraction_threshold(Hs, mass_frac)
        if not np.isfinite(tau):
            continue

        masks[ui] = (Hs >= tau)

        if dbg_path is not None:
            # Dump every slice so we can inspect continuity and gaps
            np.save(dbg_path / f"slice_{ui:03d}_Hs.npy", Hs)
            np.save(dbg_path / f"slice_{ui:03d}_mask.npy", masks[ui])

            # Also save a tiny metadata file once
            meta_path = dbg_path / "_meta_support_volume.txt"
            if not meta_path.exists():
                meta_path.write_text(
                    f"pop={pop}\n"
                    f"grid_res_xy={grid_res_xy}\n"
                    f"grid_res_u={grid_res_u}\n"
                    f"u_min={u_min}\n"
                    f"u_max={u_max}\n"
                    f"mass_frac={mass_frac}\n"
                    f"sigma_xy={sigma_xy}\n"
                    f"sigma_u={sigma_u}\n"
                    f"min_points_per_slice={min_points_per_slice}\n"
                )

    # --- Phase B: smooth masks along time (u) only ---
    vol = masks.astype(float)
    if sigma_u and sigma_u > 0:
        vol = gaussian_filter(vol, sigma=(sigma_u, 0.0, 0.0), mode="nearest")

    # keep in [0,1] range
    vol = np.clip(vol, 0.0, 1.0)

    if dbg_path is not None:
        np.save(dbg_path / f"_vol_smoothed.npy", vol)
        np.save(dbg_path / f"_masks_raw.npy", masks.astype(np.uint8))
        np.save(dbg_path / f"_counts.npy", counts)
        np.save(dbg_path / f"_us.npy", us)

    return vol, xs, ys, us