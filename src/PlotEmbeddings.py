#!/usr/bin/env python3
"""
PlotEmbeddings.py

Plotting-only orchestration class for embedding trajectories and population-level
support geometry.

Design goals (current cleanup pass)
----------------------------------
- PlotEmbeddings owns: figure, embeddings df, population labels/colors, and core layers
  (raw/sample trajectories, medians, legend).
- No global mutable state / tunables. All knobs are explicit method args.
- Support-tube geometry (mesh + rings) lives in src/support_tube.py and is imported
  lazily inside methods to avoid heavy deps unless needed.

NOTE
----
You said you moved old helper functions (volume builders, marching cubes surface extraction,
etc.) into src/old_support_tube.py. This file no longer depends on them.

Next steps (after this cleanup)
-------------------------------
- Implement src/support_tube.py with:
    - Phase A: per-u slice mass-fraction mask
    - Phase B: u-smoothing + optional closing + mesh + ring extraction
- Wire add_support_tube() / add_support_rings() to those functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

# ----------------------- tunables ------------------------------------------
sample_line_width = 2
sample_line_alpha = 0.5

median_line_width = 5
median_line_alpha = 1.0

# ------------------------- small helpers ----------------------------------


def _require_columns(df: pd.DataFrame, cols: Sequence[str], *, df_name: str = "embeddings_df") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{df_name} is missing required columns: {missing}")


def _safe_positive_min(x: np.ndarray, *, fallback: float = 1.0) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    x = x[x > 0]
    if x.size == 0:
        return float(fallback)
    return float(np.min(x))


@dataclass(frozen=True)
class PlotRanges:
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]
    z_ticks: np.ndarray
    z_ticktext: List[str]


# ------------------------- main class -------------------------------------


class PlotEmbeddings:
    """
    PlotEmbeddings is a plotting orchestrator.

    It expects an embeddings_df with at least columns:
      - sample_id
      - X1, X2
      - time

    Labels/colors are optional and are merged in, producing:
      - population column
      - color column (hex strings)
    """

    def __init__(
        self,
        *,
        embeddings_df: pd.DataFrame,
        labels_df: Optional[pd.DataFrame] = None,
        colors_df: Optional[pd.DataFrame] = None,
        generation_time: Optional[float] = None,
        xcol: str = "X1",
        ycol: str = "X2",
        tcol: str = "time",
        sample_id_col: str = "sample_id",
        popcol: str = "population",
        default_population: str = "unlabelled_pop",
        default_color: str = "#808080",
        default_palette: Optional[Sequence[str]] = None,
    ):
        self.xcol = xcol
        self.ycol = ycol
        self.tcol = tcol
        self.sample_id_col = sample_id_col
        self.popcol = popcol
        self.default_population = default_population
        self.default_color = default_color

        _require_columns(embeddings_df, [sample_id_col, xcol, ycol, tcol], df_name="embeddings_df")

        df = embeddings_df.copy()

        # ---- attach population labels (optional) ----
        if labels_df is None:
            df[self.popcol] = default_population
            colors_df = None  # trigger default colors if no labels provided
        else:
            _require_columns(labels_df, [sample_id_col, popcol], df_name="labels_df")
            df = df.merge(labels_df[[sample_id_col, popcol]], on=sample_id_col, how="left")
            df[self.popcol] = df[self.popcol].fillna(default_population)

        # ---- convert time to years if generation_time provided ----
        self.generation_time = generation_time
        if generation_time is not None:
            df[self.tcol] = df[self.tcol].astype(float) * float(generation_time)

        # ---- attach colors (optional) ----
        if default_palette is None:
            default_palette = px.colors.qualitative.Safe

        if colors_df is None:
            pops = df[self.popcol].astype(str).unique().tolist()
            colors_dict = {p: default_palette[i % len(default_palette)] for i, p in enumerate(pops)}
            df["color"] = df[self.popcol].astype(str).map(colors_dict)
        else:
            _require_columns(colors_df, [popcol, "color"], df_name="colors_df")
            cdf = colors_df[[popcol, "color"]].copy()
            cdf[popcol] = cdf[popcol].astype(str)
            cdf["color"] = cdf["color"].astype(str)

            df = df.merge(cdf, on=popcol, how="left")
            df["color"] = df["color"].fillna(default_color)

            unique_colors = df[[self.popcol, "color"]].drop_duplicates()
            colors_dict = {row[self.popcol]: row["color"] for _, row in unique_colors.iterrows()}

        self.embeddings_df = df
        self.colors_dict: Dict[str, str] = dict(colors_dict)

        self.populations = df[self.popcol].astype(str).unique()
        self.num_populations = df[self.popcol].nunique()

        self.times = df[self.tcol].unique()
        self.fig: Optional[go.Figure] = None

        # stored ranges for downstream geometry layers if useful
        self._plot_ranges: Optional[PlotRanges] = None

    # ------------------------- plot setup ---------------------------------

    def init_plot(
        self,
        *,
        time_scale: str = "log",
        axis_background: str = "whitesmoke",
        pad_frac: float = 0.05,
        aspectratio: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """
        Initialize an empty Plotly Figure with consistent axis ranges.

        time_scale: "log" or "linear" (Plotly z-axis type)
        """
        if time_scale not in {"log", "linear"}:
            raise ValueError("time_scale must be 'log' or 'linear'")

        df = self.embeddings_df
        self.fig = go.Figure()

        # Axis labels
        x_label = self.xcol
        y_label = self.ycol
        z_label = "Time (generations ago)" if self.generation_time is None else "Time (years ago)"

        axis_style = dict(
            showline=True,
            showgrid=True,
            showbackground=True,
            linecolor="black",
            linewidth=1,
            ticks="outside",
            tickcolor="black",
            gridcolor="white",
            backgroundcolor=axis_background,
            tickmode="array",
        )

        # Equal scaling for X and Y
        x_min = float(df[self.xcol].min())
        x_max = float(df[self.xcol].max())
        y_min = float(df[self.ycol].min())
        y_max = float(df[self.ycol].max())

        common_min = float(min(x_min, y_min))
        common_max = float(max(x_max, y_max))
        pad = float(pad_frac) * (common_max - common_min if common_max > common_min else 1.0)
        common_min -= pad
        common_max += pad

        x_range = (common_min, common_max)
        y_range = (common_min, common_max)

        # Z ticks (decades) for log scale; for linear, still use nice decade ticks if possible
        t = df[self.tcol].to_numpy(dtype=float)
        min_time = _safe_positive_min(t, fallback=1.0)
        max_time = float(np.nanmax(t[np.isfinite(t)])) if np.isfinite(t).any() else 10.0
        max_time = max(max_time, min_time * 1.01)

        z_min_decade = int(np.floor(np.log10(min_time)))
        z_max_decade = int(np.ceil(np.log10(max_time)))
        z_ticks = (10.0 ** np.arange(z_min_decade, z_max_decade + 1)).astype(float)

        self._plot_ranges = PlotRanges(
            x_range=x_range,
            y_range=y_range,
            z_ticks=z_ticks,
            z_ticktext=[str(int(z)) for z in z_ticks],
        )

        self.fig.update_layout(
            scene=dict(
                xaxis=dict(title=x_label, range=list(x_range), zeroline=True, **axis_style),
                yaxis=dict(title=y_label, range=list(y_range), zeroline=True, **axis_style),
                zaxis=dict(
                    title=z_label,
                    tickvals=z_ticks,
                    ticktext=[str(int(z)) for z in z_ticks],
                    type=time_scale,
                    **axis_style,
                ),
                aspectmode="manual",
                aspectratio=dict(x=float(aspectratio[0]), y=float(aspectratio[1]), z=float(aspectratio[2])),
            )
        )

    def save_static(
        self,
        filename: str,
        *,
        format: str = "png",
        width: int = 600,
        height: int = 600,
        scale: int = 2,
        eye: Dict[str, float] = {"x": 1.8, "y": 1.8, "z": 0.3},
        up: Dict[str, float] = {"x": 0.0, "y": 0.0, "z": 1.0},
    ) -> None:
        if self.fig is None:
            raise RuntimeError("Call init_plot() before saving.")
        self.fig.update_layout(scene=dict(camera=dict(eye=eye, up=up)))
        pio.write_image(fig=self.fig, file=f"{filename}.{format}", format=format, width=width, height=height, scale=scale)

    # ------------------------- core layers --------------------------------

    def add_legend(self, *, marker_size: int = 10) -> None:
        if self.fig is None:
            raise RuntimeError("Call init_plot() before adding layers.")
        for pop in self.populations:
            color = self.colors_dict.get(pop, self.default_color)
            self.fig.add_trace(
                go.Scatter3d(
                    x=[None],
                    y=[None],
                    z=[None],
                    mode="markers",
                    marker=dict(size=marker_size, color=color),
                    name=str(pop),
                    legendgroup=str(pop),
                    showlegend=True,
                )
            )

    def add_sample_trajectories(
        self,
        *,
        line_width: float = 2.0,
        alpha: float = 0.5,
        max_samples: Optional[int] = None,
        sample_stride: int = 1,
    ) -> None:
        """
        Add one polyline per sample_id.
        Can be heavy; max_samples/sample_stride can thin.
        """
        if self.fig is None:
            raise RuntimeError("Call init_plot() before adding layers.")

        df = self.embeddings_df

        samples = df[self.sample_id_col].unique()
        samples = samples[:: max(1, int(sample_stride))]

        if max_samples is not None:
            samples = samples[: int(max_samples)]

        for sample in samples:
            sample_data = df[df[self.sample_id_col] == sample]
            if sample_data.empty:
                continue

            population = str(sample_data[self.popcol].values[0])
            color = self.colors_dict.get(population, self.default_color)

            self.fig.add_trace(
                go.Scatter3d(
                    x=sample_data[self.xcol],
                    y=sample_data[self.ycol],
                    z=sample_data[self.tcol],
                    mode="lines",
                    line=dict(color=color, width=float(line_width)),
                    opacity=float(alpha),
                    name=f"{population} - {sample}",
                    hovertemplate=f"{population}: {sample}<br>Time: %{{z:.2f}}<extra></extra>",
                    showlegend=False,
                    legendgroup=population,
                )
            )

    def add_population_medians(self, *, line_width: float = 5.0, alpha: float = 1.0) -> None:
        if self.fig is None:
            raise RuntimeError("Call init_plot() before adding layers.")

        df = self.embeddings_df

        median_df = (
            df.groupby([self.popcol, self.tcol], as_index=False)
            .agg(
                **{
                    self.xcol: (self.xcol, "median"),
                    self.ycol: (self.ycol, "median"),
                    "color": ("color", "first"),
                }
            )
            .sort_values([self.popcol, self.tcol])
        )

        for population in self.populations:
            pop = str(population)
            pop_data = median_df[median_df[self.popcol].astype(str) == pop]
            if pop_data.empty:
                continue

            color = self.colors_dict.get(pop, self.default_color)
            self.fig.add_trace(
                go.Scatter3d(
                    x=pop_data[self.xcol],
                    y=pop_data[self.ycol],
                    z=pop_data[self.tcol],
                    mode="lines",
                    line=dict(color=color, width=float(line_width)),
                    name=f"{pop} Median",
                    hovertemplate=f"{pop} Median<br>Time: %{{z:.2f}}<extra></extra>",
                    showlegend=True,
                    legendgroup=pop,
                    opacity=float(alpha),
                )
            )

    # ------------------------- support geometry hooks ----------------------

    def add_support_tube(
        self,
        *,
        pops: Optional[Sequence[str]] = None,
        # support volume / mask parameters
        grid_res_xy: int = 256,
        grid_res_u: int = 256,
        mass_frac: float = 0.90,
        sigma_xy: float = 1.5,
        sigma_u: float = 1.0,
        min_points_per_slice: int = 5,
        time_eps: float = 1.0,
        # mesh parameters
        tau: float = 0.5,
        u_smooth_sigma: float = 0.0,
        xy_smooth_sigma: float = 0.0,
        close_r_u: int = 0,
        close_r_xy: int = 0,
        opacity: float = 0.25,
        # plotting
        name_suffix: str = "tube",
        visible: str = "on",  # "on" | "legendonly" | "off"
        verbose: bool = False,
        x_edges: Optional[np.ndarray] = None,
        y_edges: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add a 3D support tube (Mesh3d) for each population.

        This is a thin wrapper that delegates all geometry work to src/support_tube.py.

        NOTE: support_tube.py does not exist yet in this step; this method is wired
        for the next step. It will raise a clear error until implemented.
        """
        if self.fig is None:
            raise RuntimeError("Call init_plot() before adding layers.")
        if visible not in {"on", "legendonly", "off"}:
            raise ValueError("visible must be 'on', 'legendonly', or 'off'")

        if pops is None:
            pops = [str(p) for p in self.populations]
        else:
            pops = [str(p) for p in pops]

        if visible == "off":
            return

        try:
            from src import support_tube  # type: ignore
        except Exception as e:
            raise ImportError(
                "add_support_tube() requires src/support_tube.py to exist and be importable. "
                "Next step: implement support_tube.compute_mesh_traces(...)."
            ) from e

        vis_flag = True if visible == "on" else "legendonly"

        for pop in pops:
            color = self.colors_dict.get(pop, self.default_color)

            # Expected API (to implement next):
            # returns a dict with keys: X, Y, Z, faces (i,j,k)
            mesh = support_tube.compute_support_tube_mesh(
                df=self.embeddings_df,
                pop=pop,
                xcol=self.xcol,
                ycol=self.ycol,
                tcol=self.tcol,
                popcol=self.popcol,
                x_edges=x_edges, 
                y_edges=y_edges,
                grid_res_xy=int(grid_res_xy),
                grid_res_u=int(grid_res_u),
                mass_frac=float(mass_frac),
                sigma_xy=float(sigma_xy),
                sigma_u=float(sigma_u),
                min_points_per_slice=int(min_points_per_slice),
                time_eps=float(time_eps),
                tau=float(tau),
                u_smooth_sigma=float(u_smooth_sigma),
                xy_smooth_sigma=float(xy_smooth_sigma),
                close_r_u=int(close_r_u),
                close_r_xy=int(close_r_xy),
                verbose=bool(verbose),
            )

            if mesh is None:
                if verbose:
                    print(f"[{pop}] support_tube mesh: None (no surface)")
                continue

            X, Y, Z = mesh["X"], mesh["Y"], mesh["Z"]
            faces = mesh["faces"]  # (F,3)

            self.fig.add_trace(
                go.Mesh3d(
                    x=X,
                    y=Y,
                    z=Z,
                    i=faces[:, 0],
                    j=faces[:, 1],
                    k=faces[:, 2],
                    color=color,
                    opacity=float(opacity),
                    name=f"{pop} {name_suffix}",
                    legendgroup=pop,
                    showlegend=False,
                    visible=vis_flag,
                    hoverinfo="skip",
                )
            )

    def add_support_rings(
        self,
        *,
        pops: Optional[Sequence[str]] = None,
        # support volume / mask parameters
        grid_res_xy: int = 256,
        grid_res_u: int = 256,
        mass_frac: float = 0.90,
        sigma_xy: float = 1.5,
        sigma_u: float = 1.0,
        min_points_per_slice: int = 5,
        time_eps: float = 1.0,
        # ring binning parameters
        nbins_old: int = 24,
        recent_bin_width: int = 1,
        override_window: int = 12,
        override_valid_frac: float = 0.7,
        tau: float = 0.5,
        tau_recent: Optional[float] = None,
        min_boundary_points: int = 50,
        # plotting
        line_width_recent: float = 5.0,
        line_width_old: float = 4.0,
        opacity: float = 0.96,
        name_suffix: str = "rings",
        visible: str = "on",  # "on" | "legendonly" | "off"
        verbose: bool = False,
        x_edges: Optional[np.ndarray] = None,
        y_edges: Optional[np.ndarray] = None,
    ) -> None:
        """
        Add mask-derived ring hull outlines for each population.

        Delegates geometry to src/support_tube.py (to be implemented next).
        """
        if self.fig is None:
            raise RuntimeError("Call init_plot() before adding layers.")
        if visible not in {"on", "legendonly", "off"}:
            raise ValueError("visible must be 'on', 'legendonly', or 'off'")

        if pops is None:
            pops = [str(p) for p in self.populations]
        else:
            pops = [str(p) for p in pops]

        if visible == "off":
            return

        try:
            from src import support_tube  # type: ignore
        except Exception as e:
            raise ImportError(
                "add_support_rings() requires src/support_tube.py to exist and be importable. "
                "Next step: implement support_tube.compute_ring_traces(...)."
            ) from e

        vis_flag = True if visible == "on" else "legendonly"
        tau_recent_val = float(tau) if tau_recent is None else float(tau_recent)

        for pop in pops:
            color = self.colors_dict.get(pop, self.default_color)

            # Expected API (to implement next):
            # returns list of dicts: each has x,y,z arrays, kind ("recent"/"old"), and hover fields
            rings = support_tube.compute_support_tube_rings(
                df=self.embeddings_df,
                pop=pop,
                xcol=self.xcol,
                ycol=self.ycol,
                tcol=self.tcol,
                popcol=self.popcol,
                x_edges=x_edges, 
                y_edges=y_edges,
                grid_res_xy=int(grid_res_xy),
                grid_res_u=int(grid_res_u),
                mass_frac=float(mass_frac),
                sigma_xy=float(sigma_xy),
                sigma_u=float(sigma_u),
                min_points_per_slice=int(min_points_per_slice),
                time_eps=float(time_eps),
                nbins_old=int(nbins_old),
                recent_bin_width=int(recent_bin_width),
                override_window=int(override_window),
                override_valid_frac=float(override_valid_frac),
                tau=float(tau),
                tau_recent=float(tau_recent_val),
                min_boundary_points=int(min_boundary_points),
                verbose=bool(verbose),
            )

            if not rings:
                if verbose:
                    print(f"[{pop}] support_tube rings: empty")
                continue

            for idx, r in enumerate(rings):
                kind = r.get("kind", "old")
                lw = float(line_width_recent) if kind == "recent" else float(line_width_old)

                self.fig.add_trace(
                    go.Scatter3d(
                        x=r["x"],
                        y=r["y"],
                        z=r["z"],
                        mode="lines",
                        line=dict(width=lw, color=color),
                        opacity=float(opacity),
                        name=f"{pop} {name_suffix}",
                        legendgroup=pop,
                        showlegend=(idx == 0),
                        visible=vis_flag,
                        hovertemplate=r.get("hovertemplate", f"{pop} {kind}<extra></extra>"),
                    )
                )