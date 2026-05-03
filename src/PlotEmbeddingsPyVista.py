#!/usr/bin/env python3
"""PyVista-based 3D embeddings plotter.

Parallel to PlotEmbeddings.py but uses PyVista/VTK for proper depth sorting
and tube rendering. Keeps the same public interface so it can eventually
replace PlotEmbeddings if validated.

Key differences from the Plotly version:
- Sample trajectories rendered as tubes (cylinders along splines) rather than
  lines, giving proper 3D occlusion and lighting.
- True OpenGL depth buffering — no painter's-algorithm overprinting.
- HTML export via trame (requires pyvista[trame]).
- time_stride parameter to downsample time points, controlling render cost.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import pyvista as pv

TUBE_RADIUS_FRAC = 0.005   # tube radius as fraction of XY data range


class PlotEmbeddingsPyVista:
    """PyVista-based 3D embeddings plotter.

    Constructor takes the same arguments as PlotEmbeddings so the two classes
    are interchangeable.
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
    ):
        self.xcol = xcol
        self.ycol = ycol
        self.tcol = tcol
        self.sample_id_col = sample_id_col
        self.popcol = popcol
        self.default_color = default_color
        self.generation_time = generation_time

        df = embeddings_df.copy()

        if labels_df is None:
            df[self.popcol] = default_population
        else:
            df = df.merge(labels_df[[sample_id_col, popcol]], on=sample_id_col, how="left")
            df[self.popcol] = df[self.popcol].fillna(default_population)

        if generation_time is not None:
            df[self.tcol] = df[self.tcol].astype(float) * float(generation_time)

        # Build color mapping
        try:
            import plotly.express as px
            default_palette = px.colors.qualitative.Safe
        except ImportError:
            default_palette = [
                "#0072B2", "#E69F00", "#009E73", "#CC79A7",
                "#56B4E9", "#D55E00", "#F0E442", "#808080",
            ]

        if colors_df is None:
            pops = df[self.popcol].astype(str).unique().tolist()
            colors_dict = {p: default_palette[i % len(default_palette)] for i, p in enumerate(pops)}
            df["color"] = df[self.popcol].astype(str).map(colors_dict)
        else:
            cdf = colors_df[[popcol, "color"]].copy()
            cdf[popcol] = cdf[popcol].astype(str)
            df = df.merge(cdf, on=popcol, how="left")
            df["color"] = df["color"].fillna(default_color)
            colors_dict = {
                str(row[popcol]): str(row["color"])
                for _, row in colors_df[[popcol, "color"]].drop_duplicates().iterrows()
            }

        self.embeddings_df = df
        self.colors_dict: Dict[str, str] = colors_dict
        self.populations = df[self.popcol].astype(str).unique()

        # Set in init_plot
        self.plotter: Optional[pv.Plotter] = None
        self.time_scale: Optional[str] = None
        self.zcol: str = tcol
        self._xy_range: float = 1.0
        self._x_range: Tuple[float, float] = (-1.0, 1.0)
        self._y_range: Tuple[float, float] = (-1.0, 1.0)
        self._z_range: Tuple[float, float] = (0.0, 1.0)
        self._z_tick_vals: Optional[np.ndarray] = None
        self._z_tick_labels: Optional[List[str]] = None

    # ------------------------------------------------------------------ #
    #  Plot setup
    # ------------------------------------------------------------------ #

    def init_plot(
        self,
        *,
        time_scale: str = "log",
        time_stride: int = 1,
        xy_lims: Optional[float] = None,
        min_time: Optional[float] = None,
        max_time: Optional[float] = None,
        background_color: str = "white",
        window_size: Tuple[int, int] = (1200, 900),
    ) -> None:
        if time_scale not in {"log", "linear"}:
            raise ValueError("time_scale must be 'log' or 'linear'")

        self.time_scale = time_scale
        df = self.embeddings_df.copy()

        # Filter by time range
        if min_time is not None:
            df = df[df[self.tcol].astype(float) >= float(min_time)]
        if max_time is not None:
            df = df[df[self.tcol].astype(float) <= float(max_time)]

        # Subsample time points
        if time_stride > 1:
            all_times = np.sort(df[self.tcol].unique())
            df = df[df[self.tcol].isin(all_times[::time_stride])]

        # Compute z column (log-transform for log scale)
        if time_scale == "log":
            self.zcol = "__z_log"
            df[self.zcol] = np.log10(np.maximum(df[self.tcol].astype(float), 1e-10))
        else:
            self.zcol = self.tcol

        self.embeddings_df = df

        # XY axis ranges
        if xy_lims is not None:
            self._xy_range = float(xy_lims)
            self._x_range = (-float(xy_lims), float(xy_lims))
            self._y_range = (-float(xy_lims), float(xy_lims))
        else:
            x_min = float(df[self.xcol].min())
            x_max = float(df[self.xcol].max())
            y_min = float(df[self.ycol].min())
            y_max = float(df[self.ycol].max())
            common_min = min(x_min, y_min)
            common_max = max(x_max, y_max)
            pad = 0.05 * max(common_max - common_min, 1.0)
            self._xy_range = common_max - common_min
            self._x_range = (common_min - pad, common_max + pad)
            self._y_range = (common_min - pad, common_max + pad)

        z_min = float(df[self.zcol].min())
        z_max = float(df[self.zcol].max())
        self._z_range = (z_min, z_max)

        # Z-axis tick labels
        if time_scale == "log":
            decade_min = int(np.floor(z_min))
            decade_max = int(np.ceil(z_max))
            raw_ticks = 10.0 ** np.arange(decade_min, decade_max + 1)
            self._z_tick_vals = np.log10(raw_ticks)
            self._z_tick_labels = [str(int(t)) for t in raw_ticks]
        else:
            self._z_tick_vals = np.linspace(z_min, z_max, 6)
            self._z_tick_labels = [str(int(t)) for t in self._z_tick_vals]

        self.plotter = pv.Plotter(window_size=list(window_size), off_screen=True)
        self.plotter.background_color = background_color

        # Add a second fill light from below-left to soften shadows on tubes
        fill_light = pv.Light(position=(-1, -1, -1), light_type="scene light", intensity=0.4)
        self.plotter.add_light(fill_light)

    # ------------------------------------------------------------------ #
    #  Layers
    # ------------------------------------------------------------------ #

    def add_sample_trajectories(
        self,
        *,
        tube_radius: Optional[float] = None,
        alpha: float = 0.8,
        sample_stride: int = 1,
    ) -> None:
        if self.plotter is None:
            raise RuntimeError("Call init_plot() before adding layers.")

        df = self.embeddings_df
        if tube_radius is None:
            tube_radius = max(self._xy_range * TUBE_RADIUS_FRAC, 1e-6)

        samples = df[self.sample_id_col].unique()
        if sample_stride > 1:
            samples = samples[::sample_stride]

        # Shuffle to avoid systematic population overprinting
        rng = np.random.default_rng(seed=0)
        samples = rng.permutation(samples)

        for sample in samples:
            sample_data = (
                df[df[self.sample_id_col] == sample]
                .sort_values(self.tcol)
            )
            if len(sample_data) < 2:
                continue

            population = str(sample_data[self.popcol].values[0])
            color = self.colors_dict.get(population, self.default_color)

            pts = np.column_stack([
                sample_data[self.xcol].values.astype(float),
                sample_data[self.ycol].values.astype(float),
                sample_data[self.zcol].values.astype(float),
            ])

            spline = pv.Spline(pts, n_points=len(pts))
            tube = spline.tube(radius=tube_radius)
            self.plotter.add_mesh(
                tube,
                color=color,
                opacity=alpha,
                smooth_shading=True,
                ambient=0.3,
                diffuse=0.7,
                specular=0.1,
            )

    def add_legend(self) -> None:
        if self.plotter is None:
            raise RuntimeError("Call init_plot() before adding layers.")
        legend_entries = [
            [pop, self.colors_dict.get(pop, self.default_color)]
            for pop in self.populations
        ]
        self.plotter.add_legend(
            legend_entries,
            bcolor="white",
            border=True,
            size=(0.15, min(0.05 * len(self.populations) + 0.05, 0.4)),
        )

    def show_axes(self) -> None:
        if self.plotter is None:
            raise RuntimeError("Call init_plot() before adding layers.")
        z_label = (
            "log10(Time)"
            if self.time_scale == "log"
            else ("Time (years ago)" if self.generation_time else "Time (generations ago)")
        )
        self.plotter.show_bounds(
            bounds=[
                self._x_range[0], self._x_range[1],
                self._y_range[0], self._y_range[1],
                self._z_range[0], self._z_range[1],
            ],
            xtitle="X1",
            ytitle="X2",
            ztitle=z_label,
            grid=True,
            location="outer",
            font_size=10,
        )

    # ------------------------------------------------------------------ #
    #  Output
    # ------------------------------------------------------------------ #

    def save_html(self, path: str) -> None:
        """Export interactive HTML via trame (requires pyvista[trame])."""
        if self.plotter is None:
            raise RuntimeError("Call init_plot() before saving.")
        self.plotter.export_html(path)

    def save_static(
        self,
        path: str,
        *,
        camera_position: Optional[list] = None,
    ) -> None:
        if self.plotter is None:
            raise RuntimeError("Call init_plot() before saving.")
        if camera_position is not None:
            self.plotter.camera_position = camera_position
        self.plotter.screenshot(path)
