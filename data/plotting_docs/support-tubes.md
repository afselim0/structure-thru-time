# Support Tubes

## What is a support tube?

A support tube is a population-level geometric envelope that summarizes where the bulk of sample trajectories lie in embedding space as a function of time. Rather than plotting thousands of individual paths, the tube provides a compact representation of population structure through time.

The tube is rendered as a semi-transparent 3D surface in (X, Y, Time) space, where X and Y are embedding coordinates and Time is plotted on a log-scaled axis. Optionally, ring contour outlines are overlaid at discrete time slices for additional detail.

The construction proceeds in four phases:

**Phase A -- Per-slice masks.** Time is log-transformed and discretized into slices. For each slice, a robust center (median or mean) is computed, the closest `mass_frac` fraction of points is retained, and a convex hull is rasterized onto a global XY grid.

**Phase B -- Temporal smoothing.** Empty slices are filled by copying from the nearest non-empty neighbor. Optional Gaussian smoothing along the time axis regularizes small fluctuations, producing a continuous float volume in [0, 1].

**Phase C -- Surface extraction.** The volume is thresholded and marching cubes extracts a triangulated mesh surface. Voxel coordinates are mapped back to real embedding and time coordinates.

**Phase D -- Ring outlines.** Contour polylines are extracted per time-bin from the thresholded volume, providing cross-sectional outlines at discrete time points.

### Interpretation

- The tube is not a convex hull of trajectories; it captures the central `mass_frac` of the population at each time slice.
- The tube is not guaranteed to be smooth under sudden demographic shifts. Abrupt jumps reflect genuine structural discontinuities (e.g., instantaneous migration events).


## Pipeline usage

### Standalone

```bash
python utils/plot_support_tubes.py \
  --input path/to/embeddings.parquet \
  --colors path/to/pop_colors.csv \
  --labels path/to/pop_labels.csv \
  --output_path output/dir \
  --output_name my_tubes.html \
  --grid_res_xy 50 \
  --grid_res_u 50 \
  --mass_frac 0.90
```

### Via Snakemake (empirical)

The `plot_support_tubes` rule is opt-in (not in default targets). Run it explicitly from `results/empirical/`:

```bash
cd results/empirical
snakemake plot_support_tubes --config \
    dataset=Wohns \
    trees_dir=../../data/trees/wohns_chopped \
    n_samples=2 \
    N0=10000 \
    min_time=1 \
    max_time=4000 \
    min_window_size=100 \
    max_window_size=1500 \
    n_timepoints=4000 \
    output_dir=Wohns/all_1 \
    grid_res_xy=50 \
    grid_res_u=50 \
    --cores all
```

Tube-specific config keys (all optional, have sensible defaults):

| Key | Default | Description |
|-----|---------|-------------|
| `grid_res_xy` | 256 | XY grid resolution |
| `grid_res_u` | 256 | Number of time slices |
| `mass_frac` | 0.90 | Fraction of population mass |
| `sigma_u` | 1.0 | Temporal smoothing |

### Interactive tuning (dev only)

The sandbox script provides a wider set of experimental knobs:

```bash
python sandbox/convex_hull_sandbox.py \
  --outputs_dir /tmp/test \
  --embeddings path/to/embeddings.parquet \
  --labels path/to/pop_labels.csv \
  --colors path/to/pop_colors.csv \
  --show_tube --show_median \
  --grid_res_xy 50 --grid_res_u 50 \
  --mass_frac 0.90
```


## Parameter reference

### Primary (most users adjust these)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `grid_res_xy` | 256 | Resolution of the XY grid. Lower values (50-100) render faster and often look cleaner. Higher values capture finer spatial detail. |
| `grid_res_u` | 256 | Number of time slices along the log-time axis. More slices capture finer temporal detail. |
| `mass_frac` | 0.90 | Fraction of population mass captured per time slice. 0.90 = keep the closest 90% of points to the robust center. Higher values produce wider tubes. |
| `sigma_u` | 1.0 | Gaussian smoothing width along the time axis (in units of time-slice bins). Larger values produce smoother tubes at the cost of temporal resolution. |

### Secondary (occasionally useful)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tau` | 0.5 | Threshold for converting the smoothed volume to a binary mask before marching cubes. Usually left at 0.5. |
| `trim_center` | `"median"` | Robust center used for radial trimming: `"median"` or `"mean"`. Median is more robust to outliers. |
| `min_points_per_slice` | 5 | Minimum number of sample points in a time slice for it to produce a valid mask. |
| `time_eps` | 1.0 | Floor value for time before log-transform: `u = log10(max(t, time_eps))`. Prevents log(0). |

### Advanced (rarely touched)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `post_smooth_u` | 0.0 | Additional Gaussian smoothing along time after volume construction. |
| `post_smooth_xy` | 0.0 | Additional Gaussian smoothing in XY after volume construction. |
| `close_r_u` | 0 | Morphological closing radius along time axis (fills small temporal gaps). |
| `close_r_xy` | 0 | Morphological closing radius in XY (fills small spatial holes). |
| `nbins_old` | 24 | Number of time bins for "old" ring outlines. |
| `recent_bin_width` | 1 | Bin width (in slices) for recent ring outlines. |
| `override_window` | 12 | Number of recent slices checked for the recent-ring override heuristic. |
| `override_valid_frac` | 0.7 | Fraction of recent slices that must have valid contours to enable the recent-ring mode. |


## Architecture

| File | Role |
|------|------|
| `utils/support_tube.py` | Geometry engine: builds support volumes, extracts mesh surfaces and ring contours. No plotting code. |
| `src/PlotEmbeddings.py` | Plotting orchestrator: manages the Plotly figure, delegates geometry to `utils/support_tube.py` via `add_support_tube()` and `add_support_rings()`. |
| `utils/plot_support_tubes.py` | CLI wrapper: reads data, calls PlotEmbeddings methods, writes HTML output. Used by Snakemake. |
| `sandbox/convex_hull_sandbox.py` | Dev-only parameter playground. Not used by the pipeline. |
