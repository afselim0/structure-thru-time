---
name: figure-designer
description: Design, write and review publication-quality manuscript figures for CoaSTT (matplotlib + rendered 3D embedding panels). Use whenever creating or editing a script in results/figures/plotting_scripts/, changing results/figures/plotting_style.py, or when a figure needs fixing, resizing, restyling or reviewing.
---

# Figure designer

You produce figures for a LaTeX article. Every figure is drawn by one script in
`results/figures/plotting_scripts/<figure_name>.py` from precomputed tables in
`results/figure_data/<figure_name>/`, using the house style in
`results/figures/plotting_style.py`. Work in three phases: **design → write → review**.
Never skip the review; it is where most figures fail.

Reference figure for what "good" looks like: `results/figures/sim_figs/double_SS_2d.pdf`
(rasterise it with `pdftoppm -r 110 -png` and look at it if in doubt).

## Hard rules

| Rule | Value / where enforced |
| --- | --- |
| Width | ≤ 6.5 in (double column, default) or 3.25 in (single column). `WIDTHS`, checked by `save_figure` |
| Height | ≤ 8.5 in. `MAX_HEIGHT`, checked by `save_figure` |
| Outputs | PDF (vector, for the paper) to `results/figures/{sim,app}_figs/<name>.pdf` **and** PNG (for inspection) to `{sim,app}_figs_png/<name>.png`. Both via `save_figure` |
| Font | Latin Modern Roman (shipped in `results/figures/fonts/`), Computer Modern math (`mathtext.fontset = cm`). Never `usetex` |
| Font sizes | from `FONT_SIZES` in plotting_style.py (see table below). Nothing below `FONT_SIZES["min"]` = 7 pt; `save_figure` raises otherwise |
| Panel labels | bold upper-case `A, B, C…`, 11 pt, inside the axes at the top-left, via `add_panel_label(ax, "A")`. If it collides with an element, push it slightly further up (`dy>0`) or left (`dx<0`), never anywhere else |
| Layout | `GridSpec` with strips in **inches** (`inch_gridspec`, `add_axes_inches`). Never `constrained_layout`, `tight_layout` or `plt.subplots` defaults for the final layout |
| Gaps | minimal between subplots; no overlap of any two elements (labels, ticks, legends, titles, images) |
| Computation | none. Scripts read tables/YAML and draw. No msprime, no optimizer, no Procrustes, no `results/outputs/`. Exception: `demesdraw` drawing a demes file |
| Colours | population colours come from the figure's `pop_colors.csv`, never hard-coded. Categorical fallback `okabe_ito`. **Black** reserved for truth/analytic, **grey** (`SIMULATED_COLOR`) for simulated-tree results. Sequential `viridis`, diverging `RdBu_r` |
| Curves | truth = dashed black (`TRUTH_STYLE`); estimate = solid colour (`ESTIMATE_STYLE`); replicates = thin translucent (`REPLICATE_STYLE`) |
| Axis labels | sentence case, units in parentheses. Time is always `"Time (generations ago)"` (`TIME_LABEL`) or `"Time (years ago)"`, never plain "generations". Log time axes use `10^k` ticks (`log_axis`) |
| Aligned 3D panels | whenever a figure shows more than one 3D embeddings panel, the embeddings must be Procrustes-aligned to a common reference with `procrustes_align_trajectories_df` (`src/OptimizationFunctions.py`: reference frame = the reference's first timepoint, matched by `individual_id`). Alignment is data preparation, so it happens in the figure's prep script, which writes the aligned tables; the plotting script never aligns. If the tables are not aligned (frames are visibly rotated/reflected relative to each other), fix the prep script |
| Legends | one shared legend per figure in its own strip when panels share an encoding; inside a panel only when it fits without covering data |
| No argparse | none. Constants at the top of the script are the only knobs |
| Style changes | if a figure needs something every figure will want, put it in `plotting_style.py`, not in the script |

### Font size table (edit in `plotting_style.py` → `FONT_SIZES`)

| Element | key | pt |
| --- | --- | --- |
| Figure / row / column titles | `title` | 11 |
| Panel letters (bold) | `panel_label` | 11 |
| Axis labels, in-panel text, legend title | `base` | 9 |
| Tick labels | `tick` | 8 |
| Legend entries | `legend` | 8 |
| Small annotations, colourbar ticks | `annotation` | 7 |
| Absolute floor | `min` | 7 |

Use the keys (`fontsize=FONT_SIZES["annotation"]`), never literal numbers, so a
later change to the table propagates.

## Phase 1: design

Before code, write the panel plan and put it in the script's header docstring:

1. List the panels with their letters, what each shows, and the data file/columns it uses.
2. Choose the width class (default `"double"` = 6.5 in). Sketch the strips: for each
   row and column give an inch budget including **margins for labels** (a y-axis
   label + tick labels needs ≈0.5–0.6 in; an x-axis label + ticks ≈0.4 in; a title
   strip ≈0.25 in; a legend strip ≈0.3–0.45 in). The strips must sum exactly to
   the figure size: axes that touch the figure edge lose their labels to the size check.
3. Raster panels (3D embeddings from `PlotEmbeddings.save_static`) dictate their
   own aspect ratio: probe it first (`_, ar = plotter.save_static(..., scale=1)`) and
   derive the panel width from the height (`w = h * ar`) or vice versa. Probe and
   final render must share `width`/`height`.
4. Decide the single legend location and what belongs in `params.csv` (time windows,
   chosen timepoints, `N_max`) versus what is a plotting constant at the top of the script.

If the data for a panel does not exist in `results/figure_data/<figure_name>/`, stop
and say so: the fix is in `utils/runs/prepare_figure_data/<figure_name>.py`, not in
the plotting script. Do not compute it in the plotting script.

## Phase 2: write

Copy [`template.py`](template.py) and fill it in. The skeleton is fixed:

```
#%% header docstring: panel list (A, B, …), data files, prep script
imports; from results.figures.plotting_style import *
FIGURE = "<figure_name>"; KIND = "sim" | "app"
# ---- plotting parameters (the only knobs) ----  inch budgets, timepoints, colours…
# ---- data ----  load_data() -> dict of DataFrames / dicts; no computation beyond indexing
# ---- panels ----  one function per panel: panel_a(ax, data), panel_b(ax, data)…
# ---- assembly ----  assemble(data) -> fig : builds strips, calls each panel, labels, legend
#%% if __name__ == "__main__": save_figure(assemble(load_data()), FIGURE, KIND)
```

Conventions inside a script:

- Keep the `#%%` cell markers so the script also runs interactively cell by cell.
- Each panel function takes an `ax` (or a `SubplotSpec` when it needs sub-axes)
  and the loaded data, and returns handles it wants the assembler to reuse
  (legend handles, colour mappings). It sets its own labels, scales and limits.
- `assemble()` owns every inch: builds the figure with `new_figure`, the grid with
  `inch_gridspec`, places panels, calls `add_panel_label`, places the shared legend.
- Panel letters go in reading order (left→right, top→bottom); one letter per
  logical panel group, as in the reference figure (C labels a 2×4 block).
- Shared axes limits across comparable panels: compute them once from the data
  (that is indexing, not computation) and pass them in.
- Tick density: at ≤1.5 in panels use `MaxNLocator(nbins=2–3)`; decade ticks only
  on log axes.
- Every text call takes `fontsize=FONT_SIZES[...]` only when it deviates from the
  rcParams default; otherwise omit it.
- Demography schematics: `demesdraw.tubes(graph, ax=..., colours=pop_colors_dict, log_time=..., max_time=..., labels="xticks")`
  (API: https://grahamgower.github.io/demesdraw/latest/api.html). Then set the
  y label to `TIME_LABEL` and align its y limits with the neighbouring embedding panel.
- Raster panels: `place_image(ax, img)`; the axis has no spines/ticks. Text inside
  the raster (Plotly axis titles) must also read at ≥ 7 pt at final size: raise
  `width`/`height` in `save_static` to shrink text relative to the plot, or set
  `font_size`.

## Phase 3: review (mandatory)

Run the script, then **look at the PNG** (`Read` the file in
`results/figures/{sim,app}_figs_png/`). For fine detail rasterise the PDF at higher
resolution: `pdftoppm -r 300 -png <pdf> <out>` and view crops. Check every item:

- [ ] `save_figure` ran without raising (width, height, minimum font size).
- [ ] Nothing is clipped at the figure edge; nothing overlaps: tick labels vs.
      neighbouring panel, axis labels vs. panel letters, legend vs. data, titles vs. tick labels.
- [ ] Panel letters present, bold, upper-case, in order, at the top-left inside corner, consistent offsets.
- [ ] Gaps between subplots are minimal and even; panels in a row share a baseline.






- [ ] Data ranges: shared limits where panels are compared; no panel is an unreadable blob (use insets, as in the reference figure).
- [ ] Nothing was computed in the script that belongs in the prep script.

Fix, rerun, re-inspect until every box is ticked. Report the final saved size printed
by `save_figure` and anything you could not resolve (e.g. missing figure data).

## Lessons already learned (do not rediscover)

- `savefig.pad_inches` is 0: the inch strips *are* the saved page, so every label
  needs a strip to live in. If `save_figure` reports an overflow of a few hundredths
  of an inch, a label is poking out of the figure, not a rounding issue.
- 3D embedding panels narrower than ~2.5 in cannot carry legible Plotly tick
  labels (7 pt at final size means ~30 px on a 700 px canvas, which swamps the plot).
  For small multiples hide them (`plotter.fig.update_layout(scene=dict(xaxis=dict(showticklabels=False, title=""), ...))`,
  keep the planes so every panel crops identically) and give the time direction once,
  with a labelled arrow in the row-title strip. See `two_pop_migration.py`.
- A rotated row title must be shorter than its row is tall; break it into two lines
  (`"Recent\nmigration"`) rather than shrinking it below the title size.
- Panel letters on raster panels: `add_panel_label(ax, "A", dx=2, dy=0)`; the
  plot's top-left corner is white in the standard camera.
- Every render in a figure should share `RENDER_KW` and `xy_lims` so one cheap
  probe (`scale=1`) gives the aspect ratio of all panels.

## Running

```
cd /project/jnovembre/afselim/structure-thru-time
export PATH=/project/jnovembre/afselim/CoaSTT_env/bin:$PATH PYTHONPATH=$PWD
python results/figures/plotting_scripts/<figure_name>.py
```

`from results.figures.plotting_style import *` needs the repo root on `PYTHONPATH`
(the `generate_*_figs.sh` drivers set it). 3D panels need kaleido's Chrome; if
`ChromeNotFoundError` appears run once:
`SSL_CERT_FILE=$(python -c "import certifi;print(certifi.where())") python -c "from kaleido import get_chrome_sync; print(get_chrome_sync())"`.
New sim figures are picked up by `generate_sim_figs.sh` automatically; empirical
(`app`) figures must be listed in `generate_app_figs.sh` and excluded in `generate_sim_figs.sh`.
