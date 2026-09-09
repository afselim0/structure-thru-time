#%%
"""
The complex four-population example under increasing migration: three embeddings
panels (A-C), one per migration rate, and the pair coalescence rates against the
no-migration 1/(2Ne) baseline (D).

Data: results/figure_data/complex_4_pops_migration/, written by
utils/runs/prepare_figure_data/complex_4_pops_migration.py
"""
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from results.figures.plotting_style import *
import sys
import os
filename = os.path.basename(__file__).split(".")[0]
os.chdir(os.path.dirname(__file__))
sys.path.append("../../../src")
from PlotEmbeddings import *

FIGURE = "complex_4_pops_migration"
all_embeddings = pd.read_parquet(figure_data(FIGURE, "embeddings.parquet"))
coalescence_rates = pd.read_parquet(figure_data(FIGURE, "coalescence_rates.parquet"))
inverse_2Ne = pd.read_parquet(figure_data(FIGURE, "inverse_2Ne.parquet"))
pop_labels = pd.read_csv(figure_data(FIGURE, "pop_labels.csv"))
pop_colors = pd.read_csv(figure_data(FIGURE, "pop_colors.csv"))

A_color = pop_colors.loc[pop_colors.population == "A", "color"].values[0]
B_color = pop_colors.loc[pop_colors.population == "B", "color"].values[0]

# the embeddings panels, weakest migration first
PANEL_M = ["0.0001", "0.001", "0.01"]
panel_embeddings = [(f"m = {m}", all_embeddings[all_embeddings["m"] == m])
                    for m in PANEL_M]

times = np.sort(all_embeddings["time"].unique()).astype(int)

# %% build the three embeddings plotters
# all three share the same time window and xy limits, so the panels are directly
# comparable: only the migration rate differs between them. The time axis is log-scaled,
# so it starts at the first non-zero timepoint rather than at 0
MIN_TIME, MAX_TIME = float(times[times > 0].min()), float(times.max())
XY_LIMS = max(
    np.abs(df[["X1", "X2"]].values).max()
    for _, df in panel_embeddings
) * 1.05

plotters = []
for _, embeddings_df in panel_embeddings:
    plotter = PlotEmbeddings(
        embeddings_df=embeddings_df, labels_df=pop_labels, colors_df=pop_colors,
        sample_id_col="individual_id",
    )
    plotter.init_plot(min_time=MIN_TIME, max_time=MAX_TIME, xy_lims=XY_LIMS)
    plotter.add_sample_trajectories()
    plotters.append(plotter)

# the three panels share a camera angle and data extent, so one probe render fixes the
# aspect ratio for all of them; their widths follow from it and the rates panel takes
# whatever width is left over
# each panel is ROW_H inches tall at 500 dpi; the cropped aspect ratio depends on this
# canvas size, so the probe here and the panel renders below must share it
RENDER_KW = dict(width=1500, height=1500)
_, aspect_ratio = plotters[0].save_static(**RENDER_KW, scale=1)  # cheap probe: scale does not affect the ratio

# %% figure layout
ROW_H = 3.0                       # height of the embeddings panels
RATES_W = 3.4                     # width of the rates panel's cell, incl. its label margins
RATES_MARGIN = dict(left=0.6, right=0.15, top=0.35, bottom=0.5)  # inches around the rates axes

# the embeddings panels' width is fixed by their rendered aspect ratio, so the figure
# width follows from them plus the space the rates panel needs
embeddings_w = ROW_H * aspect_ratio
rates_w = RATES_W
FIG_W, FIG_H = 3 * embeddings_w + rates_w, 3.2

if rates_w <= RATES_MARGIN["left"] + RATES_MARGIN["right"]:
    raise ValueError(
        f"RATES_W={RATES_W}in leaves no room for the rates axes inside its margins"
    )

fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=500)
# margins pinned to the full figure so the ratios below map to real inches
outer = gridspec.GridSpec(
    nrows=1, ncols=4,
    width_ratios=[embeddings_w, embeddings_w, embeddings_w, rates_w], wspace=0,
    left=0, right=1, top=ROW_H / FIG_H, bottom=0, figure=fig,
)
embeddings_axes = [fig.add_subplot(outer[0, i]) for i in range(3)]
rates_cell = fig.add_subplot(outer[0, 3])
rates_cell.axis("off")

# the rates panel needs room for its tick/axis labels, which the inch-exact grid above
# does not reserve; it is placed by hand inside its cell, inset by RATES_MARGIN
cell_pos = rates_cell.get_position()
rates_ax = fig.add_axes([
    cell_pos.x0 + RATES_MARGIN["left"] / FIG_W,
    cell_pos.y0 + RATES_MARGIN["bottom"] / FIG_H,
    cell_pos.width - (RATES_MARGIN["left"] + RATES_MARGIN["right"]) / FIG_W,
    cell_pos.height - (RATES_MARGIN["top"] + RATES_MARGIN["bottom"]) / FIG_H,
])

# %% panel d: pair coalescence rates against the no-migration 1/(2Ne) baseline
# 1/(2Ne) is the rate a pair would coalesce at with no migration, so the gap between
# each dashed/dotted curve and its solid baseline is what migration contributes
LINESTYLES = {
    "0.0001": "--",
    "0.001": (0, (3, 1, 1, 1)),
    "0.01": (0, (1, 1)),
}
POP_COLORS = {"A": A_color, "B": B_color}

for pop, color in POP_COLORS.items():
    baseline = inverse_2Ne[inverse_2Ne["population"] == pop]
    rates_ax.plot(baseline["time"], baseline["inverse_2Ne"], color=color)

for m, linestyle in LINESTYLES.items():
    for pop, color in POP_COLORS.items():
        curve = coalescence_rates[(coalescence_rates["m"] == m)
                                  & (coalescence_rates["population"] == pop)]
        rates_ax.plot(curve["time"], curve["rate"], color=color, linestyle=linestyle)

rates_ax.set_xscale("log")
rates_ax.set_yscale("log")
rates_ax.set_xlabel("Time (generations)")
rates_ax.set_ylabel("Coalescence rate")

# two keys rather than one combined one: colour carries the population and linestyle
# carries the migration rate, so listing every combination would need six entries
population_handles = [
    Line2D([], [], color=A_color, label="A"),
    Line2D([], [], color=B_color, label="B"),
]
rate_handles = [Line2D([], [], color="black", label="$1/(2N)$")] + [
    Line2D([], [], color="black", linestyle=LINESTYLES[m], label=f"m = {m}")
    for m in ("0.0001", "0.001", "0.01")
]
# both keys sit in the upper-left corner, which the curves leave empty: they all rise
# towards the right, so a lower-left key would be drawn over them
population_legend = rates_ax.legend(
    handles=population_handles, loc="upper left", fontsize=10, title="Population",
    title_fontsize=10, handlelength=1.2, handletextpad=0.4, labelspacing=0.2,
    borderpad=0.2,
)
rates_ax.add_artist(population_legend)
rates_ax.legend(
    handles=rate_handles, loc="upper left", bbox_to_anchor=(0.32, 1.0), fontsize=10,
    handlelength=1.6, handletextpad=0.4, labelspacing=0.2, borderpad=0.2,title_fontsize=10
)

# %% panels a-c: the static 3D embeddings, rendered last so each axis's position is final
SUBFIG_LABEL_SIZE = 12
TITLE_SIZE = 10

for letter, (title, _), plotter, ax in zip("ABC", panel_embeddings, plotters, embeddings_axes):
    img, _ = plotter.save_static(**RENDER_KW)
    img.save(f"../sim_figs_png/{filename}_{letter}.png")
    ax.set_axis_off()
    ax.imshow(img, aspect="equal")

# labels and titles are placed against each embeddings axis's gridspec position
label_baseline = max(ax.get_position().y1 for ax in embeddings_axes + [rates_cell])
for letter, (title, _), ax in zip("ABC", panel_embeddings, embeddings_axes):
    pos = ax.get_position()
    fig.text(pos.x0, label_baseline + 0.01, letter, ha="left", va="bottom",
             fontsize=SUBFIG_LABEL_SIZE, fontweight="bold")
    fig.text((pos.x0 + pos.x1) / 2, label_baseline + 0.01, title,
             ha="center", va="bottom", fontsize=TITLE_SIZE)

fig.text(cell_pos.x0, label_baseline + 0.01, "D", ha="left", va="bottom",
         fontsize=SUBFIG_LABEL_SIZE, fontweight="bold")

fig.savefig(f"../sim_figs/{filename}.pdf")
# %%
