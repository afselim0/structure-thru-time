#%%
"""
Pairwise coalescent rates recovered from ARGs inferred by different methods, against
the analytic truth and the estimate on the true trees, for a population split (A) and
a single sawtooth population (B).

Data: results/figure_data/arg_rates_comparison/, written by
utils/runs/prepare_figure_data/arg_rates_comparison.py. Every source is already on
one time grid per demography, so this script only draws.
"""
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from results.figures.plotting_style import *

os.chdir(os.path.dirname(__file__))

FIGURE = os.environ.get("ARG_RATES_FIGURE_DATA", "arg_rates_comparison")

DEMOGRAPHIES = ["split", "sawtooth"]
PANEL_TITLES = {"split": "A", "sawtooth": "B"}
# analytic truth in black, the estimate on the true trees in grey, one Okabe-Ito
# colour per inference method
SOURCE_COLORS = {
    "true": "#000000",
    "sim_trees": "#7F7F7F",
    "tsinfer": okabe_ito[0],
    "relate": okabe_ito[1],
    "singer": okabe_ito[2],
}
SOURCE_LABELS = {
    "true": "analytic truth",
    "sim_trees": "true trees",
    "tsinfer": "tsinfer + tsdate",
    "relate": "Relate",
    "singer": "SINGER + POLEGON",
}
# between-population rates fall to the 1e-16 clip before the split, so the axis is
# floored well above it
RATE_FLOOR = 1e-6

mean_rates = pd.read_parquet(figure_data(FIGURE, "mean_rates.parquet"))
params = pd.read_csv(figure_data(FIGURE, "params.csv"))


def param(demography, name, cast=float):
    value = params.loc[(params["demography"] == demography)
                       & (params["parameter"] == name), "value"].iloc[0]
    return cast(value)

# %%
fig, axs = plt.subplots(1, 2, figsize=(7, 3), sharey=True)

for ax, demography in zip(axs, DEMOGRAPHIES):
    demography_means = mean_rates[mean_rates["demography"] == demography]

    # every pair of every source as a thin line first, so no source's cloud is
    # drawn over another's mean
    for source, color in SOURCE_COLORS.items():
        rates = pd.read_parquet(
            figure_data(FIGURE, f"rates_{demography}_{source}.parquet"))
        times = rates.columns.values.astype(float)
        # rasterized: ~12M vector segments across both panels would make the PDF
        # tens of MB and slow to open. The means, axes and all text stay vector.
        for row in rates.to_numpy():
            ax.plot(times, row, color=color, lw=0.15, alpha=0.15, zorder=1,
                    rasterized=True)

    # then the group means on top: colour names the source, style the grouping
    for source, color in SOURCE_COLORS.items():
        for group, style in (("within", "-"), ("between", ":")):
            group_mean = demography_means[(demography_means["source"] == source)
                                          & (demography_means["pair_type"] == group)]
            if group_mean.empty:
                continue
            ax.plot(group_mean["time"], group_mean["rate"], color=color,
                    ls=style, lw=1.6, zorder=3)

    ax.set_xscale(param(demography, "time_scale", str))
    ax.set_yscale("log")
    ax.set_xlim(param(demography, "min_time"), param(demography, "max_time"))
    ax.set_ylim(RATE_FLOOR, param(demography, "max_rate_plot"))
    ax.set_xlabel("Time (generations ago)")
    ax.set_title(PANEL_TITLES[demography], loc="left", fontweight="bold")

axs[0].set_ylabel("Coalescent rate")

# one legend for the whole figure: colour names the rate source, line style the
# within/between grouping (the sawtooth panel has only within pairs)
source_handles = [Line2D([], [], color=c, lw=1.8, label=SOURCE_LABELS[s])
                  for s, c in SOURCE_COLORS.items()]
group_handles = [Line2D([], [], color="black", lw=1.8, ls="-", label="within"),
                 Line2D([], [], color="black", lw=1.8, ls=":", label="between")]
fig.legend(handles=source_handles + group_handles, loc="lower center",
           bbox_to_anchor=(0.5, -0.02), fontsize=8, ncol=7, columnspacing=1.2)

fig.tight_layout(rect=(0, 0.06, 1, 1))
filename = os.path.basename(__file__).split(".")[0]
fig.savefig(f"../app_figs/{filename}.pdf", dpi=400)
fig.savefig(f"../app_figs_png/{filename}.png", dpi=300)
print(f"Saved ../app_figs/{filename}.pdf and ../app_figs_png/{filename}.png")
# %%
