#%%
"""
Compare Gaussian and Cauchy kernels, highlighting how a fixed band of
"moderate similarity" on the y-axis corresponds to very different distance
ranges on the x-axis -- the mechanism behind the t-SNE crowding problem.

Data: results/figure_data/GaussianvsCauchyKernels/, written by
utils/runs/prepare_figure_data/GaussianvsCauchyKernels.py
"""
import matplotlib.pyplot as plt
import pandas as pd
import os

# Relative paths below are anchored at this script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# This figure deliberately does not import plotting_style: it is drawn with
# matplotlib's defaults (in particular an untrimmed savefig bbox).
FIGURE = "GaussianvsCauchyKernels"
DATA = os.path.join("..", "..", "figure_data", FIGURE)
curves = pd.read_parquet(os.path.join(DATA, "kernels.parquet"))
bands = pd.read_csv(os.path.join(DATA, "similarity_band.csv")).set_index("kernel")
params = pd.read_csv(os.path.join(DATA, "params.csv")).set_index("parameter")["value"]

y_low, y_high = params["y_low"], params["y_high"]
x_gauss_low, x_gauss_high = bands.loc["gaussian", ["x_low", "x_high"]]
x_cauchy_low, x_cauchy_high = bands.loc["cauchy", ["x_low", "x_high"]]
#%%
fig, ax = plt.subplots(figsize=(6, 4), dpi=500)

# Plot the two kernels.
ax.plot(curves["x"], curves["gaussian"], color="#1f77b4", linewidth=2.5,
        label="Gaussian: $\\exp(-x^2/2)$")
ax.plot(curves["x"], curves["cauchy"], color="#d62728", linewidth=2.5,
        label="Cauchy: $1/(1+x^2)$")

# Shade the "moderate similarity" band on the y-axis.
ax.axhspan(y_low, y_high, color="gray", alpha=0.15, zorder=0)
ax.text(5.7, (y_low + y_high) / 2, "moderate\nsimilarity", fontsize=9,
        color="dimgray", va="center", ha="right", style="italic")

# Highlight the corresponding x-range for each kernel with a shaded vertical span.
ax.axvspan(x_gauss_low, x_gauss_high, color="#1f77b4", alpha=0.15, zorder=0)
ax.axvspan(x_cauchy_low, x_cauchy_high, color="#d62728", alpha=0.15, zorder=0)

# Annotate the two distance ranges with arrows.
def annotate_range(x_low, x_high, y_pos, color, label):
    ax.annotate(
        "", xy=(x_high, y_pos), xytext=(x_low, y_pos),
        arrowprops=dict(arrowstyle="<->", color=color, lw=1.8),
    )
    ax.text((x_low + x_high) / 2, y_pos + 0.02, label, color=color,
            fontsize=9, ha="center", fontweight="bold")

annotate_range(x_gauss_low, x_gauss_high, 0.34, "#1f77b4",
                f"Gaussian: x $\\in$ [{x_gauss_low:.2f}, {x_gauss_high:.2f}]")
annotate_range(x_cauchy_low, x_cauchy_high, 0.40, "#d62728",
                f"Cauchy: x $\\in$ [{x_cauchy_low:.2f}, {x_cauchy_high:.2f}]")

# Dotted guide lines from the band edges down to the x-axis, for each kernel.
for x_val, y_val, color in [
    (x_gauss_low, y_high, "#1f77b4"), (x_gauss_high, y_low, "#1f77b4"),
    (x_cauchy_low, y_high, "#d62728"), (x_cauchy_high, y_low, "#d62728"),
]:
    ax.plot([x_val, x_val], [0, y_val], color=color, linestyle=":", linewidth=1, alpha=0.6)

ax.set_xlim(0, 6)
ax.set_ylim(0, 1)
ax.set_xlabel("distance $x$", fontsize=11)
ax.set_ylabel("similarity $f(x)$", fontsize=11)
ax.legend()
ax.grid(alpha=0.25)

fig.tight_layout()
filename = os.path.basename(__file__).split(".")[0]
fig.savefig(f"../sim_figs/{filename}.pdf")
# %%
