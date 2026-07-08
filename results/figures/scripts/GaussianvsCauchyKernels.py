"""
Compare Gaussian and Cauchy kernels, highlighting how a fixed band of
"moderate similarity" on the y-axis corresponds to very different distance
ranges on the x-axis -- the mechanism behind the t-SNE crowding problem.
"""
import numpy as np
import matplotlib.pyplot as plt

# Both kernels normalized to peak at 1, for direct illustrative comparison.
# Gaussian: exp(-x^2 / (2*sigma^2)), sigma=1 -> exp(-x^2/2)
# Cauchy:   1 / (1 + x^2)
x = np.linspace(0, 6, 2000)
gaussian = np.exp(-x**2 / 2)
cauchy = 1.0 / (1.0 + x**2)

# Define the "moderate similarity" band on the y-axis.
y_low, y_high = 0.05, 0.25

# For each kernel, find the x-range where the curve falls within the band.
# Since both kernels are monotonically decreasing, invert analytically:
#   Gaussian: y = exp(-x^2/2)      -> x = sqrt(-2 ln y)
#   Cauchy:   y = 1/(1+x^2)        -> x = sqrt(1/y - 1)
def gaussian_inv(y):
    return np.sqrt(-2 * np.log(y))

def cauchy_inv(y):
    return np.sqrt(1 / y - 1)

x_gauss_low, x_gauss_high = gaussian_inv(y_high), gaussian_inv(y_low)
x_cauchy_low, x_cauchy_high = cauchy_inv(y_high), cauchy_inv(y_low)

fig, ax = plt.subplots(figsize=(9, 6))

# Plot the two kernels.
ax.plot(x, gaussian, color="#1f77b4", linewidth=2.5, label="Gaussian: $\\exp(-x^2/2)$")
ax.plot(x, cauchy, color="#d62728", linewidth=2.5, label="Cauchy: $1/(1+x^2)$")

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
ax.set_ylim(0, 1.05)
ax.set_xlabel("distance $x$", fontsize=11)
ax.set_ylabel("similarity $f(x)$", fontsize=11)
ax.set_title(
    "Same similarity band, very different distance ranges:\n"
    "the Cauchy tail spreads 'moderate similarity' over a much wider distance range",
    fontsize=11,
)
ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
ax.grid(alpha=0.25)

fig.tight_layout()
fig.savefig("/mnt/user-data/outputs/gaussian_vs_cauchy_kernel.png", dpi=200)
print("Gaussian range width:", x_gauss_high - x_gauss_low)
print("Cauchy range width:", x_cauchy_high - x_cauchy_low)