#!/usr/bin/env python3
"""Schematic of the 100-deme linear stepping-stone model (SS_N100_m_001.yaml).

100 demes arranged along the X axis, each rendered as a vertical tube through
time (Z axis). Migration connections shown at evenly-spaced time slices;
bidirectional arrows drawn at the top layer.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

N_DEMES = 100
T = 1.0        # normalized time axis
SPACING = 2.2  # distance between deme centers
r3 = 0.44      # cylinder radius
N_th = 24      # angular resolution
N_z  = 2       # height resolution for cylinder side
N_r  = 8       # radial resolution for cap discs
N_MIGRATION_LEVELS = 6

# Color gradient: purple → cyan → green → yellow → red (matches 2D scheme)
_cmap = LinearSegmentedColormap.from_list(
    "ss1d",
    ["#8000ff", "#00b5eb", "#80ffb4", "#ffb360", "#ff0000"],
    N=N_DEMES,
)
COLORS = [_cmap(i / (N_DEMES - 1)) for i in range(N_DEMES)]

theta  = np.linspace(0, 2 * np.pi, N_th)
z_side = np.array([0.0, T])
th_g, z_g = np.meshgrid(theta, z_side)

fig = plt.figure(figsize=(22, 6))
ax = fig.add_subplot(1, 1, 1, projection="3d")
span = (N_DEMES - 1) * SPACING
ax.set_box_aspect([span, 4, span * 0.22])

# Cylinders
for i in range(N_DEMES):
    cx = float(i) * SPACING

    # Side surface
    ax.plot_surface(
        cx + r3 * np.cos(th_g),
        r3 * np.sin(th_g),
        z_g,
        color=COLORS[i], alpha=0.85, linewidth=0, shade=True,
    )

    # Top cap
    r_cap = np.linspace(0, r3, N_r)
    th_c, r_c = np.meshgrid(theta, r_cap)
    z_cap = np.full_like(th_c, T)
    ax.plot_surface(
        cx + r_c * np.cos(th_c),
        r_c * np.sin(th_c),
        z_cap,
        color=COLORS[i], alpha=0.85, linewidth=0, shade=False,
    )

# Migration lines at evenly-spaced time slices
for z_lev in np.linspace(0, T, N_MIGRATION_LEVELS):
    for i in range(N_DEMES - 1):
        ax.plot(
            [i * SPACING + r3, (i + 1) * SPACING - r3], [0, 0], [z_lev, z_lev],
            color="#888888", lw=0.8, alpha=0.6,
        )

# Bidirectional arrows at the top layer
gap = SPACING - 2 * r3
L   = gap * 0.42
for i in range(N_DEMES - 1):
    ax.quiver(i * SPACING + r3,           0, T,  L, 0, 0, color="#444444", lw=0.8,
              arrow_length_ratio=0.4, normalize=False)
    ax.quiver((i + 1) * SPACING - r3,     0, T, -L, 0, 0, color="#444444", lw=0.8,
              arrow_length_ratio=0.4, normalize=False)

ax.set_xlabel("")
ax.set_zlabel("Time", fontsize=9, labelpad=4)
ax.set_yticks([])
ax.set_xticks(np.arange(0, N_DEMES, 10) * SPACING)
ax.set_xticklabels(np.arange(0, N_DEMES, 10))
ax.tick_params(labelsize=7)
ax.view_init(elev=22, azim=-65)
ax.dist = 5  # zoom in (default ~10)

out = "../../results/figures/sim_figs/stepping_stones_1d_schematic.pdf"
import os
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, bbox_inches="tight", dpi=300)
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
print(f"Saved to {out}")
plt.tight_layout()
plt.show()
