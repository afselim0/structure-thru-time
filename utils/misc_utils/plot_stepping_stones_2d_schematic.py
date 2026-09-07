#!/usr/bin/env python3
"""Schematic of the 5x5 stepping-stone model.

Panel 1: flat 2D grid with bidirectional migration arrows.
Panel 2: 3D perspective with each deme as a tube through time and
         migration connections shown at evenly-spaced time slices.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

GRID = 5
COLORS = [
    "#8000ff","#6c1fff","#5641fd","#4062fa","#2c7ef7",
    "#169bf2","#00b5eb","#14cae5","#2adddd","#40ecd4",
    "#54f6cb","#6afdc0","#80ffb4","#94fda8","#abf69b",
    "#c0eb8d","#d4dd80","#ebca70","#ffb360","#ff9b52",
    "#ff7e41","#ff5f30","#ff4121","#ff1f10","#ff0000",
]

fig = plt.figure(figsize=(9, 4))

# ── Panel 1: flat 2D grid ──────────────────────────────────────────────────
ax = fig.add_subplot(1, 2, 1)
ax.set_aspect("equal")
ax.axis("off")

r = 0.35
lw = 1.2

for i in range(GRID * GRID):
    row, col = divmod(i, GRID)
    for (dx, dy) in [(1, 0), (0, -1)]:
        nr, nc = row - dy, col + dx
        if 0 <= nr < GRID and 0 <= nc < GRID:
            x0, y0 = col + dx * r, -row + dy * r
            x1, y1 = col + dx * (1 - r), -row + dy * (1 - r)
            ax.annotate(
                "", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="<->", color="#aaaaaa",
                                lw=lw, mutation_scale=8),
                zorder=1,
            )

for i in range(GRID * GRID):
    row, col = divmod(i, GRID)
    ax.add_patch(mpatches.Circle(
        (col, -row), radius=r,
        facecolor=COLORS[i], edgecolor="white", linewidth=1.0, zorder=2,
    ))
    ax.text(col, -row, f"{i}", ha="center", va="center",
            fontsize=6, color="white", fontweight="bold", zorder=3)

ax.set_xlim(-r - 0.2, GRID - 1 + r + 0.2)
ax.set_ylim(-(GRID - 1) - r - 0.2, r + 0.2)

# ── Panel 2: 3D tubes through time ────────────────────────────────────────
ax3 = fig.add_subplot(1, 2, 2, projection="3d")
ax3.set_box_aspect([1, 1, 1.6])

T = 1.0          # normalized time axis
r3 = 0.28        # cylinder radius
N_th = 24        # angular resolution
N_z  = 2         # height resolution for cylinder side
N_r  = 8         # radial resolution for cap discs
N_migration_levels = 5

theta = np.linspace(0, 2 * np.pi, N_th)
z_side = np.array([0.0, T])
th_g, z_g = np.meshgrid(theta, z_side)

# Draw cylinders (side + top cap)
for i in range(GRID * GRID):
    row, col = divmod(i, GRID)
    cx, cy = float(col), float(-row)

    # Side surface
    ax3.plot_surface(
        cx + r3 * np.cos(th_g), cy + r3 * np.sin(th_g), z_g,
        color=COLORS[i], alpha=0.85, linewidth=0, shade=True,
    )

    # Top cap
    r_cap = np.linspace(0, r3, N_r)
    th_c, r_c = np.meshgrid(theta, r_cap)
    z_cap = np.full_like(th_c, T)
    ax3.plot_surface(
        cx + r_c * np.cos(th_c), cy + r_c * np.sin(th_c), z_cap,
        color=COLORS[i], alpha=0.85, linewidth=0, shade=False,
    )

# Migration connections at evenly-spaced time slices
for z_lev in np.linspace(0, T, N_migration_levels):
    for i in range(GRID * GRID):
        row, col = divmod(i, GRID)
        if col + 1 < GRID:   # right
            ax3.plot([col + r3, col + 1 - r3], [-row, -row], [z_lev, z_lev],
                     color="#888888", lw=0.8, alpha=0.6)
        if row + 1 < GRID:   # down
            ax3.plot([col, col], [-row - r3, -(row + 1) + r3], [z_lev, z_lev],
                     color="#888888", lw=0.8, alpha=0.6)

# Bidirectional arrows at the top layer
gap = 1 - 2 * r3          # distance between adjacent cylinder surfaces
L   = gap * 0.42           # each arrow covers ~42% of the gap
for i in range(GRID * GRID):
    row, col = divmod(i, GRID)
    if col + 1 < GRID:     # horizontal pair
        ax3.quiver(col + r3,       -row, T,  L,  0, 0, color="#444444", lw=0.8,
                   arrow_length_ratio=0.4, normalize=False)
        ax3.quiver(col + 1 - r3,   -row, T, -L,  0, 0, color="#444444", lw=0.8,
                   arrow_length_ratio=0.4, normalize=False)
    if row + 1 < GRID:     # vertical pair
        ax3.quiver(col, -row - r3,       T,  0, -L, 0, color="#444444", lw=0.8,
                   arrow_length_ratio=0.4, normalize=False)
        ax3.quiver(col, -(row + 1) + r3, T,  0,  L, 0, color="#444444", lw=0.8,
                   arrow_length_ratio=0.4, normalize=False)

ax3.set_xlabel("X", fontsize=8, labelpad=2)
ax3.set_ylabel("Y", fontsize=8, labelpad=2)
ax3.set_zlabel("Time", fontsize=8, labelpad=2)
ax3.set_xticks(range(GRID))
ax3.set_yticks([-(GRID - 1) + i for i in range(GRID)])
ax3.set_yticklabels(range(GRID))
ax3.tick_params(labelsize=6)
ax3.view_init(elev=22, azim=-55)

# ── Save ──────────────────────────────────────────────────────────────────
out = "../../results/figures/sim_figs/stepping_stones_2d_schematic.pdf"
fig.savefig(out, bbox_inches="tight", dpi=300)
fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=300)
print(f"Saved to {out}")
plt.tight_layout()
plt.show()
