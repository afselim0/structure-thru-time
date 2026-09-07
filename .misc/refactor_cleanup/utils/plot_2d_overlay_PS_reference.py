#!/usr/bin/env python3
"""Single 2-D overlay plot across all PS runs, with one PS as the reference.

Reference PS: larger, fully opaque points + bold labels.
Other PSs:    small, semi-transparent points, no labels.
Each PS gets a slightly different lightness tone and marker shape.
All are Procrustes-aligned to the reference PS.

Exports a single PNG.
"""
import argparse
import colorsys
import glob
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy.linalg import svd

sys.path.append("../src")
sys.path.append("../../src")
sys.path.append("src")

# Marker cycle for each PS (index → shape)
PS_MARKERS = ['o', 's', '^', 'D', 'v', 'p', 'P', 'h']


def _discover_embeddings(ps_data_dir):
    matches = sorted(
        m for m in glob.glob(os.path.join(ps_data_dir, "embeddings.*.parquet"))
        if "loss" not in os.path.basename(m)
    )
    if not matches:
        raise FileNotFoundError(f"No embeddings*.parquet in {ps_data_dir}")
    if len(matches) > 1:
        matches.sort(key=os.path.getmtime)
    return matches[-1]


def _hex_to_hls(hex_color):
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16)/255, int(h[2:4], 16)/255, int(h[4:6], 16)/255
    return colorsys.rgb_to_hls(r, g, b)


def _hls_to_hex(h, l, s):
    r, g, b = colorsys.hls_to_rgb(h, max(0.0, min(1.0, l)), s)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


def _ps_tone(base_hex, ps_pos, n_ps, l_spread=0.14):
    """Return a lightness-shifted variant of base_hex for the given PS position."""
    h, l, s = _hex_to_hls(base_hex)
    offsets  = np.linspace(-l_spread, l_spread, n_ps)
    return _hls_to_hex(h, l + offsets[ps_pos], s)


def main():
    parser = argparse.ArgumentParser(
        description="Single 2-D overlay of all PS runs with one reference PS highlighted"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--base_dir")
    group.add_argument("--inputs", nargs="+")

    parser.add_argument("--n_ps",         type=int,   default=8)
    parser.add_argument("--reference_ps", type=int,   required=True)
    parser.add_argument("--colors",       required=True)
    parser.add_argument("--labels",       required=True)
    parser.add_argument("--output_path",  required=True)
    parser.add_argument("--output_name",  default=None)
    parser.add_argument("--min_time",     type=float, default=1000)
    parser.add_argument("--max_time",     type=float, default=100000)
    parser.add_argument("--lims",         type=str,   default="None")
    parser.add_argument("--l_spread",     type=float, default=0.14,
                        help="Lightness range (±) across PS runs")
    args = parser.parse_args()

    if args.output_name is None:
        args.output_name = f"2d_overlay_PS{args.reference_ps}_reference.png"

    lims = None if args.lims == "None" else float(args.lims)

    if args.inputs:
        input_paths = args.inputs
        ps_indices  = list(range(len(input_paths)))
    else:
        ps_indices  = list(range(args.n_ps))
        input_paths = []
        for k in ps_indices:
            p = _discover_embeddings(os.path.join(args.base_dir, f"chimps_PS{k}", "data"))
            input_paths.append(p)
            print(f"PS{k}: {os.path.basename(p)}")

    n_ps      = len(ps_indices)
    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)
    base_color_map = colors_df.set_index("population")["color"].to_dict()

    dfs = []
    for path in input_paths:
        df = pd.read_parquet(path)
        df = df[(df["time"] >= args.min_time) & (df["time"] <= args.max_time)]
        dfs.append(df)

    id_col = "individual_id" if "individual_id" in dfs[0].columns else "sample_id"
    if id_col == "individual_id" and "individual_id" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df["individual_id"] = labels_df["sample_id"] // 2
    pop_map = labels_df.drop_duplicates(subset=id_col).set_index(id_col)["population"]

    most_recent = min(df["time"].min() for df in dfs)
    print(f"Most recent time: {most_recent:.0f} gen")

    slices = []
    for df in dfs:
        s = df[df["time"] == most_recent].copy()
        s["population"] = s[id_col].map(pop_map).fillna("unknown")
        slices.append(s)

    # Procrustes-align all to the reference PS
    ref_pos    = ps_indices.index(args.reference_ps) if args.reference_ps in ps_indices else 0
    ref_slice  = slices[ref_pos].set_index(id_col)
    common_ids = sorted(set.intersection(*[set(s[id_col].values) for s in slices]))
    ref_xy     = ref_slice.loc[common_ids, ["X1", "X2"]].values
    for i in range(len(slices)):
        if i == ref_pos:
            continue
        s_idx    = slices[i].set_index(id_col)
        src_xy   = s_idx.loc[common_ids, ["X1", "X2"]].values
        src_mean = src_xy.mean(axis=0)
        ref_mean = ref_xy.mean(axis=0)
        U, _, Vt = svd((src_xy - src_mean).T @ (ref_xy - ref_mean))
        R        = U @ Vt
        all_xy   = slices[i][["X1", "X2"]].values
        rotated  = (all_xy - src_mean) @ R + ref_mean
        slices[i] = slices[i].copy()
        slices[i]["X1"] = rotated[:, 0]
        slices[i]["X2"] = rotated[:, 1]

    # Per-PS tone and marker
    markers  = (PS_MARKERS * ((n_ps // len(PS_MARKERS)) + 1))[:n_ps]
    # Build colour maps per PS: {ps_position: {population: toned_hex}}
    ps_color_maps = []
    for pos in range(n_ps):
        cm = {pop: _ps_tone(base_color_map.get(pop, "#808080"), pos, n_ps, args.l_spread)
              for pop in base_color_map}
        ps_color_maps.append(cm)

    # Shared axis limits
    all_x = np.concatenate([s["X1"].dropna().values for s in slices])
    all_y = np.concatenate([s["X2"].dropna().values for s in slices])
    if lims is None:
        cx   = (all_x.max() + all_x.min()) / 2
        cy   = (all_y.max() + all_y.min()) / 2
        half = max(all_x.max() - all_x.min(), all_y.max() - all_y.min()) / 2 * 1.06
        xlim = (cx - half, cx + half)
        ylim = (cy - half, cy + half)
    else:
        xlim = (-lims, lims)
        ylim = (-lims, lims)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Non-reference PSs first (behind), each with its own tone + shape
    for pos, (s, k) in enumerate(zip(slices, ps_indices)):
        if k == args.reference_ps:
            continue
        colors = s["population"].map(ps_color_maps[pos]).fillna("#808080")
        ax.scatter(s["X1"], s["X2"],
                   c=colors, s=5,
                   marker=markers[pos],
                   alpha=0.45, linewidths=0, zorder=3)

    # Reference PS on top — larger, opaque, with labels
    ref_s    = slices[ref_pos]
    ref_colors = ref_s["population"].map(ps_color_maps[ref_pos]).fillna("#808080")
    ax.scatter(ref_s["X1"], ref_s["X2"],
               c=ref_colors, s=22,
               marker=markers[ref_pos],
               alpha=1.0, linewidths=0.4, edgecolors="white", zorder=5)
    label_offset = 0.015 * (xlim[1] - xlim[0])
    for _, sample in ref_s.iterrows():
        pop   = sample["population"]
        color = ps_color_maps[ref_pos].get(pop, "#808080")
        ax.text(sample["X1"], sample["X2"] - label_offset,
                f"{pop[:2]}-{sample[id_col]}",
                fontsize=5, ha="center", va="top",
                color=color, fontweight="bold")

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal")
    ax.set_xlabel("X1", fontsize=8)
    ax.set_ylabel("X2", fontsize=8)
    ax.set_title(
        f"PS{args.reference_ps} reference  |  all PS aligned  |  t = {most_recent:.0f} gen",
        fontsize=9, fontweight="bold", pad=4,
    )
    ax.tick_params(labelsize=6)

    # Legend: one entry per PS (shape + representative colour)
    active_pops = [p for p in labels_df["population"].unique() if p in base_color_map]
    # Representative colour per PS = tone of the first active population
    rep_pop = active_pops[0] if active_pops else list(base_color_map.keys())[0]
    ps_legend = [
        Line2D([0], [0], marker=markers[pos], linestyle="none",
               color=ps_color_maps[pos].get(rep_pop, "#808080"),
               markersize=6 if k == args.reference_ps else 4,
               alpha=1.0 if k == args.reference_ps else 0.7,
               label=f"PS{k}" + (" [REF]" if k == args.reference_ps else ""))
        for pos, k in enumerate(ps_indices)
    ]
    pop_legend = [
        mpatches.Patch(facecolor=base_color_map[p], label=p) for p in active_pops
    ]
    ax.legend(handles=ps_legend + pop_legend, fontsize=6, loc="best",
              frameon=True, framealpha=0.9, ncol=2)

    os.makedirs(args.output_path, exist_ok=True)
    out = os.path.join(args.output_path, args.output_name)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
