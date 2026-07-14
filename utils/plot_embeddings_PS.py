#!/usr/bin/env python3
"""Multi-panel 3-D trajectory plot across SINGER posterior sample runs (PS0–PS7).
Procrustes-aligns X1/X2 of each PS to PS0 before rendering.
"""
import argparse
import glob
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
import matplotlib.patches as mpatches
from PIL import Image
from scipy.linalg import svd

sys.path.append("../src")
sys.path.append("../../src")
sys.path.append("src")

from PlotEmbeddings import PlotEmbeddings


def _discover_embeddings(ps_data_dir):
    matches = sorted(
        m for m in glob.glob(os.path.join(ps_data_dir, "embeddings.*.parquet"))
        if "loss" not in os.path.basename(m)
    )
    if not matches:
        raise FileNotFoundError(f"No embeddings*.parquet in {ps_data_dir}")
    if len(matches) > 1:
        matches.sort(key=os.path.getmtime)
        print(f"  [warn] multiple embeddings in {ps_data_dir}; using newest")
    return matches[-1]


def _render_to_pil(plotter, eye, width=500, height=500, scale=2):
    plotter.fig.update_layout(
        scene=dict(
            camera=dict(eye=eye, up=dict(x=0, y=0, z=1)),
            zaxis=dict(title=""),
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    img_bytes = pio.to_image(
        plotter.fig, format="png", width=width, height=height, scale=scale
    )
    return Image.open(io.BytesIO(img_bytes))


def _autocrop(img: Image.Image, pad: int = 6) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    non_bg = (arr < 240).any(axis=2)
    rows = np.where(non_bg.any(axis=1))[0]
    cols = np.where(non_bg.any(axis=0))[0]
    if rows.size == 0:
        return img
    r0 = max(0,          rows[0]  - pad)
    r1 = min(img.height, rows[-1] + pad + 1)
    c0 = max(0,          cols[0]  - pad)
    c1 = min(img.width,  cols[-1] + pad + 1)
    return img.crop((c0, r0, c1, r1))


def main():
    parser = argparse.ArgumentParser(
        description="Multi-panel 3-D embeddings plot across PS runs"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--inputs", nargs="+",
                       help="Embeddings parquet files, in PS order")
    group.add_argument("--base_dir",
                       help="Directory containing PS0/, PS1/, … subdirs")

    parser.add_argument("--n_ps", type=int, default=8)
    parser.add_argument("--colors",  required=True)
    parser.add_argument("--labels",  required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--min_time", type=float, default=50)
    parser.add_argument("--max_time", type=float, default=100000)
    parser.add_argument("--lims", type=str, default="None")
    parser.add_argument("--time_scale", type=str, default="log")
    parser.add_argument("--param_label", type=str, default=None)
    parser.add_argument("--eye_x", type=float, default=1.8)
    parser.add_argument("--eye_y", type=float, default=1.8)
    parser.add_argument("--eye_z", type=float, default=0.3)
    parser.add_argument("--panel_px", type=int, default=500)
    parser.add_argument("--panel_scale", type=int, default=2)
    args = parser.parse_args()

    lims = None if args.lims == "None" else float(args.lims)
    eye  = dict(x=args.eye_x, y=args.eye_y, z=args.eye_z)

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

    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)

    print("Loading PS embeddings...")
    dfs = []
    for path in input_paths:
        df = pd.read_parquet(path)
        df = df[df["time"] >= args.min_time]
        dfs.append(df)

    id_label = "individual_id" if "individual_id" in dfs[0].columns else "sample_id"
    if id_label == "individual_id" and "individual_id" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df["individual_id"] = labels_df["sample_id"] // 2
    active_samples = set(dfs[0][id_label].unique())
    labels_df  = labels_df[labels_df[id_label].isin(active_samples)]
    populations = labels_df["population"].unique()
    colors_df  = colors_df[colors_df["population"].isin(populations)]

    # Procrustes-rotate PS1–PS7 onto PS0 (rotation + reflection, no scaling)
    most_recent = min(df["time"].min() for df in dfs)
    ref_slice   = dfs[0][dfs[0]["time"] == most_recent].set_index(id_label)
    common_ids  = sorted(
        set.intersection(*[set(df[df["time"] == most_recent][id_label].values) for df in dfs])
    )
    ref_xy = ref_slice.loc[common_ids, ["X1", "X2"]].values
    for i in range(1, len(dfs)):
        slice_i  = dfs[i][dfs[i]["time"] == most_recent].set_index(id_label)
        src_xy   = slice_i.loc[common_ids, ["X1", "X2"]].values
        src_mean = src_xy.mean(axis=0)
        ref_mean = ref_xy.mean(axis=0)
        U, _, Vt = svd((src_xy - src_mean).T @ (ref_xy - ref_mean))
        R        = U @ Vt
        all_xy   = dfs[i][["X1", "X2"]].values
        rotated  = (all_xy - src_mean) @ R + ref_mean
        dfs[i]   = dfs[i].copy()
        dfs[i]["X1"] = rotated[:, 0]
        dfs[i]["X2"] = rotated[:, 1]

    if lims is None:
        all_x = np.concatenate([df["X1"].dropna().values for df in dfs])
        all_y = np.concatenate([df["X2"].dropna().values for df in dfs])
        cmn_min = min(all_x.min(), all_y.min())
        cmn_max = max(all_x.max(), all_y.max())
        pad  = 0.05 * (cmn_max - cmn_min)
        lims = max(abs(cmn_min - pad), abs(cmn_max + pad))
        print(f"Auto xy limits: ±{lims:.1f}")

    n     = len(ps_indices)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    images = []
    for df, k in zip(dfs, ps_indices):
        print(f"Rendering PS{k}...")
        plotter = PlotEmbeddings(
            embeddings_df=df,
            labels_df=labels_df,
            colors_df=colors_df,
            sample_id_col=id_label,
        )
        plotter.init_plot(
            time_scale=args.time_scale,
            xy_lims=lims,
            min_time=args.min_time,
            max_time=args.max_time,
        )
        plotter.add_sample_trajectories()
        img = _render_to_pil(plotter, eye, width=args.panel_px, height=args.panel_px,
                             scale=args.panel_scale)
        images.append(_autocrop(img))

    median_ratio = min(float(np.median([img.height / img.width for img in images])), 1.4)
    panel_w = 3.5
    panel_h = panel_w * median_ratio

    color_map = colors_df.set_index("population")["color"].to_dict()
    legend_handles = [
        mpatches.Patch(facecolor=color_map[pop], label=pop)
        for pop in populations if pop in color_map
    ]

    caption_in   = 0.32 if args.param_label else 0.0
    legend_ncols = min(len(legend_handles), 8)
    legend_nrows = int(np.ceil(len(legend_handles) / legend_ncols)) if legend_handles else 1
    legend_in    = legend_nrows * 0.22 + 0.15
    fig_h = panel_h * nrows + legend_in + caption_in

    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, fig_h))
    axes = np.array(axes).reshape(nrows, ncols)
    fig.subplots_adjust(hspace=0.06, wspace=0.02)

    for idx, (img, k) in enumerate(zip(images, ps_indices)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        ax.imshow(img, aspect="auto")
        ax.set_title(f"PS{k}", fontsize=9, fontweight="bold", pad=2)
        ax.axis("off")

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    caption_frac = caption_in / fig_h
    legend_frac  = (legend_in + caption_in) / fig_h

    if args.param_label:
        fig.text(0.5, 0.005, args.param_label, ha="center", va="bottom",
                 fontsize=8, color="#555555")

    fig.legend(handles=legend_handles, loc="lower center",
               ncol=legend_ncols,
               fontsize=9, frameon=True, framealpha=0.9,
               bbox_to_anchor=(0.5, caption_frac + 0.01))

    plt.tight_layout(rect=[0, legend_frac, 1, 1], pad=0.3)
    os.makedirs(args.output_path, exist_ok=True)
    out_path = os.path.join(args.output_path, args.output_name)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"Saved PS embeddings panel to {out_path}")


if __name__ == "__main__":
    main()
