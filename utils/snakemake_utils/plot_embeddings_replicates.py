#!/usr/bin/env python3
import argparse
import io
import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.io as pio
import matplotlib.patches as mpatches
from PIL import Image, ImageDraw
from scipy.linalg import svd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

from PlotEmbeddings import PlotEmbeddings


def _render_to_pil(plotter, eye, width=500, height=500, scale=2):
    """Render a PlotEmbeddings figure to a PIL Image without writing to disk."""
    plotter.fig.update_layout(
        scene=dict(
            camera=dict(eye=eye, up=dict(x=0, y=0, z=1)),
            zaxis=dict(title=""),
        ),
        showlegend=False,
        margin=dict(l=0, r=0, t=0, b=0),
    )
    try:
        img_bytes = pio.to_image(
            plotter.fig, format="png", width=width, height=height, scale=scale
        )
    except Exception as e:  # kaleido needs a Chrome binary; compute nodes may lack one
        print(f"WARNING: static export failed ({type(e).__name__}); drawing a placeholder panel. "
              "One-time fix per user: SSL_CERT_FILE=$(python -c \"import certifi;print(certifi.where())\") "
              "python -c \"from kaleido import get_chrome_sync; print(get_chrome_sync())\"",
              file=sys.stderr)
        img = Image.new("RGB", (width * scale, height * scale), (235, 235, 235))
        ImageDraw.Draw(img).text((10, 10), "static export unavailable\n(no Chrome for kaleido)", fill=(80, 80, 80))
        return img
    return Image.open(io.BytesIO(img_bytes))


def _autocrop(img: Image.Image, pad: int = 6) -> Image.Image:
    """Trim near-white margins from a PIL Image on all four sides."""
    arr = np.array(img.convert("RGB"))
    # Plotly background is light grey/white — treat pixels >= 240 on all channels as bg
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
        description="Multi-panel plot of replicate embeddings from the same camera angle"
    )
    parser.add_argument("--inputs", nargs="+", required=True,
                        help="Replicate embeddings parquet files, in seed order")
    parser.add_argument("--seeds", nargs="+", type=int, required=True,
                        help="Seed values corresponding to --inputs")
    parser.add_argument("--colors", required=True, help="Population colors CSV")
    parser.add_argument("--labels", required=True, help="Population labels CSV")
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--output_name", required=True)
    parser.add_argument("--min_time", type=float, default=50)
    parser.add_argument("--max_time", type=float, default=2000)
    parser.add_argument("--lims", type=str, default="None",
                        help="Shared xy axis half-range. 'None' = auto from data.")
    parser.add_argument("--time_scale", type=str, default="log")
    parser.add_argument("--param_label", type=str, default=None,
                        help="Optional parameter string shown as figure subtitle")
    parser.add_argument("--eye_x", type=float, default=1.8)
    parser.add_argument("--eye_y", type=float, default=1.8)
    parser.add_argument("--eye_z", type=float, default=0.3)
    parser.add_argument("--panel_px", type=int, default=500,
                        help="Width and height of each panel in pixels before scaling")
    parser.add_argument("--panel_scale", type=int, default=2)
    args = parser.parse_args()

    assert len(args.inputs) == len(args.seeds), \
        "--inputs and --seeds must have the same number of elements"

    lims = None if args.lims == "None" else float(args.lims)
    eye = dict(x=args.eye_x, y=args.eye_y, z=args.eye_z)

    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)

    # Load and time-filter all replicates
    print("Loading replicate embeddings...")
    dfs = []
    for path in args.inputs:
        df = pd.read_parquet(path)
        df = df[df["time"] >= args.min_time]
        dfs.append(df)

    # Subset labels/colors to samples present in the data
    id_label = "individual_id" if "individual_id" in dfs[0].columns else "sample_id"
    if id_label == "individual_id" and "individual_id" not in labels_df.columns:
        labels_df = labels_df.copy()
        labels_df["individual_id"] = labels_df["sample_id"] // 2
    active_samples = set(dfs[0][id_label].unique())
    labels_df = labels_df[labels_df[id_label].isin(active_samples)]
    populations = labels_df["population"].unique()
    colors_df = colors_df[colors_df["population"].isin(populations)]

    # Procrustes alignment: rotate+reflect each replicate's X1/X2 to match seed 0,
    # using the most recent time slice as the reference, applied to all time points.
    most_recent = min(df["time"].min() for df in dfs)
    ref_slice = dfs[0][dfs[0]["time"] == most_recent].set_index(id_label)
    common_ids = sorted(
        set.intersection(*[set(df[df["time"] == most_recent][id_label].values) for df in dfs])
    )
    ref_xy = ref_slice.loc[common_ids, ["X1", "X2"]].values
    for i in range(1, len(dfs)):
        slice_i = dfs[i][dfs[i]["time"] == most_recent].set_index(id_label)
        src_xy  = slice_i.loc[common_ids, ["X1", "X2"]].values
        src_mean = src_xy.mean(axis=0)
        ref_mean = ref_xy.mean(axis=0)
        U, _, Vt = svd((src_xy - src_mean).T @ (ref_xy - ref_mean))
        R = U @ Vt  # orthogonal Procrustes: allow rotation and reflection
        all_xy  = dfs[i][["X1", "X2"]].values
        rotated = (all_xy - src_mean) @ R + ref_mean
        dfs[i] = dfs[i].copy()
        dfs[i]["X1"] = rotated[:, 0]
        dfs[i]["X2"] = rotated[:, 1]

    # Shared xy axis limits across all replicates
    if lims is None:
        all_x = np.concatenate([df["X1"].dropna().values for df in dfs])
        all_y = np.concatenate([df["X2"].dropna().values for df in dfs])
        common_min = min(all_x.min(), all_y.min())
        common_max = max(all_x.max(), all_y.max())
        pad = 0.05 * (common_max - common_min)
        lims = max(abs(common_min - pad), abs(common_max + pad))
        print(f"Auto xy limits: ±{lims:.1f}")

    # Grid layout: up to 4 columns
    n = len(dfs)
    ncols = min(n, 4)
    nrows = int(np.ceil(n / ncols))

    # Render each replicate to a PIL image
    images = []
    for df, seed in zip(dfs, args.seeds):
        print(f"Rendering seed {seed}...")
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

    # Derive panel height from the median cropped aspect ratio so axes fit tightly
    # Cap at 1.4 — uncropped 3D plots can have very tall aspect ratios that blow up figure height
    median_ratio = min(float(np.median([img.height / img.width for img in images])), 1.4)
    panel_w = 3.5
    panel_h = panel_w * median_ratio

    # Build legend handles from population colours
    color_map = colors_df.set_index("population")["color"].to_dict()
    legend_handles = [
        mpatches.Patch(facecolor=color_map[pop], label=pop)
        for pop in populations if pop in color_map
    ]

    # Reserve a bottom strip sized to the actual legend: ~0.22" per row + padding
    caption_in   = 0.32 if args.param_label else 0.0
    legend_ncols = min(len(legend_handles), 8)
    legend_nrows = int(np.ceil(len(legend_handles) / legend_ncols)) if legend_handles else 1
    legend_in    = legend_nrows * 0.22 + 0.15
    fig_h = panel_h * nrows + legend_in + caption_in

    fig, axes = plt.subplots(nrows, ncols, figsize=(panel_w * ncols, fig_h))
    axes = np.array(axes).reshape(nrows, ncols)
    fig.subplots_adjust(hspace=0.06, wspace=0.02)

    for idx, (img, seed) in enumerate(zip(images, args.seeds)):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        ax.imshow(img, aspect="auto")
        ax.set_title(f"seed {seed}", fontsize=9, fontweight="bold", pad=2)
        ax.axis("off")

    # Hide unused panels
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].axis("off")

    # Caption at very bottom, legend clearly above it
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
    print(f"Saved panel figure to {out_path}")


if __name__ == "__main__":
    main()
