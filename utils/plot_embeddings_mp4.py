import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from pathlib import Path

try:
    import imageio_ffmpeg  # noqa: F401
except ImportError:
    pass


def make_mp4(
    parquet_path: str,
    out_mp4: str = "",
    fps: int = 20,
    bounce: bool = True,
    point_size: float = 10.0,
    alpha: float = 0.8,
    dpi: int = 120,
    min_time: float = 0,
    max_time: float = 3000,
    colors_path: str | None = None,
    labels_path: str | None = None,
):
    parquet_path = Path(parquet_path)

    if out_mp4:
        out_mp4 = Path(out_mp4)
    else:
        out_mp4 = parquet_path.with_suffix(".mp4")

    print("make_mp4: reading parquet", parquet_path, flush=True)

    df = pd.read_parquet(parquet_path)

    time_col = "time"
    x_col = "X1"
    y_col = "X2"
    id_col = "sample_id"

    # ---- Time filter early
    df = df[(df[time_col] >= min_time) & (df[time_col] <= max_time)].copy()
    if df.empty:
        raise ValueError("No rows left after time filtering")

    # ---- Unique times (sorted)
    times = np.sort(df[time_col].unique())
    T = len(times)

    # ---- Stable sample order: union of all samples ever seen
    sample_order = np.array(sorted(df[id_col].unique()))
    N = len(sample_order)

    # ---- Map times and samples to integer indices
    t_codes = pd.Categorical(
        df[time_col], categories=times, ordered=True
    ).codes
    s_codes = pd.Index(sample_order).get_indexer(df[id_col].to_numpy())

    # ---- Pack coordinates: (T, N, 2), NaN = not yet present
    xy = np.full((T, N, 2), np.nan, dtype=np.float32)
    xy[t_codes, s_codes, 0] = df[x_col].to_numpy(dtype=np.float32)
    xy[t_codes, s_codes, 1] = df[y_col].to_numpy(dtype=np.float32)

    # ---- Colors (constant over time)
    sample_colors = None
    if colors_path and labels_path:
        labels_df = pd.read_csv(labels_path)
        colors_df = pd.read_csv(colors_path)

        if labels_df[id_col].duplicated().any():
            raise ValueError("Duplicate sample_id in labels file")
        if colors_df["population"].duplicated().any():
            raise ValueError("Duplicate population in colors file")

        merged = labels_df.merge(colors_df, on="population", how="left")
        if merged["color"].isna().any():
            missing = merged.loc[merged["color"].isna(), "population"].unique()
            raise ValueError(f"Missing colors for populations: {missing}")

        color_map = dict(zip(merged[id_col], merged["color"]))
        sample_colors = np.array(
            [color_map.get(sid, "#808080") for sid in sample_order],
            dtype=object,
        )
        
    sample_labels = None
    if labels_path is not None:
        labels_df = pd.read_csv(labels_path)

        # Map sample_id -> 3-letter population code
        pop_map = dict(
            zip(labels_df[id_col], labels_df["population"].str[:3].str.upper())
        )

        sample_labels = np.array(
            [pop_map.get(sid, "") for sid in sample_order],
            dtype=object,
        )

    # ---- Axis limits with padding
    xmin, xmax = np.nanmin(xy[:, :, 0]), np.nanmax(xy[:, :, 0])
    ymin, ymax = np.nanmin(xy[:, :, 1]), np.nanmax(xy[:, :, 1])
    dx = xmax - xmin
    dy = ymax - ymin
    pad = 0.03

    # ---- Bounce order
    idxs = list(range(T))
    if bounce and T > 1:
        idxs = idxs + idxs[-2:0:-1]

    # ---- Figure
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    ax.set_xlim(xmin - pad * dx, xmax + pad * dx)
    ax.set_ylim(ymin - pad * dy, ymax + pad * dy)
    ax.set_aspect("equal", adjustable="box")

    title = ax.set_title("")
   
    time_text = ax.text(
        0.02, 0.98, "",
        transform=ax.transAxes,
        ha="left", va="top",
    )
    time_text.set_animated(True)

    scat = ax.scatter(
        xy[0, :, 0],
        xy[0, :, 1],
        s=point_size,
        alpha=alpha,
        rasterized=True,
        animated=True,
    )
    title.set_animated(True)

    if sample_colors is not None:
        scat.set_color(sample_colors)
        
        texts = []
    if sample_labels is not None:
        for i in range(N):
            txt = ax.text(
                np.nan, np.nan, sample_labels[i],
                ha="center", va="bottom",
                color=sample_colors[i],
                fontsize=7,
            )
            txt.set_animated(True)
            txt.set_visible(False)  # hidden until sample appears
            texts.append(txt)

    fig.tight_layout()

    # ---- Draw once, cache background for blitting
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(ax.bbox)

    print("Writing to:", out_mp4)

    with imageio.get_writer(
        out_mp4,
        format="FFMPEG",
        fps=fps,
        codec="libx264",
        quality=8,
        macro_block_size=None,
    ) as writer:
        for ti in idxs:
            fig.canvas.restore_region(background)

            coords = xy[ti]
            scat.set_offsets(coords)
            ax.draw_artist(scat)
            
            # Update labels
            if sample_labels is not None:
                for i, txt in enumerate(texts):
                    x, y = coords[i]
                    if np.isnan(x) or np.isnan(y):
                        txt.set_visible(False)
                    else:
                        txt.set_position((x, y))
                        txt.set_visible(True)
                        ax.draw_artist(txt)
            
            time_text.set_text(f"t = {times[ti]}")
            ax.draw_artist(time_text)

            fig.canvas.blit(ax.bbox)

            buf = np.asarray(fig.canvas.buffer_rgba())
            writer.append_data(buf[:, :, :3])

    plt.close(fig)
    print(f"Wrote {out_mp4} with {len(idxs)} frames.")

    output_basename = out_mp4.stem
    # ---- Tiny HTML wrapper
    html = f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0">
      <video src="{out_mp4.name}" autoplay loop muted playsinline
             style="width:100vw;height:100vh;object-fit:contain;"></video>
    </body>
    </html>
    """
    (out_mp4.parent / f"{output_basename}.html").write_text(html)



if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("parquet", help="Path to parquet file")
    p.add_argument("--out", default="")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--no-bounce", action="store_true")
    p.add_argument("--size", type=float, default=10.0)
    p.add_argument("--alpha", type=float, default=0.8)
    p.add_argument("--min-time", type=float, default=0)
    p.add_argument("--max-time", type=float, default=3000)
    p.add_argument("--colors", required=True, help="CSV with columns: population,color")
    p.add_argument("--labels", required=True, help="CSV with columns: sample_id,population")

    args = p.parse_args()

    make_mp4(
        args.parquet,
        out_mp4=args.out,
        fps=args.fps,
        bounce=not args.no_bounce,
        point_size=args.size,
        alpha=args.alpha,
        min_time=args.min_time,
        max_time=args.max_time,
        colors_path=args.colors,
        labels_path=args.labels,
    )

