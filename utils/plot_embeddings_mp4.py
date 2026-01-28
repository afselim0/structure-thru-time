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
    pass  # rely on system ffmpeg if present

def make_mp4(
    parquet_path: str,
    out_mp4: str = '',
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
    
    if out_mp4:
        out_mp4 = Path(out_mp4)
    else:
        out_mp4 = f"{parquet_path.removesuffix('.parquet')}.mp4"
        out_mp4 = Path(out_mp4)
        
    parquet_path = Path(parquet_path)
   

    df = pd.read_parquet(parquet_path)

    time_col = "time"
    x_col = "X1"
    y_col = "X2"
    id_col = "sample_id"

    # Time filter early
    df = df[(df[time_col] >= min_time) & (df[time_col] <= max_time)].copy()
    if df.empty:
        raise ValueError(f"No rows left after filtering {min_time} <= {time_col} <= {max_time}.")

    sample_colors = None
    if colors_path and labels_path:
        labels_df = pd.read_csv(labels_path)
        colors_df = pd.read_csv(colors_path)

        active_samples = df[id_col].unique()
        labels_df = labels_df[labels_df[id_col].isin(active_samples)].copy()

        # merge labels->colors on population
        merged = labels_df.merge(colors_df, on="population", how="left")

        if merged["color"].isna().any():
            missing_pops = merged.loc[merged["color"].isna(), "population"].unique()
            raise ValueError(f"Missing colors for populations: {missing_pops}")

        # create mapping: sample_id -> color
        color_map = dict(zip(merged[id_col], merged["color"]))

        # We'll use a consistent sample order for ALL frames
        sample_order = np.array(sorted(df[id_col].unique()))
        sample_colors = np.array([color_map.get(sid, "#808080") for sid in sample_order], dtype=object)
    else:
        # still enforce consistent order even if no colors
        sample_order = np.array(sorted(df[id_col].unique()))

    # Sort once so groupby preserves sample_order within each time (we'll still reindex to be safe)
    df.sort_values([time_col, id_col], inplace=True)

    # Unique times
    times = np.sort(df[time_col].unique())
    T = len(times)

    # Precompute xy arrays, ordered by sample_order, per time
    # (This keeps colors aligned to points.)
    xy_by_time = {}
    for t, g in df.groupby(time_col, sort=True):
        # reindex to sample_order so rows align to sample_colors
        g2 = g.set_index(id_col).reindex(sample_order)
        xy_by_time[t] = g2[[x_col, y_col]].to_numpy()

    # Axis limits
    xmin, xmax = df[x_col].min(), df[x_col].max()
    ymin, ymax = df[y_col].min(), df[y_col].max()

    # Bounce order
    idxs = list(range(T))
    if bounce and T > 1:
        idxs = idxs + idxs[-2:0:-1]

    # Figure
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")
    title = ax.set_title("")

    # Initialize scatter with correct number of points
    init_xy = xy_by_time[times[0]]
    scat = ax.scatter(
        init_xy[:, 0],
        init_xy[:, 1],
        s=point_size,
        alpha=alpha,
        rasterized=True,
    )

    if sample_colors is not None:
        scat.set_color(sample_colors)

    fig.tight_layout()
    
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
            t = times[ti]
            scat.set_offsets(xy_by_time[t])
            title.set_text(f"t = {t}")

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            frame = buf[:, :, :3].copy()

            writer.append_data(frame)

    plt.close(fig)
    print(f"Wrote {out_mp4} with {len(idxs)} frames (times {times.min()}..{times.max()}).")

    output_basename = out_mp4.name.removesuffix(".html")
    html = f"""
    <!DOCTYPE html>
    <html>
    <body style="margin:0">
        <video src="{out_mp4.name}" autoplay loop muted playsinline></video>
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

