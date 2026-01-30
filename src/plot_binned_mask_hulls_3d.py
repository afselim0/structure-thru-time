#!/usr/bin/env python3
"""
plot_binned_mask_hulls_3d.py

Hybrid hull visualization:
  - Older times: binned-mask hulls from support_volume masks
  - Recent times (where u-slices are sparse / holey): exact per-time convex hulls from raw points

Key files in debug_root/<pop>/:
  - slice_XXX_mask.npy
  - _us.npy
  - _counts.npy (recommended)
  - _meta_support_volume.txt
"""

import argparse
import os
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go

SLICE_RE = re.compile(r"slice_(\d{3})_mask\.npy$")


# ------------------------- small utilities ---------------------------------

def parse_meta_txt(path: str) -> Dict[str, str]:
    meta: Dict[str, str] = {}
    if not os.path.exists(path):
        return meta
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or "=" not in line:
            continue
        k, v = line.split("=", 1)
        meta[k.strip()] = v.strip()
    return meta


def get_float(meta: Dict[str, str], key: str, default: Optional[float] = None) -> Optional[float]:
    if key not in meta:
        return default
    try:
        return float(meta[key])
    except Exception:
        return default


def get_int(meta: Dict[str, str], key: str, default: Optional[int] = None) -> Optional[int]:
    if key not in meta:
        return default
    try:
        return int(float(meta[key]))
    except Exception:
        return default


def boundary_pixels_4n(binary_mask: np.ndarray) -> np.ndarray:
    m = (binary_mask > 0).astype(np.uint8)
    if m.sum() == 0:
        return np.zeros((0, 2), dtype=float)

    p = np.pad(m, ((1, 1), (1, 1)), mode="constant", constant_values=0)
    c = p[1:-1, 1:-1]
    up = p[:-2, 1:-1]
    dn = p[2:, 1:-1]
    lf = p[1:-1, :-2]
    rt = p[1:-1, 2:]
    b = (c == 1) & ((up == 0) | (dn == 0) | (lf == 0) | (rt == 0))
    yy, xx = np.where(b)
    return np.stack([xx.astype(float), yy.astype(float)], axis=1)


def convex_hull_2d(points: np.ndarray) -> np.ndarray:
    if points.shape[0] < 3:
        return points

    pts = np.unique(points, axis=0)
    if pts.shape[0] < 3:
        return pts

    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in pts[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return np.array(lower[:-1] + upper[:-1], dtype=float)


def trim_central(points: np.ndarray, keep_frac: float = 0.95) -> np.ndarray:
    if points.shape[0] < 3:
        return points
    keep_frac = float(keep_frac)
    if keep_frac >= 1.0:
        return points
    c = points.mean(axis=0)
    d = np.linalg.norm(points - c, axis=1)
    thr = np.quantile(d, keep_frac)
    return points[d <= thr]


def list_pop_dirs(debug_root: str) -> List[str]:
    root = Path(debug_root)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])


def load_masks_us_counts(pop_dir: str) -> Tuple[np.ndarray, np.ndarray, Dict[int, str]]:
    pop_path = Path(pop_dir)
    us_path = pop_path / "_us.npy"
    if not us_path.exists():
        raise FileNotFoundError(f"Missing {us_path}")
    us = np.load(us_path)

    counts_path = pop_path / "_counts.npy"
    counts = np.array([], dtype=int)
    if counts_path.exists():
        try:
            counts = np.load(counts_path).astype(int)
        except Exception:
            counts = np.array([], dtype=int)

    paths: Dict[int, str] = {}
    for f in pop_path.glob("slice_*_mask.npy"):
        m = SLICE_RE.search(str(f))
        if not m:
            continue
        idx = int(m.group(1))
        paths[idx] = str(f)

    if not paths:
        raise FileNotFoundError(f"No slice_XXX_mask.npy found in {pop_dir}")

    return us, counts, paths


def edges_from_centers(us: np.ndarray) -> np.ndarray:
    us = np.asarray(us, dtype=float)
    if us.ndim != 1 or us.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    if us.size == 1:
        du = 1.0
        return np.array([us[0] - 0.5 * du, us[0] + 0.5 * du], dtype=float)

    mid = 0.5 * (us[:-1] + us[1:])
    first = us[0] - (mid[0] - us[0])
    last = us[-1] + (us[-1] - mid[-1])
    return np.concatenate([[first], mid, [last]]).astype(float)


def infer_counts_from_points(
    df: pd.DataFrame,
    pop: str,
    popcol: str,
    tcol: str,
    u_edges: np.ndarray,
    time_eps: float = 1.0,
) -> np.ndarray:
    sub = df[df[popcol] == pop]
    if sub.empty:
        return np.zeros(len(u_edges) - 1, dtype=int)
    t = sub[tcol].to_numpy(dtype=float)
    t_pos = np.maximum(t, time_eps)
    u = np.log10(t_pos)
    u_bin = np.searchsorted(u_edges, u, side="right") - 1
    u_bin = np.clip(u_bin, 0, len(u_edges) - 2)
    return np.bincount(u_bin, minlength=len(u_edges) - 1).astype(int)


def recent_region_end_from_window(
    counts: np.ndarray,
    *,
    min_points_per_slice: int,
    window: int,
    valid_frac: float,
) -> int:
    """
    Return ui0 = first index where a window of size `window` has >= valid_frac "valid" slices.
    Recent region is [0, ui0) (more recent).
    If never becomes dense, returns len(counts).
    """
    counts = np.asarray(counts, dtype=int)
    U = int(counts.size)
    if U == 0:
        return 0

    valid = (counts >= int(min_points_per_slice)).astype(np.int8)
    W = int(max(1, window))
    vf = float(valid_frac)

    if W == 1:
        # degenerate: first valid slice defines the transition
        idxs = np.where(valid == 1)[0]
        return int(idxs[0]) if idxs.size else U

    # sliding window sum
    # sums[i] = number of valid slices in valid[i:i+W]
    sums = np.convolve(valid, np.ones(W, dtype=int), mode="valid")
    thresh = vf * W

    hits = np.where(sums >= thresh)[0]
    if hits.size == 0:
        return U
    return int(hits[0])


def reconstruct_xy_edges_for_pop(
    df: pd.DataFrame,
    pop: str,
    grid_res_xy: int,
    xy_pad_frac: float,
    xcol: str,
    ycol: str,
    popcol: str,
) -> Tuple[np.ndarray, np.ndarray]:
    sub = df[df[popcol] == pop]
    if sub.empty:
        raise ValueError(f"No points found in parquet for population={pop!r} (popcol={popcol!r})")

    x = sub[xcol].to_numpy(dtype=float)
    y = sub[ycol].to_numpy(dtype=float)

    x_min, x_max = float(np.nanmin(x)), float(np.nanmax(x))
    y_min, y_max = float(np.nanmin(y)), float(np.nanmax(y))

    x_pad = xy_pad_frac * (x_max - x_min if x_max > x_min else 1.0)
    y_pad = xy_pad_frac * (y_max - y_min if y_max > y_min else 1.0)

    x_min -= x_pad
    x_max += x_pad
    y_min -= y_pad
    y_max += y_pad

    x_edges = np.linspace(x_min, x_max, grid_res_xy + 1)
    y_edges = np.linspace(y_min, y_max, grid_res_xy + 1)
    return x_edges, y_edges


def pixels_to_xy(
    bnd_pixels: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    grid_res_xy: int,
) -> np.ndarray:
    if bnd_pixels.shape[0] == 0:
        return np.zeros((0, 2), dtype=float)

    W = grid_res_xy
    H = grid_res_xy

    x0, x1 = float(x_edges[0]), float(x_edges[-1])
    y0, y1 = float(y_edges[0]), float(y_edges[-1])

    xs = x0 + (bnd_pixels[:, 0] + 0.5) * (x1 - x0) / W
    ys = y0 + (bnd_pixels[:, 1] + 0.5) * (y1 - y0) / H
    return np.stack([xs, ys], axis=1)


# ------------------------- main plotting -----------------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--debug_root", required=True)
    ap.add_argument("--embeddings_parquet", required=True)

    ap.add_argument("--labels_csv", default=None)
    ap.add_argument("--colors_csv", default=None)
    ap.add_argument("--popcol", default="population")
    ap.add_argument("--pops", default="ALL")

    ap.add_argument("--xcol", default="X1")
    ap.add_argument("--ycol", default="X2")
    ap.add_argument("--tcol", default="time")

    ap.add_argument("--nbins", type=int, default=24)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--min_boundary_points", type=int, default=50)

    ap.add_argument("--xy_pad_frac", type=float, default=0.05)
    ap.add_argument("--time_eps", type=float, default=1.0)

    # Hybrid detection + override
    ap.add_argument("--min_points_per_slice", type=int, default=None,
                    help="If None, uses meta min_points_per_slice else 5")
    ap.add_argument("--override_window", type=int, default=12,
                    help="Window size in u-slices used to decide when things become 'dense'")
    ap.add_argument("--override_valid_frac", type=float, default=0.7,
                    help="Fraction of valid slices in the window to declare 'dense'")

    ap.add_argument("--force_recent_t_max", type=float, default=None,
                    help="Optional manual override: treat times <= this as 'recent exact-hull region'")
    ap.add_argument("--force_recent_u_max", type=float, default=None,
                    help="Optional manual override: treat u <= this as 'recent exact-hull region'")

    ap.add_argument("--recent_trim_frac", type=float, default=0.95)
    ap.add_argument("--recent_min_points", type=int, default=10)
    ap.add_argument("--recent_max_hulls", type=int, default=80)

    ap.add_argument("--max_raw_points", type=int, default=200000)
    ap.add_argument("--raw_point_size", type=float, default=2.0)
    ap.add_argument("--raw_alpha", type=float, default=0.20)

    ap.add_argument("--z_mode", choices=["u_linear", "time_log"], default="time_log")
    ap.add_argument("--z_exaggeration", type=float, default=3.0)

    ap.add_argument("--out_dir", default="sandbox/_debug_figures")
    ap.add_argument("--out_name", default="binned_mask_hulls_3d.html")

    args = ap.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.embeddings_parquet)

    if args.labels_csv is not None:
        lab = pd.read_csv(args.labels_csv)
        if "sample_id" not in lab.columns or "population" not in lab.columns:
            raise ValueError("labels_csv must contain columns: sample_id,population")
        df = df.merge(lab[["sample_id", "population"]], on="sample_id", how="left")
        df[args.popcol] = df["population"].fillna("unlabelled_pop")
    else:
        if args.popcol not in df.columns:
            df[args.popcol] = "unlabelled_pop"

    colors_dict = {}
    if args.colors_csv is not None:
        cdf = pd.read_csv(args.colors_csv)
        if "population" in cdf.columns and "color" in cdf.columns:
            colors_dict = dict(zip(cdf["population"].astype(str), cdf["color"].astype(str)))

    available = list_pop_dirs(args.debug_root)
    if args.pops.strip().upper() == "ALL":
        pops = available
    else:
        pops = [p.strip() for p in args.pops.split(",") if p.strip()]
        missing = [p for p in pops if p not in available]
        if missing:
            raise ValueError(f"Requested pops not found under debug_root: {missing}. Found: {available}")

    fig = go.Figure()

    # raw points
    x = df[args.xcol].to_numpy(dtype=float)
    y = df[args.ycol].to_numpy(dtype=float)
    t = df[args.tcol].to_numpy(dtype=float)
    t_pos = np.clip(t, 1e-12, None)
    u_all = np.log10(np.maximum(t, args.time_eps))

    if args.z_mode == "u_linear":
        z_raw = u_all
        zaxis_type = "linear"
        zaxis_title = "u = log10(time)"
    else:
        z_raw = t_pos
        zaxis_type = "log"
        zaxis_title = "time"

    n = len(df)
    if n > args.max_raw_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=args.max_raw_points, replace=False)
        x_raw = x[idx]; y_raw = y[idx]; z_raw_ds = z_raw[idx]
        pop_raw = df.iloc[idx][args.popcol].astype(str).to_numpy()
    else:
        x_raw = x; y_raw = y; z_raw_ds = z_raw
        pop_raw = df[args.popcol].astype(str).to_numpy()

    for p in sorted(set(pop_raw.tolist())):
        m = (pop_raw == p)
        if m.sum() == 0:
            continue
        marker = dict(size=args.raw_point_size, opacity=args.raw_alpha)
        c = colors_dict.get(p, None)
        if c is not None:
            marker["color"] = c
        fig.add_trace(
            go.Scatter3d(
                x=x_raw[m], y=y_raw[m], z=z_raw_ds[m],
                mode="markers",
                name=f"raw {p}",
                marker=marker,
                showlegend=True,
                legendgroup=p,
            )
        )

    # hulls
    for pop in pops:
        pop_dir = str(Path(args.debug_root) / pop)
        meta = parse_meta_txt(str(Path(pop_dir) / "_meta_support_volume.txt"))

        grid_res_xy = get_int(meta, "grid_res_xy", None)
        us, counts, mask_paths = load_masks_us_counts(pop_dir)

        if grid_res_xy is None:
            any_mask = np.load(mask_paths[min(mask_paths.keys())])
            grid_res_xy = int(any_mask.shape[0])

        x_edges, y_edges = reconstruct_xy_edges_for_pop(
            df=df,
            pop=pop,
            grid_res_xy=grid_res_xy,
            xy_pad_frac=args.xy_pad_frac,
            xcol=args.xcol,
            ycol=args.ycol,
            popcol=args.popcol,
        )

        u_edges = edges_from_centers(us)

        mpps = args.min_points_per_slice
        if mpps is None:
            mpps = get_int(meta, "min_points_per_slice", None)
        if mpps is None:
            mpps = 5
        mpps = int(mpps)

        if counts.size != us.size:
            counts = infer_counts_from_points(
                df=df,
                pop=pop,
                popcol=args.popcol,
                tcol=args.tcol,
                u_edges=u_edges,
                time_eps=args.time_eps,
            )

        ui0 = recent_region_end_from_window(
            counts,
            min_points_per_slice=mpps,
            window=args.override_window,
            valid_frac=args.override_valid_frac,
        )

        # translate ui0 -> u threshold
        u_recent_max = float(u_edges[ui0]) if ui0 > 0 else float(u_edges[0])

        # optional manual forcing
        if args.force_recent_u_max is not None:
            u_recent_max = max(u_recent_max, float(args.force_recent_u_max))
        if args.force_recent_t_max is not None:
            u_recent_max = max(u_recent_max, float(np.log10(max(float(args.force_recent_t_max), args.time_eps))))

        t_recent_max = 10.0 ** u_recent_max

        print(
            f"[{pop}] recent override region: ui0={ui0}/{len(us)} "
            f"(mpps={mpps}, window={args.override_window}, frac={args.override_valid_frac}) "
            f"-> u_recent_max={u_recent_max:.6g} (t_recent_max={t_recent_max:.6g})"
        )

        # load masks
        max_ui = int(np.max(list(mask_paths.keys())))
        masks = []
        for ui in range(max_ui + 1):
            if ui in mask_paths:
                masks.append(np.load(mask_paths[ui]).astype(np.uint8))
            else:
                masks.append(np.zeros((grid_res_xy, grid_res_xy), dtype=np.uint8))
        masks = np.stack(masks, axis=0)

        if masks.shape[0] != us.size:
            if masks.shape[0] > us.size:
                masks = masks[:us.size]
            else:
                pad = np.zeros((us.size - masks.shape[0], grid_res_xy, grid_res_xy), dtype=np.uint8)
                masks = np.concatenate([masks, pad], axis=0)

        hull_color = colors_dict.get(pop, None)

        # A) recent exact per-time hulls
        sub = df[df[args.popcol] == pop].copy()
        if not sub.empty:
            tt = sub[args.tcol].to_numpy(dtype=float)
            uu = np.log10(np.maximum(tt, args.time_eps))
            recent = sub[uu <= u_recent_max].copy()

            if not recent.empty:
                times = np.sort(recent[args.tcol].unique())
                if len(times) > args.recent_max_hulls:
                    step = int(np.ceil(len(times) / args.recent_max_hulls))
                    times = times[::step]
                    print(f"[{pop}] recent hulls decimated: step={step} -> {len(times)} hulls")

                for ti in times:
                    pts = recent.loc[recent[args.tcol] == ti, [args.xcol, args.ycol]].to_numpy(dtype=float)
                    if pts.shape[0] < args.recent_min_points:
                        continue
                    pts = trim_central(pts, keep_frac=args.recent_trim_frac)
                    if pts.shape[0] < 3:
                        continue
                    hull = convex_hull_2d(pts)
                    if hull.shape[0] < 3:
                        continue
                    hull_closed = np.vstack([hull, hull[0]])

                    if args.z_mode == "u_linear":
                        zc = float(np.log10(max(float(ti), args.time_eps)))
                        z_line = np.full(hull_closed.shape[0], zc, dtype=float)
                    else:
                        zc = float(max(float(ti), 1e-12))
                        z_line = np.full(hull_closed.shape[0], zc, dtype=float)

                    line = dict(width=5)
                    if hull_color is not None:
                        line["color"] = hull_color

                    fig.add_trace(
                        go.Scatter3d(
                            x=hull_closed[:, 0],
                            y=hull_closed[:, 1],
                            z=z_line,
                            mode="lines",
                            name=f"{pop} recent",
                            legendgroup=pop,
                            showlegend=False,
                            line=line,
                            opacity=0.98,
                            hovertemplate=f"pop={pop}<br>recent_exact<br>t={float(ti):.3g}<extra></extra>",
                        )
                    )

        # B) older binned masks (exclude recent region by u threshold)
        umin, umax = float(np.min(us)), float(np.max(us))
        edges_u = np.linspace(umin, umax, args.nbins + 1)
        centers_u = 0.5 * (edges_u[:-1] + edges_u[1:])

        for bi in range(args.nbins):
            lo, hi = edges_u[bi], edges_u[bi + 1]
            zc_u = float(centers_u[bi])

            in_bin = np.where((us >= lo) & (us < hi) & (us > u_recent_max))[0]
            if in_bin.size == 0:
                continue

            frac = masks[in_bin].mean(axis=0)
            bin_mask = (frac >= args.tau).astype(np.uint8)

            bnd_pix = boundary_pixels_4n(bin_mask)
            if bnd_pix.shape[0] < args.min_boundary_points:
                continue

            pts_xy = pixels_to_xy(bnd_pix, x_edges, y_edges, grid_res_xy)
            hull = convex_hull_2d(pts_xy)
            if hull.shape[0] < 3:
                continue

            hull_closed = np.vstack([hull, hull[0]])

            if args.z_mode == "u_linear":
                z_line = np.full(hull_closed.shape[0], zc_u, dtype=float)
            else:
                z_line = np.full(hull_closed.shape[0], 10.0 ** zc_u, dtype=float)

            line = dict(width=4)
            if hull_color is not None:
                line["color"] = hull_color

            fig.add_trace(
                go.Scatter3d(
                    x=hull_closed[:, 0],
                    y=hull_closed[:, 1],
                    z=z_line,
                    mode="lines",
                    name=f"{pop} bin{bi:02d}",
                    legendgroup=pop,
                    showlegend=False,
                    line=line,
                    opacity=0.95,
                    hovertemplate=(
                        f"pop={pop}<br>bin={bi:02d}<br>"
                        f"u=[{lo:.3f},{hi:.3f})<br>k={int(in_bin.size)}<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        title=f"Hybrid hulls: binned masks + recent exact hulls (nbins={args.nbins}, tau={args.tau}, z_mode={args.z_mode})",
        scene=dict(
            xaxis_title=args.xcol,
            yaxis_title=args.ycol,
            zaxis_title=zaxis_title,
            zaxis=dict(type=zaxis_type),
            aspectmode="manual",
            aspectratio=dict(x=1, y=1, z=max(0.1, float(args.z_exaggeration))),
        ),
        margin=dict(l=0, r=0, t=60, b=0),
        legend=dict(itemsizing="constant"),
    )

    out_path = str(Path(args.out_dir) / args.out_name)
    fig.write_html(out_path, include_plotlyjs="cdn")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()