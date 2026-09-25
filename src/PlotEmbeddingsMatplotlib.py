"""Matplotlib renderer for CoaSTT 3D embedding panels (X1, X2, log time).

The scene is projected orthographically by hand and drawn with ordinary 2D artists, so text uses the house font at true
point sizes, the frame is vector, and trajectories are depth-sorted segments kept as vector or rasterised into one
layer.  No kaleido or Chrome.

Geometry.  X1-X2 are rotated (a gauge freedom) so the most recent slice's first principal axis runs across the screen.
The floor is a square fitted to the data footprint (or +-xy_lim when panels must share a scale); its half-side is the
unit of length.  Time is log-scaled onto a vertical axis whose height, in floor half-sides, is chosen automatically so
the drawing exactly fills the axes it is given, without distortion.  Tick lengths and label offsets are in points.

    plotter = PlotEmbeddingsMatplotlib(df, colours)   # df: individual_id, time, X1, X2; colours: individual_id -> colour
    plotter.draw(ax)                                  # fills ax (any shape; square is typical)

Unlike PlotEmbeddings (Plotly), this needs no kaleido/Chrome and draws directly into a matplotlib axes of a larger
figure, so it can replace the save_static -> place_image round trip in the figure scripts.
"""
import numpy as np
from matplotlib import ticker, rcParams
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.font_manager import FontProperties
from matplotlib.markers import MarkerStyle
from matplotlib.path import Path
from matplotlib.text import Text

PLANE_COLOR, GRID_COLOR, EDGE_COLOR = "whitesmoke", "white", "black"
FRAME_LW, GRID_LW = 0.6, 0.6
PAD_PT = 1.5                                   # blank margin around the drawing, points


def _fmt_xy(v):
    a = abs(v)
    if v == 0: return "0"
    if a >= 1e6: return f"{v/1e6:g}M"
    if a >= 1e3: return f"{v/1e3:g}k"
    return f"{v:g}"


def _log_ticks(lo, hi):
    k0, k1 = np.ceil(np.log10(lo)), np.floor(np.log10(hi))
    major = 10.0 ** np.arange(k0, k1 + 1)
    minor = np.array([m * 10.0 ** k for k in np.arange(k0 - 1, k1 + 1) for m in range(2, 10)])
    return major, minor[(minor >= lo) & (minor <= hi)]


def orient(df, azim=45, t=None):
    """Rotate X1-X2 so the first principal axis of the slice at time t (default: most recent) lies along the screen
    horizontal for a camera at `azim`; heavier tail to the right.  Returns a copy."""
    t = df.time.min() if t is None else t
    X = df.loc[df.time == t, ["X1", "X2"]].to_numpy(float); X = X - X.mean(0)
    u = np.linalg.svd(X, full_matrices=False)[2][0]
    if np.sum((X @ u) ** 3) < 0: u = -u
    a = np.radians(azim); h = np.array([np.cos(a), -np.sin(a)])
    th = np.arctan2(h[1], h[0]) - np.arctan2(u[1], u[0])
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    out = df.copy(); out[["X1", "X2"]] = df[["X1", "X2"]].to_numpy(float) @ R.T
    return out


class PlotEmbeddingsMatplotlib:
    def __init__(self, df, colours, tmin=None, tmax=None, xy_lim=None, azim=45, elev=8, orient_view=True,
                 orient_time=None, floor_margin=0.06):
        """tmin/tmax: time range (default: smallest positive and largest time in df; earlier rows are dropped).
        xy_lim: fixed square floor +-xy_lim (share it across panels that must be compared); default fits the data.
        orient_view: rotate X1-X2 as in `orient`."""
        t = df.time.to_numpy(float)
        self.tmin = float(t[t > 0].min() if tmin is None else tmin); self.tmax = float(t.max() if tmax is None else tmax)
        df = df[(df.time >= self.tmin) & (df.time <= self.tmax)]
        if orient_view: df = orient(df, azim=azim, t=orient_time)
        self.azim, self.elev = float(azim), float(elev)
        xy = df[["X1", "X2"]].to_numpy(float)
        if xy_lim is None:
            lo, hi = xy.min(0), xy.max(0); self.c = (lo + hi) / 2; self.u = (hi - lo).max() / 2 * (1 + floor_margin)
        else:
            self.c = np.zeros(2); self.u = float(xy_lim)
        df = df.sort_values(["individual_id", "time"])
        self.x = (df.X1.to_numpy(float) - self.c[0]) / self.u
        self.y = (df.X2.to_numpy(float) - self.c[1]) / self.u
        self.w = self.zfrac(df.time.to_numpy(float))
        ids = df.individual_id.to_numpy()
        self.seg = np.flatnonzero(ids[1:] == ids[:-1])                      # segment i joins rows i and i+1
        self.seg_colour = np.array([colours[i] for i in ids[self.seg]], dtype=object)
        starts = np.r_[0, np.flatnonzero(ids[1:] != ids[:-1]) + 1, len(ids)]
        self.paths = [slice(a, b) for a, b in zip(starts[:-1], starts[1:])]  # one row range per individual
        self.path_colour = np.array([colours[ids[sl.start]] for sl in self.paths], dtype=object)
        self.t_major, self.t_minor = _log_ticks(self.tmin, self.tmax)
        # which floor corners sit where on screen
        a = np.radians(self.azim)
        sx_of = lambda x, y: x * np.cos(a) - y * np.sin(a)
        dp_of = lambda x, y: x * np.sin(a) + y * np.cos(a)                  # towards the back
        corners = [(sx, sy) for sx in (-1.0, 1.0) for sy in (-1.0, 1.0)]
        depth = [dp_of(*c) for c in corners]; scr = [sx_of(*c) for c in corners]
        self.front = corners[int(np.argmin(depth))]; self.back = corners[int(np.argmax(depth))]
        self.left = corners[int(np.argmin(scr))]; self.right = corners[int(np.argmax(scr))]

    def zfrac(self, t):
        t = np.clip(np.asarray(t, float), self.tmin, self.tmax)
        return (np.log10(t) - np.log10(self.tmin)) / (np.log10(self.tmax) - np.log10(self.tmin))

    def _screen(self, x, y, w, za):
        a, e = np.radians(self.azim), np.radians(self.elev)
        x, y, w = np.asarray(x, float), np.asarray(y, float), np.asarray(w, float)
        y1 = x * np.sin(a) + y * np.cos(a); z = za * w
        return x * np.cos(a) - y * np.sin(a), y1 * np.sin(e) + z * np.cos(e), y1 * np.cos(e) - z * np.sin(e)

    # ---- layout ------------------------------------------------------------------------------------------------
    def _items(self, fig, opts):
        """Everything that occupies space: rows of (x, y, w, dx_pt, dy_pt).  Geometry has zero offsets; each text
        contributes the four corners of its box.  Also returns the text specs to draw."""
        ren = fig.canvas.get_renderer()
        def size(s, fs):
            t = Text(0, 0, s, fontproperties=FontProperties(size=fs)); t.set_figure(fig)
            bb = t.get_window_extent(ren); return bb.width * 72 / fig.dpi, bb.height * 72 / fig.dpi
        rows, texts = [], []
        def text(s, x, y, w, off, ha, va, rot, fs):
            tw, th = size(s, fs)
            x0 = {"left": 0, "center": -tw / 2, "right": -tw}[ha]; y0 = {"bottom": 0, "center": -th / 2, "top": -th}[va]
            box = np.array([[x0, y0], [x0 + tw, y0], [x0 + tw, y0 + th], [x0, y0 + th]])
            r = np.radians(rot); R = np.array([[np.cos(r), -np.sin(r)], [np.sin(r), np.cos(r)]])
            for cx, cy in box @ R.T + np.asarray(off, float): rows.append((x, y, w, cx, cy))
            texts.append(dict(s=s, xyz=(x, y, w), off=tuple(off), ha=ha, va=va, rot=rot, fs=fs, width=tw))
            return tw
        ft, fl, tl = opts["ft"], opts["fl"], opts["tick_len"]
        # geometry: floor and wall tops, trajectories
        for cx in (-1, 1):
            for cy in (-1, 1):
                rows.append((cx, cy, 0, 0, 0)); rows.append((cx, cy, 1, 0, 0))
        gx, gy, gw = self.x, self.y, self.w
        # time axis on the left
        (lx, ly), (rx, ry) = self.left, self.right
        if opts["show_time_ticks"]:
            wmax = max(text(f"$10^{{{int(np.log10(t))}}}$", lx, ly, self.zfrac(t), (-(tl + 2), 0), "right", "center", 0, ft)
                       for t in self.t_major) if len(self.t_major) else 0
        else:
            wmax = 0
        if opts["time_label"]:
            text(opts["time_label"], lx, ly, 0.5, (-(tl + 2 + wmax + 3), 0), "center", "bottom", 90, fl)
        g = opts["generation_time"]
        if g:
            ymaj, _ = _log_ticks(self.tmin * g, self.tmax * g)
            wr = max(text(f"$10^{{{int(np.log10(v))}}}$", rx, ry, self.zfrac(v / g), (tl + 2, 0), "left", "center", 0, ft)
                     for v in ymaj) if len(ymaj) else 0
            text(f"Time (years ago, g = {g:g} years)", rx, ry, 0.5, (tl + 2 + wr + 3, 0), "center", "top", 90, fl)
        # X1 / X2 along the two front floor edges
        (fx, fy) = self.front
        for axis, other in ((0, (-fx, fy)), (1, (fx, -fy))):                  # X1 edge varies x, X2 edge varies y
            p0 = np.array(self.front, float); p1 = np.array(other, float)
            s0 = np.array(self._screen(*p0, 0, 1)[:2]); s1 = np.array(self._screen(*p1, 0, 1)[:2])
            v = s1 - s0; ang = np.degrees(np.arctan2(v[1], v[0]))
            ang = ang - 180 if ang > 90 else ang + 180 if ang < -90 else ang
            n = np.array([v[1], -v[0]]) / np.linalg.norm(v)
            if n[1] > 0: n = -n                                              # outward = downward on screen
            k = int(np.flatnonzero(p1 != p0)[0])                              # the coordinate that varies on this edge
            loc = ticker.MaxNLocator(nbins=opts["n_xy_ticks"], steps=[1, 2, 5, 10])
            lo, hi = self.c[k] - self.u, self.c[k] + self.u
            vals = [u for u in loc.tick_values(lo, hi) if lo <= u <= hi]
            ticks = []
            for u in vals:
                q = p0.copy(); q[k] = (u - self.c[k]) / self.u
                if abs(q[k] - p0[k]) < 0.24: continue                         # clear of the shared front corner
                ticks.append((u, q))
            spec = dict(k=k, p0=p0, p1=p1, n=n, ang=ang, ticks=ticks, grid=vals)
            opts.setdefault("edges", []).append(spec)
            if opts["show_xy_ticks"]:
                for u, q in ticks:
                    text(_fmt_xy(u), q[0], q[1], 0, tuple(n * (tl + 1.5 + 0.55 * ft)), "center", "center", ang, ft)
            if opts["xy_labels"][axis]:
                mid = (p0 + p1) / 2; off = (tl + 1.5 + 1.2 * ft + 0.75 * fl) if opts["show_xy_ticks"] else (2 + 0.7 * fl)
                text(opts["xy_labels"][axis], mid[0], mid[1], 0, tuple(n * off), "center", "center", ang, fl)
        R = np.array(rows, float)
        G = np.c_[gx, gy, gw, np.zeros((len(gx), 2))]
        return np.vstack([R, G]), texts

    def _fit(self, pts, W, H, za):
        """Scale k (points per unit) and bounds for a given time-axis height za, fitting inside W x H points."""
        sx, sy, _ = self._screen(pts[:, 0], pts[:, 1], pts[:, 2], za); ox, oy = pts[:, 3], pts[:, 4]
        ext = lambda s, o, k: (np.max(k * s + o) - np.min(k * s + o))
        def solve(s, o, target):
            lo, hi = 0.0, 1.0
            while ext(s, o, hi) < target: hi *= 2
            for _ in range(60):
                mid = (lo + hi) / 2
                if ext(s, o, mid) <= target: lo = mid
                else: hi = mid
            return max(lo, 1e-9)
        kw, kh = solve(sx, ox, W), solve(sy, oy, H)
        return kw, kh, (sx, sy, ox, oy)

    def draw(self, ax, z_aspect="auto", lw=0.3, alpha=0.55, rasterized=False, time_label="Time (generations ago)",
             xy_labels=("X1", "X2"), show_time_ticks=True, show_xy_ticks=True, minor_ticks=True, planes=True,
             tick_fontsize=None, label_fontsize=None, tick_len=3.0, n_xy_ticks=4, generation_time=None,
             depth_sort="individual"):
        """Draw into `ax`, filling it.  z_aspect: time-axis height in floor half-sides, or 'auto' to choose it so the
        drawing has the axes' shape exactly.  generation_time: years per generation; adds a years axis on the right.
        Output is vector by default.  depth_sort: 'individual' draws each trajectory as one path, ordered far to near by
        its mean depth (clean with transparency, small PDFs); 'segment' orders every segment by depth (exact occlusion
        where trajectories cross, but each joint is painted twice, visible as beading with alpha < 1).  rasterized: turn
        the trajectories into one bitmap layer (not for publication PDFs)."""
        fig = ax.figure
        fs = lambda v, key: FontProperties(size=v if v is not None else rcParams[key]).get_size_in_points()
        opts = dict(ft=fs(tick_fontsize, "xtick.labelsize"), fl=fs(label_fontsize, "axes.labelsize"), tick_len=tick_len,
                    show_time_ticks=show_time_ticks, show_xy_ticks=show_xy_ticks, time_label=time_label,
                    xy_labels=xy_labels, generation_time=generation_time, n_xy_ticks=n_xy_ticks)
        pts, texts = self._items(fig, opts)
        bb = ax.get_window_extent(fig.canvas.get_renderer())
        W, H = bb.width * 72 / fig.dpi - 2 * PAD_PT, bb.height * 72 / fig.dpi - 2 * PAD_PT
        if z_aspect == "auto":                                  # width does not depend on za: fix k from it, solve za
            kw, _, _ = self._fit(pts, W, H, 1.0)
            h_at = lambda za: self._fit(pts, W, H, za)[1] >= kw   # True while the drawing still fits vertically at kw
            lo, hi = 0.05, 50.0
            if not h_at(lo): za = lo
            elif h_at(hi): za = hi
            else:
                for _ in range(50):
                    mid = np.sqrt(lo * hi)
                    if h_at(mid): lo = mid
                    else: hi = mid
                za = lo
        else:
            za = float(z_aspect)
        kw, kh, (sx, sy, ox, oy) = self._fit(pts, W, H, za); k = min(kw, kh)
        X, Y = k * sx + ox, k * sy + oy
        x0 = (X.min() - (W - (X.max() - X.min())) / 2 - PAD_PT) / k; y0 = (Y.min() - (H - (Y.max() - Y.min())) / 2 - PAD_PT) / k
        ax.set_xlim(x0, x0 + (W + 2 * PAD_PT) / k); ax.set_ylim(y0, y0 + (H + 2 * PAD_PT) / k); ax.axis("off")
        self.z_aspect = za

        S = lambda x, y, w: np.c_[self._screen(x, y, w, za)[:2]]
        (lx, ly), (rx, ry), (bx, by) = self.left, self.right, self.back
        # planes and grid: floor and the two walls meeting at the back corner
        if planes:
            floor = S([-1, 1, 1, -1], [-1, -1, 1, 1], [0, 0, 0, 0])
            walls = [S([bx, bx, bx, bx], [-1, 1, 1, -1], [0, 0, 1, 1]), S([-1, 1, 1, -1], [by, by, by, by], [0, 0, 1, 1])]
            ax.add_collection(PolyCollection([floor] + walls, facecolors=PLANE_COLOR, edgecolors="none", zorder=0))
            grid = []
            for e in opts["edges"]:
                k_ = e["k"]
                for u in e["grid"]:
                    v = (u - self.c[k_]) / self.u
                    if k_ == 0: grid += [S([v, v], [-1, 1], [0, 0]), S([v, v], [by, by], [0, 1])]
                    else: grid += [S([-1, 1], [v, v], [0, 0]), S([bx, bx], [v, v], [0, 1])]
            for t in self.t_major:
                w = self.zfrac(t); grid += [S([bx, bx], [-1, 1], [w, w]), S([-1, 1], [by, by], [w, w])]
            ax.add_collection(LineCollection(grid, colors=GRID_COLOR, linewidths=GRID_LW, zorder=1))
        # frame: time axis, the two front floor edges, and the years axis if requested
        (fx, fy) = self.front
        frame = [S([lx, lx], [ly, ly], [0, 1]), S([fx, lx], [fy, ly], [0, 0]), S([fx, rx], [fy, ry], [0, 0])]
        if generation_time: frame.append(S([rx, rx], [ry, ry], [0, 1]))
        ax.add_collection(LineCollection(frame, colors=EDGE_COLOR, linewidths=FRAME_LW, zorder=1.5, capstyle="projecting"))
        # trajectories, far to near
        sx_, sy_, dp = self._screen(self.x, self.y, self.w, za); P = np.c_[sx_, sy_]
        if depth_sort == "segment":
            i = self.seg; lines = np.stack([P[i], P[i + 1]], 1); order = np.argsort(-(dp[i] + dp[i + 1])); cols = self.seg_colour
        else:
            lines = [P[sl] for sl in self.paths]; order = np.argsort([-dp[sl].mean() for sl in self.paths]); cols = self.path_colour
            lines = [lines[j] for j in order]
        ax.add_collection(LineCollection(lines if depth_sort != "segment" else lines[order], colors=cols[order], linewidths=lw,
                                         alpha=alpha, zorder=2, rasterized=rasterized, joinstyle="round", capstyle="round"))
        # ticks: one-sided markers of fixed point length
        def tickmark(xy, direction, length):
            if len(xy) == 0: return
            ax.plot(*np.asarray(xy).T, ls="none", marker=MarkerStyle(Path([(0, 0), tuple(direction)])), markersize=2 * length,
                    markeredgewidth=FRAME_LW, markeredgecolor=EDGE_COLOR, zorder=1.6, clip_on=False)
        if show_time_ticks:
            tickmark(S([lx] * len(self.t_major), [ly] * len(self.t_major), self.zfrac(self.t_major)), (-1, 0), tick_len)
            if minor_ticks:
                tickmark(S([lx] * len(self.t_minor), [ly] * len(self.t_minor), self.zfrac(self.t_minor)), (-1, 0), tick_len / 2)
            if generation_time:
                ymaj, ymin = _log_ticks(self.tmin * generation_time, self.tmax * generation_time)
                tickmark(S([rx] * len(ymaj), [ry] * len(ymaj), self.zfrac(ymaj / generation_time)), (1, 0), tick_len)
                if minor_ticks:
                    tickmark(S([rx] * len(ymin), [ry] * len(ymin), self.zfrac(ymin / generation_time)), (1, 0), tick_len / 2)
        if show_xy_ticks:
            for e in opts["edges"]:
                if e["ticks"]: tickmark(S(*np.array([q for _, q in e["ticks"]]).T, np.zeros(len(e["ticks"]))), e["n"], tick_len)
        for t in texts:
            xy = S(*[[c] for c in t["xyz"]])[0]
            ax.annotate(t["s"], xy, xytext=t["off"], textcoords="offset points", ha=t["ha"], va=t["va"], rotation=t["rot"],
                        rotation_mode="anchor", fontsize=t["fs"], annotation_clip=False)
        return ax
