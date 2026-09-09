#%%
"""
House style for every manuscript figure, plus the small helpers the plotting
scripts share. Importing this module applies the style (rcParams) as a side
effect, so `from results.figures.plotting_style import *` is all a script needs.

Rules the helpers enforce (see .claude/skills/figure-designer/SKILL.md):
  * width  <= 6.5 in (double column) or 3.25 in (single column); height <= 8.5 in
  * Latin Modern Roman everywhere (the LaTeX article font), Computer Modern math
  * no text smaller than FONT_SIZES["min"] points at final size
  * layout in inches via GridSpec strips; PDF + PNG written side by side
"""
import os
import warnings

import matplotlib
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.text
from cycler import cycler
from matplotlib import rcParams

FIGURES_DIR = os.path.dirname(os.path.abspath(__file__))

# Tables the figures are drawn from, written by utils/runs/prepare_figure_data/.
# Plotting scripts read only from here: data acquisition lives in the prep scripts.
FIGURE_DATA = os.path.abspath(os.path.join(FIGURES_DIR, "..", "figure_data"))


def figure_data(figure_name, *parts):
    """Path to one of a figure's prepared tables, e.g. figure_data("complex_4_pops", "embeddings.parquet")."""
    return os.path.join(FIGURE_DATA, figure_name, *parts)


# ---------------------------------------------------------------------------
# Sizes. Everything is in inches at final (printed) size; nothing is scaled later.
# ---------------------------------------------------------------------------
WIDTHS = {"single": 3.25, "double": 6.5}   # journal column widths
MAX_WIDTH = WIDTHS["double"]
MAX_HEIGHT = 8.5                            # leaves room for a caption on a page
PNG_DPI = 200                               # the PNG is for inspection only

# Font sizes in points. Edit here, not in the scripts: every script reads this
# table (directly, or through the rcParams set below).
FONT_SIZES = {
    "title": 11,        # figure / row / column titles
    "panel_label": 11,  # the bold A, B, C letters
    "base": 9,          # axis labels, legend titles, in-panel text
    "tick": 8,          # tick labels
    "legend": 8,        # legend entries
    "annotation": 7,    # small in-panel annotations, colorbar ticks
    "min": 7,           # nothing may be smaller than this
}

# ---------------------------------------------------------------------------
# Colours.
# ---------------------------------------------------------------------------
# Categorical fallback (colourblind-safe, distinguishable in greyscale). Population
# colours are never chosen here: they come from the figure's pop_colors.csv.
okabe_ito = [
    '#0072B2',  # blue
    '#D55E00',  # vermillion
    '#009E73',  # green
    '#E69F00',  # orange
    '#CC79A7',  # pink
    '#56B4E9',  # sky blue
    '#F0E442',  # yellow
    '#000000',  # black
]
TRUTH_COLOR = "black"        # reserved: the true / analytic quantity
SIMULATED_COLOR = "0.45"     # reserved: results from simulated (true) trees
SEQUENTIAL_CMAP = "viridis"
DIVERGING_CMAP = "RdBu_r"

# Line conventions for the three kinds of curve that recur across figures.
TRUTH_STYLE = dict(color=TRUTH_COLOR, linestyle="--", linewidth=1.2, zorder=3)
ESTIMATE_STYLE = dict(linestyle="-", linewidth=1.2, zorder=2)             # colour from the data
REPLICATE_STYLE = dict(linestyle="-", linewidth=0.5, alpha=0.35, zorder=1)  # colour from the data

# ---------------------------------------------------------------------------
# Fonts: Latin Modern Roman, shipped in results/figures/fonts/ so the figures
# render identically on every machine. Math is Computer Modern, which Latin
# Modern is a copy of, so text and formulas match.
# ---------------------------------------------------------------------------
FONTS_DIR = os.path.join(FIGURES_DIR, "fonts")
FONT_FAMILY = "Latin Modern Roman"


def register_fonts():
    for fname in sorted(os.listdir(FONTS_DIR)):
        if fname.endswith((".otf", ".ttf")):
            fm.fontManager.addfont(os.path.join(FONTS_DIR, fname))


def setup_plotting_style():
    register_fonts()

    # -- Output / vector settings --
    rcParams['savefig.format'] = 'pdf'
    rcParams['pdf.fonttype'] = 42       # embed fonts as text, not paths (editable/searchable in PDF)
    rcParams['ps.fonttype'] = 42
    rcParams['svg.fonttype'] = 'none'
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.0    # the inch strips are the page: no extra border
    rcParams['savefig.transparent'] = False
    rcParams['figure.dpi'] = 150        # on-screen preview only
    rcParams['savefig.dpi'] = PNG_DPI

    # -- Fonts (match a LaTeX article) --
    rcParams['font.family'] = FONT_FAMILY
    rcParams['font.serif'] = [FONT_FAMILY, 'STIXGeneral', 'DejaVu Serif']
    rcParams['mathtext.fontset'] = 'cm'
    rcParams['font.size'] = FONT_SIZES["base"]
    rcParams['axes.titlesize'] = FONT_SIZES["title"]
    rcParams['axes.labelsize'] = FONT_SIZES["base"]
    rcParams['xtick.labelsize'] = FONT_SIZES["tick"]
    rcParams['ytick.labelsize'] = FONT_SIZES["tick"]
    rcParams['legend.fontsize'] = FONT_SIZES["legend"]
    rcParams['legend.title_fontsize'] = FONT_SIZES["base"]
    rcParams['figure.titlesize'] = FONT_SIZES["title"]
    rcParams['text.color'] = 'black'
    rcParams['axes.labelcolor'] = 'black'
    rcParams['axes.unicode_minus'] = False

    # -- Axes --
    rcParams['axes.spines.bottom'] = True
    rcParams['axes.spines.left'] = True
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    rcParams['axes.grid'] = False
    rcParams['axes.linewidth'] = 0.8
    rcParams['axes.labelpad'] = 3
    rcParams['axes.titlepad'] = 4

    # -- Ticks --
    rcParams['axes.edgecolor'] = 'black'
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8
    rcParams['xtick.major.size'] = 3
    rcParams['ytick.major.size'] = 3
    rcParams['xtick.minor.visible'] = False
    rcParams['ytick.minor.visible'] = False
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.major.pad'] = 2.5
    rcParams['ytick.major.pad'] = 2.5

    # -- Lines and markers --
    rcParams['lines.linewidth'] = 1.2
    rcParams['lines.markersize'] = 4
    rcParams['lines.markeredgewidth'] = 0.6
    rcParams['lines.solid_capstyle'] = 'round'

    # -- Bars / patches --
    rcParams['patch.linewidth'] = 0.6
    rcParams['patch.edgecolor'] = 'black'
    rcParams['patch.force_edgecolor'] = True

    # -- Legend --
    rcParams['legend.frameon'] = False
    rcParams['legend.handlelength'] = 1.5
    rcParams['legend.labelspacing'] = 0.3
    rcParams['legend.borderaxespad'] = 0.4
    rcParams['legend.columnspacing'] = 1.0

    rcParams['axes.prop_cycle'] = cycler(color=okabe_ito)
    rcParams['errorbar.capsize'] = 2

    # Layout is done by hand in inches (see inch_gridspec); auto layout fights
    # with panel labels, colourbars and raster panels.
    rcParams['figure.figsize'] = (WIDTHS["double"], 4)
    rcParams['figure.constrained_layout.use'] = False
    rcParams['figure.autolayout'] = False


setup_plotting_style()


# ---------------------------------------------------------------------------
# Layout helpers. A figure is a stack/row of strips whose sizes are given in
# inches; margins are pinned to the figure edge so the ratios *are* inches.
# ---------------------------------------------------------------------------
def new_figure(width="double", height=4.0):
    """
    Figure of the given width class ("single" / "double", or a number of inches)
    and height in inches. Raises if the size breaks the page limits.
    """
    w = WIDTHS.get(width, width)
    if w > MAX_WIDTH + 1e-6:
        raise ValueError(f"figure width {w} in exceeds {MAX_WIDTH} in")
    if height > MAX_HEIGHT + 1e-6:
        raise ValueError(f"figure height {height} in exceeds {MAX_HEIGHT} in")
    return plt.figure(figsize=(w, height))


def inch_gridspec(fig_or_spec, heights=None, widths=None, hgap=0.0, wgap=0.0):
    """
    GridSpec whose rows/columns are `heights`/`widths` in inches, with fixed gaps
    (also in inches) between them. Pass the figure for the outer grid, or a
    SubplotSpec cell to subdivide it. Either list may be omitted for one row/col.

    The gaps are realised as extra rows/columns, so the returned grid has index
    helpers: use grid.cell(i, j) to get the SubplotSpec of content cell (i, j).
    """
    heights = list(heights) if heights is not None else None
    widths = list(widths) if widths is not None else None

    def with_gaps(sizes, gap):
        if sizes is None:
            return [1.0], lambda i: 0
        out = []
        for k, s in enumerate(sizes):
            if k:
                out.append(gap)
            out.append(s)
        return out, lambda i: 2 * i

    h_ratios, h_index = with_gaps(heights, hgap)
    w_ratios, w_index = with_gaps(widths, wgap)

    if isinstance(fig_or_spec, matplotlib.figure.FigureBase):
        gs = gridspec.GridSpec(len(h_ratios), len(w_ratios), figure=fig_or_spec,
                               height_ratios=h_ratios, width_ratios=w_ratios,
                               left=0, right=1, bottom=0, top=1, hspace=0, wspace=0)
    else:
        gs = gridspec.GridSpecFromSubplotSpec(len(h_ratios), len(w_ratios),
                                              subplot_spec=fig_or_spec,
                                              height_ratios=h_ratios, width_ratios=w_ratios,
                                              hspace=0, wspace=0)
    gs.cell = lambda i, j=0: gs[h_index(i), w_index(j)]
    gs.total_height = sum(h_ratios) if heights is not None else None
    gs.total_width = sum(w_ratios) if widths is not None else None
    return gs


def add_axes_inches(fig, x, y, w, h, **kw):
    """Axes at (x, y) from the figure's bottom-left, w x h, all in inches."""
    fw, fh = fig.get_size_inches()
    return fig.add_axes([x / fw, y / fh, w / fw, h / fh], **kw)


def add_panel_label(ax, letter, dx=4.0, dy=-2.0, **kw):
    """
    Bold upper-case panel letter at the top-left INSIDE corner of `ax`, offset
    by (dx, dy) points from that corner. If it collides with an element, push it
    with a larger negative dx (left) or a positive dy (up): e.g. dx=-14 puts it
    left of the y axis, dy=4 puts it above the frame.
    """
    kw.setdefault("fontsize", FONT_SIZES["panel_label"])
    kw.setdefault("fontweight", "bold")
    return ax.annotate(letter.upper(), xy=(0, 1), xycoords="axes fraction",
                       xytext=(dx, dy), textcoords="offset points",
                       ha="left", va="top", annotation_clip=False, zorder=10, **kw)


def place_image(ax, img, **imshow_kw):
    """Show a rendered (raster) panel in `ax` with no axes furniture."""
    ax.set_axis_off()
    imshow_kw.setdefault("aspect", "equal")
    imshow_kw.setdefault("interpolation", "lanczos")
    return ax.imshow(img, **imshow_kw)


def log_axis(ax, axis="y", base=10.0):
    """Log scale with decade ticks labelled 10^k only (no minor ticks)."""
    which = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log", base=base)
    which.set_major_locator(ticker.LogLocator(base=base))
    which.set_major_formatter(ticker.LogFormatterSciNotation(base=base))
    which.set_minor_locator(ticker.NullLocator())
    which.set_minor_formatter(ticker.NullFormatter())


def clean_axes(ax, keep=("left", "bottom")):
    """Hide every spine not in `keep`; hide ticks on hidden sides."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)
    ax.tick_params(left="left" in keep, bottom="bottom" in keep,
                   labelleft="left" in keep, labelbottom="bottom" in keep)


TIME_LABEL = "Time (generations ago)"
TIME_LABEL_YEARS = "Time (years ago)"


# ---------------------------------------------------------------------------
# Output. Every figure is written as PDF (manuscript) and PNG (inspection).
# ---------------------------------------------------------------------------
def check_figure(fig, max_width=MAX_WIDTH, max_height=MAX_HEIGHT):
    """
    Raise if the figure, once saved with a tight bounding box, is wider than
    max_width or taller than max_height, or if any visible text is smaller than
    FONT_SIZES["min"]. Returns the tight (width, height) in inches.
    """
    pad = rcParams['savefig.pad_inches']
    renderer = fig.canvas.get_renderer()
    bbox = fig.get_tightbbox(renderer)
    w = bbox.width + 2 * pad
    h = bbox.height + 2 * pad
    if w > max_width + 0.01:
        raise ValueError(f"saved width {w:.2f} in exceeds {max_width} in "
                         f"(figsize {fig.get_size_inches()[0]:.2f}): something sticks out")
    if h > max_height + 0.01:
        raise ValueError(f"saved height {h:.2f} in exceeds {max_height} in")
    small = sorted({round(t.get_fontsize(), 1) for t in fig.findobj(matplotlib.text.Text)
                    if t.get_visible() and t.get_text().strip()
                    and t.get_fontsize() < FONT_SIZES["min"] - 1e-6})
    if small:
        raise ValueError(f"text smaller than {FONT_SIZES['min']} pt: sizes {small}")
    return w, h


def save_figure(fig, name, kind="sim", check=True):
    """
    Save `results/figures/<kind>_figs/<name>.pdf` and `<kind>_figs_png/<name>.png`
    (kind: "sim" or "app"). Runs check_figure first unless check=False.
    """
    fig.canvas.draw()
    if check:
        w, h = check_figure(fig)
    else:
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        w, h = bbox.width, bbox.height
    pdf_dir = os.path.join(FIGURES_DIR, f"{kind}_figs")
    png_dir = os.path.join(FIGURES_DIR, f"{kind}_figs_png")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(png_dir, exist_ok=True)
    pdf = os.path.join(pdf_dir, f"{name}.pdf")
    png = os.path.join(png_dir, f"{name}.png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=PNG_DPI)
    print(f"saved {pdf} and {png}  ({w:.2f} x {h:.2f} in)")
    return pdf, png


def check_font():
    """Warn if Latin Modern did not load (the PDF would silently fall back)."""
    found = fm.findfont(fm.FontProperties(family=FONT_FAMILY), fallback_to_default=True)
    if "lmroman" not in os.path.basename(found).lower():
        warnings.warn(f"{FONT_FAMILY} not found; matplotlib fell back to {found}")


check_font()
# %%
