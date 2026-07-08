#%%
from matplotlib import rcParams
from cycler import cycler
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

def setup_plotting_style():
    # -- Output / vector settings --
    rcParams['savefig.format'] = 'pdf'
    rcParams['pdf.fonttype'] = 42       # embed fonts as actual text, not paths (editable/searchable in PDF)
    rcParams['ps.fonttype'] = 42
    rcParams['svg.fonttype'] = 'none'   # keep text as text if exporting SVG
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.02
    rcParams['savefig.transparent'] = False
    rcParams['figure.dpi'] = 500        # for on-screen preview only; irrelevant for vector export
    rcParams['savefig.dpi'] = 'figure'  # dpi setting only matters if you rasterize elements

    # -- Axes --
    rcParams['axes.spines.bottom'] = True
    rcParams['axes.spines.left'] = True
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    rcParams['axes.grid'] = False
    rcParams['axes.linewidth'] = 1.2    # 2pt reads as heavy/cartoonish at print size; 1–1.3 is crisper
    rcParams['axes.labelpad'] = 4
    rcParams['axes.titlepad'] = 6
    rcParams['axes.unicode_minus'] = False  # avoids a missing-glyph minus sign with STIX in some cases

    # -- Ticks and tick labels --
    rcParams['axes.edgecolor'] = 'black'
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
    rcParams['xtick.major.width'] = 1.2
    rcParams['ytick.major.width'] = 1.2
    rcParams['xtick.major.size'] = 4
    rcParams['ytick.major.size'] = 4
    rcParams['xtick.minor.visible'] = False
    rcParams['ytick.minor.visible'] = False
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    rcParams['xtick.labelsize'] = 9
    rcParams['ytick.labelsize'] = 9

    # -- Fonts (match LaTeX) --
    rcParams['font.size'] = 11
    rcParams['font.family'] = 'STIXGeneral'
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['text.color'] = 'black'
    rcParams['axes.labelcolor'] = 'black'
    rcParams['axes.titlesize'] = 12
    rcParams['axes.labelsize'] = 11

    # -- Lines and markers --
    rcParams['lines.linewidth'] = 1.8
    rcParams['lines.markersize'] = 5
    rcParams['lines.markeredgewidth'] = 0.8
    rcParams['lines.solid_capstyle'] = 'round'

    # -- Bars --
    rcParams['patch.linewidth'] = 0.8
    rcParams['patch.edgecolor'] = 'black'
    rcParams['patch.force_edgecolor'] = True   # gives bars a defined edge instead of looking flat/plastic

    # -- Legend --
    rcParams['legend.frameon'] = False
    rcParams['legend.fontsize'] = 9
    rcParams['legend.handlelength'] = 1.5
    rcParams['legend.labelspacing'] = 0.3
    rcParams['legend.borderaxespad'] = 0.5

    # -- Color cycle (colorblind-safe, works for lines/bars/scatter) --
    # Based on Okabe-Ito palette — distinguishable in grayscale and for common CVD types
    rcParams['axes.prop_cycle'] = cycler(color=okabe_ito)

    # -- Figure size --
    rcParams['figure.figsize'] = (6, 4)  # 6x6 is unusually square for most single-panel figures; 3:2 reads better
    rcParams['figure.constrained_layout.use'] = True  # better auto-spacing than tight_layout for multi-panel figs

    # -- Errorbars (common in scientific figures) --
    rcParams['errorbar.capsize'] = 3

setup_plotting_style()
# %%