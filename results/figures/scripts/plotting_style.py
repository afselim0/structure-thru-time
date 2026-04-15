from matplotlib import rcParams
import matplotlib.pyplot as plt

def setup_plotting_style():
    # ---------------------------
    # Figure size (A4 / journal friendly)
    # ---------------------------
    rcParams['figure.figsize'] = (7, 4.5)   # page width and a reasonable height
    rcParams['figure.dpi'] = 100

    # ---------------------------
    # Axes (clean + seamless)
    # ---------------------------
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.left'] = False
    rcParams['axes.spines.bottom'] = False

    rcParams['axes.facecolor'] = 'white'
    rcParams['axes.edgecolor'] = 'black'
    rcParams['axes.linewidth'] = 0.8

    # Subtle ggplot-style grid
    rcParams['axes.grid'] = True
    rcParams['grid.color'] = '#E5E5E5'
    rcParams['grid.linewidth'] = 0.8
    rcParams['grid.linestyle'] = '-'
    rcParams['axes.axisbelow'] = True

    # ---------------------------
    # Ticks (minimal)
    # ---------------------------
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'

    rcParams['xtick.major.size'] = 3
    rcParams['ytick.major.size'] = 3
    rcParams['xtick.major.width'] = 0.8
    rcParams['ytick.major.width'] = 0.8

    rcParams['xtick.minor.size'] = 0
    rcParams['ytick.minor.size'] = 0

    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'

    # ---------------------------
    # Fonts (LaTeX-consistent)
    # ---------------------------
    rcParams['text.usetex'] = True   # <-- requires LaTeX installed
    rcParams['font.family'] = 'serif'

    # Match typical LaTeX document scaling
    rcParams['font.size'] = 9
    rcParams['axes.labelsize'] = 9
    rcParams['axes.titlesize'] = 10
    rcParams['xtick.labelsize'] = 8
    rcParams['ytick.labelsize'] = 8
    rcParams['legend.fontsize'] = 8

    rcParams['text.color'] = 'black'
    rcParams['axes.labelcolor'] = 'black'

    # ---------------------------
    # Lines & markers
    # ---------------------------
    rcParams['lines.linewidth'] = 1.5
    rcParams['lines.markersize'] = 4

    # ggplot-like color cycle
    rcParams['axes.prop_cycle'] = plt.cycler(color=[
        '#4C72B0',  # blue
        '#55A868',  # green
        '#C44E52',  # red
        '#8172B2',  # purple
        '#CCB974',  # mustard
        '#64B5CD'   # cyan
    ])

    # ---------------------------
    # Legend
    # ---------------------------
    rcParams['legend.frameon'] = False

    # ---------------------------
    # Savefig (publication quality)
    # ---------------------------
    rcParams['savefig.dpi'] = 600
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.pad_inches'] = 0.02

setup_plotting_style()

# %%
