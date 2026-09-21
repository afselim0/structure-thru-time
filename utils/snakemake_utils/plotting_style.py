#%%
import numpy as np
from matplotlib import rcParams


def set_time_xlim(ax, min_time, times, time_scale):
    """Anchor a time axis at min_time, guarding the log case.

    A log axis cannot show t=0, and matplotlib silently ignores a non-positive
    limit and autoscales instead. min_time comes from rates_min_time, which is
    0 whenever pdf_min_time is, so on a log axis fall back to the first positive
    timepoint -- the shallowest time the axis can actually show.
    """
    times = np.asarray(times, dtype=float)
    left = float(min_time)
    if time_scale == "log" and left <= 0:
        positive = times[times > 0]
        if positive.size == 0:
            return
        left = positive.min()
    ax.set_xlim(left, times.max())


def setup_plotting_style():
    # -- Axes --
    rcParams['axes.spines.bottom'] = True
    rcParams['axes.spines.left'] = True
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    rcParams['axes.grid'] = False
    rcParams['axes.linewidth'] = 2
    # -- Ticks and tick labels --
    rcParams['axes.edgecolor'] = 'black'
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
    rcParams['xtick.major.width'] = 2
    rcParams['ytick.major.width'] = 2
    rcParams['xtick.major.size'] = 5
    rcParams['ytick.major.size'] = 5
    # -- Fonts --
    rcParams['font.size'] = 12
    # rcParams['font.family'] = 'sans-serif'
    # rcParams['font.sans-serif'] = ['DejaVu Sans']
    # rcParams['mathtext.fontset'] = 'dejavusans'
    # plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']

    rcParams['font.family'] = 'STIXGeneral'
    rcParams['mathtext.fontset'] = 'stix'
    rcParams['text.color'] = 'black'
    rcParams['axes.labelcolor'] = 'black'
    # -- Figure size --
    rcParams['figure.figsize'] = (6, 6)
    # -- Saving Options --
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.dpi'] = 500
setup_plotting_style()
# %%
