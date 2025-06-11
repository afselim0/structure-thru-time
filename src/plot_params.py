from matplotlib import rcParams
import matplotlib.pyplot as plt
def set_custom_rcparams():
    # -- Axes --
    rcParams['axes.spines.bottom'] = True
    rcParams['axes.spines.left'] = True
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    rcParams['axes.grid'] = False
    rcParams['grid.color'] = 'black'
    rcParams['grid.linewidth'] = 0.5
    rcParams['axes.linewidth'] = 1
    # -- Ticks and tick labels --
    rcParams['axes.edgecolor'] = 'black'
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
    rcParams['xtick.major.width'] = 2
    rcParams['ytick.major.width'] = 0
    rcParams['xtick.major.size'] = 5
    rcParams['ytick.major.size'] = 0
    # -- Fonts --
    rcParams['font.size'] = 12
    rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
    rcParams['text.color'] = 'black'
    rcParams['axes.labelcolor'] = 'black'
    # -- Figure size --
    rcParams['figure.figsize'] = (6, 6)
    # -- Saving Options --
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.dpi'] = 500
    rcParams['savefig.transparent'] = False
    # -- Plot Styles --
    rcParams['lines.linewidth'] = 3
set_custom_rcparams()
