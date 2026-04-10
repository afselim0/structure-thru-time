#%%
from matplotlib import rcParams
def setup_plotting_style():
    # -- Axes --
    rcParams['axes.spines.bottom'] = True
    rcParams['axes.spines.left'] = True
    rcParams['axes.spines.right'] = False
    rcParams['axes.spines.top'] = False
    rcParams['axes.grid'] = False
    rcParams['axes.linewidth'] = 1
    # -- Ticks and tick labels --
    rcParams['axes.edgecolor'] = 'black'
    rcParams['xtick.color'] = 'black'
    rcParams['ytick.color'] = 'black'
    rcParams['xtick.major.width'] = 1
    rcParams['ytick.major.width'] = 1
    rcParams['xtick.major.size'] = 5
    rcParams['ytick.major.size'] = 5
    # -- Fonts --
    rcParams['font.size'] = 12
    rcParams['font.family'] = 'sans-serif'
    # rcParams['font.sans-serif'] = ['DejaVu Sans']
    # rcParams['mathtext.fontset'] = 'dejavusans'
    rcParams['font.sans-serif'] = ['Helvetica']
    rcParams['text.color'] = 'black'
    rcParams['axes.labelcolor'] = 'black'
    # -- Figure size --
    rcParams['figure.figsize'] = (6, 6)
    # -- Saving Options --
    rcParams['savefig.bbox'] = 'tight'
    rcParams['savefig.dpi'] = 500

setup_plotting_style()
# %%
