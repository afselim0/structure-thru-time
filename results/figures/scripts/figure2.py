#%%
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import sys
import os
import pandas as pd
import os
import demes
import demesdraw
import yaml
import matplotlib.gridspec as gridspec
import matplotlib.colors as colors
from scipy.stats import spearmanr
os.chdir(os.path.dirname(__file__))
sys.path.append("../misc_scripts")
from plotting_style import *

file_name = os.path.basename(__file__).split(".")[0]
# %%
def plot_demography(demes_file_path, pop_colors, max_time, min_time, ax):
    graph = demes.load(demes_file_path)
    pop_colors_dict = dict(zip(pop_colors['population'], pop_colors['color']))
    demesdraw.tubes(
        graph,
        ax=ax,
        colours=pop_colors_dict,
        max_time=max_time,
        log_time=True,
        labels = None
    )
    ax.set_ylim(bottom=min_time)

def parse_config_file(config_file_path):
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    demes_file_path = f"../../../data/demes/main_1/{config['demography']}.yaml"
    output_path  = os.path.join(current_dir,"../../../results/simulations/outputs/", config['demography'])
    pop_colors_path = os.path.join(output_path, "sim/data/pop_colors.csv")
    min_time = config['min_time']
    max_time = config['max_time']
    N_max = config['N_max']


    PDF_TAG    = f".V{config['version']}.FS{1.0}.maxT{config['max_time']}.nT{config['n_timepoints']}"
    RATES_TAG  = f".minWS{config['min_window_size']}.maxWS{config['max_window_size']}.poly{config['poly']}" + PDF_TAG
    STT_TAG    = f".Nmin{config['N_min']}.Nmax{config['N_max']}.L1[None].L2[None]" + RATES_TAG
    plot_tag   = f".minTPlot{config['min_time']}.maxTPlot{config['max_time']}.xylims{config['embeddings_plot_lims']}" + STT_TAG
    embeddings_plot_name   = "embeddings" + plot_tag + "_v1.png"
    sim_static_plot_path   = os.path.join(output_path, "sim/plots/", embeddings_plot_name)
    infer_static_plot_path = os.path.join(output_path, "tsinfer/plots/", embeddings_plot_name)

    sim_rates_path = os.path.join(output_path, "sim/data/", "rates" + RATES_TAG + ".parquet")
    sim_reconstructed_path = os.path.join(output_path, "sim/data/", "reconstructed_rates" + STT_TAG + ".parquet")
    infer_rates_path = os.path.join(output_path, "tsinfer/data/", "rates" + RATES_TAG + ".parquet")
    infer_reconstructed_path = os.path.join(output_path, "tsinfer/data/", "reconstructed_rates" + STT_TAG + ".parquet")

    return demes_file_path, pop_colors_path, N_max, min_time, max_time, sim_static_plot_path, infer_static_plot_path, sim_rates_path, sim_reconstructed_path, infer_rates_path, infer_reconstructed_path

def reconstructed_vs_input_all_times(input_rates, reconstructed_rates, N_max, ax):
    input_rates += 1/N_max
    input_rates.columns = input_rates.columns.astype(float)
    reconstructed_rates.columns = reconstructed_rates.columns.astype(float)

    assert input_rates.shape == reconstructed_rates.shape, "Input and reconstructed rates must have the same shape"
    assert np.all(input_rates.columns.values == reconstructed_rates.columns.values), "Input and reconstructed rates must have the same time points"
    assert input_rates.index.equals(reconstructed_rates.index), "Input and reconstructed rates must have the same samples"
    times = input_rates.columns.astype(float)

    # Normalize times to [0, 1] for the colormap
    norm = colors.Normalize(vmin=min(times), vmax=max(times))
    cmap = cm.viridis
    for t, time in enumerate(times):
        ax.scatter(
            input_rates.iloc[:, t],
            reconstructed_rates.iloc[:, t],
            alpha=0.4,
            s=1,
            color=cmap(norm(time))
        )

    # y=x line for reference
    ax.plot([np.min(input_rates.values), np.max(input_rates.values)], [np.min(input_rates.values), np.max(input_rates.values)], 'k--', lw=2)

    ## Add r^2 value to the plot
    # input_rate_values = input_rates.values.flatten()
    # reconstructed_rate_values = reconstructed_rates.values.flatten()
    # r_squared = np.corrcoef(input_rate_values, reconstructed_rate_values)[0, 1] ** 2
    # corr, p_value = spearmanr(input_rate_values, reconstructed_rate_values)
    # ax.text(0.05, 0.95, f"$r^2$ = {r_squared:.3f}", transform=ax.transAxes, fontsize=8, verticalalignment='top')
    # ax.text(0.05, 0.90, f"Spearman $\\rho$ = \n {corr:.3f}", transform=ax.transAxes, fontsize=8, verticalalignment='top')

# ------ Choosing model to plot ------
model = "../../../data/demes/main_1_config/complex_4_pops_config.yaml"
demes_file_path, pop_colors_path, N_max, min_time, max_time, sim_static_plot_path, infer_static_plot_path, sim_rates_path, sim_reconstructed_path, infer_rates_path, infer_reconstructed_path = parse_config_file(model)
pop_colors = pd.read_csv(pop_colors_path)
sim_input_rates = pd.read_parquet(sim_rates_path)
sim_reconstructed_rates = pd.read_parquet(sim_reconstructed_path)
infer_input_rates = pd.read_parquet(infer_rates_path)
infer_reconstructed_rates = pd.read_parquet(infer_reconstructed_path)

#%%
#----------------------------------------------------------
# Constructing the figure
#----------------------------------------------------------
PLOT_WIDTH  = 6.5
PLOT_HEIGHT = 6.5/3
DPI = 600

fig = plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=DPI)
# gs = gridspec.GridSpec(
#     nrows=2, ncols=4,
#     width_ratios=[0.5, 1.5, 0.5, 0.5],
#     height_ratios=[2/3, 1/3],
#     wspace = 0, hspace = 0.2,
#     figure=fig
# )
gs = gridspec.GridSpec(
    nrows=1, ncols=3,
    width_ratios=[1, 3, 2],
    wspace = 0,
    figure=fig
)

deme_ax = fig.add_subplot(gs[0])      # spans both rows
sim_ax = fig.add_subplot(gs[1])      # spans both rows
# infer_ax = fig.add_subplot(gs[0, 2:4])    # top row, spans 2 columns
sim_eval_ax = fig.add_subplot(gs[2])      # bottom row, col 2
# infer_eval_ax = fig.add_subplot(gs[1, 3])      # bottom row, col 3

# Adding demeography plot
plot_demography(demes_file_path, pop_colors, max_time, min_time, deme_ax)
deme_ax.grid(False)
deme_ax.spines['left'].set_visible(True)
deme_ax.set_ylabel("Time (generations ago)")

# Adding static plots
sim_img = plt.imread(sim_static_plot_path)
infer_img = plt.imread(infer_static_plot_path)
sim_ax.imshow(sim_img)
# infer_ax.imshow(infer_img)
sim_ax.axis('off')
# infer_ax.axis('off')

# Adding evaluation plots
reconstructed_vs_input_all_times(sim_input_rates, sim_reconstructed_rates, N_max, sim_eval_ax)
sim_eval_ax.set_xlabel(r"input rates")
sim_eval_ax.set_ylabel(r"reconstructed rates")
sim_eval_ax.yaxis.tick_right()
sim_eval_ax.yaxis.set_label_position("right")

sim_eval_ax.ticklabel_format(axis='both', style='sci', scilimits=(0, 0), useMathText=True)
# Colorbar lenged
# times = sim_input_rates.columns.astype(float)
# cmap = cm.viridis
# norm = colors.Normalize(vmin=min(times), vmax=max(times))
# sm = cm.ScalarMappable(norm=norm, cmap=cmap)

# cbar = fig.colorbar(sm, ax=sim_eval_ax, pad = 0.25)
# cbar.set_label("time", loc = "bottom")
# cbar.formatter = mticker.ScalarFormatter(useMathText=True)
# cbar.formatter.set_powerlimits((0, 0))  # force scientific notation
# cbar.update_ticks()
# %%
fig.savefig(f"../outputs/{file_name}.png", dpi=DPI)
# %%
