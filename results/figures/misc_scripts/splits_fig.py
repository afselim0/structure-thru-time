#%%
import matplotlib.pyplot as plt
from results.figures.misc_scripts.plotting_style import *
import numpy as np
import pandas as pd
import os
import demes
import demesdraw
import yaml
import matplotlib.gridspec as gridspec

def plot_demography(demes_file_path, pop_colors, max_time, min_time, ax):
    setup_plotting_style()
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


    PDF_TAG    = f".V{config['version']}.FS{1.0}.maxT{config['max_time']}.nT{config['n_timepoints']}"
    RATES_TAG  = f".minWS{config['min_window_size']}.maxWS{config['max_window_size']}.poly{config['poly']}" + PDF_TAG
    STT_TAG    = f".Nmin{config['N_min']}.Nmax{config['N_max']}.L1[None].L2[None]" + RATES_TAG
    plot_tag   = f".minTPlot{config['min_time']}.maxTPlot{config['max_time']}.xylims{config['embeddings_plot_lims']}" + STT_TAG
    embeddings_plot_name   = "embeddings" + plot_tag + "_v1.png"
    sim_static_plot_path   = os.path.join(output_path, "sim/plots/", embeddings_plot_name)
    infer_static_plot_path = os.path.join(output_path, "tsinfer/plots/", embeddings_plot_name)

    return demes_file_path, pop_colors_path, min_time, max_time, sim_static_plot_path, infer_static_plot_path

# ------ Choosing model to plot ------
model = "../../../data/demes/main_1_config/complex_4_pops_config.yaml"
demes_file_path, pop_colors_path, min_time, max_time, sim_static_plot_path, infer_static_plot_path = parse_config_file(model)
pop_colors = pd.read_csv(pop_colors_path)

#----------------------------------------------------------
# Constructing the figure
#----------------------------------------------------------
PLOT_WIDTH  = 7
PLOT_HEIGHT = 7/3
DPI = 600

fig = plt.figure(figsize=(PLOT_WIDTH, PLOT_HEIGHT), dpi=DPI)
gs = gridspec.GridSpec(
    nrows=2, ncols=4,
    width_ratios=[0.5, 1.5, 0.5, 0.5],
    height_ratios=[2/3, 1/3],
    wspace = 0, hspace = 0.2,
    figure=fig
)

deme_ax = fig.add_subplot(gs[:, 0])      # spans both rows
sim_ax = fig.add_subplot(gs[:, 1])      # spans both rows
infer_ax = fig.add_subplot(gs[0, 2:4])    # top row, spans 2 columns
sim_eval_ax = fig.add_subplot(gs[1, 2])      # bottom row, col 2
infer_eval_ax = fig.add_subplot(gs[1, 3])      # bottom row, col 3

# Adding demeography plot
plot_demography(demes_file_path, pop_colors, max_time, min_time, deme_ax)
deme_ax.grid(False)
deme_ax.spines['left'].set_visible(True)
deme_ax.set_ylabel("Time (generations ago)")

# Adding static plots
sim_img = plt.imread(sim_static_plot_path)
infer_img = plt.imread(infer_static_plot_path)
sim_ax.imshow(sim_img)
infer_ax.imshow(infer_img)
sim_ax.axis('off')
infer_ax.axis('off')

# Adding evaluation plots
# %%
