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
#%%
#######################################
# ------ Choosing models to plot ------
#######################################

# a) Split
split = "../../../data/demes/main_1_config/split_config.yaml"

# b) Split with migration
split_mig = "../../../data/demes/main_1_config/split_mig_Nm_01_config.yaml"

# c) Split with sawtooth population size change
split_sawtooth = "../../../data/demes/main_1_config/split_sawtooth_config.yaml"

# d1) Migration change
mig_change = "../../../data/demes/main_1_config/mig_change_config.yaml"

# d2) anomaly zone
anomaly_zone = "../../../data/demes/main_1_config/anomaly_zone_config.yaml"

# examples_to_plot = [split, split_mig, split_sawtooth, mig_change]
examples_to_plot = [split, split, split, split]
#%%
#----------------------------------------------------------
# Constructing the figure
#----------------------------------------------------------
PAGE_WIDTH  = 7
PAGE_HEIGHT = 8
DPI = 600

fig = plt.figure(figsize=(PAGE_WIDTH, PAGE_HEIGHT), dpi=DPI)
gs = gridspec.GridSpec(
    nrows=len(examples_to_plot), ncols=4,
    width_ratios=[1, 2.5, 2.5, 1],
    height_ratios=[PAGE_HEIGHT/len(examples_to_plot)]*len(examples_to_plot),
    wspace = 0, hspace = 0.2
)

for i,example in enumerate(examples_to_plot):
    # Defining the axes for the current example
    ax_demography = fig.add_subplot(gs[i, 0])
    ax_sim_plot = fig.add_subplot(gs[i, 1])
    ax_infer_plot = fig.add_subplot(gs[i, 2])
    ax_analysis = fig.add_subplot(gs[i, 3])
    if i == 0:
        ax_demography.set_title("Demography", fontsize=10)
        ax_sim_plot.set_title("Simulated Trees", fontsize=10)
        ax_infer_plot.set_title("Inferred Trees", fontsize=10)
        ax_analysis.set_title("Distance between clusters", fontsize=10)

    demes_file_path, pop_colors_path, min_time, max_time, sim_static_plot_path, infer_static_plot_path = parse_config_file(example)
    pop_colors = pd.read_csv(pop_colors_path)
    plot_demography(demes_file_path, pop_colors, max_time, min_time, ax_demography)
    ax_demography.set_ylabel("Time (generations)", fontsize=10)
    ax_demography.tick_params(axis='y', labelsize=10)
    try:
        sim_static_plot = plt.imread(sim_static_plot_path)
        infer_static_plot = plt.imread(infer_static_plot_path)

        ax_sim_plot.imshow(sim_static_plot)
        ax_sim_plot.axis('off')
        ax_infer_plot.imshow(infer_static_plot)
        ax_infer_plot.axis('off')
    except FileNotFoundError:
        print(f"Warning: Could not find static plot for {example}. Skipping.")
        continue

    ax_analysis.invert_xaxis()
    ax_analysis.yaxis.set_ticks_position('right')
    ax_analysis.yaxis.set_label_position('right')
    ax_analysis.spines['left'].set_visible(False)
    ax_analysis.spines['right'].set_visible(True)
    ax_analysis.set_xticklabels([])
    ax_analysis.set_yticklabels([])
    ax_analysis.set_xticks([])
    ax_analysis.set_yticks([])

    ax_analysis.set_xlabel("distance", fontsize=10)
    ax_analysis.set_ylabel("Time (generations ago)", fontsize=10)

fig.savefig("../outputs/sim_results.png", dpi=DPI, bbox_inches='tight')
# %%