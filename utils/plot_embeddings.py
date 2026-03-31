#%%
#!/usr/bin/env python3
import argparse
import sys
import pandas as pd
sys.path.append("../../src")
sys.path.append("../src")
sys.path.append("src")
# import support_tube

from PlotEmbeddings import PlotEmbeddings

def main():
    parser = argparse.ArgumentParser(description="Plot embeddings")
    parser.add_argument("--input", required=True, help="Path to embeddings parquet file")
    parser.add_argument("--colors", required=True, help="Path to population colors parquet file")
    parser.add_argument("--labels", required=True, help="Path to population labels parquet file")
    parser.add_argument("--output_path", required=True, help="Output directory for plots")
    parser.add_argument("--output_name", required=True, help="Output filename for plot")
    parser.add_argument("--min_time", type=float,help="Minimum time for plotting")
    parser.add_argument("--max_time", type=float,help="Maximum time for plotting")
    parser.add_argument("--lims", type=str, help="Minimum time for plotting")
    parser.add_argument("--generation_time", default=None, type=float,help="Generation time in years")

    args = parser.parse_args()
    embeddings_df = pd.read_parquet(args.input)
    colors_df = pd.read_csv(args.colors)
    labels_df = pd.read_csv(args.labels)
    output_path = args.output_path
    output_name = args.output_name
    min_time = args.min_time
    max_time = args.max_time
    generation_time = args.generation_time
    if args.lims == "None":
        lims = None
    else:
        lims = float(args.lims)
    
    # subsetting labels_df and colors_df to active samples only
    active_samples = embeddings_df.sample_id.unique()
    labels_df = labels_df[labels_df['sample_id'].isin(active_samples)]
    populations = labels_df['population'].unique()
    colors_df = colors_df[colors_df['population'].isin(populations)]

    output_basename = output_name.removesuffix(".html")
    print(f"Building Embeddings plots for embeddings file: {args.input}")

    # embeddings_df = embeddings_df[embeddings_df['time'] >= min_time]
    plotter = PlotEmbeddings(embeddings_df = embeddings_df,labels_df = labels_df,colors_df = colors_df)#, generation_time = generation_time)
    plotter.init_plot(time_scale="log",xy_lims=lims, min_time=min_time, max_time=max_time)
    plotter.add_sample_trajectories()
    # plotter.add_population_medians()
    plotter.add_legend()
    plotter.fig.show()
    plotter.save_static(filename=f"{output_path}/{output_basename}_v1", format="png", width=600, height=600, scale=2, eye=dict(x=1.8, y=1.8, z=0.3), up=dict(x=0, y=0, z=1))
    plotter.save_static(filename=f"{output_path}/{output_basename}_v2", format="png", width=600, height=600, scale=2, eye=dict(x=-0.75, y=-1.5, z=0.3), up=dict(x=0, y=0, z=1))
    plotter.fig.update_layout(title_text=output_basename)
    plotter.fig.write_html(f"{output_path}/{output_basename}.html")
    if len(populations) <=5:
        plotter.add_support_tube()
        plotter.fig.write_html(f"{output_path}/convexhull_{output_basename}.html")

        plotter = PlotEmbeddings(embeddings_df = embeddings_df,labels_df = labels_df,colors_df = colors_df)#, generation_time = generation_time)
        plotter.init_plot(time_scale="log",xy_lims=lims, min_time=min_time, max_time=max_time)
        plotter.add_support_tube()
        plotter.fig.write_html(f"{output_path}/convexhullONLY_{output_basename}.html")

    
if __name__ == "__main__":
    main()

# %%
embeddings_df = pd.read_parquet("../results/empirical/Wohns/all.npp.2/data/embeddings.Nmin1000.Nmax10000000.L1[None].L2[None].minWS100.maxWS1500.V1.FS1.0.maxT4000.nT4000.parquet")
colors_df = pd.read_csv("../data/metadata/wohns_pop_colors.csv")
labels_df = pd.read_csv("../data/metadata/wohns_pop_labels.csv")
output_path = "../results/empirical/Wohns/all.npp.2/plots"
output_name = "poster_plot"
lims = 7e6
generation_time = 28
min_time = 50 * generation_time
max_time = 4000 * generation_time

active_samples = embeddings_df.sample_id.unique()
labels_df = labels_df[labels_df['sample_id'].isin(active_samples)]
populations = labels_df['population'].unique()
colors_df = colors_df[colors_df['population'].isin(populations)]

# output_basename = output_name.removesuffix(".html")

# embeddings_df = embeddings_df[embeddings_df['time'] >= min_time]
# plotter = PlotEmbeddings(embeddings_df = embeddings_df,labels_df = labels_df,colors_df = colors_df, generation_time = generation_time)
# plotter.init_plot(time_scale="log",xy_lims=lims, min_time=min_time, max_time=max_time)
# plotter.add_sample_trajectories()
# # plotter.add_population_medians()
# plotter.add_legend()


# z_ticks = [2000, 10000, 60000, 100000]
# plotter.fig.update_layout(
#     scene=dict(
#     zaxis=dict(tickvals=z_ticks, ticktext=[str(int(z)) for z in z_ticks]
# )))
# plotter.fig.show()
# plotter.save_static(filename=f"{output_path}/{output_basename}_v1", format="png", width=600, height=600, scale=2, eye=dict(x=1.8, y=1.8, z=0.3), up=dict(x=0, y=0, z=1))
# plotter.save_static(filename=f"{output_path}/{output_basename}_v2", format="png", width=600, height=600, scale=2, eye=dict(x=-0.75, y=-1.5, z=0.3), up=dict(x=0, y=0, z=1))
# plotter.fig.write_html(f"{output_path}/{output_basename}.html")

# %%
colors_labels_df = colors_df.merge(labels_df, on="population", how="inner")
import matplotlib.pyplot as plt
times = [20, 500, 2500]
fig, axs = plt.subplots(1,3, figsize=(15,5), dpi=500, sharex=True, sharey=True)
for time in times:
    subset = embeddings_df[embeddings_df['time'] == time]
    axs[times.index(time)].scatter(subset['X1'], subset['X2'], c=subset['sample_id'].map(colors_labels_df.set_index('sample_id')['color']), s=10)
    axs[times.index(time)].set_title(f"time={time*generation_time} years ago", fontsize=24)

subset = embeddings_df[embeddings_df['time'] == times[0]]
for idx, row in subset.iterrows():
    label = colors_labels_df[colors_labels_df['sample_id'] == row['sample_id']]['population'].values[0][:3]
    sample_id = row['sample_id']
    axs[0].text(row['X1'], row['X2'], label, fontsize=8, alpha=0.7, c=subset['sample_id'].map(colors_labels_df.set_index('sample_id')['color'])[idx])
plt.tight_layout()

#%%
fig, ax = plt.subplots(1, figsize=(3,3), dpi=500)
time = times[2]
subset = embeddings_df[embeddings_df['time'] == time]
ax.scatter(subset['X1'], subset['X2'], c=subset['sample_id'].map(colors_labels_df.set_index('sample_id')['color']), s=10)
for idx, row in subset.iterrows():
    label = colors_labels_df[colors_labels_df['sample_id'] == row['sample_id']]['population'].values[0][:3]
    sample_id = row['sample_id']
    ax.text(row['X1'], row['X2'], label, fontsize=8, alpha=0.7, c=subset['sample_id'].map(colors_labels_df.set_index('sample_id')['color'])[idx])
ax.tick_params(axis='both', labelsize=8)
ax.xaxis.set_major_locator(plt.MaxNLocator(3))
ax.yaxis.set_major_locator(plt.MaxNLocator(3))

# %%
## Replotting SS
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors
# embeddings_old = pd.read_parquet("../../structure_thru_time_older/results/SS_50_1/embeddings.parquet")
# embeddings_df = pd.DataFrame(np.zeros((10000,4)), columns=["sample_id", "time", "X1", "X2"])
# embeddings_df["sample_id"] = embeddings_old[embeddings_old.component == 0]["sample_id"].values
# embeddings_df["time"] = embeddings_old[embeddings_old.component == 0]["time"].values
# embeddings_df["X1"] = embeddings_old[embeddings_old.component == 0]["embedding"].values
# embeddings_df["X2"] = embeddings_old[embeddings_old.component == 1]["embedding"].values

# sample_ids = np.arange(100)
# pop_ids = np.repeat(np.array(["pop"+str(i) for i in range(50)]), 2)
# pop_labels = pd.DataFrame({"sample_id": sample_ids, "population": pop_ids})

# cmap = plt.get_cmap("rainbow")(np.linspace(0,1,50))
# colors = [mcolors.to_hex(cmap[i]) for i in range(50)]

# pop_colors = pd.DataFrame({"population": [f"pop{i}" for i in range(50)], "color": colors})

# plotter = PlotEmbeddings(embeddings_df = embeddings_df,labels_df = pop_labels,colors_df = pop_colors)
# plotter.init_plot(time_scale="log")
# plotter.add_sample_trajectories()
# plotter.fig.write_html(f"SS_50_1_plot.html")
# # %%
