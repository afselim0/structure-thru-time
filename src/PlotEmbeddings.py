#%%
# Sample_id should be individual id in the case of individuals=True
import numpy as np
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
import plotly.express as px
import plotly.io as pio

# -- Axes --
rcParams['axes.spines.bottom'] = True
rcParams['axes.spines.left'] = True
rcParams['axes.spines.right'] = False
rcParams['axes.spines.top'] = False
rcParams['axes.grid'] = False
# -- Ticks and tick labels --
rcParams['axes.edgecolor'] = 'black'
rcParams['xtick.color'] = 'black'
rcParams['ytick.color'] = 'black'
rcParams['xtick.major.width'] = 2
rcParams['ytick.major.width'] = 2
rcParams['xtick.major.size'] = 5
rcParams['ytick.major.size'] = 5
# -- Fonts --
rcParams['font.size'] = 10
rcParams['font.family'] = 'serif'
# plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'sans-serif']
rcParams['text.color'] = 'black'
rcParams['axes.labelcolor'] = 'black'
# -- Figure size --
rcParams['figure.figsize'] = (6, 6)
# -- Saving Options --
rcParams['savefig.bbox'] = 'tight'
rcParams['savefig.dpi'] = 500
# %%
class PlotEmbeddings:
    def __init__(self, 
                 embeddings_df: pd.DataFrame,
                 idx_to_sample_df: pd.DataFrame,
                 sample_to_pop_df: pd.DataFrame,
                 individuals: bool = False,
                 ):
        if individuals:
            assert "individual_id" in idx_to_sample_df.columns, "idx_to_sample_df must contain 'individual_id' column when individuals=True"
            idx_to_sample_df = idx_to_sample_df.drop(columns=["sample_id"]).rename(columns={"individual_id": "sample_id"})
            idx_to_sample_df = idx_to_sample_df[idx_to_sample_df["idx"] % 2 == 0]
            idx_to_sample_df["idx"] = idx_to_sample_df["idx"] // 2
            
        embeddings_df = embeddings_df.merge(
            idx_to_sample_df,
            on='idx',
            how='left'
        ).merge(
            sample_to_pop_df,
            on='sample_id',
            how='left'
        )
        self.num_ancestries = embeddings_df["type"].nunique()
        self.num_populations = embeddings_df["population"].nunique()
        self.populations = embeddings_df["population"].unique()
        if "color" not in embeddings_df.columns:
            if self.num_ancestries < 10:
                cmap = plt.get_cmap("Set1")
                colors = cmap(np.arange(self.num_populations))[:,:-1]
                colors_dict = {label:mcolors.to_hex(colors[i]) for i,label in enumerate(self.populations)}
            else:
                cmap = plt.get_cmap("rainbow")
                colors = cmap(np.linspace(0,255,self.num_populations).astype(int))[:,:-1]
                colors_dict = {label:mcolors.to_hex(colors[i]) for i,label in enumerate(self.populations)}
            embeddings_df["color"] = embeddings_df["population"].map(colors_dict)

        self.colors_dict = colors_dict
        self.embeddings_df = embeddings_df

    def dynamic_plot(self,log_time: bool = False):
        fig = px.line_3d(self.embeddings_df, x="X1", y="X2", z="time", color='population', line_group='idx', color_discrete_map=self.colors_dict,log_z=log_time)
        # Set xlim and ylim to be the same
        x_min, x_max = self.embeddings_df["X1"].min(), self.embeddings_df["X1"].max()
        y_min, y_max = self.embeddings_df["X2"].min(), self.embeddings_df["X2"].max()
        lim_min = min(x_min, y_min)
        lim_max = max(x_max, y_max)
        fig.update_layout(scene=dict(
            xaxis=dict(range=[lim_min, lim_max]),
            yaxis=dict(range=[lim_min, lim_max])
        ))
        fig.update_layout(showlegend=True)
        # pio.write_html(fig, file=os.path.join(self.output_dir,"dynamic_3d_plot.html"))
        return fig
