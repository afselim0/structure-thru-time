#%%
import pandas as pd
embeddings = pd.read_parquet("embeddings.parquet")
colors = pd.read_csv("sample_colors.csv")
sample_ids = embeddings["sample_id"].unique()
colors_dict = {}
for sample_id in sample_ids:
    color = colors[colors["sample_id"] == sample_id]["hex_colour"].values[0]
    colors_dict[sample_id] = color
# %%
