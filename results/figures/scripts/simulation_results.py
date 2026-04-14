#%%
"""
Script for generation figure describing the embeddings model in the paper
a) Rates as a funciton of time in the sawtooth model
b) Embeddings at two timepoints
c) 3D embeddings at all time points
d) distance between embeddings at different time points
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
from PIL import Image
from plotly import graph_objects as go
from matplotlib.lines import Line2D

from plotting_style import *
sys.path.append("../../../src")
sys.path.append("../../../utils")
from compute_rates_true import compute_true_rates, expand_true_rates
from preprocess_demography import preprocess_demography
from LatentSpaceEmbeddings import LatentSpaceEmbeddings
from PlotEmbeddings import PlotEmbeddings

def compute_distances(embeddings_df):
    embeddings_pairs = embeddings_df.merge(embeddings_df, on="time", suffixes=("_1", "_2"))
    embeddings_pairs = embeddings_pairs[embeddings_pairs["sample_id_1"] < embeddings_pairs["sample_id_2"]]
    embeddings_pairs["distance"] = np.sqrt(
    (embeddings_pairs["X1_1"] - embeddings_pairs["X1_2"])**2 +
    (embeddings_pairs["X2_1"] - embeddings_pairs["X2_2"])**2)

    D_df = (embeddings_pairs
    .set_index(["sample_id_1", "sample_id_2", "time"])["distance"]
    .unstack("time"))

    return D_df
# 