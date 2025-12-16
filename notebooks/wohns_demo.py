#%%
import numpy as np
import pandas as pd
from tqdm import tqdm
import tskit
import sys
sys.path.append("../src/")
import seaborn as sns
import matplotlib.pyplot as plt
from pathos.multiprocessing import ProcessingPool as Pool

from CountCoalEvents import CountCoalEvents
from LatentSpaceEmbeddings import LatentSpaceEmbeddings
from PlotEmbeddings import PlotEmbeddings
#%%
## Loading Data
### Printing all population label in Wohns dataset
population_labels = pd.read_csv("../data/trees/wohns/metadata.csv")
populations = population_labels["population"].unique()
print(populations)
#%%
### Computing counts for samples with a set of labels
chosen_populations = ["YRI","CEU","CHB","JPT"] # Adjust as needed
# chosen_populations = ["JPT"] # Adjust as needed
sample_ids = population_labels[population_labels["population"].isin(chosen_populations)]["sample_id"].values
population_sample_dict = {pop: population_labels[population_labels["population"] == pop]["sample_id"].values for pop in chosen_populations}

output_dir = "../tests/wohns/OOA4"
ts_dir = "../data/trees/wohns/"
time_points = np.logspace(1, np.log10(20000), 200, base=10)
window_sizes = np.array([100]*len(time_points))

CoalCounter = CountCoalEvents(
    output_dir = output_dir,
    ts_dir = ts_dir,
    time_points = time_points,
    window_sizes = window_sizes,
    sample_ids = sample_ids,
    num_posterior_samples = None,
    thinning_fraction = 4
)
# CoalCounter.count_coal_events_total(num_threads=4)
# aggregated_counts = CoalCounter.aggregate_counts()
#%%
N,L = CoalCounter.count_coal_events_contig(CoalCounter.ts_files[20])
# %%
