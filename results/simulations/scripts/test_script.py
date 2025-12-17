#%%
import demes
import msprime
from preprocess_demography import preprocess_demography
from plot_demography import plot_demography
from simulate_trees import sim_contigs
from compute_rates_true import compute_true_rates, expand_true_rates
from plot_rates import plot_rates
from run_pca import run_pca,plot_pca

demes_file_path = "../../../data/demes/main/split.yaml"
graph = demes.load(demes_file_path)
demography = msprime.Demography.from_demes(graph)
max_time = int(2*demography.events[-1].time)
n_samples_per_pop = 20
delta_t = 10

pop_labels, pop_colors = preprocess_demography(demes_file_path, n_samples_per_pop)
demography_fig = plot_demography(demes_file_path, pop_colors, max_time=max_time)
pca_df = run_pca(demes_file_path, n_samples_per_pop, sequence_length=1e7, recombination_rate=1e-8, num_components=2)
pca_fig = plot_pca(pca_df, pop_labels, pop_colors)
pop_rates_df = compute_true_rates(demes_file_path, max_time=max_time, delta_t=delta_t)
sample_rates_df = expand_true_rates(pop_rates_df, pop_labels)
rates_true_fig = plot_rates(sample_rates_df, pop_colors, pop_labels)
sim_contigs(demes_file_path, "../outputs/split/trees/sim", pop_labels)
# %%
