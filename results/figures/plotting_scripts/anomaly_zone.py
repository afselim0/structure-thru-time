#%%
import msprime
import tskit
import demes
import numpy as np
from tqdm import tqdm
from results.figures.plotting_style import *
import matplotlib.pyplot as plt
import os
filename = os.path.basename(__file__).split(".")[0]

# Relative paths below are anchored at this script's directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def complex_4_pops(Ne, Ne_bottleneck, Ne_A_recent, Ne_B_recent, Ne_C_recent, C_recent_growth_stop_time, A_B_split_time, AB_C_split_time, ABC_D_split_time):
    # b = lambda x: np.log(2/3 + (5*np.exp(2*x)-2)/(6*(3*np.exp(2*x)-2*np.exp(x))))
    # recent_split_branch_length = (b(distant_split_branch_length/Ne) / 2)*Ne
    # AB_C_split_time = A_B_split_time + recent_split_branch_length
    # ABC_D_split_time = AB_C_split_time + distant_split_branch_length

    growth_rate_A = (1/A_B_split_time)*np.log(Ne_A_recent/Ne_bottleneck)
    growth_rate_B = (1/A_B_split_time)*np.log(Ne_B_recent/Ne_bottleneck)


    C_growth_timepoints = np.logspace(np.log10(C_recent_growth_stop_time),np.log10(AB_C_split_time), num=4)
    growth_rate_C_1 = 1/(C_growth_timepoints[1]-C_growth_timepoints[0])*np.log(Ne_C_recent/Ne_bottleneck)
    growth_rate_C_2 = 1/(C_growth_timepoints[2]-C_growth_timepoints[1])*np.log(Ne_bottleneck/Ne_C_recent)
    growth_rate_C_3 = 1/(C_growth_timepoints[3]-C_growth_timepoints[2])*np.log(Ne_C_recent/Ne_bottleneck)

    demography = msprime.Demography()

    demography.add_population(name="A", initial_size= Ne_A_recent, growth_rate=growth_rate_A, default_sampling_time = 0)
    demography.add_population(name="B", initial_size= Ne_B_recent, growth_rate=growth_rate_B, default_sampling_time = 0)
    demography.add_population(name="C", initial_size= Ne_C_recent, growth_rate=0, default_sampling_time = 0)
    demography.add_population(name="D", initial_size= Ne, growth_rate=0, default_sampling_time = 0)

    demography.add_population(name="AB", initial_size= Ne_bottleneck)
    demography.add_population(name="ABC", initial_size= Ne_bottleneck)
    demography.add_population(name="ABCD", initial_size= Ne)

    demography.add_population_split(time=A_B_split_time, derived=["A", "B"], ancestral="AB")
    demography.add_population_split(time=AB_C_split_time, derived=["AB", "C"], ancestral="ABC")
    demography.add_population_split(time=ABC_D_split_time, derived=["ABC", "D"], ancestral="ABCD")

    demography.add_population_parameters_change(time=C_growth_timepoints[0], population="C", growth_rate=growth_rate_C_1)
    demography.add_population_parameters_change(time=C_growth_timepoints[1], population="C", growth_rate=growth_rate_C_2)
    demography.add_population_parameters_change(time=C_growth_timepoints[2], population="C", growth_rate=growth_rate_C_3)


    demography.sort_events()
    return demography

def preorder_to_topology(preorder, n_leaves=4, leaf_labels = {0: "A", 1: "B", 2: "C", 3: "D"}):
    if leaf_labels is None:
        leaf_labels = {i: i for i in range(n_leaves)}

    pos = [0]

    def parse():
        label = preorder[pos[0]]
        pos[0] += 1
        if label < n_leaves:
            return leaf_labels[label]
        else:
            left = parse()
            right = parse()
            return (left, right)

    result = parse()
    if pos[0] != len(preorder):
        raise ValueError(f"Extra tokens left over after parsing: {preorder[pos[0]:]}")
    return result

# %%
Ne = 3e4
Ne_bottleneck = Ne/10
A_B_split_time = 1000
AB_C_split_time = 1200
ABC_D_split_time = 1400

Ne_A_recent = 2*Ne
Ne_B_recent = 4*Ne
Ne_C_recent = 4*Ne
C_recent_growth_stop_time = 30

demography = complex_4_pops(Ne, Ne_bottleneck, Ne_A_recent, Ne_B_recent, Ne_C_recent, C_recent_growth_stop_time, A_B_split_time, AB_C_split_time, ABC_D_split_time)

num_replicates = int(1e4)
ts_repicates = msprime.sim_ancestry(
    demography=demography,
    samples = {"A":1, "B":1, "C":1, "D":1},
    ploidy=1,
    num_replicates=num_replicates
)
ranks = np.zeros((num_replicates,2), dtype=int)
for i, ts in tqdm(enumerate(ts_repicates)):
    tree = ts.first()
    rank = tree.rank()
    ranks[i,0] = rank.shape
    ranks[i,1] = rank.label
# %%
unique_ranks, counts = np.unique(ranks, return_counts=True, axis=0)
counts_order = np.argsort(counts)[::-1]
unique_ranks = unique_ranks[counts_order, :]
counts = counts[counts_order]
probabilities = counts / num_replicates
topologies = []
for rank in unique_ranks:
    ts = tskit.Tree.unrank(4, tuple(rank))
    topologies.append(preorder_to_topology(ts.preorder()))
# %%
unique_ranks_str = unique_ranks.astype(str)
unique_ranks_str = [",".join(rank) for rank in unique_ranks_str]

# convert nested topology tuples (e.g. (('A','B'),('C','D'))) to Newick-style labels for the x-axis
def topology_to_label(topology):
    if isinstance(topology, tuple):
        return "(" + ",".join(topology_to_label(child) for child in topology) + ")"
    return topology

# canonical form (children sorted) so topologies that differ only in child order still match
def canonical_topology(topology):
    if isinstance(topology, tuple):
        children = [canonical_topology(child) for child in topology]
        return tuple(sorted(children, key=topology_to_label))
    return topology

topology_labels = [topology_to_label(t) for t in topologies]

correct_topology = ("D", ("C", ("A", "B")))
bar_colors = [
    "tab:red" if canonical_topology(t) == canonical_topology(correct_topology) else "tab:blue"
    for t in topologies
]

fig, ax = plt.subplots(figsize=(6, 4), dpi=500)
ax.bar(topology_labels, probabilities, color=bar_colors)
ax.set_xticks(range(len(topology_labels)))
ax.set_xticklabels(topology_labels, rotation=45, ha="right")
for tick_label, color in zip(ax.get_xticklabels(), bar_colors):
    tick_label.set_color(color)
ax.set_ylabel("Probability")
ax.set_xlabel("Topology")
fig.tight_layout()
fig.savefig(f"../sim_figs/{filename}.pdf", dpi=500)
# %%
