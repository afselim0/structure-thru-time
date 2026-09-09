"""
Figure data: anomaly zone.

Simulates the four-population demography and tabulates how often each unlabelled
tree topology comes up, so the figure only has to draw the bars. The most probable
topology is not the species tree in this parameter regime -- that is the anomaly
the figure shows -- so the correct topology is recorded alongside the counts.
"""
import os
import sys

import numpy as np
import pandas as pd
import msprime
import tskit
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import out_dir, report, write_table

FIGURE = "anomaly_zone"

# demography parameters (unchanged from the figure this replaces)
Ne = 3e4
Ne_bottleneck = Ne / 10
A_B_split_time = 1000
AB_C_split_time = 1200
ABC_D_split_time = 1400
Ne_A_recent = 2 * Ne
Ne_B_recent = 4 * Ne
Ne_C_recent = 4 * Ne
C_recent_growth_stop_time = 30

NUM_REPLICATES = int(1e4)
# the replicates are drawn from a seeded generator so the tabulated probabilities
# are reproducible; the original figure left this unseeded
RANDOM_SEED = 42

CORRECT_TOPOLOGY = ("D", ("C", ("A", "B")))


def complex_4_pops(Ne, Ne_bottleneck, Ne_A_recent, Ne_B_recent, Ne_C_recent,
                   C_recent_growth_stop_time, A_B_split_time, AB_C_split_time,
                   ABC_D_split_time):
    growth_rate_A = (1 / A_B_split_time) * np.log(Ne_A_recent / Ne_bottleneck)
    growth_rate_B = (1 / A_B_split_time) * np.log(Ne_B_recent / Ne_bottleneck)

    C_growth_timepoints = np.logspace(
        np.log10(C_recent_growth_stop_time), np.log10(AB_C_split_time), num=4)
    growth_rate_C_1 = 1 / (C_growth_timepoints[1] - C_growth_timepoints[0]) * np.log(Ne_C_recent / Ne_bottleneck)
    growth_rate_C_2 = 1 / (C_growth_timepoints[2] - C_growth_timepoints[1]) * np.log(Ne_bottleneck / Ne_C_recent)
    growth_rate_C_3 = 1 / (C_growth_timepoints[3] - C_growth_timepoints[2]) * np.log(Ne_C_recent / Ne_bottleneck)

    demography = msprime.Demography()

    demography.add_population(name="A", initial_size=Ne_A_recent, growth_rate=growth_rate_A, default_sampling_time=0)
    demography.add_population(name="B", initial_size=Ne_B_recent, growth_rate=growth_rate_B, default_sampling_time=0)
    demography.add_population(name="C", initial_size=Ne_C_recent, growth_rate=0, default_sampling_time=0)
    demography.add_population(name="D", initial_size=Ne, growth_rate=0, default_sampling_time=0)

    demography.add_population(name="AB", initial_size=Ne_bottleneck)
    demography.add_population(name="ABC", initial_size=Ne_bottleneck)
    demography.add_population(name="ABCD", initial_size=Ne)

    demography.add_population_split(time=A_B_split_time, derived=["A", "B"], ancestral="AB")
    demography.add_population_split(time=AB_C_split_time, derived=["AB", "C"], ancestral="ABC")
    demography.add_population_split(time=ABC_D_split_time, derived=["ABC", "D"], ancestral="ABCD")

    demography.add_population_parameters_change(time=C_growth_timepoints[0], population="C", growth_rate=growth_rate_C_1)
    demography.add_population_parameters_change(time=C_growth_timepoints[1], population="C", growth_rate=growth_rate_C_2)
    demography.add_population_parameters_change(time=C_growth_timepoints[2], population="C", growth_rate=growth_rate_C_3)

    demography.sort_events()
    return demography


def preorder_to_topology(preorder, n_leaves=4, leaf_labels=None):
    """Turn tskit's preorder node list into a nested (left, right) tuple of leaf names."""
    if leaf_labels is None:
        leaf_labels = {0: "A", 1: "B", 2: "C", 3: "D"}

    pos = [0]

    def parse():
        label = preorder[pos[0]]
        pos[0] += 1
        if label < n_leaves:
            return leaf_labels[label]
        left = parse()
        right = parse()
        return (left, right)

    result = parse()
    if pos[0] != len(preorder):
        raise ValueError(f"Extra tokens left over after parsing: {preorder[pos[0]:]}")
    return result


def topology_to_label(topology):
    """Newick-style label, e.g. (D,(C,(A,B)))."""
    if isinstance(topology, tuple):
        return "(" + ",".join(topology_to_label(child) for child in topology) + ")"
    return topology


def canonical_topology(topology):
    """Children sorted, so topologies differing only in child order compare equal."""
    if isinstance(topology, tuple):
        children = [canonical_topology(child) for child in topology]
        return tuple(sorted(children, key=topology_to_label))
    return topology


def main():
    destination = out_dir(FIGURE)

    demography = complex_4_pops(
        Ne, Ne_bottleneck, Ne_A_recent, Ne_B_recent, Ne_C_recent,
        C_recent_growth_stop_time, A_B_split_time, AB_C_split_time, ABC_D_split_time)

    replicates = msprime.sim_ancestry(
        demography=demography,
        samples={"A": 1, "B": 1, "C": 1, "D": 1},
        ploidy=1,
        num_replicates=NUM_REPLICATES,
        random_seed=RANDOM_SEED,
    )

    # each tree's unlabelled shape is summarised by tskit's (shape, label) rank
    ranks = np.zeros((NUM_REPLICATES, 2), dtype=int)
    for i, ts in tqdm(enumerate(replicates), total=NUM_REPLICATES):
        rank = ts.first().rank()
        ranks[i, 0] = rank.shape
        ranks[i, 1] = rank.label

    unique_ranks, counts = np.unique(ranks, return_counts=True, axis=0)
    order = np.argsort(counts)[::-1]          # most frequent topology first
    unique_ranks = unique_ranks[order, :]
    counts = counts[order]

    topologies = [preorder_to_topology(tskit.Tree.unrank(4, tuple(rank)).preorder())
                  for rank in unique_ranks]
    correct = canonical_topology(CORRECT_TOPOLOGY)

    table = pd.DataFrame({
        "rank_shape": unique_ranks[:, 0],
        "rank_label": unique_ranks[:, 1],
        "topology": [topology_to_label(t) for t in topologies],
        "count": counts,
        "probability": counts / NUM_REPLICATES,
        "is_correct_topology": [canonical_topology(t) == correct for t in topologies],
    })

    params = pd.DataFrame([
        {"parameter": "Ne", "value": Ne},
        {"parameter": "Ne_bottleneck", "value": Ne_bottleneck},
        {"parameter": "Ne_A_recent", "value": Ne_A_recent},
        {"parameter": "Ne_B_recent", "value": Ne_B_recent},
        {"parameter": "Ne_C_recent", "value": Ne_C_recent},
        {"parameter": "C_recent_growth_stop_time", "value": C_recent_growth_stop_time},
        {"parameter": "A_B_split_time", "value": A_B_split_time},
        {"parameter": "AB_C_split_time", "value": AB_C_split_time},
        {"parameter": "ABC_D_split_time", "value": ABC_D_split_time},
        {"parameter": "num_replicates", "value": NUM_REPLICATES},
        {"parameter": "seed", "value": RANDOM_SEED},
    ])

    written = [
        write_table(table, os.path.join(destination, "topology_probabilities.csv"), index=False),
        write_table(params, os.path.join(destination, "params.csv"), index=False),
    ]
    report(written)


if __name__ == "__main__":
    main()
