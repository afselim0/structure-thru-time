from tqdm import tqdm
import numpy as np # type: ignore
import pandas as pd
from numba import njit

@njit
def update_matrices(N, C, pair, tmrca, times, times_upper):
    i, j = pair
    for k in range(len(times)):
        if tmrca > times[k]: # Has not coalesced yet
            if tmrca <= times_upper[k]: # Within the sliding window
                N[i, j, k] += 1
                # N[j, i, k] += 1
        else: # Has coalesced
            C[i, j, k] += 1
            # C[j, i, k] += 1
            break


def count_coal_events(trees,times,delta=100):
    """
    Count the number of coalescent events in a set of trees using a sliding window.
    
    Parameters:
    - trees: An iterator over tskit.Tree objects.
    - times: A list of time points that define the lower bound of the sliding window
    - delta: The width of the sliding window
    
    Returns:
    - N: A num_samplesxnum_samplesxlen(times) numpy array where N[i,j,k] is the number of coalescent events between samples i and j in the k-th time window.
    - L: A num_samplesxnum_samplesxlen(times) numpy array where L[i,j,k] is the number of uncoalesced loci between samples i and j at the lower bound of the k-th time window.
    """
    if times[0] !=0:
        times = np.concatenate(([0], times))
        remove_later = True
    else:
        remove_later = False

    n_times = len(times)
    times_upper = times + delta

    for k,tree in tqdm(enumerate(trees)):
        if k==0:
            n_samples = tree.num_samples()
            N = np.zeros((n_samples, n_samples, n_times), dtype=np.uint16) # matrix of number of coalescent events within sliding window 
            C = np.zeros((n_samples, n_samples, n_times), dtype=np.uint16) # matrix specifying if i and j have coalesesced by t
            L0 = np.zeros((n_samples, n_samples), dtype=np.uint16) # matrix counting number of loci where pairs i and j were chosen
            pairs = np.column_stack(np.triu_indices(n_samples, k=1)) 
            num_pairs = pairs.shape[0]

        # Randomly select pairs of samples to count coalescent events
        mask = np.random.random(size=num_pairs) < 2/n_samples
        active_pairs = pairs[mask]
        tmrca_active_pairs = np.array([tree.tmrca(ip, jp) for ip, jp in active_pairs])

        for pair,tmrca in zip(active_pairs,tmrca_active_pairs):
            L0[pair[0],pair[1]] += 1
            # L0[pair[1],pair[0]] += 1
            update_matrices(N,C, pair,tmrca,times,times_upper)
    L = L0[:,:,np.newaxis]-np.cumsum(C, axis=2)  # Number of uncoalesced loci

    if remove_later:
        N = N[:, :, 1:]  # Remove the first time point if it was added
        L = L[:, :, 1:]

    return N, L


def create_counts_df(N, L, times, population_samples_dict):
    n_total = N.shape[0]
    sample_to_population = {}
    for population, sample_ids in population_samples_dict.items():
        for sample_id in sample_ids:
            sample_to_population[sample_id] = population

    sample_1_inx = np.repeat(np.arange(n_total), n_total * len(times))
    sample_2_inx = np.tile(np.repeat(np.arange(n_total), len(times)), n_total)
    time_inx = np.tile(np.arange(len(times)), n_total * n_total)
    time = np.tile(times, n_total * n_total)
    num_uncoalesced = L.flatten()
    num_coal_events = N.flatten()
    sample_1_population = [sample_to_population[sample] for sample in sample_1_inx]
    sample_2_population = [sample_to_population[sample] for sample in sample_2_inx]

    df = pd.DataFrame({
        "sample_1_inx": sample_1_inx,
        "sample_2_inx": sample_2_inx,
        "time_inx": time_inx,
        "time": time,
        "num_uncoalesced": num_uncoalesced,
        "num_coal_events": num_coal_events,
        "sample_1_population": sample_1_population,
        "sample_2_population": sample_2_population
    })

    return df