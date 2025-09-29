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
        mask = np.random.random(size=num_pairs) < 0.1/n_samples
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

    N = N + N.transpose((1,0,2))
    L = L + L.transpose((1,0,2))
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

def compute_local_ancestry(tree,census_time,population_id_name):
    local_ancestry = np.full(tree.num_samples(),np.nan,dtype=object)
    for node in tree.nodes():
        if tree.time(node) != census_time:
            continue
        pop = tree.population(node)
        leaves = np.array(list(tree.leaves(node)))
        local_ancestry[leaves] = population_id_name[pop]
    return local_ancestry

def count_coal_events_admix(trees,times,census_time,population_id_name,delta=100):
    if times[0] !=0:
        times = np.concatenate(([0], times))
        remove_later = True
    else:
        remove_later = False

    n_times = len(times)
    times_upper = times + delta

    for k,tree in tqdm(enumerate(trees)):
        if k==0:
            local_ancestry = compute_local_ancestry(tree,census_time,population_id_name)
            source_populations = np.unique(local_ancestry)
            source_pop_index = {pop: i for i, pop in enumerate(source_populations)}
            K = len(source_populations)

            n_samples = tree.num_samples()
            N = np.zeros((n_samples*K, n_samples*K, n_times), dtype=np.uint16) # matrix of number of coalescent events within sliding window 
            C = np.zeros((n_samples*K, n_samples*K, n_times), dtype=np.uint16) # matrix specifying if i and j have coalesesced by t
            L0 = np.zeros((n_samples*K, n_samples*K), dtype=np.uint16) # matrix counting number of loci where pairs i and j were chosen
            pairs = np.column_stack(np.triu_indices(n_samples, k=1)) 
            num_pairs = pairs.shape[0]

            extant_populations = [population_id_name[pop] for pop in np.unique([tree.population(sample) for sample in np.sort(list(tree.samples()))])]
            num_extant_populations = len(extant_populations)
            n_pop = int(n_samples / num_extant_populations)

        # Randomly select pairs of samples to count coalescent events
        mask = np.random.random(size=num_pairs) < 2/n_samples
        active_pairs = pairs[mask]
        tmrca_active_pairs = np.array([tree.tmrca(ip, jp) for ip, jp in active_pairs])

        local_ancestry = compute_local_ancestry(tree,census_time,population_id_name)
        for pair,tmrca in zip(active_pairs,tmrca_active_pairs):
            pair[0] = pair[0] * K + source_pop_index[local_ancestry[pair[0]]]
            pair[1] = pair[1] * K + source_pop_index[local_ancestry[pair[1]]]

            L0[pair[0],pair[1]] += 1
            update_matrices(N,C,pair,tmrca,times,times_upper)
    L = L0[:,:,np.newaxis]-np.cumsum(C, axis=2)  # Number of uncoalesced loci
    if remove_later:
        N = N[:, :, 1:]  # Remove the first time point if it was added
        L = L[:, :, 1:]

    remove_indices = np.zeros(N.shape[0], dtype=bool)
    for i in range(n_samples):
        if i == n_samples-1:
            break
        for k1 in range(K):
            inx1 = i * K + k1
            if np.all(N[inx1,:,:]==0):
                remove_indices[inx1] = True
            else:
                for j in range(i+1, n_samples):
                    for k2 in range(K):
                        inx2 = j * K + k2
                        imputed_time_inxs = np.where(N[inx1,inx2,:]==0)[0]
                        imputed_time_inxs = imputed_time_inxs[imputed_time_inxs != 0]
                        if k1 != k2:
                            L[inx1, inx2, imputed_time_inxs] = 0
                        if N[inx1, inx2, 0] == 0:
                            non_zero_types = np.arange(K)
                            non_zero_types = non_zero_types[non_zero_types != k1]

                            N[inx1, inx2, 0] = N[i*K + non_zero_types, inx2, 0].reshape(len(non_zero_types),).mean(axis=0)

    for i in range(n_samples):
        L[i*K:(i+1)*K, i*K:(i+1)*K,:] = 0

    N = N[~remove_indices,:,:]
    N = N[:,~remove_indices,:]
    L = L[~remove_indices,:,:]
    L = L[:,~remove_indices,:]
    
    population_samples_dict = {}
    i=0
    for k,pop in enumerate(extant_populations):
        n_active = sum(~remove_indices[k*K*n_pop:(k+1)*K*n_pop])
        population_samples_dict[pop] = np.arange(i,i+n_active)
        i+=n_active
        
    return N, L, population_samples_dict


# #%%
# L0 = 2000
# n=50
# K=3
# delta = 50
# min_time = 10
# max_time = 5000
# times = np.logspace(np.log10(min_time),np.log10(max_time),100,base=10)


# ts_list = msprime.sim_ancestry(
#     samples = samples,
#     demography = demography,
#     sequence_length=1,
#     ploidy = 1,
#     num_replicates = int((L0*n*K)/2)
#     )
# trees = (ts.first() for ts in ts_list)
# #%%
# N,L = count_coal_events_admix(trees,times,census_time,population_id_name,delta=delta)
# # %%
# embeddings_through_time = compute_embeddings_through_time(R, times)
# embeddings_df = create_embeddings_df(embeddings_through_time, times, population_samples_dict)

# fig = px.line_3d(embeddings_df, x="X1", y="X2", z="time",log_z=True, color='population',line_group='sample_id',
#     color_discrete_map=colors_dict
# )
# x1 = embeddings_df["X1"].values
# x2 = embeddings_df["X2"].values

# min_value = min(np.min(x1),np.min(x2)) * 1.05
# max_value = max(np.max(x1),np.max(x2)) * 1.05

# fig.update_layout(showlegend=True)

# fig.update_layout(
#     scene=dict(
#         xaxis=dict(range=[min_value, max_value]),
#         yaxis=dict(range=[min_value, max_value]),
#     )
# )

# pio.write_html(fig, file="test.html")
# # %%
# R=N/L
# remove_indices = np.zeros(R.shape[0], dtype=bool)
# for i in range(n_samples):
#     if i == n_samples-1:
#         break
#     for k1 in range(K):
#         inx1 = i * K + k1
#         if np.all(R[inx1,:,:]==0):
#             remove_indices[inx1] = True
#         else:
#             for j in range(i+1, n_samples):
#                 for k2 in range(K):
#                     inx2 = j * K + k2
#                     imputed_time_inxs = np.where(R[inx1,inx2,:]==0)[0]
#                     if len(imputed_time_inxs) > 0:
#                         non_zero_types = np.arange(K)
#                         non_zero_types = non_zero_types[non_zero_types != k1]

#                         R[inx1, inx2, imputed_time_inxs] = R[i*K + non_zero_types, inx2, imputed_time_inxs].reshape(len(non_zero_types),len(imputed_time_inxs)).mean(axis=0)
# # %%
# x=R[200,202:,10]
# y=R[201,202:,10]
# # %%
# K=2
# i=100
# k1=0
# j=101
# k2=1
# inx1= i * K + k1
# inx2= j * K + k2
# # %%
# R=N/L
# for j in range(i+1, n_samples):
#     for k2 in range(K):
#         inx2 = j * K + k2
#         imputed_time_inxs = np.where(R[inx1,inx2,:]==0)[0]
#         if len(imputed_time_inxs) > 0:
#             non_zero_types = np.arange(K)
#             non_zero_types = non_zero_types[non_zero_types != k1]

#             R[inx1, inx2, imputed_time_inxs] = R[i*K + non_zero_types, inx2, imputed_time_inxs].reshape(len(non_zero_types),len(imputed_time_inxs)).mean(axis=0)
# # %%
# sns.heatmap(R[100:102,100:,0])
# # %%
# R_adjusted = R.copy()
# for i in range(100,200):
#     if i%2 == 0:
#         R_adjusted[i,i+3::2,1:] = np.nan
#     else:
#         R_adjusted[i,i+1::2,1:] = np.nan   
    

# for i in range(200):
#     if i%2 == 0:
#         R_adjusted[i,100+i+3::2,1:] = np.nan
#     else:
#         R_adjusted[i,100+i+1::2,1:] = np.nan   
# # %%
# N = copy(N_old)
# L = copy(L_old)
# remove_indices = np.zeros(N.shape[0], dtype=bool)
# for i in range(n_samples):
#     if i == n_samples-1:
#         break
#     for k1 in range(K):
#         inx1 = i * K + k1
#         if np.all(N[inx1,:,:]==0):
#             remove_indices[inx1] = True
#         else:
#             for j in range(i+1, n_samples):
#                 for k2 in range(K):
#                     inx2 = j * K + k2
#                     imputed_time_inxs = np.where(N[inx1,inx2,:]==0)[0]
#                     imputed_time_inxs = imputed_time_inxs[imputed_time_inxs != 0]
#                     if k1 != k2:
#                         L[inx1, inx2, imputed_time_inxs] = 0
#                     if N[inx1, inx2, 0] == 0:
#                         non_zero_types = np.arange(K)
#                         non_zero_types = non_zero_types[non_zero_types != k1]

#                         N[inx1, inx2, 0] = N[i*K + non_zero_types, inx2, 0].reshape(len(non_zero_types),).mean(axis=0)


# for i in range(n_samples):
#     L[i*K:(i+1)*K, i*K:(i+1)*K,:] = 0

# # R = R[~remove_indices,:,:]
# # R = R[:,~remove_indices,:]
# N = N[~remove_indices,:,:]
# N = N[:,~remove_indices,:]
# L = L[~remove_indices,:,:]
# L = L[:,~remove_indices,:]

# R = N/L
# #%%
# remove_indices = np.zeros(N.shape[0], dtype=bool)
#     for i in range(n_samples):
#         if i == n_samples-1:
#             break
#         for k1 in range(K):
#             inx1 = i * K + k1
#             if np.all(N[inx1,:,:]==0):
#                 remove_indices[inx1] = True
#             else:
#                 for j in range(i+1, n_samples):
#                     for k2 in range(K):
#                         inx2 = j * K + k2
#                         imputed_time_inxs = np.where(N[inx1,inx2,:]==0)[0]
#                         if len(imputed_time_inxs) > 0:
#                             non_zero_types = np.arange(K)
#                             non_zero_types = non_zero_types[non_zero_types != k1]

#                             N[inx1, inx2, imputed_time_inxs] = N[i*K + non_zero_types, inx2, imputed_time_inxs].reshape(len(non_zero_types),len(imputed_time_inxs)).mean(axis=0)
#                             C[inx1, inx2, imputed_time_inxs] = C[i*K + non_zero_types, inx2, imputed_time_inxs].reshape(len(non_zero_types),len(imputed_time_inxs)).mean(axis=0)


#     L = L0[:,:,np.newaxis]-np.cumsum(C, axis=2)  # Number of uncoalesced loci

#     if remove_later:
#         N = N[:, :, 1:]  # Remove the first time point if it was added
#         L = L[:, :, 1:]


#     # R = N/L
#     # max_R  = np.nanmax(R)
#     for i in range(n_samples):
#         # R[i*K:(i+1)*K, i*K:(i+1)*K,:] = max_R
#         L[i*K:(i+1)*K, i*K:(i+1)*K,:] = 0

#     # R = R[~remove_indices,:,:]
#     # R = R[:,~remove_indices,:]
#     N = N[~remove_indices,:,:]
#     N = N[:,~remove_indices,:]
#     L = L[~remove_indices,:,:]
#     L = L[:,~remove_indices,:]

#%%
# L0=10000
# ts_list,metadata,population_samples_dict = prepare_from_simulation(demography,n=n,L0=L0)
# trees = (ts.first() for ts in ts_list)
# %%
