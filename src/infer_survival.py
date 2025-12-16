 #%% A script to infer rates in different ways
"""
1- Truly independent trees
2- Trees simulated on a chromosome with recombination
3- Trees simulated on a chromosome with recombination and then inferred using tsinfer
4- Trees inferred from empirical data
"""
import msprime
import numpy as np
import pandas as pd
from CountCoalEvents import CountCoalEvents
import demes
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
def compute_loss_gradient(X_flat, R, N0, d=2):
    # Mask for valid (non-nan) R entries
    n = R.shape[0]
    X = X_flat.reshape(n, d)

    valid_mask = ~np.isnan(R)
    distances_squared = squareform(pdist(X, 'sqeuclidean'))
    valid_R = R[valid_mask]
    valid_distances_squared = distances_squared[valid_mask] + N0
    inv_distances_squared = 1.0 / valid_distances_squared

    # Compute loss
    loss_values = inv_distances_squared + valid_R * np.log(valid_distances_squared)
    loss = np.nansum(loss_values)

    # Compute gradient
    grad_coeff = np.zeros((n, n))
    grad_coeff[valid_mask] = valid_R * inv_distances_squared - inv_distances_squared**2
    row_sums = np.sum(grad_coeff, axis=1)
    grad = (row_sums[:, np.newaxis] * X - grad_coeff @ X)

    return loss, grad.ravel()

def compute_embeddings(X_init, R, N0, d=2, max_iter=20000, tol=1e-10):
    # Optimize using L-BFGS-B
    result = minimize(compute_loss_gradient, X_init.ravel(), args=(R, N0, d), method='L-BFGS-B',
                      jac=True, options={'maxiter': max_iter, 'disp': True, 'gtol': tol})
    X_opt = result.x.reshape(X_init.shape)
    return X_opt, result.fun

# %%
def compute_true_survival(demography,time_points,L=int(1e6)):
    populations = [pop.name for pop in demography.populations if pop.default_sampling_time == 0]
    population_pairs = []
    for i in range(len(populations)):
        for j in range(i,len(populations)):
            population_pairs.append((populations[i],populations[j]))
    df = pd.DataFrame(columns=time_points,index = population_pairs)
    for i,pair in enumerate(population_pairs):
        coal_times_summary = np.zeros_like(time_points,np.uint32)
        samples = {pop:0 for pop in populations}
        samples[pair[0]] += 1
        samples[pair[1]] += 1
        trees = msprime.sim_ancestry(demography=demography,
                                    samples=samples,
                                    sequence_length=1,
                                    ploidy=1,
                                    num_replicates=L,
                                    random_seed=42)
        for tree in tqdm(trees,total=L,desc=f"Inferring true survival for pair {pair}"):
            tmrca = tree.first().tmrca(0,1)
            idx = np.searchsorted(time_points,tmrca,"left")
            coal_times_summary[idx-1] += 1

        survival_prob = 1 - np.cumsum(coal_times_summary)/L
        survival_prob = np.concatenate(([1],survival_prob[:-1]))

        df.iloc[i] = survival_prob
    df["inf"] = 0

    return df

    
#%%
graph = demes.load("../data/deme_files/stdpopsim_deme_files/OutOfAfrica_4J17.yaml")
demography = msprime.Demography.from_demes(graph)
max_time = (int(demography.events[-1].time*2)//1000)*1000
time_points = np.arange(0,max_time+1,5)

true_survival_df = compute_true_survival(demography,time_points,L=int(1e6))
# %%
rates = (survival[:,:-10:10]-survival[:,10::10])/(50*survival[:,:-10:10])
rates_df = pd.DataFrame(rates,columns=time_points[:-10:10],index=true_survival_df.index)
rates_times = rates_df.columns.values
fig,ax = plt.subplots(1,2,figsize=(12,6),dpi=500)
for i,row in enumerate(true_survival_df.iterrows()):
    pop1,pop2 = row[0]
    ax[0].plot(time_points,row[1][:-1],label=f"{pop1}-{pop2}",alpha=0.5)
    ax[1].plot(time_points[:-10:10],rates[i,:],alpha=0.5)
ax[0].legend()
ax[0].set_xscale("log")
ax[1].set_xscale("log")
# %%
rates_df.iloc[1:4,:np.where(rates_times>119000/29)[0][0]] = 0
rates_df.iloc[5:7,:np.where(rates_times>46000/29)[0][0]] = 0
rates_df.iloc[8,:np.where(rates_times>9000/29)[0][0]] = 0
# %%
n = 50
n_samples = n*len(populations)
samples_pop_dict = {}
for i in range(n_samples):
    if i < n:
        pop = populations[0]
    elif i < 2*n:
        pop = populations[1]
    elif i < 3*n:
        pop = populations[2]
    else:
        pop = populations[3]
    samples_pop_dict[i] = pop

pairs = []
for i in range(n_samples):
    for j in range(i+1,n_samples):
        pairs.append((i,j))


expanded_rates_df = pd.DataFrame(columns=rates_df.columns,index=pairs)
for k,pair in enumerate(pairs):
    pop1 = samples_pop_dict[pair[0]]
    pop2 = samples_pop_dict[pair[1]]
    pop_pair_idx = np.where([(pop1,pop2)==p for p in population_pairs])[0][0]
    expanded_rates_df.iloc[k] = rates_df.iloc[pop_pair_idx]

# %%
expanded_rates_df += 1e-8
LSE = LatentSpaceEmbeddings(expanded_rates_df)
embeddings = LSE.compute_embeddings_through_time()
embeddings_df = LSE.get_embeddings_df()
# %%
import plotly.express as px
import plotly.io as pio
colors_dict = {}
for i in range(n_samples):
    if i < n:
        col = "blue"
    elif i < 2*n:
        col = "orange"
    elif i < 3*n:
        col = "green"
    else:
        col = "red"
    colors_dict[i] = col
embeddings_df["pop"] = embeddings_df["sample_id"].map(samples_pop_dict)
fig = px.line_3d(embeddings_df, x="X1", y="X2", z="time", color='pop', line_group='sample_id', color_discrete_map=colors_dict,log_z=True)
pio.write_html(fig, file="true_indep_dynamic_3d_plot.html")
# %%
# from sklearn.manifold import smacof
expanded_rates_df = pd.DataFrame(columns=rates_df.columns,index=pairs)
for k,pair in enumerate(pairs):
    pop1 = samples_pop_dict[pair[0]]
    pop2 = samples_pop_dict[pair[1]]
    pop_pair_idx = np.where([(pop1,pop2)==p for p in population_pairs])[0][0]
    expanded_rates_df.iloc[k] = rates_df.iloc[pop_pair_idx]
# expanded_rates_df += 1e-7
LSE = LatentSpaceEmbeddings(expanded_rates_df)
rate_times = rates_df.columns.values
R = LSE.R.copy()
#%%
idx = 7
R_t = R[:,:,idx]
N0 = LSE.N0
# N0=1
X_init = np.random.rand(200,2)
# D = np.sqrt(1/R_t-N0)
# D[np.arange(D.shape[0]),np.arange(D.shape[0])] = 0
X,loss = compute_embeddings(X_init, R_t, N0)
# X,_ = smacof(D)
X = X-np.mean(X,axis=0)
for k in range(4):
    if k>=2:
        plt.scatter(X[k*n:(k+1)*n,0],X[k*n:(k+1)*n,1],label=populations[k])
plt.title(f"Time = {rate_times[idx]} generations, {rate_times[idx]*29} years")
plt.legend()
print(rates_df.iloc[:2,50])
# plt.xlim(-1500,1500)
# plt.ylim(-1500,1500)

# %%
idx = 7
plt.scatter(embeddings[100:150,0,idx],embeddings[100:150,1,idx])
plt.scatter(embeddings[150:200,0,idx],embeddings[150:200,1,idx])
# %%
