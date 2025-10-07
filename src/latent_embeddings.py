#%%
import numpy as np
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import smacof
#%%
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
    loss = 0.5 * np.nansum(loss_values)

    # Compute gradient
    grad_coeff = np.zeros((n, n))
    grad_coeff[valid_mask] = valid_R * inv_distances_squared - inv_distances_squared**2
    row_sums = np.sum(grad_coeff, axis=1)
    grad = (row_sums[:, np.newaxis] * X - grad_coeff @ X)

    return loss, grad.ravel()

def solve(X, R, N0):
    max_iters = 10000
    X_opt = X.copy()
    result = minimize(compute_loss_gradient, X_opt.ravel(), args=(R, N0), method='L-BFGS-B', 
                     jac=True, options={'maxiter': max_iters, 'disp': True, 'gtol': 1e-9})
    return result.x.reshape(X.shape), result.fun

def compute_embeddings_through_time(N, L, delta, n_components = 2):
    n = N.shape[0]
    with np.errstate(divide='ignore', invalid='ignore'):
        R = N/(delta*L) 
    # R += np.nanmin(R[R!=0])/10
    Ne = 1/R
    N0 = np.nanquantile(Ne[Ne!=np.inf], 0.02)
    n_timepoints = R.shape[-1]
    X_over_time = np.full((n,n_components,n_timepoints),np.nan)
    X = np.random.rand(n,n_components)

    for k in tqdm(range(n_timepoints)):
        R_t  = R[:,:,k]
        X, _ = solve(X, R_t, N0)
        if k==0:
            X -= X.mean(axis=0)
        X_over_time[:,:,k] = X
    return X_over_time

def create_embeddings_df(embeddings_through_time, times, population_samples_dict):
    """
    Creates a DataFrame to store the embeddings through time.

    Args:
        embeddings_through_time (np.ndarray): A nxKxnum_timepoints matrix of embeddings through time.
        times (np.ndarray): List of time points.
        population_samples_dict (dict): Dictionary mapping population IDs to sample IDs.

    Returns:
        pd.DataFrame: DataFrame with columns [sample_id, population_id, time_inx, time, X1, X2, ...].
    """
    n, _, num_timepoints = embeddings_through_time.shape
    sample_ids = np.arange(n)
    time = np.tile(times, n)
    X1_flat = embeddings_through_time[:,0,:].flatten()
    X2_flat = embeddings_through_time[:,1,:].flatten()

    # Map sample IDs to population IDs
    sample_to_population = {}
    for population, sample_list in population_samples_dict.items():
        for sample_id in sample_list:
            sample_to_population[sample_id] = population
    population_ids = [sample_to_population[sample_id] for sample_id in np.repeat(sample_ids, num_timepoints)]

    # Create the DataFrame
    df = pd.DataFrame({
        "sample_id": np.repeat(sample_ids, num_timepoints),
        "population": population_ids,
        "time": time,
        "X1": X1_flat,
        "X2": X2_flat
    })

    return df

# %%
