#%%
import numpy as np # type: ignore
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
#%%
def compute_loss_gradient(X, R, delta, lambda_0, beta=1):
    # Mask for valid (non-nan) R entries
    valid_mask = ~np.isnan(R)

    D = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # Difference matrix
    dist_sq = np.sum(D ** 2, axis=-1)  # Distance squared matrix
    exp_term = delta * lambda_0 * np.exp(-beta * dist_sq)

    # Only compute loss where R is not nan
    loss_term_1 = np.zeros_like(R)
    loss_term_2 = np.zeros_like(R)
    loss_term_1[valid_mask] = (1 - R[valid_mask]) * exp_term[valid_mask]
    loss_term_2[valid_mask] = R[valid_mask] * dist_sq[valid_mask]
    loss_matrix = loss_term_1 + loss_term_2

    # Set diagonal to zero (only for valid entries)
    np.fill_diagonal(loss_matrix, 0)
    loss = np.nansum(loss_matrix)  # Sum ignoring nan

    # Gradient: only consider valid pairs
    grad_coeff = np.zeros_like(R)
    grad_coeff[valid_mask] = 2 * beta * (R[valid_mask] - (1 - R[valid_mask]) * exp_term[valid_mask])
    grad = np.einsum('ijk,ij->ik', D, grad_coeff)

    return loss, grad

# def compute_loss_gradient(X, R, delta, lambda_0, beta=1):
#     D = X[:, np.newaxis, :] - X[np.newaxis, :, :]  # Difference matrix
#     dist_sq = np.sum(D ** 2, axis=-1)  # Distance squared matrix
#     exp_term = delta * lambda_0 * np.exp(-beta * dist_sq)

#     # Compute loss terms directly
#     loss_term_1 = (1 - R) * exp_term
#     loss_term_2 = R * dist_sq
#     loss_matrix = loss_term_1 + loss_term_2

#     # Set diagonal to zero
#     np.fill_diagonal(loss_matrix, 0)
#     loss = np.sum(loss_matrix)  # Sum all elements

#     # Gradient computation
#     grad_coeff = 2 * beta * (R - (1 - R) * exp_term)
#     grad = np.einsum('ijk,ij->ik', D, grad_coeff)

#     return loss, grad



def bfgs_optimization(X_init, R, delta, lambda_0, beta=1, max_iter=10000, tol=1e-6, verbose=False):
    # Store original shape
    original_shape = X_init.shape
    
    # Wrapper function that handles reshaping for scipy.optimize
    def loss_and_grad(x):
        """Reshape x, compute loss/grad, then flatten grad."""
        X = x.reshape(original_shape)
        # loss, grad = compute_loss_gradient(X, R, delta, beta, grad_noise,reg_term=reg_term)
        loss, grad = compute_loss_gradient(X, R, delta, lambda_0, beta)
        return loss, grad.ravel()
    
    # Run BFGS optimization
    result = minimize(
        loss_and_grad,
        X_init.ravel(),  # Flatten initial guess
        method='BFGS',
        jac=True,  # We're providing the gradient
        options={
            'maxiter': max_iter,
            'gtol': tol,
            'disp': verbose
        }
    )
    
    # Reshape back to original format
    X_optimized = result.x.reshape(original_shape)
    final_loss = result.fun
    
    if verbose:
        print(f"\nOptimization results:")
        print(f"Success: {result.success}")
        print(f"Final loss: {final_loss:.6f}")
        print(f"Iterations: {result.nit}")
        print(f"Function evaluations: {result.nfev}")
    
    return X_optimized, final_loss

def compute_cluster_diameter(n,L0,delta,lambda_0,Ne):
    L = np.full((n, n), L0)
    L[np.arange(n), np.arange(n)] = 0 
    N = np.random.binomial(L, delta / Ne, (n, n))

    with np.errstate(divide='ignore', invalid='ignore'):
        R = N / L
    R[L == 0] = np.nan
    X_init = np.random.rand(n, 2)

    X,_ = bfgs_optimization(X_init, R, delta, lambda_0)

    diameter = np.max(X[:,0])-np.min(X[:,0])

    return diameter

def best_init(R,num_seeds,num_multipliers,Ne_average,delta=100,lambda_0=1,n=50):
    seeds = np.random.randint(0, 10000, size=num_seeds)
    multipliers = np.logspace(0, num_multipliers-1, num_multipliers,base=10)

    losses = np.zeros((num_seeds, num_multipliers))
    max_distance = np.zeros((num_seeds, num_multipliers))

    avg_diameter = compute_cluster_diameter(n,L0=10000,delta=100,lambda_0=lambda_0,Ne=Ne_average)

    for i,seed in enumerate(tqdm(seeds)):
        for j,m in enumerate(multipliers):
            np.random.seed(seed)
            X_init = np.random.rand(n, 2) * m
            X,loss = bfgs_optimization(X_init, R, delta, lambda_0)
            losses[i,j] = loss
            max_distance[i,j] = np.max(X[:,0])-np.min(X[:,0])

    min_loss = np.inf
    valid_max_distances = max_distance > 2 * avg_diameter

    if np.all(~valid_max_distances): # None of the present multipliers give a max latent space distance that is double the average diameter
        i, j = np.unravel_index(np.argmin(losses), losses.shape)
        best_seed = seeds[i]
        best_multiplier = multipliers[j]
    else:           
        for seed_inx,mult_inx in zip(*np.where(max_distance>2*avg_diameter)):
            if losses[seed_inx,mult_inx] < min_loss:
                min_loss = losses[seed_inx,mult_inx]
                best_seed = seeds[seed_inx]
                best_multiplier = multipliers[mult_inx]

    np.random.seed(best_seed)
    X_init = np.random.rand(n, 2) * best_multiplier

    return X_init

def compute_embeddings_through_time(N, L, times , delta, sample_times = None,n_components = 2):
# def compute_embeddings_through_time(R, times ,sample_times = None,n_components = 2, X_init=None, alpha=None):
    if sample_times is None:
        sample_times = np.zeros(N.shape[0])
    n = N.shape[0]

    with np.errstate(divide='ignore', invalid='ignore'):
        R = N / L
    
    # R += 1e-10
    lambda_0 = (2*np.nanquantile(R,0.95))/delta
    # lambda_0 = np.nanquantile(R,0.99)/delta
    R += 1e-10

    # Ne_average = 1/np.nanmean(R/delta)
    # R_0 = R[:,:,0]
    # R_0 = np.triu(R_0) + np.triu(R_0, k=1).T
    # X_init = best_init(R_0,num_seeds=20,num_multipliers=4,Ne_average=Ne_average,delta=delta,lambda_0=lambda_0,n=n)

    # alpha = np.nanquantile(R,0.99) # Set alpha to 99th percentile of R
    # if alpha is None:
    #     alpha = np.nanquantile(R,0.995)


    n_timepoints = R.shape[-1]
    X_over_time = np.full((n,n_components,n_timepoints),np.nan)
    X_init = np.random.rand(n, n_components)*1e4

    for k in tqdm(range(n_timepoints)):
        # inx = n_timepoints-k-1 # Reverse order for time
        inx = k
        R_t  = R[:,:,inx]
        # R_t[np.arange(n), np.arange(n)] = 0 # Set diagonal to zero
        # R_t = np.triu(R_t) + np.triu(R_t, k=1).T
        if k==0:
            X, _ = bfgs_optimization(X_init, R_t, delta, lambda_0)
            X = X-np.mean(X, axis=0)
        else:
            extant_sample_indices = np.where(sample_times<=times[inx])[0]
            R_t_extant = R_t[np.ix_(extant_sample_indices,extant_sample_indices)]
            X_init_extant = X_init[extant_sample_indices,:]
            X_extant, _ = bfgs_optimization(X_init_extant, R_t_extant, delta, lambda_0)
            X = np.full((n,n_components),np.nan)
            X[extant_sample_indices, :] = X_extant
            X, _ = bfgs_optimization(X_init, R_t, delta, lambda_0)
            # X = procrustes_adjustment(X_init, X)  # Procrustes adjustment to align embeddings

        X_over_time[:,:,inx] = X
        X_init = X

        # mean_abs_X = np.mean(np.abs(X_over_time),axis=(0,1))
        # max_mean_abs_X = np.max(mean_abs_X)
        # if np.mean(np.abs(X)) < max_mean_abs_X/10:
        #     X_init = np.random.rand(n, n_components)
        # else:
        #     jitter = 0.01 * np.std(X) * np.random.randn(*X.shape)
        #     X_init = X + jitter
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

def procrustes_adjustment(X_reference, Y):
    
    X_reference_centered = X_reference - np.mean(X_reference, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    
    
    cov_matrix = np.dot(Y_centered.T, X_reference_centered)
    
    
    U, _, Vt = np.linalg.svd(cov_matrix)
    rotation_matrix = np.dot(U, Vt)
    
    
    Y_rotated = np.dot(Y_centered, rotation_matrix)
    
    return Y_rotated


# %%
