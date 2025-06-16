#%%
import numpy as np # type: ignore
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd
#%%
def compute_loss_gradient(X, R, alpha, beta=1):
   D = X[:, np.newaxis, :] - X[np.newaxis, :, :] # Difference matrix
   dist_sq = np.sum(D ** 2, axis=-1) # Distance squared matrix
   exp_term = alpha*np.exp(-beta*dist_sq) 
   loss_term_1 = (1-R)*exp_term
   loss_term_2 = R*dist_sq
   loss = loss_term_1 + loss_term_2
   loss[np.arange(loss.shape[0]),np.arange(loss.shape[0])] = 0
   loss = np.sum(loss)

   grad_coeff = 2 * beta * (R-(1-R)*exp_term)
   grad = np.einsum('ijk,ij->ik', D, grad_coeff)
   
   return loss, grad


def bfgs_optimization(X_init, R, alpha, beta=1, max_iter=10000, tol=1e-3, verbose=False):
    # Store original shape
    original_shape = X_init.shape
    
    # Wrapper function that handles reshaping for scipy.optimize
    def loss_and_grad(x):
        """Reshape x, compute loss/grad, then flatten grad."""
        X = x.reshape(original_shape)
        # loss, grad = compute_loss_gradient(X, R, alpha, beta, grad_noise,reg_term=reg_term)
        loss, grad = compute_loss_gradient(X, R, alpha, beta)
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

def compute_embeddings_through_time(N, L, times ,sample_times = None,n_components = 2, X_init=None, alpha=None):
    if sample_times is None:
        sample_times = np.zeros(N.shape[0])
    n = N.shape[0]

    if X_init is None:
        X_init = np.random.rand(n, n_components)
    R = N/(L+1)
    # alpha = np.quantile(R,0.99) # Set alpha to 99th percentile of R
    if alpha is None:
        alpha = np.quantile(R,0.995)


    n_timepoints = R.shape[-1]
    X_over_time = np.full((n,n_components,n_timepoints),np.nan)

    for k in tqdm(range(n_timepoints)):
        # inx = n_timepoints-k-1 # Reverse order for time
        inx = k
        R_t  = R[:,:,inx]
        R_t[np.arange(n), np.arange(n)] = 0 # Set diagonal to zero
        R_t = np.triu(R_t) + np.triu(R_t, k=1).T
        if k==0:
            X, _ = bfgs_optimization(X_init, R_t, alpha)
            X = X-np.mean(X, axis=0)
        else:
            extant_sample_indices = np.where(sample_times<=times[inx])[0]
            R_t_extant = R_t[np.ix_(extant_sample_indices,extant_sample_indices)]
            X_init_extant = X_init[extant_sample_indices,:]
            X_extant, _ = bfgs_optimization(X_init_extant, R_t_extant, alpha)
            X = np.full((n,n_components),np.nan)
            X[extant_sample_indices, :] = X_extant
            X, _ = bfgs_optimization(X_init, R_t, alpha)
            X = procrustes_adjustment(X_init, X)  # Procrustes adjustment to align embeddings

        X_over_time[:,:,inx] = X
        mean_abs_X = np.mean(np.abs(X_over_time),axis=(0,1))
        max_mean_abs_X = np.max(mean_abs_X)
        if np.mean(np.abs(X)) < max_mean_abs_X/10:
            X_init = np.random.rand(n, n_components)
        else:
            jitter = 0.01 * np.std(X) * np.random.randn(*X.shape)
            X_init = X + jitter
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
