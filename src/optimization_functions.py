#%%
#!/usr/bin/env python3
"""
This module contains functions for computing the loss and the gradient for the optimization of latent space embeddings based on pairwise rates between samples over time.
Loss and gradient can be computed for:
1. All samples at a given time slice
2. A given sample over all time slices
3. All samples over all time slices
The module also contains functions for performing optimization using L-BFGS-B for a given sample, a given time slice, or all samples and time slices jointly.
"""

GTOL_TIMEPOINT     = 1e-12
FTOL_TIMEPOINT     = 1e-8
GTOL_SAMPLE        = 1e-12
FTOL_SAMPLE        = 1e-12
MAX_ITER_TIMEPOINT = 6000
MAX_ITER_SAMPLE    = 6000

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

############################################################
# General Functions
############################################################

def compute_f(D,
              N_min,
              link="inverse_sq_distance"): 
    """Compute the link function f and its derivative f_prime based on the distance matrix D and minimum value N_min. f(x) is the function that maps distances to rates.

    Args:
        D (np.ndarray): Distance matrix between pairs of samples.
        N_min (float): Minimum rate that could be modelled, where samples with rate = 1/N_min or higher will have zero distances between them.
        N_max (float): Maximum rate that could be modelled, where samples with rate = N_max or lower will have infinite distances between them.
        link (str, optional): Type of link function to compute. Defaults to "inverse_distance".

    Returns:
        tuple: f and f_prime arrays.
    """
    if link == "inverse_distance":                  # f = 1/(D + N_min)                         
        inverse_distance = 1/(D + N_min)            
        f = inverse_distance 
        f_prime = -inverse_distance**2
    elif link == "inverse_sq_distance":             # f = 1/(D^2 + N_min)
        inverse_distance_sq = 1/(D**2 + N_min)
        f = inverse_distance_sq
        f_prime = -2*D*inverse_distance_sq**2
    elif link == "exponential":                     # f = 1/N_min * exp(-D)
        f = (1/N_min)*np.exp(-D)
        f_prime = -f
    
    return f,f_prime

def procrustes_single_time(X_reference, Y):
    X_reference_centered = X_reference - np.mean(X_reference, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)
    cov_matrix = np.dot(Y_centered.T, X_reference_centered)
    U, _, Vt = np.linalg.svd(cov_matrix)
    rotation_matrix = np.dot(U, Vt)
    Y_rotated = np.dot(Y_centered, rotation_matrix)
    return Y_rotated

def procrustes_alignment(X):
    X_aligned = np.zeros_like(X)
    n_timepoints = X.shape[2]
    X_reference = X[:,:,0]
    X_reference = X_reference - np.mean(X_reference, axis=0)
    X_aligned[:,:,0] = X_reference
    for t in range(1,n_timepoints):
        Y = X[:,:,t]
        Y_aligned = procrustes_single_time(X_reference,Y)
        X_aligned[:,:,t] = Y_aligned
    return X_aligned

def compute_difference_operators(num_timepoints, max_order, delta_t):
    D1 = np.zeros((num_timepoints-1,num_timepoints))
    np.fill_diagonal(D1, -1)
    for t in range(num_timepoints-1):
        D1[t, t+1] = 1
    D1 /= delta_t
    D1 = np.vstack((D1, np.zeros((1,num_timepoints)))) # Pad to make square, where the derivative at last time is zero (or could be forward difference if you prefer)

    D = np.eye(num_timepoints)
    D_dict = {}
    for k in range(1,max_order+1):
        D = D1 @ D
        D_dict[k] = D
    return D_dict

def compute_M(f,f_prime,D,R,valid_mask):
    """Computes a useful matrix M that is used in gradient computation

    Args:
        f (np.ndarray): Array of function values.
        f_prime (np.ndarray): Array of function derivatives.
        D (np.ndarray): Distance matrix between pairs of samples.
        R (np.ndarray): Rate matrix between pairs of samples.
        valid_mask (np.ndarray): Boolean mask indicating valid entries.

    Returns:
        np.ndarray: Matrix M used in gradient computation.
    """
    M = np.zeros_like(R)
    M[valid_mask] = (f_prime[valid_mask]/D[valid_mask])*(1-R[valid_mask]/(f[valid_mask]))
    return M

def compute_unpenalized_loss(f,R,valid_mask):
    """Computes the loss function without penalty.
    Can be computed for:
      1. a given time slice (nxn)
      2. a given sample (nxT)
      3. the whole dataset (nxnxT)

    Args:
        f (np.ndarray): Array of function values at a given time slice.
        R (np.ndarray): Rate matrix between pairs of samples at a given time slice.
        valid_mask (np.ndarray): Boolean mask indicating valid entries.

    Returns:
        float: Unpenalized loss value.
    """
    loss_values = f[valid_mask] - R[valid_mask] * np.log(f[valid_mask])
    return 0.5*np.nansum(loss_values)

def compute_total_unpenalized_loss(X,R,N_min,N_max,epsilon=None):
    """Computes the total unpenalized loss function over all time points.

    Args:
        X (np.ndarray): Array of coordinates for all samples over all time points. (nxdxT)
        R (np.ndarray): Rates matrix between pairs of samples over all time points. (nxn*T)
        N_min (float): Minimum value for the function computation.
        epsilon (float, optional): Scaling factor for loss, defining the standard error of rate estimates. Defaults to None.

    Returns:
        float: Total unpenalized loss value.
    """
    R = R.copy()
    R += 1/N_max 
    n,_,T = R.shape
    D = np.linalg.norm(X[:, None, :, :] - X[None, :, :, :], axis=2) # (nxn*T) Distance between pairs of samples over time
    f,_ = compute_f(D,N_min)
    valid_mask = ~np.isnan(R)
    loss = compute_unpenalized_loss(f,R,valid_mask)

    if epsilon is not None:
        loss /= epsilon

    return loss

############################################################
# Coordinate descent over time slices
## Coordinate descent over time slices, where at each time slice we optimize the coordinates for all samples using L-BFGS-B optimization, initializing with the solution from the previous time slice.
## This can be done in both forward and backward directions, and takes into account ancient samples that are not present in all time points.
############################################################
def compute_unpenalized_gradient_timepoint(X_t,f_t,f_prime_t,D_t,R_t,valid_mask_t):
    """Computes the contribution to the gradient for a given time point from the unpenalized term in the loss function.

    Args:
        X_t (np.ndarray): Array of coordinates for all samples at the t-th time point. (nxd)
        f_t (np.ndarray): Array of function values for the t-th time point. (nxn)
        f_prime_t (np.ndarray): Array of function derivatives for the t-th time point. (nxn)
        D_t (np.ndarray): Distance matrix between pairs of samples at the t-th time point. (nxn)
        R_t (np.ndarray): Rate matrix between pairs of samples at the t-th time point. (nxn)
        valid_mask_t (np.ndarray): Boolean mask indicating valid entries for the t-th time point. (nxn)

    Returns:
        np.ndarray: Gradient matrix for the t-th time point. (nxd)
    """
    M = compute_M(f_t,f_prime_t,D_t,R_t,valid_mask_t)
    grad = (np.sum(M, axis=1)[:, np.newaxis] * X_t - M @ X_t)
    return grad

def compute_loss_gradient_timepoint(X_t_flat, R_t, N_min, N_max = np.inf, epsilon = None):
    """Wrapper function to to compute both loss and gradient for minimization for a given time point

    Args:
        X_t_flat (np.ndarray): Flattened array of coordinates for all samples at the t-th time point. (n*d,)
        R_t (np.ndarray): Rates matrix between pairs of samples at the t-th time point. (nxn)
        N_min (float): Minimum value for the function computation.
        epsilon (float, optional): Scaling factor for loss and gradient, defining the standard error of rate estimates. Defaults to None.
        smoothing_params (dict, optional): Dictionary containing smoothing parameters for the penalty term. Defaults to None.
            Keys should be "l1" and "l2" for L1 and L2 penalties respectively in addition to "diff_operator" that contains the finite difference operator to compute DX.

    Returns:
        tuple: A tuple containing the loss (float) and the gradient (np.ndarray) for the t-th time point.
    """
    R_t = R_t.copy()
    R_t += 1/N_max # To avoid issues with zero rates, which would lead to infinite distances and zero gradients, we add a small value to all rates. This value is determined by the N_max parameter, where samples with rate = N_max or lower will have infinite distances between them.
    n,_ = R_t.shape
    X_t = X_t_flat.reshape((n, -1))
    valid_mask = ~np.isnan(R_t)
    D_t = squareform(pdist(X_t, 'euclidean'))
    f_t, f_prime_t = compute_f(D_t, N_min)
    loss = compute_unpenalized_loss(f_t, R_t, valid_mask)
    grad = compute_unpenalized_gradient_timepoint(X_t,f_t,f_prime_t,D_t,R_t,valid_mask)

    if epsilon is not None:
        loss /= epsilon
        grad /= epsilon

    return loss, grad.ravel()

def optimize_timepoint(X_t_init, R_t, N_min=1, N_max=np.inf, epsilon=None, max_iter=MAX_ITER_TIMEPOINT, gtol=GTOL_TIMEPOINT, ftol=FTOL_TIMEPOINT):
    """A single coordinate descent step.
    For a given time point, finds the optimal coordinates for all samples by minimizing the unpenalized loss function using L-BFGS-B optimization.

    Args:
        X_t_init (np.ndarray): Initial coordinates for all samples at the t-th time point. (n*d,)
        R_t (np.ndarray): Rates matrix between pairs of samples at the t-th time point. (nxn)
        N_min (float): Hyperparameter for the function computation, where samples with rate = 1/N_min or higher will have zero distances between them.
        N_max (float): Hyperparameter for the function computation, where samples with rate = 0 will have distances of N_max between them.
        epsilon (float): Scaling factor for loss and gradient, defining the standard error of rate estimates.
        max_iter (int): Maximum number of iterations for the optimizer.
        gtol (float): Tolerance for the gradient.
        ftol (float): Tolerance for the function value.

    Returns:
        tuple: A tuple containing the optimized coordinates (np.ndarray) and the final loss value (float) for the t-th time point.
    """
    result = minimize(compute_loss_gradient_timepoint, 
                      X_t_init.ravel(), args=(R_t, N_min, N_max, epsilon), 
                      method='L-BFGS-B', 
                      jac=True, 
                      options={'maxiter': max_iter, 'disp': False, 'gtol': gtol, 'ftol': ftol})
    X_opt = result.x.reshape(X_t_init.shape)
    return X_opt, result.fun, result.nit, result.message

def coordinate_descent_over_time(
    R,
    N_min=1,
    N_max=np.inf,
    n_components=2,
    max_iter=MAX_ITER_TIMEPOINT,
    gtol=GTOL_TIMEPOINT,
    ftol=FTOL_TIMEPOINT,
    epsilon=None,
    X_init=None,
    sampled_by_time=None,
    direction="backward",
):
    """Coordinate descent optimization over time slices.
        iterates over time points, optimizing coordinates for all samples at each time point using the initialization from the previous time point. The order of optimization over time points can be specified by the direction parameter.
        backward means starting from the last time point and moving to the present, while forward means starting from the first time point and moving to the past.
        takes into account ancient samples that are not present in all time points

    Args:
        X_init (np.ndarray): Initial coordinates for all samples over all time points. (n*d*T,)
        R (np.ndarray): Rates matrix between pairs of samples over all time points. (n*n*T)
        N_min (float): Hyperparameter for the function computation, where samples with rate = 1/N_min or higher will have zero distances between them.
        N_max (float): Hyperparameter for the function computation, where samples with rate = 0 will have distances of N_max between them.
        epsilon (float): Scaling factor for loss and gradient, defining the standard error of rate estimates.
        max_iter (int): Maximum number of iterations for the optimizer.
        gtol (float): Tolerance for the gradient.
        ftol (float): Tolerance for the function value.
        sampled_by_time (np.ndarray): Boolean matrix indicating which samples are present at each time point. (n*T)
        n_components (int): Number of dimensions for the latent space embeddings.

    Returns:
        tuple: A tuple containing the optimized coordinates (np.ndarray) and the final loss value (float) for all time points.
    """

    n_samples, _, n_timepoints = R.shape

    if sampled_by_time is None:
        sampled_by_time = np.ones((n_samples, n_timepoints), dtype=bool)

    if X_init is None:
        X_init = np.random.rand(n_samples, n_components, n_timepoints)

    X = np.full((n_samples, n_components, n_timepoints), np.nan)

    loss_before = []
    loss_after = []
    n_iters = []
    messages = []

    # define traversal order
    time_order = range(n_timepoints) if direction == "forward" else range(n_timepoints - 1, -1, -1)

    prev_t_idx = None

    for step, t_idx in enumerate(tqdm(time_order)):

        extant_mask = sampled_by_time[:, t_idx] # True if sample is present at this time point, False if sample is ancient and not present at this time point
        extant_idx = np.where(extant_mask)[0]   # Indices of samples that are present at this time point

        # --- Initialize in GLOBAL space ---
        if step == 0:
            X_global = X_init[:, :, t_idx].copy()
        else:
            # start from previous timepoint
            X_global = X[:, :, prev_t_idx].copy()

            prev_mask = sampled_by_time[:, prev_t_idx] # True if sample is present at the previous time point, False if sample is ancient and not present at this time point

            # samples that are present now but were not before
            new_mask = extant_mask & ~prev_mask

            if np.any(new_mask):
                X_global[new_mask, :] = X_init[new_mask, :, t_idx] # For new ancient samples that are not present in the previous time point, initialize with the provided initial coordinates for that time point

        # --- Restrict to extant samples (ONE clean projection step) ---
        X_t = X_global[extant_idx, :]
        R_t = R[:, :, t_idx][np.ix_(extant_idx, extant_idx)]
        X_init_t = X_init[extant_idx, :, t_idx]

        # --- Loss before ---
        loss_t_before, _ = compute_loss_gradient_timepoint(
            X_init_t, R_t, N_min, N_max, epsilon
        )

        # --- Optimize ---
        X_t, loss_t_after, n_iter, message = optimize_timepoint(
            X_t, R_t, N_min, N_max, epsilon, max_iter, gtol, ftol
        )

        # --- Store results back in GLOBAL tensor ---
        if step == 0:
            X_t -= np.mean(X_t, axis=0)

        X[extant_idx, :, t_idx] = X_t

        # --- Logging ---
        loss_before.append(loss_t_before)
        loss_after.append(loss_t_after)
        n_iters.append(n_iter)
        messages.append(message)

        prev_t_idx = t_idx

    # --- loss tracking ---
    loss_change = np.array(loss_before) - np.array(loss_after)
    total_loss_before = np.sum(loss_before)

    loss_over_time = total_loss_before - np.cumsum(loss_change)
    loss_df = pd.DataFrame({
        "step": range(1, n_timepoints + 1),
        "loss": loss_over_time,
        "n_iterations": n_iters,
        "message": messages,
    })

    return X, loss_df


def optimize_timepoints_sequentially(
    R,
    N_min=1,
    N_max=np.inf,
    n_components=2,
    max_iter=MAX_ITER_TIMEPOINT,
    gtol=GTOL_TIMEPOINT,
    ftol=FTOL_TIMEPOINT,
    epsilon=None,
    X_init=None,
    sampled_by_time=None,
    start_direction="backward",
    n_epochs=3):
    """ Repeatedly performs coordinate descent optimization over time slices.
    """
    loss_df = pd.DataFrame(columns=["epoch", "step", "loss", "n_iterations", "message"])
    cycle_order = (
    ["backward", "forward"]
    if start_direction == "backward"
    else ["forward", "backward"])
    cycles = [cycle_order[i % 2] for i in range(n_epochs)] # Alternate between forward and backward cycles for n_epochs epochs

    for epoch in range(n_epochs):
        # First cycle, either backward or forward depending on start_direction
        print(f"Starting epoch {epoch+1}/{n_epochs}, direction: {cycles[epoch]}")
        X,loss_df_cycle = coordinate_descent_over_time(R=R, N_min=N_min, N_max=N_max, max_iter=max_iter, n_components=n_components, gtol=gtol, ftol=ftol, epsilon=epsilon, X_init=X_init, sampled_by_time=sampled_by_time, direction=cycles[epoch])
        loss_df_cycle["epoch"] = epoch+1
        loss_df = pd.concat([loss_df, loss_df_cycle], ignore_index=True)

        X_init = X.copy() # Start next cycle with the optimized coordinates from the previous cycle

        # X,loss_df_cycle_2 = coordinate_descent_over_time(R=R, N_min=N_min, N_max=N_max, max_iter=max_iter, n_components=n_components, gtol=gtol, ftol=ftol, epsilon=epsilon, X_init=X_init, sampled_by_time=sampled_by_time, direction=cycle_order[1])
        # epoch_number = 2*epoch + 2
        # loss_df_cycle_2["epoch"] = epoch_number
        # loss_df = pd.concat([loss_df, loss_df_cycle_2], ignore_index=True)

        # X_init = X.copy()
        
    return X,loss_df

############################################################
# Coordinate descent over samples
############################################################

def compute_penalty_loss_gradient(X,smoothing_params):
    """Computes the contribution to the loss and the gradient from the penalty term that encourages smoothing.
    l_penalty = sum_k {l_k^{1} |DX|_1 + l_k^{2} ||DX||_2^2}

    Args:
        X (np.ndarray): Array of coordinates for all samples over all time points. (nxdxT)
                        or Array of coordinates for a given sample over all time points. (dxT)
        smoothing_params (dict): Dictionary containing smoothing parameters for the penalty term.
                                Keys should be "l1" and "l2" for L1 and L2 penalties respectively in addtion to "diff_operator" that contains the finite difference operator to compute DX.
    """

    loss = 0
    grad = np.zeros_like(X)
    n_timepoints = X.shape[-1]

    for k,D in smoothing_params["diff_operators"].items():
        l1 = smoothing_params["l1"].get(k, 0)
        l2 = smoothing_params["l2"].get(k, 0)
        DX = X @ D.T
        loss += l1 * np.sum(np.abs(DX)) + l2 * np.sum(DX**2)
        grad += l1 * np.sign(DX) @ D + 2 * l2 * DX @ D
    
    return loss, grad

def compute_unpenalized_gradient_sample(X_i,X,f_i,f_i_prime,D_i,R_i,valid_mask_i):
    """Computes the contribution to the gradient for a given sample from the unpenalized term in the loss function.

    Args:
        X_i (np.ndarray): Array of coordinates for the i-th sample. (dxT)
        X (np.ndarray): Array of coordinates for all samples. (nxdxT)
        f_i (np.ndarray): Array of function values for the i-th sample. (nxT)
        f_i_prime (np.ndarray): Array of function derivatives for the i-th sample. (nxT)
        D_i (np.ndarray): Distance matrix between the i-th sample and all other samples over all time points. (nxT)
        R_i (np.ndarray): Rates matrix between the i-th sample and all other samples over all time points. (nxT)
        valid_mask_i (np.ndarray): Boolean mask indicating valid entries for the i-th sample. (nxT)
    """
    M = compute_M(f_i,f_i_prime,D_i,R_i,valid_mask_i)
    M_sum = np.sum(M, axis=0)  # (T,)
    term_1 = M_sum * X_i  # (d, T)
    term_2 = (X * M[:,None,:]).sum(axis=0)  # (d, T) = sum_i (X_i * M_i) where M_i is (T,) and X_i is (d, T)
    # term_2 =  np.einsum("it,ipt -> pt", M, X) # (Equivilent)
    grad = term_1 - term_2
    return grad

def compute_loss_gradient_sample(X_i_flat,X,R_i,N_min,epsilon = None, smoothing_params=None):
    """Wrapper function to to compute both loss and gradient for minimization for a given sample

    Args:
        X_i_flat (np.ndarray): Flattened array of coordinates for the i-th sample. (d*T,)
        X (np.ndarray): Array of coordinates for all samples. (nxdxT)
        R_i (np.ndarray): Rates matrix between the i-th sample and all other samples over all time points. (nxT)
        N_min (float): Minimum value for the function computation.
        epsilon (float, optional): Scaling factor for loss and gradient, defining the standard error of rate estimates. Defaults to None.
        smoothing_params (dict, optional): Dictionary containing smoothing parameters for the penalty term. Defaults to None.
            Keys should be "l1" and "l2" for L1 and L2 penalties respectively in addtion to "diff_operator" that contains the finite difference operator to compute DX.

    Returns:
        tuple: A tuple containing the loss (float) and the gradient (np.ndarray) for the i-th sample.
    """
    n,d,T = X.shape
    X_i = X_i_flat.reshape((d, T))  # already (d, T)
    D_i = np.linalg.norm(X_i - X, axis=1) # (nxT) Distance between sample i and everyone else over time
    f_i, f_i_prime = compute_f(D_i, N_min) # (nxT) Function values and derivatives for sample i with respect to all other samples over time
    valid_mask_i = ~np.isnan(R_i)
    loss = compute_unpenalized_loss(f_i, R_i, valid_mask_i)
    grad = compute_unpenalized_gradient_sample(X_i,X,f_i,f_i_prime,D_i,R_i,valid_mask_i)

    if epsilon is not None:
        loss /= epsilon
        grad /= epsilon

    if smoothing_params is not None:
        penalty_loss, penalty_grad = compute_penalty_loss_gradient(X_i, smoothing_params)
        loss += penalty_loss
        grad += penalty_grad

    return loss, grad.ravel()


def optimize_sample(X_i_init, X, R_i, N_min, epsilon, smoothing_params, max_iter, tol):
    """A single coordinate descent step.
    For a given sample, finds the optimal coordinates over all time points by minimizing the unpenalized loss function using L-BFGS-B optimization.

    Args:
        X_i_init (np.ndarray): Initial coordinates for all samples at the i-th sample. (d*T,)
        R_i (np.ndarray): Rates matrix between pairs of samples at the i-th sample. (nxT)
        N_min (float): Minimum value for the function computation.
        epsilon (float): Scaling factor for loss and gradient, defining the standard error of rate estimates.
        smoothing_params (dict): Dictionary containing smoothing parameters for the penalty term. Keys should be "l1" and "l2" for L1 and L2 penalties respectively in addition to "diff_operator" that contains the finite difference operator to compute DX.
        max_iter (int): Maximum number of iterations for the optimizer.
        tol (float): Tolerance for the optimizer.

    Returns:
        tuple: A tuple containing the optimized coordinates (np.ndarray) and the final loss value (float) for the t-th time point.
    """
    # Optimize using L-BFGS-B for a given sample
    result = minimize(compute_loss_gradient_sample, 
                      X_i_init.ravel(), args=(X, R_i, N_min, epsilon, smoothing_params), 
                      method='L-BFGS-B', 
                      jac=True, 
                      options={'maxiter': max_iter, 'disp': False, 'gtol': tol, 'ftol': tol})
    X_opt = result.x.reshape(X_i_init.shape)
    return X_opt, result.fun


# def compute_loss_gradient_all(X_flat, R, N_min, epsilon = None, smoothing_params=None):
#     """Wrapper function to compute both loss and gradient for minimization for all time points

#     Args:
#         X_flat (np.ndarray): Flattened array of coordinates for all samples over all time points. (n*d*T,)
#         R (np.ndarray): Rates matrix between pairs of samples over all time points. (nxn*T)
#         N_min (float): Minimum value for the function computation.
#         epsilon (float, optional): Scaling factor for loss and gradient, defining the standard error of rate estimates. Defaults to None.
#         smoothing_params (dict, optional): Dictionary containing smoothing parameters for the penalty term. Defaults to None.
#             Keys should be "l1" and "l2" for L1 and L2 penalties respectively in addition to "diff_operator" that contains the finite difference operator to compute DX.

#     Returns:
#         tuple: A tuple containing the loss (float) and the gradient (np.ndarray) for all time points.
#     """
#     n,_,T = R.shape
#     X = X_flat.reshape((n, -1, T))  # (n, d, T)
#     valid_mask = ~np.isnan(R)
#     D = np.linalg.norm(X[:, None, :, :] - X[None, :, :, :], axis=2) # (nxn*T) Distance between pairs of samples over time
#     f,_ = compute_f(D,N_min)
#     loss = compute_unpenalized_loss(f,R,valid_mask)
#     grad = np.zeros_like(X)
#     for t in range(T):
#         grad[:,:,t] = compute_loss_gradient_timepoint(X[:,:,t], R[:,:,t], N_min, N_max=np.inf, epsilon=epsilon)[1].reshape((n, -1))

#     if epsilon is not None:
#         loss /= epsilon
#         grad /= epsilon
    
#     if smoothing_params is not None:
#         penalty_loss, penalty_grad = compute_penalty_loss_gradient(X, smoothing_params)
#         loss += penalty_loss
#         grad += penalty_grad

#     return loss, grad.ravel()


# def optimize_all(X_init, R, N_min, epsilon, smoothing_params, max_iter, tol):
#     """Joint minimization of the total unpenalized loss function using L-BFGS

#     Args:
#         X_init (np.ndarray): Initial coordinates for all samples over all time points. (n*d*T,)
#         R (np.ndarray): Rates matrix between pairs of samples over all time points. (n*n*T)
#         N_min (float): Minimum value for the function computation.
#         epsilon (float): Scaling factor for loss and gradient, defining the standard error of rate estimates.
#         smoothing_params (dict): Dictionary containing smoothing parameters for the penalty term. Keys should be "l1" and "l2" for L1 and L2 penalties respectively in addition to "diff_operator" that contains the finite difference operator to compute DX.
#         max_iter (int): Maximum number of iterations for the optimizer.
#         tol (float): Tolerance for the optimizer.
#     Returns:
#         tuple: A tuple containing the optimized coordinates (np.ndarray) and the final loss value (float) for all time points.
#     """
#     result = minimize(compute_loss_gradient_all, 
#                       X_init.ravel(), args=(R, N_min, epsilon, smoothing_params), 
#                       method='L-BFGS-B', 
#                       jac=True, 
#                       options={'maxiter': max_iter, 'disp': False, 'gtol': tol, 'ftol': tol})
#     X_opt = result.x.reshape(X_init.shape)
#     return X_opt, result.fun
