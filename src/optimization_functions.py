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
MAX_ITER_TIMEPOINT = 6000

GTOL_SAMPLE        = 1e-16
FTOL_SAMPLE        = 1e-12
MAX_ITER_SAMPLE    = 6000

GTOL_JOINT         = 1e-16
FTOL_JOINT         = 1e-8
MAX_ITER_JOINT     = 6000


import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import cvxpy as cp

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

# Common loss functions

def compute_unpenalized_loss(f,R,W,valid_mask):
    """Computes the loss function without penalty.
    Can be computed for:
      1. a given time slice (nxn)
      2. a given sample (nxT)
      3. the whole dataset (nxnxT)

    Args:
        f (np.ndarray): Array of function values at a given time slice.
        R (np.ndarray): Rate matrix between pairs of samples at a given time slice.
        W (np.ndarray): Weight matrix for the time slice.
        valid_mask (np.ndarray): Boolean mask indicating valid entries.

    Returns:
        float: Unpenalized loss value.
    """
    loss_values = W[valid_mask] * (f[valid_mask] - R[valid_mask] * np.log(f[valid_mask]))
    return 0.5 * np.sum(loss_values)
    
def compute_total_unpenalized_loss(X,R,N_min,N_max,W):
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
    loss = compute_unpenalized_loss(f,R,W,valid_mask)

    return loss

############################################################
# Coordinate descent over time slices
## Coordinate descent over time slices, where at each time slice we optimize the coordinates for all samples using L-BFGS-B optimization, initializing with the solution from the previous time slice.
## This can be done in both forward and backward directions, and takes into account ancient samples that are not present in all time points.
############################################################

def compute_unpenalized_gradient_timepoint(X_t, f_t, f_prime_t, D_t, R_t, W_t, valid_mask_t):
    """Gradient of the weighted unpenalized loss at a single timepoint.

    Args:
        X_t (np.ndarray): Coordinates at time t. (n, d)
        f_t, f_prime_t (np.ndarray): Link function values and derivatives. (n, n)
        D_t (np.ndarray): Pairwise distance matrix. (n, n)
        R_t (np.ndarray): Rate matrix. (n, n)
        W_t (np.ndarray): Weight matrix. (n, n)
        valid_mask_t (np.ndarray): Boolean mask for valid entries. (n, n)

    Returns:
        np.ndarray: Gradient. (n, d)
    """
    M  = compute_M(f_t, f_prime_t, D_t, R_t, valid_mask_t)
    WM = np.zeros_like(M)
    WM[valid_mask_t] = W_t[valid_mask_t] * M[valid_mask_t]
    return (np.sum(WM, axis=1)[:, np.newaxis] * X_t - WM @ X_t)


def compute_loss_gradient_timepoint(X_t_flat, R_t, W_t, N_min, N_max=np.inf):
    """Loss and gradient for a single timepoint.

    Args:
        X_t_flat (np.ndarray): Flattened coordinates. (n*d,)
        R_t (np.ndarray): Rate matrix. (n, n)
        W_t (np.ndarray): Weight matrix. (n, n)
        N_min (float): Link function parameter.
        N_max (float): Link function parameter.

    Returns:
        tuple: (loss, gradient)
    """
    R_t        = R_t.copy()
    R_t       += 1 / N_max
    n, _       = R_t.shape
    X_t        = X_t_flat.reshape((n, -1))
    valid_mask = ~np.isnan(R_t)
    D_t        = squareform(pdist(X_t, 'euclidean'))
    f_t, f_prime_t = compute_f(D_t, N_min)
    loss = compute_unpenalized_loss(f_t, R_t, W_t, valid_mask)
    grad = compute_unpenalized_gradient_timepoint(X_t, f_t, f_prime_t, D_t, R_t, W_t, valid_mask)
    return loss, grad.ravel()


def optimize_timepoint(X_t_init, R_t, W_t, N_min, N_max,
                       max_iter, gtol, ftol):
    """Single L-BFGS-B step for one timepoint.

    Args:
        X_t_init (np.ndarray): Initial coordinates. (n, d)
        R_t (np.ndarray): Rate matrix. (n, n)
        W_t (np.ndarray): Weight matrix. (n, n)
        N_min, N_max (float): Link function parameters.
        max_iter (int): Maximum L-BFGS-B iterations.
        gtol, ftol (float): Convergence tolerances.

    Returns:
        tuple: (X_opt, loss, n_iterations, message)
    """
    result = minimize(
        compute_loss_gradient_timepoint,
        X_t_init.ravel(),
        args=(R_t, W_t, N_min, N_max),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'disp': False, 'gtol': gtol, 'ftol': ftol}
    )
    return result.x.reshape(X_t_init.shape), result.fun, result.nit, result.message


def coordinate_descent_over_time(
    R,
    W,
    N_min          = 1,
    N_max          = np.inf,
    n_components   = 2,
    X_init         = None,
    sampled_by_time = None,
    direction      = "backward",
    max_iter       = MAX_ITER_TIMEPOINT,
    gtol           = GTOL_TIMEPOINT,
    ftol           = FTOL_TIMEPOINT,
):
    """Single-pass coordinate descent over timepoints (default: backward).

    Args:
        R (np.ndarray): Rate tensor. (n, n, T)
        W (np.ndarray): Weight tensor. (n, n, T)
        N_min, N_max (float): Link function parameters.
        n_components (int): Latent space dimensionality.
        max_iter (int): Maximum L-BFGS-B iterations per timepoint.
        gtol, ftol (float): Convergence tolerances.
        X_init (np.ndarray, optional): Initial coordinates. (n, d, T)
        sampled_by_time (np.ndarray, optional): Boolean presence mask. (n, T)
        direction (str): "backward" (default) or "forward".

    Returns:
        tuple: (X, loss_df)
    """
    n_samples, _, n_timepoints = R.shape

    if sampled_by_time is None:
        sampled_by_time = np.ones((n_samples, n_timepoints), dtype=bool)
    if X_init is None:
        X_init = np.random.rand(n_samples, n_components, n_timepoints)

    X = np.full((n_samples, n_components, n_timepoints), np.nan)
    loss_before, loss_after, n_iters, messages = [], [], [], []

    time_order = range(n_timepoints) if direction == "forward" \
                 else range(n_timepoints - 1, -1, -1)
    prev_t_idx = None

    for step, t_idx in enumerate(tqdm(time_order)):
        extant_mask = sampled_by_time[:, t_idx]
        extant_idx  = np.where(extant_mask)[0]

        # --- Initialize ---
        if step == 0:
            X_global = X_init[:, :, t_idx].copy()
        else:
            X_global = X[:, :, prev_t_idx].copy()
            new_mask = extant_mask & ~sampled_by_time[:, prev_t_idx]
            if np.any(new_mask):
                X_global[new_mask, :] = X_init[new_mask, :, t_idx]

        # --- Restrict to extant samples ---
        X_t      = X_global[extant_idx, :]
        R_t      = R[:, :, t_idx][np.ix_(extant_idx, extant_idx)]
        W_t      = W[:, :, t_idx][np.ix_(extant_idx, extant_idx)]
        X_init_t = X_init[extant_idx, :, t_idx]

        # # Per-timepoint normalization keeps gradient at a workable scale for L-BFGS
        # w_tot_t  = 0.5 * np.nansum(W_t)
        # scale   = w_tot_t / w_tot

        # --- Loss before ---
        loss_t_before, _ = compute_loss_gradient_timepoint(
            X_init_t, R_t, W_t, N_min, N_max)

        # --- Optimize ---
        X_t, loss_t_after, n_iter, message = optimize_timepoint(
            X_t, R_t, W_t, N_min, N_max, max_iter, gtol, ftol)

        # --- Center (first step only) and Procrustes align ---
        if step == 0:
            X_t -= np.mean(X_t, axis=0)
        X_t = procrustes_single_time(X_global[extant_idx, :], X_t)
        X[extant_idx, :, t_idx] = X_t

        loss_before.append(loss_t_before)
        loss_after.append(loss_t_after)
        n_iters.append(n_iter)
        messages.append(message)
        prev_t_idx = t_idx

    loss_change    = np.array(loss_before) - np.array(loss_after)
    loss_over_time = np.sum(loss_before) - np.cumsum(loss_change)
    loss_df = pd.DataFrame({
        "step":         range(1, n_timepoints + 1),
        "loss":         loss_over_time,
        "n_iterations": n_iters,
        "message":      messages,
    })
    return X, loss_df


# def optimize_timepoints_sequentially(
#     R,
#     W,
#     N_min           = 1,
#     N_max           = np.inf,
#     n_components    = 2,
#     max_iter        = MAX_ITER_TIMEPOINT,
#     gtol            = GTOL_TIMEPOINT,
#     ftol            = FTOL_TIMEPOINT,
#     X_init          = None,
#     sampled_by_time = None,
#     start_direction = "backward",
#     n_epochs        = 1,
# ):
#     """Coordinate descent over time slices, optionally over multiple alternating epochs.

#     Args:
#         R (np.ndarray): Rate tensor. (n, n, T)
#         W (np.ndarray): Weight tensor. (n, n, T)
#         N_min, N_max (float): Link function parameters.
#         n_components (int): Latent space dimensionality.
#         max_iter (int): Maximum L-BFGS-B iterations per timepoint.
#         gtol, ftol (float): Convergence tolerances.
#         X_init (np.ndarray, optional): Initial coordinates. (n, d, T)
#         sampled_by_time (np.ndarray, optional): Boolean presence mask. (n, T)
#         start_direction (str): First sweep direction — "backward" (default) or "forward".
#         n_epochs (int): Number of sweeps. Default 1 (single backward pass).

#     Returns:
#         tuple: (X, loss_df)
#     """
#     cycle_order = (["backward", "forward"] if start_direction == "backward"
#                    else ["forward", "backward"])
#     cycles  = [cycle_order[i % 2] for i in range(n_epochs)]
#     loss_df = pd.DataFrame(columns=["epoch", "step", "loss", "n_iterations", "message"])

#     for epoch in range(n_epochs):
#         print(f"Epoch {epoch + 1}/{n_epochs}  direction: {cycles[epoch]}")
#         X, loss_df_epoch = coordinate_descent_over_time(
#             R               = R,
#             W               = W,
#             N_min           = N_min,
#             N_max           = N_max,
#             n_components    = n_components,
#             max_iter        = max_iter,
#             gtol            = gtol,
#             ftol            = ftol,
#             X_init          = X_init,
#             sampled_by_time = sampled_by_time,
#             direction       = cycles[epoch],
#         )
#         loss_df_epoch["epoch"] = epoch + 1
#         loss_df  = pd.concat([loss_df, loss_df_epoch], ignore_index=True)
#         X_init   = X.copy()

#     return X, loss_df
    
############################################################
# Coordinate descent over samples
############################################################

def compute_penalty_loss_gradient(X, smoothing_params):
    """Penalty loss and gradient for L1 (group norm) and L2 on finite differences.

    L1 uses the exact group norm — no pseudo-Huber needed since L2 on the same
    order provides the curvature near zero that L-BFGS requires.

    Args:
        X (np.ndarray): Coordinates, shape (d, T) for one sample or (n, d, T) for all.
        smoothing_params (dict): Keys "l1", "l2" (dicts of {order: lambda}),
                                 "diff_operators" ({order: matrix}).
    Returns:
        tuple: (loss, grad) with grad matching X's shape.
    """
    loss = 0.0
    grad = np.zeros_like(X)

    for k, D in smoothing_params["diff_operators"].items():
        l1 = smoothing_params["l1"].get(k, 0)
        l2 = smoothing_params["l2"].get(k, 0)
        DX = X @ D.T                                              # (..., d, T)

        if l2:
            loss += l2 * np.sum(DX ** 2)
            grad += 2 * l2 * DX @ D

        if l1:
            sq_norms  = np.sum(DX ** 2, axis=-2, keepdims=True)  # (..., 1, T)
            norms     = np.sqrt(sq_norms)                          # (..., 1, T)
            safe_norm = np.where(norms > 0, norms, 1.0)           # DX=0 → grad=0
            loss += l1 * np.sum(norms)
            grad += l1 * (DX / safe_norm) @ D

    return loss, grad


def compute_unpenalized_gradient_sample(X_i, X, f_i, f_i_prime, D_i, R_i, W_i, valid_mask_i):
    """Gradient of the weighted unpenalized loss w.r.t. sample i's trajectory.

    Uses the full symmetric sum with a 0.5 factor, consistent with
    compute_unpenalized_loss, so loss and gradient are properly matched.

    Args:
        X_i (np.ndarray): Coordinates for sample i. (d, T)
        X (np.ndarray): Coordinates for all samples. (n, d, T)
        f_i, f_i_prime (np.ndarray): Link values and derivatives. (n, T)
        D_i (np.ndarray): Distances from sample i to all others. (n, T)
        R_i (np.ndarray): Rates from sample i to all others. (n, T)
        W_i (np.ndarray): Weights for sample i's pairs. (n, T)
        valid_mask_i (np.ndarray): Valid entry mask. (n, T)

    Returns:
        np.ndarray: Gradient w.r.t. X_i. (d, T)
    """
    M   = compute_M(f_i, f_i_prime, D_i, R_i, valid_mask_i)
    WM = np.zeros_like(M)
    WM[valid_mask_i] = W_i[valid_mask_i] * M[valid_mask_i]
    return 0.5 * (np.sum(WM, axis=0) * X_i
                  - (X * WM[:, np.newaxis, :]).sum(axis=0))


def compute_loss_gradient_sample(X_i_flat, X, R_i, W_i, N_min, N_max=np.inf,
                                  smoothing_params=None):
    """Loss and gradient for sample i's trajectory over all timepoints.

    Args:
        X_i_flat (np.ndarray): Flattened coordinates for sample i. (d*T,)
        X (np.ndarray): Coordinates for all samples. (n, d, T)
        R_i (np.ndarray): Rates from sample i to all others. (n, T)
        W_i (np.ndarray): Weights for sample i's pairs. (n, T)
        N_min, N_max (float): Link function parameters.
        smoothing_params (dict, optional): Smoothing configuration.

    Returns:
        tuple: (loss, gradient)
    """
    R_i = R_i.copy()
    R_i += 1 / N_max
    n, d, T   = X.shape
    X_i       = X_i_flat.reshape((d, T))
    D_i       = np.linalg.norm(X_i - X, axis=1)        # (n, T)
    f_i, f_i_prime = compute_f(D_i, N_min)
    valid_mask_i   = ~np.isnan(R_i)

    loss = compute_unpenalized_loss(f_i, R_i, W_i, valid_mask_i)
    grad = compute_unpenalized_gradient_sample(
               X_i, X, f_i, f_i_prime, D_i, R_i, W_i, valid_mask_i)

    if smoothing_params is not None:
        pen_loss, pen_grad = compute_penalty_loss_gradient(X_i, smoothing_params)
        loss += pen_loss
        grad += pen_grad

    return loss, grad.ravel()


def optimize_sample(X_i_init, X, R_i, W_i, N_min, N_max, smoothing_params,
                    gtol=GTOL_SAMPLE, ftol=FTOL_SAMPLE, max_iter=MAX_ITER_SAMPLE):
    """L-BFGS-B step for one sample's trajectory.

    Args:
        X_i_init (np.ndarray): Initial coordinates. (d, T)
        X (np.ndarray): All sample coordinates. (n, d, T)
        R_i (np.ndarray): Rates for sample i. (n, T)
        W_i (np.ndarray): Weights for sample i. (n, T)
        N_min, N_max (float): Link function parameters.
        smoothing_params (dict): Smoothing configuration.
        gtol, ftol (float): Convergence tolerances.
        max_iter (int): Maximum iterations.

    Returns:
        tuple: (X_opt, loss, n_iterations, message)
    """
    result = minimize(
        compute_loss_gradient_sample,
        X_i_init.ravel(),
        args=(X, R_i, W_i, N_min, N_max, smoothing_params),
        method='L-BFGS-B',
        jac=True,
        options={'maxiter': max_iter, 'disp': False, 'gtol': gtol, 'ftol': ftol}
    )
    return result.x.reshape(X_i_init.shape), result.fun, result.nit, result.message


def coordinate_descent_over_samples(X_init, R, W, N_min, N_max,
                                     smoothing_params, sampled_by_time=None,
                                     n_epochs=1, max_iter=MAX_ITER_SAMPLE,
                                     gtol=GTOL_SAMPLE, ftol=FTOL_SAMPLE):
    """Coordinate descent over samples, optimizing each trajectory independently.

    Args:
        X_init (np.ndarray): Initial coordinates. (n, d, T)
        R (np.ndarray): Rate tensor. (n, n, T)
        W (np.ndarray): Weight tensor. (n, n, T)
        N_min, N_max (float): Link function parameters.
        smoothing_params (dict): Smoothing configuration.
        sampled_by_time (np.ndarray, optional): Presence mask. (n, T)
        n_epochs (int): Number of passes over all samples.
        max_iter (int): Maximum L-BFGS-B iterations per sample.
        gtol, ftol (float): Convergence tolerances.

    Returns:
        tuple: (X, loss_df)
    """
    n_samples, _, n_timepoints = R.shape
    X = X_init.copy()
    loss_df = pd.DataFrame(columns=["epoch", "step", "loss", "n_iterations", "message"])

    for epoch in range(n_epochs):
        loss_before, loss_after, n_iters, messages = [], [], [], []
        print(f"Epoch {epoch + 1}/{n_epochs}")

        for i in tqdm(range(n_samples)):
            time_mask = (sampled_by_time[i, :] if sampled_by_time is not None
                         else np.ones(n_timepoints, dtype=bool))

            R_i      = R[:, i, :][:, time_mask].copy()
            W_i      = W[:, i, :][:, time_mask].copy()
            X_i_init = X[i, :, :][:, time_mask].copy()
            X_slice  = X[:, :, time_mask]

            loss_before_i, _ = compute_loss_gradient_sample(
                X_i_init.ravel(), X_slice, R_i, W_i, N_min, N_max, smoothing_params)

            X_i_opt, loss_after_i, n_iter, message = optimize_sample(
                X_i_init, X_slice, R_i, W_i, N_min, N_max, smoothing_params,
                gtol=gtol, ftol=ftol, max_iter=max_iter)

            X[i, :, time_mask] = X_i_opt          # only update present timepoints
            loss_before.append(loss_before_i)
            loss_after.append(loss_after_i)
            n_iters.append(n_iter)
            messages.append(message)

        loss_change    = np.array(loss_before) - np.array(loss_after)
        loss_over_time = np.sum(loss_before) - np.cumsum(loss_change)
        loss_df_epoch  = pd.DataFrame({
            "step":         range(1, n_samples + 1),
            "loss":         loss_over_time,
            "n_iterations": n_iters,
            "message":      messages,
            "epoch":        epoch + 1,
        })
        loss_df = pd.concat([loss_df, loss_df_epoch], ignore_index=True)

    return X, loss_df


############################################################
# Joint optimization over all samples and time points
############################################################

def compute_loss_gradient_all(X_flat, R, W, N_min, N_max,
                               sampled_by_time=None, smoothing_params=None):
    """Total weighted loss and gradient over all samples and timepoints.

    Sums per-sample contributions; the 0.5 factor inside compute_loss_gradient_sample
    ensures each pair is counted once, so no correction is needed here.

    Args:
        X_flat (np.ndarray): Flattened coordinates. (n*d*T,)
        R (np.ndarray): Rate tensor. (n, n, T)
        W (np.ndarray): Weight tensor. (n, n, T)
        N_min, N_max (float): Link function parameters.
        sampled_by_time (np.ndarray, optional): Presence mask. (n, T)
        smoothing_params (dict, optional): Smoothing configuration.

    Returns:
        tuple: (total_loss, gradient)
    """
    n, _, T = R.shape
    X       = X_flat.reshape((n, -1, T))
    total_loss = 0.0
    total_grad = np.zeros_like(X)

    for i in range(n):
        time_mask = (sampled_by_time[i, :] if sampled_by_time is not None
                    else np.ones(T, dtype=bool))
        R_i   = R[:, i, :][:, time_mask].copy()
        W_i   = W[:, i, :][:, time_mask].copy()
        X_i   = X[i][:, time_mask]              # ← fixed: was X[i, :, time_mask]
        X_obs = X[:, :, time_mask]              # already fine (single advanced index)

        loss_i, grad_i = compute_loss_gradient_sample(
            X_i.ravel(), X_obs, R_i, W_i, N_min, N_max, smoothing_params)

        total_loss += loss_i
        total_grad[i][:, time_mask] += grad_i.reshape(X_i.shape)   # ← fixed
    return total_loss, total_grad.ravel()


def optimize_jointly(X_init, R, W,
                     N_min, N_max, smoothing_params,
                     sampled_by_time=None, max_iter=MAX_ITER_JOINT,
                     gtol=GTOL_JOINT, ftol=FTOL_JOINT):
    """Joint L-BFGS-B optimization over all samples and timepoints.

    Args:
        X_init (np.ndarray): Initial coordinates. (n, d, T)
        R (np.ndarray): Rate tensor. (n, n, T)
        W (np.ndarray): Weight tensor. (n, n, T)
        N_min, N_max (float): Link function parameters.
        smoothing_params (dict): Smoothing configuration.
        sampled_by_time (np.ndarray, optional): Presence mask. (n, T)
        max_iter (int): Maximum iterations.
        gtol, ftol (float): Convergence tolerances.

    Returns:
        tuple: (X_opt, progress_df)
    """
    progress = []

    def callback(x):
        loss, grad = compute_loss_gradient_all(
            x, R, W, N_min, N_max, sampled_by_time, smoothing_params)
        progress.append({
            "iteration":    len(progress) + 1,
            "loss":         loss,
            "proj_grad_max": np.max(np.abs(grad)),
        })
        print(f"  iter {progress[-1]['iteration']:4d}  "
              f"loss={loss:.10g}  "
              f"max|grad|={progress[-1]['proj_grad_max']:.6g}")

    result = minimize(
        compute_loss_gradient_all,
        X_init.ravel(),
        args=(R, W, N_min, N_max, sampled_by_time, smoothing_params),
        method='L-BFGS-B',
        jac=True,
        callback=callback,
        options={'maxiter': max_iter, 'disp': False, 'gtol': gtol, 'ftol': ftol}
    )
    print(f"Finished: {result.message}  "
          f"(loss={result.fun:.10g}, iterations={result.nit})")

    return result.x.reshape(X_init.shape), pd.DataFrame(progress)

############################################################
# Smoothing the trajectories individually
############################################################
def smooth_trajectories(X, smoothing_params, sampled_by_time=None):
    """
    Smooth trajectories in X using L1 and/or L2 difference penalties.

    Parameters
    ----------
    X : np.ndarray, shape (n, d, T)
        Positions of n particles in d-dimensional space over T timepoints.
        NaN values indicate missing observations.
    smoothing_params : dict or None
        Output of _parse_smoothing_params. Expected keys:
            "l2"            : dict {order: lambda_value}
            "l1"            : dict {order: lambda_value}
            "diff_operators": dict {order: np.ndarray (T, T)}
        If None, X is returned unchanged.
    sampled_by_time : np.ndarray of bool, shape (n, T), optional
        Whether particle i exists at timepoint t. If None, all timepoints
        are assumed observed for all particles.

    Returns
    -------
    X_smooth : np.ndarray, shape (n, d, T)
        Smoothed trajectories. Timepoints where sampled_by_time is False
        are left as NaN.
    """
    if smoothing_params is None:
        return X.copy()

    l2      = smoothing_params.get("l2", {})
    l1      = smoothing_params.get("l1", {})
    D_dict  = smoothing_params.get("diff_operators", {})
    has_l1  = bool(l1)
    has_l2  = bool(l2)

    n, d, T = X.shape
    X_smooth = np.full_like(X, np.nan)

    if sampled_by_time is None:
        sampled_by_time = np.ones((n, T), dtype=bool)

    for i in tqdm(range(n)):
        obs = sampled_by_time[i]          # (T,) boolean mask
        obs_idx = np.where(obs)[0]        # indices of observed timepoints
        T_i = len(obs_idx)

        if T_i == 0:
            continue

        # Slice difference operators to observed timepoints only:
        #   rows select which finite differences to penalise (only those
        #   entirely within the observed support are meaningful),
        #   columns select the observed timepoints.
        D_dict_i = {order: Dk[np.ix_(obs_idx, obs_idx)]
                    for order, Dk in D_dict.items()}

        if has_l2 and not has_l1:
            A = np.eye(T_i)
            for order, lam in l2.items():
                Dk = D_dict_i[order]
                A += lam * (Dk.T @ Dk)
            L = np.linalg.cholesky(A)

            for dim in range(d):
                x = X[i, dim, obs_idx]
                x_smooth = np.linalg.solve(L.T, np.linalg.solve(L, x))
                X_smooth[i, dim, obs_idx] = x_smooth

        elif has_l1:
            for dim in range(d):
                x = X[i, dim, obs_idx]
                x_var = cp.Variable(T_i)

                cost = cp.sum_squares(x_var - x)

                for order, lam in l2.items():
                    Dk = D_dict_i[order]
                    cost = cost + lam * cp.sum_squares(Dk @ x_var)

                for order, lam in l1.items():
                    Dk = D_dict_i[order]
                    cost = cost + lam * cp.norm1(Dk @ x_var)

                prob = cp.Problem(cp.Minimize(cost))
                prob.solve()

                if x_var.value is None:
                    raise RuntimeError(
                        f"CVXPY failed for particle {i}, dim {dim}. "
                        f"Solver status: {prob.status}"
                    )
                X_smooth[i, dim, obs_idx] = x_var.value

        else:
            X_smooth[i, dim, obs_idx] = X[i, dim, obs_idx]

    return X_smooth