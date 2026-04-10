#!/usr/bin/env python3
"""
This module contains functions for computing the loss and the gradient for the optimization of latent space embeddings based on pairwise rates between samples over time.
Loss and gradient can be computed for:
1. All samples at a given time slice
2. A given sample over all time slices
3. All samples over all time slices
The module also contains functions for performing optimization using L-BFGS-B for a given sample, a given time slice, or all samples and time slices jointly.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform

def compute_f(D, N_min, link="inverse_distance"): 
    """Compute the link function f and its derivative f_prime based on the distance matrix D and minimum value N_min. f(x) is the function that maps distances to rates.

    Args:
        D (np.ndarray): Distance matrix between pairs of samples.
        N_min (float): Maxmimum rate that could be modelled, where samples with rate = 1/N_min or higher will have zero distances between them.
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
    elif link == "exponential":                     # f = 1/N_min * exp(-D^2)
        D_sq = D**2
        f = (1/N_min)*np.exp(-D_sq)
        f_prime = -2*D*f
    return f,f_prime

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
    """Computes the contribution to the first term of the loss function.
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
    R_t += 1/N_max
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

def optimize_timepoint(X_t_init, R_t, N_min, N_max, epsilon, max_iter, tol):
    """A single coordinate descent step.
    For a given time point, finds the optimal coordinates for all samples by minimizing the unpenalized loss function using L-BFGS-B optimization.

    Args:
        X_t_init (np.ndarray): Initial coordinates for all samples at the t-th time point. (n*d,)
        R_t (np.ndarray): Rates matrix between pairs of samples at the t-th time point. (nxn)
        N_min (float): Hyperparameter for the function computation, where samples with rate = 1/N_min or higher will have zero distances between them.
        N_max (float): Hyperparameter for the function computation, where samples with rate = 0 will have distances of N_max between them.
        epsilon (float): Scaling factor for loss and gradient, defining the standard error of rate estimates.
        max_iter (int): Maximum number of iterations for the optimizer.
        tol (float): Tolerance for the optimizer.

    Returns:
        tuple: A tuple containing the optimized coordinates (np.ndarray) and the final loss value (float) for the t-th time point.
    """
    result = minimize(compute_loss_gradient_timepoint, 
                      X_t_init.ravel(), args=(R_t, N_min, N_max, epsilon), 
                      method='L-BFGS-B', 
                      jac=True, 
                      options={'maxiter': max_iter, 'disp': False, 'gtol': tol, 'ftol': tol})
    X_opt = result.x.reshape(X_t_init.shape)
    return X_opt, result.fun

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

def optimize_all(X_init, R, N_min, epsilon, smoothing_params, max_iter, tol):
    """Joint minimization of the total unpenalized loss function using L-BFGS

    Args:
        X_init (np.ndarray): Initial coordinates for all samples over all time points. (n*d*T,)
        R (np.ndarray): Rates matrix between pairs of samples over all time points. (n*n*T)
        N_min (float): Minimum value for the function computation.
        epsilon (float): Scaling factor for loss and gradient, defining the standard error of rate estimates.
        smoothing_params (dict): Dictionary containing smoothing parameters for the penalty term. Keys should be "l1" and "l2" for L1 and L2 penalties respectively in addition to "diff_operator" that contains the finite difference operator to compute DX.
        max_iter (int): Maximum number of iterations for the optimizer.
        tol (float): Tolerance for the optimizer.
    Returns:
        tuple: A tuple containing the optimized coordinates (np.ndarray) and the final loss value (float) for all time points.
    """
    result = minimize(compute_loss_gradient_all, 
                      X_init.ravel(), args=(R, N_min, epsilon, smoothing_params), 
                      method='L-BFGS-B', 
                      jac=True, 
                      options={'maxiter': max_iter, 'disp': False, 'gtol': tol, 'ftol': tol})
    X_opt = result.x.reshape(X_init.shape)
    return X_opt, result.fun

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