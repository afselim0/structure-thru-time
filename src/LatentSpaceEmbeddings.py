#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

EPSILON = 1e-16


def compute_f(D, N_min, N_max, link="inverse_sq_distance"):
    # N_max = np.inf
    D_sq = D**2
    if link == "inverse_distance":
        D_max = N_max - N_min
        f = 1/(D + N_min)
        f_prime = -1/(D + N_min)**2

        f[D > D_max] = 0
        f_prime[D > D_max] = 0
    elif link == "inverse_sq_distance":
        D_max = np.sqrt(N_max-N_min)
        f = 1/(D_sq + N_min)
        f_prime = -2*D/(D_sq + N_min)**2

        f[D > D_max] = 0
        f_prime[D > D_max] = 0
    elif link == "exponential":
        f = (1/N_min)*np.exp(-D_sq)
        f_prime = -2*D*f
    return f,f_prime

def compute_M(f,f_prime,D,R,valid_mask):
    M = np.zeros_like(R)
    M[valid_mask] = (f_prime[valid_mask]/D[valid_mask])*(1-R[valid_mask]/(f[valid_mask] + EPSILON))
    return M

def compute_loss(f,R,valid_mask):
    loss_values = f[valid_mask] - R[valid_mask] * np.log(f[valid_mask] + EPSILON)
    return 0.5*np.nansum(loss_values)

def compute_gradient(X,f,f_prime,D,R,valid_mask):
    M = compute_M(f,f_prime,D,R,valid_mask)
    grad = (np.sum(M, axis=1)[:, np.newaxis] * X - M @ X)
    return grad

def compute_loss_gradient(X_flat,R,N_min,N_max,compute_f=compute_f,change_penalty = 0, X_prev_flat = None):
    # Compute f is a function that computes f and f_prime given Distances and N0
    n_samples = R.shape[0]
    X = X_flat.reshape((n_samples, -1))
    valid_mask = ~np.isnan(R)
    D = squareform(pdist(X, 'euclidean'))
    f,f_prime = compute_f(D,N_min,N_max)
    loss = compute_loss(f,R,valid_mask)
    grad = compute_gradient(X,f,f_prime,D,R,valid_mask)

    # if X_prev_flat is not None and change_penalty > 0:
    #     X_prev = X_prev_flat.reshape((n_samples, -1))
    #     change_diff = X - X_prev

        # loss += change_penalty * 0.5 * np.sum(change_diff**2)
        # grad += change_penalty * change_diff
        # loss += change_penalty * np.sum(np.abs(change_diff))
        # grad += change_penalty * np.sign(change_diff)

    return loss, grad.ravel()

def compute_embeddings(X_init, R, N_min, N_max,change_penalty=0, X_prev=None, d=2, max_iter=30000, tol=1e-16):
    # Optimize using L-BFGS-B
    # X_prev_flat = X_prev.ravel() if X_prev is not None else None
    X_prev_flat = None
    result = minimize(compute_loss_gradient, X_init.ravel(), args=(R, N_min, N_max, compute_f, change_penalty, X_prev_flat), method='L-BFGS-B',
                      jac=True, options={'maxiter': max_iter, 'disp': False,'ftol':tol, 'gtol': tol})
    # print(f"BFGS solved with message: {result.message}, nit={result.nit},nfev={result.nfev},fun={result.fun}\n")
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

class LatentSpaceEmbeddings:
    def __init__(self,
                 rates_df: pd.DataFrame,
                 sample_times: list = None,
                 N_min = None,
                 N_max = None,
                 n_components: int = 2):
        self.samples  = np.unique([x for pair in rates_df.index.values for x in pair])
        self.num_samples = len(self.samples)
        self.sample_idx_map = {sample: idx for idx, sample in enumerate(self.samples)}
        self.idx_sample_map = {idx: sample for idx, sample in enumerate(self.samples)}
        self.time_points = rates_df.columns.values
        self.num_timepoints = len(self.time_points)
        self.n_components = n_components

        R = np.full((self.num_samples, self.num_samples, self.num_timepoints), np.nan)
        for (s1, s2), row in rates_df.iterrows():
            i = self.sample_idx_map[s1]
            j = self.sample_idx_map[s2]
            R[i, j, :] = row.values
            R[j, i, :] = row.values
                
        self.R = R
    
        if sample_times is None:
            self.sample_times = np.array([0]*self.num_samples)
        else:
            self.sample_times = sample_times

        self.embeddings = None

        if N_min is not None and N_max is not None:
            self.N_min = N_min
            self.N_max = N_max
        else:
            # Selecting N_min as 2nd percentile of Ne estimates
            with np.errstate(divide='ignore', invalid='ignore'):
                Ne = 1/(R + 1e-10)
            self.N_min = np.nanquantile(Ne, 0.01)
            self.N_max = np.inf

        sampled_by_time  = self.sample_times[:, np.newaxis] <= self.time_points
        self.sampled_by_time = sampled_by_time
        for t in range(self.num_timepoints):
            R[~sampled_by_time[:,t], :, t] = np.nan
            R[:, ~sampled_by_time[:,t], t] = np.nan

        # if individuals:
        #     R_individuals = np.full((int(self.num_idx/2), int(self.num_idx/2), self.num_timepoints), np.nan)
        #     for i in range(int(self.num_idx/2)):
        #         individual_i_rates = R[2*i:2*i+2, :, :]
        #         for j in range(i+1, int(self.num_idx/2)):
        #             individual_i_j_rates = individual_i_rates[:, 2*j:2*j+2, :]
        #             if np.all(np.isnan(individual_i_j_rates)):
        #                 continue
        #             R_individuals[i, j, :] = np.nanmean(individual_i_j_rates, axis=(0,1))
        #             R_individuals[j, i, :] = R_individuals[i, j, :]

        #     self.R = R_individuals
        #     self.num_idx = R_individuals.shape[0]
        #     self.sampled_by_time = self.sampled_by_time[::2, :]

    def compute_embeddings(self,X_init,time_idx, change_penalty=0, X_prev=None):
        # Compute embeddings for a specific time index
        R = self.R[:,:,time_idx]
        # Remove indices of samples that have not been sampled yet
        extant_samples = np.where(self.sampled_by_time[:,time_idx])[0]
        R = R[extant_samples,:,]
        R = R[:,extant_samples,]
        X = X_init[extant_samples, :]
        X,loss = compute_embeddings(X_init, R, self.N_min, self.N_max, change_penalty=change_penalty, X_prev=X_prev)
        return X, loss

    def compute_embeddings_through_time(self, X_init = None, change_penalty=0):
        embeddings = np.full((self.num_samples,self.n_components,self.num_timepoints),np.nan)
        if X_init is not None:
            X = X_init
        else:
            X = np.random.rand(self.num_samples,self.n_components)
        for k in tqdm(range(self.num_timepoints)):
            idx = len(self.time_points) - k - 1
            # idx = k
            X, _ = self.compute_embeddings(X_init = X, time_idx=idx, change_penalty=change_penalty, X_prev=X if k > 0 else None)
            if k==0:
                X -= X.mean(axis=0)
            embeddings[:,:,idx] = X
        embeddings = procrustes_alignment(embeddings)
        self.embeddings = embeddings
        return embeddings

    def get_embeddings_df(self):
        assert self.embeddings is not None, "Embeddings have not been computed yet. Please run compute_embeddings_through_time() first."
        sample_ids = [self.idx_sample_map[i] for i in np.arange(self.num_samples)]
        times = np.tile(self.time_points, self.num_samples)
        X1_flat = self.embeddings[:,0,:].flatten()
        X2_flat = self.embeddings[:,1,:].flatten()

        # Create the DataFrame
        df = pd.DataFrame({
            "sample_id": np.repeat(sample_ids, self.num_timepoints),
            "time": times,
            "X1": X1_flat,
            "X2": X2_flat
        })
        return df
