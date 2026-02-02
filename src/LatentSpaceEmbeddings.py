#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

# def compute_gradient(X,R,D_prime,valid_mask):
#     M = np.zeros_like(R)
#     M[valid_mask] = R[valid_mask]/D_prime[valid_mask] - 1/D_prime[valid_mask]**2
#     grad = (np.sum(M, axis=1)[:, np.newaxis] * X - M @ X)
#     return grad

# def compute_loss(R,D_prime):
#     loss_values = 1/D_prime + R * np.log(D_prime)
#     return 0.5*np.nansum(loss_values)

# def compute_D_prime(X,N0):
#     distances_squared = squareform(pdist(X, 'sqeuclidean'))
#     D_prime = distances_squared + N0
#     return D_prime

# def compute_loss_gradient(X_flat,R,N0):
#     n_samples = R.shape[0]
#     X = X_flat.reshape((n_samples, -1))
#     valid_mask = ~np.isnan(R)

#     D_prime = compute_D_prime(X,N0)
#     loss = compute_loss(R,D_prime)
#     grad = compute_gradient(X,R,D_prime,valid_mask)
#     return loss, grad.ravel()
 
inverse_distance=False

def compute_f(D, N0):
    D_sq = D**2

    if inverse_distance:
        f = 1/(D_sq + N0)
        f_prime = -2*D/(D_sq + N0)**2
    else:
        f = (1/N0)*np.exp(-D_sq)
        f_prime = -2*D*f
    return f,f_prime

def compute_M(f,f_prime,D,R,valid_mask):
    M = np.zeros_like(R)
    M[valid_mask] = (f_prime[valid_mask]/D[valid_mask])*(1-R[valid_mask]/f[valid_mask])
    return M

def compute_loss(f,R,valid_mask):
    loss_values = f[valid_mask] - R[valid_mask] * np.log(f[valid_mask])
    return 0.5*np.nansum(loss_values)

def compute_gradient(X,f,f_prime,D,R,valid_mask):
    M = compute_M(f,f_prime,D,R,valid_mask)
    grad = (np.sum(M, axis=1)[:, np.newaxis] * X - M @ X)
    return grad

def compute_loss_gradient(X_flat,R,N0,compute_f=compute_f):
    # Compute f is a function that computes f and f_prime given Distances and N0
    n_samples = R.shape[0]
    X = X_flat.reshape((n_samples, -1))
    valid_mask = ~np.isnan(R)
    D = squareform(pdist(X, 'euclidean'))
    f,f_prime = compute_f(D,N0)
    loss = compute_loss(f,R,valid_mask)
    grad = compute_gradient(X,f,f_prime,D,R,valid_mask)
    return loss, grad.ravel()

def compute_embeddings(X_init, R, N0, d=2, max_iter=10000, tol=1e-9):
    # Optimize using L-BFGS-B
    result = minimize(compute_loss_gradient, X_init.ravel(), args=(R, N0), method='L-BFGS-B',
                      jac=True, options={'maxiter': max_iter, 'disp': False, 'gtol': tol})
    print(f"BFGS solved with message: {result.message}, nit={result.nit},nfev={result.nfev},fun={result.fun}\n")
    X_opt = result.x.reshape(X_init.shape)
    return X_opt, result.fun

class LatentSpaceEmbeddings:
    def __init__(self,
                 rates_df: pd.DataFrame,
                 sample_times: list = None,
                 N0 = None,
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

        if N0 is not None:
            self.N0 = N0
        else:
            # Selecting N0 as 2nd percentile of Ne estimates
            with np.errstate(divide='ignore', invalid='ignore'):
                Ne = 1/(R + 1e-10)
            self.N0 = np.nanquantile(Ne, 0.01)
            # self.N0 = np.nanmin(Ne[Ne!=np.inf])/2

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

    def compute_embeddings(self,X_init,time_idx):
        # Compute embeddings for a specific time index
        R = self.R[:,:,time_idx]
        # Remove indices of samples that have not been sampled yet
        extant_samples = np.where(self.sampled_by_time[:,time_idx])[0]
        R = R[extant_samples,:,]
        R = R[:,extant_samples,]
        X = X_init[extant_samples, :]
        X,loss = compute_embeddings(X_init, R, self.N0)
        return X, loss

    def compute_embeddings_through_time(self,X_init = None):
        embeddings = np.full((self.num_samples,self.n_components,self.num_timepoints),np.nan)
        if X_init is not None:
            X = X_init
        else:
            X = np.random.rand(self.num_samples,self.n_components)
        for k in tqdm(range(self.num_timepoints)):
            idx = len(self.time_points) - k - 1
            # idx = k
            X, _ = self.compute_embeddings(X, idx)
            if k==0:
                X -= X.mean(axis=0)
            embeddings[:,:,idx] = X
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


# class LatentSpaceEmbeddings:
#     def __init__(self,
#                  R: np.ndarray,
#                  time_points: list,
#                  individuals: bool = False,
#                  compute_loss_gradient: callable = compute_loss_gradient,
#                  sample_times: list = None,
#                  n_components: int = 2):
        
#         self.R = R
#         self.time_points = time_points
#         self.compute_loss_gradient = compute_loss_gradient
#         self.num_idx = R.shape[0]
#         self.num_timepoints = R.shape[2]
#         self.n_components = n_components
#         if sample_times is None:
#             self.sample_times = np.array([0]*self.num_idx)
#         else:
#             self.sample_times = sample_times

#         self.embeddings = None
#         with np.errstate(divide='ignore', invalid='ignore'):
#             Ne = 1/R
#         self.N0 = np.nanquantile(Ne[Ne!=np.inf], 0.02)
#         sampled_by_time  = self.sample_times[:, np.newaxis] <= self.time_points
#         self.sampled_by_time = sampled_by_time
#         for t in range(self.num_timepoints):
#             R[~sampled_by_time[:,t], :, t] = np.nan
#             R[:, ~sampled_by_time[:,t], t] = np.nan

#         # if id_df is None:
#         #     sample_ids = np.arange(self.num_idx)
#         #     index_to_sample_id = {idx: sample_id for idx, sample_id in enumerate(sample_ids)}
#         #     id_df = pd.DataFrame(list(index_to_sample_id.items()), columns=["idx", "sample_id"])
#         #     id_df["individual_id"] = id_df["sample_id"] // 2
#         #     id_df["type"] = 0
#         #     id_df.sort_values(by="idx", inplace=True)
#         #     id_df.reset_index(drop=True, inplace=True)
        
#         if individuals:
#             R_individuals = np.full((int(self.num_idx/2), int(self.num_idx/2), self.num_timepoints), np.nan)
#             for i in range(int(self.num_idx/2)):
#                 individual_i_rates = R[2*i:2*i+2, :, :]
#                 for j in range(i+1, int(self.num_idx/2)):
#                     individual_i_j_rates = individual_i_rates[:, 2*j:2*j+2, :]
#                     if np.all(np.isnan(individual_i_j_rates)):
#                         continue
#                     R_individuals[i, j, :] = np.nanmean(individual_i_j_rates, axis=(0,1))
#                     R_individuals[j, i, :] = R_individuals[i, j, :]

#             self.R = R_individuals
#             self.num_idx = R_individuals.shape[0]
#             self.sampled_by_time = self.sampled_by_time[::2, :]

        
#         # self.id_df = id_df


#     def compute_embeddings(self, X_init, time_inx, max_iter = 10000, tol = 1e-9):
#         # Compute embeddings for a specific time index
#         # Remove indices of samples that have not been sampled yet
#         extant_samples = np.where(self.sampled_by_time[:,time_inx])[0]
#         R = self.R[extant_samples,:,time_inx]
#         R = self.R[:,extant_samples,time_inx]
#         X = X_init[extant_samples, :]
#         # Optimize using L-BFGS-B
#         result = minimize(self.compute_loss_gradient, X.ravel(), args=(R, self.N0), method='L-BFGS-B',
#                   jac=True, options={'maxiter': max_iter, 'disp': False, 'gtol': tol})
#         # Reshape result back to original dimensions, filling in NaNs for unsampled indices
#         X_opt = np.full((self.num_idx,self.n_components),np.nan)
#         X_opt[extant_samples, :] = result.x.reshape(X.shape)
#         return X_opt, result.fun

#     def compute_embeddings_through_time(self,X_init = None):
#         X_over_time = np.full((self.num_idx,self.n_components,self.num_timepoints),np.nan)
#         if X_init is not None:
#             X = X_init
#         else:
#             X = np.random.rand(self.num_idx,self.n_components)

#         for k in tqdm(range(self.num_timepoints)):
#             X, _ = self.compute_embeddings(X, k)
#             if k==0:
#                 X -= X.mean(axis=0)
#             X_over_time[:,:,k] = X

#         self.embeddings = X_over_time
#         return X_over_time
    
#     def create_embeddings_df(self):
#         assert self.embeddings is not None, "Embeddings have not been computed yet. Please run compute_embeddings_through_time() first."
#         n, _, num_timepoints = self.embeddings.shape
#         sample_ids = np.arange(n)
#         times = np.tile(self.time_points, n)
#         X1_flat = self.embeddings[:,0,:].flatten()
#         X2_flat = self.embeddings[:,1,:].flatten()

#         # Create the DataFrame
#         df = pd.DataFrame({
#             "idx": np.repeat(sample_ids, num_timepoints),
#             "time": times,
#             "X1": X1_flat,
#             "X2": X2_flat
#         })
#         return df
