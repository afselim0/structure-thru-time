#%%
# Add an option for an indidivual-based rather than haplotype-based computation of X
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
# %%
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

class LatentSpaceEmbeddings:
    def __init__(self,
                 R: np.ndarray,
                 time_points: list,
                 individuals: bool = False,
                 compute_loss_gradient: callable = compute_loss_gradient,
                 sample_times: list = None,
                 n_components: int = 2):
        
        self.R = R
        self.time_points = time_points
        self.compute_loss_gradient = compute_loss_gradient
        self.num_idx = R.shape[0]
        self.num_timepoints = R.shape[2]
        self.n_components = n_components
        if sample_times is None:
            self.sample_times = np.array([0]*self.num_idx)
        else:
            self.sample_times = sample_times

        self.embeddings = None
        with np.errstate(divide='ignore', invalid='ignore'):
            Ne = 1/R
        self.N0 = np.nanquantile(Ne[Ne!=np.inf], 0.02)
        sampled_by_time  = self.sample_times[:, np.newaxis] <= self.time_points
        self.sampled_by_time = sampled_by_time
        for t in range(self.num_timepoints):
            R[~sampled_by_time[:,t], :, t] = np.nan
            R[:, ~sampled_by_time[:,t], t] = np.nan

        # if id_df is None:
        #     sample_ids = np.arange(self.num_idx)
        #     index_to_sample_id = {idx: sample_id for idx, sample_id in enumerate(sample_ids)}
        #     id_df = pd.DataFrame(list(index_to_sample_id.items()), columns=["idx", "sample_id"])
        #     id_df["individual_id"] = id_df["sample_id"] // 2
        #     id_df["type"] = 0
        #     id_df.sort_values(by="idx", inplace=True)
        #     id_df.reset_index(drop=True, inplace=True)
        
        if individuals:
            R_individuals = np.full((int(self.num_idx/2), int(self.num_idx/2), self.num_timepoints), np.nan)
            for i in range(int(self.num_idx/2)):
                individual_i_rates = R[2*i:2*i+2, :, :]
                for j in range(i+1, int(self.num_idx/2)):
                    individual_i_j_rates = individual_i_rates[:, 2*j:2*j+2, :]
                    if np.all(np.isnan(individual_i_j_rates)):
                        continue
                    R_individuals[i, j, :] = np.nanmean(individual_i_j_rates, axis=(0,1))
                    R_individuals[j, i, :] = R_individuals[i, j, :]

            self.R = R_individuals
            self.num_idx = R_individuals.shape[0]
            self.sampled_by_time = self.sampled_by_time[::2, :]

        
        # self.id_df = id_df


    def compute_embeddings(self, X_init, time_inx, max_iter = 10000, tol = 1e-9):
        # Compute embeddings for a specific time index
        # Remove indices of samples that have not been sampled yet
        extant_samples = np.where(self.sampled_by_time[:,time_inx])[0]
        R = self.R[extant_samples,:,time_inx]
        R = self.R[:,extant_samples,time_inx]
        X = X_init[extant_samples, :]
        # Optimize using L-BFGS-B
        result = minimize(self.compute_loss_gradient, X.ravel(), args=(R, self.N0), method='L-BFGS-B',
                  jac=True, options={'maxiter': max_iter, 'disp': False, 'gtol': tol})
        # Reshape result back to original dimensions, filling in NaNs for unsampled indices
        X_opt = np.full((self.num_idx,self.n_components),np.nan)
        X_opt[extant_samples, :] = result.x.reshape(X.shape)
        return X_opt, result.fun

    def compute_embeddings_through_time(self,X_init = None):
        X_over_time = np.full((self.num_idx,self.n_components,self.num_timepoints),np.nan)
        if X_init is not None:
            X = X_init
        else:
            X = np.random.rand(self.num_idx,self.n_components)

        for k in tqdm(range(self.num_timepoints)):
            X, _ = self.compute_embeddings(X, k)
            if k==0:
                X -= X.mean(axis=0)
            X_over_time[:,:,k] = X

        self.embeddings = X_over_time
        return X_over_time
    
    def create_embeddings_df(self):
        assert self.embeddings is not None, "Embeddings have not been computed yet. Please run compute_embeddings_through_time() first."
        n, _, num_timepoints = self.embeddings.shape
        sample_ids = np.arange(n)
        times = np.tile(self.time_points, n)
        X1_flat = self.embeddings[:,0,:].flatten()
        X2_flat = self.embeddings[:,1,:].flatten()

        # Create the DataFrame
        df = pd.DataFrame({
            "idx": np.repeat(sample_ids, num_timepoints),
            "time": times,
            "X1": X1_flat,
            "X2": X2_flat
        })
        return df
    

