#!/usr/bin/env python3
import numpy as np
import pandas as pd
from optimization_functions import *
from itertools import product as iterproduct
from tqdm import tqdm

STDERR_ROLLING_WINDOW= 10
TIMEPOINT_TOL = 1e-12
SMOOTHING_TOL = 1e-12
MAX_ITER = 6000


class LatentSpaceEmbeddings:
    def __init__(self,
                 rates_df: pd.DataFrame,
                 sample_times: list = None,
                 N_min = 1,
                 N_max = np.inf,
                 l1_penalty = None,
                 l2_penalty = None,
                 individuals = False,
                 n_timepoints = None,
                 n_components: int = 2):
        
        # self.stderr = np.nanmean(rates_df.T.rolling(window=STDERR_ROLLING_WINDOW).std().mean())
        # self.multiplier = np.nanquantile(rates_df.values,0.01)**2 # A multiplier to scale the loss and gradient when computing the first pass over the time points, to help with convergence.
        self.multiplier = 1
        if n_timepoints is not None:
            time_step = len(rates_df.columns) // n_timepoints
            rates_df = rates_df.iloc[:, ::int(time_step)]
        rates_df.columns = rates_df.columns.astype(float)
        self.samples  = np.unique([x for pair in rates_df.index.values for x in pair])
        self.num_samples = len(self.samples)
        self.sample_idx_map = {sample: idx for idx, sample in enumerate(self.samples)}
        self.idx_sample_map = {idx: sample for idx, sample in enumerate(self.samples)}
        self.time_points = rates_df.columns.values
        self.num_timepoints = len(self.time_points)
        self.n_components = n_components
        self.delta_t = self.time_points[1] - self.time_points[0]

        R = np.full((self.num_samples, self.num_samples, self.num_timepoints), np.nan)
        for (s1, s2), row in rates_df.iterrows():
            i = self.sample_idx_map[s1]
            j = self.sample_idx_map[s2]
            R[i, j, :] = row.values
            R[j, i, :] = row.values
                
        self.R = R
        self.N_min = N_min
        self.N_max = N_max
    
        if sample_times is None:
            self.sample_times = np.zeros(self.num_samples, dtype=float)
        else:
            self.sample_times = sample_times
            print(f"Number of sample times provided: {len(self.sample_times)}")
            print(f"Number of samples: {self.num_samples}")


        self.embeddings = None
        self.covariances = None

        sampled_by_time  = self.sample_times[:, np.newaxis] <= self.time_points
        self.sampled_by_time = sampled_by_time
        for t in range(self.num_timepoints):
            R[~sampled_by_time[:,t], :, t] = np.nan
            R[:, ~sampled_by_time[:,t], t] = np.nan

        # Parse penalties
        self._parse_smoothing_params(l1_penalty, l2_penalty)

        if individuals:
            R_individuals = np.full((int(self.num_samples/2), int(self.num_samples/2), self.num_timepoints), np.nan)
            for i in range(int(self.num_samples/2)):
                individual_i_rates = R[2*i:2*i+2, :, :]
                for j in range(i+1, int(self.num_samples/2)):
                    individual_i_j_rates = individual_i_rates[:, 2*j:2*j+2, :]
                    if np.all(np.isnan(individual_i_j_rates)):
                        continue
                    R_individuals[i, j, :] = np.nanmean(individual_i_j_rates, axis=(0,1))
                    R_individuals[j, i, :] = R_individuals[i, j, :]

            self.R = R_individuals
            self.num_samples = R_individuals.shape[0]
            self.sampled_by_time = self.sampled_by_time[::2, :]

    def _parse_smoothing_params(self, l1_penalty, l2_penalty):
        if l1_penalty is not None or l2_penalty is not None:
            assert l1_penalty is None or isinstance(l1_penalty, dict), "l1_penalty must be a dictionary"
            assert l2_penalty is None or isinstance(l2_penalty, dict), "l2_penalty must be a dictionary"
            max_l1_order = max(l1_penalty.keys()) if l1_penalty is not None else 0
            max_l2_order = max(l2_penalty.keys()) if l2_penalty is not None else 0
            max_order = max(max_l1_order, max_l2_order)
            difference_operators = compute_difference_operators(self.num_timepoints, max_order, self.delta_t)
            self.smoothing_params = {
                "l1": l1_penalty if l1_penalty is not None else {},
                "l2": l2_penalty if l2_penalty is not None else {},
                "diff_operators": difference_operators
            }
        else:
            self.smoothing_params = None

    def compute_embeddings_jointly(self, X_init=None):
        if X_init is None:
            X_init = np.random.rand(self.num_samples, self.n_components, self.num_timepoints)
        else:
            assert X_init.shape == (self.num_samples, self.n_components, self.num_timepoints), "X_init must have shape (num_samples, n_components, num_timepoints)"
        X, loss = optimize_all(X_init, self.R, self.N_min, self.N_max, self.smoothing_params)
        X = procrustes_alignment(X)
        self.embeddings = X

    def compute_embeddings(self, X_init, time_idx):
        R = self.R[:, :, time_idx]

        extant = np.where(self.sampled_by_time[:, time_idx])[0]
        R_sub = R[np.ix_(extant, extant)]

        X0 = X_init[extant, :]
        X_sub, loss = compute_embeddings(X0, R_sub, self.N0)

        # Re-expand to full size
        X_full = np.full_like(X_init, np.nan, dtype=float)
        X_full[extant, :] = X_sub

        return X_full, loss

    def compute_embeddings_through_time(self, X_init=None, direction="backward", tol=TIMEPOINT_TOL, max_iter=MAX_ITER,
                                         compute_uncertainty=False, n_observations=None):
        """Computes embeddings through time by optimizing each time slice with and initializing with previous solution

        Args:
            X_init (np.ndarray, optional): initial solution for first time slice. Defaults to None.
            direction (str, optional): "backward" or "forward", determines the order of optimization. Defaults to "backward".
            tol (float, optional): tolerance for convergence. Defaults to TIMEPOINT_TOL.
            max_iter (int, optional): maximum number of iterations. Defaults to MAX_ITER.
            compute_uncertainty (bool): if True, compute per-sample Laplace covariances via the L-BFGS inverse Hessian.
            n_observations (float, optional): effective number of independent coalescence observations per rate estimate.
                Scales the posterior covariance by 1/n_observations. Without this the covariances reflect 1 Poisson
                observation per rate, which is typically far too uncertain.

        Returns:
            np.ndarray, float: (n x n_components x T) embeddings, total loss
        """
        X = np.full((self.num_samples, self.n_components, self.num_timepoints), np.nan)
        total_loss = 0
        if X_init is not None:
            X_t = X_init
        else:
            X_t = np.random.rand(self.num_samples, self.n_components)

        if compute_uncertainty:
            covariances = np.full(
                (self.num_samples, self.n_components, self.n_components, self.num_timepoints), np.nan
            )

        for k in tqdm(range(self.num_timepoints)):
            if direction == "backward":
                time_idx = len(self.time_points) - k - 1
            elif direction == "forward":
                time_idx = k
            R_t = self.R[:, :, time_idx]
            extant_samples = np.where(self.sampled_by_time[:, time_idx])[0]
            R_t = R_t[np.ix_(extant_samples, extant_samples)]
            X_t = X_t[extant_samples, :]

            if compute_uncertainty:
                X_t, loss, hess_inv = optimize_timepoint(
                    X_t, R_t, self.N_min, self.N_max, epsilon=self.multiplier,
                    tol=tol, max_iter=max_iter, return_hess_inv=True,
                )
                try:
                    covs = extract_sample_covariances(hess_inv, len(extant_samples), self.n_components)
                    if n_observations is not None:
                        # Each rate has n_observations independent Poisson observations → Hessian
                        # is n_observations× larger → covariance is n_observations× smaller.
                        covs = covs / (n_observations * self.multiplier)
                    covariances[extant_samples, :, :, time_idx] = covs
                except Exception:
                    pass  # leave NaN on Hessian failure
            else:
                X_t, loss = optimize_timepoint(
                    X_t, R_t, self.N_min, self.N_max, epsilon=self.multiplier,
                    tol=tol, max_iter=max_iter,
                )

            if k == 0:
                X_t -= X_t.mean(axis=0)
            X[extant_samples, :, time_idx] = X_t
            X_t = X[:, :, time_idx]
            total_loss += loss

        X = procrustes_alignment(X)
        self.embeddings = X
        if compute_uncertainty:
            self.covariances = covariances
        return X, total_loss
    
    def compute_embeddings_coordinate_descent(self, X_init = None, num_epochs=10):
        if X_init is None:
            X_init = np.random.rand(self.num_samples, self.n_components)
            X_init, _ = optimize_timepoint(X_init, X = None, R_t=self.R[:, :, -1], N_min= self.N_min, epsilon= self.stderr, smoothing_params= None, max_iter=MAX_ITER, tol=TIMEPOINT_TOL)
            X_init = np.repeat(X_init[:, :, np.newaxis], self.num_timepoints, axis=2)
        else:
            assert X_init.shape == (self.num_samples, self.n_components, self.num_timepoints), "X_init must have shape (num_samples, n_components, num_timepoints)"
        
        X = X_init.copy()
        for _ in range(num_epochs):
            total_loss = 0
            for i in tqdm(range(self.num_samples)):
                X_i = X[i,:,:]
                R_i = self.R[i,:,:]
                X_i_opt,X_i_loss = optimize_sample(X_i, X, R_i, self.N_min, self.N_max, self.smoothing_params, max_iter=MAX_ITER, tol=SMOOTHING_TOL)
                X[i,:,:] = X_i_opt
                X = X - np.mean(X, axis=0, keepdims=True)  # Centering step to remove translation indeterminacy
                total_loss += X_i_loss
        # X = procrustes_alignment(X)
        self.embeddings = X
        return X, total_loss
    
    def compute_embeddings_two_pass(self):
        X,loss = self.compute_embeddings_through_time(smooth=False)
        X,loss = self.compute_embeddings_coordinate_descent(X_init=X, num_epochs=1)
        return X, loss

    def get_embeddings_df(self):
        assert self.embeddings is not None, (
            "Embeddings have not been computed yet. "
            "Please run compute_embeddings_through_time() first."
        )

        n, d, T = self.embeddings.shape

        sample_ids = [self.idx_sample_map[i] for i in range(n)]
        times = np.tile(self.time_points, n)

        # Base dataframe
        df = pd.DataFrame({
            "sample_id": np.repeat(sample_ids, T),
            "time": times,
        })

        # Add each embedding dimension
        for dim in range(d):
            df[f"X{dim+1}"] = self.embeddings[:, dim, :].flatten()

        return df

    def get_covariances_df(self):
        assert self.covariances is not None, (
            "Covariances have not been computed. "
            "Run compute_embeddings_through_time(compute_uncertainty=True) first."
        )

        n, d, _, T = self.covariances.shape
        sample_ids = [self.idx_sample_map[i] for i in range(n)]

        df = pd.DataFrame({
            "sample_id": np.repeat(sample_ids, T),
            "time": np.tile(self.time_points, n),
        })

        for i, j in iterproduct(range(d), range(d)):
            df[f"C{i+1}{j+1}"] = self.covariances[:, i, j, :].flatten()

        return df