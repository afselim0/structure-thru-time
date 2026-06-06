#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from optimization_functions import *

class LatentSpaceEmbeddings:
    def __init__(self,
                 rates_df:              pd.DataFrame,
                 se_df:                 pd.DataFrame = None,
                 sample_times:          list         = None,
                 N_min:                 float        = 1,
                 N_max:                 float        = np.inf,
                 n_components:          int          = 2,
                 l1_strengths:         dict         = None,
                 l2_strengths:         dict         = None,
                 weight_cap_percentile: float        = 95,
                 default_weights:       int        = 1e6):

        rates_df.columns = rates_df.columns.astype(float)

        # --- Metadata ---
        self.samples         = np.unique([x for pair in rates_df.index for x in pair])
        self.num_samples     = len(self.samples)
        self.sample_idx_map  = {s: i for i, s in enumerate(self.samples)}
        self.idx_sample_map  = {i: s for i, s in enumerate(self.samples)}
        self.time_points     = rates_df.columns.values
        self.num_timepoints  = len(self.time_points)
        self.n_components    = n_components
        self.delta_t         = self.time_points[1] - self.time_points[0]
        self.N_min           = N_min
        self.N_max           = N_max

        # --- Sample times ---
        if sample_times is None:
            self.sample_times = np.zeros(self.num_samples, dtype=float)
        else:
            self.sample_times = np.asarray(sample_times, dtype=float)
            n_ancient = (self.sample_times != 0).sum()
            if n_ancient > 0:
                print(f"Ancient samples: {n_ancient} / {self.num_samples}")

        # --- Build R and W ---
        n, T = self.num_samples, self.num_timepoints
        R = np.full((n, n, T), np.nan)
        W = np.full((n, n, T), np.nan)

        for (s1, s2), row in rates_df.iterrows():
            i = self.sample_idx_map[s1]
            j = self.sample_idx_map[s2]
            r = row.values
            R[i, j, :] = r
            R[j, i, :] = r
            if se_df is not None:
                se = se_df.loc[(s1, s2)].values
                w = (r+(1/self.N_max)) / (se) ** 2 #Adding 1/N_max to avoid zero weights when r is zero
                W[i, j, :] = w
                W[j, i, :] = w
            else:
                W[i, j, :] = default_weights
                W[j, i, :] = default_weights
            
        W[np.isinf(W)] = np.nanmax(W)*10 # Cap infinite weights to be an order of magnitude higher than the maximum finite weight (can happen when se is zero)
        w_cap = np.nanpercentile(W, weight_cap_percentile) # Cap weights at a high percentile to avoid extreme outliers dominating the optimization
        W = np.clip(W, 0, w_cap)

        # --- Mask ancient samples in both R and W ---
        sampled_by_time = self.sample_times[:, np.newaxis] <= self.time_points
        self.sampled_by_time = sampled_by_time
        for t in range(T):
            absent          = ~sampled_by_time[:, t]
            R[absent, :, t] = np.nan
            R[:, absent, t] = np.nan
            W[absent, :, t] = np.nan
            W[:, absent, t] = np.nan

        self.R = R
        self.W = W

        # --- Precompute reference loss and total weight for scale-matching ---
        ii, jj      = np.triu_indices(self.num_samples, k=1)
        r_upper     = self.R[ii, jj, :]              # (n_pairs, T)
        w_upper     = self.W[ii, jj, :]
        valid       = np.isfinite(r_upper) & np.isfinite(w_upper) & (r_upper > 0)
        r_v, w_v    = r_upper[valid], w_upper[valid]
        
        self.saturated_loss = np.sum(w_v * r_v * (1 - np.log(r_v)))
        
        # self.w_tot  = w_v.sum()
        # # Per-timepoint: used for ftol in stage-1 per-slice optimization
        # wl = np.where(valid, w_upper * r_upper * (1 - np.log(np.where(valid, r_upper, 1))), 0.0)
        # w  = np.where(valid, w_upper, 0.0)
        # w_tot_t       = w.sum(axis=0)                                         # (T,)
        # self.saturated_loss_t = np.where(w_tot_t > 0, wl.sum(axis=0) / w_tot_t, np.nan)  # (T,)
        # self.w_tot_t    = w_tot_t                   
        #                           # (T,)
        self.l1_strengths = l1_strengths
        self.l2_strengths = l2_strengths
        self.covariances  = None

    def _parse_smoothing_params(self, l1_strengths, l2_strengths):
        """Compute smoothing weights from transferable strengths and the current embeddings.

        Must be called after stage-1 embeddings are available (self.embeddings is not None),
        since the weights are calibrated to the actual roughness of the stage-1 solution.

        Args:
            l1_strengths (dict or None): {order: alpha} dimensionless L1 strengths.
            l2_strengths (dict or None): {order: alpha} dimensionless L2 strengths.
        """

        if l1_strengths is None and l2_strengths is None:
            self.smoothing_params = None
            return

        if self.embeddings is None:
            raise ValueError(
                "Stage-1 embeddings must be set before computing smoothing weights. "
                "Run optimize_timepoints_sequentially first"
            )

        assert l1_strengths is None or isinstance(l1_strengths, dict), "l1_strengths must be a dict"
        assert l2_strengths is None or isinstance(l2_strengths, dict), "l2_strengths must be a dict"

        l1_strengths = l1_strengths if l1_strengths is not None else {}
        l2_strengths = l2_strengths if l2_strengths is not None else {}

        max_order = max(
            max(l1_strengths.keys()) if l1_strengths else 0,
            max(l2_strengths.keys()) if l2_strengths else 0,
        )

        diff_operators = compute_difference_operators(
            self.num_timepoints, max_order, self.delta_t
        )

        l1_weights, l2_weights = compute_smoothing_weights(
            X              = self.embeddings,
            saturated_loss       = self.saturated_loss,
            l1_strengths   = l1_strengths,
            l2_strengths   = l2_strengths,
            diff_operators = diff_operators,
        )

        self.smoothing_params = {
            "l1":             l1_weights,
            "l2":             l2_weights,
            "diff_operators": diff_operators,
        }

    def compute_embeddings_through_time(self,
                                        X_init = None,
                                        target_precision = 1e-8,
                                        compute_uncertainty = False,
                                        exact_hessian = True):
        """Compute embeddings by coordinate descent over time points

        Args:
            X_init (_type_, optional): initial solution over all time points. Defaults to None.
            target_precision (float, optional): target precision for the optimization.
            compute_uncertainty (bool): if True, compute per-sample Laplace covariances.
            exact_hessian (bool): if True (default), use exact analytical Hessian for
                covariance estimation; if False, use L-BFGS approximation.

        Returns:
            np.ndarray, pd.DataFrame: (nxn_componentsxtimepoints) array of embeddings and loss dataframe with columns ["epoch","step","loss","n_iterations","message"]
        """
        # ftol_t = target_precision * min(np.mean(self.saturated_loss_t), 1.0)
        result = coordinate_descent_over_time(R = self.R, W = self.W, N_min = self.N_min, N_max = self.N_max, n_components = self.n_components, X_init = X_init, sampled_by_time = self.sampled_by_time, return_covariances = compute_uncertainty, exact_hessian = exact_hessian)#, ftol=ftol_t)
        if compute_uncertainty:
            X, loss_df, covariances = result
            self.covariances = covariances
        else:
            X, loss_df = result
        # X = procrustes_alignment(X)
        self.embeddings = X
        self._parse_smoothing_params(self.l1_strengths, self.l2_strengths)
        print(self.smoothing_params)
        return X, loss_df

    def compute_embeddings_sample_coordinate_descent(self, X_init = None, n_epochs = 1):
        X_init = X_init if X_init is not None else np.random.rand(self.num_samples, self.n_components, self.num_timepoints)
        X, loss_df = coordinate_descent_over_samples(X_init, self.R, self.W, self.N_min, self.N_max, self.smoothing_params, self.sampled_by_time, n_epochs=n_epochs)
        # X = procrustes_alignment(X)
        self.embeddings = X
        return X, loss_df
    
    def compute_embeddings_jointly(self, X_init=None, precision=1e-5):
        X_init = X_init if X_init is not None else np.random.rand(self.num_samples, self.n_components, self.num_timepoints)
        ftol = precision * min(self.saturated_loss, 1.0)
        X, loss_df = optimize_jointly(X_init, self.R, self.W, self.N_min, self.N_max, self.smoothing_params, self.sampled_by_time, ftol=ftol)
        self.embeddings = X
        X = procrustes_alignment(X)
        return X, loss_df
    
    def smooth_embeddings_independently(self, X):
        X_smooth = smooth_trajectories(X, self.smoothing_params, self.sampled_by_time)
        self.embeddings = X_smooth
        return X_smooth

    def get_embeddings_df(self, id_label: str = "sample_id"):
        """Return the embeddings as a long dataframe.

        Args:
            id_label (str): Name of the identifier column.  Use "sample_id" for
                phased (per-haplotype) data and "individual_id" for unphased
                (per-individual) data, so the column matches the entity that the
                rates_df index referred to.
        """
        assert self.embeddings is not None, (
            "Embeddings have not been computed yet. "
            "Please run compute_embeddings_through_time() first."
        )

        n, d, T = self.embeddings.shape

        sample_ids = [self.idx_sample_map[i] for i in range(n)]
        times = np.tile(self.time_points, n)

        # Base dataframe
        df = pd.DataFrame({
            id_label: np.repeat(sample_ids, T),
            "time": times,
        })

        # Add each embedding dimension
        for dim in range(d):
            df[f"X{dim+1}"] = self.embeddings[:, dim, :].flatten()

        df.columns = df.columns.astype(str) # To ensure safe parquet saving with multi-index columns
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
            "time":      np.tile(self.time_points, n),
        })
        for i in range(d):
            for j in range(d):
                df[f"C{i+1}{j+1}"] = self.covariances[:, i, j, :].flatten()
        df.columns = df.columns.astype(str)
        return df

def compute_smoothing_weights(X, saturated_loss, l1_strengths, l2_strengths, diff_operators):
    n, d, T    = X.shape
    l1_weights = {}
    l2_weights = {}

    for k in set(list(l1_strengths.keys()) + list(l2_strengths.keys())):
        D    = diff_operators[k]
        DkX  = X @ D.T          # (n, d, T)

        if l2_strengths.get(k, 0) > 0:
            l2_weights[k] = l2_strengths[k] * saturated_loss / np.sum(DkX ** 2)

        if l1_strengths.get(k, 0) > 0:
            l1_weights[k] = l1_strengths[k] * saturated_loss / np.sum(np.linalg.norm(DkX, axis=1))

    return l1_weights, l2_weights
# # %%
# rates_path = "../results/simulations/outputs/split/sim/data/rates.minWS100.maxWS200.V1.FS1.0.maxT10000.nT2000.parquet"
# se_path =  "../results/simulations/outputs/split/sim/data/se_rates.minWS100.maxWS200.V1.FS1.0.maxT10000.nT2000.parquet"
# rates_df = pd.read_parquet(rates_path)
# se_df = pd.read_parquet(se_path)

# rates_df = rates_df[rates_df.columns[:700]]
# se_df = se_df[se_df.columns[:700]]

# LSE = LatentSpaceEmbeddings(rates_df, se_df, N_min=1, N_max=np.inf, n_components=2, individuals=True)
# X, loss_df = LSE.compute_embeddings_through_time(target_precision=1e-7)
# # %%
# l1_strengths = {2: 1}
# l2_strengths = None

# LSE._parse_smoothing_params(l1_strengths, l2_strengths)
# smoothing_params = LSE.smoothing_params
# # %%
# smoothing_params["l1"]
# # %%
# penalty_loss, penalty_grad = compute_penalty_loss_gradient(LSE.embeddings, LSE.smoothing_params)
# # %%
# print(penalty_grad)
# # %%
