#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from optimization_functions import *

class LatentSpaceEmbeddings:
    def __init__(self,
                 rates_df:              pd.DataFrame,
                 se_df:                 pd.DataFrame = None,
                 admix_rates_df:        pd.DataFrame = None,
                 admix_se_df:           pd.DataFrame = None,
                 sample_times:          list         = None,
                 N_min:                 float        = 1,
                 N_max:                 float        = np.inf,
                 individuals:           bool         = False,
                 n_components:          int          = 2,
                 l1_strengths:         dict         = None,
                 l2_strengths:         dict         = None,
                 weight_cap_percentile: float        = .95,
):

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
                with np.errstate(divide='ignore', invalid='ignore'):
                    w = np.where(
                        r > 0,
                        r / se ** 2,     # Poisson-motivated: λ/σ² = effective exposure
                        1.0 / se ** 2    # r=0 fallback: pure precision; se=0 → inf → capped
                    )
                W[i, j, :] = w
                W[j, i, :] = w
            else:
                W[i, j, :] = 1.0
                W[j, i, :] = 1.0

        # --- Cap extreme weights ---
        cap = np.nanpercentile(W, weight_cap_percentile)
        W   = np.clip(W, 0, cap)

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

        # --- Average over individuals ---
        if individuals:
            self._average_over_individuals()

        # --- Precompute reference loss and total weight for scale-matching ---
        ii, jj      = np.triu_indices(self.num_samples, k=1)
        r_upper     = self.R[ii, jj, :]              # (n_pairs, T)
        w_upper     = self.W[ii, jj, :]
        valid       = np.isfinite(r_upper) & np.isfinite(w_upper) & (r_upper > 0)
        r_v, w_v    = r_upper[valid], w_upper[valid]
        self.w_tot  = w_v.sum()
        self.saturated_loss = np.sum(w_v * r_v * (1 - np.log(r_v))) / self.w_tot

        # Per-timepoint: used for ftol in stage-1 per-slice optimization
        wl = np.where(valid, w_upper * r_upper * (1 - np.log(np.where(valid, r_upper, 1))), 0.0)
        w  = np.where(valid, w_upper, 0.0)
        w_tot_t       = w.sum(axis=0)                                         # (T,)
        self.saturated_loss_t = np.where(w_tot_t > 0, wl.sum(axis=0) / w_tot_t, np.nan)  # (T,)
        self.w_tot_t    = w_tot_t                                             # (T,)
        self.l1_strengths = l1_strengths
        self.l2_strengths = l2_strengths

        # --- Build admix arrays if provided ---
        self.num_ancestry_components = None
        self.R_admix_dict         = None
        self.W_admix_dict         = None
        self.admix_samples        = None
        self.w_tot_admix            = None
        self.saturated_loss_admix   = None
        self.saturated_loss_t_admix = None
        self.w_tot_t_admix          = None
        self.idx_sample_map_admix      = None
        self.sample_idx_map_admix      = None
        self.n_admix_samples      = None
        self.n_non_admix_samples  = None
        self.R_admix = None
        self.W_admix = None
        

        if admix_rates_df is not None:
            admix_rates_df = admix_rates_df.copy()
            admix_rates_df.columns = admix_rates_df.columns.astype(float)
            self._build_admix_arrays(admix_rates_df, admix_se_df, weight_cap_percentile)

    # ------------------------------------------------------------------
    # Admix array construction
    # ------------------------------------------------------------------
    def _build_admix_arrays(self, admix_rates_df, admix_se_df, weight_cap_percentile):
        """Build per-ancestry R_admix_dict / W_admix_dict and a combined R_admix / W_admix.

        Per-ancestry arrays (R_admix_dict[k], W_admix_dict[k]):
          Shape (n, n, T) where n = self.num_samples, ordered by self.sample_idx_map.
          - Non-admix × non-admix block : copied from self.R / self.W
          - Non-admix × admix-k blocks  : filled from admix_rates_df
          - Admix-k × admix-k block     : filled from admix_rates_df
          - All other blocks            : NaN

        Combined arrays (R_admix, W_admix):
          Shape (n_non + K*n_admix, n_non + K*n_admix, T), ordered by sample_idx_map_admix.
          - Non-admix × non-admix block         : copied from self.R / self.W
          - Non-admix × admix-k blocks          : copied from R_admix_dict[k]
          - Admix-k × admix-k diagonal blocks   : copied from R_admix_dict[k]
          - Off-diagonal admix-k × admix-j blocks (k≠j): NaN

        admix_rates_df index: (float_id_1, float_id_2) where the integer part is the
        sample base ID and the decimal digit is the ancestry (0 = non-admixed).
        """
        T = self.num_timepoints

        # --- Helpers to extract base ID and integer ancestry type from float sample IDs ---
        def _base(fid):
            return int(round(fid))

        def _type_int(fid):
            return int(round((fid - _base(fid)) * 10))

        # --- Identify admixed sample base IDs and ancestry keys from the df index ---
        all_float_ids     = sorted(set(fid for pair in admix_rates_df.index for fid in pair))
        admixed_float_ids = [fid for fid in all_float_ids if _type_int(fid) != 0]
        ancestry_keys     = sorted(set(_type_int(fid) for fid in admixed_float_ids))

        self.admix_samples       = sorted(set(_base(fid) for fid in admixed_float_ids))
        self.n_admix_samples     = len(self.admix_samples)
        self.n_non_admix_samples = self.num_samples - self.n_admix_samples

        print(
            f"Admix arrays: {self.num_samples} samples, "
            f"admixed base IDs {self.admix_samples}, ancestries {ancestry_keys}"
        )

        # --- Build extended sample ordering for combined arrays ---
        # Layout: non-admixed samples (self.sample_idx_map order),
        # then admixed samples grouped by ancestry type k, sorted by base ID within each k.
        admix_base_set    = set(self.admix_samples)
        non_admix_samples = [s for s in self.samples if int(s) not in admix_base_set]

        admixed_typed_ids = [
            round(base_id + k * 0.1, 10)
            for k in ancestry_keys
            for base_id in sorted(self.admix_samples)
        ]
        ordered_samples = non_admix_samples + admixed_typed_ids

        self.sample_idx_map_admix = {s: i for i, s in enumerate(ordered_samples)}
        self.idx_sample_map_admix = {i: s for i, s in enumerate(ordered_samples)}

        # --- Precompute index arrays shared across all per-ancestry iterations ---
        non_admix_R_indices = np.array([self.sample_idx_map[s] for s in non_admix_samples])

        # --- Initialise per-ancestry dicts ---
        self.R_admix_dict           = {}
        self.W_admix_dict           = {}
        self.w_tot_admix            = {}
        self.saturated_loss_admix   = {}
        self.saturated_loss_t_admix = {}
        self.w_tot_t_admix          = {}

        for k in ancestry_keys:

            # --- Allocate arrays (all NaN); shape mirrors self.R / self.W ---
            n   = self.num_samples
            R_k = np.full((n, n, T), np.nan)
            W_k = np.full((n, n, T), np.nan)

            # --- Copy non-admix × non-admix block from self.R / self.W ---
            ix = np.ix_(non_admix_R_indices, non_admix_R_indices)
            R_k[ix] = self.R[ix]
            W_k[ix] = self.W[ix]

            # --- Fill non-admix × admix-k and admix-k × admix-k blocks ---
            for (s1, s2), row in admix_rates_df.iterrows():
                s1_type = _type_int(s1)
                s2_type = _type_int(s2)

                # Keep only pairs belonging to ancestry k:
                # non-admix(.0) × admix-k, or admix-k × admix-k
                if not (
                    (s1_type == k and s2_type == k) or
                    (s1_type == 0  and s2_type == k) or
                    (s1_type == k  and s2_type == 0)
                ):
                    continue

                b1, b2 = _base(s1), _base(s2)
                if b1 not in self.sample_idx_map:
                    raise KeyError(
                        f"Base ID {b1} (from {s1!r}) not found in self.sample_idx_map"
                    )
                if b2 not in self.sample_idx_map:
                    raise KeyError(
                        f"Base ID {b2} (from {s2!r}) not found in self.sample_idx_map"
                    )

                i1 = self.sample_idx_map[b1]
                i2 = self.sample_idx_map[b2]
                r  = row.values

                R_k[i1, i2, :] = r
                R_k[i2, i1, :] = r

                # Compute weights from SE if available, else uniform
                if admix_se_df is not None:
                    se = admix_se_df.loc[(s1, s2)].values
                    with np.errstate(divide='ignore', invalid='ignore'):
                        w = np.where(r > 0, r / se ** 2, 1.0 / se ** 2)
                else:
                    w = np.ones_like(r)

                W_k[i1, i2, :] = w
                W_k[i2, i1, :] = w

            # --- Cap extreme weights (finite values only) ---
            finite_w = W_k[np.isfinite(W_k)]
            cap      = np.nanpercentile(finite_w, weight_cap_percentile)
            W_k      = np.clip(W_k, 0, cap)

            # --- Apply ancient-sample masking, inherited from self.sampled_by_time ---
            for t in range(T):
                absent             = ~self.sampled_by_time[:, t]
                R_k[absent, :, t]  = np.nan
                R_k[:, absent, t]  = np.nan
                W_k[absent, :, t]  = np.nan
                W_k[:, absent, t]  = np.nan

            self.R_admix_dict[k] = R_k
            self.W_admix_dict[k] = W_k

            # --- Precompute saturated loss and total weight for ancestry k ---
            ii, jj  = np.triu_indices(n, k=1)
            r_upper = R_k[ii, jj, :]
            w_upper = W_k[ii, jj, :]
            valid   = np.isfinite(r_upper) & np.isfinite(w_upper) & (r_upper > 0)
            r_v, w_v = r_upper[valid], w_upper[valid]

            self.w_tot_admix[k]          = w_v.sum()
            self.saturated_loss_admix[k] = np.sum(w_v * r_v * (1 - np.log(r_v))) / self.w_tot_admix[k]

            wl      = np.where(valid, w_upper * r_upper * (1 - np.log(np.where(valid, r_upper, 1))), 0.0)
            w_arr   = np.where(valid, w_upper, 0.0)
            w_tot_t = w_arr.sum(axis=0)
            self.saturated_loss_t_admix[k] = np.where(w_tot_t > 0, wl.sum(axis=0) / w_tot_t, np.nan)
            self.w_tot_t_admix[k]          = w_tot_t

        # --- Build combined R_admix / W_admix of shape (n_non + K*n_admix, n_non + K*n_admix, T) ---
        K       = len(ancestry_keys)
        m       = self.n_non_admix_samples + K * self.n_admix_samples
        R_admix = np.full((m, m, T), np.nan)
        W_admix = np.full((m, m, T), np.nan)

        # Index arrays for non-admixed samples in the combined matrix
        non_admix_admix_indices = np.array([
            self.sample_idx_map_admix[s] for s in non_admix_samples
        ])

        # --- Non-admix × non-admix block: copy from self.R / self.W ---
        ix = np.ix_(non_admix_R_indices,     non_admix_R_indices)
        iy = np.ix_(non_admix_admix_indices, non_admix_admix_indices)
        R_admix[iy] = self.R[ix]
        W_admix[iy] = self.W[ix]

        # --- Per-ancestry off-diagonal and diagonal blocks ---
        for k_anc in ancestry_keys:
            R_k = self.R_admix_dict[k_anc]
            W_k = self.W_admix_dict[k_anc]

            # Float IDs and their positions in the combined matrix for ancestry k_anc
            admixed_k_float_ids   = [
                round(base_id + k_anc * 0.1, 10) for base_id in sorted(self.admix_samples)
            ]
            admix_k_admix_indices = np.array([
                self.sample_idx_map_admix[fid] for fid in admixed_k_float_ids
            ])

            # Base ID positions in self.sample_idx_map, used to slice R_admix_dict[k_anc]
            admix_k_R_indices = np.array([
                self.sample_idx_map[_base(fid)] for fid in admixed_k_float_ids
            ])

            # Non-admix × admix-k and admix-k × non-admix blocks
            iy_non   = np.ix_(non_admix_admix_indices, admix_k_admix_indices)
            iy_admix = np.ix_(admix_k_admix_indices,   non_admix_admix_indices)
            ix_non   = np.ix_(non_admix_R_indices,     admix_k_R_indices)
            ix_admix = np.ix_(admix_k_R_indices,       non_admix_R_indices)

            R_admix[iy_non]   = R_k[ix_non]
            R_admix[iy_admix] = R_k[ix_admix]
            W_admix[iy_non]   = W_k[ix_non]
            W_admix[iy_admix] = W_k[ix_admix]

            # Admix-k × admix-k diagonal block
            iy_kk = np.ix_(admix_k_admix_indices, admix_k_admix_indices)
            ix_kk = np.ix_(admix_k_R_indices,     admix_k_R_indices)
            R_admix[iy_kk] = R_k[ix_kk]
            W_admix[iy_kk] = W_k[ix_kk]

        # --- At t=0, initialise cross-ancestry pairs of the same base sample to 1/N_min ---
        # For each admixed sample, pairs (sample_id.k1, sample_id.k2) with k1 ≠ k2 are
        # otherwise NaN; setting them to 1/N_min encodes maximal differentiation at t=0.
        for base_id in self.admix_samples:
            cross_ancestry_indices = np.array([
                self.sample_idx_map_admix[round(base_id + k_anc * 0.1, 10)]
                for k_anc in ancestry_keys
            ])
            for ia in cross_ancestry_indices:
                for ib in cross_ancestry_indices:
                    if ia != ib:
                        R_admix[ia, ib, 0] = 1.0 / self.N_min
                        W_admix[ia, ib, 0] = np.nanmax(W_admix)  # Assign max weight to enforce this constraint
        self.R_admix = R_admix
        self.W_admix = W_admix
    # ------------------------------------------------------------------
    # Existing private helpers (unchanged)
    # ------------------------------------------------------------------

    def _average_over_individuals(self):
        """Average R and sum W over haplotype pairs within each individual."""
        n_ind = self.num_samples // 2
        T     = self.num_timepoints
        R_ind = np.full((n_ind, n_ind, T), np.nan)
        W_ind = np.full((n_ind, n_ind, T), np.nan)

        for i in range(n_ind):
            for j in range(i + 1, n_ind):
                R_block = self.R[2*i:2*i+2, 2*j:2*j+2, :]
                W_block = self.W[2*i:2*i+2, 2*j:2*j+2, :]
                if not np.all(np.isnan(R_block)):
                    R_ind[i, j, :] = np.nanmean(R_block, axis=(0, 1))
                    R_ind[j, i, :] = R_ind[i, j, :]
                    W_ind[i, j, :] = np.nansum(W_block, axis=(0, 1))
                    W_ind[j, i, :] = W_ind[i, j, :]

        self.R               = R_ind
        self.W               = W_ind
        self.num_samples     = n_ind
        self.sampled_by_time = self.sampled_by_time[::2, :]

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

    # ------------------------------------------------------------------
    # Existing public methods (unchanged)
    # ------------------------------------------------------------------

    def compute_embeddings_through_time(self,
                                        X_init = None,
                                        target_precision = 1e-8,
                                        direction = "backward"):
        """Compute embeddings by coordinate descent over time points

        Args:
            X_init (_type_, optional): initial solution over all time points. Defaults to None.
            target_precision (float, optional): target precision for the optimization.
            n_epochs (int, optional): number of cycles to pass over timepoints. Defaults to 1.

        Returns:
            np.ndarray, pd.DataFrame: (nxn_componentsxtimepoints) array of embeddings and loss dataframe with columns ["epoch","step","loss","n_iterations","message"]
        """
        ftol_t = target_precision * min(np.mean(self.saturated_loss_t), 1.0)
        X, loss_df = coordinate_descent_over_time(R = self.R, W = self.W, w_tot = self.w_tot, N_min = self.N_min,N_max = self.N_max,n_components = self.n_components,X_init = X_init, sampled_by_time = self.sampled_by_time, ftol=ftol_t, direction=direction)
        # X = procrustes_alignment(X)
        self.embeddings = X
        self._parse_smoothing_params(self.l1_strengths, self.l2_strengths)
        print(self.smoothing_params)
        return X, loss_df  

    def compute_embeddings_sample_coordinate_descent(self, X_init = None, n_epochs = 1):
        X_init = X_init if X_init is not None else np.random.rand(self.num_samples, self.n_components, self.num_timepoints)
        X, loss_df = coordinate_descent_over_samples(X_init, self.R, self.W, self.w_tot, self.N_min, self.N_max, self.smoothing_params, self.sampled_by_time, n_epochs=n_epochs)        
        # X = procrustes_alignment(X)
        self.embeddings = X
        return X, loss_df
    
    def compute_embeddings_jointly(self, X_init=None, precision=1e-5):
        X_init = X_init if X_init is not None else np.random.rand(self.num_samples, self.n_components, self.num_timepoints)
        ftol = precision * min(self.saturated_loss, 1.0)
        X, loss_df = optimize_jointly(X_init, self.R, self.W, self.w_tot, self.N_min, self.N_max, self.smoothing_params, self.sampled_by_time, ftol=ftol)
        self.embeddings = X
        X = procrustes_alignment(X)
        return X, loss_df
    
    def smooth_embeddings_independently(self, X):
        X_smooth = smooth_trajectories(X, self.smoothing_params, self.sampled_by_time)
        self.embeddings = X_smooth
        return X_smooth

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

        df.columns = df.columns.astype(str) # To ensure safe parquet saving with multi-index columns
        return df

    # ------------------------------------------------------------------
    # New admix embedding methods
    # ------------------------------------------------------------------

    def compute_embeddings_through_time_admix(self,
                                              X_init=None,
                                              target_precision=1e-8):
        """Compute embeddings for the extended (non-admix + typed admix) sample set
        using coordinate descent over time points.

        Uses R_admix and W_admix instead of R and W. The returned array has shape
        (m, n_components, T) where m = n_non_admix + K * n_admix.

        If X_init is None and self.embeddings is available (i.e.
        compute_embeddings_through_time() has already been run), the initialisation
        is built automatically via _expand_embeddings_to_admix(), so that admixed
        positions start from their base sample's fitted trajectory.

        Args:
            X_init (np.ndarray, optional): Initial solution of shape (m, n_components, T).
                Defaults to None; falls back to self.embeddings expansion if available,
                then to the default initialisation inside coordinate_descent_over_time.
            target_precision (float, optional): Target precision for the optimisation.

        Returns:
            np.ndarray, pd.DataFrame: embeddings array and loss dataframe.
        """
        # if self.R_admix_dict is None:
        #     raise ValueError(
        #         "R_admix is not available. Pass admix_rates_df when constructing the object."
        #     )

        # X_init = self.embeddings if self.embeddings is not None else np.random.rand(self.num_samples, self.n_components, self.num_timepoints)
        # X_admix = np.full((len(self.idx_sample_map_admix), self.n_components, self.num_timepoints), np.nan)
        # non_admix_idxs = np.array([self.idx_sample_map_admix[k] for k in list(self.idx_sample_map_admix)[:self.n_non_admix_samples]])
        # X_admix[:self.n_non_admix_samples] = X_init[non_admix_idxs,:,:]

        # for k, R_k in self.R_admix_dict.items():
        #     print(f"Optimizing ancestry {k} with {self.n_admix_samples} admixed samples...")
        #     ftol = target_precision * min(np.mean(self.saturated_loss_t_admix[k]), 1.0)
        #     X_k, loss_df_k = coordinate_descent_over_time_admixed(
        #         R_k,
        #         self.W_admix_dict[k],
        #         self.w_tot_admix[k],
        #         self.admix_samples,
        #         self.N_min,
        #         self.N_max,
        #         self.n_components,
        #         X_init,
        #         ftol=ftol,
        #     )
        #     X_k_sliced = X_k[np.array(self.admix_samples), :, :]
        #     start_idx = self.n_non_admix_samples + ((k-1) * self.n_admix_samples)
        #     end_idx = start_idx + self.n_admix_samples
        #     X_admix[start_idx:end_idx,:,:] = X_k_sliced

        # self.embeddings_admix = X_admix

        X_init = self._expand_embeddings_to_admix(self.embeddings) if X_init is None and self.embeddings is not None else X_init
        print(X_init[:,0,0])
        ftol_t = target_precision * min(np.mean(self.saturated_loss_t), 1.0)
        w_tot_t = np.mean(np.nansum(self.W_admix,axis=(0,1)))
        X_admix, loss_df = coordinate_descent_over_time(R = self.R_admix, W = self.W_admix, w_tot = w_tot_t, N_min = self.N_min,N_max = self.N_max,n_components = self.n_components,X_init = X_init, ftol=ftol_t, direction="forward")
        # X = procrustes_alignment(X)
        self.embeddings_admix = X_admix
        # self._parse_smoothing_params(self.l1_strengths, self.l2_strengths)
        # print(self.smoothing_params)
        return X_admix, loss_df

    def compute_embeddings_jointly_admix(self, X_init=None, precision=1e-5):
        """Jointly optimise embeddings for the extended (non-admix + typed admix) sample set.

        Uses R_admix and W_admix instead of R and W. The returned array has shape
        (m, n_components, T) where m = n_non_admix + K * n_admix.

        If X_init is None and self.embeddings is available, the initialisation
        is built automatically via _expand_embeddings_to_admix().

        Args:
            X_init (np.ndarray, optional): Initial solution of shape (m, n_components, T).
                Defaults to None; falls back to self.embeddings expansion if available,
                then random.
            precision (float, optional): Convergence tolerance relative to saturated loss.

        Returns:
            np.ndarray, pd.DataFrame: embeddings array and loss dataframe.
        """
        if self.R_admix is None:
            raise ValueError(
                "R_admix is not available. Pass admix_rates_df when constructing the object."
            )

        m = self.R_admix.shape[0]

        if X_init is None:
            if self.embeddings is not None:
                X_init = self._expand_embeddings_to_admix(self.embeddings)
            else:
                X_init = np.random.rand(m, self.n_components, self.num_timepoints)

        ftol = precision * min(self.saturated_loss_admix, 1.0)

        X, loss_df = optimize_jointly(
            X_init,
            self.R_admix,
            self.W_admix,
            self.w_tot_admix,
            self.N_min,
            self.N_max,
            self.smoothing_params,
            self._sampled_by_time_admix,
            ftol=ftol,
        )

        self.embeddings_admix = X
        X = procrustes_alignment(X)
        return X, loss_df

    def get_embeddings_admix_df(self):
        """Return embeddings_admix as a tidy DataFrame, analogous to get_embeddings_df()."""
        assert hasattr(self, "embeddings_admix") and self.embeddings_admix is not None, (
            "Admix embeddings have not been computed yet. "
            "Please run compute_embeddings_through_time_admix() or "
            "compute_embeddings_jointly_admix() first."
        )

        n, d, T = self.embeddings_admix.shape
        sample_ids = [self.idx_admix_map[i] for i in range(n)]
        times = np.tile(self.time_points, n)

        df = pd.DataFrame({
            "sample_id": np.repeat(sample_ids, T),
            "time": times,
        })

        for dim in range(d):
            df[f"X{dim+1}"] = self.embeddings_admix[:, dim, :].flatten()

        df.columns = df.columns.astype(str)
        return df
    
    def _expand_embeddings_to_admix(self, X):
        """Expand a (num_samples, n_components, T) embedding array into the
        (m, n_components, T) layout expected by R_admix / W_admix.

        Mapping rules
        -------------
        - Non-admix rows in R_admix  <- rows of X at the corresponding non-admix
        indices in self.sample_idx_map.
        - Admix typed rows in R_admix <- rows of X at the index of the *base*
        sample (integer part of the float ID) in self.sample_idx_map. All K
        typed copies of the same base sample start from the same position.

        Args:
            X (np.ndarray): embeddings of shape (num_samples, n_components, T).

        Returns:
            np.ndarray: expanded embeddings of shape (m, n_components, T).
        """
        m       = len(self.sample_idx_map_admix)
        X_admix = np.zeros((m, self.n_components, self.num_timepoints), dtype=X.dtype)

        for admix_idx, sample_id in self.idx_sample_map_admix.items():
            # Non-admixed sample_id is an integer; admixed typed sample_id is a
            # float (e.g. 3.1) — in both cases the base ID is the integer part.
            base_id = int(round(sample_id))
            if base_id not in self.sample_idx_map:
                raise KeyError(
                    f"Base sample ID {base_id!r} (derived from {sample_id!r}) "
                    f"not found in self.sample_idx_map."
                )
            X_admix[admix_idx] = X[self.sample_idx_map[base_id]]

        return X_admix

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
# %%
# def _build_admix_arrays(self, admix_rates_df, admix_se_df, weight_cap_percentile):
#     """Build R_admix and W_admix of shape (m, m, T).

#     Index layout (row/col order in the new arrays):
#     [0 .. n_non_admix-1]                     : non-admixed samples (same order as self.samples)
#     [n_non_admix .. n_non_admix + K*n_admix-1]: admixed samples, ordered by
#                                                 (base_id_0 type_1, base_id_0 type_2, ...,
#                                                     base_id_1 type_1, ...)

#     Non-admixed sample IDs are integers.
#     Admixed sample IDs are floats, e.g. 3.1 = sample 3, type 1.
#     All admix × admix blocks are NaN.
#     Non-admix × admix (and admix × non-admix) blocks are filled from admix_rates_df.
#     Non-admix × non-admix block is copied from self.R / self.W.
#     """
#     T = self.num_timepoints

#     # --- Parse admixed IDs from the index ---
#     # admix_rates_df rows: (admixed_float_id, non_admixed_int_id)
#     admixed_typed_ids = sorted(set(
#         s1 for (s1, s2) in admix_rates_df.index
#     ))  # list of floats, e.g. [3.1, 3.2, 5.1, 5.2]

#     # Identify unique base IDs and unique types
#     def _base(fid):
#         return int(round(fid))        # integer part: the original sample id

#     def _type(fid):
#         # Round to avoid floating-point noise; type is the first decimal digit
#         return round(fid - _base(fid), 10)

#     base_ids   = sorted(set(_base(fid) for fid in admixed_typed_ids))
#     type_keys  = sorted(set(_type(fid) for fid in admixed_typed_ids))
#     K          = len(type_keys)
#     n_admix    = len(base_ids)
#     n_non      = self.num_samples - n_admix   # non-admixed count
#     m          = n_non + K * n_admix          # total rows/cols

#     print(f"Admix arrays: {n_non} non-admixed + {n_admix} admixed × {K} types = {m} total")

#     # --- Identify non-admixed samples within self.samples ---
#     # Non-admixed sample IDs are integers (stored as int or float-with-zero-decimal).
#     # Admixed base IDs (integers) are present in self.samples alongside the non-admixed ones.
#     admix_base_id_set = set(base_ids)
#     non_admix_samples = [s for s in self.samples if int(s) not in admix_base_id_set]
#     assert len(non_admix_samples) == n_non, (
#         f"Expected {n_non} non-admixed samples, found {len(non_admix_samples)}: {non_admix_samples}"
#     )

#     # Indices of non-admixed samples inside self.R / self.W
#     non_admix_R_indices = np.array([self.sample_idx_map[s] for s in non_admix_samples])

#     # --- Build ordered sample list for R_admix ---
#     # Layout: [non-admixed samples | admixed_base_0.type_1, admixed_base_0.type_2, ...]
#     admix_typed_order = [round(base_id + tk, 10) for base_id in base_ids for tk in type_keys]
#     admix_samples_list = non_admix_samples + admix_typed_order   # length m

#     admix_idx_map = {s: i for i, s in enumerate(admix_samples_list)}
#     idx_admix_map = {i: s for i, s in enumerate(admix_samples_list)}

#     self.admix_samples  = admix_samples_list
#     self.admix_idx_map  = admix_idx_map
#     self.idx_admix_map  = idx_admix_map

#     # Indices of non-admixed samples inside R_admix (trivially 0..n_non-1,
#     # but kept explicit to mirror the logic used for self.R slicing)
#     non_admix_admix_indices = np.array([admix_idx_map[s] for s in non_admix_samples])

#     # --- Allocate (all NaN; admix×admix block stays NaN by construction) ---
#     R_admix = np.full((m, m, T), np.nan)
#     W_admix = np.full((m, m, T), np.nan)

#     # --- Fill non-admix × non-admix block ---
#     # Slice the relevant rows/cols out of self.R / self.W and place them at
#     # the corresponding positions in R_admix / W_admix.
#     ix = non_admix_R_indices          # positions in self.R  (shape n_non)
#     iy = non_admix_admix_indices      # positions in R_admix (shape n_non)

#     # Use np.ix_ to select the (n_non × n_non) sub-block from self.R and write
#     # it into the correct (n_non × n_non) sub-block of R_admix.
#     R_admix[np.ix_(iy, iy)] = self.R[np.ix_(ix, ix)]
#     W_admix[np.ix_(iy, iy)] = self.W[np.ix_(ix, ix)]

#     # --- Fill non-admix × admix blocks from admix_rates_df ---
#     for (s_admix, s_non), row in admix_rates_df.iterrows():
#         # s_admix is a float (e.g. 3.1), s_non is a non-admixed int/float
#         if s_non not in admix_idx_map:
#             raise KeyError(
#                 f"Non-admixed sample {s_non!r} from admix_rates_df not found in admix_idx_map"
#             )
#         if s_admix not in admix_idx_map:
#             raise KeyError(
#                 f"Admixed sample {s_admix!r} from admix_rates_df not found in admix_idx_map. "
#                 f"Expected one of: {admix_typed_order}"
#             )

#         i_non   = admix_idx_map[s_non]    # index in [0 .. n_non-1]
#         i_admix = admix_idx_map[s_admix]  # index in [n_non .. m-1]
#         r = row.values

#         R_admix[i_non,   i_admix, :] = r
#         R_admix[i_admix, i_non,   :] = r

#         if admix_se_df is not None:
#             se = admix_se_df.loc[(s_admix, s_non)].values
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 w = np.where(
#                     r > 0,
#                     r / se ** 2,
#                     1.0 / se ** 2
#                 )
#         else:
#             w = np.ones_like(r)

#         W_admix[i_non,   i_admix, :] = w
#         W_admix[i_admix, i_non,   :] = w

#     # --- Cap extreme weights (finite values only) ---
#     finite_w = W_admix[np.isfinite(W_admix)]
#     cap = np.nanpercentile(finite_w, weight_cap_percentile)
#     W_admix = np.clip(W_admix, 0, cap)

#     # --- Apply ancient-sample masking for non-admixed samples ---
#     # Admixed typed samples are assumed present at all time points.
#     # Non-admixed masking is inherited from self.sampled_by_time, but we
#     # must map from self.sampled_by_time rows (indexed by self.sample_idx_map)
#     # to the non-admixed rows in R_admix (indexed by non_admix_admix_indices).
#     sampled_by_time_admix = np.ones((m, T), dtype=bool)
#     for k_non, (r_idx, a_idx) in enumerate(zip(non_admix_R_indices, non_admix_admix_indices)):
#         sampled_by_time_admix[a_idx, :] = self.sampled_by_time[r_idx, :]

#     self._sampled_by_time_admix = sampled_by_time_admix  # stored for use in new methods

#     for t in range(T):
#         absent = ~sampled_by_time_admix[:, t]
#         R_admix[absent, :,      t] = np.nan
#         R_admix[:,      absent, t] = np.nan
#         W_admix[absent, :,      t] = np.nan
#         W_admix[:,      absent, t] = np.nan

#     self.R_admix = R_admix
#     self.W_admix = W_admix

#     # --- Precompute admix reference loss and total weight ---
#     ii, jj  = np.triu_indices(m, k=1)
#     r_upper = R_admix[ii, jj, :]
#     w_upper = W_admix[ii, jj, :]
#     valid   = np.isfinite(r_upper) & np.isfinite(w_upper) & (r_upper > 0)
#     r_v, w_v = r_upper[valid], w_upper[valid]

#     self.w_tot_admix          = w_v.sum()
#     self.saturated_loss_admix = np.sum(w_v * r_v * (1 - np.log(r_v))) / self.w_tot_admix

#     wl    = np.where(valid, w_upper * r_upper * (1 - np.log(np.where(valid, r_upper, 1))), 0.0)
#     w_arr = np.where(valid, w_upper, 0.0)
#     w_tot_t = w_arr.sum(axis=0)
#     self.saturated_loss_t_admix = np.where(w_tot_t > 0, wl.sum(axis=0) / w_tot_t, np.nan)
#     self.w_tot_t_admix          = w_tot_t

    def _expand_embeddings_to_admix(self, X):
        """Expand a (num_samples, n_components, T) embedding array into the
        (m, n_components, T) layout expected by R_admix / W_admix.

        Mapping rules
        -------------
        - Non-admix rows in R_admix  <- rows of X at the corresponding non-admix
          indices in self.R  (looked up via self.sample_idx_map).
        - Admix typed rows in R_admix <- rows of X at the index of the *base*
          sample (integer part of the float ID) in self.R.  All K typed copies
          of the same base sample start from the same position in the embedding.

        Args:
            X (np.ndarray): embeddings of shape (num_samples, n_components, T).

        Returns:
            np.ndarray: expanded embeddings of shape (m, n_components, T).
        """
        m = len(self.admix_samples)
        X_admix = np.zeros((m, self.n_components, self.num_timepoints), dtype=X.dtype)

        for admix_idx, sample_id in self.idx_admix_map.items():
            # For non-admixed samples sample_id is an integer; for admixed typed
            # samples it is a float (e.g. 3.1) and we want the base id (3).
            if isinstance(sample_id, float) and sample_id != int(sample_id):
                base_id = int(round(sample_id))
            else:
                base_id = sample_id

            if base_id not in self.sample_idx_map:
                raise KeyError(
                    f"Base sample id {base_id!r} (derived from admix id {sample_id!r}) "
                    f"not found in self.sample_idx_map."
                )

            X_admix[admix_idx] = X[self.sample_idx_map[base_id]]

        return X_admix