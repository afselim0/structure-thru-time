#%%
#!/usr/bin/env python3
"""Latent-space embeddings of time-varying population structure.

Wraps the two optimization stages in OptimizationFunctions: a sequential, unpenalized
pass over time slices (damped Newton on the exact Hessian, one slice at a time),
followed by a joint pass over all slices (L-BFGS-B) with a group-fused-lasso temporal
smoothing penalty.

The public embeddings array is (n, d, T) with NaN at (sample, time) entries where the
sample is not yet present.  Internally the joint optimizer works on a packed vector of
present coordinates only.
"""

from typing import Optional

import numpy as np
import pandas as pd

from OptimizationFunctions import (
    LinkFunction,
    compute_difference_operators,
    coordinate_descent_over_time,
    optimize_jointly,
    DEFAULT_WEIGHTS,
    FTOL_JOINT,
    GTOL_JOINT,
    MAXCOR_JOINT,
    FTOL_NEWTON,
    MAX_ITER_NEWTON,
)


class LatentSpaceEmbeddings:
    """Fit latent trajectories X_i(t) to pairwise coalescent rates over time."""

    def __init__(self,
                 rates_df:             pd.DataFrame,
                 weights_df:           Optional[pd.DataFrame] = None,
                 sample_times:         Optional[list]         = None,
                 N_min:                float                  = 1,
                 N_max:                float                  = np.inf,
                 n_components:         int                    = 2,
                 l1_strengths:         Optional[dict]         = None,
                 pseudo_huber_factor:  float                  = 1e-3,
                 default_weights:      float                  = DEFAULT_WEIGHTS,
                 link_function:        str                    = "inverse_distance"):
        """Build the rate and weight tensors from long-format rates.

        Args:
            rates_df: Pairwise rates, indexed by (sample_1, sample_2), columns time.
            weights_df: Matching loss weights from compute_coal_rate_weights; if
                None, uniform weights.
            sample_times: Sampling time per sample; non-zero marks an ancient sample.
            N_min, N_max: Link function parameters.
            n_components: Latent dimensionality d.
            l1_strengths: {order: gamma} dimensionless group-fused-lasso strengths.
            pseudo_huber_factor: eps_k = pseudo_huber_factor * median group norm of
                the stage-1 solution's k-th differences.
            default_weights: Uniform weight used when weights_df is None; defaults
                to DEFAULT_WEIGHTS, the same fallback the optimizer applies.
            link_function: Link name.
        """
        # Read the column labels as floats without mutating the caller's DataFrame
        time_points = rates_df.columns.astype(float).values

        # --- Metadata ---
        self.samples         = np.unique([x for pair in rates_df.index for x in pair])
        self.num_samples     = len(self.samples)
        self.sample_idx_map  = {s: i for i, s in enumerate(self.samples)}
        self.idx_sample_map  = {i: s for i, s in enumerate(self.samples)}
        self.time_points     = time_points
        self.num_timepoints  = len(self.time_points)
        self.n_components    = n_components
        self.delta_t         = self.time_points[1] - self.time_points[0]
        self.N_min           = N_min
        self.N_max           = N_max
        self.link_function   = link_function
        self.pseudo_huber_factor = pseudo_huber_factor
        self.embeddings      = None
        self.smoothing_params = None

        # The difference operators are plain finite differences, which assume a
        # uniform grid; a log-spaced grid would need divided differences
        if not np.allclose(np.diff(self.time_points), self.delta_t):
            raise ValueError(
                "The time grid is not uniform. The finite-difference smoothing "
                "operators assume uniform spacing; non-uniform grids require "
                "divided-difference operators, which are not yet supported. "
                f"delta_t={self.delta_t}, observed spacings range over "
                f"[{np.min(np.diff(self.time_points))}, {np.max(np.diff(self.time_points))}]."
            )

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
            if weights_df is not None:
                w = weights_df.loc[(s1, s2)].values
                W[i, j, :] = w
                W[j, i, :] = w

        if weights_df is None:
            W = np.where(np.isnan(R), np.nan, float(default_weights))
        elif np.any(np.isfinite(W) & (W <= 0)):
            raise ValueError(
                f"{int(np.sum(np.isfinite(W) & (W <= 0)))} non-positive weights in "
                "weights_df; w = S_hat / (dt * sum Phi^2) is strictly positive, so "
                "this indicates a corrupt or mismatched weights file."
            )

        # --- Mask self-pairs ---
        # A sample has no rate against itself.  The diagonal is never written above
        # (rates_df is indexed by unordered pairs, so it holds no (s, s) row), but
        # NaN it explicitly rather than relying on that: a finite R_ii would reach
        # the loss with d_ii = 0, pinning each sample to itself.  This mirrors
        # default_valid_mask, which drops the diagonal on the optimizer's side.
        diag = np.arange(n)
        R[diag, diag, :] = np.nan
        W[diag, diag, :] = np.nan

        # --- Mask ancient samples in both R and W ---
        sampled_by_time = self.sample_times[:, np.newaxis] <= self.time_points
        self.sampled_by_time = sampled_by_time
        for t in range(T):
            absent          = ~sampled_by_time[:, t]
            R[absent, :, t] = np.nan
            R[:, absent, t] = np.nan
            W[absent, :, t] = np.nan
            W[:, absent, t] = np.nan

        # Stored as float32 to halve the footprint of the two (n, n, T) tensors;
        # all downstream accumulation promotes to float64
        self.R = R.astype(np.float32)
        self.W = W.astype(np.float32)

        # --- Precompute the saturated loss, used to scale the smoothing penalty ---
        ii, jj   = np.triu_indices(self.num_samples, k=1)
        r_upper  = self.R[ii, jj, :].astype(np.float64)
        w_upper  = self.W[ii, jj, :].astype(np.float64)
        valid    = np.isfinite(r_upper) & np.isfinite(w_upper) & (r_upper > 0)
        r_v, w_v = r_upper[valid], w_upper[valid]

        self.saturated_loss = float(np.sum(w_v * r_v * (1 - np.log(r_v))))

        self.l1_strengths = l1_strengths
        self.covariances  = None

    def _parse_smoothing_params(self, l1_strengths: Optional[dict]) -> None:
        """Calibrate the smoothing weights against the stage-1 solution's roughness.

        Must be called after stage-1 embeddings are available, since both alpha_1 and
        the pseudo-Huber eps are set from the actual roughness of that solution.

        Args:
            l1_strengths: {order: gamma} dimensionless strengths, or None.
        """
        if not l1_strengths:
            self.smoothing_params = None
            return

        if self.embeddings is None:
            raise ValueError(
                "Stage-1 embeddings must be set before computing smoothing weights. "
                "Run compute_embeddings_through_time first."
            )
        if not isinstance(l1_strengths, dict):
            raise TypeError(f"l1_strengths must be a dict, got {type(l1_strengths).__name__}.")

        max_order = max(l1_strengths.keys())
        diff_operators = compute_difference_operators(
            self.num_timepoints, max_order, self.delta_t
        )

        l1_weights, eps_by_order = compute_smoothing_weights(
            X                   = self.embeddings,
            saturated_loss      = self.saturated_loss,
            l1_strengths        = l1_strengths,
            diff_operators      = diff_operators,
            pseudo_huber_factor = self.pseudo_huber_factor,
            sampled_by_time     = self.sampled_by_time,
            delta_t             = self.delta_t,
        )

        self.smoothing_params = {
            "l1":             l1_weights,
            "eps":            eps_by_order,
            "diff_operators": diff_operators,
            "delta_t":        self.delta_t,
        }

    def compute_embeddings_through_time(self,
                                        X_init:              Optional[np.ndarray] = None,
                                        p:                   float = 1,
                                        link:                str   = "inverse_distance",
                                        compute_uncertainty: bool  = False,
                                        max_iter:            int   = MAX_ITER_NEWTON,
                                        newton_ftol:         float = FTOL_NEWTON,
                                        limit_blas_threads:  bool  = True):
        """Stage 1: sequential, unpenalized coordinate descent over time points.

        Args:
            X_init: (n, d, T) initial solution; random if None.
            p: Link function exponent.
            link: Link function name.
            compute_uncertainty: If True, compute per-sample Laplace covariances
                from the exact analytical Hessian.
            max_iter: Maximum damped-Newton iterations per slice.
            newton_ftol: Relative loss change at which each slice stops.
            limit_blas_threads: If True (default), pin BLAS to one thread for the
                duration of the sweep.  This stage does dense linear algebra per
                slice, and this cluster's OpenBLAS (MAX_THREADS=64, NO_AFFINITY)
                thread-thrashes catastrophically on matrices this small: an 80x80 eigh
                takes 869 ms unpinned against 0.58 ms pinned.

        Returns:
            (embeddings (n, d, T), loss dataframe).
        """
        link_fn = LinkFunction(name=link, N_min=self.N_min, p=p)

        def _run():
            return coordinate_descent_over_time(
                R                      = self.R,
                W                      = self.W,
                link                   = link_fn,
                N_max                  = self.N_max,
                n_components           = self.n_components,
                X_init                 = X_init,
                sampled_by_time        = self.sampled_by_time,
                max_iter               = max_iter,
                return_covariances     = compute_uncertainty,
                newton_ftol            = newton_ftol,
            )

        if limit_blas_threads:
            try:
                from threadpoolctl import threadpool_limits
                with threadpool_limits(limits=1):
                    result = _run()
            except ImportError:
                print("threadpoolctl unavailable; BLAS threads not pinned. On clusters "
                      "with a many-threaded OpenBLAS this makes the sequential stage "
                      "much slower -- set OMP_NUM_THREADS=1 before starting python.")
                result = _run()
        else:
            result = _run()
        if compute_uncertainty:
            X, loss_df, covariances = result
            self.covariances = covariances
        else:
            X, loss_df = result

        self.embeddings = X
        self._parse_smoothing_params(self.l1_strengths)
        return X, loss_df

    def compute_embeddings_jointly(self,
                                   X_init:           Optional[np.ndarray] = None,
                                   p:                float = 1,
                                   link:             str   = "inverse_distance",
                                   target_precision: float = FTOL_JOINT,
                                   gtol_factor:      float = GTOL_JOINT,
                                   maxcor:           int   = MAXCOR_JOINT):
        """Stage 2: joint optimization over all samples and timepoints, with smoothing.

        Args:
            X_init: (n, d, T) initial solution; random if None, but is normally the
                stage-1 solution.
            p: Link function exponent.
            link: Link function name.
            target_precision: ftol; 1e-9 for exploratory sweeps, 1e-12 for final runs.
            gtol_factor: gtol is gtol_factor * max|grad| at the initial point.
            maxcor: L-BFGS memory; cost is O(maxcor * n * d * T).

        Returns:
            (embeddings (n, d, T), progress dataframe).
        """
        if X_init is None:
            X_init = np.random.rand(self.num_samples, self.n_components, self.num_timepoints)
        link_fn = LinkFunction(name=link, N_min=self.N_min, p=p)
        X, loss_df = optimize_jointly(
            X_init           = X_init,
            R                = self.R,
            W                = self.W,
            link             = link_fn,
            N_max            = self.N_max,
            smoothing_params = self.smoothing_params,
            sampled_by_time  = self.sampled_by_time,
            target_precision = target_precision,
            gtol_factor      = gtol_factor,
            maxcor           = maxcor,
        )
        self.embeddings = X
        return X, loss_df

    def get_embeddings_df(self, id_label: str = "sample_id") -> pd.DataFrame:
        """Return the embeddings as a long dataframe.

        Args:
            id_label: Name of the identifier column.  Use "sample_id" for
                per-haplotype data and "individual_id" for per-individual data, so the
                column matches the entity that the rates_df index referred to.
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

        df.columns = df.columns.astype(str)  # safe parquet saving with multi-index columns
        return df

    def embeddings_df_to_array(self, df: pd.DataFrame, id_label: str = "sample_id") -> np.ndarray:
        """Reconstruct the (n, d, T) embeddings array from a long dataframe.

        Inverse of get_embeddings_df(): reorders the rows onto this object's
        (sample, time) grid and reshapes back to (n, d, T).  Useful for loading
        previously saved embeddings without rerunning the optimization.

        Args:
            df: Long dataframe with columns [id_label, "time", X1..Xd].
            id_label: Identifier column name used when the df was written.
        """
        dim_cols = sorted(
            [c for c in df.columns if c.startswith("X") and c[1:].isdigit()],
            key=lambda c: int(c[1:]),
        )
        n, d, T = len(self.samples), len(dim_cols), self.num_timepoints

        # Reindex onto the canonical (sample, time) grid so the reshape matches
        # the row-major (n, T) flatten used in get_embeddings_df.
        df = df.set_index([id_label, "time"])
        sample_ids = [self.idx_sample_map[i] for i in range(n)]
        grid = pd.MultiIndex.from_product([sample_ids, self.time_points])
        ordered = df.loc[grid, dim_cols].values  # (n*T, d), row-major over (n, T)

        return ordered.reshape(n, T, d).transpose(0, 2, 1)

    def get_covariances_df(self) -> pd.DataFrame:
        """Return the per-sample Laplace covariance blocks as a long dataframe."""
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


def compute_smoothing_weights(
    X: np.ndarray,
    saturated_loss: float,
    l1_strengths: dict,
    diff_operators: dict[int, np.ndarray],
    pseudo_huber_factor: float = 1e-3,
    sampled_by_time: Optional[np.ndarray] = None,
    delta_t: Optional[float] = None,
) -> tuple[dict[int, float], dict[int, float]]:
    """Calibrate alpha_1 and the pseudo-Huber eps against a stage-1 solution.

    For each order k, with u = (D_k X_i)_t the k-th difference of sample i's
    trajectory at time t:

        eps_k       = pseudo_huber_factor * median_{i,t} ||u||
        alpha_1^(k) = gamma_k * saturated_loss / sum_{i,t} sqrt(||u||^2 + eps_k^2)

    The alpha_1 denominator uses the same smoothed form as the penalty itself, for
    consistency.  With ancient samples the stage-1 solution contains NaN, which would
    otherwise make every alpha_1 come out NaN: each sample's differences are taken on
    its own present time window, exactly as the penalty does, so the calibration set
    is the same set of groups the penalty actually evaluates.

    Args:
        X: (n, d, T) stage-1 solution, possibly containing NaN.
        saturated_loss: Reference loss scale from the rates.
        l1_strengths: {order: gamma}.
        diff_operators: {order: (T-k, T) matrix} for full-length samples.
        pseudo_huber_factor: Multiplier setting eps_k relative to the median norm.
        sampled_by_time: (n, T) presence mask; all-present if None.
        delta_t: Time-grid spacing, needed to build shorter-window operators.

    Returns:
        ({order: alpha_1}, {order: eps}).
    """
    n, _, T = X.shape
    if sampled_by_time is None:
        sampled_by_time = np.ones((n, T), dtype=bool)
    operator_cache: dict[int, dict[int, np.ndarray]] = {T: diff_operators}

    def group_norms(k: int) -> np.ndarray:
        """||(D_k X_i)_t|| over every sample, on that sample's present window."""
        collected = []
        block_lengths = sampled_by_time.sum(axis=1)
        for block_length in np.unique(block_lengths):
            rows = np.where(block_lengths == block_length)[0]
            block_length = int(block_length)
            if block_length <= k:
                print(f"  Order {k}: {len(rows)} sample(s) have a present window of "
                      f"only {block_length} timepoint(s) and contribute no order-{k} "
                      f"penalty: {rows.tolist()}")
                continue
            if block_length not in operator_cache:
                if delta_t is None:
                    raise ValueError(
                        "delta_t is required to build difference operators for "
                        "ancient samples' shorter time windows."
                    )
                operator_cache[block_length] = compute_difference_operators(
                    block_length, max(l1_strengths.keys()), delta_t
                )
            ops = operator_cache[block_length]
            if k not in ops:
                continue
            starts = np.argmax(sampled_by_time[rows], axis=1)
            X_block = np.stack([X[i, :, s:s + block_length] for i, s in zip(rows, starts)])
            collected.append(np.linalg.norm(X_block @ ops[k].T, axis=1).ravel())
        return np.concatenate(collected) if collected else np.array([])
    l1_weights: dict[int, float] = {}
    eps_by_order: dict[int, float] = {}

    for k, gamma in l1_strengths.items():
        if not gamma:
            continue
        if k not in diff_operators:
            raise ValueError(
                f"No difference operator of order {k} was built; the time grid has "
                f"only {X.shape[2]} points, which is too short for that order."
            )
        norms = group_norms(k)
        finite_norms = norms[np.isfinite(norms)]
        if finite_norms.size == 0:
            raise ValueError(
                f"No finite order-{k} differences in the stage-1 solution, so the "
                "smoothing strength cannot be calibrated. Either every sample's "
                "present window is shorter than the difference order, or the stage-1 "
                "solution is all NaN."
            )

        eps_k = pseudo_huber_factor * float(np.median(finite_norms))
        smoothed_total = float(np.sum(np.sqrt(finite_norms ** 2 + eps_k ** 2)))

        l1_weights[k] = gamma * saturated_loss / smoothed_total
        eps_by_order[k] = eps_k

        # If most groups sit below eps the penalty is effectively quadratic
        # everywhere and eps_k is too large to be a mere smoothing of the kink
        frac_below = float(np.mean(finite_norms < eps_k))
        print(f"Smoothing order {k}: alpha_1={l1_weights[k]:.6g}, eps={eps_k:.6g}, "
              f"{frac_below:.2%} of groups below eps")
        if frac_below > 0.05:
            print(f"  WARNING: {frac_below:.2%} of order-{k} groups have ||u|| < eps. "
                  "eps is large relative to the roughness of the stage-1 solution; "
                  "consider reducing pseudo_huber_factor.")

    return l1_weights, eps_by_order
