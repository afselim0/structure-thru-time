#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from tqdm import tqdm

MIN_RATE = 1e-12

def adaptive_local_linear_filter(minimum_window_size, maximum_window_size, n_timepoints, time_step):
    minimum_num_idxs = int(minimum_window_size // time_step)
    if minimum_num_idxs % 2 == 0:
        minimum_num_idxs += 1
    if minimum_num_idxs < 7:
        minimum_num_idxs = 7
    maximum_num_idxs = int(maximum_window_size // time_step)
    if maximum_num_idxs % 2 == 0:
        maximum_num_idxs += 1
    if maximum_num_idxs < minimum_num_idxs:
        maximum_num_idxs = minimum_num_idxs

    # t_min = minimum_num_idxs // 2
    # t_max = n_timepoints - maximum_num_idxs // 2
    t_min = minimum_num_idxs // 2
    t_max = n_timepoints - maximum_num_idxs // 2

    if t_max <= t_min:
        raise ValueError("Time series too short for the requested window sizes.")

    window_sizes = np.linspace(minimum_num_idxs, maximum_num_idxs, t_max - t_min).astype(int)

    S = np.zeros((n_timepoints, n_timepoints))
    # A        = np.zeros((n_timepoints, n_timepoints))
    # se_scale = np.full(n_timepoints, np.nan)

    # Interior: symmetric window growing from minimum to maximum bandwidth
    for k, t in enumerate(range(t_min, t_max)):
        m = window_sizes[k]
        if m % 2 == 0:
            m += 1
        h    = (m - 1) // 2
        xloc = np.arange(-h, h + 1) * time_step   # S1 = 0 by symmetry
        S2   = (xloc ** 2).sum()
        S[t, t - h : t + h + 1] = xloc / S2        # slope:     j*dt / S2
        # A[t, t - h : t + h + 1] = 1.0 / m          # intercept: local mean
        # se_scale[t]              = 1.0 / np.sqrt(S2) # SE = s_hat / sqrt(S2)

    # Early-time boundary (t < t_min): one fixed window [0, minimum_num_idxs),
    # slope evaluated at each t within it via one-sided local linear fit.
    for t in range(t_min):
        xloc = (np.arange(minimum_num_idxs) - t) * time_step
        S1 = xloc.sum()
        S2 = (xloc ** 2).sum()
        n  = minimum_num_idxs
        S[t, :minimum_num_idxs] = (n * xloc - S1) / (n * S2 - S1 ** 2)

    # Late-time boundary (t >= t_max): one fixed window [n_timepoints - maximum_num_idxs, n_timepoints),
    # slope evaluated at each t within it via one-sided local linear fit.
    start = n_timepoints - maximum_num_idxs
    for t in range(t_max, n_timepoints):
        xloc = (np.arange(maximum_num_idxs) - (t - start)) * time_step
        S1 = xloc.sum()
        S2 = (xloc ** 2).sum()
        n  = maximum_num_idxs
        S[t, start:] = (n * xloc - S1) / (n * S2 - S1 ** 2)

    # return S, t_min, t_max
    return S


def compute_coal_rates(pdf_df, minimum_window_size, maximum_window_size):
    times     = pdf_df.columns.astype(float)
    time_step = times[1] - times[0]
    n_T       = len(times)

    survival_df = 1 - pdf_df.cumsum(axis=1)
    H = -np.log(survival_df.values.clip(1e-10))   # (n_pairs, n_T); clip avoids -inf at S=0

    S = adaptive_local_linear_filter(minimum_window_size, maximum_window_size, n_T, time_step)

    print("Multiplying cumulative hazard matrix with derivative weights to compute rates")
    rates = H @ S.T   # (n_pairs, n_T)

    rates_df = pd.DataFrame(rates, columns=times, index=pdf_df.index)
    return rates_df


def compute_coal_rates_bootstrap_se(
    pdf_df: pd.DataFrame,
    bootstrap_dir: str,
    minimum_window_size: float,
    maximum_window_size: float,
    n_replicates: int,
    subsample_size: int,
) -> pd.DataFrame:
    """Estimate SE of coalescent rate estimates by block bootstrap over tree files.

    For each of ``n_replicates`` bootstrap replicates, ``subsample_size`` count
    files are drawn without replacement from ``bootstrap_dir``.  The sampled
    counts are summed and normalised using the corresponding columns of
    ``coverage_df`` to form a bootstrap PDF, from which rates are computed using
    the same derivative filter matrix as ``compute_coal_rates``.  The SE at
    each (pair, timepoint) is the standard deviation across replicates.

    Parameters
    ----------
    pdf_df : pd.DataFrame
        Full PDF estimate (output of ``CountCoalEvents.compute_coal_time_pdf``).
        Used to set the time grid and index.
    bootstrap_dir : str
        Directory containing per-file ``.parquet`` count files, as written by
        ``CountCoalEvents.compute_coal_time_pdf``.
    minimum_window_size : float
        Passed to ``adaptive_local_linear_filter``.
    maximum_window_size : float
        Passed to ``adaptive_local_linear_filter``.
    n_replicates : int
        Number of bootstrap replicates.
    subsample_size : int
        Number of count files drawn without replacement per replicate.

    Returns
    -------
    pd.DataFrame
        SE estimates with the same index and columns as ``pdf_df``.
    """
    import os

    times     = pdf_df.columns.astype(float)
    time_step = times[1] - times[0]
    n_T       = len(times)

    coverage_df = pd.read_parquet(os.path.join(bootstrap_dir, "coverage.parquet"))

    # Build derivative filter once — shared across all replicates
    S = adaptive_local_linear_filter(minimum_window_size, maximum_window_size, n_T, time_step)

    # Discover available count files (exclude coverage.parquet)
    available_files = sorted(
        f for f in os.listdir(bootstrap_dir)
        if f.endswith(".parquet") and f != "coverage.parquet"
    )
    file_basenames = [f.removesuffix(".parquet") for f in available_files]
    n_files = len(available_files)

    if n_files == 0:
        raise ValueError(f"No count parquet files found in {bootstrap_dir}.")

    missing = set(file_basenames) - set(coverage_df.columns)
    if missing:
        raise ValueError(f"coverage_df is missing columns for files: {missing}")

    # Align coverage_df columns to the order of discovered files
    coverage_df = coverage_df[file_basenames]
    coverage_vals = coverage_df.values  # (n_pairs, n_files)

    if subsample_size > n_files:
        raise ValueError(
            f"subsample_size={subsample_size} exceeds the number of available count files ({n_files})."
        )

    # Load all count files once into a (n_files, n_pairs, n_T) array
    print("Loading all bootstrap count files into memory …")
    n_pairs = len(pdf_df)
    all_counts = np.zeros((n_files, n_pairs, n_T))
    for i, fname in enumerate(tqdm(available_files, desc="Loading count files")):
        all_counts[i] = pd.read_parquet(os.path.join(bootstrap_dir, fname)).values

    replicate_rates = np.zeros((n_replicates, n_pairs, n_T))

    for rep in tqdm(range(n_replicates), desc="Bootstrap replicates"):
        sampled_idxs = np.random.choice(n_files, size=subsample_size, replace=False)

        # Sum counts and coverage for sampled files
        counts_sum = all_counts[sampled_idxs].sum(axis=0)          # (n_pairs, n_T)
        total_len  = coverage_vals[:, sampled_idxs].sum(axis=1)     # (n_pairs,)

        with np.errstate(invalid="ignore"):
            pdf_rep = counts_sum / total_len[:, np.newaxis]

        # Survival → cumulative hazard → rates
        survival_rep = 1 - np.cumsum(pdf_rep, axis=1)
        H_rep = -np.log(survival_rep.clip(1e-10))
        replicate_rates[rep] = H_rep @ S.T

    # SE = std across replicates
    se = np.std(replicate_rates, axis=0)   # (n_pairs, n_T)

    return pd.DataFrame(se, columns=times, index=pdf_df.index)
#%%
# def adaptive_local_linear_filter(minimum_window_size, maximum_window_size, n_timepoints, time_step):
#     # -------------------------
#     # Window sizes (in indices)
#     # -------------------------
#     minimum_num_idxs = int(minimum_window_size // time_step)
#     if minimum_num_idxs % 2 == 0:
#         minimum_num_idxs += 1
#     if minimum_num_idxs < 7:
#         minimum_num_idxs = 7

#     maximum_num_idxs = int(maximum_window_size // time_step)
#     if maximum_num_idxs % 2 == 0:
#         maximum_num_idxs += 1
#     if maximum_num_idxs < minimum_num_idxs:
#         maximum_num_idxs = minimum_num_idxs

#     # -------------------------
#     # Boundaries
#     # -------------------------
#     start_boundary = minimum_num_idxs // 2
#     end_boundary = n_timepoints - maximum_num_idxs // 2

#     window_sizes = np.linspace(
#         minimum_num_idxs,
#         maximum_num_idxs,
#         end_boundary - start_boundary
#     ).astype(int)

#     S = np.zeros((n_timepoints, n_timepoints))

#     # # =====================================================
#     # # 1. Recent-time boundary: quadratic fit (y = a + b x + c x^2)
#     # # =====================================================
#     # idx = np.arange(start_boundary)
#     # x = (idx - idx.mean()) * time_step

#     # X = np.column_stack([np.ones(start_boundary), x, x**2])
#     # XtX_inv = np.linalg.inv(X.T @ X)
#     # P = XtX_inv @ X.T # P @ y gives coefficients [a,b,c]

#     # # Filling in [0-start_boundary] rows of S
#     # for t in range(start_boundary):
#     #     xt = x[t]
#     #     D = np.array([0.0, 1.0, 2.0 * xt])
#     #     S[t, :start_boundary] = D @ P # Obtaining 2cx + b, the deivative at x=xt

#     # =====================================================
#     # 1. Recent-time boundary: one-sided local linear (forward)
#     # =====================================================
#     for t in range(start_boundary):
#         h = minimum_num_idxs // 2
#         right = min(n_timepoints, t + h + 1)

#         idx = np.arange(t, right)
#         xloc = (idx - t) * time_step   # forward offsets

#         n = len(xloc)
#         S1 = xloc.sum()
#         S2 = (xloc**2).sum()

#         coeffs = (n * xloc - S1) / (n * S2 - S1**2)
#         S[t, t:right] = coeffs


#     # =====================================================
#     # 2. Interior: symmetric local linear
#     # =====================================================
#     # Filling in [start_boundary-end_boundary] rows of S
#     for k, t in enumerate(range(start_boundary, end_boundary)):
#         m = window_sizes[k]
#         if m % 2 == 0:
#             m += 1
#         h = (m - 1) // 2

#         j = np.arange(-h, h + 1)
#         coeffs = (3 * j) / (h * (h + 1) * (2 * h + 1)) / time_step
#         S[t, t - h : t + h + 1] = coeffs

#     # =====================================================
#     # 3. Deep-time boundary: one-sided local linear
#     # =====================================================
#     # Filling in [end_boundary-n_timepoints] rows of S
#     for t in range(end_boundary, n_timepoints):
#         h = maximum_num_idxs // 2
#         left = max(0, t - h)

#         idx = np.arange(left, n_timepoints)
#         xloc = (idx - t) * time_step

#         n = len(xloc)
#         S1 = xloc.sum()
#         S2 = (xloc**2).sum()

#         coeffs = (n * xloc - S1) / (n * S2 - S1**2)
#         S[t, left:] = coeffs
#     return S

# def compute_coal_rates(pdf_df, minimum_window_size, maximum_window_size, poly_deg = 1, min_rate = MIN_RATE):
#     times = pdf_df.columns.astype(float)
#     time_step = times[1]-times[0]
#     survival_df = 1 - pdf_df.cumsum(axis=1)
#     H = -np.log(survival_df)
#     W = adaptive_local_linear_filter(minimum_window_size, maximum_window_size, len(times), time_step, poly_deg)
#     # assert np.all(H.iloc[:,1:] >= H.iloc[:,:-1]), "cumulative hazard must be non-decreasing"
#     rates = H.values @ W.T
#     rates[rates<min_rate] = 0
#     rates_df = pd.DataFrame(rates,columns=times,index=survival_df.index)
#     return rates_df
# %%
# def adaptive_local_poly_derivative_matrix(
#     minimum_window_size,
#     maximum_window_size,
#     n_timepoints,
#     time_step,
#     poly_deg=1,
# ):
#     """
#     Build weight matrix W such that h = H @ W.T gives the smoothed
#     derivative of H at each time point.

#     Interior points: symmetric local polynomial of growing bandwidth.
#     Boundary points: single polynomial fit over the fixed boundary
#                      window, derivative evaluated at each v within it.

#     Parameters
#     ----------
#     minimum_window_size : float
#         Minimum full window size in time units (start boundary window).
#     maximum_window_size : float
#         Maximum full window size in time units (end boundary window).
#     n_timepoints : int
#         Number of time points T (indices 0 .. T-1).
#     time_step : float
#         Grid spacing Δ_t.
#     poly_deg : int
#         Degree p of the local polynomial (must be >= 1).

#     Returns
#     -------
#     W : np.ndarray, shape (T, T)
#         Weight matrix. Smoothed derivative is h = H @ W.T.
#     """
#     assert poly_deg >= 1, "Polynomial degree must be >= 1"

#     # -------------------------
#     # Window sizes (in indices)
#     # -------------------------
#     minimum_num_idxs = int(minimum_window_size // time_step)
#     if minimum_num_idxs % 2 == 0:
#         minimum_num_idxs += 1
#     if minimum_num_idxs < poly_deg + 2:  # Need at least p+1 points to fit, plus one more for derivative
#         minimum_num_idxs = poly_deg + 2

#     maximum_num_idxs = int(maximum_window_size // time_step)
#     if maximum_num_idxs % 2 == 0:
#         maximum_num_idxs += 1
#     if maximum_num_idxs < minimum_num_idxs:
#         maximum_num_idxs = minimum_num_idxs

#     # -------------------------
#     # Boundary cutoffs
#     # -------------------------
#     start_boundary = minimum_num_idxs // 2
#     end_boundary   = n_timepoints - maximum_num_idxs // 2

#     # Interior window sizes, forced odd
#     window_sizes = np.linspace(
#         minimum_num_idxs,
#         maximum_num_idxs,
#         end_boundary - start_boundary,
#     ).astype(int)
#     window_sizes = np.where(window_sizes % 2 == 0, window_sizes + 1, window_sizes)

#     W = np.zeros((n_timepoints, n_timepoints))

#     def _derivative_vector(t_eval, poly_deg):
#         """
#         Coefficient vector d(t) such that d(t)^T a = f'(t_eval),
#         where f(t) = sum_k a_k t^k.
#         d_k = k * t_eval^{k-1}
#         """
#         d = np.zeros(poly_deg + 1)
#         for k in range(1, poly_deg + 1):
#             d[k] = k * (t_eval ** (k - 1))
#         return d

#     # =====================================================
#     # 1. Start boundary: single polynomial over [0, m_min)
#     #    evaluated at each v in [0, start_boundary)
#     # =====================================================
#     indices_start = np.arange(minimum_num_idxs)
#     t_start       = indices_start * time_step                          # absolute times
#     Phi_start     = np.vander(t_start, N=poly_deg + 1, increasing=True)
#     Phi_start_inv = np.linalg.pinv(Phi_start)                         # (p+1, m_min)

#     for v in range(start_boundary):
#         t_eval = v * time_step
#         d      = _derivative_vector(t_eval, poly_deg)                  # (p+1,)
#         W[v, indices_start] = d @ Phi_start_inv                        # (m_min,)

#     # =====================================================
#     # 2. Interior: symmetric window, growing bandwidth
#     # =====================================================
#     for k, v in enumerate(range(start_boundary, end_boundary)):
#         m       = window_sizes[k]
#         h       = (m - 1) // 2
#         indices = np.arange(v - h, v + h + 1)
#         tau     = (indices - v) * time_step                            # centered at v
#         Phi     = np.vander(tau, N=poly_deg + 1, increasing=True)
#         W[v, indices] = np.linalg.pinv(Phi)[1, :]                     # a_1 row

#     # =====================================================
#     # 3. End boundary: single polynomial over [T-m_max, T)
#     #    evaluated at each v in [end_boundary, T)
#     # =====================================================
#     indices_end = np.arange(n_timepoints - maximum_num_idxs, n_timepoints)
#     t_end       = indices_end * time_step                              # absolute times
#     Phi_end     = np.vander(t_end, N=poly_deg + 1, increasing=True)
#     Phi_end_inv = np.linalg.pinv(Phi_end)                             # (p+1, m_max)

#     for v in range(end_boundary, n_timepoints):
#         t_eval = v * time_step
#         d      = _derivative_vector(t_eval, poly_deg)                  # (p+1,)
#         W[v, indices_end] = d @ Phi_end_inv                            # (m_max,)

#     return W