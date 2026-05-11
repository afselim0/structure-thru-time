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

    t_min = minimum_num_idxs // 2
    t_max = n_timepoints - maximum_num_idxs // 2

    if t_max <= t_min:
        raise ValueError("Time series too short for the requested window sizes.")

    window_sizes = np.linspace(minimum_num_idxs, maximum_num_idxs, t_max - t_min).astype(int)

    S        = np.zeros((n_timepoints, n_timepoints))
    A        = np.zeros((n_timepoints, n_timepoints))
    se_scale = np.full(n_timepoints, np.nan)

    for k, t in enumerate(range(t_min, t_max)):
        m = window_sizes[k]
        if m % 2 == 0:
            m += 1
        h    = (m - 1) // 2
        xloc = np.arange(-h, h + 1) * time_step   # S1 = 0 by symmetry
        S2   = (xloc ** 2).sum()
        S[t, t - h : t + h + 1] = xloc / S2        # slope:     j*dt / S2
        A[t, t - h : t + h + 1] = 1.0 / m          # intercept: local mean
        se_scale[t]              = 1.0 / np.sqrt(S2) # SE = s_hat / sqrt(S2)

    return S, A, se_scale, t_min, t_max


def compute_coal_rates(pdf_df, minimum_window_size, maximum_window_size, min_rate=MIN_RATE):
    times     = pdf_df.columns.astype(float)
    time_step = times[1] - times[0]
    n_T       = len(times)

    survival_df = 1 - pdf_df.cumsum(axis=1)
    H = -np.log(survival_df.values.clip(1e-10))   # (n_pairs, n_T); clip avoids -inf at S=0

    S, A, se_scale, t_min, t_max = adaptive_local_linear_filter(
        minimum_window_size, maximum_window_size, n_T, time_step
    )

    print("Multiplying cumulative rate matrix with weights to compute slopes (rates)")
    rates_raw  = H @ S.T   # (n_pairs, n_T)
    print("Multiplying cumulative rate matrix with weights to compute intercepts")
    intercepts = H @ A.T   # (n_pairs, n_T)

    # SE from local linear residuals (computed on unclipped rates)
    print("Computing standard errors")
    se = np.full_like(rates_raw, np.nan)
    for t in tqdm(range(t_min, t_max)):
        m    = np.count_nonzero(A[t])              # = 2h+1, non-zero at all window points
        h    = (m - 1) // 2
        l, r = t - h, t + h + 1
        xloc      = np.arange(-h, h + 1) * time_step          # (m,)
        fitted    = (intercepts[:, [t]]
                     + rates_raw[:, [t]] * xloc[np.newaxis, :]) # (n_pairs, m)
        residuals = H[:, l:r] - fitted                          # (n_pairs, m)
        s_hat     = np.sqrt(np.sum(residuals**2, axis=1) / (m - 2))
        se[:, t]  = s_hat * se_scale[t]

    # Clip and mask
    rates = rates_raw.copy()
    # rates[rates < min_rate] = 0
    rates = rates[:, t_min:t_max]
    se = se[:, t_min:t_max]

    rates_df = pd.DataFrame(rates, columns=times[t_min:t_max], index=pdf_df.index)
    se_df    = pd.DataFrame(se,    columns=times[t_min:t_max], index=pdf_df.index)
    return rates_df, se_df
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