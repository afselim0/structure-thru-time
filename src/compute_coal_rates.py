#%%
#!/usr/bin/env python3
from os import times

import numpy as np
import pandas as pd

MIN_RATE = 1e-12

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

def adaptive_local_poly_derivative_matrix(
    minimum_window_size,
    maximum_window_size,
    n_timepoints,
    time_step,
    poly_deg=3,
):
    """
    Build weight matrix W such that h = H @ W.T gives the smoothed
    derivative of H at each time point.

    Interior points: symmetric local polynomial of growing bandwidth.
    Boundary points: single polynomial fit over the fixed boundary
                     window, derivative evaluated at each v within it.

    Parameters
    ----------
    minimum_window_size : float
        Minimum full window size in time units (start boundary window).
    maximum_window_size : float
        Maximum full window size in time units (end boundary window).
    n_timepoints : int
        Number of time points T (indices 0 .. T-1).
    time_step : float
        Grid spacing Δ_t.
    poly_deg : int
        Degree p of the local polynomial (must be >= 1).

    Returns
    -------
    W : np.ndarray, shape (T, T)
        Weight matrix. Smoothed derivative is h = H @ W.T.
    """
    assert poly_deg >= 1, "Polynomial degree must be >= 1"

    # -------------------------
    # Window sizes (in indices)
    # -------------------------
    minimum_num_idxs = int(minimum_window_size // time_step)
    if minimum_num_idxs % 2 == 0:
        minimum_num_idxs += 1
    if minimum_num_idxs < poly_deg + 2:  # Need at least p+1 points to fit, plus one more for derivative
        minimum_num_idxs = poly_deg + 2

    maximum_num_idxs = int(maximum_window_size // time_step)
    if maximum_num_idxs % 2 == 0:
        maximum_num_idxs += 1
    if maximum_num_idxs < minimum_num_idxs:
        maximum_num_idxs = minimum_num_idxs

    # -------------------------
    # Boundary cutoffs
    # -------------------------
    start_boundary = minimum_num_idxs // 2
    end_boundary   = n_timepoints - maximum_num_idxs // 2

    # Interior window sizes, forced odd
    window_sizes = np.linspace(
        minimum_num_idxs,
        maximum_num_idxs,
        end_boundary - start_boundary,
    ).astype(int)
    window_sizes = np.where(window_sizes % 2 == 0, window_sizes + 1, window_sizes)

    W = np.zeros((n_timepoints, n_timepoints))

    def _derivative_vector(t_eval, poly_deg):
        """
        Coefficient vector d(t) such that d(t)^T a = f'(t_eval),
        where f(t) = sum_k a_k t^k.
        d_k = k * t_eval^{k-1}
        """
        d = np.zeros(poly_deg + 1)
        for k in range(1, poly_deg + 1):
            d[k] = k * (t_eval ** (k - 1))
        return d

    # =====================================================
    # 1. Start boundary: single polynomial over [0, m_min)
    #    evaluated at each v in [0, start_boundary)
    # =====================================================
    indices_start = np.arange(minimum_num_idxs)
    t_start       = indices_start * time_step                          # absolute times
    Phi_start     = np.vander(t_start, N=poly_deg + 1, increasing=True)
    Phi_start_inv = np.linalg.pinv(Phi_start)                         # (p+1, m_min)

    for v in range(start_boundary):
        t_eval = v * time_step
        d      = _derivative_vector(t_eval, poly_deg)                  # (p+1,)
        W[v, indices_start] = d @ Phi_start_inv                        # (m_min,)

    # =====================================================
    # 2. Interior: symmetric window, growing bandwidth
    # =====================================================
    for k, v in enumerate(range(start_boundary, end_boundary)):
        m       = window_sizes[k]
        h       = (m - 1) // 2
        indices = np.arange(v - h, v + h + 1)
        tau     = (indices - v) * time_step                            # centered at v
        Phi     = np.vander(tau, N=poly_deg + 1, increasing=True)
        W[v, indices] = np.linalg.pinv(Phi)[1, :]                     # a_1 row

    # =====================================================
    # 3. End boundary: single polynomial over [T-m_max, T)
    #    evaluated at each v in [end_boundary, T)
    # =====================================================
    indices_end = np.arange(n_timepoints - maximum_num_idxs, n_timepoints)
    t_end       = indices_end * time_step                              # absolute times
    Phi_end     = np.vander(t_end, N=poly_deg + 1, increasing=True)
    Phi_end_inv = np.linalg.pinv(Phi_end)                             # (p+1, m_max)

    for v in range(end_boundary, n_timepoints):
        t_eval = v * time_step
        d      = _derivative_vector(t_eval, poly_deg)                  # (p+1,)
        W[v, indices_end] = d @ Phi_end_inv                            # (m_max,)

    return W

def compute_coal_rates(pdf_df, minimum_window_size, maximum_window_size, poly_deg = 1, min_rate = MIN_RATE):
    times = pdf_df.columns.astype(float)
    time_step = times[1]-times[0]
    survival_df = 1 - pdf_df.cumsum(axis=1)
    H = -np.log(survival_df)
    W = adaptive_local_poly_derivative_matrix(minimum_window_size, maximum_window_size, len(times), time_step, poly_deg)
    rates = H.values @ W.T
    rates[rates<min_rate] = 0
    rates_df = pd.DataFrame(rates,columns=times,index=survival_df.index)
    return rates_df
# # %%
# pdf_df = pd.read_parquet("../results/simulations/outputs/zigzag/sim/data/coal_pdf.V1.FS1.0.maxT40000.nT4000.parquet")
# true_rates = pd.read_parquet("../results/simulations/outputs/zigzag/sim/data/rates_true.parquet").mean()
# minimum_window_size = 1
# maximum_window_size = 0
# poly_deg = 1
# rates = compute_coal_rates(pdf_df, minimum_window_size, maximum_window_size, poly_deg, min_rate = MIN_RATE)
# mean_rates = rates.mean()
# # %%
# import matplotlib.pyplot as plt
# plt.plot(times, true_rates, label="True")
# plt.plot(times, mean_rates, label="Estimated")
# plt.legend()
# plt.xlabel("Time")
# plt.ylabel("Coalescence Rate")
# plt.xscale("log")
# # %%
# for row in rates.itertuples():
#     plt.plot(times, row[1:], alpha=0.1)
# plt.plot(times, true_rates, label="True", color="black", linewidth=2)
# plt.xscale("log")
# plt.xlabel("Time")
# plt.ylabel("Coalescence Rate")
# # %%
# plt.plot(rates.iloc[0,:])
# plt.xscale("log")
# # %%
