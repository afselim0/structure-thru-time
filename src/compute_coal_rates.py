#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd

def adaptive_local_linear_filter(minimum_window_size, maximum_window_size, n_timepoints, time_step):
    # -------------------------
    # Window sizes (in indices)
    # -------------------------
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

    # -------------------------
    # Boundaries
    # -------------------------
    start_boundary = minimum_num_idxs // 2
    end_boundary = n_timepoints - maximum_num_idxs // 2

    window_sizes = np.linspace(
        minimum_num_idxs,
        maximum_num_idxs,
        end_boundary - start_boundary
    ).astype(int)

    S = np.zeros((n_timepoints, n_timepoints))

    # # =====================================================
    # # 1. Recent-time boundary: quadratic fit (y = a + b x + c x^2)
    # # =====================================================
    # idx = np.arange(start_boundary)
    # x = (idx - idx.mean()) * time_step

    # X = np.column_stack([np.ones(start_boundary), x, x**2])
    # XtX_inv = np.linalg.inv(X.T @ X)
    # P = XtX_inv @ X.T # P @ y gives coefficients [a,b,c]

    # # Filling in [0-start_boundary] rows of S
    # for t in range(start_boundary):
    #     xt = x[t]
    #     D = np.array([0.0, 1.0, 2.0 * xt])
    #     S[t, :start_boundary] = D @ P # Obtaining 2cx + b, the deivative at x=xt

    # =====================================================
    # 1. Recent-time boundary: one-sided local linear (forward)
    # =====================================================
    for t in range(start_boundary):
        h = minimum_num_idxs // 2
        right = min(n_timepoints, t + h + 1)

        idx = np.arange(t, right)
        xloc = (idx - t) * time_step   # forward offsets

        n = len(xloc)
        S1 = xloc.sum()
        S2 = (xloc**2).sum()

        coeffs = (n * xloc - S1) / (n * S2 - S1**2)
        S[t, t:right] = coeffs


    # =====================================================
    # 2. Interior: symmetric local linear
    # =====================================================
    # Filling in [start_boundary-end_boundary] rows of S
    for k, t in enumerate(range(start_boundary, end_boundary)):
        m = window_sizes[k]
        if m % 2 == 0:
            m += 1
        h = (m - 1) // 2

        j = np.arange(-h, h + 1)
        coeffs = (3 * j) / (h * (h + 1) * (2 * h + 1)) / time_step
        S[t, t - h : t + h + 1] = coeffs

    # =====================================================
    # 3. Deep-time boundary: one-sided local linear
    # =====================================================
    # Filling in [end_boundary-n_timepoints] rows of S
    for t in range(end_boundary, n_timepoints):
        h = maximum_num_idxs // 2
        left = max(0, t - h)

        idx = np.arange(left, n_timepoints)
        xloc = (idx - t) * time_step

        n = len(xloc)
        S1 = xloc.sum()
        S2 = (xloc**2).sum()

        coeffs = (n * xloc - S1) / (n * S2 - S1**2)
        S[t, left:] = coeffs
    return S

def compute_coal_rates(pdf_df, minimum_window_size, maximum_window_size):
    times = pdf_df.columns
    time_step = times[1]-times[0]
    survival_df = 1 - pdf_df.cumsum(axis=1)
    H = -np.log(survival_df)
    S = adaptive_local_linear_filter(minimum_window_size, maximum_window_size, len(times), time_step)
    rates = H.values @ S.T
    rates_df = pd.DataFrame(rates,columns=times,index=survival_df.index)
    return rates_df