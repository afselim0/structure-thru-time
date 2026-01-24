#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def gaussian_smoothing_matrix(T, time_step, sigma, truncate=2):
    K = int(truncate * sigma / time_step)
    offsets = np.arange(-K, K + 1)
    kernel = np.exp(-(offsets*time_step)**2 / (2 * sigma**2))

    G = np.zeros((T, T))
    for t in range(T):
        for k, w in zip(offsets, kernel):
            j = t + k
            if 0 <= j < T:
                G[t, j] += w

        # normalize each row
        G[t] /= G[t].sum()

    return G

def compute_coal_rates(pdf_df, window_size, local_poly_order = 3):
    """Compute pairwise coalescent rates

    Args:
        pdf_df (pd.DataFrame): PDF computed from compute_coal_time_pdf
        window_size (int): window size used to calculate derivatives of the negative log survival function

    Returns:
        pd.DataFrame: A dataframe for pairwise coalescent rates where each row corresponds to a sample pair, and each column corresponds to a time bin. The values represent the estimated coalescent rate within the respective time bin for the sample pair.
    """
    times = pdf_df.columns
    time_step = times[1]-times[0]
    survival_df = 1 - pdf_df.cumsum(axis=1)
    H = -np.log(survival_df)

    m = int(window_size//time_step)
    if m < local_poly_order+1:
        m = local_poly_order+1
     # Cumulative hazard function (cumulative coalescent rate)
    rates = savgol_filter(H,window_length=m,polyorder=local_poly_order,deriv=1,delta=time_step,axis=1)
    rates_df = pd.DataFrame(rates,columns=times,index=survival_df.index)
    return rates_df
# # %%
# pdf_df = pd.read_parquet("../results/empirical/DeManuel/all_npp.10/data/pdf.max_t100000.num_t20000.fp1.parquet")
# time_step = 5
# survival_df = 1 - pdf_df.cumsum(axis=1)
# H = -np.log(survival_df)
# t = pdf_df.columns
# x = H.iloc[100,:].values
# # %%
# window_length = 100
# r1 = savgol_filter(x,window_length=int(window_length//time_step),polyorder=1,deriv=1,delta=time_step)
# r2 = savgol_filter(x,window_length=int(window_length//time_step),polyorder=2,deriv=1,delta=time_step)
# r3 = savgol_filter(x,window_length=int(window_length//time_step),polyorder=3,deriv=1,delta=time_step)
# plt.plot(t, r1, label='polyorder=1')
# plt.plot(t, r2, label='polyorder=2')
# plt.plot(t, r3, label='polyorder=3')
# plt.xscale('log')
# plt.axvline(x=window_length/2, color='gray', linestyle='--', label='window size')
# # %%
# import tskit
