#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

from tqdm import tqdm
sys.path.append('../../../../src')
from plot_params import *
from scipy.spatial import ConvexHull

def embeddings_df_to_matrix(df):
    n = df['sample_id'].nunique()
    num_t_points = df['time'].nunique()
    times = df['time'].unique()
    df_sorted = df.sort_values(['sample_id', 'time'])
    X_through_time = df_sorted[['X1', 'X2']].values
    X_through_time = X_through_time.reshape((n, num_t_points, 2)).transpose(0, 2, 1)
    
    return X_through_time, times

def zigzag_Ne(times):
    growth_1_rate = 8.99448e-05
    decline_1_rate = -0.00035977
    growth_2_rate = 0.0014391
    decline_2_rate = -0.005756
    growth_3_rate = 0.023025
    growth_1_time = 34133.31
    decline_1_time = 8533.33
    growth_2_time = 2133.33
    decline_2_time = 533.33
    growth_3_time = 133.33
    growth_3_end_time = 33.333

    max_Ne = 7156
    min_Ne = 71560

    Ne_through_time = []
    # Ne_average_through_time = []
    for time in times[::-1]:
        if time > growth_1_time:
            Ne = max_Ne
            # Ne_average = max_Ne
        elif growth_1_time >= time > decline_1_time:
            Ne = max_Ne * np.exp(growth_1_rate * (growth_1_time - time))
            # Ne_average = Ne * ((delta * (-growth_1_rate))/(1-np.exp(growth_1_rate*delta)))
        elif decline_1_time >= time > growth_2_time:
            Ne = min_Ne * np.exp(decline_1_rate * (decline_1_time - time))
            # Ne_average = Ne * ((delta * (-decline_1_rate))/(1-np.exp(decline_1_rate*delta)))
        elif growth_2_time >= time > decline_2_time:
            Ne = max_Ne * np.exp(growth_2_rate * (growth_2_time - time))
            # Ne_average = Ne * ((delta * (-growth_2_rate))/(1-np.exp(growth_2_rate*delta)))
        elif decline_2_time >= time > growth_3_time:
            Ne = min_Ne * np.exp(decline_2_rate * (decline_2_time - time))
            # Ne_average = Ne * ((delta * (-decline_2_rate))/(1-np.exp(decline_2_rate*delta)))
        elif growth_3_time >= time > growth_3_end_time:
            Ne = max_Ne * np.exp(growth_3_rate * (growth_3_time - time))
            # Ne_average = Ne * ((delta * (-growth_3_rate))/(1-np.exp(growth_3_rate*delta)))
        else:
            Ne = min_Ne
            # Ne_average = min_Ne
        Ne_through_time.append(Ne)
        # Ne_average_through_time.append(Ne_average)

    return Ne_through_time[::-1]#, Ne_average_through_time[::-1]

def Average_Ne(Ne_through_time,times,delta=100):
    finer_time_points = np.linspace(times[0], times[-1]+500, 50000)
    dt = finer_time_points[1] - finer_time_points[0]
    Ne_through_time = zigzag_Ne(finer_time_points)
    Ne_ave = []
    for i,time in enumerate(times):
        mask = (finer_time_points > time) & (finer_time_points <= time + delta)
        recip_Ne = dt/(finer_time_points[-1]-finer_time_points[0]) * np.sum(1/np.array(Ne_through_time)[mask][:-1])
        Ne_ave.append(1/recip_Ne)
    return Ne_ave

def centroids_through_time(X_through_time,K):
    n = int(X_through_time.shape[0]/K)
    centroids = np.zeros((K,2,X_through_time.shape[2]))
    for k in range(K):
        centroids[k,:,:] = np.nanmean(X_through_time[k*n:(k+1)*n,:,:],axis=0)
    return np.array(centroids)
# %%
## 1-Zigzag
delta = 100
data = pd.read_parquet("../../outputs/main/zigzag/embeddings.parquet")
X_through_time, times = embeddings_df_to_matrix(data)
Ne_through_time = zigzag_Ne(times)
Average_Ne_through_time = Average_Ne(Ne_through_time, times, delta)
log_Ne_through_time = np.log(Ne_through_time)
log_Average_Ne_through_time = np.log(Average_Ne_through_time)
areas = []
radii = []
for i in range(len(times)):
    X = X_through_time[:,:,i]
    hull = ConvexHull(X)
    area = hull.volume 
    radius = np.sqrt(area/np.pi)
    areas.append(area)
    radii.append(radius)

rcParams['font.size'] = 12
fig, ax1 = plt.subplots(figsize=(6, 6), dpi=500)

color = "blue"
ax1.set_xlabel("Time")
ax1.set_ylabel(r"$\sqrt{\mathrm{r}}$", color=color)
ax1.plot(times, np.sqrt(radii), color=color, label="Hull Diameter", lw=2)
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_xscale("log")

ax2 = ax1.twinx()
color = "black"
ax2.set_ylabel(r"$\ln(\mathrm{N_e})$", color=color)
# ax2.plot(times, log_Ne_through_time, color=color, label="ln(Ne)",lw=2,ls=":")
ax2.plot(times, log_Average_Ne_through_time, color=color, label="smoothed ln(Ne)",lw=2,ls = "--")
ax2.tick_params(axis='y', labelcolor=color)
ax2.set_xscale("log")
ax2.spines['right'].set_visible(True)  # Ensure right y-axis line is visible
ax1.tick_params(axis='y', which='major', length=5, width=2)
ax2.tick_params(axis='y', which='major', length=5, width=2)

# fig.legend(loc="upper right", bbox_to_anchor=(1.6,1), bbox_transform=ax1.transAxes)
# fig.legend()
fig.tight_layout()
plt.show()
fig.savefig("zigzag_diameter_vs_Ne.png", dpi=500)
# %%
## 2-Split
delta = 100
data = pd.read_parquet("../../outputs/main/split/embeddings.parquet")
X_through_time, times = embeddings_df_to_matrix(data)
centroids = centroids_through_time(X_through_time, K=2)
distances_through_time = np.linalg.norm(centroids[0,:,:]-centroids[1,:,:], axis=0)

fig,ax = plt.subplots(figsize=(6,6), dpi=500)
ax.plot(times, distances_through_time, color="black", lw=2)
ax.axvline(x=10000, color="red", ls="--", lw=2,label="Split time")
ax.set_xlabel("Time")
ax.set_ylabel("Distance between clusters")
ax.set_ylim(0, np.max(distances_through_time)*1.1)
fig.tight_layout()
fig.legend()
fig.savefig("split_distance_through_time.png", dpi=500)
# %%
## 3-Split_mig
delta = 100
data = pd.read_parquet("../../outputs/main/split_mig_Nm_1/embeddings.parquet")
X_through_time, times = embeddings_df_to_matrix(data)
centroids = centroids_through_time(X_through_time, K=2)
distances_through_time = np.linalg.norm(centroids[0,:,:]-centroids[1,:,:], axis=0)

fig,ax = plt.subplots(figsize=(6,6), dpi=500)
ax.plot(times, distances_through_time, color="black", lw=2)
ax.axvline(x=5000, color="red", ls="--", lw=2,label="Split time")
ax.axvline(x=2500, color="blue", ls=":", lw=2,label="Migration time")
ax.set_xlabel("Time")
ax.set_ylabel("Distance between clusters")
ax.set_ylim(0, np.max(distances_through_time)*1.1)
fig.tight_layout()
fig.legend()
fig.savefig("split_mig_distance_through_time.png", dpi=500)
# %%
## 4-Mig_change
delta = 100
data = pd.read_parquet("../../outputs/main/mig_change/embeddings.parquet")
X_through_time, times = embeddings_df_to_matrix(data)
centroids = centroids_through_time(X_through_time, K=3)
distances_through_time_A_B = np.linalg.norm(centroids[0,:,:]-centroids[1,:,:], axis=0)
distances_through_time_A_C = np.linalg.norm(centroids[0,:,:]-centroids[2,:,:], axis=0)
distances_through_time_B_C = np.linalg.norm(centroids[1,:,:]-centroids[2,:,:], axis=0)

fig,ax = plt.subplots(figsize=(6,6), dpi=500)
ax.plot(times, distances_through_time_A_B, color="red", lw=2,label="A-B")
ax.plot(times, distances_through_time_A_C, color="darkblue", lw=2,label="A-C")
ax.plot(times, distances_through_time_B_C, color="darkgreen", lw=2,label="B-C")
ax.axvline(x=100, color="black", ls="--", lw=2,label="A-B migration start")
ax.axvline(x=2000, color="black", ls=":", lw=2,label="Split time")
ax.set_ylim(0, np.max(distances_through_time_A_B)*1.1)
ax.set_xlabel("Time")
ax.set_ylabel("Distance between clusters")
ax.set_xscale("log")
fig.tight_layout()
fig.legend(fontsize=8)
fig.savefig("mig_change_distance_through_time.png", dpi=500)
# %%
## 5-SS
delta = 100
data = pd.read_parquet("../../outputs/main/SS_N100_m_001/embeddings.parquet")
X_through_time, times = embeddings_df_to_matrix(data)
centroids = centroids_through_time(X_through_time, K=100)
distances_through_time = np.linalg.norm(centroids[0,:,:]-centroids[-1,:,:], axis=0)
fig,ax = plt.subplots(figsize=(6,6), dpi=500)
ax.set_ylim(0, np.max(distances_through_time)*1.1)
ax.plot(times, distances_through_time, color="black", lw=2)
ax.set_xlabel("Time")
ax.set_ylabel("Distance between first and last deme")
ax.set_xscale("log")
fig.tight_layout()
fig.savefig("SS_distance_through_time.png", dpi=500)

# %%