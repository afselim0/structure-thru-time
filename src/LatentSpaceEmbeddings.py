#!/usr/bin/env python3
import numpy as np
import pandas as pd
from optimization_functions import *
from tqdm import tqdm

class LatentSpaceEmbeddings:
    def __init__(self,
                 rates_df: pd.DataFrame,
                 sample_times: list = None,
                 N_min = None,
                 N_max = None,
                 n_components: int = 2):
        self.samples  = np.unique([x for pair in rates_df.index.values for x in pair])
        self.num_samples = len(self.samples)
        self.sample_idx_map = {sample: idx for idx, sample in enumerate(self.samples)}
        self.idx_sample_map = {idx: sample for idx, sample in enumerate(self.samples)}
        self.time_points = rates_df.columns.values
        self.num_timepoints = len(self.time_points)
        self.n_components = n_components

        R = np.full((self.num_samples, self.num_samples, self.num_timepoints), np.nan)
        for (s1, s2), row in rates_df.iterrows():
            i = self.sample_idx_map[s1]
            j = self.sample_idx_map[s2]
            R[i, j, :] = row.values
            R[j, i, :] = row.values
                
        self.R = R
        if N_max is not None:
            self.R += 1/N_max
    
        if sample_times is None:
            self.sample_times = np.array([0]*self.num_samples)
        else:
            self.sample_times = sample_times

        self.embeddings = None

        if N_min is not None:
            self.N_min = N_min
        else:
            # Selecting N_min as 1st percentile of Ne estimates
            with np.errstate(divide='ignore', invalid='ignore'):
                Ne = 1/(R + 1e-10)
            self.N_min = np.nanquantile(Ne, 0.01)

        sampled_by_time  = self.sample_times[:, np.newaxis] <= self.time_points
        self.sampled_by_time = sampled_by_time
        for t in range(self.num_timepoints):
            R[~sampled_by_time[:,t], :, t] = np.nan
            R[:, ~sampled_by_time[:,t], t] = np.nan

        # if individuals:
        #     R_individuals = np.full((int(self.num_idx/2), int(self.num_idx/2), self.num_timepoints), np.nan)
        #     for i in range(int(self.num_idx/2)):
        #         individual_i_rates = R[2*i:2*i+2, :, :]
        #         for j in range(i+1, int(self.num_idx/2)):
        #             individual_i_j_rates = individual_i_rates[:, 2*j:2*j+2, :]
        #             if np.all(np.isnan(individual_i_j_rates)):
        #                 continue
        #             R_individuals[i, j, :] = np.nanmean(individual_i_j_rates, axis=(0,1))
        #             R_individuals[j, i, :] = R_individuals[i, j, :]

        #     self.R = R_individuals
        #     self.num_idx = R_individuals.shape[0]
        #     self.sampled_by_time = self.sampled_by_time[::2, :]

    def compute_embeddings(self,X_init,time_idx, change_penalty=0, X_prev=None):
        # Compute embeddings for a specific time index
        R = self.R[:,:,time_idx]
        # Remove indices of samples that have not been sampled yet
        extant_samples = np.where(self.sampled_by_time[:,time_idx])[0]
        R = R[extant_samples,:,]
        R = R[:,extant_samples,]
        X = X_init[extant_samples, :]
        X,loss = compute_embeddings(X_init, R, self.N_min, change_penalty=change_penalty, X_prev=X_prev)
        return X, loss

    def compute_embeddings_through_time(self, X_init = None, change_penalty=0):
        embeddings = np.full((self.num_samples,self.n_components,self.num_timepoints),np.nan)
        if X_init is not None:
            X = X_init
        else:
            X = np.random.rand(self.num_samples,self.n_components)
        for k in tqdm(range(self.num_timepoints)):
            idx = len(self.time_points) - k - 1
            # idx = k
            X, _ = self.compute_embeddings(X_init = X, time_idx=idx, change_penalty=change_penalty, X_prev=X if k > 0 else None)
            if k==0:
                X -= X.mean(axis=0)
            embeddings[:,:,idx] = X
        embeddings = procrustes_alignment(embeddings)
        self.embeddings = embeddings
        return embeddings

    def get_embeddings_df(self):
        assert self.embeddings is not None, "Embeddings have not been computed yet. Please run compute_embeddings_through_time() first."
        sample_ids = [self.idx_sample_map[i] for i in np.arange(self.num_samples)]
        times = np.tile(self.time_points, self.num_samples)
        X1_flat = self.embeddings[:,0,:].flatten()
        X2_flat = self.embeddings[:,1,:].flatten()

        # Create the DataFrame
        df = pd.DataFrame({
            "sample_id": np.repeat(sample_ids, self.num_timepoints),
            "time": times,
            "X1": X1_flat,
            "X2": X2_flat
        })
        return df
# # %%
# import matplotlib.pyplot as plt
# from plotly import express as px
# rates = pd.read_parquet("../results/simulations/outputs/split/sim/data/rates.minWS101.maxWS501.V1.FS1.0.maxT4000.nT1000.parquet")
# rates.columns = rates.columns.astype(float)
# times = rates.columns.values
# LSE = LatentSpaceEmbeddings(rates_df=rates, n_components=2)
# R = LSE.R
# #%%
# LSE.N_min = 1
# LSE.N_max = 1000000
# alpha = 0
# embeddings = LSE.compute_embeddings_through_time(change_penalty=alpha)
# embeddings_df = LSE.get_embeddings_df()
# embeddings_df["color"] = np.nan
# embeddings_df.loc[embeddings_df["sample_id"]<20, "color"] = "red"
# embeddings_df.loc[embeddings_df["sample_id"]>=20, "color"] = "blue"
# # %%
# plt.plot(times,embeddings_df[embeddings_df["sample_id"]==0][["X2"]])
# #%%
# xy_min = min(embeddings_df["X1"].min(), embeddings_df["X2"].min())
# xy_max = max(embeddings_df["X1"].max(), embeddings_df["X2"].max())
# fig = px.line_3d(embeddings_df, x="X1", y="X2", z="time", line_group="sample_id", color="color", hover_data=["sample_id"])
# fig.update_layout(
#     scene=dict(
#         xaxis=dict(range=[xy_min, xy_max]),
#         yaxis=dict(range=[xy_min, xy_max]),
#         aspectmode="cube"
#     )
# )
# fig.show()
# # %%
