#%%
import numpy as np
import msprime
import matplotlib.pyplot as plt
from plotting_style import *
from tqdm import tqdm
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def two_pop_migration(Ne_A, Ne_B, m):
    demography = msprime.Demography.isolated_model([Ne_A, Ne_B])
    demography.set_migration_rate(source=0, dest=1, rate=m)
    demography.set_migration_rate(source=1, dest=0, rate=m)
    return demography

# %%
Ne_A = 1e4
Ne_B = 1e4
Nm_list = [1e-3, 1e-2, 1e-1, 1]
times = np.arange(1,200001,100)
within_rates_dict = {}
between_rates_dict = {}
for i, Nm in enumerate(Nm_list):
    m = Nm/(Ne_A)
    demography = two_pop_migration(Ne_A, Ne_B, m)
    debugger = demography.debug()
    between_rate,s = debugger.coalescence_rate_trajectory(steps = times, lineages = {0:1, 1:1})
    within_rate,s = debugger.coalescence_rate_trajectory(steps = times, lineages = {0:2, 1:0}) 
    within_rates_dict[Nm] = within_rate
    between_rates_dict[Nm] = between_rate

#%%
# Between and within coalescence rates over time for different Nm values
fig, axs = plt.subplots(1,4,figsize=(6.5,1.75), sharey=True, dpi=500)
for i, Nm in enumerate(Nm_list):
    axs[i].plot(times, between_rates_dict[Nm], label='Between')
    axs[i].plot(times, within_rates_dict[Nm], label='Within')
    axs[i].set_title(f'Nm = {Nm}', fontsize=10)
fig.supxlabel('Time (generations ago)', fontsize=10)
axs[0].set_ylabel('Coalescence rate', fontsize=10)
axs[-1].legend(fontsize=8)
fig.savefig("../outputs/mig_coal_rate.pdf", bbox_inches='tight', dpi=500)
#%%
# Saturation limit and rate of convergence for different Nm values
m_list = np.linspace(1e-7, 1e-4, 100)
times = np.arange(1,300001,100)
plateus = []
rates = []
intercepts = []
for m in tqdm(m_list):
    demography = two_pop_migration(Ne_A, Ne_B, m)
    debugger = demography.debug()
    between_rate,s = debugger.coalescence_rate_trajectory(steps = times, lineages = {0:1, 1:1})
    A = between_rate[-1] + 1e-10
    plateus.append(A)

    y = -np.log(1-between_rate/A)
    x = times
    slope, intercept = np.polyfit(x[:200], y[:200], 1)
    rates.append(slope)
    intercepts.append(intercept)
# %%
fig,axs = plt.subplots(1,2,figsize=(3.5,1.75), dpi=500)
axs[0].plot(m_list*Ne_A, plateus)
axs[1].plot(m_list*Ne_A, rates)

axs[0].set_xlabel('Nm', fontsize=8)
axs[0].set_ylabel('Limiting coalescent rate', fontsize=8)
axs[1].set_xlabel('Nm', fontsize=8)
axs[1].set_ylabel('Rate constant', fontsize=8)
axs[0].set_title('A', loc='left', fontweight='bold', fontsize=8)
axs[1].set_title('B', loc='left', fontweight='bold', fontsize=8)
axs[0].tick_params(axis='both', which='major', labelsize=8)
axs[1].tick_params(axis='both', which='major', labelsize=8)
fig.savefig("../outputs/mig_coal_rate_2.pdf", bbox_inches='tight', dpi=500)
# %%
