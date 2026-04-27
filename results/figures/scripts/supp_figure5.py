#%%
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
os.chdir(os.path.dirname(__file__))
sys.path.append("../misc_scripts")
from plotting_style import *

fig = plt.figure(figsize=(6.5, 3.5))
filename = os.path.basename(__file__).split(".")[0]
fig.savefig(f"../outputs/{filename}.png")
# %%
