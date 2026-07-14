#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from plotting_style import *
import sys
import os
os.chdir(os.path.dirname(__file__))
sys.path.append("../../../src")
sys.path.append("../../../utils")


# %%
fig = plt.figure(figsize=(6,4))
filename = os.path.basename(__file__).split(".")[0]
fig.savefig(f"../outputs/{filename}.pdf")