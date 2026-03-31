#%%
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import sys
sys.path.append("../../utils")
from plotting_style import *
#%%
df = pd.read_csv("../metadata/pop_locs.csv",index_col=0)
df = df[df["dataset"]!="ggvp"]
pop_colors = pd.read_csv("../metadata/Wohns_pop_colors.csv",index_col=0)
pop_colors = pop_colors.rename(columns={"population":"pop"})

pop_labels = df["pop"].values
pop_labels_adjusted = []
for pop_label in pop_labels:
    pop_label = pop_label.removesuffix("SGDP")
    pop_label = pop_label.removesuffix("HGDP")
    pop_labels_adjusted.append(pop_label)

## Adjusting some names
pop_labels_adjusted[pop_labels_adjusted.index('EskimoChaplin')] = "Eskimo_Chaplin"
pop_labels_adjusted[pop_labels_adjusted.index('EskimoNaukan')] = "Eskimo_Naukan"
pop_labels_adjusted[pop_labels_adjusted.index('EskimoSireniki')] = "Eskimo_Sireniki"
pop_labels_adjusted[pop_labels_adjusted.index('IraqiJew')] = "Iraqi_Jew"
pop_labels_adjusted[pop_labels_adjusted.index('JuhoanNorth')] = "Ju_hoan_North"
pop_labels_adjusted[pop_labels_adjusted.index('KhomaniSan')] = "Khomani_San"
pop_labels_adjusted[pop_labels_adjusted.index('KhondaDora')] = "Khonda_Dora"
pop_labels_adjusted[pop_labels_adjusted.index('NorthOssetian')] = "North_Ossetian"
pop_labels_adjusted[pop_labels_adjusted.index('YemeniteJew')] = "Yemenite_Jew"
pop_labels_adjusted[pop_labels_adjusted.index('MKK')] = "Masai"
##

df["pop"] = pop_labels_adjusted
df = df.merge(pop_colors, on="pop", how="left")
# %%
url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
world = gpd.read_file(url)

fig, ax = plt.subplots(figsize=(12,6), dpi= 500)

# Plot world map
world.plot(ax=ax, color="lightgray", edgecolor=None)

# Plot your points
ax.scatter(df["lng"], df["lat"], c=df["color"], s=50)

ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

fig.savefig("pop_locs_map.png", dpi=1000)
for _, row in df.iterrows():
    ax.text(
        row["lng"],
        row["lat"],
        row["pop"][:3],
        fontsize=8,
        ha="left",
        va="bottom",
        c=row["color"]
    )

fig.show()
fig.savefig("pop_locs_map_text.png", dpi=1000)
# %%
