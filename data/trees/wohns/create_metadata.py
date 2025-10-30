#%%
import numpy as np
import pandas as pd
import json
import tskit
import glob


ts_files = glob.glob("*.trees")
ts = tskit.load(ts_files[0])
samples = ts.samples()
n = len(samples)
K = ts.num_populations
populations = np.zeros(K,dtype='<U30')
regions = np.zeros(K,dtype='<U30')
descriptions = np.zeros(K,dtype='<U50')

sample_regions = np.zeros(n,dtype='<U30')
sample_populations = np.zeros(n,dtype='<U30')
sample_descriptions = np.zeros(n,dtype='<U50')

# Get population and region names
for population in ts.populations():
    id = population.id
    metadata = json.loads(population.metadata.decode())
    try:
        description = metadata["description"]
    except:
        description = metadata["name"]

    name = metadata["name"]
    try:
        region = metadata["region"]
    except:
        region = metadata["super_population"]

    descriptions[id] = description
    populations[id] = name
    regions[id] = region


# Adjust Region Names
regions[regions=="AFR"] = "Africa"
regions[regions=="AFRICA"] = "Africa"
regions[regions=="AMERICA"] = "America"
regions[regions=="AMR"] = "America"
regions[regions=="EAST_ASIA"] = "East_Asia"
regions[regions=="EastAsia"] = "East_Asia"
regions[regions=="EAS"] = "East_Asia"
regions[regions=="CENTRAL_SOUTH_ASIA"] = "Central_South_Asia"
regions[regions=="CentralAsiaSiberia"] = "Central_Asia_Siberia"
regions[regions=="EUROPE"] = "Western_Europe"
regions[regions=="EUR"] = "Western_Europe"
regions[regions=="MIDDLE_EAST"] = "Middle_East"
regions[regions=="OCEANIA"] = "Oceania"
regions[regions=="SAS"] = "South_Asia"
regions[regions=="SouthAsia"] = "South_Asia"
regions[regions=="WestEurasia"] = "West_Eurasia"

regions[regions=="Afanasievo"] = "Ancient"
regions[regions=="Max Planck"] = "Ancient"

regions[np.where(populations=="Yemenite_Jew")] = "Middle_East"
regions[np.where(populations=="Druze")] = "Middle_East"
regions[np.where(populations=="BedouinB")] = "Middle_East"
regions[np.where(populations=="Iranian")] = "Middle_East"
regions[np.where(populations=="Jordanian")] = "Middle_East"
regions[np.where(populations=="Palestinian")] = "Middle_East"
regions[np.where(populations=="Samaritan")] = "Middle_East"


regions[np.where(populations=="English")] = "Western_Europe"
regions[np.where(populations=="Finnish")] = "Western_Europe"
regions[np.where(populations=="Norwegian")] = "Western_Europe"
regions[np.where(populations=="Saami")] = "Western_Europe"
regions[np.where(populations=="French")] = "Western_Europe"
regions[np.where(populations=="Icelandic")] = "Western_Europe"
regions[np.where(populations=="Basque")] = "Western_Europe"
regions[np.where(populations=="Bergamo")] = "Western_Europe"
regions[np.where(populations=="Sardinian")] = "Western_Europe"
regions[np.where(populations=="Spanish")] = "Western_Europe"
regions[np.where(populations=="Tuscan")] = "Western_Europe"


# Adjust Population Names

# Get population and region for each sample
samples = ts.samples()
for i,sample in enumerate(samples):
    population_id = ts.node(sample).population
    sample_regions[i] = regions[population_id]
    sample_populations[i] = populations[population_id]
    sample_descriptions[i] = descriptions[population_id]

# Create df
data = {
    "sample_id": samples,
    "population": sample_populations,
    "region": sample_regions,
    "description": sample_descriptions
}
df = pd.DataFrame(data)
df.to_csv("metadata.csv",index=False)

# %%