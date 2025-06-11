# TRASP
**T**ime-resolved **R**epresentations of **A**(RG) **S**tructure using **P**airwise-coalescent-rates

# Installations
We recommend cloning the git repo and creating a `conda` environment with the required dependencies from the `environment.yml` file

```
git clone https://github.com/afselim97/trasp.git
conda env create -f environment.yml
conda activate trasp
```

# Basic Usage
The input to **trasp** is one (or many) ARGs inferred using any inference software, and stored in the [tree_sequence](https://tskit.dev/software/tskit.html) format.
All inferred trees should be stored in a single directory, and have the suffix **.trees**

To run trasp:
'''
PATH_TO_REPP/src/run_trasp
'''

Only 3 flags are required (**-input**, **-output** and **metadata**), the rest are optional. Here is a list of available flags

| flag          | help                                                                                                                                                                   | default                                   |
|----------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------|
| **-input**    | Either the path to the directory containing inferred tree sequence files ending with **.trees** *OR* path to demes file to be simulated                                  | N/A                                       |
| **-output**   | Path to output directory                                                                                                                                                 | N/A                                       |
| **-metadata** | Path to metadata file with population label for each sample. Should be a CSV with 2 columns: ("sample_id","population"). Required only if mode is "inferred"             | N/A                                       |
| **-mode**     | Either *simulated* if input is a demes file or *inferred* if the input is a directory containing inferred trees                                                          | inferred                                  |
| **-L**        | Number of trees to be sampled from the tree sequences (or simulated)                                                                                                     | 10000                                     |
| **-min_time** | Earliest time to compute rates and embeddings                                                                                                                            | 0.01 quantile of all coalescence events   |
| **-max_time** | Deepest time to compute rates and embeddings                                                                                                                             | 0.95 quantile of all coalescence events   |
| **-num_timepoints** | Number of timepoints to compute rates and embeddings                                                                                                              | 100                                       |
| **-log_time** | If set, picks timepoints between minimum or maximum on a log scale. Otherwise, uses a linear scale.                                                                       | False                                     |
| **-delta**    | The time window for computing the rate estimates at each timepoint (In python script, can use different window sizes for each point)                                      | 200/(Ne_estimate*L)                       |


For example, running the following command will simulate a basic population split scenario, and apply the method to it.

```
src/run_trasp -input data/demes/split.yaml -output data/results/simulations/split -mode simulated -L 10000 -delta 100
```