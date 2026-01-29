# Installations
We recommend cloning the git repo and creating a `conda` environment with the required dependencies from the `environment.yml` file
```
git clone https://github.com/afselim0/structure-thru-time
conda env create -f environment.yml
conda activate STT
```

# Basic Usage
The main input to **sturcture-thru-time** a directory with a list of [tree sequences](https://tskit.dev/software/tskit.html) format. They should all end with the suffix **.trees**. If the tree sequences are samples from a posterior (singer), each sample should be saved as `{ContigID}_{PosteriorSample}.trees`

To run STT:
```
results/empirical/snakemake --config \
    trees_dir= PATH/TO/TREES \
    output_dir=PATH/TO/OUTPUT/DIR
    {param}={param_value} \
    ...
```

Parameters are passed to the snakemake following `--config` as shown above. Refer to `results/empirical/test.sh` for an example. 
The snakemake is engineered in a way where changing a parameter will only reproduce the results that depend on that parameter onward. Each output file is tagged with parameters which that output depends on. Current standard outputs include:

1. Inferred probability density function of pairwise coalescent rates (this is the step that takes most time)
2. Inferred pairwise coalescent rates as a function of time + plots of rates over time
3. Inferred embeddings of samples over time + plots of embeddings
4. Performance evaluation data + plots

The following are parameters that can be passed to `--config` and that affect downstream results
| Parameter                | Description                                                                  | Affected files |
|--------------------------|------------------------------------------------------------------------------|----------------|
| **trees_dir**            | Path to a directory containing `.trees` files                                | ALL            |
| **output_dir**           | Path to the output directory                                                 | ALL            |
| **num_posterior_samples**| Number of posterior samples (if `singer` was used for inference)             | ALL            |
| **pops**                 | Subset of populations to draw samples from (default: all)                    | ALL            |
| **n_samples**            | Number of samples per population (default: all)                              | ALL            |
| **max_time**             | Maximum time used for computing PDFs, rates, and embeddings                  | PDF            |
| **n_timepoints**         | Number of equally spaced time points from `t = 0` to `max_time`              | PDF            |
| **fraction_samples**     | Fraction of samples used per contig (default: 1.0)                           | PDF            |
| **num_threads**          | Number of parallel threads used for PDF computation                          | PDF            |
| **min_window_size**      | Minimum window size for local regression when computing rates                | Rates          |
| **max_window_size**      | Maximum window size for local regression when computing rates (linear growth)| Rates          |
| **N0**                   | Controls latent space resolution                                             | Embeddings     |
| **min_time**             | Minimum time shown in plots                                                  | Plotting       |
| **pop_labels**           | Path to population labels for each sample                                    | Plotting       |
| **pop_colors**           | Path to population colors                                                    | Plotting       |

Since we can't push large files on github, we will start gathering interesting outputs in the directory `/project/jnovembre/STT_data` on `Midway3`.  


