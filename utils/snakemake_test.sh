#!/bin/bash
: <<'Instructions'
A test script that outlines how to use the master snakemake.
The master snakemake can perform 3 main pipelines:
1- Simulating trees based on a demography, and running the STT on them
2- Inferring trees using an ARG inference method (tsinfer, relate, singer) and running STT on the inferred trees (simulations must be done first)
3- Running STT on trees that are already inferred
Instructions

# Directory names
main_demes_dir=../data/demes/main
stdpopsim_demes_dir=../data/demes/stdpopsim


# Simulated Data Examples
## Simulate
dempgraphy=split

# snakemake sim_STT --cores all --config \
#     demes_file_path="$main_demes_dir"/"$dempgraphy".yaml \
#     output_path=../results/simulations/outputs/"$dempgraphy"/sim \
#     num_contigs=2500 \
#     contig_length=1e6 \
#     n_sim_per_pop=20 \
#     max_time=10000 \
#     n_timepoints=2000 \
#     min_window_size=50 \
#     max_window_size=100 \
#     n_sim_per_pop=10 \
#     N_min=100 \
#     N_max=100000 \
#     min_time=100 \
#     l1=1e-3,1e-2 \
#     l2=1e-2 \
#     num_threads=12

## Infer
### 1. tsinfer
# infer_method=tsinfer
# snakemake infer_STT --cores all --config \
#     vcf_dir=../results/simulations/outputs/"$dempgraphy"/sim/trees \
#     output_path=../results/simulations/outputs/"$dempgraphy"/"$infer_method" \
#     infer_method="$infer_method" \
#     num_contigs=2500 \
#     contig_length=1e6 \
#     n_sim_per_pop=20 \
#     max_time=10000 \
#     n_timepoints=2000 \
#     min_window_size=50 \
#     max_window_size=1000 \
#     n_sim_per_pop=10 \
#     N_min=100 \
#     N_max=100000 \
#     min_time=100 \
#     num_posterior_samples=5 \
#     thinning_interval=10 \
#     burnin=500 \
#     num_threads=12
    
# ### 2. relate
# infer_method=relate
# snakemake infer_STT --cores all --config \
#     vcf_dir=../results/simulations/outputs/"$dempgraphy"/sim/trees \
#     output_path=../results/simulations/outputs/"$dempgraphy"/"$infer_method" \
#     infer_method="$infer_method" \
#     num_contigs=500 \
#     contig_length=5e6 \
#     n_sim_per_pop=20 \
#     max_time=10000 \
#     n_timepoints=2000 \
#     min_window_size=50 \
#     max_window_size=1000 \
#     n_sim_per_pop=10 \
#     N_min=100 \
#     N_max=100000 \
#     min_time=100 \
#     num_posterior_samples=5 \
#     thinning_interval=10 \
#     burnin=500 \
#     num_threads=12

### 3. singer
infer_method=singer
snakemake infer_STT --cores all --config \
    vcf_dir=../results/simulations/outputs/"$dempgraphy"/sim/trees \
    output_path=../results/simulations/outputs/"$dempgraphy"/"$infer_method" \
    infer_method="$infer_method" \
    num_contigs=2500 \
    contig_length=1e6 \
    n_sim_per_pop=20 \
    max_time=10000 \
    n_timepoints=2000 \
    min_window_size=50 \
    max_window_size=1000 \
    n_sim_per_pop=10 \
    N_min=100 \
    N_max=100000 \
    min_time=100 \
    num_posterior_samples=50 \
    thinning_interval=20 \
    burnin=0 \
    num_threads=12


# # Empirical Data Examples
