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
dempgraphy=zigzag

# snakemake sim_STT --cores all --config \
#     demes_file_path="$main_demes_dir"/"$dempgraphy".yaml \
#     output_path=../results/simulations/outputs/"$dempgraphy"/sim \
#     num_contigs=500 \
#     contig_length=5e6 \
#     n_sim_per_pop=20 \
#     max_time=40000 \
#     n_timepoints=2000 \
#     min_window_size=50 \
#     max_window_size=100 \
#     n_sim_per_pop=10 \
#     N_min=100 \
#     N_max=100000 \
#     min_time=100 \
#     num_threads=12 \
    # l1=1e-3,1e-2 \
    # l2=1e-2
    
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
#     num_contigs=2500 \
#     contig_length=1e6 \
#     n_sim_per_pop=20 \
#     max_time=10000 \
#     n_timepoints=2000 \
#     min_window_size=20 \
#     max_window_size=200 \
#     n_sim_per_pop=10 \
#     N_min=100 \
#     N_max=100000 \
#     min_time=100 \
#     num_posterior_samples=5 \
#     thinning_interval=10 \
#     burnin=500 \
#     num_threads=12

# ### 3. singer
# infer_method=singer
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


# Empirical Data Examples
# snakemake STT --config \
#     output_path=../results/empirical/Wohns/YRI_CEU_npp.20 \
#     trees_dir=../data/trees/wohns_chopped \
#     n_samples_per_pop=20 \
#     pops=YRI,CEU \
#     pop_colors=../data/metadata/Wohns_pop_colors.csv \
#     pop_labels=../data/metadata/Wohns_pop_labels.csv \
#     min_time=1 \
#     max_time=4000 \
#     n_timepoints=4000 \
#     min_window_size=100 \
#     max_window_size=1500 \
#     N_min=1000 \
#     N_max=1000000 \
#     num_threads=8 \
#     --cores all \

# snakemake STT --config \
#     output_path=../results/empirical/Wohns/all.npp.2 \
#     trees_dir=../data/trees/wohns_chopped \
#     n_samples_per_pop=2 \
#     pop_colors=../data/metadata/Wohns_pop_colors.csv \
#     pop_labels=../data/metadata/Wohns_pop_labels.csv \
#     min_time=1 \
#     max_time=4000 \
#     n_timepoints=4000 \
#     min_window_size=100 \
#     max_window_size=1500 \
#     N_min=1000 \
#     N_max=1000000 \
#     num_threads=4 \
#     --cores all \

snakemake STT --config \
    output_path=../results/empirical/DeManuel/all \
    trees_dir=../data/trees/DeManuel/chr21_chopped \
    pop_colors=../data/metadata/DeManuel_pop_colors.csv \
    pop_labels=../data/metadata/DeManuel_pop_labels.csv \
    min_time=2000 \
    max_time=100000 \
    n_timepoints=2000 \
    min_window_size=100 \
    max_window_size=2000 \
    N_min=1000 \
    N_max=1000000 \
    l2=0.01 \
    num_threads=4 \
    --cores all \