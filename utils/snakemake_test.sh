#!/bin/bash
main_demes_dir=../data/demes/main
stdpopsim_demes_dir=../data/demes/stdpopsim
infer_method=tsinfer

snakemake sim_STT --cores all --config \
    demes_file_path="$main_demes_dir"/zigzag.yaml \
    output_path=../results/simulations/outputs/split/sim \
    num_contigs=500 \
    contig_length=5e6 \
    n_sim_per_pop=20 \
    max_time=40000 \
    n_timepoints=10000 \
    min_window_size=200 \
    max_window_size=1000 \
    n_sim_per_pop=10 \
    N_min=1000 \
    # N_max=10000000 \
    num_threads=12

# snakemake infer_STT --cores all --config \
#     vcf_dir=../results/simulations/outputs/split/sim/trees \
#     output_path=../results/simulations/outputs/split/"$infer_method" \
#     infer_method="$infer_method" \
#     contig_length=5e6 \
#     num_posterior_samples=5 \
#     thinning_interval=10 \
#     burnin=500 \
#     num_threads=1
    
