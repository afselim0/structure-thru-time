#!/bin/bash
main_demes_dir=../data/demes/main
stdpopsim_demes_dir=../data/demes/stdpopsim
infer_method=tsinfer

snakemake sim_STT --cores all --config \
    demes_file_path="$main_demes_dir"/split.yaml \
    output_path=../results/simulations/outputs/zigzag/sim \
    num_contigs=500 \
    contig_length=5e6 \
    n_sim_per_pop=20 \
    max_time=10000 \
    n_timepoints=2000 \
    min_window_size=50 \
    max_window_size=1000 \
    n_sim_per_pop=10 \
    N_min=5000 \
    N_max=10000000000 \
    min_time=100 \
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
    
