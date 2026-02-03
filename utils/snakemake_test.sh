#!/bin/bash
main_demes_dir=../data/demes/main
stdpopsim_demes_dir=../data/demes/stdpopsim

snakemake sim --cores all --config \
    demes_file_path="$main_demes_dir"/split.yaml \
    output_path=../results/simulations/outputs/split \
    num_contigs=500 \
    contig_length=1e6 \
    n_sim_per_pop=20 \
    max_time=4000 \
    num_threads=8