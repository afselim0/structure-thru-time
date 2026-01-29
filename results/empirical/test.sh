#!/bin/bash
# snakemake --config \
#     trees_dir=../../data/trees/wohns_chopped \
#     pops=YRI,CEU,CHB,JPT \
#     n_samples=20 \
#     dataset=Wohns \
#     min_time=1 \
#     N0=1000 \
#     n_timepoints=2000 \
#     max_window_size=1500 \
#     num_threads=8 \
#     fraction=0.5 \
#     --cores all \

snakemake --config \
    trees_dir=../../data/trees/wohns_chopped \
    n_samples=2 \
    dataset=Wohns \
    min_time=1 \
    N0=10000 \
    max_time=4000 \
    min_window_size=100 \
    max_window_size=1500 \
    n_timepoints=4000 \
    output_dir=Wohns/all_1 \
    num_threads=8 \
    --cores all \
