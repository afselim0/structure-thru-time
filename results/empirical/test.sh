#!/bin/bash
snakemake --config \
    trees_dir=../../data/trees/wohns_chopped \
    pops=YRI,CEU,CHB,JPT \
    n_samples=100 \
    dataset=Wohns \
    min_time=1 \
    N0=1000 \
    n_timepoints=2000 \
    max_window_size=1000 \
    fraction=0.1 \
    --cores all \

# snakemake --config \
#     trees_dir=../../data/trees/wohns_chopped \
#     n_samples=2 \
#     dataset=Wohns \
#     output_dir=Wohns/all_1 \
#     --cores all \
