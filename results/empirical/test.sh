#!/bin/bash
snakemake --config \
    trees_dir=../../data/trees/wohns_chopped \
    output_dir=wohns/test \
    pops=YRI,CEU \
    max_time=10000 \
    n_samples=5 \
    pop_colors=./wohns_pop_colors.csv \
    species=humans \
    fraction=1 \
    --cores all
    