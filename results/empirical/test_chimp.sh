#!/bin/bash
snakemake --config \
    dataset=DeManuel \
    trees_dir=../../data/trees/DeManuel/chr21_chopped \
    min_time=100 \
    max_time=100000 \
    n_timepoints=20000 \
    num_posterior_samples=100 \
    window_size=1000 \
    N0=1000 \
    n_samples=10 \
    --cores all