#!/bin/bash
snakemake --config \
    dataset=DeManuel \
    trees_dir=../../data/trees/DeManuel/chr21_chopped \
    min_time=100 \
    max_time=100000 \
    n_timepoints=20000 \
    num_posterior_samples=100 \
    minimum_window_size=50 \
    maximum_window_size=1500 \
    N0=1000 \
    --cores all