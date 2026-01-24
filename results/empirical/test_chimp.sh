#!/bin/bash
snakemake combs --config \
    dataset=DeManuel \
    trees_dir=../../data/trees/DeManuel/chr21 \
    max_time=100000 \
    n_timepoints=1000 \
    num_posterior_samples=100 \
    --cores all \

    