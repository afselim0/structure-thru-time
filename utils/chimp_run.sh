#!/bin/bash
full_ts_dir=../../../data/external_private/chimp_data/scratch_isilon/groups/compgen/nhermosilla/PROJECTS/Structure_thru_time/pantro_data/SINGER/parallel_SINGER/pan_pbl
n=2
snakemake STT --config \
    output_path=../results/empirical/DeManuel/"$pops" \
    trees_dir="$full_ts_dir" \
    pop_colors=../data/metadata/DeManuel_pop_colors.csv \
    pop_labels=../data/metadata/DeManuel_pop_labels.csv \
    n_samples_per_pop=$n \
    num_posterior_samples=100 \
    min_time=1 \
    max_time=2000 \
    n_timepoints=2000 \
    min_window_size=100 \
    max_window_size=1000 \
    N_min=1 \
    N_max=1000000000000000000 \
    num_threads=1 \
    --cores all \