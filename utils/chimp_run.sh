#!/bin/bash
conda activate /project/jnovembre/afselim/STT
full_ts_dir=../../../data/external_private/chimp_data/scratch_isilon/groups/compgen/nhermosilla/PROJECTS/Structure_thru_time/pantro_data/SINGER/parallel_SINGER/pan_pbl
snakemake STT --config \
    output_path=../results/empirical/DeManuel/all_chroms_all_samples_1 \
    trees_dir="$full_ts_dir" \
    pop_colors=../data/metadata/DeManuel_pop_colors.csv \
    pop_labels=../data/metadata/DeManuel_pop_labels.csv \
    num_posterior_samples=100 \
    min_time=1 \
    max_time=1000 \
    n_timepoints=1000 \
    min_window_size=500 \
    max_window_size=500 \
    N_min=1 \
    N_max=100000000 \
    num_threads=16 \
    --cores all
