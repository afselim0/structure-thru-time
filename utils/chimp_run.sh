#!/bin/bash
# conda activate /project/jnovembre/afselim/STT
# full_ts_dir=../../../data/external_private/chimp_data/scratch_isilon/groups/compgen/nhermosilla/PROJECTS/Structure_thru_time/pantro_data/SINGER/parallel_SINGER/pan_pbl
# snakemake STT --config \
#     output_path=../results/empirical/DeManuel/all_chroms_all_samples_1 \
#     trees_dir="$full_ts_dir" \
#     pop_colors=../data/metadata/DeManuel_pop_colors.csv \
#     pop_labels=../data/metadata/DeManuel_pop_labels.csv \
#     num_posterior_samples=100 \
#     min_time=1 \
#     max_time=1000 \
#     n_timepoints=1000 \
#     min_window_size=500 \
#     max_window_size=500 \
#     N_min=1 \
#     N_max=100000000 \
#     num_threads=16 \
#     --cores all

chopped_ts_dir=/project/jnovembre/afselim/structure-thru-time/data/trees/DeManuel/all_chopped
snakemake STT_pop_combs --config \
    output_path=../results/empirical/DeManuel/all_chroms_all_samples_deep_time \
    trees_dir="$chopped_ts_dir" \
    pop_colors=../data/metadata/DeManuel_pop_colors.csv \
    pop_labels=../data/metadata/DeManuel_pop_labels.csv \
    num_posterior_samples=100 \
    min_time=250 \
    max_time=100000 \
    n_timepoints=5000 \
    min_window_size=500 \
    max_window_size=5000 \
    N_min=1 \
    N_max=1000000000 \
    embeddings_plot_lims=1000000 \
    num_threads=12 \
    --cores all
