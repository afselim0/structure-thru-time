#!/bin/bash
#SBATCH --job-name=Chimps_deep
#SBATCH --account=pi-jnovembre
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /software/bin/miniconda3/etc/profile.d/conda.sh
conda activate /project/jnovembre/afselim/STT


# full_ts_dir=../../../data/external_private/chimp_data/scratch_isilon/groups/compgen/nhermosilla/PROJECTS/Structure_thru_time/pantro_data/SINGER/parallel_SINGER/pan_pbl
chopped_ts_dir=/project/jnovembre/afselim/structure-thru-time/data/trees/DeManuel/all_chopped
snakemake STT --config \
    output_path=../results/empirical/DeManuel/all_chroms_all_samples_deep_time \
    trees_dir="$chopped_ts_dir" \
    pop_colors=../data/metadata/DeManuel_pop_colors.csv \
    pop_labels=../data/metadata/DeManuel_pop_labels.csv \
    num_posterior_samples=100 \
    min_time=1 \
    max_time=100000 \
    n_timepoints=5000 \
    min_window_size=500 \
    max_window_size=5000 \
    N_min=1 \
    N_max=1000000000 \
    num_threads=36 \
    --cores all
