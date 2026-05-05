#!/bin/bash
#SBATCH --job-name=Chimps_deep
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre
#SBATCH --cpus-per-task=40
#SBATCH --mem-per-cpu=2G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /software/bin/miniconda3/etc/profile.d/conda.sh
conda activate /project/jnovembre/afselim/STT


# full_ts_dir=../../../data/external_private/chimp_data/scratch_isilon/groups/compgen/nhermosilla/PROJECTS/Structure_thru_time/pantro_data/SINGER/parallel_SINGER/pan_pbl
# chopped_ts_dir = /project/jnovembre/afselim/structure-thru-time/data/trees/DeManuel/all_chopped
# snakemake STT --configfile /project/jnovembre/afselim/structure-thru-time/data/empirical_runs/chimps_all.yaml --cores all
snakemake STT --configfile /project/jnovembre/afselim/structure-thru-time/data/empirical_runs/chimps_all_10000.yaml --cores all
