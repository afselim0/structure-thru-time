#!/bin/bash
#SBATCH --job-name=wohns_all_replicates
#SBATCH --account=pi-jnovembre
#SBATCH --partition=bigmem
#SBATCH --cpus-per-task=4
#SBATCH --mem=1000G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/midway3/mianniravn/conda/STT

cd /project2/jnovembre/mianniravn/stt-0526/structure-thru-time/utils

snakemake --unlock

snakemake STT_replicates \
    --configfile=../data/empirical_runs/wohns_all_replicates.yaml \
    --cores all
