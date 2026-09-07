#!/bin/bash
#SBATCH --job-name=4pop_replicates
#SBATCH --account=pi-jnovembre
#SBATCH --partition=caslake
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/midway3/mianniravn/conda/STT

cd /project2/jnovembre/mianniravn/stt-0526/structure-thru-time/utils

snakemake STT_replicates \
    --configfile=../data/empirical_runs/wohns_4pop_replicates.yaml \
    --cores all
