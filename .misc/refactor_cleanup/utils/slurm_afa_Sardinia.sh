#!/bin/bash
#SBATCH --job-name=afa_Sardinia
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre-hm
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
conda activate /scratch/midway3/mianniravn/conda/STT

snakemake STT --configfile /project2/jnovembre/mianniravn/stt-0526/structure-thru-time/data/empirical_runs/wohns_afa_Sardinia.yaml --cores all
