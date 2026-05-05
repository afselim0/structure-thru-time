#!/bin/bash
#SBATCH --job-name=Chop_trees
#SBATCH --account=pi-jnovembre
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

source /software/bin/miniconda3/etc/profile.d/conda.sh
conda activate /project/jnovembre/afselim/STT


python chop_chimp_trees.py