#!/bin/bash
#SBATCH --job-name=SS_demes
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=4G
#SBATCH --time=06:00:00
#SBATCH --output=/project/jnovembre/afselim/structure-thru-time/job_logs//%x_%A_%a.out
#SBATCH --error=/project/jnovembre/afselim/structure-thru-time/job_logs//%x_%A_%a.err

CONFIG_FILES_PATH=/project/jnovembre/afselim/structure-thru-time/data/demes/SS_demes_config
SCRIPT=$(realpath "$0")

# Entry point: submit p2 array only
if [[ -z "$SLURM_ARRAY_TASK_ID" ]]; then
    mapfile -t P2_FILES < <(ls "${CONFIG_FILES_PATH}"/*_p2.yaml | sort)
    N_P2=${#P2_FILES[@]}

    P2_JOB=$(sbatch --array=1-${N_P2} "$SCRIPT" | awk '{print $4}')
    echo "Submitted p2 array: job ${P2_JOB} (${N_P2} tasks)"
    exit 0
fi

# Worker: run the config assigned to this task
source /software/python-anaconda-2022.05-el8-x86_64/etc/profile.d/conda.sh
conda activate /project/jnovembre/afselim/STT

mapfile -t CONFIG_FILES < <(ls "${CONFIG_FILES_PATH}"/*_p2.yaml | sort)
CONFIG=${CONFIG_FILES[$SLURM_ARRAY_TASK_ID - 1]}

cd /project/jnovembre/afselim/structure-thru-time/utils

snakemake STT_sim --configfile "${CONFIG}" --cores all --rerun-incomplete --nolock
