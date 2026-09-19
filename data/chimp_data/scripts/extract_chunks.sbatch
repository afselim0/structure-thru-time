#!/bin/bash
# STEP 9 as a SLURM job, because the login node cannot hold it.
#
#   sbatch scripts/extract_chunks.sbatch
#
# drivers/9_extract_chunks.sh claims this "runs in minutes, on the login node". That is true
# of a small dataset and false of this one: extract_ts_chunks.py decompresses a WHOLE
# chromosome tree sequence before slicing it at chunk boundaries, and chr1 (224 Mb, 112
# chunks) does not fit in the ~8 GiB a midway3 login node allows. Measured: the process was
# OOM-killed having written zero files.
#
# 120 GB matches what merge_chunks needed for the same reason -- both hold a chromosome-scale
# tree sequence in memory. Single-threaded, so one CPU; the work is sequential per chromosome.
#SBATCH --job-name=chimp_extract
#SBATCH --account=pi-jnovembre
#SBATCH --partition=caslake
#SBATCH --qos=caslake
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=logs/chimp_extract-%j.out
set -euo pipefail
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"
cd "$SLURM_SUBMIT_DIR"
echo "START extract $(date -Is) on $(hostname)"
bash drivers/9_extract_chunks.sh "$@"
echo "END extract $(date -Is)"
