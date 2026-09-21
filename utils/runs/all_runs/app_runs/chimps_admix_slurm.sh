#!/bin/bash
# Submit the admixed chimp CoaSTT run as a single SLURM job.
#
# Unlike chimps_slurm.sh (which chains three configs onto one shared pdf), this is a
# single job: chimps_admix.yaml has its own output_path (results/outputs/chimps/admix)
# and its own pdf grid (pdf_maxT200000 vs 450000), so it shares nothing with the
# chimps/no_admix runs and needs no dependency on them.
#
# Usage:  sbatch utils/runs/local_runs/app_runs/chimps_admix_slurm.sh
#SBATCH --job-name=coastt_chimps_admix
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=36:00:00
#SBATCH --output=/project/jnovembre/afselim/structure-thru-time/utils/runs/job_logs/%x_%j.out
#SBATCH --error=/project/jnovembre/afselim/structure-thru-time/utils/runs/job_logs/%x_%j.err
set -euo pipefail

REPO=/project/jnovembre/afselim/structure-thru-time
CONDA_ENV="${COASTT_CONDA_ENV:-/project/jnovembre/afselim/CoaSTT_env}"

# snakemake's highspy ships a libstdc++ that shadows the env's; numpy then fails with a
# GLIBCXX error unless the env's lib comes first. chimps_admix.sh sets LD_LIBRARY_PATH
# from CONDA_PREFIX, so that variable has to be exported too -- setting only
# LD_LIBRARY_PATH here is not enough, because the inner script overwrites it and
# CONDA_PREFIX would be empty. The rules invoke a bare `python`, so the env also has to
# be on PATH, not merely imported from.
export PATH="$CONDA_ENV/bin:$PATH"
export CONDA_PREFIX="$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_ENV/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

mkdir -p "$REPO/utils/runs/job_logs"
echo "host $(hostname), ${SLURM_CPUS_PER_TASK:-?} cpus, started $(date)"

# A private snakemake working directory, mirroring utils/ with the repo's real
# directories symlinked in so the snakefile's relative paths (../src, ../results,
# ../data, snakemake_utils) all resolve. Writes still land in the repo through the
# symlinks; only .snakemake/ is private.
#
# This matters because other CoaSTT jobs may be running from utils/ at the same time:
# concurrent snakemakes sharing a working directory race on .snakemake/iocache --
# drop_iocache() has each deleting the others' latest.pkl -- and a job dies within
# seconds of starting, during DAG construction, with a FileNotFoundError that looks
# nothing like a workflow error. --nolock does not help: it suppresses the directory
# lock, not the shared .snakemake state, and there is no flag to relocate that (it is
# derived from the working directory).
#
# Kept outside the repo, as the SINGER runs' workdirs are: the state is large,
# regenerable and untracked. $SCRATCH on midway3 is the shared /scratch/midway3, so
# the path includes $USER.
WORKDIR_ROOT="${WORKDIR_ROOT:-${SCRATCH:-/tmp}/${USER}/coastt_chimps_work}"
work="$WORKDIR_ROOT/chimps_admix"
rm -rf "$work"
mkdir -p "$work/utils"
for item in src data results; do
    ln -sfn "$REPO/$item" "$work/$item"
done
for item in snakemake_utils misc_utils arg_infer_utils data_utils runs snakefile; do
    ln -sfn "$REPO/utils/$item" "$work/utils/$item"
done
cd "$work/utils"

# Match the compute_pdf / GNN workers to the CPUs actually allocated rather than the
# config's own num_threads, which is written for a different machine.
snakemake STT \
    --configfile "$REPO/data/configs/app_configs/chimps_admix.yaml" \
    --config num_threads=${SLURM_CPUS_PER_TASK:-16} \
    --cores all --rerun-incomplete --nolock "$@"

echo "finished $(date)"
