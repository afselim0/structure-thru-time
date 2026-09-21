#!/bin/bash
# Extract ONE posterior draw into its own ts_chunks/draw_<rep>/ directory.
#
#   sbatch --partition=<p> --account=<a> scripts/extract_draw.sbatch \
#          <SRC> <OUT> <CONFIG> <REP> <EXTRACT_PY>
#
# Not run by hand -- scripts/extract_chunks.sh submits one of these per draw once it has been
# given --partition and --account. It resolves which replicates to take and passes each job an
# explicit index, so the jobs cannot disagree about which draws the set is made of.
#
# --partition, --account and --job-name come from the submitting script rather than from
# #SBATCH lines here, because they are the cluster-specific part; everything below is the
# resource shape of the work itself, which does not change between clusters.
#
# 120 GB because extract_ts_chunks.py decompresses a WHOLE chromosome tree sequence before
# slicing it at chunk boundaries -- the same reason merge_chunks needed it. That is per
# chromosome and per draw, and nothing is shared between draws, which is exactly why these run
# as separate jobs: peak memory is the same whether one job does eight draws or eight jobs do
# one each, so the fan-out is free. Single-threaded, so one CPU; chromosomes are sequential
# within a job.
#
# 4 h matches what the whole-set extraction took when it ran as a single job over all 23
# chromosomes for one draw.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:$PATH"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

SRC=$1
OUT=$2
CONFIG=$3
REP=$4
EXTRACT=$5

# Every path arrives as an argument, and none is derived from $0 or $BASH_SOURCE. SLURM copies
# the batch script into its own spool directory before running it, so inside the job those
# point at /var/spool/slurm/... and a ROOT computed from them lands nowhere near the repo.
# The submitting script knows the real paths; it passes them.
[[ -f "$EXTRACT" ]] || { echo "extractor not found at ${EXTRACT}" >&2; exit 1; }
[[ -d "$SRC" ]]     || { echo "singer output not found at ${SRC}" >&2; exit 1; }
[[ -f "$CONFIG" ]]  || { echo "config not found at ${CONFIG}" >&2; exit 1; }

echo "START draw ${REP} $(date -Is) on $(hostname)"

# --config is NOT optional. extract_ts_chunks.py lives under human_data/, and its default
# config path is relative to its OWN location -- so without this it would read the human
# burn-in floor and apply it to chimpanzee output.
#
# --draw, not --n-draws: this job owns exactly one replicate. It writes only
# <OUT>/draw_<REP>/, so the concurrent jobs never touch the same directory.
python "$EXTRACT" \
    --singer-output "$SRC" \
    --out-dir "$OUT" \
    --config "$CONFIG" \
    --draw "$REP"

echo "END draw ${REP} $(date -Is)"
