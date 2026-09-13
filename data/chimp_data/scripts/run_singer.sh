#!/usr/bin/env bash
# Run singer-snakemake on the chimpanzee panel.
#
#   bash scripts/run_singer.sh                            # all 23 autosomes
#   bash scripts/run_singer.sh --chrom chr21              # one chromosome
#   bash scripts/run_singer.sh --config slurm_caslake     # shared partition (see below)
#   bash scripts/run_singer.sh --pilot --chrom chr21      # reduced MCMC, for smoke tests
#   bash scripts/run_singer.sh -n --chrom chr21           # dry run: show the plan only
#
# A real launch detaches and returns immediately; -n stays in the foreground, prints the job
# plan and submits nothing. Use it to check paths and see how much work is outstanding.
#
# ALL scientific parameters come from drivers/singer_config.yaml. This script overrides only
# the keys that identify a run -- input-dir, output-dir, chromosomes -- plus --pilot, via
# --config, which snakemake applies after --configfile.
#
# WHY THIS IS NOT AN SBATCH ARRAY. The first genome-wide attempt ran one array task per
# chromosome with a flat --mem, which forced every rule to share one memory number. That
# number was sized from run_singer's per-chunk cost and did not cover merge_chunks, whose
# peak scales with total chromosome size: six of 23 chromosomes were OOM-killed there after
# 12-16 h of completed inference. Here snakemake submits each RULE as its own SLURM job, so
# merge_chunks gets its own allocation -- see the note in drivers/slurm_jnovembre.config.
#
# Produces:
#   results/chimp/singer-snakemake_outputs/   the workflow's own output tree
#   results/chimp/ts_chunks/                  built afterwards by scripts/extract_chunks.sh
#
# Run state (snakemake's .snakemake directory) goes to a scratch workdir OUTSIDE the repo:
# it is large, regenerable, and must not be committed. Which chunks are finished is recorded
# by the `.singer.done` sentinels under results/, not by the workdir, so deleting a workdir
# loses no completed work and a restart picks up where the last one stopped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SINGER="${ROOT}/../singer-snakemake"
DRIVERS="${ROOT}/drivers"

export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:${PATH}"
# CoaSTT_env's libstdc++ must lead or numpy fails to import on midway3 compute nodes.
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

# Guard against a snakemake SLURM-executor bug: one NODE_FAIL makes it cache the literal
# string "None assigned" as a failed node, after which EVERY submission carries
# `--exclude=None assigned` and is rejected by SLURM. See scripts/slurm_executor_patch.py.
# Injected via sitecustomize so it applies to the driver without editing site-packages, which
# would not survive rebuilding the environment from environment.yml.
#
# NOT hypothetical here: the first genome-wide run hit 5 NODE_FAILs and then logged 94,384
# failed run_polegon submissions, every one "sbatch: error: Unable to open file assigned".
# POLEGON dating never ran, so merge_chunks could never start.
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"

CHROM=""
PILOT=0
DRYRUN=0
# Cluster config, by basename of a drivers/<name>.config file:
#   slurm_jnovembre = the dedicated partition (192 CPUs, 48 h QOS ceiling), no queueing.
#   slurm_caslake   = the shared partition (9,120 CPUs, 36 h ceiling), higher throughput but
#                     jobs queue behind other users.
CLUSTER_CONFIG="slurm_jnovembre"
# $SCRATCH on midway3 is the SHARED root (/scratch/midway3), not a per-user directory, so the
# user component has to be appended explicitly -- writing straight to $SCRATCH is a permission
# error. Falls back to /tmp when $SCRATCH is unset.
if [[ -n "${SCRATCH:-}" ]]; then
    WORKDIR_ROOT="${WORKDIR_ROOT:-${SCRATCH}/${USER}/singer_work}"
else
    WORKDIR_ROOT="${WORKDIR_ROOT:-/tmp/${USER}/singer_work}"
fi
NAME="chimp"

while [[ $# -gt 0 ]]; do
    case $1 in
        --chrom)   CHROM=$2; shift 2 ;;
        --config)  CLUSTER_CONFIG=$2; shift 2 ;;
        --pilot)   PILOT=1; shift ;;
        -n|--dry-run) DRYRUN=1; shift ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

# Accept either `--config slurm_caslake` or `--config drivers/slurm_caslake.config`.
CLUSTER_FILE="${DRIVERS}/${CLUSTER_CONFIG##*/}"
CLUSTER_FILE="${CLUSTER_FILE%.config}.config"
[[ -f "$CLUSTER_FILE" ]] || {
    echo "no such cluster config: ${CLUSTER_FILE}" >&2
    echo "available:" >&2; ls "${DRIVERS}"/*.config >&2
    exit 1
}

input="${ROOT}/SingerSnakemake_Input"
[[ -d "$input" ]] || { echo "no input directory at ${input} -- run drivers 1-7 first" >&2; exit 1; }

# snakemake's --profile takes a DIRECTORY containing a file literally named config.yaml, but a
# directory per cluster is a poor fit for a drivers/ folder meant to be read at a glance. Stage
# the chosen flat .config into a throwaway profile directory instead.
PROFILE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/singer_profile.XXXXXXXX")"
trap 'rm -rf "$PROFILE_DIR"' EXIT
cp "$CLUSTER_FILE" "${PROFILE_DIR}/config.yaml"

# The pilot exists to exercise the pipeline quickly, not to produce usable posteriors.
PILOT_OVERRIDES=()
SUFFIX=""
if [[ $PILOT -eq 1 ]]; then
    # Cut the NUMBER of samples but keep the shared thinning interval, so a pilot draw comes
    # from the same chain behaviour as a production one. Lowering thin as well would make the
    # pilot a different MCMC regime rather than a smaller version of the real run.
    PILOT_OVERRIDES=(singer-mcmc-samples=4 polegon-mcmc-samples=4 singer-mcmc-resumes=20)
    SUFFIX="_pilot"
fi

name="${NAME}${SUFFIX}"
out="${ROOT}/results/${name}/singer-snakemake_outputs"
work="${WORKDIR_ROOT}/${name}"
mkdir -p "$work" "${ROOT}/logs"

overrides=(input-dir="$input" output-dir="$out")
[[ -n "$CHROM" ]] && overrides+=(chromosomes="[\"${CHROM}\"]")
overrides+=("${PILOT_OVERRIDES[@]}")

# --dry-run stays in the FOREGROUND and submits nothing: the point is to see the plan.
if [[ $DRYRUN -eq 1 ]]; then
    echo "[${name}] dry run (workdir ${work}, cluster config ${CLUSTER_CONFIG})"
    snakemake \
        --snakefile "${SINGER}/workflow/Snakefile" \
        --configfile "${DRIVERS}/singer_config.yaml" \
        --config "${overrides[@]}" \
        --directory "$work" \
        --profile "$PROFILE_DIR" \
        -n
    echo
    echo "Dry run only -- nothing was submitted."
    exit 0
fi

log="${ROOT}/logs/${name}.snakemake.log"
echo "[${name}] launching (workdir ${work}, cluster config ${CLUSTER_CONFIG})"
# setsid + disown: without them the driver is a grandchild of the invoking shell and gets
# reaped when that shell exits, which silently kills a multi-day run.
setsid nohup snakemake \
    --snakefile "${SINGER}/workflow/Snakefile" \
    --configfile "${DRIVERS}/singer_config.yaml" \
    --config "${overrides[@]}" \
    --directory "$work" \
    --profile "$PROFILE_DIR" \
    >> "$log" 2>&1 < /dev/null &
disown

# The staged profile must outlive this script: the driver keeps running after it exits and
# snakemake re-reads the profile in each submitted job.
trap - EXIT

echo "  (cluster config staged at ${PROFILE_DIR}; remove it once the run finishes)"
echo
echo "Monitor with:"
echo "  tail -f ${log}"
echo "  squeue -u \$USER"
echo
echo "When it finishes, build the CoaSTT input directory:"
echo "  bash ${ROOT}/scripts/extract_chunks.sh"
