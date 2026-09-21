#!/bin/bash
# Submit the three chimp CoaSTT runs as SLURM jobs, chained on the shared pdf.
#
#   1. chimps.yaml                 rates over [40, 4000]        (builds the pdf)
#   2. chimps_deep.yaml            rates over [4000, 400000]
#   3. chimps_per_haplotype.yaml   rates over [400, 40000], per-haplotype
#
# All three declare the SAME pdf grid (pdf_min_time 0, pdf_max_time 450000,
# pdf_bin_size 20) and the same output_path, so they share one
# coal_pdf.V1.FS1.0.pdf_minT0.pdf_maxT450000.pdf_bin20.parquet. That file is the
# expensive part of the run -- counting coalescent spans over the whole contig set
# -- and computing rates from it is cheap, which is the point of separating the pdf
# grid from the rate interval. So job 1 runs alone and builds it; 2 and 3 are held
# on `afterok:1` and then run concurrently, each doing only its own rates,
# embeddings and plots.
#
# Their outputs do not collide: the rate interval and the window sizes are in
# RATES_TAG, and per_haplotype adds `.perHap`, so each config writes its own
# rates/weights/embeddings/plots beside the one shared pdf. Checked against the
# resolved DAGs, jobs 2 and 3 have exactly four output files in common --
# coal_pdf.*.parquet, pca.V1.csv, pop_labels.csv, pop_colors.csv -- and all four
# are genuinely independent of both the rate interval and per_haplotype. Job 1
# creates them, so by the time 2 and 3 start, snakemake sees them present and skips
# those rules; nothing is written twice. (This is why 2 and 3 must be held on 1
# rather than all three going at once: without it, all three would build the pdf.)
#
# Usage:
#   sbatch utils/runs/local_runs/app_runs/chimps_slurm.sh     # submit the chain
#   bash   utils/runs/local_runs/app_runs/chimps_slurm.sh -n  # print the plan only
#
# Despite the #SBATCH block below (kept so `sbatch <this file>` works as it did
# before), this is a SUBMITTER: it runs in seconds and submits the three real jobs.
# generate_app_results_slurm.sh excludes *_slurm.sh from its array for that reason.
#SBATCH --job-name=coastt_chimps_submit
#SBATCH --account=pi-jnovembre
#SBATCH --partition=jnovembre
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=00:10:00
#SBATCH --output=/project/jnovembre/afselim/structure-thru-time/utils/runs/job_logs/%x_%j.out
#SBATCH --error=/project/jnovembre/afselim/structure-thru-time/utils/runs/job_logs/%x_%j.err
set -euo pipefail

DRY_RUN=0
[[ "${1:-}" == "-n" ]] && DRY_RUN=1

REPO=/project/jnovembre/afselim/structure-thru-time
CONDA_ENV="${COASTT_CONDA_ENV:-/project/jnovembre/afselim/CoaSTT_env}"
ACCOUNT=pi-jnovembre
PARTITION=jnovembre
LOG_DIR="$REPO/utils/runs/job_logs"

# Each job gets its own snakemake working directory. Jobs 2 and 3 run at the same
# time against one output_path, and concurrent snakemakes sharing a working
# directory race on .snakemake/iocache -- drop_iocache() has each deleting the
# other's latest.pkl -- which kills a job within seconds of starting, during DAG
# construction, with a FileNotFoundError that looks nothing like a workflow error.
# --nolock does not help: it suppresses the directory lock, not the shared
# .snakemake state, and there is no flag to relocate that (it is derived from the
# working directory). The fix is a per-job workdir of symlinks back into the real
# tree: writes still land in the repo, but .snakemake/ is private.
#
# These live outside the repo, as the SINGER runs' workdirs do -- the state is
# large, regenerable, and nothing tracks it. $SCRATCH on midway3 is the shared
# /scratch/midway3, so the path includes $USER.
WORKDIR_ROOT="${WORKDIR_ROOT:-${SCRATCH:-/tmp}/${USER}/coastt_chimps_work}"

# cpus  mem-per-cpu  walltime, per config.
#
# Job 1 carries the pdf: compute_pdf counts coalescent spans over the full contig
# set into per-contig (pairs x 22,501 timepoints) arrays, so it needs the wide
# allocation and the long clock. The counting is resumable -- an interrupted run
# validates and reuses the .npy files already written -- so a timeout costs only
# the contigs in flight. Jobs 2 and 3 read that pdf and only fit embeddings, which
# is bounded by n_timepoints_emb rather than by the contig count.
resources() {
    case "$1" in
        chimps)               echo "16 8G 36:00:00" ;;
        chimps_deep)          echo "16 6G 12:00:00" ;;   # nTe 2000, the largest embedding
        chimps_per_haplotype) echo "16 6G 12:00:00" ;;   # nTe 1000, but 2x the pairs
    esac
}

mkdir -p "$LOG_DIR"

submit() {
    local cfg="$1" dep="${2:-}"
    read -r cpus mem time <<< "$(resources "$cfg")"

    local depflag=()
    [[ -n "$dep" ]] && depflag=(--dependency="afterok:$dep")

    local args=(
        --account="$ACCOUNT" --partition="$PARTITION"
        --job-name="coastt_$cfg"
        --nodes=1 --ntasks=1
        --cpus-per-task="$cpus" --mem-per-cpu="$mem" --time="$time"
        --output="$LOG_DIR/coastt_${cfg}_%j.out"
        --error="$LOG_DIR/coastt_${cfg}_%j.err"
        "${depflag[@]}"
    )

    # snakemake's highspy ships a libstdc++ that shadows the env's, and numpy then
    # fails with a GLIBCXX error; the env's lib has to come first. The rules invoke
    # a bare `python`, so the env also has to be on PATH, not just imported from.
    local payload
    payload=$(cat <<EOS
#!/bin/bash
set -euo pipefail
export PATH="$CONDA_ENV/bin:\$PATH"
export CONDA_PREFIX="$CONDA_ENV"
export LD_LIBRARY_PATH="$CONDA_ENV/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"

echo "host \$(hostname), \${SLURM_CPUS_PER_TASK:-?} cpus, started \$(date)"

# A private workdir mirroring utils/, with the repo's real directories symlinked in
# so the snakefile's relative paths (../src, ../results, ../data, snakemake_utils)
# all resolve. Rebuilt each run: it holds only links plus .snakemake.
work="$WORKDIR_ROOT/$cfg"
rm -rf "\$work"
mkdir -p "\$work/utils"
for item in src data results; do
    ln -sfn "$REPO/\$item" "\$work/\$item"
done
for item in snakemake_utils misc_utils arg_infer_utils data_utils runs snakefile; do
    ln -sfn "$REPO/utils/\$item" "\$work/utils/\$item"
done
cd "\$work/utils"

# Match compute_pdf's workers to the CPUs actually allocated rather than the
# config's own num_threads, which is written for a different machine.
snakemake STT \
    --configfile "$REPO/data/configs/app_configs/$cfg.yaml" \
    --config num_threads=\${SLURM_CPUS_PER_TASK:-$cpus} \
    --cores all --rerun-incomplete --nolock

echo "finished \$(date)"
EOS
)

    if [[ $DRY_RUN -eq 1 ]]; then
        echo "[dry] sbatch --parsable ${args[*]}  <<< '<$cfg payload>'" >&2
        echo "<${cfg}_jobid>"
        return 0
    fi

    sbatch --parsable "${args[@]}" <<< "$payload"
}

# 1. builds the shared pdf; 2 and 3 wait on it, then run side by side.
base_jid=$(submit chimps)
echo "submitted chimps (builds the shared pdf): job $base_jid" >&2

for cfg in chimps_deep chimps_per_haplotype; do
    jid=$(submit "$cfg" "$base_jid")
    echo "submitted $cfg: job $jid (afterok:$base_jid)" >&2
done

echo >&2
echo "Note: 2 and 3 start only if 1 succeeds. If 1 times out, its coalescent counts" >&2
echo "are resumable -- resubmit this script and compute_pdf picks up where it stopped," >&2
echo "or submit 2 and 3 without a dependency once the pdf parquet exists." >&2
