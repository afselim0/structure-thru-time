#!/bin/bash
# Submit the ARG-inference rates comparison as 8 SLURM jobs.
#
# Per demography: one `sim` job (simulate + analytic true rates + estimate on the
# true trees) followed by three method jobs (tsinfer, relate, singer+POLEGON),
# each held until that demography's sim job succeeds. All eight run
# utils/misc_utils/arg_rates_comparison.py, which drives the ordinary snakemake
# pipeline.
#
# Usage:
#   bash arg_rates_comparison_slurm.sh            # submit
#   bash arg_rates_comparison_slurm.sh -n         # print the sbatch commands only
#
# This is a submitter, not a run script: it must NOT be picked up as an array task
# by ../../generate_app_results_slurm.sh, which excludes *_slurm.sh for that reason.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
runs_dir="$script_dir/../.."          # utils/runs
utils_dir="$runs_dir/.."              # utils
repo_root="$utils_dir/.."

DRY_RUN=0
[[ "${1:-}" == "-n" ]] && DRY_RUN=1

ACCOUNT=pi-jnovembre
PARTITION=jnovembre
CONDA_ENV="${COASTT_CONDA_ENV:-/project/jnovembre/afselim/CoaSTT_env}"
NUM_CONTIGS="${NUM_CONTIGS:-100}"
ROOT="${ROOT:-../results/outputs/misc_analyses/arg_rates_comparison}"
LOG_DIR="$runs_dir/job_logs"
mkdir -p "$LOG_DIR"

# stage -> "cpus mem_flag time". SINGER is the long pole (~9 min/contig on split,
# ~18 min on sawtooth at 20 diploids/pop), so it gets the widest allocation.
resources() {
    case "$1" in
        sim)     echo "24 --mem-per-cpu=4G 04:00:00" ;;
        tsinfer) echo "24 --mem-per-cpu=4G 06:00:00" ;;
        relate)  echo "24 --mem-per-cpu=4G 04:00:00" ;;
        singer)  echo "48 --mem-per-cpu=2G 03:00:00" ;;   # measured: 29 min split, ~70 min sawtooth, 22 GB peak
    esac
}

submit() {
    local demo="$1" stage="$2" dep="${3:-}"
    read -r cpus mem time <<< "$(resources "$stage")"

    local depflag=()
    [[ -n "$dep" ]] && depflag=(--dependency="afterok:$dep")

    local args=(
        --account="$ACCOUNT" --partition="$PARTITION"
        --job-name="arg_rates_${demo}_${stage}"
        --cpus-per-task="$cpus" "$mem" --time="$time"
        --output="$LOG_DIR/arg_rates_${demo}_${stage}_%j.out"
        --error="$LOG_DIR/arg_rates_${demo}_${stage}_%j.err"
        "${depflag[@]}"
    )

    # The payload: activate the env (LD_LIBRARY_PATH first, or numpy fails under
    # snakemake in a batch job), then run the wrapper from utils/.
    local payload
    payload=$(cat <<EOS
#!/bin/bash
set -euo pipefail
source "\$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
export LD_LIBRARY_PATH="\$CONDA_PREFIX/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
cd "$(cd "$utils_dir" && pwd)"
python misc_utils/arg_rates_comparison.py --demography $demo --stage $stage --num_contigs $NUM_CONTIGS --num_threads "\${SLURM_CPUS_PER_TASK:-$cpus}" --root "$ROOT"
EOS
)

    if [[ $DRY_RUN -eq 1 ]]; then
        # the sbatch line goes to stderr so stdout stays the job id alone
        echo "[dry] sbatch --parsable ${args[*]} <<< '<$demo/$stage payload>'" >&2
        echo "<${demo}_${stage}_jobid>"
        return 0
    fi

    local jid
    jid=$(sbatch --parsable "${args[@]}" <<< "$payload")
    echo "$jid"
}

for demo in split sawtooth; do
    sim_jid=$(submit "$demo" sim)
    echo "submitted $demo sim: job $sim_jid" >&2
    for stage in tsinfer relate singer; do
        jid=$(submit "$demo" "$stage" "$sim_jid")
        echo "submitted $demo $stage: job $jid (afterok:$sim_jid)" >&2
    done
done
