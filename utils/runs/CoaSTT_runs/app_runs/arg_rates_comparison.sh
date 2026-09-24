#!/bin/bash
# The ARG-inference rates comparison: how well CoaSTT's rate estimates survive
# each ARG inference method, against the analytic truth.
#
#     bash arg_rates_comparison.sh           # all 8 stages here, in sequence
#     bash arg_rates_comparison.sh --slurm   # as 8 chained SLURM jobs
#     bash arg_rates_comparison.sh --slurm -n
#
# Per demography: one `sim` stage (simulate + analytic true rates + estimate on
# the true trees) followed by three method stages (tsinfer, relate,
# singer+POLEGON), which need the simulated trees the sim stage writes. Under
# --slurm the three method jobs are held on that demography's sim job with
# afterok and then run concurrently; locally the stages just run in order.
#
# All eight stages run utils/misc_utils/arg_rates_comparison.py, which drives
# the ordinary snakemake pipeline.
#
# NUM_CONTIGS, ROOT and STAGES can be overridden in the environment. Leaving sim
# out of STAGES submits the method stages with no dependency, on existing sims:
#     STAGES=singer bash arg_rates_comparison.sh --slurm
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

NUM_CONTIGS="${NUM_CONTIGS:-500}"
ROOT="${ROOT:-../results/outputs/misc_analyses/arg_rates_comparison}"
STAGES="${STAGES:-sim tsinfer relate singer}"

# stage -> resources, sized for 500 contigs. SINGER is the long pole (~9 min/contig
# on split, ~18 min on sawtooth at 20 diploids/pop), so it gets the widest allocation.
stage_resources() {
    case "$1" in
        sim)     coastt_resources cpus=24 mem_per_cpu=4G time=12:00:00 ;;
        tsinfer) coastt_resources cpus=12 mem_per_cpu=4G time=48:00:00 ;;
        relate)  coastt_resources cpus=12 mem_per_cpu=4G time=48:00:00 ;;
        # measured at 100 contigs and 1000 iterations: 29 min split, ~70 min
        # sawtooth, 22 GB peak; now 5x the contigs and 2.5x the iterations
        singer)  coastt_resources cpus=48 mem_per_cpu=2G time=48:00:00 ;;
    esac
}

# ------------------------------------------------------------------ local ----
if [[ $COASTT_SLURM -eq 0 ]]; then
    cd "$COASTT_UTILS"
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    cpus="$(nproc 2>/dev/null || echo 12)"

    for demo in sawtooth split; do
        for stage in $STAGES; do
            echo "=== $demo / $stage ==="
            python misc_utils/arg_rates_comparison.py \
                --demography "$demo" --stage "$stage" \
                --num_contigs "$NUM_CONTIGS" --num_threads "$cpus" --root "$ROOT"
        done
    done
    exit 0
fi

# ------------------------------------------------------------------ slurm ----
coastt_load_slurm_config

submit_stage() {
    local demo="$1" stage="$2" dep="${3:-}"
    stage_resources "$stage"

    local depflag=()
    [[ -n "$dep" ]] && depflag=(--dependency="afterok:$dep")

    coastt_sbatch "arg_rates_${demo}_${stage}" "${depflag[@]}" -- \
        python misc_utils/arg_rates_comparison.py \
            --demography "$demo" --stage "$stage" \
            --num_contigs "$NUM_CONTIGS" --num_threads "$COASTT_CPUS" --root "$ROOT"
}

for demo in sawtooth split; do
    sim_jid=""
    if [[ " $STAGES " == *" sim "* ]]; then
        sim_jid=$(submit_stage "$demo" sim)
        echo "submitted $demo sim: job $sim_jid" >&2
    fi
    for stage in $STAGES; do
        [[ "$stage" == sim ]] && continue
        jid=$(submit_stage "$demo" "$stage" "$sim_jid")
        echo "submitted $demo $stage: job $jid${sim_jid:+ (afterok:$sim_jid)}" >&2
    done
done
