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
# NUM_CONTIGS and ROOT can be overridden in the environment.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

NUM_CONTIGS="${NUM_CONTIGS:-100}"
ROOT="${ROOT:-../results/outputs/misc_analyses/arg_rates_comparison}"

# stage -> resources. SINGER is the long pole (~9 min/contig on split, ~18 min
# on sawtooth at 20 diploids/pop), so it gets the widest allocation.
stage_resources() {
    case "$1" in
        sim)     coastt_resources cpus=24 mem_per_cpu=4G time=04:00:00 ;;
        tsinfer) coastt_resources cpus=24 mem_per_cpu=4G time=06:00:00 ;;
        relate)  coastt_resources cpus=24 mem_per_cpu=4G time=04:00:00 ;;
        # measured: 29 min split, ~70 min sawtooth, 22 GB peak
        singer)  coastt_resources cpus=48 mem_per_cpu=2G time=03:00:00 ;;
    esac
}

# ------------------------------------------------------------------ local ----
if [[ $COASTT_SLURM -eq 0 ]]; then
    cd "$COASTT_UTILS"
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    cpus="$(nproc 2>/dev/null || echo 12)"

    for demo in split sawtooth; do
        for stage in sim tsinfer relate singer; do
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
        python "$COASTT_UTILS/misc_utils/arg_rates_comparison.py" \
            --demography "$demo" --stage "$stage" \
            --num_contigs "$NUM_CONTIGS" --num_threads "$COASTT_CPUS" --root "$ROOT"
}

for demo in split sawtooth; do
    sim_jid=$(submit_stage "$demo" sim)
    echo "submitted $demo sim: job $sim_jid" >&2
    for stage in tsinfer relate singer; do
        jid=$(submit_stage "$demo" "$stage" "$sim_jid")
        echo "submitted $demo $stage: job $jid (afterok:$sim_jid)" >&2
    done
done
