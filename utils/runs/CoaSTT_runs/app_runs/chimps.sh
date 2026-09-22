#!/bin/bash
# CoaSTT run: the chimpanzee dataset, from the inferred tree sequences.
#
#     bash chimps.sh            # run the four in order, here
#     bash chimps.sh --slurm    # submit them as a chained job set
#     bash chimps.sh --slurm -n # print the plan only
#
# Four runs, in this order because they share the pdf:
#
#   1. chimps_recent.yaml          rates over [50, 4000]        (builds the pdf)
#   2. chimps_deep.yaml            rates over [4000, 400000]
#   3. chimps_per_haplotype.yaml   rates over [400, 400000], per-haplotype
#   4. chimps_recent.yaml          STT_replicates, 8 seeds
#
# 4 reruns the recent config under the STT_replicates target, which reuses that
# config's rates and re-solves the embeddings once per seed in `seeds`. Its
# outputs are the embeddings_seed<n>.* parquets and the four replicates_* panels;
# it writes no rates or pdf of its own, so it holds on 1 like 2 and 3 and then
# runs alongside them. Without a `seeds` list the target collapses to rates+pdf
# and produces nothing, which is why chimps_recent.yaml carries one.
#
# All four declare the SAME pdf grid (pdf_min_time 0, pdf_max_time 450000,
# pdf_bin_size 20) and the same output_path, so they share one
# coal_pdf.V1.FS1.0.pdf_minT0.pdf_maxT450000.pdf_bin20.parquet. That file is the
# expensive part of the run -- counting coalescent spans over 1,334 contigs
# (73 GB of .trees) into 5,886 pair x 10,000 timepoint arrays, holding ~1.5 TB of
# intermediates in counts_dir until the pdf is written -- while computing rates
# from that pdf is cheap, which is the point of separating the pdf grid from the
# rate interval.
# So job 1 runs alone and builds it; 2 and 3 are held on `afterok:1` and then
# run concurrently, each doing only its own rates, embeddings and plots.
#
# Their outputs do not collide: the rate interval and the window sizes are in
# RATES_TAG, and per_haplotype adds `.perHap`, so each config writes its own
# rates/weights/embeddings/plots beside the one shared pdf. Jobs 2 and 3 have
# only four output files in common -- coal_pdf.*.parquet, pca.V1.csv,
# pop_labels.csv, pop_colors.csv -- and all four are independent of both the
# rate interval and per_haplotype. Job 1 creates them, so by the time 2 and 3
# start, snakemake sees them present and skips those rules; nothing is written
# twice. (This is why 2 and 3 must be held on 1 rather than all three going at
# once: without it, all three would build the pdf.)
#
# The counting is resumable -- an interrupted run validates and reuses the .npy
# files already written -- so a timeout costs only the contigs in flight, but
# the first pass needs the long wall clock.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

BASE_CFG=chimps_recent
DEPENDENT_CFGS=(chimps_deep chimps_per_haplotype)

if [[ $COASTT_SLURM -eq 0 ]]; then
    for cfg in "$BASE_CFG" "${DEPENDENT_CFGS[@]}"; do
        echo "=== $cfg ===" >&2
        coastt_run STT --configfile "../data/configs/app_configs/$cfg.yaml"
    done
    echo "=== $BASE_CFG (replicates) ===" >&2
    coastt_run STT_replicates --configfile "../data/configs/app_configs/$BASE_CFG.yaml"
    exit $?
fi

coastt_load_slurm_config

# Submit one config, optionally held on another job. The job name carries the
# target as well as the config, since the replicates job reuses chimps_recent.
submit_chimps() {
    local cfg="$1" dep="${2:-}" target="${3:-STT}" name="${4:-coastt_$1}"
    local depflag=()
    [[ -n "$dep" ]] && depflag=(--dependency="afterok:$dep")

    COASTT_JOB_NAME="$name" \
    coastt_sbatch "$name" "${depflag[@]}" -- \
        snakemake "$target" \
            --configfile "$COASTT_REPO/data/configs/app_configs/$cfg.yaml" \
            --config "num_threads=$COASTT_CPUS" \
            --cores all --rerun-incomplete --nolock
}

# Job 1 carries the pdf, so it gets the wide allocation and the long clock.
coastt_resources cpus=16 mem_per_cpu=8G time=36:00:00
base_jid=$(submit_chimps "$BASE_CFG")
echo "submitted $BASE_CFG (builds the shared pdf): job $base_jid" >&2

# 2 and 3 read that pdf and only fit embeddings, which is bounded by
# n_timepoints_emb rather than by the contig count. Both are nTe 2000, and
# per_haplotype has twice the pairs, so they share one lighter profile.
coastt_resources cpus=16 mem_per_cpu=6G time=12:00:00
for cfg in "${DEPENDENT_CFGS[@]}"; do
    jid=$(submit_chimps "$cfg" "$base_jid")
    echo "submitted $cfg: job $jid (afterok:$base_jid)" >&2
done

# Job 4: replicate optimizations on the recent config's rates. Eight seeds on
# the nTe198 grid, run one after another inside the one job, so it gets the
# longer clock the three-job profile above does not need.
coastt_resources cpus=16 mem_per_cpu=6G time=24:00:00
rep_jid=$(submit_chimps "$BASE_CFG" "$base_jid" STT_replicates "coastt_${BASE_CFG}_replicates")
echo "submitted $BASE_CFG replicates: job $rep_jid (afterok:$base_jid)" >&2

echo >&2
echo "Note: 2, 3 and 4 start only if 1 succeeds. If 1 times out, its coalescent counts" >&2
echo "are resumable -- rerun this script and compute_pdf picks up where it stopped." >&2
