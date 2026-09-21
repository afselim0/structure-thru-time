#!/bin/bash
# CoaSTT run: the chimpanzee dataset, from the inferred tree sequences.
#
#     bash chimps.sh            # run chimps.yaml here
#     bash chimps.sh --slurm    # submit all three configs as a chained job set
#     bash chimps.sh --slurm -n # print the plan only
#
# Locally this runs the one config, chimps.yaml. Under --slurm it submits the
# three chimp configs, chained on the shared pdf:
#
#   1. chimps.yaml                 rates over [40, 4000]        (builds the pdf)
#   2. chimps_deep.yaml            rates over [4000, 400000]
#   3. chimps_per_haplotype.yaml   rates over [400, 40000], per-haplotype
#
# All three declare the SAME pdf grid (pdf_min_time 0, pdf_max_time 450000,
# pdf_bin_size 20) and the same output_path, so they share one
# coal_pdf.V1.FS1.0.pdf_minT0.pdf_maxT450000.pdf_bin20.parquet. That file is the
# expensive part of the run -- counting coalescent spans over 1,334 contigs
# (73 GB of .trees) into 1,431 pair x 10,000 timepoint arrays, ~115 MB per contig
# and ~153 GB of intermediates in counts_dir -- and computing rates from it is
# cheap, which is the point of separating the pdf grid from the rate interval.
# So job 1 runs alone and builds it; 2 and 3 are held on `afterok:1` and then
# run concurrently, each doing only its own rates, embeddings and plots.
#
# Their outputs do not collide: the rate interval and the window sizes are in
# RATES_TAG, and per_haplotype adds `.perHap`, so each config writes its own
# rates/weights/embeddings/plots beside the one shared pdf. Checked against the
# resolved DAGs, jobs 2 and 3 have exactly four output files in common --
# coal_pdf.*.parquet, pca.V1.csv, pop_labels.csv, pop_colors.csv -- and all four
# are genuinely independent of both the rate interval and per_haplotype. Job 1
# creates them, so by the time 2 and 3 start, snakemake sees them present and
# skips those rules; nothing is written twice. (This is why 2 and 3 must be held
# on 1 rather than all three going at once: without it, all three would build
# the pdf.)
#
# The counting is resumable -- an interrupted run validates and reuses the .npy
# files already written -- so a timeout costs only the contigs in flight, but
# the first pass needs the long wall clock.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

# Job 1 carries the pdf, so it gets the wide allocation and the long clock. Jobs
# 2 and 3 read that pdf and only fit embeddings, which is bounded by
# n_timepoints_emb rather than by the contig count.
coastt_resources cpus=16 mem_per_cpu=8G time=36:00:00

if [[ $COASTT_SLURM -eq 0 ]]; then
    coastt_run STT --configfile ../data/configs/app_configs/chimps.yaml
    exit $?
fi

coastt_load_slurm_config

# Submit one config, optionally held on another job.
submit_chimps() {
    local cfg="$1" dep="${2:-}"
    local depflag=()
    [[ -n "$dep" ]] && depflag=(--dependency="afterok:$dep")

    COASTT_JOB_NAME="coastt_$cfg" \
    coastt_sbatch "coastt_$cfg" "${depflag[@]}" -- \
        snakemake STT \
            --configfile "$COASTT_REPO/data/configs/app_configs/$cfg.yaml" \
            --config "num_threads=$COASTT_CPUS" \
            --cores all --rerun-incomplete --nolock
}

base_jid=$(submit_chimps chimps)
echo "submitted chimps (builds the shared pdf): job $base_jid" >&2

# nTe 2000 is the largest embedding; per_haplotype is nTe 1000 but twice the
# pairs. Both are embedding-bound, so they share one lighter profile.
coastt_resources cpus=16 mem_per_cpu=6G time=12:00:00
for cfg in chimps_deep chimps_per_haplotype; do
    jid=$(submit_chimps "$cfg" "$base_jid")
    echo "submitted $cfg: job $jid (afterok:$base_jid)" >&2
done

echo >&2
echo "Note: 2 and 3 start only if 1 succeeds. If 1 times out, its coalescent counts" >&2
echo "are resumable -- rerun this script and compute_pdf picks up where it stopped." >&2
