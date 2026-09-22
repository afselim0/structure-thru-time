#!/bin/bash
# CoaSTT run: chimps_posteriors.yaml over every SINGER posterior draw.
#
#     bash chimps_posterior_samples.sh            # every draw, sequentially
#     bash chimps_posterior_samples.sh --slurm    # one job array, one task per draw
#     bash chimps_posterior_samples.sh --slurm -n # print the plan only
#
# The draws are the ts_chunks/draw_* directories, discovered at run time rather
# than listed here: a new draw is picked up by adding its directory. Each is a
# full independent posterior sample of the same 1,334 contigs, so the runs share
# nothing and can go in any order or all at once.
#
# Each task runs the unmodified chimps_posteriors.yaml with two keys overridden --
# trees_dir to that draw's tree sequences and output_path to its own directory
# under results/outputs/chimps/posterior_samples/<draw>. Everything else (the
# pdf grid, the rate interval, the embedding parameters) is the config's, so the
# draws are comparable and differ only in their input ARGs. Nothing is written
# to results/outputs/chimps/no_admix, which chimps.sh owns.
#
# Each task repeats the whole pipeline including its own compute_pdf, since a
# pdf is specific to the tree sequences it was counted from. Measured on one
# draw: counting runs at ~72 contigs/min, so a cold pass over the 1,334 contigs
# is ~20 min, and the embeddings that follow are small here (nTe198 -- the joint
# stage is quadratic in the timepoint count, so this config is nowhere near the
# nTe2000 runs). 4 h is a comfortable ceiling for the whole pipeline.
#
# While it counts, a task holds ~31 GB of per-contig .npy intermediates in
# counts_dir, released once the pdf parquet is written. chimps_posteriors.yaml
# caps pdf_max_time at 5000 (250 timepoints), which is what keeps that small --
# chimps_recent.yaml would be ~1.5 TB per draw on the same contigs.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

CONFIG=chimps_posteriors
TREES_ROOT="$COASTT_REPO/data/chimp_data/results/chimp/ts_chunks"
OUTPUT_ROOT="$COASTT_REPO/results/outputs/chimps/posterior_samples"

# Every draw directory that actually holds tree sequences, sorted numerically so
# draw_2 precedes draw_10.
DRAWS=()
for d in "$TREES_ROOT"/*/; do
    [[ -n "$(echo "$d"*.trees 2>/dev/null | grep -v '\*')" ]] || continue
    DRAWS+=("$(basename "$d")")
done
IFS=$'\n' DRAWS=($(printf '%s\n' "${DRAWS[@]}" | sort -V)) ; unset IFS

if [[ ${#DRAWS[@]} -eq 0 ]]; then
    echo "No draw directories with .trees files under $TREES_ROOT" >&2
    exit 1
fi
echo "${#DRAWS[@]} posterior draws: ${DRAWS[*]}" >&2

# 4G/cpu is the nodes' native ratio (257 GB / 64 cpus), so cpus alone sets both
# the workers and the memory: 20 x 4G = 80 GB. A task peaked at 57.5 GB early in
# compute_pdf, so 80 leaves real headroom while still fitting several tasks per
# node -- at 128 GB none fit at all and the array sat on (Resources).
coastt_resources cpus=20 mem_per_cpu=4G time=04:00:00

if [[ $COASTT_SLURM -eq 0 ]]; then
    for draw in "${DRAWS[@]}"; do
        echo "=== $draw ===" >&2
        coastt_run STT \
            --configfile "../data/configs/app_configs/$CONFIG.yaml" \
            --config "trees_dir=$TREES_ROOT/$draw" \
                     "output_path=$OUTPUT_ROOT/$draw"
    done
    exit $?
fi

coastt_load_slurm_config

# One array task per draw. The draw list is baked into the payload as a bash
# array and indexed by SLURM_ARRAY_TASK_ID, so the tasks are identical but for
# the index -- which is what lets this be one array rather than N submissions.
# run_common.sh gives each task its own snakemake workdir (the task id is in the
# path), so the tasks do not race on .snakemake state.
# %N caps how many tasks run at once. One at a time keeps the counts_dir peak to
# a single draw's worth regardless of how many draws there are; raise it once the
# allocation has room to spare.
MAX_CONCURRENT=1

coastt_sbatch "coastt_chimps_posterior" \
    --array="0-$((${#DRAWS[@]} - 1))%$MAX_CONCURRENT" -- \
    bash -c '
        draws=('"${DRAWS[*]}"')
        draw="${draws[$SLURM_ARRAY_TASK_ID]}"
        echo "draw $draw (task $SLURM_ARRAY_TASK_ID)"
        snakemake STT \
            --configfile '"$COASTT_REPO"'/data/configs/app_configs/'"$CONFIG"'.yaml \
            --config "trees_dir='"$TREES_ROOT"'/$draw" \
                     "output_path='"$OUTPUT_ROOT"'/$draw" \
                     "num_threads='"$COASTT_CPUS"'" \
            --cores all --rerun-incomplete --nolock
    '

echo >&2
echo "Note: while counting, each task holds ~31 GB in counts_dir, freed once its" >&2
echo "pdf is written. The array runs $MAX_CONCURRENT task(s) at a time (%N on --array)." >&2
