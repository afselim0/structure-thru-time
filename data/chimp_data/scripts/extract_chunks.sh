#!/usr/bin/env bash
# Cut the last N posterior draws of each chromosome into per-chunk tree sequences for CoaSTT.
#
#   bash scripts/extract_chunks.sh                                   # 8 draws, in this shell
#   bash scripts/extract_chunks.sh --partition caslake \
#                                  --account pi-jnovembre            # 8 draws, 8 SLURM jobs
#   bash scripts/extract_chunks.sh --n-draws 4 --partition caslake \
#                                  --account pi-jnovembre            # 4 draws, 4 SLURM jobs
#   bash scripts/extract_chunks.sh --name chimp_pilot                # another results/<name>
#   bash scripts/extract_chunks.sh --partition caslake --account pi-jnovembre -n   # dry run
#
# Writes results/<name>/ts_chunks/draw_<rep>/<chrom>_<start>_<end>.trees -- one directory per
# posterior draw, each FLAT, with every chromosome's chunks together and no per-chromosome
# subdirectories. That is what CountCoalEvents wants: it globs *.trees in ONE directory and
# turns each filename into a contig id (src/CountCoalEvents.py:181).
#
# WHY ONE DIRECTORY PER DRAW, and never one directory holding several. The draws are repeat
# measurements of the same genome, not extra genome. Pooling them would enter every region
# into the likelihood N times over, which does not average the posterior -- it multiplies the
# evidence and shrinks the confidence in whatever the chain happened to do. So each draw is a
# complete, self-contained CoaSTT input: point a run at one, and compare runs ACROSS draws to
# see how much of the embedding is posterior uncertainty rather than signal.
#
# WHY ONE SLURM JOB PER DRAW. Extraction decompresses a WHOLE chromosome tree sequence before
# slicing it, which is why this needs ~120 GB and cannot run on a login node (see
# extract_chunks.sbatch). That cost is per (chromosome, draw) and none of it is shared between
# draws, so N draws in one job is N times the wall clock for the same peak memory. Fanning out
# over N jobs buys the whole speed-up at no extra memory per job. They are independent --
# different input draws, different output directories, no shared state.
#
# SLURM is opt-in because --account and --partition are cluster-specific and this repo should
# still run on a machine that has no scheduler at all. Supply BOTH to fan out; supply neither
# and the draws are extracted sequentially in this shell.
#
# The chimpanzee panel is a single sample set, so `chimp` is the only name under results/;
# the human side has one per set because it has eight. The extraction script itself is
# shared -- ../human_data/scripts/extract_ts_chunks.py -- rather than duplicated, since
# nothing in it is human-specific.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:${PATH}"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

NAME="chimp"
N_DRAWS=8
PARTITION=""
ACCOUNT=""
DRYRUN=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --n-draws)   N_DRAWS=$2; shift 2 ;;
        --name)      NAME=$2; shift 2 ;;
        --partition) PARTITION=$2; shift 2 ;;
        --account)   ACCOUNT=$2; shift 2 ;;
        -n|--dry-run) DRYRUN=1; shift ;;
        -*) echo "unknown argument: $1" >&2; exit 1 ;;
        # A bare word is the results/<name> to build, kept for the old positional form.
        *)  NAME=$1; shift ;;
    esac
done

[[ "$N_DRAWS" =~ ^[0-9]+$ && "$N_DRAWS" -ge 1 ]] || {
    echo "--n-draws must be a positive integer, got '${N_DRAWS}'" >&2; exit 1; }

SRC="${ROOT}/results/${NAME}/singer-snakemake_outputs"
OUT="${ROOT}/results/${NAME}/ts_chunks"
CONFIG="${ROOT}/drivers/singer_config.yaml"
# Resolved to an absolute path HERE and passed down to the SLURM jobs. They cannot work it out
# for themselves: SLURM runs a batch script from its own spool directory, so $BASH_SOURCE
# inside the job points at /var/spool/slurm/... rather than at this repo.
EXTRACT="$(cd "${ROOT}/../human_data/scripts" && pwd)/extract_ts_chunks.py"
[[ -d "$SRC" ]] || { echo "no singer output at $SRC" >&2; exit 1; }
[[ -f "$EXTRACT" ]] || { echo "no extractor at $EXTRACT" >&2; exit 1; }

# --- gate: every chromosome must offer the SAME draws ----------------------------------
#
# Two separate things can go wrong with replicate numbering, and this gate covers the one
# extract_ts_chunks.py cannot see when the draws are split across jobs. Each job is handed one
# replicate index and checks only that IT exists, so nothing downstream would notice that one
# chromosome stops two draws short of the others.
#
# extract_ts_chunks.py takes the N highest replicates per chromosome, which is correct only if
# replicates accumulate in order. A crashed or OOM-killed merge_chunks breaks that: the rule is
# parameterised per replicate and the jobs run concurrently, so a kill removes whichever were
# in flight -- not necessarily the last ones. On the first genome-wide run chr8 came back
# missing replicates 1, 6 AND 15 while keeping 14, so it would have silently contributed a
# different, less burnt-in draw than every other chromosome.
#
# That is invisible downstream: the file is valid, the chunk count is right, and CoaSTT has
# no way to know one contig came from a different point in the chain. So refuse to build a
# mixed dataset, and say exactly which chromosome to re-merge.
declare -A TOP
MAXTOP=-1
for d in "${SRC}"/*/; do
    c=$(basename "$d")
    [[ -d "${d}trees" ]] || continue
    top=$(ls "${d}trees/${c}."*.tsz 2>/dev/null \
          | sed "s|.*/${c}\.\([0-9]*\)\.tsz|\1|" | sort -n | tail -1)
    [[ -n "$top" ]] || { echo "  $c: no .tsz replicates yet" >&2; continue; }
    TOP[$c]=$top
    (( top > MAXTOP )) && MAXTOP=$top
done
[[ ${#TOP[@]} -gt 0 ]] || { echo "no finished chromosomes under $SRC" >&2; exit 1; }

BEHIND=()
for c in "${!TOP[@]}"; do
    (( TOP[$c] < MAXTOP )) && BEHIND+=("$c (has ${TOP[$c]}, expected ${MAXTOP})")
done
if [[ ${#BEHIND[@]} -gt 0 ]]; then
    echo "REFUSING to build a mixed-replicate dataset." >&2
    echo "These chromosomes are missing replicate ${MAXTOP}:" >&2
    printf '  %s\n' "${BEHIND[@]}" >&2
    echo >&2
    echo "Re-run the merge step for them -- the SINGER output survives on disk, so this" >&2
    echo "costs minutes, not the hours of inference:" >&2
    echo "  bash scripts/run_singer.sh" >&2
    exit 1
fi
echo "All ${#TOP[@]} chromosomes at replicate ${MAXTOP}."

# The draws to extract: the N highest, oldest first. Derived HERE rather than inside each job
# so that every job is handed an explicit index and they cannot disagree about which draws the
# set is made of -- the same reason the gate above exists.
FIRST=$((MAXTOP - N_DRAWS + 1))
if (( FIRST < 0 )); then
    echo "REFUSING: asked for ${N_DRAWS} draws but the chain only reaches replicate ${MAXTOP}." >&2
    echo "Lower --n-draws to $((MAXTOP + 1)) or fewer." >&2
    exit 1
fi
DRAWS=($(seq "$FIRST" "$MAXTOP"))
echo "Extracting ${N_DRAWS} draw(s): ${DRAWS[*]} -> ${OUT}/draw_<rep>/"

# --- fan out over SLURM, one job per draw ----------------------------------------------
#
# Only when BOTH --partition and --account are given: they have no portable default, and
# guessing one produces a job that is rejected at submission or charged to the wrong account.
if [[ -n "$PARTITION" && -n "$ACCOUNT" ]]; then
    mkdir -p "${ROOT}/logs"
    if [[ $DRYRUN -eq 1 ]]; then
        for r in "${DRAWS[@]}"; do
            echo "would submit draw ${r}  (partition ${PARTITION}, account ${ACCOUNT})"
        done
        echo "dry run -- nothing submitted"
        exit 0
    fi
    jids=()
    for r in "${DRAWS[@]}"; do
        jid=$(sbatch --parsable \
            --partition="$PARTITION" \
            --account="$ACCOUNT" \
            --job-name="chimp_extract_draw${r}" \
            "${ROOT}/scripts/extract_draw.sbatch" "$SRC" "$OUT" "$CONFIG" "$r" "$EXTRACT" \
            | tail -1)
        jids+=("$jid")
        echo "submitted draw ${r} as ${jid}"
    done
    echo
    echo "${#jids[@]} jobs submitted. Watch them with:"
    echo "  squeue -j $(IFS=,; echo "${jids[*]}")"
    echo "  tail -f logs/chimp_extract_draw*-*.out"
    echo
    echo "Each writes ${OUT}/draw_<rep>/. When they have all finished, reclaim disk with"
    echo "scripts/prune_chunk_draws.py --keep-from ${FIRST} (dry run by default)."
    exit 0
fi

if [[ -n "$PARTITION" || -n "$ACCOUNT" ]]; then
    echo "NOTE: --partition and --account are only honoured together; got only" \
         "$([[ -n "$PARTITION" ]] && echo --partition || echo --account)." >&2
    echo "      Running the draws sequentially in this shell instead." >&2
fi

# --- otherwise: sequential, in this shell -----------------------------------------------
#
# --config is NOT optional here. extract_ts_chunks.py lives under human_data/, and its default
# config path is relative to its OWN location -- so without this it would read the human
# burn-in floor and apply it to chimpanzee output.
if [[ $DRYRUN -eq 1 ]]; then
    echo "would extract draws ${DRAWS[*]} sequentially in this shell"
    exit 0
fi
python "$EXTRACT" \
    --singer-output "$SRC" \
    --out-dir "$OUT" \
    --config "$CONFIG" \
    --n-draws "$N_DRAWS"

# Now that extraction has SUCCEEDED, the draws below the ones we kept are dead weight:
# merge_chunks writes every draw, of which only these N are used. Report what could be freed,
# but do not delete it automatically -- reclaiming disk is not worth doing behind someone's
# back, and the dry run prints the exact command.
echo
python "${ROOT}/scripts/prune_chunk_draws.py" \
    --singer-output "$SRC" \
    --ts-chunks "${OUT}/draw_${MAXTOP}" \
    --keep-from "$FIRST" || true
echo "To reclaim that space, re-run the command above with --apply"
