#!/usr/bin/env bash
# STEP 9 -- cut the finished ARGs into the per-chunk tree sequences CoaSTT reads.
#
#   bash drivers/9_extract_chunks.sh                                  # 8 draws, here, serially
#   bash drivers/9_extract_chunks.sh --partition caslake \
#                                    --account pi-jnovembre           # 8 draws, 8 parallel jobs
#   bash drivers/9_extract_chunks.sh --n-draws 4 --partition caslake \
#                                    --account pi-jnovembre           # 4 draws, 4 parallel jobs
#   bash drivers/9_extract_chunks.sh --name chimp_pilot               # a pilot run
#
# Needs step 8 to have FINISHED. Writes, for each of the last N posterior draws,
#   results/chimp/ts_chunks/draw_<rep>/<chrom>_<start>_<end>.trees
# and defaults to N = 8: the last 8 of the chain's 20 draws, the first 12 being burn-in.
#
# One directory per draw, each a complete CoaSTT input on its own. They are NOT pooled: the
# draws are repeat measurements of one genome, so putting them in a single directory would
# enter every region into the likelihood eight times over. Point a run at one draw; compare
# runs across draws to separate posterior uncertainty from signal.
#
# Within a draw the directory is flat -- every chromosome's chunks together, no subdirectories
# -- because CountCoalEvents globs *.trees and turns each filename into a contig id.
#
# Refuses to run if the chromosomes disagree on which replicate is their latest, which
# happens when a merge job was killed: the extraction takes the highest-numbered replicates
# per chromosome, so one chromosome missing its top draw would silently contribute a
# different, less burnt-in posterior sample than the rest. Re-run step 8 to fill the gap --
# it redoes only the missing merges, not the inference behind them.
#
# Give it --partition AND --account and it submits one SLURM job per draw, which is the way
# to run it: each job decompresses whole chromosome tree sequences and needs ~120 GB, more
# than a login node allows, and the draws share nothing so they cost no more in parallel than
# in sequence. Without both flags the draws are extracted one after another in this shell,
# which works anywhere but will be OOM-killed on a midway3 login node.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
bash "${ROOT}/scripts/extract_chunks.sh" "$@"

# Only claim completion when something was actually built. A dry run printed a plan, and a
# SLURM fan-out has merely SUBMITTED the jobs -- the draws do not exist until they finish, and
# saying otherwise here would send someone to an empty directory.
for a in "$@"; do
    case $a in
        -n|--dry-run) exit 0 ;;
        --partition|--account) exit 0 ;;
    esac
done
echo
echo "Pipeline complete. CoaSTT input: results/chimp/ts_chunks/draw_<rep>/"
