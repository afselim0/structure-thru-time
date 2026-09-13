#!/usr/bin/env bash
# STEP 9 -- cut the finished ARGs into the per-chunk tree sequences CoaSTT reads.
#
#   bash drivers/9_extract_chunks.sh                # the production run
#   bash drivers/9_extract_chunks.sh chimp_pilot    # a pilot run
#
# Needs step 8 to have FINISHED. Writes results/chimp/ts_chunks/<chrom>_<start>_<end>.trees
#
# Flat -- every chromosome's chunks in one directory, no subdirectories -- because
# CountCoalEvents globs *.trees and turns each filename into a contig id.
#
# Refuses to run if the chromosomes disagree on which replicate is their latest, which
# happens when a merge job was killed: the extraction takes the highest-numbered replicate
# per chromosome, so one chromosome missing its top draw would silently contribute a
# different, less burnt-in posterior sample than the rest. Re-run step 8 to fill the gap --
# it redoes only the missing merges, not the inference behind them.
#
# Runs in minutes, on the login node.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
bash "${ROOT}/scripts/extract_chunks.sh" "$@"
echo
echo "Pipeline complete. CoaSTT input: results/chimp/ts_chunks/"
