#!/usr/bin/env bash
# STEP 4 -- cut the finished ARGs into the per-chunk tree sequences CoaSTT reads.
#
#   bash drivers/4_extract_chunks.sh ALL      # one set
#   bash drivers/4_extract_chunks.sh          # every set with singer-snakemake output
#
# Needs step 3 to have FINISHED. Writes results/<SET>/ts_chunks/<chrom>_<start>_<end>.trees
#
# Flat -- every chromosome's chunks in one directory, no subdirectories -- because
# CountCoalEvents globs *.trees and turns each filename into a contig id. The chr prefix is
# kept deliberately: a contig id starting with a digit is awkward in Python and pandas.
#
# Writes ONE draw per chunk. CoaSTT treats each .trees file as an independent contig, so
# several posterior draws of the same region would enter that region into the likelihood more
# than once.
#
# Two things it refuses to do, both of which would otherwise fail silently:
#
#   * take a draw from inside the burn-in implied by drivers/singer_config.yaml.
#     singer-snakemake applies singer-mcmc-burnin only to its statistics and diagnostic plots
#     -- merge_chunks writes a tree sequence for every draw, pre-convergence ones included --
#     so this is the last point an unconverged draw can be caught.
#   * proceed when chromosomes disagree on which replicate is their latest, which happens when
#     a merge job was killed. Extraction takes the highest-numbered replicate per chromosome,
#     so one chromosome missing its top draw would silently contribute a different, less
#     burnt-in posterior sample than the rest. Re-run step 3 to fill the gap -- it redoes only
#     the missing merges, not the inference behind them.
#
# Runs in minutes, on the login node.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
bash "${ROOT}/scripts/extract_chunks.sh" "$@"
echo
echo "Pipeline complete. CoaSTT input: results/<SET>/ts_chunks/"
