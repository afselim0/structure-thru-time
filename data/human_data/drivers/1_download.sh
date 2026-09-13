#!/usr/bin/env bash
# STEP 1 -- download the raw HGDP data and derive the shared per-chromosome resources.
#
#   bash drivers/1_download.sh
#
# Writes resources/raw/          the statphase VCF, accessibility mask, metadata, ancestral FASTA
#        resources/grch38/       ancestral/, mask/ (complemented to INACCESSIBLE), hapmap/
#        resources/hgdp/         metadata.tsv, sets/*.csv, statphase/<chrom>.bcf (GT-only)
#
# Everything here is shared by all 8 sample sets and independent of the cluster, so this runs
# once and never needs repeating. ~15 GB and a few hours, mostly the 6.3 GB source VCF.
#
# ONE FILE MUST BE PLACED BY HAND FIRST. science.org serves its supplements behind bot
# protection and returns 403 to every scripted request, so the deCODE recombination map cannot
# be downloaded automatically. Grab Data S3 from doi:10.1126/science.aau1043 and put it at
# resources/raw/aau1043_datas3.gz (gzip it if it arrives uncompressed). This step stops with
# these instructions if it is missing.
#
# Ends with verify_resources.py, ~200 checks over the derived files.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
bash "${ROOT}/scripts/download.sh" "$@"
echo "Step 1 complete. Next: bash drivers/2_subset.sh"
