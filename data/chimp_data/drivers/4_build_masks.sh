#!/usr/bin/env bash
# STEP 4 -- build the accessibility masks.
#
#   bash drivers/4_build_masks.sh
#
# Needs step 2 (genmap mappability) and step 1 (filter.txt, to trim).
# Writes resources/panTro6/mask/<chrom>.mask.bed
#        and restricts intermediates/filter_positions/<chrom>.txt to accessible sequence
#
# mask.bed holds INACCESSIBLE intervals. Storing the opposite polarity inverts the analysis
# with no error raised anywhere, which is why verify (step 7) checks the accessible fraction
# against a band rather than just checking the file exists.
#
# Mappability (genmap k=36, e=0, threshold < 1.0) union assembly gaps. Nothing else: excess
# heterozygosity was dropped because it needed >= 4 called samples per subspecies to test and
# this panel is too small for that to be reliable (ellioti n=5).
#
# 23-task array, ~30-60 min per task.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
submit_and_wait build_mask_all.sbatch
echo "Step 4 complete. Next: bash drivers/5_phase.sh"
