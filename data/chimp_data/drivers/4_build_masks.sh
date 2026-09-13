#!/usr/bin/env bash
# STEP 4 -- build the accessibility masks.
#
#   bash drivers/4_build_masks.sh
#
# Needs steps 1 and 2 (excess-het flags from the filtering, mappability from genmap).
# Writes resources/panTro6/mask/<chrom>.mask.bed
#        and restricts intermediates/filter_positions/<chrom>.txt to accessible sequence
#
# mask.bed holds INACCESSIBLE intervals. Storing the opposite polarity inverts the analysis
# with no error raised anywhere, which is why verify (step 7) checks the accessible fraction
# against a band rather than just checking the file exists.
#
# Mappability union assembly gaps union excess-heterozygosity regions. Excess het is computed
# WITHIN subspecies, never pooled: the panel is deeply structured, and pooling produces
# heterozygote deficit by the Wahlund effect -- the wrong direction entirely.
#
# 23-task array, ~30-60 min per task.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
submit_and_wait build_mask_all.sbatch
echo "Step 4 complete. Next: bash drivers/5_phase.sh"
