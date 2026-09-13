#!/usr/bin/env bash
# STEP 5 -- phase with SHAPEIT5.
#
#   bash drivers/5_phase.sh
#
# Needs step 1. Reads  intermediates/filtered/<chrom>.sitefiltered.vcf.gz
#              Writes intermediates/phased/<chrom>.vcf.gz
#
# SHAPEIT5 phase_common, no reference panel and no genetic map (--hmm-ne 10000). Beagle was
# tested and lost on the same chromosome with the same metric: 4.66% switch error against
# SHAPEIT5's 4.03%, and worse in every subspecies. Beagle is ~9x faster, which is not the
# binding constraint across 23 chromosomes.
#
# --hmm-ne was swept at 10,000 / 25,000 / 30,000 and made no difference (0.06-point spread,
# non-monotonic, against a 0.20-point swing between chromosomes at the SAME setting), so no
# "best" value should be quoted from that sweep.
#
# 23-task array, ~4-10 h per task. The longest step in the prep.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
submit_and_wait phase_all.sbatch
echo "Step 5 complete. Next: bash drivers/6_assemble.sh"
