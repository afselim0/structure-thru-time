#!/usr/bin/env bash
# STEP 2 -- build the whole-genome mappability index.
#
#   bash drivers/2_genmap_index.sh
#
# Reads  resources/raw/panTro6.fa.gz
# Writes resources/panTro6/genmap/   (the index, plus per-base mappability)
#
# Independent of step 1 -- both can be submitted at the same time. Step 4 needs this.
#
# genmap at k=36, e=0: unique 36-mers with exact matching. This is the ONLY component of the
# accessibility mask, which is why the chimpanzee mask retains 79-91% against the human
# prep's ~74%. That is a mask measuring less, not cleaner data: depth-based callability is
# impossible on a variant-only callset, and RepeatMasker was dropped deliberately (it covers
# 87% of the assembly, and most of that sequence is uniquely mappable and perfectly callable).
#
# Single job, ~40 min, 64 GB. Run once -- it is genome-wide, not per chromosome.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
submit_and_wait genmap_index.sbatch
echo "Step 2 complete. Next: bash drivers/3_build_resources.sh"
