#!/usr/bin/env bash
# STEP 3 -- build the recombination maps and ancestral-state FASTAs.
#
#   bash drivers/3_build_resources.sh
#
# Writes resources/panTro6/hapmap/<chrom>.hapmap
#        resources/panTro6/ancestral/<chrom>.ancestral.fa.gz
#
# Independent of steps 1 and 2. Neither of these could be downloaded; the human prep gets
# both from a public release, and here each has to be built.
#
# The hapmap lifts the Stevison ellioti map hg18 -> hg38 -> panTro6 and scales it with ONE
# global scalar across all chromosomes (per-chromosome scaling would erase the real
# obligate-crossover signal). The ancestral FASTA projects hg38 bases through the
# panTro6->hg38 chain, and refuses to write output unless per-strand identity against the
# panTro6 reference clears 90% -- the gate that caught a minus-strand coordinate bug which a
# pooled identity check would have passed. See the two scripts for the full reasoning.
#
# 23-task array, ~1-3 h per task.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
submit_and_wait build_resources.sbatch
echo "Step 3 complete. Next: bash drivers/4_build_masks.sh"
