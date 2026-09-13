#!/usr/bin/env bash
# STEP 6 -- assemble the singer-snakemake input directory.
#
#   bash drivers/6_assemble.sh
#
# Needs steps 3, 4 and 5. Writes SingerSnakemake_Input/<chrom>.{vcf.gz,tbi,filter.txt,
# meta.csv,mask.bed,hapmap,ancestral.fa.gz} -- 7 files x 23 chromosomes = 161.
#
# Merges the multiallelic records back into the phased VCF, unphased. SHAPEIT5 drops them,
# but they must be PRESENT for the workflow to count them into filtered_positions, which
# scales adj_mu. They are discarded before any haplotype is built and exist purely to be
# counted; omitting them would inflate mu by roughly 8%.
#
# Also writes meta.csv in VCF sample order -- the workflow zips by index with no ID matching,
# so the order is load-bearing -- and symlinks the shared per-chromosome resources.
#
# 23-task array, ~20-40 min per task.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
submit_and_wait assemble_all.sbatch
echo "Step 6 complete. Next: bash drivers/7_verify.sh"
