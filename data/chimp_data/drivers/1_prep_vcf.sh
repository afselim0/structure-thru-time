#!/usr/bin/env bash
# STEP 1 -- split the raw callset by chromosome, drop bonobos, apply the site filters.
#
#   bash drivers/1_prep_vcf.sh
#
# Reads  resources/vcf/full.vcf.gz   (28 GB, 61 samples, GenBank accession names)
# Writes intermediates/filtered/<chrom>.sitefiltered.vcf.gz
#        intermediates/filter_positions/<chrom>.txt
#
# Three things happen per chromosome: the 7 bonobos are dropped and then sites no chimpanzee
# carries; the contig is renamed from its GenBank accession (CM009259.2) to the UCSC name
# (chr21); and the three site filters run (GATK hard filters, allelic imbalance, depth).
#
# Filtering happens HERE, before phasing, and not by preference. SHAPEIT5 emits GT and
# nothing else, so FORMAT/AD and FORMAT/DP -- which the allelic-imbalance and depth filters
# need -- do not survive it. It also measures better: 3.59% switch error against 4.32% for
# the reverse order. See scripts/filter_sites.py.
#
# 23-task array, ~2-4 h per task.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
submit_and_wait prep_vcf.sbatch
echo "Step 1 complete. Next: bash drivers/2_genmap_index.sh"
