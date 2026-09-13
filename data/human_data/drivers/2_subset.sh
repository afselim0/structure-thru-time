#!/usr/bin/env bash
# STEP 2 -- cut the shared resources into the 8 per-set singer-snakemake input directories.
#
#   bash drivers/2_subset.sh                                     # local, 4 sets at a time
#   bash drivers/2_subset.sh --jobs 8                            # local, all 8 at once
#   bash drivers/2_subset.sh --slurm-partition jnovembre \
#                            --slurm-account pi-jnovembre        # one 8-task SLURM array
#
# Needs step 1. Writes SingerSnakemake_SampleSets/<SET>/, holding per chromosome:
#
#   chr{N}.vcf.gz(.tbi)      subset to this set's samples, trimmed to segregating SNPs
#   chr{N}.meta.csv          sample metadata, in VCF order -- the order is load-bearing
#   chr{N}.sample.mask.bed   lightly-missing genotypes, for singer-snakemake to impute
#   chr{N}.mask.bed          symlink to the shared inaccessible-interval mask
#   chr{N}.hapmap            symlink to the shared recombination map
#   chr{N}.ancestral.fa.gz   symlink to the shared ancestral FASTA
#
# Local is the default and usually the better choice: this is a few minutes of bcftools per
# set, so queue wait typically exceeds the work. Each set writes a `.complete` sentinel, so a
# re-run skips finished sets and resumes a partial one.
#
# Ends with verify_sets.py, 2,320 checks across all 8 sets.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
bash "${ROOT}/scripts/subset.sh" "$@"
echo "Step 2 complete. Next: bash drivers/3_run_singer.sh <SET>"
