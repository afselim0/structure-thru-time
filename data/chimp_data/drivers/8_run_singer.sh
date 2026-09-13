#!/usr/bin/env bash
# STEP 8 -- infer the ARGs with SINGER, and date them with POLEGON.
#
#   bash drivers/8_run_singer.sh                          # all 23 autosomes
#   bash drivers/8_run_singer.sh --chrom chr21            # one chromosome
#   bash drivers/8_run_singer.sh --config slurm_caslake   # shared partition
#   bash drivers/8_run_singer.sh --pilot --chrom chr21    # reduced MCMC, smoke test
#   bash drivers/8_run_singer.sh -n                       # dry run: print the plan only
#
# Needs step 7. Writes results/chimp/singer-snakemake_outputs/<chrom>/trees/<chrom>.<rep>.tsz
#
# Unlike steps 1-7 this does NOT block: it launches a detached snakemake that submits one
# SLURM job per rule and returns immediately. Monitor with `tail -f logs/chimp.snakemake.log`
# and `squeue -u $USER`. Expect roughly a day for the whole genome.
#
# Scientific parameters come from drivers/singer_config.yaml; cluster settings from
# drivers/slurm_jnovembre.config (default) or drivers/slurm_caslake.config. Try --pilot first
# on one chromosome if you are running this for the first time.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
bash "${ROOT}/scripts/run_singer.sh" "$@"
