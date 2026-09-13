#!/usr/bin/env bash
# STEP 3 -- infer the ARGs with SINGER, and date them with POLEGON.
#
#   bash drivers/3_run_singer.sh ALL                              # one set, whole genome
#   bash drivers/3_run_singer.sh ALL --config slurm_caslake       # shared partition
#   bash drivers/3_run_singer.sh ALL --chrom chr21                # one chromosome
#   bash drivers/3_run_singer.sh OCEANIA --pilot --chrom chr22    # reduced MCMC, smoke test
#   bash drivers/3_run_singer.sh ALL -n                           # dry run: print the plan only
#   bash drivers/3_run_singer.sh                                  # all 8 sets
#
# Needs step 2. Writes results/<SET>/singer-snakemake_outputs/<chrom>/trees/<chrom>.<rep>.tsz
#
# Unlike steps 1-2 this does NOT block: it launches a detached snakemake that submits one SLURM
# job per rule and returns immediately. Monitor with `tail -f logs/<SET>.snakemake.log` and
# `squeue -u $USER`. Expect roughly 14 h of wall clock for ALL genome-wide at 500 concurrent
# jobs, before queueing -- ~6,700 CPU-h.
#
# Scientific parameters come from drivers/singer_config.yaml; cluster settings from
# drivers/slurm_jnovembre.config (default) or drivers/slurm_caslake.config.
#
# TO RUN THIS ON YOUR OWN CLUSTER: copy one of the drivers/slurm_*.config files, change the
# account and partition in it, and pass it as `--config slurm_<yourcluster>`. Nothing else
# changes. Try --pilot on one chromosome first.
#
# Run state goes to $SCRATCH/$USER/singer_work/<SET>, outside the repository. It is disposable:
# which chunks are finished is recorded by the .singer.done sentinels under results/, so a
# rerun rebuilds its plan from results/ and queues only what is genuinely outstanding.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
require_env
require_submodule
bash "${ROOT}/scripts/run_singer.sh" "$@"
echo "Once the jobs finish: bash drivers/4_extract_chunks.sh <SET>"
