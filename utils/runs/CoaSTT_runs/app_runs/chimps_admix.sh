#!/bin/bash
# CoaSTT run: the admixed chimpanzee dataset.
#
#     bash chimps_admix.sh           # here
#     bash chimps_admix.sh --slurm   # as a SLURM job
#
# Local ancestry for the NC and "W x C" individuals against two size-matched
# reference panels (EasternCentral = 5 Eastern + 5 Central individuals, West =
# 10 Western; 20 haplotypes each), then the whole rates/embeddings/plotting
# pipeline once genome-wide and once per ancestry panel.
#
# Sized like chimps.sh, which starts from the same 1,334 tree sequences (73 GB)
# and is dominated by the same compute_pdf counting step. Two differences push
# the wall clock up rather than down, so the 36 h is kept:
#   * infer_local_ancestry (GNN) walks every local tree of every contig before
#     any counting starts;
#   * compute_pdf then writes a per-ancestry pdf for each panel on top of the
#     genome-wide one, i.e. three counting passes rather than one.
# The counting is resumable -- an interrupted run validates and reuses the .npy
# files already written -- so a timeout costs only the contigs in flight.
#
# Unlike chimps.sh this is a single job: chimps_admix.yaml has its own
# output_path (results/outputs/chimps/admix) and its own pdf grid
# (pdf_maxT200000 vs 450000), so it shares nothing with the chimps/no_admix runs
# and can go alongside them with no dependency.
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

coastt_resources cpus=16 mem_per_cpu=8G time=36:00:00

coastt_run STT --configfile ../data/configs/app_configs/chimps_admix.yaml
