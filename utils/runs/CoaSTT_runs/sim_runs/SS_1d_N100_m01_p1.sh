#!/bin/bash
# CoaSTT run: SS_1d_N100_m01_p1
#
#     bash SS_1d_N100_m01_p1.sh           # here
#     bash SS_1d_N100_m01_p1.sh --slurm   # as a SLURM job
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

coastt_resources cpus=24 mem_per_cpu=4G time=08:00:00

coastt_run STT_sim --configfile ../data/configs/sim_configs/SS_1d_N100_m01_p1.yaml
