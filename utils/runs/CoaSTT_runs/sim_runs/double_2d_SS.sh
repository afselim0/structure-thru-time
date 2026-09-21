#!/bin/bash
# CoaSTT run: double_2d_SS
#
#     bash double_2d_SS.sh           # here
#     bash double_2d_SS.sh --slurm   # as a SLURM job
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

coastt_resources cpus=12 mem_per_cpu=4G time=12:00:00

coastt_run STT_sim --configfile ../data/configs/sim_configs/double_2d_SS.yaml
