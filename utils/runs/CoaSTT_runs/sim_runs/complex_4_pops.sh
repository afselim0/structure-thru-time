#!/bin/bash
# CoaSTT run: complex_4_pops
#
#     bash complex_4_pops.sh           # here
#     bash complex_4_pops.sh --slurm   # as a SLURM job
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../run_common.sh"

coastt_resources cpus=12 mem_per_cpu=4G time=12:00:00

coastt_run STT_sim --configfile ../data/configs/sim_configs/complex_4_pops.yaml
