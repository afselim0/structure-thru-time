#!/usr/bin/env bash
# Run singer-snakemake on chr22 for all 8 HGDP sample sets.
#
#   bash runs/run_chr22.sh [SET ...]     # default: all 8
#
# Each set gets its OWN workdir under runs/work/<SET>. Snakemake keeps per-run state in
# <workdir>/.snakemake, and concurrent runs sharing one directory race on .snakemake/iocache
# -- `--nolock` is not enough. The workdir only needs to hold that state; input and output
# paths in the config are absolute.
#
# The driver snakemake processes are launched in the background and submit their own SLURM
# jobs via the profile, so this returns immediately. Watch progress with:
#   tail -f runs/work/<SET>/snakemake.log
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SINGER="${ROOT}/../singer-snakemake"

export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:${PATH}"
# CoaSTT_env's libstdc++ must lead or numpy fails to import on midway3 compute nodes.
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

SETS=("$@")
if [[ ${#SETS[@]} -eq 0 ]]; then
    SETS=(AFRICA ALL AMERICA CENTRAL_SOUTH_ASIA EAST_ASIA EUROPE MIDDLE_EAST OCEANIA)
fi

for s in "${SETS[@]}"; do
    cfg="${ROOT}/runs/chr22_configs/${s}.yaml"
    work="${ROOT}/runs/work/${s}"
    [[ -f "$cfg" ]] || { echo "no config for ${s}" >&2; exit 1; }
    mkdir -p "$work"

    echo "[${s}] launching (workdir ${work})"
    nohup snakemake \
        --snakefile "${SINGER}/workflow/Snakefile" \
        --configfile "$cfg" \
        --directory "$work" \
        --profile "${ROOT}/slurm_profile" \
        > "${work}/snakemake.log" 2>&1 &

    # Stagger: simultaneous starts contend on the shared conda prefix and the SLURM submit host.
    sleep 20
done

echo
echo "All ${#SETS[@]} launched. Monitor with:"
echo "  tail -f ${ROOT}/runs/work/<SET>/snakemake.log"
echo "  squeue -u \$USER"
