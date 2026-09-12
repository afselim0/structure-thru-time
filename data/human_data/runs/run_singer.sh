#!/usr/bin/env bash
# Run singer-snakemake on the HGDP sample sets, then cut the final posterior draw into the
# per-chunk tree sequences CoaSTT consumes.
#
#   bash runs/run_singer.sh --chrom chr22                    # all 8 sets, chr22
#   bash runs/run_singer.sh --profile caslake_profile ...    # shared partition (see below)
#   bash runs/run_singer.sh --chrom chr22 OCEANIA            # one set
#   bash runs/run_singer.sh OCEANIA                          # one set, all chromosomes
#   bash runs/run_singer.sh --pilot --chrom chr22 OCEANIA    # reduced MCMC, for smoke tests
#
# ALL scientific parameters come from ../singer_config.yaml. This script overrides only the
# keys that identify a run -- input-dir, output-dir, chromosomes -- plus the two documented
# per-set exceptions below, via --config, which snakemake applies after --configfile. There are
# deliberately no per-set config files to drift out of sync.
#
# Exception 1: ALL is stratified by region, not population. It is 4 samples from each of 54
# populations, so stratifying by population gives 1,485 cross-coalescence pairs each backed by
# 8 haplotypes -- too sparse to estimate. The 7 regions give 28 pairs with real support. Every
# other set is a single region, where `population` is the informative level.
#
# Exception 2: --pilot lowers the MCMC lengths, to exercise the pipeline rather than to produce
# usable posteriors.
#
# Produces, per set:
#   results/<SET>/singer-snakemake_outputs/   the workflow's own output tree
#   results/<SET>/ts_chunks/                  chr<N>_<start>_<end>.trees, flat -- CoaSTT input
#
# Each set gets its own workdir under runs/work/<SET>: snakemake keeps run state in
# <workdir>/.snakemake and concurrent runs sharing a directory race on .snakemake/iocache
# (--nolock is not sufficient).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SINGER="${ROOT}/../singer-snakemake"

export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:${PATH}"
# CoaSTT_env's libstdc++ must lead or numpy fails to import on midway3 compute nodes.
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

CHROM=""
PILOT=0
# slurm_profile = the dedicated jnovembre partition (192 CPUs, 12h walltime).
# caslake_profile = the shared partition: 9120 CPUs, no time limit, but jobs queue behind other
# users. It requests one CPU and right-sized memory per rule, which backfills far better on a
# shared queue than the blanket 16 GB the jnovembre profile uses.
PROFILE="slurm_profile"
SETS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --chrom) CHROM=$2; shift 2 ;;
        --profile) PROFILE=$2; shift 2 ;;
        --pilot) PILOT=1; shift ;;
        -*) echo "unknown flag: $1" >&2; exit 1 ;;
        *) SETS+=("$1"); shift ;;
    esac
done
[[ ${#SETS[@]} -gt 0 ]] || SETS=(AFRICA ALL AMERICA CENTRAL_SOUTH_ASIA EAST_ASIA EUROPE MIDDLE_EAST OCEANIA)

# The pilot exists to exercise the pipeline quickly, not to produce usable posteriors, so it is
# the one place MCMC lengths deviate from the shared config.
PILOT_OVERRIDES=()
SUFFIX=""
if [[ $PILOT -eq 1 ]]; then
    # Cut the NUMBER of samples but keep the shared thinning interval, so a pilot draw comes
    # from the same chain behaviour as a production one. Lowering thin as well would make the
    # pilot a different MCMC regime rather than a smaller version of the real run.
    PILOT_OVERRIDES=(singer-mcmc-samples=4 polegon-mcmc-samples=4)
    SUFFIX="_pilot"
fi

for s in "${SETS[@]}"; do
    input="${ROOT}/SingerSnakemake_SampleSets/${s}/input"
    [[ -d "$input" ]] || { echo "no input directory for ${s}" >&2; exit 1; }

    name="${s}${SUFFIX}"
    out="${ROOT}/results/${name}/singer-snakemake_outputs"
    work="${ROOT}/runs/work/${name}"
    mkdir -p "$work"

    overrides=(input-dir="$input" output-dir="$out")
    [[ -n "$CHROM" ]] && overrides+=(chromosomes="[\"${CHROM}\"]")

    # See "Exception 1" above: ALL spans 54 populations at 4 samples each, so region is the
    # only level with enough haplotypes per group to estimate cross-coalescence.
    if [[ "$s" == "ALL" ]]; then
        overrides+=(stratify-by=region)
    fi

    overrides+=("${PILOT_OVERRIDES[@]}")

    echo "[${name}] launching (workdir ${work}, profile ${PROFILE})"
    nohup snakemake \
        --snakefile "${SINGER}/workflow/Snakefile" \
        --configfile "${ROOT}/singer_config.yaml" \
        --config "${overrides[@]}" \
        --directory "$work" \
        --profile "${ROOT}/${PROFILE}" \
        > "${work}/snakemake.log" 2>&1 &

    # Stagger: simultaneous starts contend on the shared conda prefix and the submit host.
    sleep 20
done

echo
echo "Launched ${#SETS[@]}. Monitor with:"
echo "  tail -f ${ROOT}/runs/work/<SET>${SUFFIX}/snakemake.log"
echo "  squeue -u \$USER"
echo
echo "When a set finishes, build its CoaSTT input directory:"
echo "  bash ${ROOT}/runs/extract_chunks.sh <SET>${SUFFIX}"
