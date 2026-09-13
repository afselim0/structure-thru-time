#!/usr/bin/env bash
# Run singer-snakemake on the HGDP sample sets, then cut the final posterior draw into the
# per-chunk tree sequences CoaSTT consumes.
#
#   bash scripts/run_singer.sh --chrom chr22                      # all 8 sets, chr22
#   bash scripts/run_singer.sh --config slurm_caslake ...         # shared partition (see below)
#   bash scripts/run_singer.sh --chrom chr22 OCEANIA              # one set
#   bash scripts/run_singer.sh OCEANIA                            # one set, all chromosomes
#   bash scripts/run_singer.sh --pilot --chrom chr22 OCEANIA      # reduced MCMC, for smoke tests
#   bash scripts/run_singer.sh -n --chrom chr22 OCEANIA           # dry run: show the plan only
#
# A real launch detaches and returns immediately; -n stays in the foreground, prints the job
# plan and submits nothing. Use it to check paths and see how much work is outstanding.
#
# ALL scientific parameters come from drivers/singer_config.yaml. This script overrides only
# the keys that identify a run -- input-dir, output-dir, chromosomes -- plus the two documented
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
# Run state (snakemake's .snakemake directory) goes to a scratch workdir OUTSIDE the repo, one
# per set: concurrent runs sharing a directory race on .snakemake/iocache and --nolock is not
# sufficient. That state is disposable -- which chunks are finished is recorded by the
# `.singer.done` sentinels under results/, not by the workdir, so deleting a workdir loses no
# completed work and a restart picks up where the last one stopped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SINGER="${ROOT}/../singer-snakemake"
DRIVERS="${ROOT}/drivers"

export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:${PATH}"
# CoaSTT_env's libstdc++ must lead or numpy fails to import on midway3 compute nodes.
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"
# Silences a harmless "Can't find a usable init.tcl" from this env's Tcl, which was
# built for a differently-named prefix and is found via the lib path above.
export TCL_LIBRARY="/project/jnovembre/afselim/CoaSTT_env/lib/tcl8.6"

# Guard against a snakemake SLURM-executor bug: one NODE_FAIL makes it cache the literal
# string "None assigned" as a failed node, after which EVERY submission carries
# `--exclude=None assigned` and is rejected by SLURM. See scripts/slurm_executor_patch.py.
# Injected via sitecustomize so it applies to the driver without editing site-packages, which
# would not survive rebuilding the environment from environment.yml.
export PYTHONPATH="${ROOT}/scripts:${PYTHONPATH:-}"

CHROM=""
PILOT=0
DRYRUN=0
# Cluster config, by basename of a drivers/<name>.config file:
#   slurm_jnovembre = the dedicated jnovembre partition (192 CPUs, 48 h QOS ceiling).
#   slurm_caslake   = the shared partition: 9120 CPUs, 36 h QOS ceiling, but jobs queue behind
#                     other users. It requests one CPU and right-sized memory per rule, which
#                     backfills far better on a shared queue than a blanket large request.
CLUSTER_CONFIG="slurm_jnovembre"
# Where snakemake keeps per-run state. Deliberately outside the repo: it is large (hundreds of
# MB per set), regenerable, and must not be committed. Override with WORKDIR_ROOT=... if your
# scratch lives elsewhere.
# $SCRATCH on midway3 is the shared /scratch/midway3, not a per-user directory, so include
# $USER explicitly rather than writing into a path other people also own.
WORKDIR_ROOT="${WORKDIR_ROOT:-${SCRATCH:-/tmp}/${USER}/singer_work}"
SETS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        --chrom) CHROM=$2; shift 2 ;;
        --config) CLUSTER_CONFIG=$2; shift 2 ;;
        --pilot) PILOT=1; shift ;;
        -n|--dry-run) DRYRUN=1; shift ;;
        -*) echo "unknown flag: $1" >&2; exit 1 ;;
        *) SETS+=("$1"); shift ;;
    esac
done
[[ ${#SETS[@]} -gt 0 ]] || SETS=(AFRICA ALL AMERICA CENTRAL_SOUTH_ASIA EAST_ASIA EUROPE MIDDLE_EAST OCEANIA)

# Accept either `--config slurm_caslake` or `--config drivers/slurm_caslake.config`.
CLUSTER_FILE="${DRIVERS}/${CLUSTER_CONFIG##*/}"
CLUSTER_FILE="${CLUSTER_FILE%.config}.config"
[[ -f "$CLUSTER_FILE" ]] || {
    echo "no such cluster config: ${CLUSTER_FILE}" >&2
    echo "available:" >&2; ls "${DRIVERS}"/*.config >&2
    exit 1
}

# run_singer's memory need scales with the number of haplotypes -- SINGER holds the ARG for the
# whole sample in memory -- so a single constant is wrong for a set of 8 sample sets spanning
# 28 to 216 diploids. Sizing every set for the largest starves the partition: a jnovembre node
# has 257 GB and 64 CPUs, so at 12 GB per job only 21 jobs fit per node and 43 of its CPUs sit
# idle. AMERICA measured 0.87-1.32 GB against that 12 GB request.
#
# The model is proportional through the origin, anchored on the ALL measurement (6.81 GB peak
# over 576 jobs at 432 haplotypes) = 15.76 MB per haplotype, doubled for headroom, with a 4 GB
# floor. It is deliberately anchored on the LARGEST set: that over-predicts the small ones
# (AMERICA: predicts 1.92 GB against 1.32 measured, 1.46x) rather than under-predicting them,
# and a two-point fit would give a negative intercept, which is unphysical.
#
# Retries still escalate on top of whatever this picks, so a chunk that defies the model gets
# more memory on its next attempt rather than failing.
mem_for_set() {
    local set_name=$1
    local list="${ROOT}/resources/hgdp/sets/${set_name}.csv"
    [[ -f "$list" ]] || { echo ""; return; }          # unknown set: leave the config's value
    local diploids haplotypes mb
    diploids=$(grep -c . "$list")
    haplotypes=$(( diploids * 2 ))
    # 15.76 MB/haplotype x 2 headroom = 31.52; integer arithmetic, rounded up.
    mb=$(( (haplotypes * 3152 + 99) / 100 ))
    (( mb < 4000 )) && mb=4000
    echo "$mb"
}

# The pilot exists to exercise the pipeline quickly, not to produce usable posteriors, so it is
# the one place MCMC lengths deviate from the shared config.
PILOT_OVERRIDES=()
SUFFIX=""
STAGED_PROFILES=()
if [[ $PILOT -eq 1 ]]; then
    # Cut the NUMBER of samples but keep the shared thinning interval, so a pilot draw comes
    # from the same chain behaviour as a production one. Lowering thin as well would make the
    # pilot a different MCMC regime rather than a smaller version of the real run.
    PILOT_OVERRIDES=(singer-mcmc-samples=4 polegon-mcmc-samples=4)
    SUFFIX="_pilot"
fi

for s in "${SETS[@]}"; do
    input="${ROOT}/SingerSnakemake_SampleSets/${s}"
    [[ -d "$input" ]] || { echo "no input directory for ${s}" >&2; exit 1; }

    name="${s}${SUFFIX}"
    out="${ROOT}/results/${name}/singer-snakemake_outputs"
    work="${WORKDIR_ROOT}/${name}"
    mkdir -p "$work" "${ROOT}/logs"

    # Stage this set's profile. Each set gets its own directory because run_singer's memory is
    # sized per set (see mem_for_set above) -- and because snakemake re-reads the profile inside
    # every submitted job, so it must outlive this script and must not be shared between sets.
    PROFILE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/singer_profile_${name}.XXXXXXXX")"
    STAGED_PROFILES+=("$PROFILE_DIR")
    cp "$CLUSTER_FILE" "${PROFILE_DIR}/config.yaml"
    set_mem="$(mem_for_set "$s")"
    if [[ -n "$set_mem" ]]; then
        # Replace the run_singer mem_mb line, keeping the retry escalation term. Anchored on the
        # `run_singer:` key so the other rules' mem_mb lines are untouched.
        python3 - "${PROFILE_DIR}/config.yaml" "$set_mem" <<'PY'
import re, sys
path, mem = sys.argv[1], int(sys.argv[2])
text = open(path).read()
# within the run_singer block only, rewrite the mem_mb value and keep `+ 8000 * (attempt - 1)`
pat = re.compile(r"(    run_singer:\n)(?:.*\n)*?(        mem_mb: )[^\n]+")
new, n = pat.subn(
    lambda m: (
        f"{m.group(1)}"
        f"        # Sized for this sample set by scripts/run_singer.sh; see mem_for_set there.\n"
        f"{m.group(2)}{mem} + 8000 * (attempt - 1)"
    ),
    text, count=1,
)
if n != 1:
    sys.exit(f"could not rewrite run_singer mem_mb in {path}")
open(path, "w").write(new)
PY
        echo "[${name}] run_singer memory: ${set_mem} MB (from $(grep -c . "${ROOT}/resources/hgdp/sets/${s}.csv") diploids)"
    fi

    overrides=(input-dir="$input" output-dir="$out")
    [[ -n "$CHROM" ]] && overrides+=(chromosomes="[\"${CHROM}\"]")

    # See "Exception 1" above: ALL spans 54 populations at 4 samples each, so region is the
    # only level with enough haplotypes per group to estimate cross-coalescence.
    if [[ "$s" == "ALL" ]]; then
        overrides+=(stratify-by=region)
    fi

    overrides+=("${PILOT_OVERRIDES[@]}")

    # --dry-run stays in the FOREGROUND and submits nothing: the point is to see the plan.
    # Running it detached like a real launch would defeat that.
    if [[ $DRYRUN -eq 1 ]]; then
        echo "[${name}] dry run (workdir ${work}, cluster config ${CLUSTER_CONFIG})"
        snakemake \
            --snakefile "${SINGER}/workflow/Snakefile" \
            --configfile "${DRIVERS}/singer_config.yaml" \
            --config "${overrides[@]}" \
            --directory "$work" \
            --profile "$PROFILE_DIR" \
            -n
        continue
    fi

    log="${ROOT}/logs/${name}.snakemake.log"
    echo "[${name}] launching (workdir ${work}, cluster config ${CLUSTER_CONFIG})"
    # setsid + disown: without them the driver is a grandchild of the invoking shell and gets
    # reaped when that shell exits, which silently kills a multi-day run.
    setsid nohup snakemake \
        --snakefile "${SINGER}/workflow/Snakefile" \
        --configfile "${DRIVERS}/singer_config.yaml" \
        --config "${overrides[@]}" \
        --directory "$work" \
        --profile "$PROFILE_DIR" \
        >> "$log" 2>&1 < /dev/null &
    disown

    # Stagger: simultaneous starts contend on the shared conda prefix and the submit host.
    sleep 20
done

if [[ $DRYRUN -eq 1 ]]; then
    # Nothing was submitted, so nothing will read these again.
    rm -rf "${STAGED_PROFILES[@]}"
    echo
    echo "Dry run only -- nothing was submitted."
    exit 0
fi

# The staged profiles must OUTLIVE this script: the drivers keep running after it exits and
# snakemake re-reads the profile inside every job it submits. Deleting one mid-run fails every
# job submitted afterwards, so they are left in place and listed here instead.
echo
echo "Staged cluster configs (do NOT delete while the runs are live):"
printf '  %s\n' "${STAGED_PROFILES[@]}"

echo
echo "Launched ${#SETS[@]}. Monitor with:"
echo "  tail -f ${ROOT}/logs/<SET>${SUFFIX}.snakemake.log"
echo "  squeue -u \$USER"
echo
echo "When a set finishes, build its CoaSTT input directory:"
echo "  bash ${ROOT}/scripts/extract_chunks.sh <SET>${SUFFIX}"
