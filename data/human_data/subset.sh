#!/usr/bin/env bash
# Build the 8 singer-snakemake input directories, one per sample set.
#
#   bash subset.sh                                              # local, parallel
#   bash subset.sh --jobs 4                                     # local, 4 sets at a time
#   bash subset.sh --slurm-partition jnovembre \
#                  --slurm-account pi-jnovembre                 # one job array, 8 tasks
#
# Local is the default and is usually the better choice: this is a few minutes of bcftools per
# set, so queue wait typically exceeds the work. Each set writes a `.complete` sentinel, so a
# re-run skips finished sets and resumes a partial one.
#
# Output: SingerSnakemake_SampleSets/<set>/input/, one directory per set, each holding the five
# per-chromosome files singer-snakemake expects (see scripts/make_set_inputs.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SETS_DIR="${ROOT}/resources/hgdp/sets"

JOBS=4
PARTITION=""
ACCOUNT=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --jobs) JOBS=$2; shift 2 ;;
        --slurm-partition) PARTITION=$2; shift 2 ;;
        --slurm-account) ACCOUNT=$2; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -d "$SETS_DIR" ]] || {
    echo "ERROR: ${SETS_DIR} not found -- run download.sh first." >&2
    exit 1
}

# Sorted, so the SLURM array index is stable across invocations.
mapfile -t SETS < <(cd "$SETS_DIR" && ls *.csv | sed 's/\.csv$//' | sort)
[[ ${#SETS[@]} -gt 0 ]] || { echo "ERROR: no sample sets in ${SETS_DIR}" >&2; exit 1; }

echo "Building ${#SETS[@]} sample sets: ${SETS[*]}"

if [[ -n "$PARTITION" && -n "$ACCOUNT" ]]; then
    LOGDIR="${ROOT}/logs"
    mkdir -p "$LOGDIR"
    last=$(( ${#SETS[@]} - 1 ))
    script=$(mktemp)
    cat > "$script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=hgdp_subset
#SBATCH --array=0-${last}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=${LOGDIR}/subset_%A_%a.out
#SBATCH --error=${LOGDIR}/subset_%A_%a.err
set -euo pipefail

# CoaSTT_env's libstdc++ must precede the system one or numpy fails to import on midway3.
export LD_LIBRARY_PATH="\${CONDA_PREFIX:-}/lib:\${LD_LIBRARY_PATH:-}"

SETS=(${SETS[*]})
bash "${ROOT}/scripts/make_set_inputs.sh" "\${SETS[\$SLURM_ARRAY_TASK_ID]}" "${ROOT}"
EOF
    sbatch "$script"
    rm -f "$script"
    echo "Submitted array of ${#SETS[@]} tasks; logs in ${LOGDIR}/"
    exit 0
fi

printf '%s\n' "${SETS[@]}" \
    | xargs -P "$JOBS" -I{} bash "${ROOT}/scripts/make_set_inputs.sh" {} "$ROOT"

echo
echo "== verifying sample set inputs =="
python "${ROOT}/scripts/verify_sets.py" --root "$ROOT"
