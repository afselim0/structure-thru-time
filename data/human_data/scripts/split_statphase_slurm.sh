#!/usr/bin/env bash
# Submit the statphase split as a SLURM job array, one chromosome per task.
#
#   bash scripts/split_statphase_slurm.sh --partition jnovembre --account pi-jnovembre
#   bash scripts/split_statphase_slurm.sh --partition jnovembre --account pi-jnovembre --chroms "18 19 20 21 22"
#
# The local path in split_statphase.sh walks the chromosomes in sequence, which takes a couple
# of hours for all 22. This runs them concurrently instead. Each task is independent: it reads
# a region of the shared source VCF and writes one chr{N}.bcf, so there is nothing to
# serialise and no shared state to race on.
#
# Tasks skip chromosomes that already have a .bcf with a NEWER .csi, so re-submitting after a
# partial run only redoes what is missing. A chromosome killed mid-write leaves a .bcf with no
# .csi and is correctly redone.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PARTITION=""
ACCOUNT=""
CHROMS=""
CPUS=4
MEM="8G"
TIME="04:00:00"
while [[ $# -gt 0 ]]; do
    case $1 in
        --partition) PARTITION=$2; shift 2 ;;
        --account)   ACCOUNT=$2;   shift 2 ;;
        --chroms)    CHROMS=$2;    shift 2 ;;
        --cpus)      CPUS=$2;      shift 2 ;;
        --mem)       MEM=$2;       shift 2 ;;
        --time)      TIME=$2;      shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$PARTITION" && -n "$ACCOUNT" ]] || {
    echo "usage: $0 --partition <p> --account <a> [--chroms \"1 2 3\"]" >&2
    exit 1
}

VCF="${ROOT}/resources/raw/hgdp_wgs.20190516.statphase.autosomes.vcf.gz"
OUTDIR="${ROOT}/resources/hgdp/statphase"
LOGDIR="${ROOT}/logs"
mkdir -p "$OUTDIR" "$LOGDIR"

[[ -s "$VCF" ]] || { echo "ERROR: ${VCF} not found -- run download.sh first." >&2; exit 1; }

# Default to whatever is not already finished.
if [[ -z "$CHROMS" ]]; then
    todo=()
    for n in $(seq 1 22); do
        bcf="${OUTDIR}/chr${n}.bcf"
        if [[ -f "$bcf" && -f "${bcf}.csi" && "${bcf}.csi" -nt "$bcf" ]]; then
            continue
        fi
        todo+=("$n")
    done
    CHROMS="${todo[*]:-}"
fi

read -r -a CHROM_ARRAY <<< "$CHROMS"
if [[ ${#CHROM_ARRAY[@]} -eq 0 ]]; then
    echo "Nothing to do: all 22 chromosomes are already split and indexed."
    exit 0
fi

last=$(( ${#CHROM_ARRAY[@]} - 1 ))
echo "Submitting ${#CHROM_ARRAY[@]} chromosome(s): ${CHROM_ARRAY[*]}"

script=$(mktemp)
cat > "$script" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=hgdp_split
#SBATCH --array=0-${last}
#SBATCH --partition=${PARTITION}
#SBATCH --account=${ACCOUNT}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --mem=${MEM}
#SBATCH --time=${TIME}
#SBATCH --output=${LOGDIR}/split_%A_%a.out
#SBATCH --error=${LOGDIR}/split_%A_%a.err
set -euo pipefail

# CoaSTT_env is a prefix env; its bin must lead PATH, and its lib must lead LD_LIBRARY_PATH or
# numpy fails to import against the system libstdc++ on midway3.
export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:\${PATH}"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:\${LD_LIBRARY_PATH:-}"

CHROMS=(${CHROM_ARRAY[*]})
n=\${CHROMS[\$SLURM_ARRAY_TASK_ID]}
chrom="chr\${n}"
out="${OUTDIR}/\${chrom}.bcf"

if [[ -f "\${out}.csi" && "\${out}.csi" -nt "\$out" ]]; then
    echo "[\${chrom}] already done, skipping"
    exit 0
fi

echo "[\${chrom}] extracting on \$(hostname) at \$(date)"
bcftools view --threads ${CPUS} -r "\$chrom" -v snps -Ou "${VCF}" \
  | bcftools annotate --threads ${CPUS} -x "INFO,^FORMAT/GT" -Ob -o "\$out"
bcftools index --threads ${CPUS} -f "\$out"
echo "[\${chrom}] \$(bcftools index -n "\$out") records, done at \$(date)"
EOF

sbatch "$script"
rm -f "$script"
echo "Logs will appear in ${LOGDIR}/split_<jobid>_<task>.{out,err}"
