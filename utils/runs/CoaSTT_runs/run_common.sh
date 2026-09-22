#!/bin/bash
# Shared body for the CoaSTT run scripts in sim_runs/ and app_runs/.
#
# Sourced, never executed. It lets one script serve both machines: run it with
# no flags and the pipeline runs here; run it with --slurm and the same script
# submits itself as a batch job sized by its own resource declaration.
#
#     bash CoaSTT_runs/sim_runs/split.sh                # here, now
#     bash CoaSTT_runs/sim_runs/split.sh --slurm        # as a SLURM job
#     bash CoaSTT_runs/sim_runs/split.sh --slurm -n     # print the sbatch plan
#     bash CoaSTT_runs/sim_runs/split.sh -n             # snakemake dry run, here
#
# Cluster settings (account, partition, environment, log and workdir paths)
# come from slurm.yaml beside this file -- see slurm.yaml.example. Nothing
# about a particular cluster belongs in a run script.
#
# A run script is:
#
#     set -euo pipefail
#     source "$(dirname "${BASH_SOURCE[0]}")/../run_common.sh"
#
#     coastt_resources cpus=12 mem_per_cpu=4G time=12:00:00
#     coastt_run STT_sim --configfile ../data/configs/sim_configs/split.yaml
#
# coastt_run forwards anything left on the command line to snakemake, so `-n`
# and friends reach it in both modes. Under --slurm it also passes
# `--config num_threads=<allocated cpus>` unless the script sets num_threads
# itself, so the pipeline's workers match the allocation rather than the
# config's own value, which was written for a different machine.
#
# A script that submits several dependent jobs of its own (chimps.sh,
# arg_rates_comparison.sh) parses --slurm the same way but calls
# coastt_sbatch directly instead of coastt_run.

# ---------------------------------------------------------------- paths ----

# Absolute paths, resolved from this file rather than from $0, so a run script
# works from any directory and a sourcing script does not have to re-derive
# them.
COASTT_RUNS_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COASTT_RUNS_DIR="$(cd "$COASTT_RUNS_ROOT/.." && pwd)"
COASTT_UTILS="$(cd "$COASTT_RUNS_DIR/.." && pwd)"
COASTT_REPO="$(cd "$COASTT_UTILS/.." && pwd)"

# ------------------------------------------------------- cluster config ----

# Flat `key: value` reader. Not a YAML parser: it handles the one-level files
# slurm.yaml.example documents and nothing more, which keeps the run scripts
# free of a python dependency at submit time. Strips comments and surrounding
# quotes; $VAR and ${VAR} are left intact here and expanded by the caller.
_coastt_config_get() {
    local key="$1" file="$2"
    awk -v key="$key" '
        # key: value, at the start of a line, before any trailing # comment
        $0 ~ "^[[:space:]]*" key "[[:space:]]*:" {
            sub("^[[:space:]]*" key "[[:space:]]*:[[:space:]]*", "")
            sub(/[[:space:]]*#.*$/, "")
            sub(/[[:space:]]+$/, "")
            # strip one layer of matching quotes
            if ($0 ~ /^".*"$/ || $0 ~ /^'"'"'.*'"'"'$/)
                $0 = substr($0, 2, length($0) - 2)
            print
            exit
        }' "$file"
}

# Populate COASTT_ACCOUNT, COASTT_PARTITION, ... from slurm.yaml. Called only
# on the --slurm path, so a local run never needs the file to exist.
coastt_load_slurm_config() {
    local cfg="${COASTT_SLURM_CONFIG:-$COASTT_RUNS_ROOT/slurm.yaml}"

    if [[ ! -f "$cfg" ]]; then
        cat >&2 <<EOF
No cluster config at $cfg

--slurm needs your account, partition and environment. Create it once:

    cp $COASTT_RUNS_ROOT/slurm.yaml.example $COASTT_RUNS_ROOT/slurm.yaml
    \$EDITOR $COASTT_RUNS_ROOT/slurm.yaml

(or point COASTT_SLURM_CONFIG at a copy kept elsewhere).
EOF
        return 1
    fi

    COASTT_ACCOUNT="$(_coastt_config_get account "$cfg")"
    COASTT_PARTITION="$(_coastt_config_get partition "$cfg")"
    COASTT_SBATCH_EXTRA="$(_coastt_config_get sbatch_extra "$cfg")"
    COASTT_CONDA_ENV="$(_coastt_config_get conda_env "$cfg")"
    COASTT_CONDA_SH="$(_coastt_config_get conda_sh "$cfg")"
    COASTT_PRE_ACTIVATE="$(_coastt_config_get pre_activate "$cfg")"

    # Expand $VAR/${VAR} in the path-valued settings, so workdir_root can be
    # written as $SCRATCH/$USER/... in the config. eval is confined to these
    # three keys and the file is the user's own.
    local raw
    raw="$(_coastt_config_get log_dir "$cfg")"
    eval "COASTT_LOG_DIR=\"${raw:-job_logs}\""
    raw="$(_coastt_config_get workdir_root "$cfg")"
    eval "COASTT_WORKDIR_ROOT=\"${raw:-${SCRATCH:-/tmp}/$USER/coastt_work}\""
    eval "COASTT_CONDA_ENV=\"$COASTT_CONDA_ENV\""

    # Relative log_dir resolves against utils/runs/, where the old scripts put it.
    [[ "$COASTT_LOG_DIR" != /* ]] && COASTT_LOG_DIR="$COASTT_RUNS_DIR/$COASTT_LOG_DIR"

    if [[ -z "$COASTT_CONDA_ENV" ]]; then
        echo "slurm.yaml: conda_env is required for --slurm (jobs start from a clean shell)" >&2
        return 1
    fi
}

# --------------------------------------------------------- declarations ----

# Resource request for this run, e.g.
#     coastt_resources cpus=48 mem_per_cpu=2G time=08:00:00
# Used only under --slurm. Anything unset falls back to the defaults below,
# which are the ones the old job arrays used.
COASTT_CPUS=12
COASTT_MEM_PER_CPU=4G
COASTT_TIME=12:00:00

coastt_resources() {
    local kv
    for kv in "$@"; do
        case "$kv" in
            cpus=*)        COASTT_CPUS="${kv#*=}" ;;
            mem_per_cpu=*) COASTT_MEM_PER_CPU="${kv#*=}" ;;
            mem=*)         COASTT_MEM="${kv#*=}" ;;
            time=*)        COASTT_TIME="${kv#*=}" ;;
            *) echo "coastt_resources: unknown setting '$kv'" >&2; return 1 ;;
        esac
    done
}

# ------------------------------------------------------ argument parsing ----
#
# Sourcing this file consumes --slurm (and -n/--dry-run alongside it) from the
# script's own arguments; everything else is left in COASTT_ARGS for snakemake.
# The run scripts do not each re-parse their command line.

COASTT_SLURM=0
COASTT_DRY_RUN=0
COASTT_ARGS=()

for _arg in "$@"; do
    case "$_arg" in
        --slurm) COASTT_SLURM=1 ;;
        -n|--dry-run|--dryrun) COASTT_DRY_RUN=1 ;;
        *) COASTT_ARGS+=("$_arg") ;;
    esac
done
unset _arg

# -n means "show me what would happen", which is a different thing in each
# mode: with --slurm it prints the sbatch commands and their payloads and
# submits nothing, so it must NOT also reach snakemake (that would submit a
# real job that then only dry-runs). Without --slurm there is nothing to
# submit, so it is snakemake's own dry run and is passed straight through.
if [[ $COASTT_DRY_RUN -eq 1 && $COASTT_SLURM -eq 0 ]]; then
    COASTT_ARGS+=(-n)
fi

# --------------------------------------------------------------- submit ----

# The job body: get into the environment, build a private working directory,
# and run the command. Printed into the sbatch payload by coastt_sbatch.
#
# Escaping note: this heredoc is unquoted, so `$x` is expanded now (by the
# submitting shell, baking a value into the payload) and `\$x` is expanded
# later (by the job). Getting one wrong is the classic failure here.
_coastt_job_preamble() {
    local name="$1"
    cat <<EOS
set -euo pipefail

echo "host \$(hostname), \${SLURM_CPUS_PER_TASK:-?} cpus, started \$(date)"

${COASTT_PRE_ACTIVATE:+$COASTT_PRE_ACTIVATE}

# Batch jobs start from a clean shell, so CONDA_EXE is usually unset: source
# the hook explicitly rather than relying on the submitting shell. A prefix env
# works without the hook, by PATH alone, which is the fallback.
if [[ -f "$COASTT_CONDA_SH" ]]; then
    source "$COASTT_CONDA_SH"
    conda activate "$COASTT_CONDA_ENV"
elif [[ -x "$COASTT_CONDA_ENV/bin/python" ]]; then
    export PATH="$COASTT_CONDA_ENV/bin:\$PATH"
    export CONDA_PREFIX="$COASTT_CONDA_ENV"
else
    echo "Cannot activate '$COASTT_CONDA_ENV': no conda hook at '$COASTT_CONDA_SH' and no \$CONDA_ENV/bin/python" >&2
    exit 1
fi

# snakemake's highspy ships a libstdc++ that shadows the env's, and numpy then
# fails with a GLIBCXX error as soon as snakemake imports pandas. The env's lib
# has to come first. The rules invoke a bare \`python\`, so the env has to be on
# PATH too, not merely imported from.
export LD_LIBRARY_PATH="\${CONDA_PREFIX:-$COASTT_CONDA_ENV}/lib\${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"

# A private snakemake working directory: a mirror of utils/ with the repo's real
# directories symlinked in, so the snakefile's relative paths (../src, ../data,
# ../results, snakemake_utils) all resolve and writes land in the repo, while
# .snakemake/ stays private to this job. See workdir_root in slurm.yaml for
# why concurrent jobs cannot share one.
# Array tasks share one job name, so the task id has to be in the path or the
# tasks would rm -rf each other's workdir. Unset for an ordinary job, which
# then gets the bare name as before.
work="$COASTT_WORKDIR_ROOT/$name\${SLURM_ARRAY_TASK_ID:+_\$SLURM_ARRAY_TASK_ID}"
rm -rf "\$work"
mkdir -p "\$work/utils"
for item in src data results; do
    ln -sfn "$COASTT_REPO/\$item" "\$work/\$item"
done
for item in snakemake_utils misc_utils arg_infer_utils data_utils runs snakefile; do
    ln -sfn "$COASTT_UTILS/\$item" "\$work/utils/\$item"
done
cd "\$work/utils"
EOS
}

# Submit one job.
#
#     coastt_sbatch <name> [--dependency=afterok:N] [--cpus-per-task=..] ... -- <command...>
#
# Everything before `--` is passed to sbatch on top of the account, partition,
# resource and log settings; everything after is the command the job runs,
# inside the activated environment and the private workdir. Prints the job id
# on stdout (progress goes to stderr), so callers can chain dependencies.
coastt_sbatch() {
    local name="$1"; shift

    local sbatch_args=()
    while [[ $# -gt 0 && "$1" != "--" ]]; do
        sbatch_args+=("$1"); shift
    done
    [[ "${1:-}" == "--" ]] && shift
    [[ $# -gt 0 ]] || { echo "coastt_sbatch: no command given" >&2; return 1; }

    mkdir -p "$COASTT_LOG_DIR"

    local args=(
        --job-name="$name"
        --nodes=1 --ntasks=1
        --cpus-per-task="$COASTT_CPUS"
        --time="$COASTT_TIME"
        --output="$COASTT_LOG_DIR/${name}_%j.out"
        --error="$COASTT_LOG_DIR/${name}_%j.err"
        # Each array task has its own job id, so %j alone keeps the logs
        # distinct and non-array log names are unchanged.
    )
    # mem wins over mem_per_cpu when a script asks for a whole-job total.
    if [[ -n "${COASTT_MEM:-}" ]]; then
        args+=(--mem="$COASTT_MEM")
    else
        args+=(--mem-per-cpu="$COASTT_MEM_PER_CPU")
    fi
    [[ -n "$COASTT_ACCOUNT" ]]   && args+=(--account="$COASTT_ACCOUNT")
    [[ -n "$COASTT_PARTITION" ]] && args+=(--partition="$COASTT_PARTITION")
    # Set by generate_*_results.sh when it holds a run behind an earlier phase.
    # A script that builds its own chain passes --dependency explicitly instead,
    # in which case that one wins (sbatch takes the last occurrence).
    [[ -n "${COASTT_DEPENDENCY:-}" ]] && args+=(--dependency="afterok:$COASTT_DEPENDENCY")
    # Unquoted on purpose: sbatch_extra is a whitespace-separated flag list.
    # shellcheck disable=SC2206
    [[ -n "$COASTT_SBATCH_EXTRA" ]] && args+=($COASTT_SBATCH_EXTRA)
    args+=("${sbatch_args[@]}")

    local payload
    payload="#!/bin/bash
$(_coastt_job_preamble "$name")

$(printf '%q ' "$@")

echo \"finished \$(date)\""

    if [[ $COASTT_DRY_RUN -eq 1 ]]; then
        echo "[dry] sbatch ${args[*]}" >&2
        echo "$payload" | sed 's/^/      | /' >&2
        echo "<${name}_jobid>"
        return 0
    fi

    sbatch --parsable "${args[@]}" <<< "$payload"
}

# ------------------------------------------------------------------ run ----

# Run one snakemake invocation, here or as a job.
#
#     coastt_run <target> --configfile <path> [--config KEY=VALUE ...] [snakemake flags]
#
# Paths in the arguments are relative to utils/, as everywhere else in the
# pipeline. COASTT_ARGS (whatever the user passed through) is appended.
# COASTT_JOB_NAME overrides the job name, which otherwise comes from the
# calling script's filename.
coastt_run() {
    local name="${COASTT_JOB_NAME:-coastt_$(basename "${BASH_SOURCE[1]}" .sh)}"
    local snake_args=("$@" ${COASTT_ARGS[@]+"${COASTT_ARGS[@]}"})

    if [[ $COASTT_SLURM -eq 1 ]]; then
        coastt_load_slurm_config || return 1

        # Size the pipeline's workers to the allocation rather than to the
        # config's num_threads, which was written for a different machine. Left
        # alone if the script set it, and appended as its own --config group
        # because snakemake takes --config as a trailing key=value list.
        local threads_arg=()
        [[ " ${snake_args[*]} " == *" num_threads="* ]] \
            || threads_arg=(--config "num_threads=$COASTT_CPUS")

        # --nolock: the private workdir already isolates .snakemake state, and
        # a stale lock from a killed job would otherwise block a resubmission.
        local jid
        jid=$(coastt_sbatch "$name" -- \
                  snakemake "${snake_args[@]}" "${threads_arg[@]}" \
                  --cores all --rerun-incomplete --nolock) || return 1
        echo "submitted $name: job $jid" >&2
        return 0
    fi

    # Local: run from utils/, where every pipeline path is anchored.
    cd "$COASTT_UTILS"

    # The system /lib64/libstdc++ (EL8) is older than numpy 2.5 needs, so
    # snakemake dies with `GLIBCXX_3.4.29 not found` as soon as it imports
    # pandas. Put the active env's own lib first. Harmless elsewhere.
    if [[ -n "${CONDA_PREFIX:-}" ]]; then
        export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    # Match the workers to the CPUs actually available. SLURM_CPUS_PER_TASK is
    # set when this runs inside someone's interactive allocation; otherwise use
    # the cpuset size rather than a fixed default, which silently
    # under-subscribed a 16-CPU allocation. With neither, leave the config's
    # own num_threads alone.
    local threads_arg=()
    if [[ " ${snake_args[*]} " != *" num_threads="* ]]; then
        local cpus="${SLURM_CPUS_PER_TASK:-$(nproc 2>/dev/null || true)}"
        [[ -n "$cpus" ]] && threads_arg=(--config "num_threads=$cpus")
    fi

    snakemake "${snake_args[@]}" "${threads_arg[@]}" --cores all --rerun-incomplete
}

# Run CoaSTT on inferred trees (the app_runs scripts), here or as a job.
#
#     coastt_run_inferred <config_basename> <infer_method> [KEY=VALUE ...]
#
# Each demography's config writes its simulated trees to an output_path ending
# in /sim. This redirects output_path to a sibling directory named after the
# inference method, so both tree sources live side by side under one run dir:
#   ../results/outputs/simulations/<demography>/sim/trees      <- simulated
#   ../results/outputs/simulations/<demography>/<method>/trees <- inferred
# The snakefile's vcf_dir defaults to tree_output_path with infer_method
# swapped back to "sim", so infer_trees picks up the simulated trees with no
# explicit path. The matching sim_runs script must be run first.
coastt_run_inferred() {
    local config_name="$1"; shift
    local infer_method="$1"; shift

    local configfile="$COASTT_UTILS/../data/configs/sim_configs/${config_name}.yaml"
    if [[ ! -f "$configfile" ]]; then
        echo "coastt_run_inferred: config not found: $configfile" >&2
        return 1
    fi

    # Replace the trailing "sim" component of the config's output_path with the
    # inference method, keeping the rest of the run directory intact.
    local sim_output_dir
    sim_output_dir="$(grep -E '^output_path:' "$configfile" | head -n1 | sed -E 's/^output_path:[[:space:]]*//')"
    if [[ -z "$sim_output_dir" ]]; then
        echo "coastt_run_inferred: no output_path in $configfile" >&2
        return 1
    fi
    if [[ "${sim_output_dir%/sim}" == "$sim_output_dir" ]]; then
        echo "coastt_run_inferred: output_path '$sim_output_dir' does not end in /sim" >&2
        return 1
    fi
    local output_dir="${sim_output_dir%/sim}/${infer_method}"

    COASTT_JOB_NAME="${COASTT_JOB_NAME:-coastt_$(basename "${BASH_SOURCE[1]}" .sh)}" \
    coastt_run STT_infer \
        --configfile "../data/configs/sim_configs/${config_name}.yaml" \
        --config "output_path=$output_dir" "infer_method=$infer_method" "$@"
}
