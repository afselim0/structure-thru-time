# Shared helpers for the numbered drivers. Sourced, never run directly.
#
# Each driver is a thin wrapper: it runs the matching script in ../scripts/ and waits for it,
# so that running the drivers in order reproduces the pipeline with no other knowledge. The
# real work, and all the documentation of WHY each step does what it does, lives in the
# scripts themselves.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

# Use whatever environment is already active if it has snakemake; otherwise fall back to the
# prefix env this project was developed against. A clone elsewhere just needs
# `conda activate CoaSTT_env` (or its own equivalent) before running these.
if ! command -v snakemake >/dev/null 2>&1; then
    _fallback="/project/jnovembre/afselim/CoaSTT_env"
    if [[ -x "${_fallback}/bin/snakemake" ]]; then
        export PATH="${_fallback}/bin:${PATH}"
        # This env's libstdc++ must lead or numpy fails to import on midway3 compute nodes.
        export LD_LIBRARY_PATH="${_fallback}/lib:${LD_LIBRARY_PATH:-}"
        # Putting that lib directory first also makes system binaries pick up this env's Tcl,
        # which was built for a differently-named prefix and prints a scary but harmless
        # "Can't find a usable init.tcl" on startup. Point it at the real one.
        [[ -d "${_fallback}/lib/tcl8.6" ]] && export TCL_LIBRARY="${_fallback}/lib/tcl8.6"
    fi
fi

# The interpreter that owns snakemake, not whatever `python` resolves to: on a login node with
# a system conda ahead of the project env on PATH those differ, and only the former has
# tskit/tszip.
python_bin() {
    local p
    p="$(dirname "$(command -v snakemake 2>/dev/null || echo /nonexistent)")/python"
    [[ -x "$p" ]] && echo "$p" || echo python
}

# Fail early and legibly rather than part-way through a long step.
require_env() {
    command -v snakemake >/dev/null 2>&1 || {
        echo "snakemake not found. Activate the environment first:" >&2
        echo "    conda env create -f environment.yml && conda activate CoaSTT_env" >&2
        exit 1
    }
}

require_slurm() {
    command -v sbatch >/dev/null 2>&1 || {
        echo "sbatch not found -- the cluster configs in drivers/ submit to SLURM." >&2
        exit 1
    }
}

require_submodule() {
    [[ -f "${ROOT}/../singer-snakemake/workflow/Snakefile" ]] || {
        echo "singer-snakemake submodule missing. Run: git submodule update --init" >&2
        exit 1
    }
}

# The sample sets the pipeline builds, in the order make_sample_sets.py writes them.
SAMPLE_SETS=(AFRICA ALL AMERICA CENTRAL_SOUTH_ASIA EAST_ASIA EUROPE MIDDLE_EAST OCEANIA)

require_set() {
    local s=$1
    [[ -d "${ROOT}/SingerSnakemake_SampleSets/${s}" ]] || {
        echo "no input directory for '${s}'." >&2
        echo "Run steps 1 and 2 first, then pick one of:" >&2
        printf '    %s\n' "${SAMPLE_SETS[@]}" >&2
        exit 1
    }
}
