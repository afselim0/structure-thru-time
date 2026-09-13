# Shared helpers for the numbered drivers. Sourced, never run directly.
#
# Each driver is a thin wrapper: it submits the matching script in ../scripts/ and waits for
# it, so that running the drivers in order reproduces the pipeline with no other knowledge.
# The real work, and all the documentation of WHY each step does what it does, lives in the
# scripts themselves.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

export PATH="/project/jnovembre/afselim/CoaSTT_env/bin:${PATH}"
export LD_LIBRARY_PATH="/project/jnovembre/afselim/CoaSTT_env/lib:${LD_LIBRARY_PATH:-}"

# Submit an sbatch script and block until it finishes, so `bash 1_*.sh && bash 2_*.sh` is a
# valid way to run the pipeline. --wait makes sbatch exit with the job's own status, so a
# failed array stops the chain instead of letting the next step run on missing inputs.
submit_and_wait() {
    local script=$1; shift
    echo "--> sbatch --wait scripts/${script} $*"
    sbatch --wait "scripts/${script}" "$@"
}
