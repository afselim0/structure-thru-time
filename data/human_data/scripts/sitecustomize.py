"""
Auto-applied by Python when `scripts/` is on PYTHONPATH.

Installs the guard in slurm_executor_patch.py, which stops one SLURM NODE_FAIL from poisoning
every subsequent job submission. scripts/run_singer.sh puts this directory on PYTHONPATH, so the
guard is applied to the snakemake driver without touching site-packages -- a site-packages edit
would be silently lost whenever the conda environment is rebuilt from environment.yml.

Harmless if the SLURM executor is not installed, and a no-op if already applied.
"""

try:
    from slurm_executor_patch import apply as _apply_slurm_node_guard

    _apply_slurm_node_guard()
except Exception:  # never let a guard break the interpreter
    pass
