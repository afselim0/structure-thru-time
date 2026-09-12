"""
Guard against snakemake's SLURM executor poisoning `--exclude` with a bogus node name.

THE BUG (snakemake-executor-plugin-slurm 2.8.0, __init__.py ~line 1512)

When a job ends in NODE_FAIL the executor records the node so later jobs avoid it:

    sacct_output = subprocess.check_output(f"sacct -j {jobid} -n -X -o nodelist%-256", ...)
    node = sacct_output.strip()
    if node:
        self._failed_nodes.add(node)          # <-- no validation

`sacct` reports a job's nodelist as the literal string "None assigned" when the job failed
before SLURM placed it on a node. That string passes the `if node:` test, lands in
`_failed_nodes`, and is then rendered into every subsequent submission:

    sbatch ... --exclude=None assigned ...

SLURM parses `--exclude=None`, hits the stray word `assigned`, and rejects the job:

    sbatch: error: Batch job submission failed: Invalid node name specified

which the executor surfaces as the much less obvious "Unable to open file assigned".

It is a latch: one NODE_FAIL poisons the set for the lifetime of the driver, and EVERY later
submission fails. Observed on a whole-genome run -- a single node failure was followed by 8,522
consecutive failed submissions, and the pipeline made no further progress past SINGER.

THE FIX

Wrap `set.add` on the executor's `_failed_nodes` so only plausible SLURM node names get in.
Valid names are alphanumerics, underscores, hyphens, dots and comma/bracket range syntax --
never whitespace, which is what makes "None assigned" detectable.

This is applied by importing this module before snakemake starts (runs/run_singer.sh does it
via PYTHONSTARTUP-style injection). It is deliberately NOT a patch to site-packages: that would
not survive `conda env create` from environment.yml, so a fresh clone would hit the bug again.

Remove this once upstream validates the sacct output itself.
"""

import re

# A SLURM node name or node expression. No whitespace: that is the tell.
_VALID_NODE = re.compile(r"^[A-Za-z0-9_.\-\[\],]+$")


def apply() -> bool:
    """Install the guard. Returns True if patched, False if the executor is absent."""
    try:
        from snakemake_executor_plugin_slurm import Executor
    except ImportError:
        return False

    if getattr(Executor, "_coastt_node_guard", False):
        return True

    original_post_init = Executor.__post_init__

    def guarded_post_init(self, *args, **kwargs):
        original_post_init(self, *args, **kwargs)
        real_set = self._failed_nodes

        class _GuardedNodeSet(set):
            def add(self, value):
                text = str(value).strip()
                if not text or not _VALID_NODE.match(text):
                    # "None assigned" and friends: sacct had no node to report.
                    return
                super().add(text)

        guarded = _GuardedNodeSet(
            n for n in real_set if _VALID_NODE.match(str(n).strip())
        )
        self._failed_nodes = guarded

    Executor.__post_init__ = guarded_post_init
    Executor._coastt_node_guard = True
    return True


if __name__ == "__main__":
    print("patched" if apply() else "slurm executor not installed")
