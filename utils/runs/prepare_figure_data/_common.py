"""
Shared helpers for the figure-data preparation scripts.

Each script in this directory turns raw run outputs (results/outputs/) or a fresh
simulation/optimization into the tidy tables one figure needs, under
results/figure_data/<figure_name>/. The plotting scripts in
results/figures/plotting_scripts/ then only read those tables.

Nothing here imports matplotlib or plotting_style: prep scripts produce data, not
pictures.
"""
import os
import sys

# repo root is four levels up: utils/runs/prepare_figure_data/_common.py
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "..", "..", ".."))
OUTPUTS_DIR = os.path.join(REPO_ROOT, "results", "outputs")
FIGURE_DATA_DIR = os.path.join(REPO_ROOT, "results", "figure_data")
DEMES_DIR = os.path.join(REPO_ROOT, "data", "demes")
CONFIGS_DIR = os.path.join(REPO_ROOT, "data", "configs")
METADATA_DIR = os.path.join(REPO_ROOT, "data", "metadata")


def add_src_to_path():
    """Put src/ and the snakemake/misc helper dirs on sys.path, once."""
    for rel in (("src",),
                ("utils", "snakemake_utils"),
                ("utils", "misc_utils")):
        path = os.path.join(REPO_ROOT, *rel)
        if path not in sys.path:
            sys.path.append(path)


def out_dir(figure_name):
    """Create and return results/figure_data/<figure_name>/."""
    path = os.path.join(FIGURE_DATA_DIR, figure_name)
    os.makedirs(path, exist_ok=True)
    return path


class MissingRun(FileNotFoundError):
    """Raised when a run output a figure depends on is not present on disk."""


def require(path, what=None):
    """
    Return path, or fail with a message naming what is missing.

    Several runs referenced by the figures have not been produced on this machine;
    the prep script for such a figure is expected to stop here with a clear message
    rather than fail deep inside pandas.
    """
    if not os.path.exists(path):
        label = f" ({what})" if what else ""
        raise MissingRun(f"missing run: {path}{label}")
    return path


def require_dir(path, what=None):
    """Like require(), but also insists the path is a directory."""
    require(path, what)
    if not os.path.isdir(path):
        raise MissingRun(f"missing run: {path} (not a directory)")
    return path


def find_output_file(run_dir, prefix, **params):
    """
    Resolve one of the long parameterised output filenames inside run_dir.

    The pipeline writes names like

        embeddings.d2.Nmin1000.Nmax1000000.p1.linkinverse_distance.L1[None]...parquet

    so a file is picked by its prefix (the part before the first ".", e.g.
    "embeddings", "rates", "reconstructed_rates") plus any number of parameter
    tokens that must appear in the name, given as keyword arguments:

        find_output_file(data_dir, "embeddings", p=1, d=2)

    matches a name containing both ".p1." and ".d2.". A keyword whose value is None
    contributes just the bare key as a substring.

    Fails if the run directory is absent, or if the pattern matches no file or more
    than one -- an ambiguous match means the caller has to be more specific, and
    silently taking the first would make the figure depend on directory order.
    """
    require_dir(run_dir, f"run data directory for {prefix!r}")

    # "embeddings" must not match "smooth_embeddings" or "loss_embeddings", so the
    # prefix is anchored at the start of the name and followed by a "."
    candidates = [f for f in sorted(os.listdir(run_dir))
                  if f.startswith(prefix + ".")]

    tokens = []
    for key, value in params.items():
        tokens.append(f".{key}." if value is None else f".{key}{value}.")
    for token in tokens:
        candidates = [f for f in candidates if token in f]

    described = f"{prefix}" + ("".join(f" {t.strip('.')}" for t in tokens))
    if not candidates:
        raise MissingRun(
            f"missing run: no {described!r} output in {run_dir}")
    if len(candidates) > 1:
        raise MissingRun(
            f"ambiguous match for {described!r} in {run_dir}: {candidates}")
    return os.path.join(run_dir, candidates[0])


def write_table(df, path, index=True):
    """
    Write one table, choosing the format from the extension.

    Parquet keeps dtypes and a MultiIndex intact, so it is the default for the
    numeric tables; csv is used for the small label/colour/parameter tables that
    are easier to read as text. Parquet is written with zstd rather than the
    default snappy: these tables are checked in, and the coordinates must stay at
    full precision, so compression is the only size lever available.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith(".parquet"):
        df.to_parquet(path, compression="zstd")
    elif path.endswith(".csv"):
        df.to_csv(path, index=index)
    else:
        raise ValueError(f"Unsupported table format for {path}")
    return path


def report(paths):
    """Print the tables a prep script wrote, relative to the repo root."""
    for path in paths:
        print(f"  wrote {os.path.relpath(path, REPO_ROOT)}")
