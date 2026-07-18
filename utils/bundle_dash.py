#!/usr/bin/env python3
"""Package a CoaSTT dash embedding into a self-contained zip.

Collects the dash app plus the data files it needs (embeddings, rates,
reconstructed rates, labels, colors, optional covariances), flattens all paths,
and writes a `run.sh` + `environment.yml` + `README.md` so a collaborator can
unzip and launch the interactive plot with no knowledge of this repo's layout.

Two ways to use it (run from the `utils/` directory):

1. From a run's snakemake config (recommended — one-liner, derives everything):

    python bundle_dash.py --config ../data/empirical_runs/wohns_ragsdale_chr21.yaml

2. From explicit paths (for embeddings produced outside the pipeline):

    python bundle_dash.py ../results/.../data/embeddings.<TAG>.parquet \\
        --labels <labels>.csv --colors <colors>.csv \\
        --rates <rates>.parquet --reconstructed_rates <recon>.parquet \\
        --N_max 1000000 --time_scale log --min_time 140 --max_time 58337 \\
        -o ../results/.../dash_bundle.zip

CLI flags override values derived from --config.

Rate parquets dominate bundle size; by default their timepoint columns are
auto-thinned to ~the embeddings timepoint count (near-lossless — the viewer
snaps the slider to the nearest rate column and can't resolve finer than the
embeddings grid). Use --rates-time-stride N for an explicit divisor, or 1 to
disable. Rates still exceeding the (post-thin) size limit are dropped with a
warning unless --force-rates-export is set.
"""
import argparse
import os
import shutil
import tempfile
import textwrap
import zipfile

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DASH_APP = os.path.join(THIS_DIR, "plot_embeddings_dash.py")

# Minimal runtime deps for plot_embeddings_dash.py (NOT the full CoaSTT env).
DASH_DEPS = ["dash", "plotly", "pandas", "numpy", "scipy", "pyarrow"]

# Rate parquets dominate bundle size and grow ~quadratically in sample count.
# Above this combined (post-thinning) size they are excluded unless forced.
DEFAULT_MAX_RATES_MB = 500


def _environment_yml(env_name="stt_dash"):
    """Build a minimal conda environment.yml pinned to the current interpreter's
    package versions, so the bundle reproduces a working dash environment."""
    import sys
    import importlib.metadata as md
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    lines = [f"name: {env_name}", "channels:", "  - conda-forge", "dependencies:",
             f"  - python={py}"]
    for dep in DASH_DEPS:
        try:
            lines.append(f"  - {dep}={md.version(dep)}")
        except md.PackageNotFoundError:
            lines.append(f"  - {dep}")
    return "\n".join(lines) + "\n"


def _timepoint_columns(parquet_path):
    """Return a rate parquet's timepoint data-column names in file order,
    excluding the (Multi)Index columns. Read from the schema only (no data)."""
    import pyarrow.parquet as pq
    schema = pq.read_schema(parquet_path)
    md = schema.pandas_metadata or {}
    index_cols = {ic for ic in md.get("index_columns", []) if isinstance(ic, str)}
    return [name for name in schema.names if name not in index_cols]


def _emb_timepoint_count(embeddings_path):
    """Number of distinct embedding timepoints (the dash slider grid size).
    The embeddings parquet is long-format with a 'time' column."""
    import pandas as pd
    return int(pd.read_parquet(embeddings_path, columns=["time"])["time"].nunique())


def _resolve_stride(rates_time_stride, parquet_path, emb_n_timepoints):
    """Resolve a per-parquet integer stride. 'auto' picks a stride so the kept
    timepoint count ~= the embeddings count (near-lossless); otherwise int>=1."""
    if rates_time_stride == "auto":
        if not emb_n_timepoints or emb_n_timepoints <= 0:
            return 1
        n = len(_timepoint_columns(parquet_path))
        return max(1, round(n / emb_n_timepoints))
    return max(1, int(rates_time_stride))


def _thin_rate_parquet(src, dest, stride):
    """Write a copy of rate parquet `src` to `dest` keeping every `stride`-th
    timepoint COLUMN, preserving the pair MultiIndex. Always retains the last
    timepoint. Columns kept as their original (string) names; no value changes."""
    import pandas as pd
    cols = _timepoint_columns(src)
    keep = cols[::stride]
    if keep and keep[-1] != cols[-1]:
        keep.append(cols[-1])
    # pandas' pyarrow engine restores index columns from the parquet metadata
    # regardless of `columns=`, so passing only the kept timepoint columns keeps
    # the MultiIndex intact while reading just those columns (columnar/efficient).
    pd.read_parquet(src, columns=keep).to_parquet(dest)


def derive_from_config(config_path, base_dir=THIS_DIR):
    """Reconstruct the dash inputs (file paths + plotting params) for a run from
    its snakemake config yaml, mirroring utils/Snakefile's tag construction.

    Relative paths in the config are resolved against *base_dir* (the utils/ dir,
    matching where snakemake runs). Returns a dict ready for build_bundle().
    """
    import yaml

    with open(config_path) as f:
        cfg = yaml.safe_load(f) or {}

    def r(key, default):
        return cfg[key] if key in cfg else default

    # --- params (defaults copied from Snakefile read_config calls) ----------
    output_path = r("output_path", "../results")
    version = r("version", 1)
    fraction_samples = r("fraction_samples", 1.0)
    max_time = r("max_time", 10000)
    n_timepoints = r("n_timepoints", max_time)
    min_window_size = r("min_window_size", 50)
    max_window_size = r("max_window_size", 500)
    N_min = r("N_min", 1)
    N_max = r("N_max", N_min + 100000)
    k = r("k", 1)
    link = r("link", "inverse_distance")
    l1 = r("l1", "None")
    l2 = r("l2", "None")
    if isinstance(l1, list):
        l1 = ",".join(str(x) for x in l1)
    if isinstance(l2, list):
        l2 = ",".join(str(x) for x in l2)
    n_components = r("n_components", 2)
    n_timepoints_emb = r("n_timepoints_emb", 400)
    min_time = r("min_time", 1)
    embeddings_max_time = r("embeddings_max_time", max_time)
    embeddings_plot_lims = r("embeddings_plot_lims", "None")
    time_scale = r("time_scale", "linear")
    pop_colors = r("pop_colors", os.path.join(output_path, "data/pop_colors.csv"))
    pop_labels = r("pop_labels", os.path.join(output_path, "data/pop_labels.csv"))

    # --- filename tags (must match Snakefile exactly) -----------------------
    RATES_TAG = f".minWS{min_window_size}.maxWS{max_window_size}.V{version}.FS{fraction_samples}.maxT{max_time}.nT{n_timepoints}"
    STT_TAG = f".d{n_components}.Nmin{N_min}.Nmax{N_max}.k{k}.link{link}.L1[{l1}].L2[{l2}].nTe{n_timepoints_emb}" + RATES_TAG

    def resolve(p):
        return p if os.path.isabs(p) else os.path.normpath(os.path.join(base_dir, p))

    data = resolve(os.path.join(output_path, "data"))
    emb = os.path.join(data, f"embeddings{STT_TAG}.parquet")
    rates = os.path.join(data, f"rates{RATES_TAG}.parquet")
    recon = os.path.join(data, f"reconstructed_rates{STT_TAG}.parquet")
    cov = os.path.join(data, f"covariances{STT_TAG}.parquet")

    # default output matches the Snakefile `bundle_dash` rule: plots/dash_bundle<TAG>.zip
    plots = resolve(os.path.join(output_path, "plots"))
    return dict(
        embeddings=emb,
        labels=resolve(pop_labels),
        colors=resolve(pop_colors),
        rates=rates if os.path.exists(rates) else None,
        reconstructed_rates=recon if os.path.exists(recon) else None,
        covariances=cov if os.path.exists(cov) else None,
        N_max=N_max,
        time_scale=time_scale,
        min_time=min_time,
        max_time=embeddings_max_time,
        lims=None if str(embeddings_plot_lims) == "None" else embeddings_plot_lims,
        generation_time=None,
        output=os.path.join(plots, f"dash_bundle{STT_TAG}.zip"),
    )


def build_bundle(embeddings, labels, colors, output, rates=None,
                 reconstructed_rates=None, covariances=None, N_max=None,
                 time_scale="log", min_time=None, max_time=None, lims=None,
                 generation_time=None, force_rates_export=False,
                 max_rates_mb=DEFAULT_MAX_RATES_MB, rates_time_stride="auto"):
    """Assemble the self-contained zip from resolved paths and params."""
    # (dest-basename, source-path, dash-flag-or-None-for-positional)
    items = [("embeddings.parquet", embeddings, None),
             ("pop_labels.csv", labels, "--labels"),
             ("pop_colors.csv", colors, "--colors")]

    # The rate parquets dominate bundle size (they grow ~quadratically in the
    # number of samples). First thin their timepoint columns (see _thin_rate_parquet
    # / 'auto'), then skip them entirely when the thinned combined size still
    # exceeds the limit, unless forced. Without them the bundle still has the
    # interactive embedding; only the input-vs-reconstructed rates panel is lost.
    rate_files = [("rates.parquet", rates, "--rates"),
                  ("reconstructed_rates.parquet", reconstructed_rates, "--reconstructed_rates")]
    emb_n_tp = _emb_timepoint_count(embeddings) if rates_time_stride == "auto" else None
    rate_strides = {dest: _resolve_stride(rates_time_stride, s, emb_n_tp)
                    for dest, s, _ in rate_files if s and os.path.exists(s)}
    # Estimate thinned size (~ original / stride) for the gate without reading data.
    rate_bytes = sum(os.path.getsize(s) / rate_strides[dest]
                     for dest, s, _ in rate_files if s and os.path.exists(s))
    include_rates = force_rates_export or rate_bytes <= max_rates_mb * 1e6
    if rate_bytes and not include_rates:
        thinned = any(v > 1 for v in rate_strides.values())
        note = "even after thinning " if thinned else ""
        print(f"WARNING: rate parquets total ~{rate_bytes / 1e6:.0f} MB {note}"
              f"(estimated), exceeding the {max_rates_mb:.0f} MB limit — EXCLUDING them "
              f"from the bundle (the dash rates panel will be unavailable). Increase "
              f"--rates-time-stride, raise --max-rates-mb, or pass --force-rates-export.")
    optional = (rate_files if include_rates else []) + [("covariances.parquet", covariances, "--covariances")]
    for dest, src, flag in optional:
        if src:
            items.append((dest, src, flag))

    for dest, src, _ in items:
        if not os.path.exists(src):
            raise FileNotFoundError(f"input not found: {src}")
    if not os.path.exists(DASH_APP):
        raise FileNotFoundError(f"dash app not found: {DASH_APP}")

    bundle_name = os.path.splitext(os.path.basename(output))[0]
    with tempfile.TemporaryDirectory() as tmp:
        root = os.path.join(tmp, bundle_name)
        data_dir = os.path.join(root, "data")
        os.makedirs(data_dir)

        shutil.copy2(DASH_APP, os.path.join(root, "plot_embeddings_dash.py"))
        for dest, src, _ in items:
            out = os.path.join(data_dir, dest)
            stride = rate_strides.get(dest, 1)
            if stride > 1:
                _thin_rate_parquet(src, out, stride)
                print(f"Thinned {dest}: stride {stride}, "
                      f"{os.path.getsize(src) / 1e6:.0f} -> {os.path.getsize(out) / 1e6:.0f} MB")
            else:
                shutil.copy2(src, out)

        # build the run command (paths relative to bundle root)
        cmd = ["python plot_embeddings_dash.py data/embeddings.parquet",
               "--labels data/pop_labels.csv", "--colors data/pop_colors.csv"]
        for dest, src, flag in items:
            if flag in ("--rates", "--reconstructed_rates", "--covariances"):
                cmd.append(f"{flag} data/{dest}")
        for flag, val in [("--N_max", N_max), ("--time_scale", time_scale),
                          ("--min_time", min_time), ("--max_time", max_time),
                          ("--lims", lims), ("--generation_time", generation_time)]:
            if val is not None:
                cmd.append(f"{flag} {val}")
        run_cmd = " \\\n    ".join(cmd)

        with open(os.path.join(root, "run.sh"), "w") as f:
            f.write("#!/bin/bash\n# Launch the CoaSTT embedding dash app. Open the printed http://127.0.0.1:<port> URL.\n")
            f.write('cd "$(dirname "$0")"\n')
            f.write(run_cmd + "\n")
        os.chmod(os.path.join(root, "run.sh"), 0o755)

        # minimal conda env pinned to the versions used to build this bundle
        with open(os.path.join(root, "environment.yml"), "w") as f:
            f.write(_environment_yml())

        with open(os.path.join(root, "README.md"), "w") as f:
            f.write(textwrap.dedent(f"""\
                # CoaSTT embedding — interactive dash bundle

                Self-contained interactive CoaSTT embedding.

                ## Environment (recommended: conda)
                A pinned `environment.yml` is included (only the few deps the viewer
                needs — not the full CoaSTT pipeline env):
                ```
                conda env create -f environment.yml
                conda activate stt_dash
                ```
                Or with pip into any Python 3.10+ env:
                ```
                pip install dash plotly pandas numpy scipy pyarrow
                ```

                ## Run
                ```
                ./run.sh
                ```
                Then open the `http://127.0.0.1:<port>` URL it prints (auto-picks a
                free port from 8050). The slider scrubs through time; the embedding
                panel shows each individual's position, and the rates panel compares
                input vs reconstructed coalescent rates.

                ## Contents
                - `plot_embeddings_dash.py` — the app
                - `environment.yml` — pinned conda env (`stt_dash`)
                - `data/` — {", ".join(dest for dest, _, _ in items)}
                - `run.sh` — launch command (time_scale={time_scale})
                """))

        os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as z:
            for dirpath, _, filenames in os.walk(root):
                for fn in filenames:
                    full = os.path.join(dirpath, fn)
                    z.write(full, os.path.relpath(full, tmp))

    size_mb = os.path.getsize(output) / 1e6
    print(f"Wrote {output} ({size_mb:.1f} MB)")
    print(f"Bundle root: {bundle_name}/  (unzip, then ./run.sh)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("embeddings", nargs="?", default=None, help="Path to embeddings parquet (omit when using --config)")
    p.add_argument("--config", default=None, help="snakemake config yaml to derive all paths/params from")
    p.add_argument("--base", default=THIS_DIR, help="base dir for resolving config-relative paths (default: utils/)")
    p.add_argument("--labels", default=None, help="pop_labels CSV")
    p.add_argument("--colors", default=None, help="pop_colors CSV")
    p.add_argument("--rates", default=None, help="input rates parquet")
    p.add_argument("--reconstructed_rates", default=None, help="reconstructed rates parquet")
    p.add_argument("--covariances", default=None, help="covariances parquet (optional)")
    p.add_argument("--N_max", type=float, default=None)
    p.add_argument("--time_scale", default=None, choices=["log", "linear"])
    p.add_argument("--min_time", type=float, default=None)
    p.add_argument("--max_time", type=float, default=None)
    p.add_argument("--lims", default=None)
    p.add_argument("--generation_time", type=float, default=None)
    p.add_argument("--force-rates-export", action="store_true",
                   help=f"Include the rate parquets even if they exceed the {DEFAULT_MAX_RATES_MB} MB limit")
    p.add_argument("--max-rates-mb", type=float, default=DEFAULT_MAX_RATES_MB,
                   help=f"Combined rate-parquet size (MB) above which they are excluded unless forced (default: {DEFAULT_MAX_RATES_MB})")

    def _stride_arg(s):
        if str(s).lower() == "auto":
            return "auto"
        v = int(s)
        if v < 1:
            raise argparse.ArgumentTypeError("must be an integer >= 1 or 'auto'")
        return v
    p.add_argument("--rates-time-stride", type=_stride_arg, default="auto",
                   help="Subsample rate-parquet timepoint COLUMNS to shrink the bundle: keep "
                        "every Nth timepoint (1 = no thinning). 'auto' (default) picks a per-file "
                        "stride so the kept count ~= the embeddings timepoint count (near-lossless).")
    p.add_argument("-o", "--output", default=None, help="Output .zip path (default: <output_path>/plots/dash_bundle<TAG>.zip in --config mode)")
    args = p.parse_args()

    # Start from config-derived values (if any), then let explicit CLI flags win.
    vals = derive_from_config(args.config, args.base) if args.config else {}
    explicit = {k: v for k, v in dict(
        embeddings=args.embeddings, labels=args.labels, colors=args.colors,
        rates=args.rates, reconstructed_rates=args.reconstructed_rates,
        covariances=args.covariances, N_max=args.N_max, time_scale=args.time_scale,
        min_time=args.min_time, max_time=args.max_time, lims=args.lims,
        generation_time=args.generation_time, output=args.output,
    ).items() if v is not None}
    vals.update(explicit)
    vals.setdefault("time_scale", "log")
    vals["force_rates_export"] = args.force_rates_export
    vals["max_rates_mb"] = args.max_rates_mb
    vals["rates_time_stride"] = args.rates_time_stride

    missing = [k for k in ("embeddings", "labels", "colors", "output") if not vals.get(k)]
    if missing:
        p.error(f"missing required value(s): {', '.join(missing)} "
                f"(provide via --config or explicit flags)")

    build_bundle(**vals)


if __name__ == "__main__":
    main()
