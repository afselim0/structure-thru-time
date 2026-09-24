#!/usr/bin/env python3
"""Pin bundle_dash.derive_from_config's tag construction to utils/snakefile's.

bundle_dash.py rebuilds PDF_TAG/RATES_TAG/STT_TAG by hand from the config yaml,
because it runs outside snakemake and has no access to the workflow's variables.
That duplication has drifted before -- it carried a stale ``.k{k}`` where the
snakefile writes ``.p{p}``, and a hardcoded ``n_timepoints_emb`` default of 400
against the snakefile's "every rates timepoint" -- which silently pointed the dash
bundle at files that no run would ever produce.

This test extracts the tags the snakefile would build for a real config and
compares them to the ones bundle_dash derives. It parses the snakefile's config
section rather than importing it, since the file is a snakemake workflow and not
an importable module.

Run with pytest, or directly (``python test_bundle_dash_tags.py``).
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO_ROOT, "utils", "snakemake_utils"))

SNAKEFILE = os.path.join(REPO_ROOT, "utils", "snakefile")
CONFIGS = [
    os.path.join(REPO_ROOT, "data", "configs", "sim_configs", "complex_4_pops.yaml"),
    os.path.join(REPO_ROOT, "data", "configs", "app_configs", "chimps_recent.yaml"),
]


def _snakefile_tags(config_path):
    """Evaluate the snakefile's config + tag block against one config yaml.

    Executes the file up to the tag definitions in a namespace holding only
    ``config`` and ``os``, stopping at the first line past the tags so no rule or
    filesystem access is triggered.
    """
    import yaml

    with open(config_path) as fh:
        config = yaml.safe_load(fh) or {}

    src = open(SNAKEFILE).read().splitlines(keepends=True)
    end = next(i for i, l in enumerate(src) if l.startswith("STT_TAG_NO_D"))
    block = "".join(src[: end + 1])

    ns = {"config": config, "os": os}
    exec(compile(block, SNAKEFILE, "exec"), ns)
    return {k: ns[k] for k in ("PDF_TAG", "RATES_TAG", "STT_TAG", "SMOOTH_TAG")}


def _bundle_dash_tags(config_path):
    """The same three tags, as bundle_dash reconstructs them.

    derive_from_config returns paths rather than tags, so the tags are recovered
    from the filenames it builds -- which is exactly the thing that has to match.
    """
    from bundle_dash import derive_from_config

    d = derive_from_config(config_path, base_dir=os.path.join(REPO_ROOT, "utils"))
    emb = os.path.basename(d["embeddings"]).removeprefix("smooth_")
    stt = emb[len("embeddings"):-len(".parquet")]
    rates = os.path.basename(d["output"])  # dash_bundle[_smooth]<STT_TAG>.zip
    assert rates.startswith("dash_bundle") and rates.endswith(".zip")
    return stt


def test_smoothed_runs_bundle_the_smooth_files():
    """With l1 set, the snakefile writes smooth_embeddings / smooth_reconstructed_rates."""
    import yaml
    from bundle_dash import derive_from_config

    for config_path in CONFIGS:
        cfg = yaml.safe_load(open(config_path)) or {}
        smoothing_on = str(cfg.get("l1", "None")) != "None"
        d = derive_from_config(config_path, base_dir=os.path.join(REPO_ROOT, "utils"))
        for key in ("embeddings", "output"):
            name = os.path.basename(d[key])
            assert (name.startswith("smooth_") or "_smooth." in name) == smoothing_on, (config_path, name)


def test_stt_tag_matches_the_snakefile():
    """bundle_dash bundles the embeddings a run actually produced.

    The snakefile keeps two parallel tags: STT_TAG is penalty-free (it tags the
    unsmoothed embeddings, always computed), and SMOOTH_TAG carries the real L1
    penalties. bundle_dash builds whichever the config asks for, so the comparison
    picks the matching one.
    """
    import yaml

    for config_path in CONFIGS:
        cfg = yaml.safe_load(open(config_path)) or {}
        smoothing_on = str(cfg.get("l1", "None")) != "None"
        tags = _snakefile_tags(config_path)
        expected = tags["SMOOTH_TAG"] if smoothing_on else tags["STT_TAG"]
        actual = _bundle_dash_tags(config_path)
        assert actual == expected, (
            f"{os.path.basename(config_path)} (smoothing_on={smoothing_on}):\n"
            f"  snakefile:   {expected}\n"
            f"  bundle_dash: {actual}"
        )


def test_rates_tag_is_a_suffix_of_the_stt_tag():
    """The nesting PDF_TAG in RATES_TAG in STT_TAG is what makes one pdf serve many runs."""
    for config_path in CONFIGS:
        tags = _snakefile_tags(config_path)
        assert tags["STT_TAG"].endswith(tags["RATES_TAG"]), config_path
        assert tags["RATES_TAG"].endswith(tags["PDF_TAG"]), config_path


def test_pdf_tag_carries_the_three_grid_parameters():
    for config_path in CONFIGS:
        pdf_tag = _snakefile_tags(config_path)["PDF_TAG"]
        for token in ("pdf_minT", "pdf_maxT", "pdf_bin"):
            assert token in pdf_tag, f"{token} missing from {pdf_tag}"


def test_rates_tag_carries_the_rate_interval():
    for config_path in CONFIGS:
        rates_tag = _snakefile_tags(config_path)["RATES_TAG"]
        for token in ("rates_minT", "rates_maxT", "minWS", "maxWS"):
            assert token in rates_tag, f"{token} missing from {rates_tag}"


if __name__ == "__main__":
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failures = []
    for name, fn in tests:
        try:
            fn()
        except Exception as err:                       # noqa: BLE001 - reporting harness
            failures.append((name, err))
            print(f"FAIL  {name}: {type(err).__name__}: {err}")
        else:
            print(f"pass  {name}")
    print(f"\n{len(tests) - len(failures)}/{len(tests)} passed")
    sys.exit(1 if failures else 0)
