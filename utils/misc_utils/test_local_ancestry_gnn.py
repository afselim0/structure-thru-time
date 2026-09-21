#!/usr/bin/env python3
"""Regression tests for src/LocalAncestryGNN.py on realistically-shaped inputs.

Run with pytest, or directly (``python test_local_ancestry_gnn.py``) in
environments where pytest is not installed -- the file carries a small runner at
the bottom that executes every ``test_*`` function and reports failures.

The case these exist for: a tree sequence whose span is wider than the region it
actually holds edges for. tskit represents the span before the first edge and
after the last as EMPTY trees, and ``keep_intervals(...).trim()`` on one of those
raises "Trimming a tree sequence with no edges would reduce the sequence length to
zero". Every chimp .trees chunk is shaped exactly like this -- a ~2 Mb chunk
carrying the full ~33 Mb chromosome coordinate space, so two empty flanking trees
-- which crashed infer_local_ancestry on its first real run.
"""

import os
import sys
import tempfile

import msprime
import numpy as np
import tskit

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
)

from LocalAncestryGNN import LocalAncestryInference  # noqa: E402


def _ts_with_empty_flanks(seq_len=1000.0, kept=(400.0, 600.0), seed=7):
    """A tree sequence holding edges only in the middle of its coordinate space.

    Mirrors the chimp chunks: delete_intervals leaves the flanks edge-free while
    sequence_length stays the full span, so ts.trees() yields empty trees at both
    ends.
    """
    ts = msprime.sim_ancestry(
        samples=8, sequence_length=seq_len, recombination_rate=2e-4,
        population_size=100, random_seed=seed,
    )
    ts = ts.delete_intervals([[0, kept[0]], [kept[1], seq_len]])
    n_empty = sum(1 for t in ts.trees() if t.num_edges == 0)
    assert n_empty >= 2, f"fixture should have empty flanking trees, got {n_empty}"
    return ts


def _panels(ts):
    """Split the samples into two reference panels plus a set of query samples."""
    samples = list(ts.samples())
    return samples[:4], {"PanelA": samples[4:10], "PanelB": samples[10:]}


def test_empty_flanking_trees_do_not_crash():
    """The bug: keep_intervals().trim() raises on an edge-free tree."""
    ts = _ts_with_empty_flanks()
    admixed, panels = _panels(ts)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "contig.trees")
        ts.dump(path)
        df = LocalAncestryInference(d, admixed, panels).infer_single(path)
    assert len(df) > 0
    assert set(df["sample_id"]) == set(admixed)


def test_calls_cover_only_the_region_with_edges():
    """Skipping empty trees must lose no real span, and invent none."""
    kept = (400.0, 600.0)
    ts = _ts_with_empty_flanks(kept=kept)
    admixed, panels = _panels(ts)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "contig.trees")
        ts.dump(path)
        df = LocalAncestryInference(d, admixed, panels).infer_single(path)

    # No segment may stray outside the region that actually holds edges.
    assert df["start_pos"].min() >= kept[0], df["start_pos"].min()
    assert df["end_pos"].max() <= kept[1], df["end_pos"].max()

    # And per query sample, the calls should tile that region.
    per_sample = df.groupby("sample_id").apply(
        lambda g: (g["end_pos"] - g["start_pos"]).sum(), include_groups=False
    )
    assert np.allclose(per_sample.values, kept[1] - kept[0]), per_sample.to_dict()


def test_labels_are_the_panel_labels():
    ts = _ts_with_empty_flanks()
    admixed, panels = _panels(ts)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "contig.trees")
        ts.dump(path)
        df = LocalAncestryInference(d, admixed, panels).infer_single(path)
    assert set(df["ancestry_label"].dropna()) <= set(panels), set(df["ancestry_label"])
    for label in panels:
        assert f"prob_{label}" in df.columns


def test_fully_empty_tree_sequence_yields_no_rows():
    """A chunk with no edges at all should come back empty, not raise."""
    ts = msprime.sim_ancestry(samples=8, sequence_length=500.0, population_size=100,
                              random_seed=3)
    ts = ts.delete_intervals([[0, 500.0]])
    admixed, panels = _panels(ts)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "contig.trees")
        ts.dump(path)
        df = LocalAncestryInference(d, admixed, panels).infer_single(path)
    assert len(df) == 0, df


def test_ordinary_tree_sequence_is_unaffected():
    """The skip must not change behaviour on a sequence with no empty trees."""
    ts = msprime.sim_ancestry(samples=8, sequence_length=1000.0,
                              recombination_rate=2e-4, population_size=100,
                              random_seed=11)
    assert all(t.num_edges > 0 for t in ts.trees())
    admixed, panels = _panels(ts)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "contig.trees")
        ts.dump(path)
        df = LocalAncestryInference(d, admixed, panels).infer_single(path)
    per_sample = df.groupby("sample_id").apply(
        lambda g: (g["end_pos"] - g["start_pos"]).sum(), include_groups=False
    )
    assert np.allclose(per_sample.values, ts.sequence_length), per_sample.to_dict()


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
