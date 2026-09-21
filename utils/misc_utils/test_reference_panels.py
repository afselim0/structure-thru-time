#!/usr/bin/env python3
"""Tests for reference-panel resolution in utils/snakemake_utils/infer_local_ancestry.py.

Run with pytest, or directly (``python test_reference_panels.py``) in environments
where pytest is not installed -- the file carries a small runner at the bottom that
executes every ``test_*`` function and reports failures.

``resolve_reference_panels`` is the one place where a reference population stops
being a name and becomes a set of samples, so it carries the whole behavioural
contract of the composite feature:

* the historical forms (None default, plain list) keep their exact meaning, and
* a composite group takes the first N *diploid individuals* per source
  population -- both haplotypes of each -- so panels can be size-matched.

The synthetic pop_labels below is deliberately shuffled and non-contiguous in
sample_id, so a test that passes by accident of row order will fail here.
"""

import os
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "snakemake_utils")
)

from infer_local_ancestry import (  # noqa: E402
    compute_total_ancestry_proportions,
    resolve_reference_panels,
    write_reference_panels_manifest,
)

# population -> number of diploid individuals. Mirrors the chimp dataset's shape
# (Eastern 19 / Central 15 / Western 14 / NC 5 admixed) at a smaller size.
POPS = {"Eastern": 8, "Central": 6, "Western": 7, "NC": 3}
ADMIX = ["NC"]


def _fake_pop_labels(seed=0):
    """Two haploid rows per individual, rows shuffled, sample_ids non-contiguous.

    sample_id deliberately does NOT start at 0 and skips values, so any code that
    conflates sample_id with individual_id, or relies on row order, breaks.
    """
    rows = []
    individual_id = 0
    sample_id = 1000
    for pop, n_ind in POPS.items():
        for _ in range(n_ind):
            for _hap in range(2):
                rows.append({
                    "sample_id": sample_id,
                    "population": pop,
                    "individual_id": individual_id,
                    "sample_name": f"{pop}_{individual_id}",
                })
                sample_id += 3          # non-contiguous
            individual_id += 1
    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def _legacy_panels(pop_labels, reference_pops):
    """The pre-refactor mapping, verbatim, as the back-compat oracle."""
    return {
        pop: sorted(int(s) for s in
                    pop_labels[pop_labels["population"] == pop]["sample_id"].values)
        for pop in reference_pops
    }


# --------------------------------------------------------------------------- #
#  Back-compat: the two historical forms must not move
# --------------------------------------------------------------------------- #

def test_default_none_matches_legacy():
    """reference_spec=None must reproduce the old comprehension exactly."""
    pl = _fake_pop_labels()
    got = resolve_reference_panels(pl, None, ADMIX)
    expected_pops = [p for p in pl["population"].unique() if p not in ADMIX]
    assert set(got) == set(expected_pops), got
    assert got == _legacy_panels(pl, list(got))


def test_plain_list_form_matches_legacy():
    pl = _fake_pop_labels()
    spec = {"Eastern": None, "Western": None}
    got = resolve_reference_panels(pl, spec, ADMIX)
    assert got == _legacy_panels(pl, ["Eastern", "Western"])
    assert len(got["Eastern"]) == 2 * POPS["Eastern"]


def test_default_respects_pops_filter():
    pl = _fake_pop_labels()
    got = resolve_reference_panels(pl, None, ADMIX, pops=["Eastern", "Central", "NC"])
    assert set(got) == {"Eastern", "Central"}, got


def test_default_excludes_admixed():
    pl = _fake_pop_labels()
    got = resolve_reference_panels(pl, None, ADMIX)
    assert "NC" not in got


# --------------------------------------------------------------------------- #
#  Composite groups
# --------------------------------------------------------------------------- #

def test_composite_counts_individuals_not_haplotypes():
    """5 + 5 individuals -> 20 haplotypes, 10 individuals. The core contract."""
    pl = _fake_pop_labels()
    # The chimp case in miniature: 5+5 from two populations against 7 from one.
    # (The real config asks 10 from Western; this fixture only has 7.)
    spec = {"EastCentral": {"Eastern": 5, "Central": 5}, "West": {"Western": 7}}
    got = resolve_reference_panels(pl, spec, ADMIX)

    assert len(got["EastCentral"]) == 20, len(got["EastCentral"])
    ind = pl[pl["sample_id"].isin(got["EastCentral"])]["individual_id"].nunique()
    assert ind == 10, ind
    assert len(got["West"]) == 14


def test_composite_balances_panels():
    """The whole point: unequal source populations, equal panels."""
    pl = _fake_pop_labels()
    spec = {"A": {"Eastern": 3, "Central": 3}, "B": {"Western": 6}}
    got = resolve_reference_panels(pl, spec, ADMIX)
    assert len(got["A"]) == len(got["B"]) == 12


def test_composite_takes_lowest_individual_ids():
    pl = _fake_pop_labels()
    got = resolve_reference_panels(
        pl, {"A": {"Eastern": 3}, "B": {"Western": 3}}, ADMIX)

    eastern_ind = np.sort(pl[pl["population"] == "Eastern"]["individual_id"].unique())
    chosen = set(pl[pl["sample_id"].isin(got["A"])]["individual_id"])
    assert chosen == set(eastern_ind[:3]), (chosen, eastern_ind[:3])


def test_composite_takes_both_haplotypes_of_each_individual():
    pl = _fake_pop_labels()
    got = resolve_reference_panels(pl, {"A": {"Eastern": 4}, "B": {"Western": 4}}, ADMIX)
    counts = pl[pl["sample_id"].isin(got["A"])].groupby("individual_id").size()
    assert (counts == 2).all(), counts.to_dict()


def test_composite_is_deterministic():
    pl = _fake_pop_labels()
    spec = {"A": {"Eastern": 5, "Central": 2}, "B": {"Western": 7}}
    assert resolve_reference_panels(pl, spec, ADMIX) == resolve_reference_panels(pl, spec, ADMIX)


def test_composite_independent_of_row_order():
    """Two different shuffles of the same table must give the same panels."""
    spec = {"A": {"Eastern": 5, "Central": 3}, "B": {"Western": 7}}
    a = resolve_reference_panels(_fake_pop_labels(seed=1), spec, ADMIX)
    b = resolve_reference_panels(_fake_pop_labels(seed=2), spec, ADMIX)
    assert a == b


def test_mixed_form():
    """A composite group beside an all-samples group is legal."""
    pl = _fake_pop_labels()
    got = resolve_reference_panels(
        pl, {"EastCentral": {"Eastern": 4, "Central": 4}, "Western": None}, ADMIX)
    assert len(got["EastCentral"]) == 16
    assert len(got["Western"]) == 2 * POPS["Western"]


def test_panels_are_disjoint():
    pl = _fake_pop_labels()
    for spec in (None,
                 {"Eastern": None, "Western": None},
                 {"A": {"Eastern": 4, "Central": 4}, "B": {"Western": 5}}):
        got = resolve_reference_panels(pl, spec, ADMIX)
        seen = [s for ids in got.values() for s in ids]
        assert len(seen) == len(set(seen)), spec


# --------------------------------------------------------------------------- #
#  Rejections
# --------------------------------------------------------------------------- #

def _expect(exc_substr, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except ValueError as err:
        assert exc_substr in str(err), f"wrong message: {err}"
    else:
        raise AssertionError(f"expected ValueError containing {exc_substr!r}")


def test_insufficient_individuals_raises():
    pl = _fake_pop_labels()
    _expect("only 8 are present", resolve_reference_panels,
            pl, {"A": {"Eastern": 999}, "B": {"Western": 2}}, ADMIX)


def test_unknown_source_pop_raises():
    pl = _fake_pop_labels()
    _expect("is not present in pop_labels", resolve_reference_panels,
            pl, {"A": {"Nowhere": 2}, "B": {"Western": 2}}, ADMIX)


def test_admixed_source_raises():
    pl = _fake_pop_labels()
    _expect("admixed_pops", resolve_reference_panels,
            pl, {"A": {"NC": 2}, "B": {"Western": 2}}, ADMIX)


def test_overlapping_sources_raise():
    pl = _fake_pop_labels()
    _expect("must be disjoint", resolve_reference_panels,
            pl, {"A": {"Eastern": 3}, "B": {"Eastern": 3}}, ADMIX)


def test_fewer_than_two_panels_raises():
    pl = _fake_pop_labels()
    _expect("at least 2 reference panels", resolve_reference_panels,
            pl, {"A": {"Eastern": 3}}, ADMIX)


def test_missing_columns_raise():
    pl = _fake_pop_labels().drop(columns=["individual_id"])
    _expect("missing required column", resolve_reference_panels, pl, None, ADMIX)


# --------------------------------------------------------------------------- #
#  Manifest
# --------------------------------------------------------------------------- #

def test_manifest_records_panel_composition():
    pl = _fake_pop_labels()
    panels = resolve_reference_panels(
        pl, {"EastCentral": {"Eastern": 4, "Central": 4}, "West": {"Western": 7}}, ADMIX)
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "reference_panels.csv")
        manifest = write_reference_panels_manifest(panels, pl, path)
        assert os.path.isfile(path)
        reread = pd.read_csv(path)

    assert set(reread["ancestry_label"]) == {"EastCentral", "West"}
    # The balance check a user would actually run.
    per_label = reread.groupby("ancestry_label")["sample_id"].size().to_dict()
    assert per_label == {"EastCentral": 16, "West": 14}, per_label
    sources = set(reread[reread["ancestry_label"] == "EastCentral"]["source_population"])
    assert sources == {"Eastern", "Central"}, sources
    assert len(manifest) == len(reread)


# --------------------------------------------------------------------------- #
#  The proportions guard
# --------------------------------------------------------------------------- #

def _write_lai(d, labels, sample_ids, span=1000.0):
    rows = []
    for sid in sample_ids:
        for i, lbl in enumerate(labels):
            rows.append({"sample_id": sid, "start_pos": i * span,
                         "end_pos": (i + 1) * span, "ancestry_label": lbl})
    pd.DataFrame(rows).to_parquet(os.path.join(d, "contig0.parquet"))


def test_proportions_raise_when_no_label_overlaps():
    """The silent-NaN failure mode: requested labels absent from the LAI calls."""
    with tempfile.TemporaryDirectory() as d:
        _write_lai(d, ["Eastern", "Western"], [1, 2])
        _expect("None of the requested reference labels",
                compute_total_ancestry_proportions, d, [1, 2], ["EastCentral", "West"])


def test_proportions_are_finite_and_sum_to_one():
    with tempfile.TemporaryDirectory() as d:
        _write_lai(d, ["Eastern", "Western"], [1, 2])
        prop = compute_total_ancestry_proportions(d, [1, 2], ["Eastern", "Western"])
    assert list(prop.columns) == ["sample_id", "Eastern_prop", "Western_prop"]
    vals = prop[["Eastern_prop", "Western_prop"]].values
    assert np.isfinite(vals).all(), vals
    assert np.allclose(vals.sum(axis=1), 1.0)


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
