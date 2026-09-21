#!/usr/bin/env python3
"""Tests for the rate-interval slicing in src/ComputeCoalescentRates.py.

Run with pytest, or directly (``python test_compute_rates_slicing.py``) in
environments where pytest is not installed -- the file carries a small runner at
the bottom that executes every ``test_*`` function and reports failures.

The refactor that introduced ``slice_pdf_for_rates`` also made the pdf grid
inclusive of its endpoint, which changes the number of bins and therefore the
whole bandwidth schedule. So these tests deliberately do NOT compare against
previously saved rates -- those values are expected to move. They pin the two
properties the padding is actually responsible for:

1. Discarded boundary rows never influence kept rows (slice invariance).
2. The kept endpoints carry the bandwidths the config asked for.
"""

import io
import os
import sys
from contextlib import redirect_stdout

import numpy as np
import pandas as pd

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
)

from ComputeCoalescentRates import (  # noqa: E402
    _odd_window_bins,
    adaptive_local_linear_filter,
    compute_coal_rate_weights,
    compute_coal_rates,
    slice_pdf_for_rates,
)

# (time_step, min_window_size, max_window_size, kept_bins) -- the grids of the
# shipped configs, plus a short interval where the rejected full-window pad was
# worst (43 bins where 51 were asked for).
CASES = [
    (40.0, 250.0, 4000.0, 400),   # chimps.yaml
    (4.0, 40.0, 200.0, 400),      # complex_4_pops.yaml
    (4.0, 40.0, 200.0, 100),      # short interval
    (2.0, 100.0, 200.0, 500),     # two_admix.yaml
    (10.0, 80.0, 1500.0, 1000),   # sawtooth.yaml
    (5.0, 100.0, 200.0, 300),     # split.yaml
]


def _support_width(row):
    """Number of bins the filter row spans.

    Counting nonzeros would undercount by one: the centre weight of a symmetric
    window is exactly zero because its abscissa is zero.
    """
    nz = np.nonzero(row)[0]
    return int(nz[-1] - nz[0] + 1)


def _pdf_on(times, n_pairs=6, seed=0):
    """A normalised pdf on the given grid, with survival kept strictly positive."""
    rng = np.random.default_rng(seed)
    mass = rng.random((n_pairs, len(times))) + 0.05
    mass /= mass.sum(axis=1, keepdims=True) * 1.4
    index = pd.MultiIndex.from_tuples(
        [(2 * i, 2 * i + 1) for i in range(n_pairs)], names=["sample_1", "sample_2"]
    )
    return pd.DataFrame(mass, columns=times, index=index)


def _grid_for(time_step, min_ws, max_ws, kept):
    """A pdf grid long enough to supply both pads in full, and the kept interval."""
    min_bins = _odd_window_bins(min_ws, time_step)
    max_bins = max(_odd_window_bins(max_ws, time_step), min_bins)
    n = min_bins // 2 + kept + max_bins // 2 + 1
    times = np.arange(n) * time_step
    return times, min_bins, max_bins, times[min_bins // 2], times[min_bins // 2 + kept - 1]


def test_kept_endpoints_get_the_requested_bandwidths():
    """min_window_size applies AT rates_min_time and max_window_size AT rates_max_time.

    This is the property that motivates padding by max_bins // 2 rather than the
    full max_bins: with a full-window pad the ramp is still climbing at the last
    kept row, landing e.g. at 91 bins where 101 were asked for.
    """
    for time_step, min_ws, max_ws, kept in CASES:
        times, min_bins, max_bins, rmin, rmax = _grid_for(time_step, min_ws, max_ws, kept)
        padded, keep_mask = slice_pdf_for_rates(
            _pdf_on(times), rmin, rmax, min_ws, max_ws)
        assert keep_mask.sum() == kept, (time_step, min_ws, max_ws, kept)

        S = adaptive_local_linear_filter(min_ws, max_ws, padded.shape[1], time_step)
        rows = np.flatnonzero(keep_mask)
        first, last = _support_width(S[rows[0]]), _support_width(S[rows[-1]])
        assert first == min_bins, f"first kept row {first} != min_bins {min_bins} for {(time_step, min_ws, max_ws, kept)}"
        assert last == max_bins, f"last kept row {last} != max_bins {max_bins} for {(time_step, min_ws, max_ws, kept)}"


def test_sufficient_padding_reports_no_shortfall():
    """A pdf sized for both pads must trigger neither the note nor the warning."""
    for time_step, min_ws, max_ws, kept in CASES:
        times, _, _, rmin, rmax = _grid_for(time_step, min_ws, max_ws, kept)
        buf = io.StringIO()
        with redirect_stdout(buf):
            slice_pdf_for_rates(_pdf_on(times), rmin, rmax, min_ws, max_ws)
        assert "short of" not in buf.getvalue(), buf.getvalue()


def test_slice_invariance_under_a_constant_bandwidth():
    """Padded-then-clipped rates equal full-grid rates, to machine precision.

    With min_window_size == max_window_size the bandwidth no longer depends on the
    grid length, so the two are directly comparable. This isolates exactly what
    padding is responsible for: that the rows thrown away never influenced the
    rows kept. (With a ramped bandwidth the schedule is anchored to the interval
    by design, so the two SHOULD differ and this comparison would be meaningless.)
    """
    time_step, ws, n = 4.0, 200.0, 1001
    times = np.arange(n) * time_step
    pdf = _pdf_on(times, n_pairs=8, seed=7)

    full = compute_coal_rates(pdf, ws, ws, per_haplotype=True)

    lo, hi = 200, 600
    padded, keep_mask = slice_pdf_for_rates(pdf, times[lo], times[hi], ws, ws)
    sliced = compute_coal_rates(padded, ws, ws, per_haplotype=True).loc[:, keep_mask]

    assert list(sliced.columns) == list(times[lo:hi + 1])
    diff = np.abs(sliced.values - full.values[:, lo:hi + 1]).max()
    assert diff < 1e-12, f"slice invariance broken: max abs diff {diff:.3e}"


def test_slice_carries_the_survival_baseline():
    """1 - cumsum on the slice must equal the true survival, not restart from 1.

    A plain slice would report every pair as uncoalesced at the start of the slice.
    The inflation does not cancel in the hazard -d/dt log S, because it differs per
    pair and across time within a pair, so it would corrupt every rate.
    """
    times = np.arange(600) * 4.0
    pdf = _pdf_on(times, n_pairs=6, seed=11)
    true_survival = (1 - pdf.cumsum(axis=1)).values

    padded, _ = slice_pdf_for_rates(pdf, times[200], times[400], 40.0, 200.0)
    start = int(np.flatnonzero(np.isclose(times, padded.columns[0]))[0])
    assert start > 0, "test needs a slice that actually drops a prefix"

    sliced_survival = (1 - padded.cumsum(axis=1)).values
    diff = np.abs(sliced_survival - true_survival[:, start:start + padded.shape[1]]).max()
    assert diff < 1e-12, f"survival baseline lost: max abs diff {diff:.3e}"


def test_slice_invariance_holds_for_an_interior_interval():
    """Slice invariance with a slice that drops a prefix, not just a suffix.

    Complements test_slice_invariance_under_a_constant_bandwidth by starting the
    kept region far enough in that the survival baseline matters.
    """
    time_step, ws = 4.0, 200.0
    times = np.arange(1001) * time_step
    pdf = _pdf_on(times, n_pairs=6, seed=13)

    full = compute_coal_rates(pdf, ws, ws, per_haplotype=True)
    lo, hi = 400, 800
    padded, keep_mask = slice_pdf_for_rates(pdf, times[lo], times[hi], ws, ws)
    sliced = compute_coal_rates(padded, ws, ws, per_haplotype=True).loc[:, keep_mask]

    diff = np.abs(sliced.values - full.values[:, lo:hi + 1]).max()
    assert diff < 1e-12, f"slice invariance broken: max abs diff {diff:.3e}"


def test_rates_and_weights_are_clipped_to_the_same_columns():
    """The loss pairs a rate with its weight by position; divergence must be impossible."""
    time_step, min_ws, max_ws = 4.0, 40.0, 200.0
    times = np.arange(700) * time_step
    pdf = _pdf_on(times, n_pairs=6, seed=3)

    padded, keep_mask = slice_pdf_for_rates(pdf, times[100], times[500], min_ws, max_ws)
    S = adaptive_local_linear_filter(min_ws, max_ws, padded.shape[1], time_step)

    rates = compute_coal_rates(padded, min_ws, max_ws, per_haplotype=True,
                               filter_matrix=S).loc[:, keep_mask]
    weights = compute_coal_rate_weights(padded, min_ws, max_ws, per_haplotype=True,
                                        filter_matrix=S).loc[:, keep_mask]

    assert rates.columns.equals(weights.columns)
    assert list(rates.columns) == list(times[100:501])
    assert (weights.values > 0).all()


def test_shortfall_is_reported_but_not_fatal():
    """A pdf too short for the old pad still returns, with a named warning."""
    time_step, min_ws, max_ws = 4.0, 40.0, 200.0
    times = np.arange(300) * time_step
    pdf = _pdf_on(times)

    buf = io.StringIO()
    with redirect_stdout(buf):
        padded, keep_mask = slice_pdf_for_rates(pdf, times[50], times[-1], min_ws, max_ws)
    out = buf.getvalue()

    assert "short of" in out and "rates_max_time" in out, out
    assert padded.shape[1] == keep_mask.size
    assert keep_mask.sum() == len(times) - 50


def test_rates_from_the_pdf_start_report_a_note_not_a_warning():
    """The default rates_min_time == pdf_min_time case is not a misconfiguration."""
    time_step, min_ws, max_ws = 4.0, 40.0, 200.0
    times = np.arange(600) * time_step

    buf = io.StringIO()
    with redirect_stdout(buf):
        slice_pdf_for_rates(_pdf_on(times), times[0], times[400], min_ws, max_ws)
    out = buf.getvalue()

    assert "Note:" in out and "rates_min_time" in out, out
    assert "Warning" not in out, out


def test_degenerate_interval_is_rejected():
    times = np.arange(200) * 4.0
    try:
        slice_pdf_for_rates(_pdf_on(times), times[50], times[50], 40.0, 200.0)
    except ValueError as err:
        assert "at least two" in str(err), err
    else:
        raise AssertionError("a single-column interval must raise")


def test_non_uniform_grid_is_rejected():
    times = np.array([0.0, 4.0, 8.0, 20.0, 24.0])
    try:
        slice_pdf_for_rates(_pdf_on(times), 0.0, 24.0, 4.0, 8.0)
    except ValueError as err:
        assert "uniform" in str(err), err
    else:
        raise AssertionError("a non-uniform grid must raise")


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
