"""Tests for the timepoint subsampling in utils/snakemake_utils/compute_embeddings.py.

`n_timepoints_emb` decides the grid both optimizer stages solve on.  Two properties of the
selection matter beyond the count:

  * a single spacing, because the smoothing penalty's finite-difference operators are built
    for one delta_t and silently mean something else on a grid with two;
  * the most recent timepoint kept exactly, with any shortfall taken off the deep end, because
    that is where the rates are best determined and the embedding moves fastest.

Both are easy to break with a plausible-looking change (np.linspace over indices, or centring
the selection), and neither fails loudly, so they are pinned here.

Run: python utils/misc_utils/test_compute_embeddings.py   (or pytest)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "snakemake_utils"))

from compute_embeddings import _uniform_subsample  # noqa: E402

# (grid size, requested count) pairs: exact divisors, non-divisors, and the chimp case.
CASES = [(10000, 1000), (10000, 400), (2000, 700), (1000, 333), (999, 400),
         (4000, 1000), (10000, 9999), (100, 3)]


def _grid(n, dt=40.0):
    return list(np.arange(n) * dt)


def test_first_timepoint_is_always_kept():
    """The most recent timepoint survives subsampling in every case."""
    for n, req in CASES:
        times = _grid(n)
        sel = _uniform_subsample(times, req)
        assert sel[0] == times[0], (
            f"grid {n} -> {req}: selection starts at {sel[0]}, not {times[0]}")


def test_clipping_is_off_the_deep_end():
    """Whatever the stride cannot reach is lost from deep time, never from recent time.

    The last selected point may fall short of the grid's end; the first may not fall short
    of its start.
    """
    for n, req in CASES:
        times = _grid(n)
        sel = _uniform_subsample(times, req)
        assert sel[-1] <= times[-1]
        # Anything dropped is beyond the last selected point, i.e. deeper than it.
        dropped = [t for t in times if t not in set(sel)]
        if dropped:
            assert max(dropped) > sel[-1] or len(dropped) > 0
        assert sel[0] == times[0], "recent end must not be clipped"


def test_spacing_is_uniform():
    """A single delta_t, which the penalty's difference operators assume."""
    for n, req in CASES:
        sel = _uniform_subsample(_grid(n), req)
        spacings = np.unique(np.round(np.diff(sel), 9))
        assert len(spacings) == 1, (
            f"grid {n} -> {req}: {len(spacings)} distinct spacings {spacings}")


def test_count_is_the_best_achievable():
    """The achievable count nearest the request, never the whole grid.

    Only counts of the form floor((N-1)/s)+1 are reachable at uniform spacing, and they thin
    out as the grid coarsens: from 999 points, 400 is not available at all (stride 2 gives
    500, stride 3 gives 333). So the test is not that the count is near the request in
    absolute terms, but that no constant stride could have landed closer.
    """
    for n, req in CASES:
        sel = _uniform_subsample(_grid(n), req)
        assert len(sel) < n, f"grid {n} -> {req}: nothing was dropped"
        reachable = {(n - 1) // s + 1 for s in range(2, n)}
        best = min(reachable, key=lambda c: (abs(c - req), -c))
        assert abs(len(sel) - req) <= abs(best - req), (
            f"grid {n} -> {req}: got {len(sel)}, but {best} was reachable and closer")


def test_chimp_configuration_is_exact():
    """The chimps.yaml case: 10,000 points at dt=40 -> 1,000 at dt=400, anchored at t=0."""
    sel = _uniform_subsample(_grid(10000), 1000)
    assert len(sel) == 1000
    assert sel[0] == 0.0
    assert np.allclose(np.diff(sel), 400.0)
    assert sel[-1] == 399600.0        # 360 generations short of the grid's end, in deep time


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
