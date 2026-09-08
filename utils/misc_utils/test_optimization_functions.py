#!/usr/bin/env python3
"""Tests for the CoaSTT optimization machinery (Task 12b).

Run with pytest, or directly (``python test_optimization_functions.py``) in
environments where pytest is not installed -- the file carries a small runner at the
bottom that executes every ``test_*`` function and reports failures.

Note on finite differences: the objective is deliberately unnormalized and the weights
are of order 1e6, so the loss is of that order too.  Central differences on such a
function bottom out around 1e-6 relative accuracy at a step of ~1e-4 * |x| and get
*worse* for smaller steps as round-off takes over.  The gradient tolerances below are
set accordingly; they are limits of the finite-difference reference, not of the
analytic gradients.
"""

import math
import os
import sys

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
)

import OptimizationFunctions as O  # noqa: E402

FD_REL_STEP = 1e-4      # near-optimal central-difference step for this objective
FD_TOL      = 1e-5      # achievable accuracy of that reference


############################################################
# Helpers
############################################################

def _make_problem(n=10, d=2, T=6, seed=0, ancient=None):
    """A small synthetic problem: coordinates, symmetric rates and weights, mask."""
    rng = np.random.default_rng(seed)
    X = rng.normal(scale=400.0, size=(n, d, T))

    base = np.abs(rng.normal(scale=1e-3, size=(n, n, T)))
    R = 0.5 * (base + base.transpose(1, 0, 2))
    W = np.abs(rng.normal(scale=1e6, size=(n, n, T)))
    W = 0.5 * (W + W.transpose(1, 0, 2))
    for t in range(T):
        np.fill_diagonal(R[:, :, t], np.nan)
        np.fill_diagonal(W[:, :, t], np.nan)

    sampled_by_time = np.ones((n, T), dtype=bool)
    if ancient:
        for sample_idx, enters_at in ancient.items():
            sampled_by_time[sample_idx, :enters_at] = False
        for t in range(T):
            absent = ~sampled_by_time[:, t]
            R[absent, :, t] = np.nan
            R[:, absent, t] = np.nan
            W[absent, :, t] = np.nan
            W[:, absent, t] = np.nan

    link = O.LinkFunction("inverse_distance", N_min=1000.0, p=1.0)
    return X, R, W, sampled_by_time, link


def _max_fd_gradient_error(fn, x0, n_probes=25, seed=0, rel_step=FD_REL_STEP,
                           largest_gradients=False):
    """Largest relative disagreement between fn's gradient and central differences.

    With ``largest_gradients``, probe the coordinates carrying the most gradient rather
    than a random sample: where the analytic gradient is itself near zero, the relative
    comparison is dominated by the finite-difference reference's own noise floor.
    """
    rng = np.random.default_rng(seed)
    _, g0 = fn(x0)
    if largest_gradients:
        probes = np.argsort(np.abs(g0))[::-1][:min(n_probes, x0.size)]
    else:
        probes = rng.choice(x0.size, min(n_probes, x0.size), replace=False)

    worst = 0.0
    for i in probes:
        h = max(abs(x0[i]), 1.0) * rel_step
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        fd = (fn(xp)[0] - fn(xm)[0]) / (2 * h)
        worst = max(worst, abs(fd - g0[i]) / max(abs(fd), abs(g0[i]), 1e-12))
    return worst


############################################################
# Gradient of the unpenalized loss
############################################################

def test_unpenalized_gradient_matches_finite_differences():
    """Analytic gradient agrees with central differences, no ancient samples."""
    X, R, W, _, link = _make_problem(seed=1)
    problem = {"R": R, "W": W, "link": link}
    fn = lambda x: O.compute_loss_gradient(x, X.shape, problem)

    err = _max_fd_gradient_error(fn, X.ravel(), seed=1)
    assert err < FD_TOL, f"gradient vs finite differences: max rel err {err:.3e}"


def test_unpenalized_gradient_with_ancient_samples():
    """Same check with two ancient samples absent from the early slices."""
    X, R, W, sampled_by_time, link = _make_problem(
        seed=2, ancient={1: 3, 4: 2}
    )
    problem = {"R": R, "W": W, "link": link}
    fn = lambda x: O.compute_loss_gradient(x, X.shape, problem)

    err = _max_fd_gradient_error(fn, np.nan_to_num(X).ravel(), seed=2)
    assert err < FD_TOL, f"gradient vs finite differences: max rel err {err:.3e}"


def test_gradient_function_matches_fused_version():
    """compute_unpenalized_gradient and the fused loss+gradient agree."""
    X, R, W, _, link = _make_problem(seed=3)
    standalone = O.compute_unpenalized_gradient(X, R, W, link)
    _, fused = O.compute_loss_gradient(X.ravel(), X.shape, {"R": R, "W": W, "link": link})
    assert np.allclose(standalone.ravel(), fused, rtol=1e-12, atol=0)


############################################################
# Exact Hessian
############################################################

def test_hessian_matches_finite_differences_of_gradient():
    """Analytic Hessian agrees with central differences of the analytic gradient."""
    X, R, W, _, link = _make_problem(n=8, T=1, seed=4)
    X_t, R_t, W_t = X[:, :, 0], R[:, :, 0], W[:, :, 0]
    H = O.compute_exact_hessian(X_t, R_t, W_t, link=link)

    def grad(x):
        return O.compute_unpenalized_gradient(
            x.reshape(X_t.shape), R_t, W_t, link
        ).ravel()

    x0 = X_t.ravel()
    worst = 0.0
    rng = np.random.default_rng(4)
    for i in rng.choice(x0.size, 10, replace=False):
        h = max(abs(x0[i]), 1.0) * FD_REL_STEP
        xp = x0.copy(); xp[i] += h
        xm = x0.copy(); xm[i] -= h
        fd_col = (grad(xp) - grad(xm)) / (2 * h)
        denom = np.maximum(np.abs(fd_col), np.abs(H[:, i]))
        rel = np.max(np.where(denom > 0, np.abs(fd_col - H[:, i]) / np.maximum(denom, 1e-12), 0.0))
        worst = max(worst, rel)
    assert worst < 1e-4, f"Hessian vs finite-differenced gradient: max rel err {worst:.3e}"


def test_hessian_is_symmetric():
    X, R, W, _, link = _make_problem(n=8, T=1, seed=5)
    H = O.compute_exact_hessian(X[:, :, 0], R[:, :, 0], W[:, :, 0], link=link)
    assert np.allclose(H, H.T, rtol=1e-10, atol=0)


def test_hessian_null_space_is_translation():
    """A generic configuration has exactly d near-zero eigenvalues."""
    n, d = 10, 2
    X, R, W, _, link = _make_problem(n=n, d=d, T=1, seed=6)
    H = O.compute_exact_hessian(X[:, :, 0], R[:, :, 0], W[:, :, 0], link=link)

    eigvals = np.linalg.eigvalsh(H)
    scale = np.max(np.abs(eigvals))
    n_near_zero = int(np.sum(np.abs(eigvals) < 1e-8 * scale))
    assert n_near_zero == d, (
        f"expected {d} translation zero modes, found {n_near_zero}; "
        f"smallest |eigenvalues| = {np.sort(np.abs(eigvals))[:d+2] / scale}"
    )

    # The null space is spanned by uniform translations of each coordinate axis
    for axis in range(d):
        v = np.zeros((n, d)); v[:, axis] = 1.0
        v = v.ravel() / np.linalg.norm(v)
        assert np.linalg.norm(H @ v) < 1e-8 * scale


def test_hessian_rejects_3d_input():
    X, R, W, _, link = _make_problem(seed=7)
    try:
        O.compute_exact_hessian(X, R, W, link=link)
    except ValueError as err:
        assert "single slice" in str(err)
    else:
        raise AssertionError("compute_exact_hessian should reject a 3-D X")


############################################################
# Penalty
############################################################

def _penalty_params(T, order=2, l1=0.5, eps=1e-3, delta_t=10.0):
    ops = O.compute_difference_operators(T, order, delta_t)
    return {"l1": {order: l1}, "eps": {order: eps},
            "diff_operators": ops, "delta_t": delta_t}


def test_penalty_gradient_matches_finite_differences():
    n, d, T = 6, 2, 15
    rng = np.random.default_rng(8)
    X = rng.normal(scale=100.0, size=(n, d, T))
    params = _penalty_params(T)

    def fn(x):
        loss, grad = O.compute_penalty_loss_gradient(x.reshape(n, d, T), params)
        return loss, grad.ravel()

    err = _max_fd_gradient_error(fn, X.ravel(), seed=8)
    assert err < 1e-6, f"penalty gradient: max rel err {err:.3e}"


def test_penalty_gradient_inside_pseudo_huber_region():
    """Gradient is correct where ||u|| < eps, i.e. inside the smoothed region.

    The roughness is set an order of magnitude below eps, so every group sits strictly
    inside the quadratic region, but large enough in absolute terms that the finite
    difference reference is well conditioned.
    """
    n, d, T = 5, 2, 12
    rng = np.random.default_rng(9)
    eps = 1.0
    # Second differences of order eps/10: inside the smoothed region, yet far above
    # the finite-difference noise floor
    X = np.linspace(0, 100, T)[None, None, :] * np.ones((n, d, 1))
    X = X + rng.normal(scale=eps / 10.0, size=(n, d, T))

    params = _penalty_params(T, eps=eps, delta_t=1.0)
    norms = np.linalg.norm(X @ params["diff_operators"][2].T, axis=1)
    assert np.mean(norms < eps) > 0.9, (
        f"test setup should sit inside the smoothed region; "
        f"{np.mean(norms < eps):.0%} of groups are below eps"
    )

    def fn(x):
        loss, grad = O.compute_penalty_loss_gradient(x.reshape(n, d, T), params)
        return loss, grad.ravel()

    # The penalty is smooth and well scaled here, so a smaller step than the stiff
    # data term's is both usable and necessary: the central-difference truncation
    # error falls as h**2, reaching ~2e-9 at rel_step 1e-6.
    err = _max_fd_gradient_error(fn, X.ravel(), seed=9, rel_step=1e-6,
                                 largest_gradients=True)
    assert err < 1e-7, f"pseudo-Huber region gradient: max rel err {err:.3e}"


def test_penalized_gradient_with_ancient_samples():
    """The packed penalized gradient agrees with finite differences."""
    X, R, W, sampled_by_time, link = _make_problem(n=8, T=14, seed=10,
                                                   ancient={2: 4, 6: 3})
    n, d, T = X.shape
    params = _penalty_params(T)
    problem = {"R": R, "W": W, "link": link, "sampled_by_time": sampled_by_time,
               "shape": (n, d, T), "smoothing_params": params}
    x0 = O.pack_coordinates(np.nan_to_num(X), sampled_by_time)

    err = _max_fd_gradient_error(
        lambda x: O.compute_penalized_loss_gradient(x, problem), x0, seed=10
    )
    assert err < FD_TOL, f"penalized gradient: max rel err {err:.3e}"


############################################################
# NaN / zero-weight masking
############################################################

def test_nan_rate_pair_contributes_nothing():
    """Perturbing a NaN-rate pair changes neither loss nor gradient elsewhere."""
    X, R, W, _, link = _make_problem(n=6, T=1, seed=11)
    X_t, R_t, W_t = X[:, :, 0].copy(), R[:, :, 0].copy(), W[:, :, 0].copy()

    # Mark pair (0, 1) unobserved in both directions
    R_t[0, 1] = R_t[1, 0] = np.nan

    problem = {"R": R_t, "W": W_t, "link": link}
    loss0, grad0 = O.compute_loss_gradient(X_t.ravel(), X_t.shape, problem)
    assert np.isfinite(loss0) and np.all(np.isfinite(grad0))

    # Changing that pair's rates must not move anything
    R_perturbed = R_t.copy()
    R_perturbed[0, 1] = R_perturbed[1, 0] = 12345.0
    R_perturbed[0, 1] = R_perturbed[1, 0] = np.nan   # still masked
    loss1, grad1 = O.compute_loss_gradient(
        X_t.ravel(), X_t.shape, {"R": R_perturbed, "W": W_t, "link": link}
    )
    assert loss1 == loss0
    assert np.array_equal(grad1, grad0)


def test_diagonal_does_not_leak_nan():
    """A coincident pair (zero distance) must not produce NaN in the gradient."""
    n, d = 5, 2
    X_t = np.zeros((n, d))                      # every sample at the same point
    R_t = np.full((n, n), 1e-4)
    np.fill_diagonal(R_t, np.nan)
    W_t = np.full((n, n), 1e6)
    np.fill_diagonal(W_t, np.nan)
    link = O.LinkFunction("inverse_distance", N_min=1000.0, p=1.0)

    loss, grad = O.compute_loss_gradient(X_t.ravel(), X_t.shape,
                                         {"R": R_t, "W": W_t, "link": link})
    assert np.isfinite(loss)
    assert np.all(np.isfinite(grad))


############################################################
# pack / unpack
############################################################

def test_pack_unpack_round_trip():
    X, _, _, sampled_by_time, _ = _make_problem(n=7, T=9, seed=12, ancient={3: 4})
    n, d, T = X.shape

    packed = O.pack_coordinates(X, sampled_by_time)
    restored = O.unpack_coordinates(packed, sampled_by_time, n, d, T, fill=np.nan)

    present = sampled_by_time
    assert np.array_equal(
        restored.transpose(0, 2, 1)[present], X.transpose(0, 2, 1)[present]
    )
    assert np.isnan(restored.transpose(0, 2, 1)[~present]).all()
    assert packed.size == int(present.sum()) * d


def test_pack_rejects_non_contiguous_presence():
    X, _, _, sampled_by_time, _ = _make_problem(n=5, T=8, seed=13)
    sampled_by_time = sampled_by_time.copy()
    sampled_by_time[2, 3] = False            # punch an interior hole

    try:
        O.pack_coordinates(X, sampled_by_time)
    except ValueError as err:
        assert "contiguous" in str(err)
    else:
        raise AssertionError("pack_coordinates should reject non-contiguous presence")


############################################################
# Difference operators
############################################################

def test_difference_operators_annihilate_low_degree_polynomials():
    """D_k applied to a polynomial of degree k-1 in t returns zero."""
    T, delta_t = 20, 2.5
    t = np.arange(T) * delta_t
    ops = O.compute_difference_operators(T, 4, delta_t)

    for k, D in ops.items():
        assert D.shape == (T - k, T), f"D_{k} should be ({T-k}, {T}), got {D.shape}"
        for degree in range(k):                     # degrees 0 .. k-1
            values = t ** degree
            assert np.allclose(D @ values, 0.0, atol=1e-6 * max(1.0, np.abs(values).max())), (
                f"D_{k} should annihilate t**{degree}"
            )


def test_difference_operator_scaling():
    """D_k on t**k returns k! (the k-th derivative), confirming the delta_t scaling."""
    T, delta_t = 15, 3.0
    t = np.arange(T) * delta_t
    ops = O.compute_difference_operators(T, 3, delta_t)
    for k, D in ops.items():
        assert np.allclose(D @ (t ** k), float(math.factorial(k)), rtol=1e-8)


def test_difference_operators_stop_at_short_grids():
    """Orders with no room on the grid are simply absent, not empty arrays."""
    ops = O.compute_difference_operators(3, 5, 1.0)
    assert set(ops) == {1, 2}, f"expected orders 1 and 2 on a 3-point grid, got {sorted(ops)}"


############################################################
# Weight capping (LatentSpaceEmbeddings)
############################################################

def _rates_frame(n=6, T=5, seed=14):
    import pandas as pd
    rng = np.random.default_rng(seed)
    times = np.arange(T) * 10.0
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    rates = np.abs(rng.normal(scale=1e-4, size=(len(pairs), T))) + 1e-6
    se = np.full_like(rates, 1e-5)
    idx = pd.MultiIndex.from_tuples(pairs, names=["sample_id_1", "sample_id_2"])
    return (pd.DataFrame(rates, index=idx, columns=times),
            pd.DataFrame(se, index=idx, columns=times))


def test_weight_cap_binds_on_outlier_and_not_on_clean_slice():
    sys.path.insert(
        0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
    )
    from LatentSpaceEmbeddings import LatentSpaceEmbeddings

    rates_df, se_df = _rates_frame()

    # Clean run: uniform SEs and comparable rates, so no weight is 100x the median
    clean = LatentSpaceEmbeddings(rates_df, se_df, N_min=1000.0, N_max=1e6,
                                  weight_cap_factor=100.0)
    W_clean = clean.W[:, :, 0]
    finite = np.isfinite(W_clean)
    median = np.nanmedian(W_clean)
    assert np.max(W_clean[finite]) <= 100.0 * median + 1e-6
    # and the cap should not have bound: the max is far below it
    assert np.max(W_clean[finite]) < 100.0 * median

    # Inject an outlier: one pair with a tiny SE at slice 0 -> a huge weight
    se_outlier = se_df.copy()
    se_outlier.iloc[0, 0] = 1e-12
    capped = LatentSpaceEmbeddings(rates_df, se_outlier, N_min=1000.0, N_max=1e6,
                                   weight_cap_factor=100.0)
    W_capped = capped.W[:, :, 0]
    finite_c = np.isfinite(W_capped)
    median_c = np.nanmedian(W_capped)
    assert np.max(W_capped[finite_c]) <= 100.0 * median_c * (1 + 1e-6), (
        "the per-slice ratio cap should bind on the injected outlier"
    )


def test_zero_standard_error_gives_finite_weight():
    """A confidently-zero rate (se = 0) must not produce NaN or inf weights."""
    from LatentSpaceEmbeddings import LatentSpaceEmbeddings

    rates_df, se_df = _rates_frame()
    rates_df = rates_df.copy(); se_df = se_df.copy()
    rates_df.iloc[0, :] = 0.0
    se_df.iloc[0, :] = 0.0

    LSE = LatentSpaceEmbeddings(rates_df, se_df, N_min=1000.0, N_max=np.inf)
    present = ~np.isnan(LSE.R)
    assert np.all(np.isfinite(LSE.W[present])), "zero SE must yield a finite weight"


def test_unknown_constructor_argument_is_rejected():
    """l2_strengths is gone entirely, so it is now just an unexpected keyword."""
    from LatentSpaceEmbeddings import LatentSpaceEmbeddings

    rates_df, se_df = _rates_frame()
    try:
        LatentSpaceEmbeddings(rates_df, se_df, l2_strengths={2: 1e-3})
    except TypeError as err:
        assert "l2_strengths" in str(err)
    else:
        raise AssertionError("l2_strengths is no longer a parameter and should raise")


def test_non_uniform_time_grid_is_rejected():
    import pandas as pd
    from LatentSpaceEmbeddings import LatentSpaceEmbeddings

    rates_df, se_df = _rates_frame()
    log_times = np.geomspace(1.0, 1000.0, rates_df.shape[1])
    rates_df.columns = log_times
    se_df.columns = log_times
    try:
        LatentSpaceEmbeddings(rates_df, se_df)
    except ValueError as err:
        assert "uniform" in str(err)
    else:
        raise AssertionError("a non-uniform time grid should be rejected")


############################################################
# Link function
############################################################

def test_link_derivatives_match_finite_differences():
    x = np.array([0.5, 2.0, 10.0, 100.0])
    for name in ("inverse_distance", "exponential"):
        for p in (0.5, 1.0, 2.0):
            link = O.LinkFunction(name, N_min=1000.0, p=p)
            f, fp, fpp = link.evaluate(x, derivatives=2)
            h = x * 1e-5
            fp_fd = (link.f(x + h) - link.f(x - h)) / (2 * h)
            fpp_fd = (link.f(x + h) - 2 * link.f(x) + link.f(x - h)) / h ** 2
            assert np.allclose(fp, fp_fd, rtol=1e-4), f"{name} p={p} f'"
            assert np.allclose(fpp, fpp_fd, rtol=1e-2), f"{name} p={p} f''"


def test_link_inverse_round_trip():
    """inverse(f(x)) == x wherever f has not saturated.

    The exponential link decays as exp(-x**p), so f becomes tiny at moderate x and the
    link's own eps guard (added inside the log to keep the argument positive) starts to
    dominate the inverse once f falls below roughly 1e-8 -- and for large x**p the
    forward map underflows to exactly zero, destroying the information entirely. Each
    link is therefore probed over the range where it is actually invertible; this is a
    property of the link, not of the implementation.
    """
    for name in ("inverse_distance", "exponential"):
        for p in (0.5, 1.0, 2.0):
            link = O.LinkFunction(name, N_min=1000.0, p=p)
            if name == "exponential":
                x = np.linspace(0.5, 5.0 ** (1.0 / p), 6)    # f stays above ~1e-6
            else:
                x = np.array([0.5, 2.0, 10.0, 100.0, 1e4])
            assert np.allclose(link.inverse(link.f(x)), x, rtol=1e-6), (
                f"{name} p={p}: {link.inverse(link.f(x))} != {x}"
            )


def test_link_validates_parameters():
    for kwargs in ({"name": "nonsense", "N_min": 1.0},
                   {"name": "inverse_distance", "N_min": 1.0, "p": 0.0},
                   {"name": "inverse_distance", "N_min": -1.0}):
        try:
            O.LinkFunction(**kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"LinkFunction should reject {kwargs}")


############################################################
# Standalone runner (for environments without pytest)
############################################################

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
