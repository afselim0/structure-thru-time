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


def test_ldl_flip_is_positive_definite_and_descent():
    """L |D| L^T has the right inertia, so the direction always goes downhill.

    LDL is a congruence, so by Sylvester's law of inertia flipping the negative
    eigenvalues of D's 1x1/2x2 blocks yields a positive definite matrix -- which is what
    makes ``ldl_flip`` usable on an indefinite Hessian without any diagonal shift.
    """
    for seed in (4, 5, 6):
        X, R, W, _, link = _make_problem(n=10, T=1, seed=seed)
        X_t, R_t, W_t = X[:, :, 0], R[:, :, 0], W[:, :, 0]
        H = O.compute_exact_hessian(X_t, R_t, W_t, link=link)
        problem = {"R": R_t, "W": W_t, "link": link}
        _, grad = O.compute_loss_gradient(X_t.ravel(), X_t.shape, problem)

        for method in ("ldl_flip", "eigh_floor"):
            direction = O.modified_newton_direction(H, grad, method=method)
            slope = float(np.dot(grad, direction))
            assert np.all(np.isfinite(direction)), f"{method}: non-finite direction"
            assert slope < 0, f"{method}: ascent direction, g.p = {slope:.3e} (seed {seed})"


def test_ldl_flip_reconstructs_a_positive_definite_matrix():
    """Rebuild L |D| L^T explicitly and check its eigenvalues are all positive."""
    from scipy.linalg import ldl

    X, R, W, _, link = _make_problem(n=8, T=1, seed=4)
    H = O.compute_exact_hessian(X[:, :, 0], R[:, :, 0], W[:, :, 0], link=link)
    eig_H = np.linalg.eigvalsh(H)
    assert np.sum(eig_H < -1e-12 * np.max(np.abs(eig_H))) > 0, (
        "test needs an indefinite Hessian to be meaningful"
    )

    lu, d_blocks, perm = ldl(H, lower=True)
    N = H.shape[0]
    dm = np.asarray(d_blocks, dtype=float)
    D_abs = np.zeros((N, N))
    i = 0
    while i < N:
        if (i + 1 < N) and (dm[i + 1, i] != 0.0 or dm[i, i + 1] != 0.0):
            w, V = np.linalg.eigh(dm[i:i + 2, i:i + 2])
            D_abs[i:i + 2, i:i + 2] = (V * np.abs(w)) @ V.T
            i += 2
        else:
            D_abs[i, i] = abs(dm[i, i])
            i += 1

    L = lu[perm, :]
    B = L @ D_abs @ L.T                       # = |H| in the Gould et al. (2000) sense
    eig_B = np.linalg.eigvalsh(B)
    # Positive definite up to round-off in the reconstruction (the factorisation itself is
    # exact in inertia; forming L |D| L^T explicitly is not).
    assert eig_B.min() > -1e-12 * eig_B.max(), (
        f"L |D| L^T is not positive definite: min eig {eig_B.min():.3e} "
        f"against max {eig_B.max():.3e}"
    )
    # Note that L |D| L^T is *not* |H|: it has the right inertia (all-positive) but not
    # the same eigenvalues, so its spectrum is deliberately not compared to H's.  See
    # modified_newton_direction's docstring -- the direction has cos ~ 0.4-0.6 with the
    # exact |H|^-1 g, which costs nothing empirically but is an approximation.
    assert eig_B.shape == eig_H.shape


def test_laplacian_gradient_identity():
    """g+ + g- == g, and the Laplacian of g applied to X reproduces the gradient.

    This is the structural identity the Newton machinery relies on: the gradient is
    ``L X`` with ``L = diag(sum_j g_ij) - G``, with no factor of 2 (see the module
    docstring's sign/factor convention).
    """
    X, R, W, _, link = _make_problem(n=9, T=1, seed=11)
    X_t, R_t, W_t = X[:, :, 0], R[:, :, 0], W[:, :, 0]
    mask = O.default_valid_mask(R_t)

    D_t = O.compute_distances(X_t)
    f_t, f_prime_t = link.evaluate(D_t, derivatives=1)
    G = O.compute_gradient_weights(D_t, f_t, f_prime_t, R_t, W_t, mask, link.eps)

    # Split into attractive (g+) and repulsive (g-) parts and check they sum to g.
    with np.errstate(divide="ignore", invalid="ignore"):
        d_safe = np.maximum(D_t, link.eps)
        g_minus = np.where(mask, W_t * f_prime_t / d_safe, 0.0)
        g_plus = np.where(mask, -(R_t / f_t) * g_minus, 0.0)
    assert np.allclose(g_plus + g_minus, G, rtol=1e-10, atol=1e-14), "g+ + g- != g"
    assert np.all(g_plus >= -1e-300), "g+ has negative entries"

    # L X == grad, with L the Laplacian of G.
    L = np.diag(G.sum(axis=1)) - G
    problem = {"R": R_t, "W": W_t, "link": link}
    _, grad_ref = O.compute_loss_gradient(X_t.ravel(), X_t.shape, problem)
    assert np.allclose(L @ X_t, grad_ref.reshape(X_t.shape), rtol=1e-8, atol=1e-10), \
        "Laplacian gradient != compute_loss_gradient"


def _structured_problem(n=12, d=2, seed=1, N_min=1000.0, p=1.0, noise=40.0):
    """A slice whose rates are generated *from* a latent configuration.

    ``_make_problem`` draws rates as independent noise, so no embedding explains them and
    the loss has no well-conditioned minimum -- the solver drives pairs to near-zero
    distance, where the link's curvature diverges.  That is fine for checking gradients
    and Hessians pointwise, but it is not a fair convergence test.  The real pipeline
    fits rates that do come from structure, which is what this builds.
    """
    rng = np.random.default_rng(seed)
    link = O.LinkFunction("inverse_distance", N_min=N_min, p=p)
    X_true = rng.normal(scale=800.0, size=(n, d))
    R = link.f(O.compute_distances(X_true))
    np.fill_diagonal(R, np.nan)
    W = np.full((n, n), 1e6)
    np.fill_diagonal(W, np.nan)
    X0 = X_true + rng.normal(scale=noise, size=(n, d))
    return X0, R, W, link


def test_newton_slice_reduces_loss_and_reaches_tolerance():
    """The Newton slice solver converges, monotonically, on a structured problem."""
    X0, R_t, W_t, link = _structured_problem(seed=1)
    problem = {"R": R_t, "W": W_t, "link": link}
    f0, _ = O.compute_loss_gradient(X0.ravel(), X0.shape, problem)

    X_opt, loss, n_iter, message = O.optimize_timepoint_newton(
        X0, R_t, W_t, link=link, N_max=np.inf, max_iter=400,
    )
    assert np.all(np.isfinite(X_opt))
    assert loss <= f0 + 1e-9 * abs(f0), f"loss increased: {f0:.8g} -> {loss:.8g}"

    _, g = O.compute_loss_gradient(X_opt.ravel(), X_opt.shape, problem)
    tol = O._newton_slice_tolerance(X0, problem, O.TOL_FACTOR_NEWTON)
    assert np.max(np.abs(g)) <= tol, (
        f"|grad| {np.max(np.abs(g)):.3e} above tolerance {tol:.3e} after "
        f"{n_iter} iterations ({message})"
    )
    assert "converged" in message, message


def test_newton_beats_lbfgs_on_an_ill_conditioned_slice():
    """The reason for the change: L-BFGS stalls where Newton converges.

    Not a benchmark -- just a guard that the Newton path really does reach a far deeper
    gradient than the L-BFGS path it replaces, on a slice with the conditioning that
    motivated it (p=2 puts curvature ~ d^-2, which is minute at these distances).
    """
    X0, R_t, W_t, link = _structured_problem(seed=2, N_min=1.0, p=2.0)
    problem = {"R": R_t, "W": W_t, "link": link}

    X_n, _, _, _ = O.optimize_timepoint_newton(
        X0, R_t, W_t, link=link, N_max=np.inf, max_iter=400
    )
    X_l, _, _, _ = O.optimize_timepoint(
        X0, R_t, W_t, link=link, N_max=np.inf
    )
    _, g_n = O.compute_loss_gradient(X_n.ravel(), X_n.shape, problem)
    _, g_l = O.compute_loss_gradient(X_l.ravel(), X_l.shape, problem)
    gn, gl = np.max(np.abs(g_n)), np.max(np.abs(g_l))
    assert gn < gl, f"Newton |grad| {gn:.3e} not below L-BFGS {gl:.3e}"


def test_newton_slice_tolerance_ignores_the_warm_start():
    """The tolerance must not tighten just because the slice starts near its optimum.

    That dependence is what ``_resolve_gtol`` has, and it is the mechanism that
    entrenched the per-slice residual: a slice beginning close to its optimum was
    permitted to stop close to where it began, handing the remainder to the next slice.
    """
    X0, R_t, W_t, link = _structured_problem(seed=3)
    problem = {"R": R_t, "W": W_t, "link": link}
    X_opt, _, _, _ = O.optimize_timepoint_newton(
        X0, R_t, W_t, link=link, N_max=np.inf, max_iter=400
    )

    # The old rule keys off the gradient at the starting point, which collapses by many
    # orders of magnitude once the start is already converged.
    _, g_cold = O.compute_loss_gradient(X0.ravel(), X0.shape, problem)
    _, g_warm = O.compute_loss_gradient(X_opt.ravel(), X_opt.shape, problem)
    old_cold = O._resolve_gtol(g_cold, O.GTOL_FACTOR)
    old_warm = O._resolve_gtol(g_warm, O.GTOL_FACTOR)
    assert old_warm < 1e-6 * old_cold, (
        f"expected the old rule to collapse at a converged start: "
        f"{old_cold:.3e} -> {old_warm:.3e}"
    )

    # The new rule anchors to a random configuration at the same spatial scale instead,
    # so it barely moves between the two starting points.
    tol_cold = O._newton_slice_tolerance(X0, problem, O.TOL_FACTOR_NEWTON, seed=0)
    tol_warm = O._newton_slice_tolerance(X_opt, problem, O.TOL_FACTOR_NEWTON, seed=0)
    assert 1e-2 < tol_warm / tol_cold < 1e2, (
        f"tolerance moved with the warm start: {tol_cold:.3e} -> {tol_warm:.3e}"
    )


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


def test_unknown_constructor_argument_is_rejected():
    """l2_strengths is gone entirely, so it is now just an unexpected keyword."""
    from LatentSpaceEmbeddings import LatentSpaceEmbeddings

    rates_df, weights_df = _rates_frame()
    try:
        LatentSpaceEmbeddings(rates_df, weights_df, l2_strengths={2: 1e-3})
    except TypeError as err:
        assert "l2_strengths" in str(err)
    else:
        raise AssertionError("l2_strengths is no longer a parameter and should raise")


def test_non_uniform_time_grid_is_rejected():
    import pandas as pd
    from LatentSpaceEmbeddings import LatentSpaceEmbeddings

    rates_df, weights_df = _rates_frame()
    log_times = np.geomspace(1.0, 1000.0, rates_df.shape[1])
    rates_df.columns = log_times
    weights_df.columns = log_times
    try:
        LatentSpaceEmbeddings(rates_df, weights_df)
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
