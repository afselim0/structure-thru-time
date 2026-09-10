#%%
#!/usr/bin/env python3
"""Loss, gradient and Hessian machinery for the CoaSTT latent-position model.

Model
-----
Pairwise coalescent rates ``lambda_ijt`` inferred from ARGs are fitted by a latent
position model.  Writing ``d_ij = ||X_i - X_j||`` for the latent distance and
``f_ij = f(d_ij)`` for the link function, the per-slice (unpenalized) loss is

    l(X_t) = 0.5 * sum_{i,j} w_ij [ f_ij - lambda_ij ln f_ij ]

optimized with L-BFGS-B: first sequentially over time slices (unpenalized), then
jointly over all slices with a temporal smoothing penalty.

Laplacian formulation
---------------------
Per slice, with weights ``w_ij`` and rates ``lambda_ij``, define

    g^-_ij = w_ij * f'_ij / d_ij                                   (repulsive,  <= 0)
    g^+_ij = -(lambda_ij / f_ij) * g^-_ij                          (attractive, >= 0)
    g_ij   = g^+_ij + g^-_ij = w_ij * (f'_ij/d_ij) * (1 - lambda_ij/f_ij)

    kappa^-_ij = w_ij * f''_ij
    kappa^+_ij = -w_ij * lambda_ij * [ f''_ij/f_ij - (f'_ij)^2/f_ij^2 ]
    kappa_ij   = kappa^+_ij + kappa^-_ij

SIGN / FACTOR CONVENTION.  The loss carries the ``0.5`` prefactor (each unordered
pair is counted once across the full double sum), and therefore the gradient is

    grad_X l = L X          with   L = diag(sum_j g_ij) - G,   G = (g_ij)

with **no** factor of 2.  The manuscript appendix writes a loss without the 0.5 and
so reports ``grad = 2 L X``; the code deliberately does not match the appendix here.

Convergence tolerances
----------------------
SciPy's L-BFGS-B stops on a *relative* function criterion and an *absolute* gradient
criterion:

    (f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= ftol
    max_i |proj g_i|                           <= gtol

Because the weights ``w_ij`` are of order 1e6, the objective and its gradient are of
that order too, so an absolute ``gtol`` is meaningless unless set absurdly small --
which is why the historical code carried ``GTOL_TIMEPOINT = 1e-16`` and relied on
``ftol`` alone.  The objective is deliberately left unnormalized (normalizing would
change the penalty's relative scale).  Instead ``ftol`` is the primary criterion,
exposed as ``target_precision``, and ``gtol`` is made scale-aware by setting it to
``gtol_factor * max|grad|`` evaluated at the initial point.
"""

FTOL_TIMEPOINT     = 1e-9
MAX_ITER_TIMEPOINT = 12000
MAXCOR_TIMEPOINT   = 10

FTOL_JOINT         = 1e-9
MAX_ITER_JOINT     = 2500
MAXCOR_JOINT       = 20

GTOL_FACTOR        = 1e-8
DEFAULT_WEIGHTS    = 1e6

# --- Sequential stage: damped-Newton path (the default solver) ---
MAX_ITER_NEWTON    = 1500
TOL_FACTOR_NEWTON  = 1e-10

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import pandas as pd
from scipy.linalg import ldl, solve_triangular
from scipy.optimize import minimize
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm

# The gradient is a difference of terms of order |loss|, so ~eps*|loss| is its float64
# noise floor; tolerances below it cannot be met and only burn the iteration budget.
GRAD_FLOOR_EPS     = 1e2 * np.finfo(float).eps


############################################################
# Link functions
############################################################

@dataclass(frozen=True)
class LinkFunction:
    """The link ``f`` mapping latent distances to coalescent rates.

    ``f`` must be positive and monotonically decreasing.  A small ``eps`` is added to
    every distance before evaluation, both to avoid division by zero at coincident
    points and to keep ``log f`` finite.

    Attributes:
        name: "inverse_distance" (f = 1/(N_min + x^p)) or "exponential"
            (f = exp(-x^p)/N_min).
        N_min: Minimum modellable rate scale; pairs with rate 1/N_min or higher sit
            at zero distance.
        p: Exponent of the distance in the link.
        eps: Offset added to distances before evaluation.
    """

    name: str
    N_min: float
    p: float = 1.0
    eps: float = 1e-10

    def __post_init__(self) -> None:
        # Validate once, at construction, rather than on every evaluation
        if self.name not in ("inverse_distance", "exponential"):
            raise ValueError(
                f"Unsupported link function {self.name!r}; "
                "expected 'inverse_distance' or 'exponential'."
            )
        if not self.p > 0:
            raise ValueError(f"p must be positive, got {self.p}.")
        if not self.N_min > 0:
            raise ValueError(f"N_min must be positive, got {self.N_min}.")
        if not self.eps > 0:
            raise ValueError(f"eps must be positive, got {self.eps}.")

    def evaluate(self, x: np.ndarray, derivatives: int = 1) -> tuple[np.ndarray, ...]:
        """Evaluate f and, optionally, its first and second derivatives.

        Args:
            x: Distances.
            derivatives: 0, 1 or 2 -- how many derivatives to return alongside f.

        Returns:
            (f,), (f, f') or (f, f', f'') according to ``derivatives``.
        """
        if derivatives not in (0, 1, 2):
            raise ValueError(f"derivatives must be 0, 1 or 2, got {derivatives}.")

        p, N_min = self.p, self.N_min
        x = np.asarray(x, dtype=float) + self.eps

        if self.name == "inverse_distance":                 # f = 1/(N_min + x^p)
            f = (N_min + x ** p) ** -1
            if derivatives == 0:
                return (f,)
            f_prime = -p * x ** (p - 1) * f ** 2
            if derivatives == 1:
                return f, f_prime
            f_double_prime = p * x ** (p - 2) * f ** 3 * ((1 - p) * N_min + (1 + p) * x ** p)
        else:                                               # f = exp(-x^p) / N_min
            f = (1.0 / N_min) * np.exp(-x ** p)
            if derivatives == 0:
                return (f,)
            f_prime = -p * x ** (p - 1) * f
            if derivatives == 1:
                return f, f_prime
            f_double_prime = p * x ** (p - 2) * f * (1 - p + p * x ** p)

        return f, f_prime, f_double_prime

    def f(self, x: np.ndarray) -> np.ndarray:
        """The link function itself."""
        return self.evaluate(x, derivatives=0)[0]

    def f_prime(self, x: np.ndarray) -> np.ndarray:
        """First derivative of the link."""
        return self.evaluate(x, derivatives=1)[1]

    def f_double_prime(self, x: np.ndarray) -> np.ndarray:
        """Second derivative of the link."""
        return self.evaluate(x, derivatives=2)[2]

    def inverse(self, f: np.ndarray) -> np.ndarray:
        """Map rates back to distances.

        The inner expression is clipped at zero to guard against negative bases
        arising from floating-point error near the boundary.
        """
        if self.name == "inverse_distance":
            # f = 1/(x^p + N_min)  =>  x = (1/f - N_min)^(1/p)
            inner = (1.0 / f) - self.N_min
        else:
            # f = exp(-x^p)/N_min  =>  x = (-ln(N_min * f))^(1/p)
            inner = -np.log(self.N_min * f + self.eps)
        return np.clip(inner, 0, None) ** (1.0 / self.p)


############################################################
# Helper / matrix-construction functions
############################################################

def _as_link(link: Union[LinkFunction, str], N_min: float = 1.0, p: float = 1.0) -> LinkFunction:
    """Coerce a link specification to a LinkFunction.

    Accepting a bare string keeps the many ``link="inverse_distance"`` call sites in
    the plotting scripts working without each of them constructing a LinkFunction.
    """
    if isinstance(link, LinkFunction):
        return link
    return LinkFunction(name=link, N_min=N_min, p=p)


def compute_distances(X_t: np.ndarray) -> np.ndarray:
    """Pairwise Euclidean distance matrix for one slice. (n, d) -> (n, n)"""
    return squareform(pdist(X_t, "euclidean"))


def default_valid_mask(R: np.ndarray) -> np.ndarray:
    """The default validity mask: entries whose rate is not NaN.

    This covers both the diagonal and the unsampled ancient-sample entries, since
    LatentSpaceEmbeddings writes NaN into R at both.
    """
    return ~np.isnan(R)


def compute_gradient_weights(
    D_t: np.ndarray,
    f_t: np.ndarray,
    f_prime_t: np.ndarray,
    R_t: np.ndarray,
    W_t: np.ndarray,
    valid_mask_t: np.ndarray,
    eps: float,
) -> np.ndarray:
    """The Laplacian gradient weights g_ij = w_ij (f'_ij/d_ij) (1 - lambda_ij/f_ij).

    Division by d_ij is guarded independently of the mask: ``0 * NaN`` is NaN, so an
    unguarded 0/0 on the diagonal would propagate into the gradient even where the
    weight is zero.
    """
    d_safe = np.maximum(D_t, eps)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = W_t * (f_prime_t / d_safe) * (1.0 - R_t / f_t)
    return np.where(valid_mask_t, g, 0.0)


def compute_curvature_weights(
    f_t: np.ndarray,
    f_prime_t: np.ndarray,
    f_double_prime_t: np.ndarray,
    R_t: np.ndarray,
    W_t: np.ndarray,
    valid_mask_t: np.ndarray,
) -> np.ndarray:
    """The radial curvature weights kappa_ij = kappa^+_ij + kappa^-_ij.

    kappa^-_ij = w_ij f''_ij  and  kappa^+_ij = -w_ij lambda_ij [f''_ij/f_ij - (f'_ij/f_ij)^2].
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = W_t * (
            f_double_prime_t
            - R_t * (f_double_prime_t / f_t - (f_prime_t / f_t) ** 2)
        )
    return np.where(valid_mask_t, kappa, 0.0)


def compute_difference_operators(
    n_timepoints: int, max_order: int, delta_t: float
) -> dict[int, np.ndarray]:
    """Rectangular finite-difference operators D_k of shape (T - k, T).

    D_1 is the (T-1, T) forward-difference matrix divided by delta_t; D_k is the
    composition of k such matrices of decreasing height, so D_k has shape (T-k, T)
    and carries an overall 1/delta_t**k.  No square zero-padding is applied: the
    padded row of the previous implementation propagated artificial zero rows
    through the composition.

    Args:
        n_timepoints: Length T of the (uniform) time grid.
        max_order: Highest difference order to build.
        delta_t: Spacing of the time grid.

    Returns:
        {order: operator}, for orders 1..max_order with T - order >= 1.
    """
    if delta_t <= 0:
        raise ValueError(f"delta_t must be positive, got {delta_t}.")

    def forward_difference(m: int) -> np.ndarray:
        """(m-1, m) forward difference, scaled by 1/delta_t."""
        D1 = np.zeros((m - 1, m))
        np.fill_diagonal(D1, -1.0)
        np.fill_diagonal(D1[:, 1:], 1.0)
        return D1 / delta_t

    operators: dict[int, np.ndarray] = {}
    D = np.eye(n_timepoints)
    for k in range(1, max_order + 1):
        rows = n_timepoints - k
        if rows < 1:
            break                      # no room for a k-th difference on this grid
        D = forward_difference(n_timepoints - k + 1) @ D
        operators[k] = D
    return operators


def pack_coordinates(X: np.ndarray, sampled_by_time: np.ndarray) -> np.ndarray:
    """Flatten only the present (sample, time) coordinates into a 1-D vector.

    Ancient samples have undefined positions before they are sampled; packing keeps
    those out of the optimizer's variable vector entirely.

    Args:
        X: (n, d, T) coordinates.
        sampled_by_time: (n, T) boolean presence mask.

    Returns:
        1-D array of length ``d * sampled_by_time.sum()``.
    """
    _assert_contiguous_presence(sampled_by_time)
    n, d, T = X.shape
    if sampled_by_time.shape != (n, T):
        raise ValueError(
            f"sampled_by_time has shape {sampled_by_time.shape}, expected {(n, T)}."
        )
    # (n, d, T) -> select present (sample, time) columns -> flat
    return X.transpose(0, 2, 1)[sampled_by_time].ravel()


def unpack_coordinates(
    x_flat: np.ndarray,
    sampled_by_time: np.ndarray,
    n: int,
    d: int,
    T: int,
    fill: float = 0.0,
) -> np.ndarray:
    """Inverse of pack_coordinates, filling absent entries with a finite dummy.

    Absent entries are excluded by the valid mask (their rates are NaN) so they
    contribute nothing to loss or gradient, but the fill must be finite or the
    distance computation would return NaN everywhere.
    """
    n_present = int(sampled_by_time.sum())
    if x_flat.size != n_present * d:
        raise ValueError(
            f"x_flat has {x_flat.size} entries, expected {n_present * d} "
            f"({n_present} present (sample, time) pairs x {d} dimensions)."
        )
    X_swapped = np.full((n, T, d), float(fill))
    X_swapped[sampled_by_time] = x_flat.reshape(n_present, d)
    return X_swapped.transpose(0, 2, 1)


def _assert_contiguous_presence(sampled_by_time: np.ndarray) -> None:
    """Presence must be a contiguous block of the time axis for every sample.

    Because ``sampled_by_time[i, t] = sample_time_i <= time_points[t]``, presence can
    never have interior gaps.  The ancient-sample difference operators rely on this,
    so fail loudly if it ever stops holding.
    """
    changes = np.diff(sampled_by_time.astype(np.int8), axis=1)
    # More than one transition (or an off->on after an on->off) means an interior gap
    n_blocks = (changes == 1).sum(axis=1) + sampled_by_time[:, 0].astype(int)
    bad = np.where((np.abs(changes).sum(axis=1) > 1) | (n_blocks > 1))[0]
    if bad.size:
        raise ValueError(
            "sampled_by_time must be contiguous per sample (presence is defined by "
            f"sample_time <= time_point, so it cannot have interior gaps). Samples "
            f"with non-contiguous presence: {bad.tolist()}"
        )


def _assert_finite(value: float, grad: np.ndarray, slice_index: Optional[int] = None) -> None:
    """Guard against NaN/inf leaking out of a loss/gradient evaluation."""
    where = "" if slice_index is None else f" at time slice index {slice_index}"
    if not np.isfinite(value):
        raise FloatingPointError(f"Non-finite loss ({value}){where}.")
    if not np.all(np.isfinite(grad)):
        n_bad = int((~np.isfinite(grad)).sum())
        raise FloatingPointError(
            f"Non-finite gradient{where}: {n_bad} of {grad.size} entries are NaN or inf."
        )


def _iter_slices(
    X: np.ndarray, R: np.ndarray, W: np.ndarray, valid_mask: Optional[np.ndarray]
):
    """Yield (t, X_t, R_t, W_t, mask_t) for 2-D (single slice) or 3-D input.

    Time slices are handled one at a time with (n, n) working arrays; the full
    (n, n, T) distance tensor is never materialised (at n=1000, T=700 it exceeds 5 GB).
    """
    if X.ndim == 2:
        mask = default_valid_mask(R) if valid_mask is None else valid_mask
        yield None, X, R, W, mask
    elif X.ndim == 3:
        T = X.shape[2]
        for t in range(T):
            R_t = R[:, :, t]
            mask_t = default_valid_mask(R_t) if valid_mask is None else valid_mask[:, :, t]
            yield t, X[:, :, t], R_t, W[:, :, t], mask_t
    else:
        raise ValueError(f"X must be (n, d) or (n, d, T), got shape {X.shape}.")


def _default_weights_like(R: np.ndarray) -> np.ndarray:
    """Uniform fallback weights, used when no standard errors are supplied."""
    return np.ones_like(R, dtype=float) * DEFAULT_WEIGHTS


############################################################
# Unpenalized loss
############################################################

def compute_unpenalized_loss(
    X: np.ndarray,
    R: np.ndarray,
    W: Optional[np.ndarray] = None,
    link: Union[LinkFunction, str] = "inverse_distance",
    valid_mask: Optional[np.ndarray] = None,
) -> float:
    """Weighted unpenalized loss, for one slice or for a whole trajectory.

    l = 0.5 * sum_{i,j} w_ij [ f_ij - lambda_ij ln f_ij ], summed over slices.

    Args:
        X: (n, d) for a single slice, or (n, d, T).
        R: (n, n) or (n, n, T) rates, NaN where the pair is unobserved.
        W: Matching weights; defaults to DEFAULT_WEIGHTS everywhere.
        link: LinkFunction (or a link name, using default N_min/p).
        valid_mask: Optional override; defaults to ~isnan(R) per slice.

    Returns:
        The scalar loss.
    """
    link = _as_link(link)
    if W is None:
        W = _default_weights_like(R)

    # Accumulate in float64 even when R/W are stored as float32
    total = 0.0
    for t, X_t, R_t, W_t, mask_t in _iter_slices(X, R, W, valid_mask):
        D_t = compute_distances(X_t)
        (f_t,) = link.evaluate(D_t, derivatives=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            terms = W_t * (f_t - R_t * np.log(f_t))
        total += 0.5 * float(np.sum(terms[mask_t], dtype=np.float64))
    return total


############################################################
# Unpenalized gradient
############################################################

def compute_unpenalized_gradient(
    X: np.ndarray,
    R: np.ndarray,
    W: Optional[np.ndarray] = None,
    link: Union[LinkFunction, str] = "inverse_distance",
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Gradient of the unpenalized loss, for one slice or a whole trajectory.

    Per slice this is the graph Laplacian applied to the coordinates,
    ``L_t X_t`` with ``L_t = diag(sum_j g_ij) - G`` (no factor of 2; see the module
    docstring on the sign/factor convention).

    Args:
        X: (n, d) or (n, d, T).
        R: (n, n) or (n, n, T).
        W: Matching weights; defaults to DEFAULT_WEIGHTS.
        link: LinkFunction or link name.
        valid_mask: Optional override; defaults to ~isnan(R) per slice.

    Returns:
        Gradient with the same shape as X.
    """
    link = _as_link(link)
    if W is None:
        W = _default_weights_like(R)

    grad = np.zeros(X.shape, dtype=np.float64)
    for t, X_t, R_t, W_t, mask_t in _iter_slices(X, R, W, valid_mask):
        D_t = compute_distances(X_t)
        f_t, f_prime_t = link.evaluate(D_t, derivatives=1)
        G = compute_gradient_weights(D_t, f_t, f_prime_t, R_t, W_t, mask_t, link.eps)
        grad_t = G.sum(axis=1)[:, None] * X_t - G @ X_t
        if t is None:
            grad = grad_t
        else:
            grad[:, :, t] = grad_t
    return grad


############################################################
# Exact unpenalized Hessian (single slice only)
############################################################

def compute_exact_hessian(
    X_t: np.ndarray,
    R_t: np.ndarray,
    W_t: Optional[np.ndarray] = None,
    link: Union[LinkFunction, str] = "inverse_distance",
    valid_mask_t: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Exact Hessian of the unpenalized loss at one slice, w.r.t. flattened X_t.

    The per-pair d x d block is

        B_ij = g_ij I_d + (kappa_ij - g_ij) P_ij,   P_ij = u u^T,  u = (X_i - X_j)/d_ij

    accumulated as H[i,i] += B, H[j,j] += B, H[i,j] -= B, H[j,i] -= B.  The result is
    symmetric with exactly d zero eigenvalues (translation invariance).

    This exists for the per-slice Laplace covariances.  It does **not** include the
    smoothing penalty, which couples time slices and cannot be represented in a
    per-slice block.

    Args:
        X_t: (n, d) coordinates at one slice.
        R_t: (n, n) rates.
        W_t: (n, n) weights; defaults to DEFAULT_WEIGHTS.
        link: LinkFunction or link name.
        valid_mask_t: Optional override; defaults to ~isnan(R_t).

    Returns:
        (n*d, n*d) Hessian.
    """
    if X_t.ndim != 2:
        raise ValueError(
            f"compute_exact_hessian takes a single slice of shape (n, d), got {X_t.shape}. "
            "The joint Hessian is (n*d*T)^2 and is not computable at these sizes; it is "
            "also not the right object, since the smoothing penalty couples time slices."
        )

    link = _as_link(link)
    if W_t is None:
        W_t = _default_weights_like(R_t)
    if valid_mask_t is None:
        valid_mask_t = default_valid_mask(R_t)

    n, d = X_t.shape
    D_t = compute_distances(X_t)
    f_t, f_prime_t, f_double_prime_t = link.evaluate(D_t, derivatives=2)

    # Radial (kappa) and tangential (g) curvature weights
    G     = compute_gradient_weights(D_t, f_t, f_prime_t, R_t, W_t, valid_mask_t, link.eps)
    kappa = compute_curvature_weights(f_t, f_prime_t, f_double_prime_t, R_t, W_t, valid_mask_t)

    # Unit separation vectors and their outer products, vectorized over all pairs
    d_safe = np.maximum(D_t, link.eps)
    U = (X_t[:, None, :] - X_t[None, :, :]) / d_safe[:, :, None]     # (n, n, d)
    U = np.where(valid_mask_t[:, :, None], U, 0.0)
    P = U[:, :, :, None] * U[:, :, None, :]                          # (n, n, d, d)

    # B_ij = g_ij I_d + (kappa_ij - g_ij) P_ij
    I_d = np.eye(d)
    B = G[:, :, None, None] * I_d + (kappa - G)[:, :, None, None] * P  # (n, n, d, d)

    # Off-diagonal blocks are -B; each diagonal block is the row sum of B
    H = -B.transpose(0, 2, 1, 3).copy()                              # (n, d, n, d)
    diag = B.sum(axis=1)                                             # (n, d, d)
    idx = np.arange(n)
    H[idx, :, idx, :] = diag
    return H.reshape(n * d, n * d)


############################################################
# Modified-Newton direction from the indefinite Hessian
############################################################

def modified_newton_direction(
    H: np.ndarray,
    grad: np.ndarray,
    method: str = "ldl_flip",
    delta_ratio: float = 1e-8,
) -> np.ndarray:
    """Descent direction ``p`` solving ``|H| p = -grad`` for an indefinite Hessian.

    The exact Hessian is genuinely indefinite away from a minimum -- the repulsive term
    contributes negative curvature (34 of 120 eigenvalues negative on a typical slice) --
    so the raw Newton step ``-H^-1 grad`` can point uphill.  ``H`` must therefore be
    modified into something positive definite before it is inverted.

    ``ldl_flip`` (default) takes the Bunch-Kaufman factorization ``P^T H P = L D L^T``
    (``scipy.linalg.ldl``) and flips the negative eigenvalues of ``D``'s 1x1 and 2x2
    blocks.  Because LDL is a *congruence*, Sylvester's law of inertia guarantees that
    ``L |D| L^T`` is positive definite, hence ``p`` is always a descent direction.  ``D``
    is block diagonal, so flipping it is O(N) work rather than the O(N^3) of a full
    eigendecomposition: measured 1.3 ms against 6.1 ms at n=100, 128 ms against 1471 ms
    at n=1000.

    Caveat: ``L |D| L^T`` is positive definite with the right inertia, but it is **not**
    ``|H|``.  The resulting direction has cos ~ 0.4-0.6 with the exact ``|H|^-1 grad``.
    Empirically this costs nothing -- it converges in slightly fewer iterations than the
    exact direction, and to a deeper gradient -- but it is an approximation, which is why
    ``eigh_floor`` is kept available as the validation reference.

    A Levenberg-Marquardt diagonal shift (``H + mu I``) is deliberately *not* offered.
    On this loss the most negative eigenvalue is up to 4.9x larger in magnitude than the
    largest positive one, so any ``mu`` achieving positive definiteness swamps the good
    curvature and the direction collapses to steepest descent (measured cos(p, -grad) =
    0.997-0.999).

    Args:
        H: (N, N) symmetric Hessian, N = n*d.
        grad: (N,) flattened gradient.
        method: "ldl_flip" (default) or "eigh_floor" (the O(N^3) reference).
        delta_ratio: Eigenvalues of |D| (or of H) are floored at
            ``delta_ratio * max|lambda|``, which also lifts the translation/rotation
            null modes.

    Returns:
        (N,) direction ``p`` with ``grad . p < 0``.
    """
    grad = np.ravel(grad)

    if method == "eigh_floor":
        evals, evecs = np.linalg.eigh(H)
        delta = delta_ratio * max(float(np.max(np.abs(evals))), 1e-300)
        inv = 1.0 / np.maximum(np.abs(evals), delta)
        return -(evecs * inv) @ (evecs.T @ grad)

    if method != "ldl_flip":
        raise ValueError(
            f"method must be 'ldl_flip' or 'eigh_floor', got {method!r}."
        )

    lu, d_blocks, perm = ldl(H, lower=True, check_finite=False)
    N = H.shape[0]
    dm = np.asarray(d_blocks, dtype=np.float64)
    scale = float(np.max(np.abs(np.diag(dm)))) or 1.0
    delta = delta_ratio * scale

    # Eigen-decompose the block-diagonal D one 1x1 or 2x2 block at a time (O(N) total),
    # inverting |eigenvalues| floored at delta.
    D_inv = np.zeros((N, N))
    i = 0
    while i < N:
        is_2x2 = (i + 1 < N) and (dm[i + 1, i] != 0.0 or dm[i, i + 1] != 0.0)
        if is_2x2:
            w, V = np.linalg.eigh(dm[i:i + 2, i:i + 2])
            w = np.maximum(np.abs(w), delta)
            D_inv[i:i + 2, i:i + 2] = (V / w) @ V.T
            i += 2
        else:
            D_inv[i, i] = 1.0 / max(abs(dm[i, i]), delta)
            i += 1

    # H = P L D L^T P^T  =>  |H|^-1 grad = P L^-T |D|^-1 L^-1 P^T grad
    L = lu[perm, :]
    y = solve_triangular(L, grad[perm], lower=True, unit_diagonal=True,
                         check_finite=False)
    z = D_inv @ y
    x = solve_triangular(L.T, z, lower=False, unit_diagonal=True, check_finite=False)
    p = np.empty_like(x)
    p[perm] = -x
    return p


############################################################
# Fused unpenalized loss + gradient
############################################################

def compute_loss_gradient(
    X_flat: np.ndarray,
    shape: tuple[int, ...],
    problem: dict,
) -> tuple[float, np.ndarray]:
    """Fused unpenalized loss and gradient -- the callable passed to L-BFGS-B.

    Distances, f and f' are computed once per slice and reused for both outputs.

    Args:
        X_flat: Flattened coordinates.
        shape: Shape to restore, (n, d) or (n, d, T).
        problem: {"R", "W", "link", optionally "valid_mask"}.

    Returns:
        (loss, flattened gradient).
    """
    X = X_flat.reshape(shape)
    R, W = problem["R"], problem["W"]
    link = _as_link(problem["link"])
    valid_mask = problem.get("valid_mask")
    if W is None:
        W = _default_weights_like(R)

    total_loss = 0.0
    grad = np.zeros(X.shape, dtype=np.float64)

    for t, X_t, R_t, W_t, mask_t in _iter_slices(X, R, W, valid_mask):
        # Shared quantities: distances and the link's value and slope
        D_t = compute_distances(X_t)
        f_t, f_prime_t = link.evaluate(D_t, derivatives=1)

        with np.errstate(divide="ignore", invalid="ignore"):
            terms = W_t * (f_t - R_t * np.log(f_t))
        loss_t = 0.5 * float(np.sum(terms[mask_t], dtype=np.float64))

        G = compute_gradient_weights(D_t, f_t, f_prime_t, R_t, W_t, mask_t, link.eps)
        grad_t = G.sum(axis=1)[:, None] * X_t - G @ X_t

        _assert_finite(loss_t, grad_t, t)
        total_loss += loss_t
        if t is None:
            grad = grad_t
        else:
            grad[:, :, t] = grad_t

    return total_loss, grad.ravel()


############################################################
# Penalty loss and gradient
############################################################

def compute_penalty_loss_gradient(
    X: np.ndarray, smoothing_params: dict
) -> tuple[float, np.ndarray]:
    """Group-fused-lasso penalty with pseudo-Huber smoothing, and its gradient.

    P(X) = sum_k alpha_1^(k) sum_{i,t} sqrt( ||(D_k X_i)_t||^2 + eps_k^2 )

    with gradient, writing u = D_k X_i,

        dP/dX_i = sum_k alpha_1^(k) * ( u / sqrt(||u||^2 + eps_k^2) ) @ D_k

    where ||u|| is the norm over the latent-dimension axis.  The pseudo-Huber form
    replaces the exact group norm, giving L-BFGS finite curvature near zero.

    Note: the ``sum_k eps_k`` constant offset inflates the reported penalty value by
    a known constant.  It does not affect the gradient or the minimizer.

    Args:
        X: (d, T) for one sample or (n, d, T) for all.
        smoothing_params: Keys "l1" ({order: alpha}), "eps" ({order: eps}),
            "diff_operators" ({order: matrix}).

    Returns:
        (loss, grad) with grad matching X's shape.
    """
    loss = 0.0
    grad = np.zeros_like(X, dtype=np.float64)
    eps_by_order = smoothing_params.get("eps", {})

    for k, D in smoothing_params["diff_operators"].items():
        l1 = smoothing_params["l1"].get(k, 0.0)
        if not l1:
            continue
        eps_k = eps_by_order.get(k, 0.0)

        DX = X @ D.T                                             # (..., d, T-k)
        sq_norms = np.sum(DX ** 2, axis=-2, keepdims=True)       # (..., 1, T-k)
        smoothed = np.sqrt(sq_norms + eps_k ** 2)                # (..., 1, T-k)

        loss += l1 * float(np.sum(smoothed))
        grad += l1 * (DX / smoothed) @ D

    return loss, grad


############################################################
# Penalized loss + gradient assembly
############################################################

def compute_penalized_loss_gradient(
    x_flat: np.ndarray,
    problem: dict,
) -> tuple[float, np.ndarray]:
    """Penalized loss and gradient on the packed coordinate vector.

    This is the callable passed to L-BFGS-B in the joint stage.  It unpacks the
    present coordinates, evaluates the data term slice by slice and the smoothing
    penalty per sample on that sample's present time window, then packs the gradient
    back down so absent entries never enter the optimizer's variable vector.

    Args:
        x_flat: Packed coordinates (see pack_coordinates).
        problem: {"R", "W", "link", "sampled_by_time", "shape" (n, d, T),
            "smoothing_params" (may be None), optionally "valid_mask"}.

    Returns:
        (loss, packed gradient).
    """
    n, d, T = problem["shape"]
    sampled_by_time = problem["sampled_by_time"]
    X = unpack_coordinates(x_flat, sampled_by_time, n, d, T)

    # Data term over all slices
    loss, grad_flat = compute_loss_gradient(X.ravel(), (n, d, T), problem)
    grad = grad_flat.reshape(n, d, T)

    # Smoothing penalty, per sample, on that sample's present window
    smoothing_params = problem.get("smoothing_params")
    if smoothing_params is not None:
        pen_loss, pen_grad = compute_penalty_over_samples(
            X, sampled_by_time, smoothing_params
        )
        loss += pen_loss
        grad += pen_grad

    _assert_finite(loss, grad)
    return loss, pack_coordinates(grad, sampled_by_time)


def compute_penalty_over_samples(
    X: np.ndarray,
    sampled_by_time: np.ndarray,
    smoothing_params: dict,
) -> tuple[float, np.ndarray]:
    """Apply the smoothing penalty to each sample over its present time window.

    Samples present at every timepoint share the full-length operators and are
    handled as one batch; ancient samples are grouped by the length of their present
    block, so samples sharing a sampling time reuse the same cached operator.

    Args:
        X: (n, d, T) coordinates.
        sampled_by_time: (n, T) presence mask.
        smoothing_params: As for compute_penalty_loss_gradient, plus an
            "operator_cache" entry used to memoize per-block-length operators.

    Returns:
        (penalty loss, (n, d, T) gradient).
    """
    n, d, T = X.shape
    loss = 0.0
    grad = np.zeros_like(X, dtype=np.float64)

    # Group samples by the length of their contiguous present block
    present_counts = sampled_by_time.sum(axis=1)
    for block_length in np.unique(present_counts):
        rows = np.where(present_counts == block_length)[0]
        if block_length == 0:
            continue
        params = _operators_for_block(smoothing_params, int(block_length))
        if not params["diff_operators"]:
            continue                        # window too short for any requested order

        if block_length == T:
            X_block = X[rows]                                   # (m, d, T)
        else:
            # Contiguity (asserted in pack_coordinates) makes this a simple slice
            starts = np.argmax(sampled_by_time[rows], axis=1)
            X_block = np.stack(
                [X[i, :, s:s + int(block_length)] for i, s in zip(rows, starts)]
            )

        block_loss, block_grad = compute_penalty_loss_gradient(X_block, params)
        loss += block_loss
        if block_length == T:
            grad[rows] += block_grad
        else:
            for row_pos, (i, s) in enumerate(zip(rows, starts)):
                grad[i, :, s:s + int(block_length)] += block_grad[row_pos]

    return loss, grad


def _operators_for_block(smoothing_params: dict, block_length: int) -> dict:
    """Smoothing params restricted to a present window of a given length.

    Operators are cached by (block_length, order) so that samples sharing a sampling
    time reuse the same matrix.  Orders with block_length <= k contribute no penalty
    and are dropped.
    """
    cache = smoothing_params.setdefault("operator_cache", {})
    if block_length in cache:
        operators = cache[block_length]
    else:
        max_order = max(smoothing_params["diff_operators"].keys())
        operators = compute_difference_operators(
            block_length, max_order, smoothing_params["delta_t"]
        )
        cache[block_length] = operators

    return {
        "l1": smoothing_params["l1"],
        "eps": smoothing_params.get("eps", {}),
        "diff_operators": operators,
    }


############################################################
# Sequential (per-slice) optimization
############################################################

def _resolve_gtol(grad_init: np.ndarray, gtol_factor: float) -> float:
    """Scale-aware gradient tolerance: a fraction of the initial gradient's size.

    An absolute gtol is meaningless here because the weights (and hence the
    gradients) are of order 1e6; see the module docstring.
    """
    scale = float(np.max(np.abs(grad_init)))
    return gtol_factor * scale if scale > 0 else 0.0


def _wolfe_step(
    x: np.ndarray,
    direction: np.ndarray,
    shape: tuple,
    problem: dict,
    f0: float,
    g0: np.ndarray,
    alpha0: float = 1.0,
    c1: float = 1e-4,
    c2: float = 0.9,
    amax: float = 1e6,
) -> tuple:
    """Strong-Wolfe line search along ``direction``, one fused loss+gradient call per trial.

    ``scipy.optimize.line_search`` takes separate ``f`` and ``g`` callables and therefore
    evaluates the objective twice per trial point.  At these problem sizes the line search
    dominates the per-slice runtime, so this implements the standard bracket/zoom
    (Nocedal & Wright Alg. 3.5-3.6) directly on ``compute_loss_gradient``, which already
    returns both.  Measured, that change took a test case from 120 s to 0.15 s.

    Args:
        x: (N,) flattened current point.
        direction: (N,) search direction.
        shape: Shape to restore for compute_loss_gradient.
        problem: {"R", "W", "link"} as elsewhere.
        f0, g0: Loss and flattened gradient at ``x`` (not recomputed).
        alpha0: First trial step.
        c1, c2: Armijo and curvature constants.
        amax: Cap on the step length.

    Returns:
        (alpha, x_new, f_new, g_new, n_eval); alpha == 0 signals failure to find a step,
        in which case the input point is returned unchanged.
    """
    slope0 = float(np.dot(g0, direction))
    if slope0 >= 0:                      # not a descent direction; nothing to search
        return 0.0, x, f0, g0, 0

    cache = {}

    def eval_at(a):
        if a not in cache:
            f, g = compute_loss_gradient(x + a * direction, shape, problem)
            cache[a] = (float(f), g, float(np.dot(g, direction)))
        return cache[a]

    def zoom(lo, hi, f_lo):
        for _ in range(40):
            a = 0.5 * (lo + hi)
            if abs(hi - lo) < 1e-16 * max(1.0, abs(hi)):
                break
            f_a, _, slope_a = eval_at(a)
            if not np.isfinite(f_a) or f_a > f0 + c1 * a * slope0 or f_a >= f_lo:
                hi = a
            else:
                if abs(slope_a) <= -c2 * slope0:
                    return a
                if slope_a * (hi - lo) >= 0:
                    hi = lo
                lo, f_lo = a, f_a
        return lo if lo > 0 else None

    a_prev, f_prev = 0.0, f0
    a = alpha0
    alpha = None
    for i in range(40):
        f_a, _, slope_a = eval_at(a)
        if not np.isfinite(f_a) or f_a > f0 + c1 * a * slope0 or (i > 0 and f_a >= f_prev):
            alpha = zoom(a_prev, a, f_prev)
            break
        if abs(slope_a) <= -c2 * slope0:
            alpha = a
            break
        if slope_a >= 0:
            alpha = zoom(a, a_prev, f_a)
            break
        a_prev, f_prev = a, f_a
        a = min(2.0 * a, amax)
        if a >= amax:
            alpha = amax
            break

    if alpha is None or not np.isfinite(alpha) or alpha <= 0:
        # Armijo backtracking fallback
        a = alpha0
        for _ in range(60):
            f_a, _, _ = eval_at(a)
            if np.isfinite(f_a) and f_a <= f0 + c1 * a * slope0:
                alpha = a
                break
            a *= 0.5
        else:
            return 0.0, x, f0, g0, len(cache)

    f_new, g_new, _ = eval_at(alpha)
    return alpha, x + alpha * direction, f_new, g_new, len(cache)


def _newton_slice_tolerance(
    X_t_init: np.ndarray,
    problem: dict,
    tol_factor: float,
    seed: Optional[int] = None,
) -> float:
    """Gradient tolerance that does not depend on the warm start's quality.

    ``_resolve_gtol`` anchors the tolerance to ``max|grad|`` *at the initial point*.  After
    a warm start that gradient is already small, so the tolerance tightens and loosens with
    the warm start: a slice that begins close to its optimum is permitted to stop close to
    where it began.  That is precisely the mechanism that entrenches a residual for the
    next slice to inherit -- the staircase.

    Instead anchor to the gradient scale of a *random* configuration at the same spatial
    scale (one extra gradient evaluation per slice), and floor the result at what float64
    can actually resolve.  The floor is required, not cosmetic: the gradient is a
    difference of terms of order |loss|, so ~eps*|loss| is its noise floor.  Without it,
    tolerances landed at 2.8e-17 on the ``split`` run and slices burned their whole
    iteration budget chasing round-off (92 s for 59 slices; 24x faster with the floor, at
    100% tolerance attainment).

    Args:
        X_t_init: (n, d) starting configuration, used only for its spatial scale.
        problem: {"R", "W", "link"}.
        tol_factor: Multiplier on the random-configuration gradient scale.
        seed: Seed for the probe configuration, so a sweep is reproducible.

    Returns:
        Absolute tolerance on max|grad|.
    """
    Xc = X_t_init - X_t_init.mean(axis=0)
    scale = float(np.sqrt((Xc ** 2).sum(axis=1).mean()))
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0

    probe = scale * np.random.default_rng(seed).standard_normal(X_t_init.shape)
    _, g_probe = compute_loss_gradient(probe.ravel(), X_t_init.shape, problem)
    g_scale = float(np.max(np.abs(g_probe)))

    loss_init, _ = compute_loss_gradient(X_t_init.ravel(), X_t_init.shape, problem)
    floor = GRAD_FLOOR_EPS * abs(float(loss_init))
    return max(tol_factor * g_scale, floor)


def optimize_timepoint_newton(
    X_t_init: np.ndarray,
    R_t: np.ndarray,
    W_t: Optional[np.ndarray] = None,
    link: Union[LinkFunction, str] = "inverse_distance",
    N_min: float = 1.0,
    N_max: float = np.inf,
    p: float = 1.0,
    max_iter: int = MAX_ITER_NEWTON,
    tol: Optional[float] = None,
    tol_factor: float = TOL_FACTOR_NEWTON,
    hessian_method: str = "ldl_flip",
    delta_ratio: float = 1e-8,
    tol_seed: Optional[int] = None,
) -> tuple:
    """Damped-Newton solve for one timepoint, on the exact Hessian.

    Per iteration: build the exact Hessian, modify it to positive definite via
    ``modified_newton_direction`` (``ldl_flip``), and take a strong-Wolfe step.  The
    factorization is rebuilt every iteration deliberately -- reusing one modified Hessian
    per slice (the Vladymyrov-style trick) was tested and is catastrophic on this loss
    (31,268 iterations against 151, 188x slower at n=100), because the curvature depends
    on X through f(d_ij) and changes fast.

    Against L-BFGS-B on the same problems, Newton reaches its tolerance on 100% of slices
    where L-BFGS reaches it on 8.5-20.3%: L-BFGS does not merely stop early, it *stalls*,
    halting at a median 167 of 4000 allowed iterations having missed its tolerance by a
    median factor of 15x, because a limited-memory curvature model cannot represent this
    problem's condition number of 1e6-1e9.

    Note that this does dense linear algebra per slice, so BLAS threading matters here in
    a way it did not for the L-BFGS path; see ``limit_blas_threads``.

    Args:
        X_t_init: (n, d) initial coordinates.
        R_t: (n, n) rates.
        W_t: (n, n) weights; defaults to DEFAULT_WEIGHTS.
        link: LinkFunction or link name (with N_min/p used to build one).
        N_min, N_max, p: Link parameters, used when ``link`` is a bare name.
        max_iter: Maximum Newton iterations.
        tol: Absolute tolerance on max|grad|; if None it is derived per slice by
            ``_newton_slice_tolerance``, which does not depend on the warm start.
        tol_factor: Multiplier used by that rule.
        hessian_method: "ldl_flip" (default) or "eigh_floor".
        delta_ratio: Eigenvalue floor ratio for the Hessian modification.
        tol_seed: Seed for the tolerance probe configuration.

    Returns:
        (X_opt, loss, n_iterations, message).
    """
    link = _as_link(link, N_min=N_min, p=p)
    if W_t is None:
        W_t = _default_weights_like(R_t)

    # 1/N_max keeps zero-rate pairs from carrying zero weight
    R_t = np.asarray(R_t, dtype=np.float64) + 1.0 / N_max
    problem = {"R": R_t, "W": np.asarray(W_t, dtype=np.float64), "link": link}
    shape = X_t_init.shape

    if tol is None:
        tol = _newton_slice_tolerance(X_t_init, problem, tol_factor, seed=tol_seed)

    x = np.asarray(X_t_init, dtype=np.float64).ravel().copy()
    f, g = compute_loss_gradient(x, shape, problem)
    f = float(f)

    n_iter = 0
    message = "converged: max|grad| <= tol"
    for n_iter in range(1, max_iter + 1):
        if np.max(np.abs(g)) <= tol:
            n_iter -= 1
            break
        H = compute_exact_hessian(x.reshape(shape), R_t, W_t, link=link)
        direction = modified_newton_direction(
            H, g, method=hessian_method, delta_ratio=delta_ratio
        )
        alpha, x_new, f_new, g_new, _ = _wolfe_step(
            x, direction, shape, problem, f, g
        )
        if alpha == 0.0:
            message = "stopped: line search failed to find a step"
            break
        x, f, g = x_new, f_new, g_new
    else:
        message = f"stopped: reached max_iter ({max_iter})"

    if np.max(np.abs(g)) <= tol:
        message = "converged: max|grad| <= tol"
    return x.reshape(shape), f, n_iter, message


def optimize_timepoint(
    X_t_init: np.ndarray,
    R_t: np.ndarray,
    W_t: Optional[np.ndarray] = None,
    link: Union[LinkFunction, str] = "inverse_distance",
    N_min: float = 1.0,
    N_max: float = np.inf,
    p: float = 1.0,
    max_iter: int = MAX_ITER_TIMEPOINT,
    ftol: float = FTOL_TIMEPOINT,
    gtol_factor: float = GTOL_FACTOR,
    maxcor: int = MAXCOR_TIMEPOINT,
    return_hess_inv: bool = False,
) -> tuple:
    """Single L-BFGS-B solve for one timepoint.

    Args:
        X_t_init: (n, d) initial coordinates.
        R_t: (n, n) rates.
        W_t: (n, n) weights; defaults to DEFAULT_WEIGHTS.
        link: LinkFunction or link name (with N_min/p used to build one).
        N_min, N_max, p: Link parameters, used when ``link`` is a bare name.
            1/N_max is added to the rates, so a pair with zero rate still carries
            weight.
        max_iter: Maximum L-BFGS-B iterations.
        ftol: Relative function tolerance (the primary criterion).
        gtol_factor: gtol is set to gtol_factor * max|grad| at the initial point.
        maxcor: L-BFGS memory (number of stored correction pairs).
        return_hess_inv: If True, also return the L-BFGS inverse Hessian.

    Returns:
        (X_opt, loss, n_iterations, message), plus hess_inv when requested.
    """
    link = _as_link(link, N_min=N_min, p=p)
    if W_t is None:
        W_t = _default_weights_like(R_t)

    # 1/N_max keeps zero-rate pairs from carrying zero weight
    R_t = np.asarray(R_t, dtype=np.float64) + 1.0 / N_max
    problem = {"R": R_t, "W": np.asarray(W_t, dtype=np.float64), "link": link}
    shape = X_t_init.shape

    _, grad_init = compute_loss_gradient(X_t_init.ravel(), shape, problem)
    gtol = _resolve_gtol(grad_init, gtol_factor)

    result = minimize(
        compute_loss_gradient,
        X_t_init.ravel(),
        args=(shape, problem),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iter, "disp": False, "gtol": gtol,
                 "ftol": ftol, "maxcor": maxcor},
    )
    X_opt = result.x.reshape(shape)
    if return_hess_inv:
        return X_opt, result.fun, result.nit, result.message, result.hess_inv
    return X_opt, result.fun, result.nit, result.message


def coordinate_descent_over_time(
    R: np.ndarray,
    W: Optional[np.ndarray] = None,
    link: Union[LinkFunction, str] = "inverse_distance",
    N_min: float = 1.0,
    N_max: float = np.inf,
    p: float = 1.0,
    n_components: int = 2,
    X_init: Optional[np.ndarray] = None,
    sampled_by_time: Optional[np.ndarray] = None,
    direction: str = "backward",
    max_iter: int = MAX_ITER_TIMEPOINT,
    target_precision: float = FTOL_TIMEPOINT,
    gtol_factor: float = GTOL_FACTOR,
    maxcor: int = MAXCOR_TIMEPOINT,
    return_covariances: bool = False,
    exact_hessian: bool = True,
    solver: str = "newton",
    extrapolate_warm_start: bool = True,
    tol_factor: float = TOL_FACTOR_NEWTON,
) -> tuple:
    """Single-pass coordinate descent over timepoints (default: backward).

    Each slice is solved unpenalized, warm-started from the previous slice's solution,
    then Procrustes-aligned to it.  Alignment uses only the samples present at *both*
    slices, so the growing set of ancient samples does not translate the configuration
    for reasons unrelated to structure.

    Two properties of this loss shape the default solver.  It is badly conditioned
    (per-slice Hessian condition number 1e6-1e9, worsening toward the present, because
    with ``f = 1/(N_min + d^p)`` the latent distances near the present are large and the
    curvature ~ d^-p is minute), and its Hessian is indefinite.  L-BFGS-B stalls on that
    conditioning -- it reaches its tolerance on 8.5-20.3% of slices against Newton's
    100% -- and since each slice warm-starts from the previous one, the un-converged
    residual is inherited and discharged in occasional jumps, producing staircase-like
    trajectories.  ``solver="newton"`` (the default) replaces it with damped Newton on
    the exact Hessian; ``solver="lbfgs"`` keeps the previous behaviour.

    Args:
        R: (n, n, T) rate tensor.
        W: (n, n, T) weight tensor; defaults to DEFAULT_WEIGHTS.
        link: LinkFunction or link name.
        N_min, N_max, p: Link parameters used when ``link`` is a bare name.
        n_components: Latent dimensionality d.
        X_init: (n, d, T) initial coordinates; random if None.
        sampled_by_time: (n, T) presence mask; all-present if None.
        direction: "backward" (default) or "forward".
        max_iter: Maximum iterations per slice; the L-BFGS default (MAX_ITER_TIMEPOINT)
            is used for solver="lbfgs" and MAX_ITER_NEWTON for solver="newton".
        target_precision: ftol, solver="lbfgs" only; 1e-9 for exploratory sweeps,
            1e-12 for final runs.
        gtol_factor: Scale factor for the per-slice gtol, solver="lbfgs" only.
        maxcor: L-BFGS memory, solver="lbfgs" only.
        return_covariances: If True, also return per-sample Laplace covariances,
            shape (n, d, d, T).
        exact_hessian: If True (default), use the exact analytical Hessian for those
            covariances; if False, materialise the L-BFGS approximation instead
            (faster but significantly less accurate).  Only meaningful together with
            ``return_covariances``; with solver="newton" the L-BFGS inverse Hessian is
            unavailable, so exact_hessian=False falls back to the exact one.
        solver: "newton" (default) for damped Newton on the exact Hessian, or "lbfgs"
            for the previous scipy L-BFGS-B path.
        extrapolate_warm_start: If True (default), start each slice at the linear
            extrapolation ``2 X(t+1) - X(t+2)`` rather than at ``X(t+1)``, falling back
            to the plain warm start for the first two slices and whenever the
            extrapolated point has the higher loss.  This was the single largest
            smoothness gain measured (motion_top5 0.254 -> 0.098 with Newton, 0.254 ->
            0.152 with L-BFGS) and helps both solvers, but it costs a little per-slice
            fit and overlaps with what the stage-2 smoothing penalty does explicitly,
            so it is switchable.
        tol_factor: Multiplier on the random-configuration gradient scale that sets each
            slice's tolerance, solver="newton" only (see ``_newton_slice_tolerance``).

    Returns:
        (X, loss_df) or (X, loss_df, covariances).
    """
    if direction not in ("backward", "forward"):
        raise ValueError(f"direction must be 'backward' or 'forward', got {direction!r}.")
    if solver not in ("newton", "lbfgs"):
        raise ValueError(f"solver must be 'newton' or 'lbfgs', got {solver!r}.")

    link = _as_link(link, N_min=N_min, p=p)
    n_samples, _, n_timepoints = R.shape

    if W is None:
        W = _default_weights_like(R)
    if sampled_by_time is None:
        sampled_by_time = np.ones((n_samples, n_timepoints), dtype=bool)
    if X_init is None:
        X_init = np.random.rand(n_samples, n_components, n_timepoints)

    if solver == "newton":
        newton_max_iter = MAX_ITER_NEWTON if max_iter == MAX_ITER_TIMEPOINT else max_iter
        print(
            f"Sequential (per-slice) stage: damped Newton (exact Hessian, ldl_flip), "
            f"tol=max(tol_factor*|grad| at a random config, {GRAD_FLOOR_EPS:g}*|loss|) "
            f"with tol_factor={tol_factor:g}, max_iter={newton_max_iter}, "
            f"extrapolated warm start={extrapolate_warm_start}"
        )
    else:
        print(
            f"Sequential (per-slice) stage: L-BFGS-B, ftol={target_precision:g}, "
            f"gtol=gtol_factor*max|grad_init| with gtol_factor={gtol_factor:g}, "
            f"maxcor={maxcor}, extrapolated warm start={extrapolate_warm_start}"
        )

    X = np.full((n_samples, n_components, n_timepoints), np.nan)
    loss_before, loss_after, n_iters, messages = [], [], [], []
    if return_covariances:
        covariances = np.full(
            (n_samples, n_components, n_components, n_timepoints), np.nan
        )

    time_order = (range(n_timepoints) if direction == "forward"
                  else range(n_timepoints - 1, -1, -1))
    prev_t_idx = None
    prev2_t_idx = None
    n_extrapolated = 0

    for step, t_idx in enumerate(tqdm(time_order)):
        extant_mask = sampled_by_time[:, t_idx]
        extant_idx = np.where(extant_mask)[0]

        # --- Initialize from the previous slice, seeding newly-entering samples ---
        if step == 0:
            X_global = X_init[:, :, t_idx].copy()
        else:
            X_global = X[:, :, prev_t_idx].copy()
            new_mask = extant_mask & ~sampled_by_time[:, prev_t_idx]
            if np.any(new_mask):
                X_global[new_mask, :] = X_init[new_mask, :, t_idx]

        # --- Restrict to extant samples ---
        X_t = X_global[extant_idx, :]
        R_t = np.asarray(R[:, :, t_idx][np.ix_(extant_idx, extant_idx)], dtype=np.float64)
        W_t = np.asarray(W[:, :, t_idx][np.ix_(extant_idx, extant_idx)], dtype=np.float64)

        problem_t = {"R": R_t + 1.0 / N_max, "W": W_t, "link": link}

        # --- Extrapolate the warm start along time: 2 X(t+1) - X(t+2) ---
        # The sweep currently starts each slice where the configuration was one step ago,
        # which is systematically behind a smoothly drifting trajectory; the first-order
        # Taylor step projects the last observed velocity forward.  Extrapolation happens
        # here, before the solve, on the already Procrustes-aligned previous solutions.
        # Two guards: no t+2 yet for the first two slices, and local linearity in time
        # breaks exactly at splits, bottleneck edges and admixture pulses -- the features
        # the method exists to detect -- so fall back whenever the extrapolated point has
        # the higher loss (one extra loss evaluation per slice).
        if extrapolate_warm_start and prev2_t_idx is not None:
            usable = extant_mask & sampled_by_time[:, prev_t_idx] & sampled_by_time[:, prev2_t_idx]
            if np.any(usable):
                X_cand = X_t.copy()
                extrap_in_extant = np.isin(extant_idx, np.where(usable)[0])
                cand_rows = np.where(usable)[0]
                X_cand[extrap_in_extant, :] = (
                    2.0 * X[cand_rows, :, prev_t_idx] - X[cand_rows, :, prev2_t_idx]
                )
                loss_cand, _ = compute_loss_gradient(
                    X_cand.ravel(), X_cand.shape, problem_t
                )
                loss_plain, _ = compute_loss_gradient(
                    X_t.ravel(), X_t.shape, problem_t
                )
                if np.isfinite(loss_cand) and loss_cand <= loss_plain:
                    X_t = X_cand
                    n_extrapolated += 1

        # --- Loss before ---
        loss_t_before, _ = compute_loss_gradient(X_t.ravel(), X_t.shape, problem_t)

        # --- Optimize ---
        hess_inv = None
        if solver == "newton":
            X_t, loss_t_after, n_iter, message = optimize_timepoint_newton(
                X_t_init=X_t, R_t=R_t, W_t=W_t, link=link, N_max=N_max,
                max_iter=newton_max_iter, tol_factor=tol_factor, tol_seed=int(t_idx),
            )
        else:
            need_hess_inv = return_covariances and not exact_hessian
            X_t, loss_t_after, n_iter, message, *_hess = optimize_timepoint(
                X_t_init=X_t, R_t=R_t, W_t=W_t, link=link, N_max=N_max,
                max_iter=max_iter, ftol=target_precision, gtol_factor=gtol_factor,
                maxcor=maxcor, return_hess_inv=need_hess_inv,
            )
            hess_inv = _hess[0] if _hess else None

        covs = None
        if return_covariances:
            # The L-BFGS inverse-Hessian approximation only exists on the L-BFGS path;
            # with Newton the exact Hessian is the only option available.
            covs_exact = exact_hessian or solver == "newton"
            try:
                covs = extract_sample_covariances(
                    X_t, R_t + 1.0 / N_max, W_t, link=link,
                    hess_inv=hess_inv, exact=covs_exact,
                )
            except np.linalg.LinAlgError as err:
                raise np.linalg.LinAlgError(
                    f"Covariance extraction failed at timepoint index {t_idx} "
                    f"({len(extant_idx)} extant samples): {err}"
                ) from err

        # --- Center (first step only) and Procrustes align to the previous slice ---
        if step == 0:
            X_t = X_t - np.mean(X_t, axis=0)
            X_t, rot = procrustes_single_time(
                X_global[extant_idx, :], X_t, allow_reflection=False
            )
        else:
            # Align on samples present at both slices, so the centering set does not
            # change as ancient samples enter and translate the whole configuration
            shared_mask = extant_mask & sampled_by_time[:, prev_t_idx]
            if shared_mask.sum() < n_components + 1:
                raise ValueError(
                    f"Only {int(shared_mask.sum())} samples are present at both "
                    f"timepoint indices {prev_t_idx} and {t_idx}; at least "
                    f"{n_components + 1} are needed for a non-degenerate Procrustes "
                    "rotation in {n_components} dimensions."
                )
            shared_in_extant = np.isin(extant_idx, np.where(shared_mask)[0])
            X_t, rot = procrustes_align_subset(
                X_reference=X_global[extant_idx, :],
                Y=X_t,
                subset=shared_in_extant,
                allow_reflection=False,
            )
        X[extant_idx, :, t_idx] = X_t

        if return_covariances and covs is not None:
            # x_rotated = x @ rot  →  Cov(x_rotated) = rot.T @ Cov(x) @ rot
            covariances[extant_idx, :, :, t_idx] = np.array([rot.T @ c @ rot for c in covs])

        loss_before.append(loss_t_before)
        loss_after.append(loss_t_after)
        n_iters.append(n_iter)
        messages.append(message)
        prev2_t_idx, prev_t_idx = prev_t_idx, t_idx

    if extrapolate_warm_start:
        print(f"  extrapolated warm start accepted on {n_extrapolated} of "
              f"{n_timepoints} slices")

    loss_df = pd.DataFrame({
        "step":           range(1, n_timepoints + 1),
        "loss_reduction": np.array(loss_before) - np.array(loss_after),
        "n_iterations":   n_iters,
        "message":        messages,
    })
    if return_covariances:
        return X, loss_df, covariances
    return X, loss_df


############################################################
# Joint optimization
############################################################

def optimize_jointly(
    X_init: np.ndarray,
    R: np.ndarray,
    W: Optional[np.ndarray] = None,
    link: Union[LinkFunction, str] = "inverse_distance",
    N_min: float = 1.0,
    N_max: float = np.inf,
    p: float = 1.0,
    smoothing_params: Optional[dict] = None,
    sampled_by_time: Optional[np.ndarray] = None,
    max_iter: int = MAX_ITER_JOINT,
    target_precision: float = FTOL_JOINT,
    gtol_factor: float = GTOL_FACTOR,
    maxcor: int = MAXCOR_JOINT,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Joint L-BFGS-B optimization over all samples and timepoints, with smoothing.

    The optimizer works on a packed vector of present coordinates only, so absent
    ancient-sample entries never enter the variable vector.  The returned array keeps
    the public (n, d, T) contract, with NaN at absent entries.

    Args:
        X_init: (n, d, T) initial coordinates (typically the stage-1 solution).
        R: (n, n, T) rate tensor.
        W: (n, n, T) weight tensor; defaults to DEFAULT_WEIGHTS.
        link: LinkFunction or link name.
        N_min, N_max, p: Link parameters used when ``link`` is a bare name.
        smoothing_params: Smoothing configuration; None for no penalty.
        sampled_by_time: (n, T) presence mask; all-present if None.
        max_iter: Maximum iterations.
        target_precision: ftol; 1e-9 for exploratory sweeps, 1e-12 for final runs.
        gtol_factor: gtol is gtol_factor * max|grad| at the initial point.
        maxcor: L-BFGS memory.  Memory cost is O(maxcor * n * d * T) -- at n=1000,
            d=2, T=700 with maxcor=20 that is roughly 450 MB.

    Returns:
        (X_opt, progress_df) with progress columns
        ["iteration", "loss", "proj_grad_max", "rel_change"].
    """
    link = _as_link(link, N_min=N_min, p=p)
    n, d, T = X_init.shape
    if W is None:
        W = _default_weights_like(R)
    if sampled_by_time is None:
        sampled_by_time = np.ones((n, T), dtype=bool)

    problem = {
        "R": np.asarray(R, dtype=np.float64) + 1.0 / N_max,
        "W": np.asarray(W, dtype=np.float64),
        "link": link,
        "sampled_by_time": sampled_by_time,
        "shape": (n, d, T),
        "smoothing_params": smoothing_params,
    }
    if smoothing_params is not None and "delta_t" not in smoothing_params:
        raise ValueError(
            "smoothing_params must carry 'delta_t' so that difference operators can "
            "be rebuilt on each ancient sample's present time window."
        )

    # Absent entries hold NaN in the public array; pack drops them
    x0 = pack_coordinates(np.nan_to_num(X_init, nan=0.0), sampled_by_time)

    # Cache the last evaluation so the progress callback need not recompute it
    state: dict = {"loss": None, "grad_max": None}

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        loss, grad = compute_penalized_loss_gradient(x, problem)
        state["loss"] = loss
        state["grad_max"] = float(np.max(np.abs(grad)))
        return loss, grad

    _, grad_init = compute_penalized_loss_gradient(x0, problem)
    gtol = _resolve_gtol(grad_init, gtol_factor)
    print(
        f"Joint stage: ftol={target_precision:g}, gtol={gtol:g} "
        f"(gtol_factor={gtol_factor:g}), maxcor={maxcor}"
    )

    progress: list[dict] = []

    def callback(xk: np.ndarray) -> None:
        # Read the cached values; recomputing here would double the cost of the
        # most expensive stage
        loss, grad_max = state["loss"], state["grad_max"]
        prev_loss = progress[-1]["loss"] if progress else None
        rel_change = (
            (prev_loss - loss) / max(abs(prev_loss), abs(loss), 1)
            if prev_loss is not None else np.nan
        )
        progress.append({
            "iteration":     len(progress) + 1,
            "loss":          loss,
            "proj_grad_max": grad_max,
            "rel_change":    rel_change,
        })
        print(f"  iter {progress[-1]['iteration']:4d}  "
              f"loss={loss:.10g}  "
              f"rel_change={rel_change:.3g}  "
              f"max|grad|={grad_max:.6g}")

    result = minimize(
        objective,
        x0,
        method="L-BFGS-B",
        jac=True,
        callback=callback,
        options={"maxiter": max_iter, "disp": False, "gtol": gtol,
                 "ftol": target_precision, "maxcor": maxcor},
    )
    print(f"Finished: {result.message}  "
          f"(loss={result.fun:.10g}, iterations={result.nit})")

    # Restore the public (n, d, T) contract: NaN at absent entries
    X_opt = unpack_coordinates(result.x, sampled_by_time, n, d, T, fill=np.nan)
    return X_opt, pd.DataFrame(progress)


############################################################
# Non-optimization utilities
############################################################

def extract_sample_covariances(
    X_t: np.ndarray,
    R_t: np.ndarray,
    W_t: Optional[np.ndarray] = None,
    link: Union[LinkFunction, str] = "inverse_distance",
    N_min: float = 1.0,
    p: float = 1.0,
    hess_inv=None,
    exact: bool = True,
) -> np.ndarray:
    """Extract per-sample d x d Laplace covariance blocks at one slice.

    Args:
        X_t: (n, d) coordinates at the optimum.
        R_t: (n, n) rate matrix used in optimization.
        W_t: (n, n) weight matrix.
        link: LinkFunction or link name; must match the one used in optimization.
        N_min, p: Link parameters used when ``link`` is a bare name.
        hess_inv: L-BFGS inverse Hessian (LbfgsInvHessProduct); used when exact=False.
        exact: If True (default), compute the exact Hessian analytically and
            pseudoinvert it.  If False, materialise the L-BFGS approximation.

    Returns:
        (n, d, d) array of per-sample covariance blocks.
    """
    link = _as_link(link, N_min=N_min, p=p)
    n, d = X_t.shape
    if exact:
        H = compute_exact_hessian(X_t, R_t, W_t, link=link)
        # Eigendecompose (symmetric) and invert only positive eigenvalues.  This
        # handles both the d translation-invariance zeros and any small negative
        # eigenvalues arising from numerical imprecision.
        eigvals, eigvecs = np.linalg.eigh(H)
        threshold = np.max(eigvals) * n * d * np.finfo(float).eps * 1e3
        inv_eigvals = np.where(eigvals > threshold, 1.0 / eigvals, 0.0)
        H_inv = (eigvecs * inv_eigvals) @ eigvecs.T
    else:
        if hess_inv is None:
            raise ValueError("exact=False requires hess_inv from the L-BFGS result.")
        H_inv = hess_inv @ np.eye(n * d)
    return np.array([H_inv[i * d:(i + 1) * d, i * d:(i + 1) * d] for i in range(n)])


def procrustes_single_time(
    X_reference: np.ndarray, Y: np.ndarray, allow_reflection: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Align Y to X_reference via the optimal rotation (Procrustes).

    Args:
        X_reference: (n, d) reference configuration.
        Y: (n, d) configuration to rotate.
        allow_reflection: If False, force a proper rotation (det = +1).

    Returns:
        (Y_rotated, rotation_matrix).
    """
    X_reference_centered = X_reference - np.mean(X_reference, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)

    cov_matrix = np.dot(Y_centered.T, X_reference_centered)
    U, S, Vt = np.linalg.svd(cov_matrix)

    if not allow_reflection:
        d = np.sign(np.linalg.det(np.dot(U, Vt)))
        D = np.diag([1] * (U.shape[1] - 1) + [d])
        rotation_matrix = np.dot(U, np.dot(D, Vt))
    else:
        rotation_matrix = np.dot(U, Vt)

    Y_rotated = np.dot(Y_centered, rotation_matrix)
    return Y_rotated, rotation_matrix


def procrustes_align_subset(
    X_reference: np.ndarray,
    Y: np.ndarray,
    subset: np.ndarray,
    allow_reflection: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Procrustes alignment estimated on a subset of rows, applied to all rows.

    Used when consecutive time slices share only some of their samples: the centroids
    and rotation are estimated from the shared rows only, then the same rigid
    transform is applied to every row of Y.

    Args:
        X_reference: (n, d) reference configuration.
        Y: (n, d) configuration to align.
        subset: (n,) boolean mask of rows to estimate the transform from.
        allow_reflection: If False, force a proper rotation (det = +1).

    Returns:
        (Y_aligned, rotation_matrix).
    """
    ref_centroid = np.mean(X_reference[subset], axis=0)
    y_centroid = np.mean(Y[subset], axis=0)

    cov_matrix = np.dot((Y[subset] - y_centroid).T, X_reference[subset] - ref_centroid)
    U, S, Vt = np.linalg.svd(cov_matrix)

    if not allow_reflection:
        sign = np.sign(np.linalg.det(np.dot(U, Vt)))
        D = np.diag([1] * (U.shape[1] - 1) + [sign])
        rotation_matrix = np.dot(U, np.dot(D, Vt))
    else:
        rotation_matrix = np.dot(U, Vt)

    # Apply the shared-sample transform to every row
    return np.dot(Y - y_centroid, rotation_matrix), rotation_matrix


def procrustes_align_trajectories(
    X_reference: np.ndarray, X: np.ndarray, allow_reflections: bool = True
) -> np.ndarray:
    """Align a trajectory X (n, d, T) to a reference, frame by frame.

    Frame 0 of X is aligned onto frame 0 of X_reference; every other frame is then
    aligned onto that aligned frame 0 (not onto the reference directly).

    Args:
        X_reference: (n, d, T_ref); only index 0 is used.
        X: (n, d, T).
        allow_reflections: Passed to procrustes_single_time.

    Returns:
        (n, d, T) aligned trajectory.
    """
    n_timepoints = X.shape[2]
    X_aligned = np.zeros_like(X)

    # Step 1: align frame 0 of X onto frame 0 of X_reference
    X0_aligned, _ = procrustes_single_time(
        X_reference[:, :, 0], X[:, :, 0], allow_reflection=allow_reflections
    )
    X_aligned[:, :, 0] = X0_aligned

    # Step 2: align all remaining frames onto the now-aligned frame 0
    for t in range(1, n_timepoints):
        Y_aligned, _ = procrustes_single_time(
            X0_aligned, X[:, :, t], allow_reflection=allow_reflections
        )
        X_aligned[:, :, t] = Y_aligned

    return X_aligned


def procrustes_align_trajectories_df(
    df_reference: pd.DataFrame,
    df_X: pd.DataFrame,
    id_column: str = "individual_id",
    allow_reflections: bool = True,
) -> pd.DataFrame:
    """Dataframe wrapper around procrustes_align_trajectories.

    df_reference is treated as time-varying, but only its first timepoint (the row
    with the minimum "time" value) is used as the reference frame.  Matching is done
    by id_column, at the first timepoint and across all timepoints of df_X.

    Args:
        df_reference: Columns [id_column, "time", "X_1", ..., "X_d"].
        df_X: Same schema.
        id_column: Identifier column.
        allow_reflections: Passed through to the alignment.

    Returns:
        df_X's schema with coordinate columns replaced by aligned values.
    """
    coord_cols = [c for c in df_X.columns if c not in (id_column, "time")]

    times = sorted(df_X["time"].unique())
    n_timepoints = len(times)

    # sample_id order is fixed by the first timepoint of df_X
    first_frame = df_X[df_X["time"] == times[0]]
    sample_ids = first_frame[id_column].to_numpy()
    n = len(sample_ids)
    d = len(coord_cols)

    # Build (n, d, T) array for df_X, enforcing consistent id order per frame
    X = np.zeros((n, d, n_timepoints))
    for i, t in enumerate(times):
        frame = df_X[df_X["time"] == t].set_index(id_column)
        try:
            X[:, :, i] = frame.loc[sample_ids, coord_cols].to_numpy()
        except KeyError as e:
            raise ValueError(
                f"df_X at time={t} is missing sample_id(s) present at time={times[0]}: {e}"
            )

    # Build the reference's first timepoint in the same id order
    ref_first_time = df_reference["time"].min()
    ref_frame = df_reference[df_reference["time"] == ref_first_time].set_index(id_column)
    try:
        X_ref0 = ref_frame.loc[sample_ids, coord_cols].to_numpy()
    except KeyError as e:
        raise ValueError(
            f"df_reference at time={ref_first_time} is missing sample_id(s) "
            f"present in df_X: {e}"
        )
    X_reference = X_ref0[:, :, np.newaxis]  # (n, d, 1)

    X_aligned = procrustes_align_trajectories(
        X_reference, X, allow_reflections=allow_reflections
    )

    # Reassemble into a dataframe with the same schema as df_X
    aligned_frames = []
    for i, t in enumerate(times):
        frame = df_X[df_X["time"] == t].set_index(id_column).loc[sample_ids].reset_index()
        frame[coord_cols] = X_aligned[:, :, i]
        aligned_frames.append(frame)

    result = pd.concat(aligned_frames, axis=0, ignore_index=True)
    return result[df_X.columns.tolist()]


def compute_distance_df(embeddings: pd.DataFrame, id_label: str = "sample_id") -> pd.DataFrame:
    """Pairwise Euclidean distances between samples at each timepoint.

    Args:
        embeddings: Long-format embeddings with columns id_label, "time", and one
            column per latent dimension (named "X...").
        id_label: Column identifying samples/individuals.

    Returns:
        Distances indexed by (id_label_1, id_label_2), columns are time points.
    """
    x_cols = sorted(c for c in embeddings.columns if c.startswith("X"))

    # Pivot each dimension to (n_individuals, n_times), then stack to
    # (n_individuals, n_times, n_dims). Avoids the O(n^2*T) cross-join.
    pivots = [
        embeddings.pivot_table(index=id_label, columns="time", values=c, aggfunc="first")
        for c in x_cols
    ]
    individuals = pivots[0].index.values
    times = pivots[0].columns.values
    X = np.stack([p.values for p in pivots], axis=-1)  # (n, n_times, n_dims)

    ii, jj = np.triu_indices(len(individuals), k=1)
    distances = np.linalg.norm(X[ii] - X[jj], axis=-1)  # (n_pairs, n_times)

    pair_index = pd.MultiIndex.from_arrays(
        [individuals[ii], individuals[jj]],
        names=[f"{id_label}_1", f"{id_label}_2"],
    )
    return pd.DataFrame(distances, index=pair_index, columns=times)


def reconstruct_rates(
    D_df: pd.DataFrame,
    N_min: float = 1.0,
    p: float = 1.0,
    link: Union[LinkFunction, str] = "inverse_distance",
) -> pd.DataFrame:
    """Apply the link function to a pairwise distance dataframe to recover rates.

    The link evaluates on the underlying array, so the pair index and time columns are
    restored here to keep the result aligned with the input rates dataframe.
    """
    link = _as_link(link, N_min=N_min, p=p)
    return pd.DataFrame(link.f(D_df.values), index=D_df.index, columns=D_df.columns)
