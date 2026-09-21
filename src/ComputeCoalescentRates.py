#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import os

MIN_RATE = 1e-16

def adaptive_local_linear_filter(
    minimum_window_size: float,
    maximum_window_size: float,
    n_timepoints: int,
    time_step: float,
) -> np.ndarray:
    """Build the (n_timepoints, n_timepoints) derivative-filter matrix S.

    Row t of S contains the local-linear regression weights for estimating
    d/dt H(t), where H is the cumulative hazard.  The bandwidth (number of
    bins used) grows linearly from ``minimum_window_size`` at early times to
    ``maximum_window_size`` at late times.

    Three regimes:
    - Interior [t_min, t_max): symmetric window, bandwidth grows with t.
    - Early boundary [0, t_min): fixed minimum window, one-sided fit.
    - Late boundary [t_max, n_timepoints): fixed maximum window, one-sided fit.

    The late-boundary loop reuses the full maximum window starting at index
    ``n_timepoints - maximum_num_idxs``, which is the same window as the last
    interior point — the transition is therefore smooth.

    Parameters
    ----------
    minimum_window_size : float
        Minimum bandwidth in the same time units as ``time_step``.
    maximum_window_size : float
        Maximum bandwidth in the same time units as ``time_step``.
    n_timepoints : int
        Number of time bins.
    time_step : float
        Uniform spacing between bins (in generations or years).

    Returns
    -------
    S : np.ndarray, shape (n_timepoints, n_timepoints)
        Derivative-filter matrix.  ``rates = H @ S.T`` gives the estimated
        hazard rate at each time point for each pair.  The matrix is banded
        (bandwidth <= maximum window size) and so is a natural sparse candidate,
        but at production sizes the sparse form was measured only ~1.1x faster
        than the dense matmul, which is not worth the extra type handling; it is
        returned dense.  Callers may still pass a sparse matrix in, and both
        entry points densify it.
    """
    minimum_num_idxs = int(minimum_window_size // time_step)
    if minimum_num_idxs % 2 == 0:
        minimum_num_idxs += 1
    if minimum_num_idxs < 3:
        minimum_num_idxs = 3
    maximum_num_idxs = int(maximum_window_size // time_step)
    if maximum_num_idxs % 2 == 0:
        maximum_num_idxs += 1
    if maximum_num_idxs < minimum_num_idxs:
        maximum_num_idxs = minimum_num_idxs

    t_min = minimum_num_idxs // 2
    t_max = n_timepoints - maximum_num_idxs // 2

    # The late-boundary rows are written into S[t, start:] with
    # start = n_timepoints - maximum_num_idxs, so the grid must hold at least one
    # full maximum window. t_max <= t_min alone does not catch this: for
    # n_timepoints in [18, 20] with these windows t_max > t_min, but start < 0 and
    # the slice silently shrinks, raising an opaque numpy broadcast error instead.
    if n_timepoints < maximum_num_idxs or t_max <= t_min:
        raise ValueError(
            "Time series too short for the requested window sizes: "
            f"n_timepoints={n_timepoints} must be at least maximum_window_size / "
            f"time_step = {maximum_num_idxs} bins (and more than "
            f"{minimum_num_idxs // 2 + maximum_num_idxs // 2}). Either lengthen the "
            "time grid or reduce maximum_window_size."
        )

    # One window size per interior time point; cast to int after linspace to
    # avoid fractional bandwidths.  An extra +1 parity correction is applied
    # inside the loop to keep all windows odd-sized.
    window_sizes = np.linspace(minimum_num_idxs, maximum_num_idxs, t_max - t_min).astype(int)

    S = np.zeros((n_timepoints, n_timepoints))

    # Interior: symmetric window growing from minimum to maximum bandwidth
    for k, t in enumerate(range(t_min, t_max)):
        m = window_sizes[k]
        if m % 2 == 0:
            m += 1
        h    = (m - 1) // 2
        xloc = np.arange(-h, h + 1) * time_step   # S1 = 0 by symmetry
        S2   = (xloc ** 2).sum()
        S[t, t - h : t + h + 1] = xloc / S2        # slope:     j*dt / S2

    # Early-time boundary (t < t_min): one fixed window [0, minimum_num_idxs),
    # slope evaluated at each t within it via one-sided local linear fit.
    for t in range(t_min):
        xloc = (np.arange(minimum_num_idxs) - t) * time_step
        S1 = xloc.sum()
        S2 = (xloc ** 2).sum()
        n  = minimum_num_idxs
        S[t, :minimum_num_idxs] = (n * xloc - S1) / (n * S2 - S1 ** 2)

    # Late-time boundary (t >= t_max): one fixed window [n_timepoints - maximum_num_idxs, n_timepoints),
    # slope evaluated at each t within it via one-sided local linear fit.
    start = n_timepoints - maximum_num_idxs
    for t in range(t_max, n_timepoints):
        xloc = (np.arange(maximum_num_idxs) - (t - start)) * time_step
        S1 = xloc.sum()
        S2 = (xloc ** 2).sum()
        n  = maximum_num_idxs
        S[t, start:] = (n * xloc - S1) / (n * S2 - S1 ** 2)

    return S


def _odd_window_bins(window_size: float, time_step: float) -> int:
    """Window size in whole, odd bins -- the same rounding the filter applies.

    Mirrors ``adaptive_local_linear_filter``'s own conversion so that callers can
    predict the bandwidth it will use without rebuilding the matrix.
    """
    n = int(window_size // time_step)
    if n % 2 == 0:
        n += 1
    return max(n, 3)


def slice_pdf_for_rates(
    pdf_df: pd.DataFrame,
    rates_min_time: float,
    rates_max_time: float,
    minimum_window_size: float,
    maximum_window_size: float,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Widen the rate interval by the filter's boundary width, for later clipping.

    ``adaptive_local_linear_filter`` has three regimes: rows below ``t_min`` all
    share one fixed minimum window, rows at or above ``t_max`` all share one fixed
    maximum window, and only the rows between get a symmetric window whose
    bandwidth ramps from ``minimum_window_size`` to ``maximum_window_size``. Rates
    on the boundary rows are therefore constant-bandwidth artefacts of where the
    grid happens to stop, not estimates at the requested bandwidth.

    Padding the slice by the two boundary half-widths and discarding the pad
    afterwards makes every KEPT timepoint an interior, symmetric-window row, with
    bandwidth ``minimum_window_size`` at ``rates_min_time`` and
    ``maximum_window_size`` at ``rates_max_time``. That is what lets one expensive
    pdf serve many rate intervals: the interval's ends are no longer special.

    The pads are ``min_bins // 2`` young and ``max_bins // 2`` old, in whole bins.
    The old pad is HALF the maximum window, not the full window: a full window
    pushes ``t_max`` about ``max_bins / 2`` rows past the last kept row, so the
    bandwidth ramp is still climbing where the kept region ends and the last kept
    row never reaches ``maximum_window_size``. Measured support width at the last
    kept row, target in brackets -- full window vs. this half window:

        dt=40  max_ws=4000   91 [101]  vs  101 [101]
        dt=4   max_ws=200     49 [51]  vs   51 [51]
        dt=4   max_ws=200 (short interval)
                              43 [51]  vs   51 [51]

    Do not "fix" this back to the full window; it is half the padding and the only
    one of the two that hits the requested bandwidth.

    When the pdf does not extend far enough the slice is truncated to what exists
    and a message names the short end; the filter's boundary rows then leak into
    the kept region at that end, which is the pre-existing behaviour rather than an
    error. Asking for rates from the very start of the pdf is the ordinary case of
    this -- the young pad has nowhere to go -- so it is reported as a note rather
    than a warning.

    Survival baseline
    -----------------
    Downstream, ``S(t) = 1 - cumsum(pdf)``. A plain slice would restart that cumsum
    from zero and so report every pair as having not yet coalesced at the start of
    the slice, inflating S by the mass below it. That does NOT cancel in the hazard
    ``-d/dt log S``: the inflation factor differs per pair AND across time within a
    pair (measured spread 0.089 over one slice), leaving errors ~6e-5 in the rates.

    So the mass dropped below the slice is folded into its first bin. Survival
    depends only on CUMULATIVE mass, so this makes ``1 - cumsum`` exact at every
    column of the slice, while keeping the grid uniform -- which prepending a
    baseline column would not. Only the first column's own bin mass is altered, and
    that column is inside the pad, never kept. Verified against full-grid rates at
    a constant bandwidth: 1.6e-18.

    Parameters
    ----------
    pdf_df : pd.DataFrame
        Estimated PDF of pairwise coalescent times, columns being float time
        points on a uniform grid.
    rates_min_time, rates_max_time : float
        The interval rates are wanted on. Both should land on the pdf grid; the
        nearest grid point is used.
    minimum_window_size, maximum_window_size : float
        Local-linear bandwidths at ``rates_min_time`` and ``rates_max_time``
        respectively, in the same time units as the columns.

    Returns
    -------
    padded_pdf : pd.DataFrame
        ``pdf_df`` restricted to the padded interval, columns unchanged, with the
        mass below the slice folded into its first bin (see above). It is a valid
        input to ``compute_coal_rates`` / ``compute_coal_rate_weights`` but its
        first bin is no longer that bin's own mass, so it is not a pdf of the
        interval and should not be interpreted as one.
    keep_mask : np.ndarray of bool, shape ``(padded_pdf.shape[1],)``
        True on the columns inside ``[rates_min_time, rates_max_time]``. Apply the
        SAME mask to the rates and to the weights -- the embeddings loss pairs a
        rate with its weight by position, so a divergence would be silent.
    """
    times = np.asarray(pdf_df.columns, dtype=float)
    if len(times) < 2:
        raise ValueError(f"pdf must have at least two time columns (got {len(times)}).")
    time_step = times[1] - times[0]
    _assert_uniform_grid(times, time_step)

    min_bins = _odd_window_bins(minimum_window_size, time_step)
    # Mirrors the filter's own clamp (maximum_num_idxs < minimum_num_idxs).
    max_bins = max(_odd_window_bins(maximum_window_size, time_step), min_bins)
    pad_young, pad_old = min_bins // 2, max_bins // 2

    lo = int(np.argmin(np.abs(times - rates_min_time)))
    hi = int(np.argmin(np.abs(times - rates_max_time)))
    if hi <= lo:
        raise ValueError(
            f"rates interval [{rates_min_time}, {rates_max_time}] selects "
            f"{hi - lo + 1} pdf column(s) on a grid spanning "
            f"[{times[0]:g}, {times[-1]:g}]; it must span at least two."
        )

    start, stop = lo - pad_young, hi + pad_old + 1      # stop exclusive
    short_young, short_old = max(0, -start), max(0, stop - len(times))

    if short_young:
        # Wanting rates from the very start of the pdf is the ordinary case: the
        # pad is unobtainable by construction, not by misconfiguration.
        lead = ("Note" if lo - pad_young < 0 and lo == 0 else "Warning")
        print(
            f"{lead}: the pdf starts at t={times[0]:g}, which is "
            f"{short_young * time_step:g} short of the {pad_young * time_step:g} "
            f"of padding wanted below rates_min_time={rates_min_time:g}. Padding as "
            "far as the pdf allows; the youngest kept timepoints will use the "
            "filter's fixed minimum window instead of a symmetric one."
        )
    if short_old:
        print(
            f"Warning: the pdf ends at t={times[-1]:g}, which is "
            f"{short_old * time_step:g} short of the {pad_old * time_step:g} of "
            f"padding wanted above rates_max_time={rates_max_time:g}. Padding as far "
            "as the pdf allows; the oldest kept timepoints will use the filter's "
            "fixed maximum window, i.e. less smoothing than max_window_size asks for."
        )

    start, stop = max(0, start), min(len(times), stop)

    padded_pdf = pdf_df.iloc[:, start:stop].copy()
    if start > 0:
        # Carry the survival baseline (see "Survival baseline" above). NaN rows --
        # pairs observed nowhere genome-wide -- stay NaN, as they must.
        dropped_mass = pdf_df.iloc[:, :start].sum(axis=1)
        padded_pdf.iloc[:, 0] += dropped_mass

    keep_mask = np.zeros(stop - start, dtype=bool)
    keep_mask[lo - start : hi - start + 1] = True
    return padded_pdf, keep_mask


def _assert_uniform_grid(times: np.ndarray, time_step: float) -> None:
    """The derivative filter is built on a constant bin width; enforce it.

    ``adaptive_local_linear_filter`` constructs each row's local abscissa as
    multiples of a single ``time_step``, so a non-uniform grid silently yields
    the wrong slope rather than an error.
    """
    if not np.allclose(np.diff(times), time_step, rtol=1e-9):
        raise ValueError(
            "Time grid must be uniform; the derivative filter assumes constant "
            f"spacing (got steps in [{np.diff(times).min():.6g}, {np.diff(times).max():.6g}])."
        )


def _filter_tail_sums(filter_matrix: np.ndarray, time_step: float) -> np.ndarray:
    """Phi_a(r) = sum_{b >= r} F[a, b], with the filter's own identities checked.

    Summing the estimator by parts turns the filter's action on the cumulative
    hazard into an action on its increments, and Phi is the resulting
    coefficient.  The tail is inclusive of b = r because H(t_b) already contains
    bin b's increment.

    Two identities every derivative filter must satisfy are free here: Phi_a(0)
    is the full row sum, so a filter that fails to annihilate constants is
    caught directly, and dt * sum_r Phi_a(r) is the unit derivative response.
    The second catches boundary-row normalisation errors that the first passes
    straight through.
    """
    phi = np.cumsum(filter_matrix[:, ::-1], axis=1)[:, ::-1]

    if not np.allclose(phi[:, 0], 0.0, atol=1e-10):
        raise ValueError(
            "Filter rows must sum to zero (a derivative annihilates constants); "
            f"max |row sum| = {np.abs(phi[:, 0]).max():.3e}."
        )
    unit_mass = time_step * phi.sum(axis=1)
    if not np.allclose(unit_mass, 1.0, atol=1e-8):
        raise ValueError(
            "Filter must have unit derivative response (dt * sum_r Phi_a(r) == 1); "
            f"unit mass ranges [{unit_mass.min():.6f}, {unit_mass.max():.6f}]."
        )
    return phi


def _pooled_survival(
    pdf_df: pd.DataFrame,
    per_haplotype: bool,
    sample_individal_ids: pd.DataFrame | None,
) -> pd.DataFrame:
    """S(t) = 1 - CDF(t), pooled to individual pairs unless per_haplotype.

    Pooling happens on the survival, before any non-linear step.  The pooled
    survival is a linear statistic of the underlying data, so the four
    haplotype pairs' shared-haplotype and shared-tree correlations are carried
    automatically; averaging afterwards would assume independence.

    The hazard of the pooled survival is the mixture hazard
    sum_pq S_pq lambda_pq / sum_pq S_pq -- a survival-weighted mean of the four
    haplotype-pair rates rather than their plain mean.  Rates and weights are
    both derived from this one object so they share an estimand by construction.
    """
    survival_df = 1 - pdf_df.cumsum(axis=1)

    # 1 - cumsum can drift slightly negative once the pdf sums marginally above
    # 1. A negative weight would flip the sign of its loss term, turning
    # attraction into repulsion, so the drift is caught rather than clipped
    # blindly; a genuinely exhausted pair floors at exactly zero.
    survival = survival_df.values
    if np.any(survival < -1e-9):
        raise ValueError(
            f"Survival below zero (min {survival.min():.3e}); the PDF sums to more "
            "than 1. Check the CDF construction before forming rates or weights."
        )
    survival_df = pd.DataFrame(np.clip(survival, 0.0, 1.0),
                               index=survival_df.index, columns=survival_df.columns)

    if per_haplotype:
        return survival_df

    if sample_individal_ids is None or "individual_id" not in sample_individal_ids.columns:
        sample_ids = np.unique([s for pair in survival_df.index.values for s in pair])
        # Two-column form expected by compute_between_individual_rates.
        sample_individal_ids = pd.DataFrame(
            {"sample_id": sample_ids, "individual_id": sample_ids // 2}
        )
    return compute_between_individual_rates(survival_df, sample_individal_ids)


def compute_coal_rates(
    pdf_df: pd.DataFrame,
    minimum_window_size: float,
    maximum_window_size: float,
    per_haplotype: bool = False,
    sample_individal_ids: pd.DataFrame | None = None,
    filter_matrix: sparse.spmatrix | None = None,
) -> pd.DataFrame:
    """Compute pairwise coalescent rates from a PDF of pairwise coalescent times.

    For each pair, the cumulative hazard H(t) = -log S(t) is computed from the
    empirical survival S(t) = 1 - CDF(t).  The instantaneous rate at each time
    point is then estimated as dH/dt via a local linear fit whose bandwidth grows
    from ``minimum_window_size`` to ``maximum_window_size`` (see
    ``adaptive_local_linear_filter``).

    When ``per_haplotype`` is False the four haplotype pairs of an individual
    pair are pooled on the *survival* before the derivative is taken, so the
    estimand is the mixture hazard sum_pq S_pq lambda_pq / sum_pq S_pq rather
    than the plain mean of the four haplotype-pair rates.  The two agree when
    the four survivals agree and diverge in deep time and for admixed
    individuals; on the split demography the measured divergence is a median of
    1.4e-5 (max 1.1e-3).  ``compute_coal_rate_weights`` pools identically, so
    rates and weights share one estimand by construction.

    Parameters
    ----------
    pdf_df : pd.DataFrame
        Estimated PDF of pairwise coalescent times.  Index is a MultiIndex of
        (sample_id_1, sample_id_2) pairs; columns are float time points.
    minimum_window_size : float
        Minimum local-linear bandwidth (same units as column time points).
    maximum_window_size : float
        Maximum local-linear bandwidth (same units as column time points).
    per_haplotype : bool
        If True, return per-sample-pair rates.  If False, pool the survivals of
        all cross-individual sample pairs and return per-individual-pair rates
        (the mixture hazard; see above).
    sample_individal_ids : pd.DataFrame or None
        Required when ``per_haplotype=False``.  Must have two columns, ``sample_id``
        and ``individual_id``.  If None, consecutive sample IDs are assumed to
        belong to the same diploid individual (IDs 0–1 → individual 0, IDs 2–3 →
        individual 1, etc.) — this assumption is only valid for zero-indexed,
        contiguous sample IDs.
    filter_matrix : scipy.sparse.spmatrix or None
        Precomputed derivative-filter matrix from
        ``adaptive_local_linear_filter``.  If None it is built internally.  Pass
        a precomputed matrix when calling repeatedly on the same time grid (e.g.
        in the bootstrap loop) to avoid rebuilding it each time.

    Returns
    -------
    rates_df : pd.DataFrame
        Estimated coalescent rates.  Index is a MultiIndex of pair IDs (sample
        or individual depending on ``per_haplotype``); columns are the same time points
        as ``pdf_df``.  Values are clipped to MIN_RATE to suppress numerical
        artefacts from non-monotone empirical survival estimates.
    """
    times     = pdf_df.columns.astype(float)
    time_step = times[1] - times[0]
    n_T       = len(times)
    _assert_uniform_grid(times, time_step)

    # S(t) = 1 - CDF(t) is the probability of not having coalesced by the right
    # edge of bin t, pooled over haplotype pairs first so that the rate and its
    # weight share one estimand.  H(t) = -log S(t) is the cumulative hazard.
    survival_df = _pooled_survival(pdf_df, per_haplotype, sample_individal_ids)
    H = -np.log(survival_df.values.clip(1e-16))   # (n_pairs, n_T); clip avoids -inf at S=0

    S = filter_matrix
    if S is None:
        S = adaptive_local_linear_filter(minimum_window_size, maximum_window_size, n_T, time_step)
    S = np.asarray(S.todense()) if sparse.issparse(S) else np.asarray(S)

    rates = H @ S.T

    # Clip negative rates caused by numerical noise in non-monotone empirical H(t).
    # Uses np.clip on the ndarray rather than DataFrame.clip: with numpy 2.4.6 and
    # pandas 2.2.3, DataFrame.clip(lower=...) has been observed to zero out valid
    # positive values well above the bound (a pandas/numpy version-mismatch bug).
    rates = np.clip(rates, a_min=MIN_RATE, a_max=None)

    return pd.DataFrame(rates, columns=times, index=survival_df.index)

def compute_coal_rate_weights(
    pdf_df: pd.DataFrame,
    minimum_window_size: float,
    maximum_window_size: float,
    per_haplotype: bool = False,
    sample_individal_ids: pd.DataFrame | None = None,
    filter_matrix: sparse.spmatrix | None = None,
) -> pd.DataFrame:
    """Compute the loss weights that accompany the rates from ``compute_coal_rates``.

    The pseudo-likelihood weight is w = lambda / Var(lambda_hat).  Propagating
    the Poisson weight s * L(t) through the derivative filter F and summing by
    parts gives Var(lambda_hat(t_a)) = dt * sum_r Phi_a(r)^2 lambda(t_r)/L(t_r)
    with Phi_a(r) = sum_{b >= r} F[a,b].  Under a locally constant rate the
    lambda cancels against the numerator, leaving

        w_ij(t_a) = S_hat_ij(t_a) / (dt * sum_r Phi_a(r)^2)

    up to a global constant, which is irrelevant because only relative weights
    affect the minimiser.  The weight therefore depends on the survival alone
    and never on the rate estimate, so a pair with no observed coalescence keeps
    full weight instead of the 0/0 the plug-in jackknife weight produced there.

    S_hat <= 1 and the filter term is fixed, so the weight is bounded above by
    construction and needs no capping.

    Parameters
    ----------
    pdf_df : pd.DataFrame
        Estimated PDF of pairwise coalescent times, as passed to
        ``compute_coal_rates``.  Index is a MultiIndex of (sample_id_1,
        sample_id_2) pairs; columns are float time points.
    minimum_window_size, maximum_window_size : float
        Local-linear bandwidths, matching the rates they weight.
    per_haplotype : bool
        If True, return per-sample-pair weights.  If False, the four haplotype
        pairs of an individual pair are pooled at the *survival* level before
        the weight is formed, so the shared-haplotype and shared-tree
        correlations are carried automatically.  Averaging the weights instead
        would assume independence and overstate the information by up to 4x.
    sample_individal_ids : pd.DataFrame or None
        Two columns, ``sample_id`` and ``individual_id``.  If None, consecutive
        sample IDs are assumed to pair into diploids, as in ``compute_coal_rates``.
    filter_matrix : scipy.sparse.spmatrix or None
        Precomputed filter from ``adaptive_local_linear_filter``; built
        internally when None.  Pass the same matrix used for the rates so the
        two share a bandwidth schedule.

    Returns
    -------
    weights_df : pd.DataFrame
        Weights with the same index and columns as the corresponding rates.
    """
    times     = pdf_df.columns.astype(float)
    time_step = times[1] - times[0]
    n_T       = len(times)
    _assert_uniform_grid(times, time_step)

    survival_df = _pooled_survival(pdf_df, per_haplotype, sample_individal_ids)

    S = filter_matrix
    if S is None:
        S = adaptive_local_linear_filter(minimum_window_size, maximum_window_size, n_T, time_step)
    S = np.asarray(S.todense()) if sparse.issparse(S) else np.asarray(S)

    phi = _filter_tail_sums(S, time_step)
    phi_sq_mass = time_step * (phi ** 2).sum(axis=1)          # (n_T,)
    if np.any(phi_sq_mass <= 0):
        raise ValueError(
            "A filter row has zero Phi^2 mass, so its effective bandwidth is "
            "undefined; the derivative filter has an all-zero row."
        )

    # B_eff is an effective bandwidth (a time), the smoother's analogue of the
    # window s in the exact Poisson weight s * L(t). Written this way the
    # correspondence is visible: survival stands in for the at-risk count L,
    # effective bandwidth for s.
    effective_bandwidth = 1.0 / phi_sq_mass
    weights = survival_df.values * effective_bandwidth[np.newaxis, :]
    return pd.DataFrame(weights, columns=times, index=survival_df.index)

def compute_between_individual_rates(
    rates_df: pd.DataFrame,
    sample_individal_ids: pd.DataFrame,
) -> pd.DataFrame:
    """Average per-sample-pair rates to produce per-individual-pair rates.

    For each ordered pair (ind1, ind2) with ind1 < ind2, all cross-individual
    sample-pair rates are averaged.

    Parameters
    ----------
    rates_df : pd.DataFrame
        Per-sample-pair rates.  Index is a MultiIndex of (sample_id_1,
        sample_id_2) tuples; columns are time points.
    sample_individal_ids : pd.DataFrame
        Must have two columns, ``sample_id`` and ``individual_id``, mapping each
        sample to its individual.

    Returns
    -------
    between_individual_rates : pd.DataFrame
        Per-individual-pair rates with a MultiIndex of (individual_1,
        individual_2) and the same time-point columns.

    Raises
    ------
    KeyError
        If any cross-individual sample pair is absent from ``rates_df``. Every
        individual pair must average over the same number of sample pairs, so a
        missing one is an input error rather than something to average around:
        dropping it would silently change the estimand for that pair alone.
    """
    active_samples_pairs = rates_df.index.values
    active_samples = np.unique([item for pair in active_samples_pairs for item in pair])

    # Look up the individual_id of every active sample via the sample_id column
    active_map = sample_individal_ids[sample_individal_ids["sample_id"].isin(active_samples)]
    active_individuals = active_map["individual_id"].unique()
    active_individuals_pairs = [(i1, i2) for i1 in active_individuals for i2 in active_individuals if i1 < i2]

    # Pre-allocate as float; an empty DataFrame defaults to object dtype, which
    # would propagate through .values and break downstream numerical ufuncs.
    between_individual_rates = pd.DataFrame(
        np.empty((len(active_individuals_pairs), len(rates_df.columns)), dtype=float),
        columns=rates_df.columns,
        index=pd.MultiIndex.from_tuples(active_individuals_pairs, names=["individual_1", "individual_2"]),
    )
    for ind1, ind2 in active_individuals_pairs:
        samples_ind1 = sample_individal_ids.loc[sample_individal_ids["individual_id"] == ind1, "sample_id"].values
        samples_ind2 = sample_individal_ids.loc[sample_individal_ids["individual_id"] == ind2, "sample_id"].values
        # rates_df is indexed on sample pairs in (min, max) order, so the lookup
        # key must be sorted by SAMPLE id. Iterating individuals in order does not
        # give that: an individual whose samples have the higher ids yields
        # reversed keys, which are absent from the index.
        pairs = [
            (min(s1, s2), max(s1, s2))
            for s1 in samples_ind1 for s2 in samples_ind2
        ]
        # skipna=False: a NaN sample pair (one with zero genome-wide coverage)
        # must surface as NaN rather than be quietly averaged out, which would
        # pool this individual pair over fewer sample pairs than the others.
        between_individual_rates.loc[(ind1, ind2)] = rates_df.loc[pairs].mean(skipna=False)
    return between_individual_rates