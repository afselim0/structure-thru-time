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
    S : scipy.sparse.csr_matrix, shape (n_timepoints, n_timepoints)
        Derivative-filter matrix.  ``rates = H @ S.T`` gives the estimated
        hazard rate at each time point for each pair.  Returned as a sparse CSR
        matrix because S is banded (bandwidth <= maximum window size), which
        makes the matmul against the (n_pairs, n_T) hazard matrix much cheaper
        than the dense O(n_pairs * n_T^2) product.
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

    if t_max <= t_min:
        raise ValueError("Time series too short for the requested window sizes.")

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
        If True, return per-sample-pair rates.  If False, average rates over
        all cross-individual sample pairs to return per-individual-pair rates.
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

    # S(t) = 1 - CDF(t) is the probability of not having coalesced by the right
    # edge of bin t.  H(t) = -log S(t) is the cumulative hazard at that edge.
    survival_df = 1 - pdf_df.cumsum(axis=1)
    H = -np.log(survival_df.values.clip(1e-16))   # (n_pairs, n_T); clip avoids -inf at S=0

    S = filter_matrix
    if S is None:
        S = adaptive_local_linear_filter(minimum_window_size, maximum_window_size, n_T, time_step)

    # rates = H @ S.T.  Compute as (S @ H.T).T so the sparse operand is on the
    # left of the product (the efficient form for CSR) and the result is a dense
    # ndarray of shape (n_pairs, n_T).
    rates = H @ S.T

    # Clip negative rates caused by numerical noise in non-monotone empirical H(t).
    # Uses np.clip on the ndarray rather than DataFrame.clip: with numpy 2.4.6 and
    # pandas 2.2.3, DataFrame.clip(lower=...) has been observed to zero out valid
    # positive values well above the bound (a pandas/numpy version-mismatch bug).
    rates = np.clip(rates, a_min=MIN_RATE, a_max=None)

    rates_df = pd.DataFrame(rates, columns=times, index=pdf_df.index)

    if not per_haplotype:
        if sample_individal_ids is None or "individual_id" not in sample_individal_ids.columns:
            sample_ids = rates_df.index.values
            sample_ids = np.unique([item for sample_id in sample_ids for item in sample_id])
            individual_ids = sample_ids // 2
            # Two-column form expected by compute_between_individual_rates.
            sample_individal_ids = pd.DataFrame(
                {"sample_id": sample_ids, "individual_id": individual_ids}
            )
        rates_df = compute_between_individual_rates(rates_df, sample_individal_ids)
    return rates_df

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
        internally when None.  Pass the same matrix used for the rates.

    Returns
    -------
    weights_df : pd.DataFrame
        Weights with the same index and columns as the corresponding rates.
    """
    times     = pdf_df.columns.astype(float)
    time_step = times[1] - times[0]
    n_T       = len(times)

    survival_df = 1 - pdf_df.cumsum(axis=1)

    # Pool haplotype pairs on the survival, before forming the weight: the
    # pooled survival is a linear statistic of each block's data, so its
    # sampling variance carries the haplotype-pair covariance with no
    # independence assumption.
    if not per_haplotype:
        if sample_individal_ids is None or "individual_id" not in sample_individal_ids.columns:
            sample_ids = np.unique([s for pair in survival_df.index.values for s in pair])
            sample_individal_ids = pd.DataFrame(
                {"sample_id": sample_ids, "individual_id": sample_ids // 2}
            )
        survival_df = compute_between_individual_rates(survival_df, sample_individal_ids)

    S = filter_matrix
    if S is None:
        S = adaptive_local_linear_filter(minimum_window_size, maximum_window_size, n_T, time_step)
    S = np.asarray(S.todense()) if sparse.issparse(S) else np.asarray(S)

    # Phi_a(r) = sum_{b >= r} F[a,b], the coefficient on the hazard increment
    # dLambda(t_r) after summing by parts.  The tail is inclusive of b = r
    # because H(t_b) = -log(1 - CDF(t_b)) already contains bin b's increment.
    phi = np.cumsum(S[:, ::-1], axis=1)[:, ::-1]
    filter_variance = time_step * (phi ** 2).sum(axis=1)      # (n_T,)
    if np.any(filter_variance <= 0):
        raise ValueError(
            "A filter row has zero Phi^2 mass, so its weight is undefined. This "
            "means the derivative filter has an all-zero row; check the window sizes."
        )

    weights = survival_df.values / filter_variance[np.newaxis, :]
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

    Notes
    -----
    If any cross-individual sample pair is absent from ``rates_df`` (e.g.,
    because only i < j pairs were computed), it is silently dropped from the
    mean.  Ensure ``rates_df`` contains all required pairs before calling.
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
        pairs = [(s1, s2) for s1 in samples_ind1 for s2 in samples_ind2]
        # Pairs absent from rates_df are silently dropped; see docstring note.
        between_individual_rates.loc[(ind1, ind2)] = rates_df.loc[pairs].mean()
    return between_individual_rates