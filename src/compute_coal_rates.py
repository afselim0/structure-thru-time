#%%
#!/usr/bin/env python3
import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
import os


MIN_RATE = 1e-20


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
    if minimum_num_idxs < 7:
        minimum_num_idxs = 7
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

    # S is banded; return CSR so that H @ S.T exploits the sparsity.
    return sparse.csr_matrix(S)


def compute_coal_rates(
    pdf_df: pd.DataFrame,
    minimum_window_size: float,
    maximum_window_size: float,
    phased: bool = False,
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
    phased : bool
        If True, return per-sample-pair rates.  If False, average rates over
        all cross-individual sample pairs to return per-individual-pair rates.
    sample_individal_ids : pd.DataFrame or None
        Required when ``phased=False``.  Must have two columns, ``sample_id``
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
        or individual depending on ``phased``); columns are the same time points
        as ``pdf_df``.  Values are clipped to MIN_RATE to suppress numerical
        artefacts from non-monotone empirical survival estimates.
    """
    times     = pdf_df.columns.astype(float)
    time_step = times[1] - times[0]
    n_T       = len(times)

    # S(t) = 1 - CDF(t) is the probability of not having coalesced by the right
    # edge of bin t.  H(t) = -log S(t) is the cumulative hazard at that edge.
    survival_df = 1 - pdf_df.cumsum(axis=1)
    H = -np.log(survival_df.values.clip(1e-10))   # (n_pairs, n_T); clip avoids -inf at S=0

    S = filter_matrix
    if S is None:
        S = adaptive_local_linear_filter(minimum_window_size, maximum_window_size, n_T, time_step)

    # rates = H @ S.T.  Compute as (S @ H.T).T so the sparse operand is on the
    # left of the product (the efficient form for CSR) and the result is a dense
    # ndarray of shape (n_pairs, n_T).
    rates = np.asarray((S @ H.T).T)

    rates_df = pd.DataFrame(rates, columns=times, index=pdf_df.index)

    # Clip negative rates caused by numerical noise in non-monotone empirical H(t)
    rates_df = rates_df.clip(lower=MIN_RATE)

    if not phased:
        if sample_individal_ids is None or "individual_id" not in sample_individal_ids.columns:
            # ASSUMPTION: sample IDs are zero-indexed consecutive integers; IDs
            # 2k and 2k+1 belong to individual k.  Will silently mis-assign
            # individuals if IDs are non-contiguous or non-zero-indexed.
            # print(
            #     "Mapping from sample IDs to individual IDs not provided. "
            #     "Assuming consecutive sample IDs (0,1), (2,3), ... belong to "
            #     "the same diploid individual."
            # )
            sample_ids = rates_df.index.values
            sample_ids = np.unique([item for sample_id in sample_ids for item in sample_id])
            individual_ids = sample_ids // 2
            # Two-column form expected by compute_between_individual_rates.
            sample_individal_ids = pd.DataFrame(
                {"sample_id": sample_ids, "individual_id": individual_ids}
            )
        rates_df = compute_between_individual_rates(rates_df, sample_individal_ids)
    return rates_df


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

def compute_coal_rates_bootstrap_se(
    bootstrap_dir: str,
    minimum_window_size: float,
    maximum_window_size: float,
    n_replicates: int = 100,
    subsample_size: int = None,
    phased: bool = False,
    sample_individal_ids: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Estimate the SE of full-data coalescent rate estimates by delete-d jackknife.

    The data are split across ``N`` count files (one per contig/tree file) in
    ``bootstrap_dir``.  For each of ``n_replicates`` replicates, ``subsample_size``
    (= ``m``) files are drawn **without replacement**; their counts and coverage
    are pooled into a bootstrap PDF, from which rates are computed via
    ``compute_coal_rates``.

    The raw across-replicate variance estimates the variance of an ``m``-contig
    rate estimate, not of the full ``N``-contig estimate that the caller actually
    reports.  The delete-d jackknife (Shao & Wu 1989, ``d = N - m`` deleted per
    replicate) rescales it to the full-data variance:

        Var[r_N] = (m / (N - m)) * Var_replicates[r_m]

    The ``m / (N - m)`` factor simultaneously rescales for the data budget and
    cancels the finite-population correction of without-replacement sampling, so
    the returned SE is the SE of the estimate computed on **all** ``N`` contigs.

    Validity requires both ``m`` and ``d = N - m`` to be reasonably large (the
    1/n variance scaling must hold); ``m`` near ``N/2`` is the robust default.
    The scaling treats each contig as an exchangeable unit of information; if
    contig coverages are highly heterogeneous the contig-count scaling is only
    approximate.

    Parameters
    ----------
    bootstrap_dir : str
        Directory containing per-file ``.parquet`` count files and a
        ``coverage.parquet`` file, as written by
        ``CountCoalEvents.compute_coal_time_pdf``.  The pair index and time grid
        are taken from these files.
    minimum_window_size : float
        Passed to ``compute_coal_rates``.
    maximum_window_size : float
        Passed to ``compute_coal_rates``.
    n_replicates : int
        Number of jackknife replicates (Monte Carlo subsamples).
    subsample_size : int
        Number of count files (``m``) drawn without replacement per replicate.
        Must satisfy ``0 < m < N``; ``m ~ N/2`` recommended.
    phased : bool
        Passed to ``compute_coal_rates``.  If False, the returned SE DataFrame
        has an individual-pair index.
    sample_individal_ids : pd.DataFrame or None
        Passed to ``compute_coal_rates``.  Required when ``phased=False`` and
        the sample-to-individual mapping is non-trivial.

    Returns
    -------
    pd.DataFrame
        SE of the full-data rate estimates, with the same (pair) index and
        (time) columns as the output of ``compute_coal_rates``.
    """
    coverage_df = pd.read_parquet(os.path.join(bootstrap_dir, "coverage.parquet"))

    # Discover available count files (exclude coverage.parquet)
    available_files = sorted(
        f for f in os.listdir(bootstrap_dir)
        if f.endswith(".parquet") and f != "coverage.parquet"
    )
    file_basenames = [f.removesuffix(".parquet") for f in available_files]
    n_files = len(available_files)

    if n_files == 0:
        raise ValueError(f"No count parquet files found in {bootstrap_dir}.")

    missing = set(file_basenames) - set(coverage_df.columns)
    if missing:
        raise ValueError(f"coverage_df is missing columns for files: {missing}")

    if subsample_size is None:
        subsample_size = n_files // 2
    # Delete-d jackknife needs 0 < m < N so that d = N - m > 0.
    if not (0 < subsample_size < n_files):
        raise ValueError(
            f"subsample_size must satisfy 0 < subsample_size < n_files; got "
            f"subsample_size={subsample_size}, n_files={n_files}. (m ~ n_files/2 recommended.)"
        )

    # Take the pair index and time grid from the first count file; all files and
    # coverage_df are then aligned to this index so row order is consistent.
    template = pd.read_parquet(os.path.join(bootstrap_dir, available_files[0]))
    pair_index = template.index
    times      = template.columns.astype(float)
    n_T        = len(times)
    n_sample_pairs = len(pair_index)

    coverage_df = coverage_df.loc[pair_index, file_basenames]
    coverage_vals = coverage_df.values  # (n_sample_pairs, n_files)

    # The derivative-filter matrix depends only on the time grid, so build it once
    # and reuse it across all replicates rather than rebuilding it each call.
    time_step = times[1] - times[0]
    filter_matrix = adaptive_local_linear_filter(
        minimum_window_size, maximum_window_size, n_T, time_step
    )

    # Load all count files into (n_files, n_sample_pairs, n_T), reindexing each
    # file to the template pair index so row order is guaranteed consistent.
    print("Loading all bootstrap count files into memory …")
    all_counts = np.empty((n_files, n_sample_pairs, n_T))
    for i, fname in enumerate(tqdm(available_files, desc="Loading count files")):
        file_df = pd.read_parquet(os.path.join(bootstrap_dir, fname))
        all_counts[i] = file_df.loc[pair_index].values

    replicate_rates_list: list[np.ndarray] = []

    for _ in tqdm(range(n_replicates), desc="Bootstrap replicates"):
        sampled_idxs = np.random.choice(n_files, size=subsample_size, replace=False)

        # Sum counts and coverage over sampled files
        counts_sum = all_counts[sampled_idxs].sum(axis=0)       # (n_sample_pairs, n_T)
        total_len  = coverage_vals[:, sampled_idxs].sum(axis=1)  # (n_sample_pairs,)

        with np.errstate(invalid="ignore"):
            pdf_rep = pd.DataFrame(
                counts_sum / total_len[:, np.newaxis],
                index=pair_index,
                columns=times,
            )

        rates_rep = compute_coal_rates(
            pdf_rep, minimum_window_size, maximum_window_size, phased,
            sample_individal_ids, filter_matrix=filter_matrix,
        )
        replicate_rates_list.append(rates_rep.values)

    # Stack replicates; index comes from the last replicate (consistent across all)
    replicate_rates = np.stack(replicate_rates_list, axis=0).astype(float)  # (n_replicates, n_pairs, n_T)

    # Delete-d jackknife: rescale the m-contig replicate variance to the full-N
    # variance by the factor m / (N - m), then take the square root for the SE.
    m = subsample_size
    N = n_files
    subsample_var = replicate_rates.var(axis=0, ddof=1)   # Var[r_m] across replicates
    se = np.sqrt((m / (N - m)) * subsample_var)            # SE of the full-N estimate

    return pd.DataFrame(se, columns=times, index=rates_rep.index)
