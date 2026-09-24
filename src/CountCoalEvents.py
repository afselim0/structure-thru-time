#!/usr/bin/env python3
"""Count pairwise coalescent events across tree sequences and estimate coalescence-time PDFs.

This module provides the `CountCoalEvents` class, which:
  1. Discovers and loads tree-sequence files (*.trees) from a directory.
  2. Counts the total genomic span that coalesces in discrete time bins
     for every pair of samples for each tree sequence file, and saves the counts to disk.
  3. Normalises the counts into an empirical PDF of pairwise coalescence
     times, dividing each pair by the span it is actually observed on.
count files are saved in .npy format, and the final PDF is saved in .parquet format.
The parquet has multi-index rows for sample IDs and columns for time-bin left edges.

Coverage -- the per-pair denominator -- is MEASURED, by summing each pair's
coalescence counts over a time grid carrying an overflow bin to infinity. It is
not the contig's ``sequence_length``, which is the coordinate space rather than
the data, and not the summed width of local-ancestry windows, which counts
labelled genome rather than observed genome. The three agree only when every
base is accessible to every sample; they diverge on inferred ARGs, where
inaccessible regions and per-individual sample masks leave pairs unobserved over
part of their span. Each ``{contig}.npy`` of counts therefore has a
``{contig}.coverage.npy`` written beside it.
"""

from __future__ import annotations

import os
import time
from itertools import combinations
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tskit
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Counts-file integrity
# ---------------------------------------------------------------------------

def is_valid_counts_file(path: str, expected_shape: tuple[int, int]) -> bool:
    """Check whether a saved counts ``.npy`` file loads completely.

    Counts are written with ``ndarray.dump`` (a raw pickle), which streams the
    header before the data buffer, so a run killed mid-write leaves a file that
    only fails when it is eventually loaded — deep inside a later summation.
    This validates a file the only way that is conclusive for truncation: by
    actually loading it. Every byte is read, so any file that fails to unpickle,
    or that unpickles to the wrong shape, is reported as invalid.

    The loaded array is discarded immediately, so peak memory is one array
    rather than the whole directory. Because ``dump`` stores the data buffer as
    a latin-1-encoded string, corruption inside the buffer generally breaks the
    decoding too, so this catches mid-file damage and not just truncation. It
    still cannot detect corruption that happens to leave a decodable buffer, so
    it certifies that a file loads, not that its counts are numerically right.

    Parameters
    ----------
    path : str
        Path to the ``.npy`` counts file.
    expected_shape : tuple[int, ...]
        The shape the array must have: ``(num_pairs, num_timepoints)`` for a
        counts file, ``(num_pairs,)`` for the coverage file written beside it. A
        file that loads but was written under a different sample set or time grid
        is treated as invalid so that it is recomputed rather than silently summed
        into a mismatched total.

    Returns
    -------
    bool
        True if the file loads and holds an array of *expected_shape*; False if
        it is missing, empty, truncated, unreadable, or of the wrong shape.
    """
    # Any failure to load (truncated pickle, missing file, unreadable path,
    # failed cloud-storage fetch) means the file must be recomputed. The catch
    # is deliberately broad: the point is to never let a bad file reach the
    # summation, and the cost of a false negative is only recomputation.
    try:
        arr = np.load(path, allow_pickle=True)
    except Exception:
        return False

    if not isinstance(arr, np.ndarray):
        return False
    return arr.shape == tuple(expected_shape)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CountCoalEvents:
    """Count pairwise coalescent events from a collection of tree sequences.

    Parameters
    ----------
    ts_dir : str
        Directory containing ``.trees`` files.
    sample_ids : list, optional
        Subset of sample node IDs to analyse. If *None*, every sample in the
        first tree sequence is used.
    pdf_min_time : float, optional
        Lower bound of the time grid (in generations). Only 0 is supported; see
        ``__init__``. Defaults to 0.
    pdf_max_time : float, optional
        Upper bound of the time grid (in generations), and the label of the LAST
        pdf column. If *None*, defaults to the 95th percentile of node times in
        the first tree sequence.
    pdf_bin_size : float, optional
        Width of each time bin. Must divide ``pdf_max_time - pdf_min_time``
        exactly. Defaults to 1.0.
    fraction_samples : float, optional
        Probability that a given sample is included for a given contig.
        Setting this below 1.0 introduces stochastic sub-sampling to reduce
        computation. Defaults to 1.0 (use all samples on every contig).

    Attributes
    ----------
    ts_files : list[str]
        Sorted list of ``.trees`` filenames found in *ts_dir*.
    contig_ids : list[str] | np.ndarray
        Unique contig identifiers extracted from *ts_files*.
    contig_lengths : np.ndarray
        ``sequence_length`` of each contig. This is the size of the coordinate
        space, which for a tree sequence cut out of a chromosome with
        ``keep_intervals`` is the whole chromosome rather than the piece held.
        It is used only to choose a compact counts dtype -- never as coverage,
        which is measured per pair; see ``_build_coverage_df``.
    sample_pairs : list[tuple[int, int]]
        All unique ordered pairs of sample IDs.
    samples_selected : pd.DataFrame
        Boolean mask indicating which samples are active on each contig.
    coverage_df : pd.DataFrame | None
        Per-pair, per-contig observed span. ``None`` until
        ``count_coal_spans_all_ts`` has run, because it is measured from the
        counts rather than assumed from the inputs.
    total_segments_per_pair : np.ndarray | None
        Row sums of ``coverage_df``; the genome-wide denominator of the PDF.
        ``None`` until counting has run.
    """

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        ts_dir: str,
        sample_ids: Optional[List[int]] = None,
        pdf_min_time: float = 0.0,
        pdf_max_time: Optional[float] = None,
        pdf_bin_size: float = 1.0,
        fraction_samples: float = 1.0,
    ) -> None:
        self.ts_dir = ts_dir
        self.fraction_samples = fraction_samples

        self.pdf_min_time = float(pdf_min_time)
        self.pdf_bin_size = float(pdf_bin_size)
        if self.pdf_min_time != 0.0:
            raise NotImplementedError(
                f"pdf_min_time={self.pdf_min_time} is not supported yet (only 0). "
                "The survival S = 1 - cumsum(pdf) would be missing the coalescent "
                f"mass in [0, {self.pdf_min_time}), so the survival -- and every "
                "rate derived from it -- would be wrong. The parameter exists and "
                "is carried in filenames so the grid is self-describing, but a "
                "nonzero value must be implemented before it can be used."
            )
        if self.pdf_bin_size <= 0:
            raise ValueError(f"pdf_bin_size must be positive (got {self.pdf_bin_size}).")

        print("Preprocessing tree-sequence files …")

        # --- Input validation -------------------------------------------
        if not os.path.isdir(ts_dir):
            raise FileNotFoundError(f"ts_dir is not a valid directory: {ts_dir}")

        # --- Discover tree-sequence files -------------------------------
        self.ts_files = sorted(f.name for f in Path(ts_dir).glob("*.trees"))
        if not self.ts_files:
            raise ValueError(f"No .trees files found in directory: {ts_dir}")

        # --- Resolve contigs ----------------------------------------------
        # Each .trees file is treated as a single contig; read contig lengths
        # and extract contig IDs from filenames.
        self.num_contigs = len(self.ts_files)
        self.contig_ids = [f.removesuffix(".trees") for f in self.ts_files]
        self.contig_lengths = np.array([
            tskit.load(os.path.join(self.ts_dir, cid + ".trees")).sequence_length
            for cid in tqdm(self.contig_ids, total=self.num_contigs,
                            desc="Reading contig lengths")
        ], dtype=np.uint64)

        # --- Choose compact dtype based on total genome length ----------
        total_bp = int(np.sum(self.contig_lengths))
        self.datatype = np.uint32 if total_bp < 2**32 else np.uint64

        # --- Sample metadata -------------------------------------------
        ts_first = tskit.load(os.path.join(ts_dir, self.ts_files[0]))

        if sample_ids is None:
            self.sample_ids = ts_first.samples()
        else:
            self.sample_ids = np.sort(np.asarray(sample_ids))
        self.num_samples = len(self.sample_ids)

        # --- Time grid --------------------------------------------------
        # Columns are the bin LEFT edges, pdf_min_time ... pdf_max_time INCLUSIVE:
        # column t holds the coalescent mass in [t, t + pdf_bin_size). The grid
        # therefore carries one more bin than the half-open [min, max) convention,
        # and its last column is labelled pdf_max_time rather than max - bin.
        if pdf_max_time is None:
            pdf_max_time = float(np.quantile(ts_first.nodes_time, 0.95))
        span = float(pdf_max_time) - self.pdf_min_time
        if span <= 0:
            raise ValueError(
                f"pdf_max_time ({pdf_max_time}) must exceed pdf_min_time ({self.pdf_min_time})."
            )
        n_bins = round(span / self.pdf_bin_size)
        if abs(n_bins * self.pdf_bin_size - span) > 1e-9 * max(1.0, span):
            raise ValueError(
                f"(pdf_max_time - pdf_min_time) = {span} is not an exact multiple of "
                f"pdf_bin_size = {self.pdf_bin_size} (would need {span / self.pdf_bin_size} "
                "bins). Choose a bin size that divides the span exactly."
            )
        self.pdf_max_time = self.pdf_min_time + n_bins * self.pdf_bin_size
        # num_timepoints keeps its name and meaning -- the number of pdf columns --
        # so every shape check against it stays correct; only its value gains the
        # one extra bin that makes pdf_max_time itself a column.
        self.num_timepoints = n_bins + 1

        # --- All unique sample pairs ------------------------------------
        self.sample_pairs = [
            (int(self.sample_ids[i]), int(self.sample_ids[j]))
            for i in range(self.num_samples)
            for j in range(i + 1, self.num_samples)
        ]
        self.num_pairs = len(self.sample_pairs)
        self.pairs_MultiIndex = pd.MultiIndex.from_tuples(
            self.sample_pairs, names=["sample_1", "sample_2"]
        )

        # Lookup from (sample_1, sample_2) → flat pair index
        self.pair_to_idx = {pair: k for k, pair in enumerate(self.sample_pairs)}

        # --- Stochastic sub-sampling mask (samples × contigs) -----------
        self.samples_selected = pd.DataFrame(
            np.random.rand(self.num_samples, self.num_contigs) < self.fraction_samples,
            index=self.sample_ids,
            columns=self.contig_ids
        )
        # Coverage is MEASURED from the counts, so it cannot exist until counting
        # has run. Both attributes stay None until count_coal_spans_all_ts (or
        # compute_coal_time_pdf, which will load it) populates them.
        self.coverage_df: Optional[pd.DataFrame] = None
        self.total_segments_per_pair: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    #  Time grid
    # ------------------------------------------------------------------ #
    #
    # Both grids are built as min + arange(k) * bin rather than with linspace, so
    # every label is an exact multiple of the bin size. The labels survive a
    # round-trip through str (parquet column names) and satisfy the uniform-grid
    # check in ComputeCoalescentRates._assert_uniform_grid; linspace's accumulated
    # endpoint error does not reliably do either.

    @property
    def time_bin_left_edges(self) -> np.ndarray:
        """Pdf column labels: bin left edges, pdf_min_time … pdf_max_time inclusive."""
        return self.pdf_min_time + np.arange(self.num_timepoints) * self.pdf_bin_size

    @property
    def counting_time_windows(self) -> np.ndarray:
        """Bin edges for ``ts.pair_coalescence_counts``, with the overflow bin.

        ``num_timepoints + 1`` finite edges (pdf_min_time … pdf_max_time +
        pdf_bin_size) followed by inf. The final ``[pdf_max_time + pdf_bin_size,
        inf)`` bin is used only for per-pair coverage and is dropped from the pdf;
        see ``count_coal_spans_single_ts``.
        """
        finite = self.pdf_min_time + np.arange(self.num_timepoints + 1) * self.pdf_bin_size
        return np.append(finite, np.inf)

    # ------------------------------------------------------------------ #
    #  Counting coalescent spans
    # ------------------------------------------------------------------ #

    def count_coal_spans_single_ts(
        self,
        ts_file: str,
        coal_counts_dir: str,
    ) -> None:
        """Count total genomic spans where a coalescent event occurs between each pair of samples in time bins for one tree sequence.

        For every selected sample pair, the function records how many base-pairs
        coalesce within each time bin (defined by a regular grid of width
        ``self.pdf_bin_size`` from ``self.pdf_min_time`` to ``self.pdf_max_time``
        inclusive), and saves the result to disk. Nothing is returned;
        results are saved as a .npy file at ``{coal_counts_dir}/{basename}.npy``
        (array of shape ``(num_pairs, num_timepoints)``, dtype ``self.datatype``,
        rows in the same order as ``self.sample_pairs``), where ``basename`` is
        *ts_file* with the ``.trees`` suffix removed.

        Parameters
        ----------
        ts_file : str
            Filename (not full path) of the ``.trees`` file inside
            ``self.ts_dir``.
        coal_counts_dir : str
            The directory where the unnormalised counts for this file will be
            saved as a .npy file.
        """
        ts = tskit.load(os.path.join(self.ts_dir, ts_file))
        contig_id = ts_file.removesuffix(".trees")

        # --- Determine which samples are active on this contig ----------
        active_mask = self.samples_selected[contig_id]
        active_samples = active_mask[active_mask].index.values

        # Place results into the full (all-pairs) output array
        coal_segments_total = np.zeros(
            (self.num_pairs, self.num_timepoints), dtype=self.datatype
        )
        # Per-pair observed span on this contig. Pairs not active here keep 0, so
        # they contribute neither counts nor coverage.
        coverage = np.zeros(self.num_pairs, dtype=np.float64)

        if len(active_samples) > 0:
            # Simplify the tree sequence down to the active sample set
            if len(active_samples) < ts.num_samples:
                ts = ts.simplify(samples=active_samples)

            # --- Build local pair indices that map back to the global index --
            local_pairs: list[tuple[int, int]] = []
            global_pair_idxs: list[int] = []
            for i, s1 in enumerate(active_samples):
                for j in range(i + 1, len(active_samples)):
                    s2 = active_samples[j]
                    local_pairs.append((i, j))
                    global_pair_idxs.append(self.pair_to_idx[(s1, s2)])
            global_pair_idx = np.array(global_pair_idxs)

            # --- Count pair-coalescence events in each time bin -------------
            # An extra window [pdf_max_time + pdf_bin_size, inf) is appended so that
            # every pair's coalescences are counted, including those older than the
            # time grid. That final bin is NOT part of the PDF -- it is dropped below
            # -- but its mass is needed for coverage: summing the finite bins alone
            # would undercount each pair's observed span by however much of it
            # coalesces beyond the grid, which is both large (3-4% at the default
            # 95th percentile) and pair-dependent, and so would bias the normalised
            # PDF.
            time_windows = self.counting_time_windows
            coal_segments = ts.pair_coalescence_counts(
                sample_sets=[[i] for i in range(len(active_samples))],
                indexes=local_pairs,
                span_normalise=False,
                time_windows=time_windows,
            )

            # Coverage is the span over which each pair actually coalesces anywhere
            # in time, i.e. the span on which that pair is observed at all. Taken
            # across ALL bins including the infinite one, before it is discarded.
            coverage[global_pair_idx] = coal_segments.sum(axis=1)

            # Drop the overflow bin: the PDF covers
            # [pdf_min_time, pdf_max_time + pdf_bin_size) only.
            coal_segments_total[global_pair_idx, :] = (
                coal_segments[:, :-1].astype(self.datatype)
            )

        # Write to a temporary file and rename into place, so a run killed
        # mid-write leaves no partially-written .npy that a later resume would
        # mistake for finished output (os.replace is atomic within a filesystem).
        out_path = os.path.join(coal_counts_dir, f"{contig_id}.npy")
        tmp_path = f"{out_path}.tmp.{os.getpid()}"
        try:
            coal_segments_total.dump(tmp_path)
            os.replace(tmp_path, out_path)
        except BaseException:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

        # Coverage is written second and as its own file, so that a counts file is
        # never left without one: the resume check below requires both, and a
        # missing coverage file simply forces this contig to be recounted.
        cov_path = os.path.join(coal_counts_dir, f"{contig_id}.coverage.npy")
        cov_tmp = f"{cov_path}.tmp.{os.getpid()}"
        try:
            coverage.dump(cov_tmp)
            os.replace(cov_tmp, cov_path)
        except BaseException:
            if os.path.exists(cov_tmp):
                os.remove(cov_tmp)
            raise

    def count_coal_spans_all_ts(
        self,
        coal_counts_dir: str,
        num_threads: int = 1,
        resume: bool = True,
    ) -> None:
        """Count total genomic spans where a coalescent event occurs between each pair of samples in time bins for all tree sequences in a directory.

        Each tree sequence's counts are saved to disk by
        ``count_coal_spans_single_ts`` as a ``.npy`` file; results are not
        returned or accumulated in memory.

        Parameters
        ----------
        ts_files : list[str]
            Filenames of the tree sequences to process.
        coal_counts_dir : str
            Forwarded to ``count_coal_spans_single_ts``; directory where
            per-file numpy arrays are saved.
        num_threads : int, optional
            Number of parallel workers. Defaults to 1 (serial execution).
        resume : bool, optional
            If True (default), every existing ``.npy`` file in *coal_counts_dir*
            is loaded; those that load successfully with the expected shape are
            skipped, and any that is missing, truncated or otherwise unloadable
            is recomputed. This lets a killed/timed-out run (e.g. OOM or
            wall-clock cancellation) pick up where it left off without trusting
            files the interruption may have left half-written. Validation runs
            serially and reads every counts file in full, so it is I/O-heavy but
            far cheaper than recounting; pass False to skip it and recompute
            everything.
        """
        os.makedirs(coal_counts_dir, exist_ok=True)
        ts_files = self.ts_files
        if resume:
            # A contig counts as done only if its file actually loads: a run
            # killed mid-write leaves a truncated pickle that must be redone.
            # Validation is serial so that only one counts array is held in
            # memory at a time; parallel workers would each hold their own.
            expected_shape = (self.num_pairs, self.num_timepoints)
            existing = [
                ts_file for ts_file in ts_files
                if os.path.exists(
                    os.path.join(coal_counts_dir, f"{ts_file.removesuffix('.trees')}.npy")
                )
            ]

            skipped, corrupt = [], []
            for ts_file in tqdm(existing, desc="Validating existing counts files"):
                basename = ts_file.removesuffix(".trees")
                path = os.path.join(coal_counts_dir, f"{basename}.npy")
                cov_path = os.path.join(coal_counts_dir, f"{basename}.coverage.npy")
                # BOTH files must be valid. Counts written before coverage existed
                # are treated as corrupt so they are recomputed rather than silently
                # normalised against a coverage figure that was never measured.
                if (
                    is_valid_counts_file(path, expected_shape)
                    and is_valid_counts_file(cov_path, (self.num_pairs,))
                ):
                    skipped.append(ts_file)
                else:
                    corrupt.append(ts_file)

            if corrupt:
                print(
                    f"Found {len(corrupt)} incomplete/corrupt counts file(s) that will be "
                    f"recomputed: {sorted(f.removesuffix('.trees') for f in corrupt)}"
                )
            if skipped:
                print(f"Resuming: skipping {len(skipped)} tree sequence(s) with complete counts.")

            skipped_set = set(skipped)
            ts_files = [ts_file for ts_file in ts_files if ts_file not in skipped_set]
        num_threads = min(num_threads, len(ts_files)) if ts_files else 1

        if num_threads == 1:
            # --- Serial execution -----------------------------------------
            for ts_file in tqdm(ts_files, desc="Counting coalescent segments"):
                self.count_coal_spans_single_ts(ts_file, coal_counts_dir=coal_counts_dir)
        else:
            # --- Parallel execution via pathos, imap for a live progress bar
            pool = Pool(nodes=num_threads)
            imap_or_raise(
                pool,
                lambda ts_file: self.count_coal_spans_single_ts(ts_file, coal_counts_dir=coal_counts_dir),
                ts_files,
                "Counting coalescent segments",
            )
            # pathos caches ProcessPool instances by nodes count; close()+join()
            # alone leaves the closed pool in that cache, so a later Pool(nodes=N)
            # call with the same N returns the stale closed pool. clear() evicts
            # it, forcing a fresh pool on the next construction.
            pool.clear()

        # Persist the per-pair, per-contig coverage alongside the counts so that
        # downstream consumers (e.g. CountCoalEvents_admix, variance estimation)
        # can normalise without recounting. Assembled from the per-contig coverage
        # files just written, which is why this follows the counting loop rather
        # than being built in __init__.
        self.coverage_df = self._build_coverage_df(coal_counts_dir)
        self.total_segments_per_pair = self.coverage_df.values.sum(axis=1)
        self.coverage_df.to_parquet(os.path.join(coal_counts_dir, "coverage.parquet"))

    # ------------------------------------------------------------------ #
    #  PDF estimation
    # ------------------------------------------------------------------ #

    def compute_coal_time_pdf(
        self,
        coal_counts_dir: str
    ) -> pd.DataFrame:
        """Estimate the PDF of pairwise coalescence times given the unnormalised counts.

        Loads the per-file coalescent-span counts previously saved to
        *coal_counts_dir* by ``count_coal_spans_all_ts``, sums them
        across tree sequences, and normalises by the total observed genomic
        span for each pair.

        Parameters
        ----------
        coal_counts_dir : str
            Directory containing the unnormalised ``.npy`` counts for each
            tree sequence, as saved by ``count_coal_spans_single_ts``.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by ``(sample_1, sample_2)`` with one column per
            time-bin left-edge.  Values approximate the coalescence-time PDF.

            Rows do not sum to 1: the columns span
            ``[pdf_min_time, pdf_max_time + pdf_bin_size)`` while the
            normalising coverage counts coalescences at every time, so a row sums
            to the fraction of that pair's span which coalesces inside the grid.
            A pair with zero coverage genome-wide -- observed on no contig -- is
            all-NaN, since its PDF is undefined rather than zero.
        """
        # --- Verify counts exist for every tree-sequence file ------------
        expected_basenames = {ts_file.removesuffix(".trees") for ts_file in self.ts_files}
        # "{contig}.coverage.npy" also matches *.npy; excluded so it is not taken
        # for a contig of its own.
        count_files = {
            f.name.removesuffix(".npy") for f in Path(coal_counts_dir).glob("*.npy")
            if not f.name.endswith(".coverage.npy")
        }
        missing = expected_basenames - count_files
        if missing:
            raise FileNotFoundError(
                f"coal_counts_dir '{coal_counts_dir}' is missing coalescent-count "
                f"files for {len(missing)} tree sequence(s): {sorted(missing)}"
            )

        # --- Load and sum counts across all tree sequences ----------------
        # Each file is loaded exactly once: a load failure or shape mismatch here
        # is itself the integrity check, reported against the offending contig
        # rather than as a bare UnpicklingError from deep inside the loop.
        expected_shape = (self.num_pairs, self.num_timepoints)
        coal_segments_total = np.zeros((self.num_pairs, self.num_timepoints), dtype=self.datatype)
        for basename in tqdm(sorted(expected_basenames), desc="Summing counts across tree sequences"):
            path = os.path.join(coal_counts_dir, f"{basename}.npy")
            try:
                counts = np.load(path, allow_pickle=True)
            except Exception as err:
                raise ValueError(
                    f"coalescent-count file '{path}' is incomplete or malformed and could "
                    "not be loaded. Re-run count_coal_spans_all_ts (resume=True recomputes "
                    "only the unloadable files)."
                ) from err
            if counts.shape != expected_shape:
                raise ValueError(
                    f"coalescent-count file '{path}' has shape {counts.shape}, expected "
                    f"{expected_shape}. It was written for a different sample set or time "
                    "grid; delete it and re-run count_coal_spans_all_ts."
                )
            coal_segments_total += counts

        # --- Normalise counts into a PDF --------------------------------
        pdf_df = pd.DataFrame(
            coal_segments_total,
            columns=self.time_bin_left_edges,
            index=self.pairs_MultiIndex,
        )

        # Coverage may not be in memory: this can be called on a fresh instance
        # pointed at a finished counts directory. Rebuild it from the per-contig
        # coverage files, which is exactly what count_coal_spans_all_ts did.
        if self.total_segments_per_pair is None:
            self.coverage_df = self._build_coverage_df(coal_counts_dir)
            self.total_segments_per_pair = self.coverage_df.values.sum(axis=1)

        # Normalise by the span each pair is actually observed on -- which already
        # accounts for the stochastic sub-sampling mask, since a pair that was not
        # selected on a contig contributes zero coverage there as well as zero
        # counts.
        #
        # Zero coverage on an INDIVIDUAL contig is ordinary and needs no special
        # handling: that contig adds 0 to both the counts and the coverage, so it
        # simply does not contribute to the genome-wide totals. A chunk where one
        # sample of the pair is entirely masked looks exactly like this.
        #
        # Zero coverage GENOME-WIDE is different: the pair is observed nowhere, so
        # its PDF is undefined rather than zero. NaN is the honest value --
        # emitting zeros would let an unobserved pair enter the likelihood as if
        # it had never coalesced, and emitting 0/0 would produce a silent inf.
        denom = self.total_segments_per_pair.astype(np.float64)
        unobserved = denom == 0
        if np.any(unobserved):
            print(
                f"Warning: {int(unobserved.sum())} of {self.num_pairs} pairs have "
                "zero coverage genome-wide (observed on no contig) and will be NaN "
                "in the PDF."
            )
        pdf_df = pdf_df.div(np.where(unobserved, np.nan, denom), axis=0)
        return pdf_df

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _build_coverage_df(self, coal_counts_dir: str) -> pd.DataFrame:
        """Build the per-pair, per-contig coverage dataframe from the counts run.

        Rows are ``(sample_1, sample_2)`` pairs; columns are the basenames of the
        ``.trees`` files. Each entry is the genomic span on that contig over which
        that pair actually coalesces -- i.e. the span on which the pair is
        observed -- as measured by ``count_coal_spans_single_ts`` and written to
        ``{contig_id}.coverage.npy``.

        This is measured rather than assumed. It previously used
        ``ts.sequence_length`` for every pair on a contig, which is wrong on two
        counts:

        * ``sequence_length`` is the length of the *coordinate space*, not of the
          data. Tree sequences cut out of a chromosome with ``keep_intervals``
          (as the human pipeline's ``ts_chunks`` are, to keep positions traceable
          to their true chromosomal location) report the whole chromosome, so a
          2 Mb chunk of chr1 claimed 249 Mb of coverage -- a 125x overstatement,
          and one that varies by chromosome, so it distorted the PDF unevenly.
        * it assigns every pair the same span, but a pair is only observed where
          both its samples have ancestry. Regions masked out per individual
          (imputed genotypes, detached in ``remove_partial_ancestry``) are absent
          for that individual's pairs and present for all others.

        Summing the pair-coalescence counts recovers both effects exactly and
        needs no mask files, because a pair coalesces exactly once at every
        position where it is observed. This is why the counting grid carries an
        overflow bin to infinity: the finite bins alone miss the 3-4% of span
        that coalesces beyond the grid, by a margin that differs per pair.

        Parameters
        ----------
        coal_counts_dir : str
            Directory holding ``{contig_id}.coverage.npy`` for every contig, as
            written by ``count_coal_spans_single_ts``.

        Returns
        -------
        pd.DataFrame
            Shape ``(num_pairs, num_ts_files)``.
        """
        col_names = [ts_file.removesuffix(".trees") for ts_file in self.ts_files]
        data = np.zeros((self.num_pairs, len(self.ts_files)), dtype=np.float64)
        for col_idx, contig_id in enumerate(col_names):
            cov_path = os.path.join(coal_counts_dir, f"{contig_id}.coverage.npy")
            try:
                cov = np.load(cov_path, allow_pickle=True)
            except Exception as err:
                raise FileNotFoundError(
                    f"coverage file '{cov_path}' is missing or unreadable. It is "
                    "written beside each counts file by count_coal_spans_all_ts; "
                    "re-run that (resume=True recomputes only what is missing)."
                ) from err
            if cov.shape != (self.num_pairs,):
                raise ValueError(
                    f"coverage file '{cov_path}' has shape {cov.shape}, expected "
                    f"{(self.num_pairs,)}. It was written for a different sample "
                    "set; delete it and re-run count_coal_spans_all_ts."
                )
            data[:, col_idx] = cov

        return pd.DataFrame(data, index=self.pairs_MultiIndex, columns=col_names)
def read_LAI_df(path):
    """Read a local-ancestry parquet, dropping intervals with no ancestry call.

    A null ``ancestry_label`` means the painter had no genealogical information
    over that span -- the empty flanking trees outside an inferred chunk, or a
    sample locally unattached to any reference panel. Such an interval belongs to
    no panel and must not contribute to any panel's counts.

    Dropping the rows here, rather than relying on the comparisons downstream to
    ignore them, is deliberate. ``parse_intersecting_window_edges`` matches
    labels with ``labels_1 == label``, and whether a null survives that depends
    on which null it is: ``NaN != NaN`` so a NaN interval can never match, but
    ``None == None`` is True, so a None-valued label would make every uncalled
    interval of one sample match every uncalled interval of another and be
    counted as shared ancestry. Today the parquet round-trip happens to yield
    NaN, which makes the current behaviour correct by accident of dtype rather
    than by construction; this makes it explicit.

    Args:
        path (str): path to a per-contig local ancestry parquet.

    Returns:
        pd.DataFrame: the called intervals only, with a guaranteed-null-free
        ``ancestry_label`` column.
    """
    LAI_df = pd.read_parquet(path)
    called = LAI_df[LAI_df["ancestry_label"].notna()]
    labels = called["ancestry_label"].unique()
    if any(pd.isna(label) for label in labels):
        raise ValueError(
            f"null ancestry labels survived filtering in {path}; they would be "
            "treated as an ancestry panel by the window-intersection logic."
        )
    return called


def imap_or_raise(pool, func, items, desc):
    """Run ``func`` over ``items`` in ``pool``, raising if a worker dies.

    ``pool.imap`` blocks in ``next()`` forever when a worker is killed rather
    than returning: ``multiprocess.Pool`` quietly replaces the dead process, but
    the task it was holding is never completed and never re-queued, so the parent
    waits on a result that cannot arrive. A job in that state keeps its
    allocation and looks RUNNING indefinitely -- one burned 9.5 hours before it
    was noticed, its workers idle and its progress bar on 0 while per-contig
    output files piled up on disk from the tasks that had finished.

    The usual cause is the OOM killer taking a worker: each one holds several GB
    here, so oversubscribing memory kills children without failing the job.

    Submitting with ``amap`` and polling lets the parent watch the worker roster
    instead of blocking on a result. A changed roster means a worker died, which
    is reported with the failing step named rather than left to hang.

    Args:
        pool: a pathos ``ProcessingPool``.
        func (callable): applied to each item, in a worker.
        items (list): work items.
        desc (str): progress bar label, also used in the error message.

    Returns:
        list: results, in the order of ``items``.

    Raises:
        RuntimeError: if a worker process dies before the results are ready.
    """
    result = pool.amap(func, items)
    initial_pids = {worker.pid for worker in pool._serve()._pool}

    with tqdm(total=len(items), desc=desc) as progress:
        while not result.ready():
            # _number_left counts task CHUNKS still outstanding, not items, so it
            # is only a progress signal; the true count comes from the result.
            done = len(items) - result._number_left * result._chunksize
            progress.n = max(0, min(done, len(items)))
            progress.refresh()

            current_pids = {worker.pid for worker in pool._serve()._pool}
            if current_pids != initial_pids:
                lost = sorted(initial_pids - current_pids)
                raise RuntimeError(
                    f"{desc}: a worker process died (pid {lost}) and its task was "
                    "lost, so this step would wait forever. The usual cause is the "
                    "OOM killer: each worker holds the full per-contig count "
                    "arrays, so peak memory scales with num_threads. Re-run with "
                    "fewer threads, more memory, or a coarser pdf_bin_size."
                )
            time.sleep(0.5)
        progress.n = len(items)
        progress.refresh()

    return result.get()


def pad_windows_to_sequence(window_edges, window_idxs, sequence_length):
    """Extend ancestry windows to span [0, sequence_length] for tskit.

    ``pair_coalescence_counts`` requires its ``windows`` to tile the whole
    sequence: strictly increasing, starting at 0 and ending at
    ``sequence_length``. Ancestry windows do not, for two reasons -- a contig cut
    out of a chromosome keeps the chromosome's coordinate space, so its trees
    occupy only the middle of it, and spans with no ancestry call are absent
    entirely.

    Padding is sound because the padded spans hold no trees and therefore no
    coalescences, so they contribute exactly zero to both the counts and the
    coverage they are summed into (verified in .misc/GNN_test/). The returned
    indices are shifted to match, so the padding is never selected for an
    ancestry label and never enters either total.

    Args:
        window_edges (np.ndarray): increasing window boundaries from
            ``parse_window_edges``.
        window_idxs (np.ndarray): indices into ``window_edges[:-1]`` naming the
            windows that carry the ancestry label of interest.
        sequence_length (float): ``ts.sequence_length``.

    Returns:
        (padded_edges, shifted_idxs): edges spanning ``[0, sequence_length]``,
        and ``window_idxs`` renumbered into them.
    """
    edges = np.asarray(window_edges, dtype=np.float64)
    if edges[0] < 0 or edges[-1] > sequence_length:
        raise ValueError(
            f"ancestry windows [{edges[0]}, {edges[-1]}] fall outside the tree "
            f"sequence [0, {sequence_length}]."
        )

    pad_left = edges[0] > 0
    pad_right = edges[-1] < sequence_length
    if not (pad_left or pad_right):
        return edges, np.asarray(window_idxs, dtype=int)

    padded = np.concatenate((
        [0.0] if pad_left else [],
        edges,
        [float(sequence_length)] if pad_right else [],
    ))
    return padded, np.asarray(window_idxs, dtype=int) + int(pad_left)


def parse_window_edges(LAI_df):
    """Parses the dataframe with local ancestry information to find bp positions where ancestry switches

    Args:
        LAI_df (pd.DataFrame): dataframe with local ancestry information, with columns ["sample_id", "start_pos", "end_pos", "ancestry_label"]

    Returns:
        window_edges (dict): dictionary mapping each admixed sample to positions where ancestry switches (always starts at zero and ends at the end of the contig. Intermediate positions is where ancestry switches)
        window_idxs_dict (dict): dictionary mapping each admixed sample to the indices of its window edges for each ancestry label
    """
    ancestry_labels = LAI_df["ancestry_label"].unique()
    window_edges = {}
    window_idxs_dict = {}
    # Single grouped pass instead of re-masking the full dataframe per sample/label
    label_groups = dict(LAI_df.groupby(["sample_id", "ancestry_label"])["start_pos"].apply(np.asarray))
    for admixed_sample, LAI_sample in LAI_df.groupby("sample_id"):
        start_positions = LAI_sample["start_pos"].values
        end_positions = LAI_sample["end_pos"].values
        unique_edges = np.unique(np.concatenate([start_positions, end_positions]))
        window_edges[admixed_sample] = unique_edges
        window_idxs_dict_sample = {}
        for ancestry_label in ancestry_labels:
            label_start_positions = label_groups.get((admixed_sample, ancestry_label), np.array([]))
            window_idxs_dict_sample[ancestry_label] = np.searchsorted(unique_edges, label_start_positions, side='left')
        window_idxs_dict[admixed_sample] = window_idxs_dict_sample
    return window_edges, window_idxs_dict

def parse_intersecting_window_edges(LAI_df, window_edges, window_idxs_dict):
    """Parses the dataframe with local ancestry information to find genomic spans where each pair of admixed samples have the same ancestry label.

    Args:
        LAI_df (pd.DataFrame): dataframe with local ancestry information, with columns ["sample_id", "start_pos", "end_pos", "ancestry_label"]
        window_edges (dict): per-sample window edges, as returned by parse_window_edges(LAI_df)
        window_idxs_dict (dict): per-sample, per-ancestry-label window edge indices, as returned by parse_window_edges(LAI_df)

    Returns:
        intersecting_window_edges (dict): dictionary mapping each pair of admixed samples to positions where ancestry switches for either sample (always starts at zero and ends at the end of the contig. Intermediate positions is where ancestry switches for either sample)
        intersecting_window_window_idxs_dict (dict): dictionary mapping each pair of admixed samples to the indices of their intersecting window edges. For example: intersecting_window_window_idxs_dict[(sample1, sample2)]["A"] gives the indices of the intersecting window edges where both pairs have label "A"
    """
    ancestry_labels = LAI_df["ancestry_label"].unique()
    admixed_samples = LAI_df["sample_id"].unique()

    # Per-sample lookup: for each edge index, which ancestry label is active on the interval starting there
    per_sample_interval_labels = {}
    for sample in admixed_samples:
        n_intervals = len(window_edges[sample]) - 1
        interval_labels = np.empty(n_intervals, dtype=object)
        for ancestry_label in ancestry_labels:
            interval_labels[window_idxs_dict[sample][ancestry_label]] = ancestry_label
        per_sample_interval_labels[sample] = interval_labels

    intersecting_window_edges = {}
    intersecting_window_idxs_dict = {}
    for sample_1, sample_2 in combinations(admixed_samples, 2):
        # Merge both samples' breakpoints so every interval has a single label per sample
        merged_edges = np.union1d(window_edges[sample_1], window_edges[sample_2])
        intersecting_window_edges[(sample_1, sample_2)] = merged_edges

        # For each merged interval, look up which per-sample interval it falls in
        interval_starts = merged_edges[:-1]
        idx_1 = np.searchsorted(window_edges[sample_1], interval_starts, side='right') - 1
        idx_2 = np.searchsorted(window_edges[sample_2], interval_starts, side='right') - 1
        labels_1 = per_sample_interval_labels[sample_1][idx_1]
        labels_2 = per_sample_interval_labels[sample_2][idx_2]

        pair_idxs_dict = {ancestry_label: np.flatnonzero((labels_1 == ancestry_label) & (labels_2 == ancestry_label))
                           for ancestry_label in ancestry_labels}
        intersecting_window_idxs_dict[(sample_1, sample_2)] = pair_idxs_dict

        # Drop pairs that never share an ancestry label on any interval
        if all(idxs.size == 0 for idxs in pair_idxs_dict.values()):
            del intersecting_window_edges[(sample_1, sample_2)]
            del intersecting_window_idxs_dict[(sample_1, sample_2)]

    return intersecting_window_edges, intersecting_window_idxs_dict


# ---------------------------------------------------------------------------
# Admixture-aware counting
# ---------------------------------------------------------------------------

class CountCoalEvents_admix:
    """Count pairwise coalescent events per local ancestry for admixed samples.

    Builds on a previously-run ``CountCoalEvents`` instance and its saved
    counts/coverage. For every ancestry label, only pairs where both samples
    carry that ancestry somewhere in their genome are considered:

      * non-admixed × non-admixed pairs are copied unchanged from the base
        ``CountCoalEvents`` output into every ancestry (ancestry is undefined
        for these pairs, so they contribute identically everywhere);
      * admixed × non-admixed pairs are counted only over genomic windows
        where the admixed sample carries that ancestry label;
      * admixed × admixed pairs are counted only over windows where *both*
        samples carry that ancestry label.

    Coverage for every pair is the span over which that pair actually
    coalesces within its windows, summed from the pair-coalescence counts
    (including an overflow time bin, so coalescences older than the time grid
    are not missed). It is deliberately *not* the summed width of the local
    ancestry windows: LAI windows tile the whole contig, but a pair is only
    observed where both samples have ancestry there, so the two agree only
    when every base is accessible -- true of simulated data, false of
    inferred ARGs, where inaccessible regions and per-individual sample masks
    leave the pair unobserved over part of the window.

    Each ancestry label additionally has its own ``pop_labels.csv`` that can
    exclude admixed samples not confidently assigned to it (non-admixed
    samples are always retained). Excluded admixed samples drop out of that
    ancestry's pairs entirely, so each ancestry has its own, generally
    smaller, sample-pair index.

    Parameters
    ----------
    cce : CountCoalEvents
        A fully-initialised instance whose ``count_coal_spans_all_ts`` has
        already been run against *coal_counts_dir*.
    coal_counts_dir : str
        Directory containing the per-contig ``.npy`` counts and the
        ``coverage.parquet`` file produced by ``cce``.
    LAI_dir : str
        Directory of per-contig local-ancestry parquet files, one per
        ``.trees`` file in ``cce.ts_dir``, with columns
        ``["sample_id", "start_pos", "end_pos", "ancestry_label"]``.
    ancestry_count_dirs : dict[str, str]
        Ancestry label → output directory for that ancestry's per-contig
        counts and coverage. Created if missing.
    pop_labels_paths : dict[str, str]
        Ancestry label → path to a ``pop_labels.csv`` with a ``sample_id``
        column giving the samples to include for that ancestry. Must contain
        every non-admixed sample; admixed samples not listed are excluded
        from that ancestry only.
    """

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        cce: "CountCoalEvents",
        coal_counts_dir: str,
        LAI_dir: str,
        ancestry_count_dirs: dict[str, str],
        pop_labels_paths: dict[str, str],
    ) -> None:
        self.cce = cce
        self.ts_dir = cce.ts_dir
        self.ts_files = cce.ts_files
        self.contig_ids = cce.contig_ids
        self.datatype = cce.datatype
        self.pdf_min_time = cce.pdf_min_time
        self.pdf_max_time = cce.pdf_max_time
        self.pdf_bin_size = cce.pdf_bin_size
        self.num_timepoints = cce.num_timepoints
        # Copied as plain attributes rather than recomputed: this class is a pure
        # consumer of the parent's grid, and sharing the arrays makes a divergence
        # between the whole-genome and per-ancestry pdfs impossible by construction.
        self.time_bin_left_edges = cce.time_bin_left_edges
        self.counting_time_windows = cce.counting_time_windows
        self.ancestry_count_dirs = ancestry_count_dirs

        # --- Input validation --------------------------------------------
        if not os.path.isdir(LAI_dir):
            raise FileNotFoundError(f"LAI_dir is not a valid directory: {LAI_dir}")
        for label, out_dir in ancestry_count_dirs.items():
            os.makedirs(out_dir, exist_ok=True)

        # --- Load base counts/coverage from the CountCoalEvents run -------
        expected_basenames = {ts_file.removesuffix(".trees") for ts_file in cce.ts_files}
        # Exclude the per-contig coverage vectors written beside the counts:
        # "{contig}.coverage.npy" also matches *.npy, and would otherwise be
        # treated as a contig named "{contig}.coverage".
        count_files = {
            f.name.removesuffix(".npy") for f in Path(coal_counts_dir).glob("*.npy")
            if not f.name.endswith(".coverage.npy")
        }
        missing = expected_basenames - count_files
        if missing:
            raise FileNotFoundError(
                f"coal_counts_dir '{coal_counts_dir}' is missing coalescent-count "
                f"files for {len(missing)} tree sequence(s): {sorted(missing)}"
            )
        coverage_path = os.path.join(coal_counts_dir, "coverage.parquet")
        if not os.path.isfile(coverage_path):
            raise FileNotFoundError(
                f"coal_counts_dir '{coal_counts_dir}' is missing 'coverage.parquet'. "
                "Re-run CountCoalEvents.count_coal_spans_all_ts to produce it."
            )
        self.coal_counts_dir = coal_counts_dir
        self.cce_coverage_df = pd.read_parquet(coverage_path)

        # --- Admixed / non-admixed samples, read from one LAI file --------
        # Ancestry labels and the admixed-sample set are assumed fixed across
        # contigs; a contig whose LAI file disagrees raises loudly rather than
        # silently reshaping the pair sets (see count_coal_spans_single_ts).
        first_LAI_df = read_LAI_df(os.path.join(LAI_dir, f"{cce.contig_ids[0]}.parquet"))
        self.admixed_samples = np.sort(first_LAI_df["sample_id"].unique())
        self.ancestry_labels = list(pop_labels_paths.keys())
        self.non_admixed_samples = np.setdiff1d(cce.sample_ids, self.admixed_samples)

        # --- Per-ancestry included samples, pairs, and coverage indices ---
        self.included_samples: dict[str, np.ndarray] = {}
        self.sample_pairs_per_ancestry: dict[str, list[tuple[int, int]]] = {}
        self.pair_to_idx_per_ancestry: dict[str, dict[tuple[int, int], int]] = {}
        self.pairs_MultiIndex_per_ancestry: dict[str, pd.MultiIndex] = {}
        for label, pop_labels_path in pop_labels_paths.items():
            included = np.sort(pd.read_csv(pop_labels_path)["sample_id"].unique())

            # Non-admixed samples must always be present; admixed samples may
            # be a subset. Anything outside cce.sample_ids is a bad input.
            if not np.isin(self.non_admixed_samples, included).all():
                missing_nonadmixed = np.setdiff1d(self.non_admixed_samples, included)
                raise ValueError(
                    f"pop_labels file for ancestry '{label}' ({pop_labels_path}) is "
                    f"missing non-admixed sample(s): {missing_nonadmixed.tolist()}"
                )
            if not np.isin(included, cce.sample_ids).all():
                unknown = np.setdiff1d(included, cce.sample_ids)
                raise ValueError(
                    f"pop_labels file for ancestry '{label}' ({pop_labels_path}) contains "
                    f"sample_id(s) not present in cce.sample_ids: {unknown.tolist()}"
                )

            self.included_samples[label] = included
            sample_pairs = [
                (int(included[i]), int(included[j]))
                for i in range(len(included))
                for j in range(i + 1, len(included))
            ]
            self.sample_pairs_per_ancestry[label] = sample_pairs
            self.pair_to_idx_per_ancestry[label] = {pair: k for k, pair in enumerate(sample_pairs)}
            self.pairs_MultiIndex_per_ancestry[label] = pd.MultiIndex.from_tuples(
                sample_pairs, names=["sample_1", "sample_2"]
            )

    # ------------------------------------------------------------------ #
    #  Counting coalescent spans
    # ------------------------------------------------------------------ #

    def count_coal_spans_single_ts(self, ts_file: str, LAI_dir: str) -> dict[str, np.ndarray]:
        """Count per-ancestry coalescent spans for one tree sequence.

        For every ancestry label, builds a ``(num_pairs_label, num_timepoints)``
        counts array (saved to ``{ancestry_count_dirs[label]}/{contig_id}.npy``)
        and a ``(num_pairs_label,)`` coverage vector (returned, to be assembled
        into that ancestry's ``coverage.parquet`` by the caller once all
        contigs have been processed).

        Parameters
        ----------
        ts_file : str
            Filename (not full path) of the ``.trees`` file inside
            ``self.ts_dir``.
        LAI_dir : str
            Directory of per-contig local-ancestry parquet files.

        Returns
        -------
        dict[str, np.ndarray]
            Ancestry label → this contig's per-pair coverage vector (bp),
            ordered like ``self.sample_pairs_per_ancestry[label]``.
        """
        contig_id = ts_file.removesuffix(".trees")
        ts = tskit.load(os.path.join(self.ts_dir, ts_file))
        LAI_df = read_LAI_df(os.path.join(LAI_dir, f"{contig_id}.parquet"))

        contig_admixed_samples = np.sort(LAI_df["sample_id"].unique())
        if not np.array_equal(contig_admixed_samples, self.admixed_samples):
            raise ValueError(
                f"LAI file for contig '{contig_id}' has admixed samples "
                f"{contig_admixed_samples.tolist()}, which does not match "
                f"{self.admixed_samples.tolist()} from '{self.contig_ids[0]}'."
            )

        window_edges, window_idxs_dict = parse_window_edges(LAI_df)
        intersecting_window_edges, intersecting_window_idxs_dict = parse_intersecting_window_edges(
            LAI_df, window_edges, window_idxs_dict
        )

        base_counts = np.load(os.path.join(self.coal_counts_dir, f"{contig_id}.npy"), allow_pickle=True)
        # As in CountCoalEvents.count_coal_spans_single_ts, an overflow bin
        # [pdf_max_time + pdf_bin_size, inf) is appended so that coverage counts
        # every coalescence, not just those inside the time grid. It is dropped
        # before the counts are stored; only its contribution to coverage is kept.
        time_windows = self.counting_time_windows

        # --- Initialise per-ancestry output arrays and coverage vectors ---
        counts_per_ancestry = {
            label: np.zeros((len(self.sample_pairs_per_ancestry[label]), self.num_timepoints), dtype=self.datatype)
            for label in self.ancestry_labels
        }
        coverage_per_ancestry = {
            label: np.zeros(len(self.sample_pairs_per_ancestry[label]), dtype=np.float64)
            for label in self.ancestry_labels
        }

        for label in self.ancestry_labels:
            included = self.included_samples[label]
            included_set = set(included.tolist())
            pair_to_idx = self.pair_to_idx_per_ancestry[label]
            admixed_in_label = [s for s in self.admixed_samples if s in included_set]
            non_admixed_in_label = [s for s in self.non_admixed_samples if s in included_set]

            # --- Non-admixed × non-admixed: copied unchanged -------------
            for i, s1 in enumerate(non_admixed_in_label):
                for s2 in non_admixed_in_label[i + 1:]:
                    global_idx = self.cce.pair_to_idx[(s1, s2)]
                    local_idx = pair_to_idx[(s1, s2)]
                    counts_per_ancestry[label][local_idx, :] = base_counts[global_idx, :]
                    coverage_per_ancestry[label][local_idx] = self.cce_coverage_df.at[(s1, s2), contig_id]

            # --- Admixed × non-admixed: windowed by the admixed sample's ancestry
            if admixed_in_label and non_admixed_in_label:
                for admixed_sample in admixed_in_label:
                    window_idxs = window_idxs_dict[admixed_sample].get(label, np.array([], dtype=int))
                    if window_idxs.size == 0:
                        continue
                    sample_sets = [[admixed_sample]] + [[s] for s in non_admixed_in_label]
                    indexes = [(0, i) for i in range(1, len(sample_sets))]
                    padded_edges, window_idxs = pad_windows_to_sequence(
                        window_edges[admixed_sample], window_idxs, ts.sequence_length
                    )
                    counts_windowed = ts.pair_coalescence_counts(
                        sample_sets=sample_sets,
                        indexes=indexes,
                        windows=padded_edges,
                        span_normalise=False,
                        time_windows=time_windows,
                    )
                    # Sum over this label's windows, then split the overflow bin
                    # off: counts keep the finite bins, coverage keeps everything.
                    label_all = counts_windowed[window_idxs, :, :].sum(axis=0)
                    label_counts = label_all[:, :-1].astype(self.datatype)
                    label_coverage = label_all.sum(axis=1)

                    for i, non_admixed_sample in enumerate(non_admixed_in_label):
                        pair = (min(admixed_sample, non_admixed_sample), max(admixed_sample, non_admixed_sample))
                        local_idx = pair_to_idx[pair]
                        counts_per_ancestry[label][local_idx, :] = label_counts[i, :]
                        # Measured, not the LAI window width: the pair is observed
                        # only where BOTH samples have ancestry, which inside a
                        # label's windows is less than their total width wherever
                        # either sample is masked or the region is inaccessible.
                        coverage_per_ancestry[label][local_idx] = label_coverage[i]

            # --- Admixed × admixed: windowed by both samples sharing the ancestry
            for i, s1 in enumerate(admixed_in_label):
                for s2 in admixed_in_label[i + 1:]:
                    admixed_pair = (s1, s2)
                    edges = intersecting_window_edges.get(admixed_pair)
                    idxs_dict = intersecting_window_idxs_dict.get(admixed_pair)
                    if edges is None or idxs_dict is None:
                        continue
                    window_idxs = idxs_dict.get(label, np.array([], dtype=int))
                    if window_idxs.size == 0:
                        continue
                    padded_edges, window_idxs = pad_windows_to_sequence(
                        edges, window_idxs, ts.sequence_length
                    )
                    counts_windowed = ts.pair_coalescence_counts(
                        sample_sets=[[s1], [s2]],
                        windows=padded_edges,
                        span_normalise=False,
                        time_windows=time_windows,
                    )
                    pair_all = counts_windowed[window_idxs, :].sum(axis=0)
                    local_idx = pair_to_idx[admixed_pair]
                    counts_per_ancestry[label][local_idx, :] = pair_all[:-1].astype(self.datatype)
                    # Measured over the shared-ancestry windows, for the same
                    # reason as the admixed x non-admixed case above.
                    coverage_per_ancestry[label][local_idx] = pair_all.sum()

            # Atomic write, as in CountCoalEvents.count_coal_spans_single_ts:
            # an interrupted run must not leave a half-written counts file.
            out_path = os.path.join(self.ancestry_count_dirs[label], f"{contig_id}.npy")
            tmp_path = f"{out_path}.tmp.{os.getpid()}"
            try:
                # np.save appends .npy unless the name already ends in it, so the
                # buffer is written explicitly to keep tmp_path exact.
                with open(tmp_path, "wb") as fh:
                    np.save(fh, counts_per_ancestry[label], allow_pickle=True)
                os.replace(tmp_path, out_path)
            except BaseException:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
                raise

        return coverage_per_ancestry

    def count_coal_spans_all_ts(
        self,
        LAI_dir: str,
        num_threads: int = 1,
    ) -> None:
        """Count per-ancestry coalescent spans for every tree sequence and save coverage.

        Runs ``count_coal_spans_single_ts`` over every ``.trees`` file (serially
        or via a pathos pool), then assembles each ancestry's per-contig
        coverage vectors into ``{ancestry_count_dirs[label]}/coverage.parquet``.

        Parameters
        ----------
        LAI_dir : str
            Directory of per-contig local-ancestry parquet files.
        num_threads : int, optional
            Number of parallel workers. Defaults to 1 (serial execution).
        """
        ts_files = self.ts_files
        num_threads = min(num_threads, len(ts_files))

        if num_threads == 1:
            # --- Serial execution -----------------------------------------
            per_contig_coverage = [
                self.count_coal_spans_single_ts(ts_file, LAI_dir=LAI_dir)
                for ts_file in tqdm(ts_files, desc="Counting per-ancestry coalescent segments")
            ]
        else:
            # --- Parallel execution via pathos, imap for a live progress bar
            pool = Pool(nodes=num_threads)
            per_contig_coverage = imap_or_raise(
                pool,
                lambda ts_file: self.count_coal_spans_single_ts(ts_file, LAI_dir=LAI_dir),
                ts_files,
                "Counting per-ancestry coalescent segments",
            )
            # See the matching comment in CountCoalEvents.count_coal_spans_all_ts:
            # clear() evicts this pool from pathos's process-wide cache so a
            # later Pool(nodes=N) call with the same N doesn't reuse a closed pool.
            pool.clear()

        # --- Assemble and save each ancestry's coverage.parquet -----------
        contig_ids = [ts_file.removesuffix(".trees") for ts_file in ts_files]
        for label in self.ancestry_labels:
            coverage_data = np.column_stack([contig_cov[label] for contig_cov in per_contig_coverage])
            coverage_df = pd.DataFrame(
                coverage_data,
                index=self.pairs_MultiIndex_per_ancestry[label],
                columns=contig_ids,
            )
            coverage_df.to_parquet(os.path.join(self.ancestry_count_dirs[label], "coverage.parquet"))

    # ------------------------------------------------------------------ #
    #  PDF estimation
    # ------------------------------------------------------------------ #

    def compute_coal_time_pdf(self, ancestry_label: str) -> pd.DataFrame:
        """Estimate the coalescence-time PDF for one ancestry label.

        Sums that ancestry's per-contig ``.npy`` counts and normalises by its
        ``coverage.parquet`` (both produced by ``count_coal_spans_all_ts``).

        Parameters
        ----------
        ancestry_label : str
            Ancestry label to compute the PDF for; must be a key of
            ``self.ancestry_count_dirs``.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by ``(sample_1, sample_2)`` (restricted to
            samples included for this ancestry) with one column per
            time-bin left-edge.
        """
        ancestry_dir = self.ancestry_count_dirs[ancestry_label]
        expected_basenames = {ts_file.removesuffix(".trees") for ts_file in self.ts_files}
        count_files = {f.name.removesuffix(".npy") for f in Path(ancestry_dir).glob("*.npy")}
        missing = expected_basenames - count_files
        if missing:
            raise FileNotFoundError(
                f"ancestry_count_dirs['{ancestry_label}'] ({ancestry_dir}) is missing "
                f"coalescent-count files for {len(missing)} tree sequence(s): {sorted(missing)}"
            )

        num_pairs_label = len(self.sample_pairs_per_ancestry[ancestry_label])
        coal_segments_total = np.zeros((num_pairs_label, self.num_timepoints), dtype=self.datatype)
        for basename in tqdm(expected_basenames, desc=f"Summing '{ancestry_label}' counts across tree sequences"):
            coal_segments_total += np.load(
                os.path.join(ancestry_dir, f"{basename}.npy"), allow_pickle=True
            )

        coverage_df = pd.read_parquet(os.path.join(ancestry_dir, "coverage.parquet"))
        total_segments_per_pair = coverage_df.values.sum(axis=1)

        pdf_df = pd.DataFrame(
            coal_segments_total,
            columns=self.time_bin_left_edges,
            index=self.pairs_MultiIndex_per_ancestry[ancestry_label],
        )

        # As in CountCoalEvents.compute_coal_time_pdf: zero coverage on a single
        # contig is ordinary (it contributes nothing to either total), but a pair
        # with zero coverage genome-wide is observed nowhere and its PDF is
        # undefined. That is commoner here than in the base class, because two
        # samples may simply never share this ancestry label on any contig.
        denom = total_segments_per_pair.astype(np.float64)
        unobserved = denom == 0
        if np.any(unobserved):
            print(
                f"Warning: {int(unobserved.sum())} of {denom.size} pairs never share "
                f"ancestry '{ancestry_label}' on any contig and will be NaN in the PDF."
            )
        pdf_df = pdf_df.div(np.where(unobserved, np.nan, denom), axis=0)
        return pdf_df
