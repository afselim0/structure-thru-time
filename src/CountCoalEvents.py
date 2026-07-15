#!/usr/bin/env python3
"""Count pairwise coalescent events across tree sequences and estimate coalescence-time PDFs.

This module provides the `CountCoalEvents` class, which:
  1. Discovers and loads tree-sequence files (*.trees) from a directory.
  2. Counts the total genomic span that coalesces in discrete time bins
     for every pair of samples for each tree sequence file, and saves the counts to disk.
  3. Normalises the counts into an empirical PDF of pairwise coalescence
     times.
count files are saved in .npy format, and the final PDF is saved in .parquet format.
The parquet has multi-index rows for sample IDs and columns for time-bin left edges.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import tskit
from pathos.multiprocessing import ProcessingPool as Pool
from tqdm import tqdm


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
    max_time : float, optional
        Upper bound of the time grid (in generations). If *None*, defaults to
        the 95th percentile of node times in the first tree sequence.
    num_timepoints : int, optional
        Number of equally-spaced time bins between 0 and *max_time*.
        Defaults to 100.
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
        Sequence length (bp) for each contig.
    sample_pairs : list[tuple[int, int]]
        All unique ordered pairs of sample IDs.
    samples_selected : pd.DataFrame
        Boolean mask indicating which samples are active on each contig.
    """

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        ts_dir: str,
        sample_ids: Optional[List[int]] = None,
        max_time: Optional[float] = None,
        num_timepoints: int = 100,
        fraction_samples: float = 1.0,
    ) -> None:
        self.ts_dir = ts_dir
        self.num_timepoints = num_timepoints
        self.fraction_samples = fraction_samples

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
        if max_time is None:
            max_time = float(np.quantile(ts_first.nodes_time, 0.95))
        self.max_time = max_time

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
        self.coverage_df = self._build_coverage_df()
        self.total_segments_per_pair = self.coverage_df.values.sum(axis=1).astype(self.datatype)

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
        coalesce within each time bin (defined by a regular grid from 0 to
        ``self.max_time``), and saves the result to disk. Nothing is returned;
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
            time_windows = np.linspace(0, self.max_time, self.num_timepoints + 1)
            coal_segments = ts.pair_coalescence_counts(
                sample_sets=[[i] for i in range(len(active_samples))],
                indexes=local_pairs,
                span_normalise=False,
                time_windows=time_windows,
            ).astype(self.datatype)

            coal_segments_total[global_pair_idx, :] = coal_segments

        coal_segments_total.dump(os.path.join(coal_counts_dir, f"{ts_file.removesuffix('.trees')}.npy"))

    def count_coal_spans_contigs_all_ts(
        self,
        ts_files: List[str],
        coal_counts_dir: str,
        num_threads: int = 1,
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
        """
        os.makedirs(coal_counts_dir, exist_ok=True)
        num_threads = min(num_threads, len(ts_files))

        if num_threads == 1:
            # --- Serial execution -----------------------------------------
            for ts_file in tqdm(ts_files, desc="Counting coalescent segments"):
                self.count_coal_spans_single_ts(ts_file, coal_counts_dir=coal_counts_dir)
        else:
            # --- Parallel execution via pathos -----------------------------
            pool = Pool(nodes=num_threads)
            pool.map(
                lambda ts_file: self.count_coal_spans_single_ts(ts_file, coal_counts_dir=coal_counts_dir),
                ts_files,
            )

    # ------------------------------------------------------------------ #
    #  PDF estimation
    # ------------------------------------------------------------------ #

    def compute_coal_time_pdf(
        self,
        coal_counts_dir: str
    ) -> pd.DataFrame:
        """Estimate the PDF of pairwise coalescence times given the unnormalised counts.

        Loads the per-file coalescent-span counts previously saved to
        *coal_counts_dir* by ``count_coal_spans_contigs_all_ts``, sums them
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
        """
        # --- Verify counts exist for every tree-sequence file ------------
        expected_basenames = {ts_file.removesuffix(".trees") for ts_file in self.ts_files}
        count_files = {
            f.name.removesuffix(".npy") for f in Path(coal_counts_dir).glob("*.npy")
        }
        missing = expected_basenames - count_files
        if missing:
            raise FileNotFoundError(
                f"coal_counts_dir '{coal_counts_dir}' is missing coalescent-count "
                f"files for {len(missing)} tree sequence(s): {sorted(missing)}"
            )

        # --- Load and sum counts across all tree sequences ----------------
        coal_segments_total = np.zeros((self.num_pairs, self.num_timepoints), dtype=self.datatype)
        for basename in tqdm(expected_basenames, desc="Loading coalescent counts"):
            coal_segments_total += np.load(
                os.path.join(coal_counts_dir, f"{basename}.npy"), allow_pickle=True
            )

        # --- Normalise counts into a PDF --------------------------------
        time_bin_edges = np.linspace(0, self.max_time, self.num_timepoints + 1)

        pdf_df = pd.DataFrame(
            coal_segments_total,
            columns=time_bin_edges[:-1],
            index=self.pairs_MultiIndex,
        )

        # Normalise by the total observed genome length for each pair,
        # respecting the stochastic sub-sampling mask computed in __init__.
        pdf_df = pdf_df / self.total_segments_per_pair.reshape(-1, 1)
        return pdf_df

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _build_coverage_df(self) -> pd.DataFrame:
        """Build the per-file sequence-length coverage dataframe.

        Rows are ``(sample_1, sample_2)`` pairs; columns are the basenames of
        the ``.trees`` files (without the ``.trees`` suffix).  Each entry is
        ``ts.sequence_length`` for the contig the file belongs to when *both*
        samples of that pair were selected on that contig, and zero otherwise.

        Returns
        -------
        pd.DataFrame
            Shape ``(num_pairs, num_ts_files)``.
        """
        sel = self.samples_selected.values  # (num_samples, num_contigs)

        # Both-samples-selected matrix: (num_samples, num_samples, num_contigs)
        both_selected = np.logical_and(sel[:, None, :], sel[None, :, :])

        # Index lookup to avoid O(n) list.index() calls inside the loop
        sample_row = {sid: i for i, sid in enumerate(self.sample_ids)}

        # Flatten to upper-triangle pairs × contigs
        pair_contig_active = np.array([
            both_selected[sample_row[s1], sample_row[s2], :]
            for s1, s2 in self.sample_pairs
        ])  # shape: (num_pairs, num_contigs)

        # Map each ts_file to its contig index and sequence length
        contig_id_list = list(self.contig_ids)
        col_names: list[str] = []
        data = np.zeros((self.num_pairs, len(self.ts_files)), dtype=np.float64)
        for col_idx, ts_file in enumerate(self.ts_files):
            contig_id = ts_file.removesuffix(".trees")
            contig_idx = contig_id_list.index(contig_id)
            seq_len = float(self.contig_lengths[contig_idx])
            data[:, col_idx] = pair_contig_active[:, contig_idx] * seq_len
            col_names.append(ts_file.removesuffix(".trees"))

        return pd.DataFrame(data, index=self.pairs_MultiIndex, columns=col_names)

