#!/usr/bin/env python3
"""Count pairwise coalescent events across tree sequences and estimate coalescence-time PDFs.

This module provides the `CountCoalEvents` class, which:
  1. Discovers and loads tree-sequence files (*.trees) from a directory.
  2. Counts the total genomic span that coalesces in discrete time bins
     for every pair of samples.
  3. Normalises the counts into an empirical PDF of pairwise coalescence
     times.

Supports multiple posterior ARG samples per contig (e.g. from SINGER) and
optional sub-sampling of sample pairs per contig for memory / speed trade-offs.
"""

from __future__ import annotations

import os
import re
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
    num_posterior_samples : int, optional
        Number of posterior ARG replicates per contig (for Bayesian ARG
        inference methods such as SINGER). Files must follow the naming
        convention ``{ContigID}_{PosteriorSampleID}.trees``. Defaults to 1.
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
        num_posterior_samples: int = 1,
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

        # --- Resolve contigs & posterior samples ------------------------
        self._init_contigs(num_posterior_samples)

        # --- Choose compact dtype based on total genome length ----------
        total_bp = int(np.sum(self.contig_lengths))
        self.datatype = np.uint32 if total_bp < 2**32 else np.uint64

        # --- Sample metadata -------------------------------------------
        ts_first = tskit.load(os.path.join(ts_dir, self.ts_files[0]))
        self.sample_sets = [[int(sid)] for sid in ts_first.samples()]

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
            columns=self.contig_ids,
        )

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _init_contigs(self, num_posterior_samples: int) -> None:
        """Populate contig/posterior-sample metadata from discovered files.

        Sets ``self.num_posterior_samples``, ``self.num_contigs``,
        ``self.contig_ids``, and ``self.contig_lengths``.
        """
        if num_posterior_samples == 1:
            # Each file is a single contig (no posterior replicates)
            self.num_posterior_samples = 1
            self.num_contigs = len(self.ts_files)
            self.contig_ids = [f.removesuffix(".trees") for f in self.ts_files]
            self.contig_lengths = np.array([
                tskit.load(os.path.join(self.ts_dir, cid + ".trees")).sequence_length
                for cid in tqdm(self.contig_ids, total=self.num_contigs,
                                desc="Reading contig lengths")
            ], dtype=np.uint64)
        else:
            # Multiple posterior samples per contig
            # File naming convention: {ContigID}_{PosteriorSampleID}.trees
            self.num_posterior_samples = num_posterior_samples
            self.contig_ids = np.unique([
                re.sub(r"_[0-9]+\.trees$", "", f) for f in self.ts_files
            ])
            self.num_contigs = len(self.contig_ids)

            if len(self.ts_files) % num_posterior_samples != 0:
                raise ValueError(
                    f"Number of .trees files ({len(self.ts_files)}) is not "
                    f"divisible by num_posterior_samples ({num_posterior_samples})"
                )

            # Use the first posterior sample (_0) to read sequence lengths
            self.contig_lengths = np.array([
                tskit.load(os.path.join(self.ts_dir, cid + "_0.trees")).sequence_length
                for cid in tqdm(self.contig_ids, total=self.num_contigs,
                                desc="Reading contig lengths")
            ], dtype=np.uint64)

    def _resolve_contig_id(self, ts_file: str) -> str:
        """Extract the contig ID from a tree-sequence filename."""
        if self.num_posterior_samples == 1:
            return ts_file.removesuffix(".trees")
        return re.sub(r"_[0-9]+\.trees$", "", ts_file)

    # ------------------------------------------------------------------ #
    #  Counting coalescent segments
    # ------------------------------------------------------------------ #

    def count_coal_segments(self, ts_file: str) -> np.ndarray:
        """Count coalescent segment spans in time bins for one tree sequence.

        For every selected sample pair, the function records how many base-pairs
        coalesce within each time bin (defined by a regular grid from 0 to
        ``self.max_time``).

        Parameters
        ----------
        ts_file : str
            Filename (not full path) of the ``.trees`` file inside
            ``self.ts_dir``.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_pairs, num_timepoints)`` with dtype
            ``self.datatype``.  Rows correspond to sample pairs (same order
            as ``self.sample_pairs``), columns to time bins.
        """
        ts = tskit.load(os.path.join(self.ts_dir, ts_file))
        contig_id = self._resolve_contig_id(ts_file)

        # --- Determine which samples are active on this contig ----------
        active_mask = self.samples_selected[contig_id]
        active_samples = active_mask[active_mask].index.values

        if len(active_samples) == 0:
            return np.zeros((self.num_pairs, self.num_timepoints), dtype=self.datatype)

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

        # Place results into the full (all-pairs) output array
        coal_segments_total = np.zeros(
            (self.num_pairs, self.num_timepoints), dtype=self.datatype
        )
        coal_segments_total[global_pair_idx, :] = coal_segments

        return coal_segments_total

    def count_coal_segments_contigs(self, ts_files: List[str]) -> np.ndarray:
        """Accumulate coalescent-segment counts across multiple tree sequences.

        Parameters
        ----------
        ts_files : list[str]
            Filenames of the tree sequences to process.

        Returns
        -------
        np.ndarray
            Summed array of shape ``(num_pairs, num_timepoints)``.
        """
        total_coal_segments: Optional[np.ndarray] = None

        for ts_file in tqdm(ts_files, desc="Counting coalescent segments"):
            coal_segments = self.count_coal_segments(ts_file)
            if total_coal_segments is None:
                total_coal_segments = coal_segments
            else:
                total_coal_segments += coal_segments

        return total_coal_segments  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  PDF estimation
    # ------------------------------------------------------------------ #

    def compute_coal_time_pdf(
        self,
        num_threads: int = 1,
        data_output: Optional[str] = None,
    ) -> pd.DataFrame:
        """Estimate the PDF of pairwise coalescence times.

        Counts coalescent segments across all tree-sequence files, then
        normalises by the total genomic span observed for each pair (accounting
        for sub-sampling and posterior replicates).

        Parameters
        ----------
        num_threads : int, optional
            Number of parallel workers.  When > 1, intermediate results are
            saved to disk in *data_output* and aggregated afterwards.
            Defaults to 1 (serial).
        data_output : str, optional
            Directory for temporary chunk files during parallel execution.
            **Required** when ``num_threads > 1``.

        Returns
        -------
        pd.DataFrame
            DataFrame indexed by ``(sample_1, sample_2)`` with one column per
            time-bin left-edge.  Values approximate the coalescence-time PDF.
        """
        # Cap thread count at the number of available files
        num_threads = min(num_threads, len(self.ts_files))

        if num_threads == 1:
            coal_segments_total = self.count_coal_segments_contigs(self.ts_files)
        else:
            coal_segments_total = self._count_parallel(num_threads, data_output)

        # --- Normalise counts into a PDF --------------------------------
        time_bin_edges = np.linspace(0, self.max_time, self.num_timepoints + 1)

        pdf_df = pd.DataFrame(
            coal_segments_total,
            columns=time_bin_edges[:-1],
            index=self.pairs_MultiIndex,
        )

        # Total observed genome length for each pair, respecting the
        # stochastic sub-sampling mask:  a pair is active on a contig only
        # if *both* samples were selected on that contig.
        total_segments_per_pair = self._total_segments_per_pair()

        pdf_df = pdf_df / (
            total_segments_per_pair.reshape(-1, 1) * self.num_posterior_samples
        )

        return pdf_df

    # ------------------------------------------------------------------ #
    #  Parallel helpers
    # ------------------------------------------------------------------ #

    def _count_parallel(
        self,
        num_threads: int,
        data_output: Optional[str],
    ) -> np.ndarray:
        """Run ``count_coal_segments_contigs`` in parallel across chunks.

        Each worker writes its result to a temporary ``.npy`` file; results are
        aggregated and cleaned up after all workers finish.

        Parameters
        ----------
        num_threads : int
            Number of parallel workers.
        data_output : str
            Directory for temporary chunk files.

        Returns
        -------
        np.ndarray
            Summed coalescent-segment counts across all chunks.
        """
        if data_output is None:
            raise ValueError(
                "data_output directory must be specified when using multiple threads."
            )
        os.makedirs(data_output, exist_ok=True)

        chunks = np.array_split(self.ts_files, num_threads)

        def _process_chunk(ts_files_chunk: list, chunk_id: int) -> None:
            """Count segments for a chunk and persist to disk."""
            result = self.count_coal_segments_contigs(ts_files_chunk)
            np.save(
                os.path.join(data_output, f"coal_segments_chunk_{chunk_id}.npy"),
                result,
            )

        # Dispatch via pathos
        pool = Pool(nodes=num_threads)
        pool.map(
            lambda args: _process_chunk(*args),
            [(chunk, cid) for cid, chunk in enumerate(chunks)],
        )

        # Aggregate chunk files
        coal_segments_total: Optional[np.ndarray] = None
        for chunk_id in range(len(chunks)):
            chunk_path = os.path.join(
                data_output, f"coal_segments_chunk_{chunk_id}.npy"
            )
            chunk_data = np.load(chunk_path)
            if coal_segments_total is None:
                coal_segments_total = chunk_data
            else:
                coal_segments_total += chunk_data
            os.remove(chunk_path)

        # Clean up temp directory (will only succeed if empty)
        os.rmdir(data_output)

        return coal_segments_total  # type: ignore[return-value]

    def _total_segments_per_pair(self) -> np.ndarray:
        """Compute total observed genome length for every sample pair for normalization to compute a pdf.

        A pair is considered active on a contig only when *both* members of
        the pair were selected (via ``self.samples_selected``).  The returned
        array is weighted by contig length so it represents total base-pairs
        observed.

        Returns
        -------
        np.ndarray
            1-D array of length ``num_pairs`` with dtype ``self.datatype``.
        """
        sel = self.samples_selected.values  # (num_samples, num_contigs)

        # Outer boolean AND: pair is active iff both samples are active
        # Result shape: (num_samples^2, num_contigs)
        pairs_selected = pd.DataFrame(
            np.logical_and(sel[:, None, :], sel[None, :, :]).reshape(
                self.num_samples**2, self.num_contigs
            ),
            index=pd.MultiIndex.from_product(
                [self.sample_ids, self.sample_ids],
                names=["sample_1", "sample_2"],
            ),
            columns=self.contig_ids,
        )

        # Keep only upper-triangle pairs (sample_1 < sample_2)
        mask = (
            pairs_selected.index.get_level_values("sample_1")
            < pairs_selected.index.get_level_values("sample_2")
        )
        pairs_selected = pairs_selected[mask]

        # Weight by contig lengths and sum across contigs
        return (
            (pairs_selected * self.contig_lengths)
            .values.sum(axis=1)
            .astype(self.datatype)
        )
