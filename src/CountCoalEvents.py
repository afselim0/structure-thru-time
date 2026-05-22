#%%
# #!/usr/bin/env python3
"""Count pairwise coalescent events across tree sequences and estimate empirical PDFs of pairwise-coalescence time.

This module provides the `CountCoalEvents` class, with two main workflows:

Genome-wide PDF (``compute_coal_time_pdf``)
    Computes the pairwise coalescence-time PDF across **all** samples
    (admixed and non-admixed alike), ignoring local ancestry entirely.
    Outputs a DataFrame with one row per unique sample pair (upper triangle) and columns for each time bin.
    Row indices are ``(sample_1_ID, sample_2_ID)`` integer pairs.

Ancestry-aware PDF (``compute_coal_time_pdf_admixed``)
    Computes ancestry-speicifc PDF of pairwise coalescence-time.
    Requires as an input:
        1- A list of admixed sample IDs
        2- Ancestry labels of genomic segments for each admixed sample.
    Row indices are ``(sample_1_ID.type, sample_2_ID)`` (if sample_1 is admixed and sample_2 is not),
    ``(sample_1_ID.type_A, sample_2_ID.type_A)`` (if both are admixed and carry the same label), or
    ``(sample_1_ID.type_A, sample_2_ID.type_B)`` (if both are admixed and carry different labels A≠B).
    ``subtype_float_id = sample_id + k / 10**d`` and *d* is the number of
    decimal digits required to represent the maximum rank *K* unambiguously
    (e.g. ``10.03`` for sample 10, type 3 when K ≥ 10, or ``10.3`` for K < 10).
    Cross-type pairs ``(i.A, j.B)`` and ``(i.B, j.A)`` are both included for every
    upper-triangle admixed pair ``(i, j)`` with ``i < j``; counts accumulate only
    over genomic windows where ``i`` carries label A and ``j`` carries label B
    simultaneously.

Common features
    - Discovers and loads ``.trees`` files from a directory.
    - Supports multiple posterior ARG replicates per contig (e.g. SINGER).
    - Optional stochastic sub-sampling of samples per contig.
    - All admixed samples must carry the same set of ancestry labels.
    - Admixture is described by one parquet file per contig in
      ``admixture_dir``, each with columns
      ``["sample_id", "start_pos", "end_pos", "ancestry_label"]``.
"""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
        Subset of sample node IDs to analyse.  If *None*, every sample in the
        first tree sequence is used.
    num_posterior_samples : int, optional
        Number of posterior ARG replicates per contig.  Defaults to 1.
    max_time : float, optional
        Upper bound of the time grid (generations).  Defaults to the 95th
        percentile of node times in the first tree sequence.
    num_timepoints : int, optional
        Number of equally-spaced time bins.  Defaults to 100.
    fraction_samples : float, optional
        Probability that a given sample is included per contig (sub-sampling).
        Defaults to 1.0.
    admixture_dir : str, optional
        Directory containing one parquet file per contig
        (``<contig_id>.parquet``), each with columns
        ``["sample_id", "start_pos", "end_pos", "ancestry_label"]``.
        Each row describes a contiguous genomic segment of a given ancestry
        for one admixed individual.  If *None*, no admixture handling is
        performed.  All admixed samples must share the same set of ancestry
        labels.

    Attributes
    ----------
    admixed_sample_ids : np.ndarray
        Integer IDs of admixed individuals (read from *admixture_df*).
    nonadmixed_sample_ids : np.ndarray
        Integer IDs of non-admixed individuals.
    shared_labels : list
        Sorted list of ancestry labels shared by all admixed samples.
        Empty when ``admixture_dir`` is *None*.
    admixture_labels : dict[int, list]
        Maps each admixed sample ID -> sorted list of its ancestry labels
        (identical for all admixed samples; kept for convenience).
    label_rank_decimals : int
        Number of decimal digits used to encode the ancestry rank in the
        float subtype ID (``ceil(log10(K + 1))`` where K = number of labels).
    admixed_subtype_ids : list[float]
        Virtual sub-sample IDs of the form
        ``sample_id + rank / 10**label_rank_decimals``
        for every admixed sample x ancestry label.
    all_sample_pairs : list[tuple]
        All unique integer-ID pairs (upper triangle) used by the genome-wide
        PDF.  Includes admixed x non-admixed and admixed x admixed.
    admixed_pairs : list[tuple]
        Float-subtype x non-admixed pairs used by the ancestry-aware PDF,
        plus same-type and cross-type admixed x admixed pairs.
        Categories: (a) (subtype_float, nonadmixed_int),
        (b) (subtype_float_i_A, subtype_float_j_A) for A==B (same-type),
        (c) (subtype_float_i_A, subtype_float_j_B) for A≠B (cross-type, both
        orderings registered for each upper-triangle admixed pair).
    """

    # ------------------------------------------------------------------ #
    #  Initialisation
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        ts_dir: str,
        admixture_dir: Optional[str] = None,
        sample_ids: Optional[List[int]] = None,
        num_posterior_samples: int = 1,
        max_time: Optional[float] = None,
        num_timepoints: int = 100,
        fraction_samples: float = 1.0
    ) -> None:
        self.ts_dir = ts_dir
        self.num_timepoints = num_timepoints
        self.fraction_samples = fraction_samples
        self.admixture_dir = admixture_dir

        print("Preprocessing tree-sequence files …")

        # --- Input validation -------------------------------------------
        if not os.path.isdir(ts_dir):
            raise FileNotFoundError(f"ts_dir is not a valid directory: {ts_dir}")
        if admixture_dir is not None and not os.path.isdir(admixture_dir):
            raise FileNotFoundError(f"admixture_dir is not a valid directory: {admixture_dir}")

        # --- Discover tree-sequence files -------------------------------
        self.ts_files = sorted(f.name for f in Path(ts_dir).glob("*.trees"))
        if not self.ts_files:
            raise ValueError(f"No .trees files found in directory: {ts_dir}")

        # --- Resolve contigs & posterior samples ------------------------
        self._init_contigs(num_posterior_samples)

        # --- Choose compact dtype based on total genome length ----------
        total_bp = int(np.sum(self.contig_lengths))
        self.datatype = np.uint32 if total_bp * num_posterior_samples < 2**32 else np.uint64

        # --- Sample metadata -------------------------------------------
        ts_first = tskit.load(os.path.join(ts_dir, self.ts_files[0]))
        self.sample_sets = [[int(sid)] for sid in ts_first.samples()]

        if sample_ids is None:
            all_sample_ids = ts_first.samples()
        else:
            all_sample_ids = np.sort(np.asarray(sample_ids))

        # --- Parse admixture information --------------------------------
        # _init_admixture only needs one representative parquet to discover
        # which sample IDs are admixed and what their ancestry labels are.
        # All per-contig files are expected to have the same set of samples
        # and labels; individual files are loaded on demand during counting.
        if admixture_dir is not None:
            parquet_files = sorted(Path(admixture_dir).glob("*.parquet"))
            parquet_contigs = [pf.name.removesuffix(".parquet") for pf in parquet_files]
            assert len(parquet_files) == len(self.contig_ids), f"Number of admixture parquet file {len(parquet_files)}, must match number of tree-sequence files {len(self.contig_ids)}"
            assert set(parquet_contigs) == set(self.contig_ids), "Admixture parquet files must match tree-sequence files by name"
            if not parquet_files:
                raise ValueError(f"No .parquet files found in admixture_dir: {admixture_dir}")
            _representative_df = pd.read_parquet(parquet_files[0])
            self._init_admixture(_representative_df, all_sample_ids)
        else:
            self._init_admixture(None, all_sample_ids)

        # sample_ids used throughout = nonadmixed + admixed (integer IDs)
        self.sample_ids = np.sort(
            np.concatenate([self.nonadmixed_sample_ids, self.admixed_sample_ids])
        ).astype(int)
        self.num_samples = len(self.sample_ids)

        # --- Time grid --------------------------------------------------
        if max_time is None:
            max_time = float(np.quantile(ts_first.nodes_time, 0.95))
        self.max_time = max_time

        # --- Build genome-wide pairs (all integer IDs, upper triangle) ----
        # Every unique unordered pair of integer sample IDs (admixed or not).
        # These are the row indices of the genome-wide PDF DataFrame.
        all_ids = self.sample_ids  # sorted integer IDs
        self.all_sample_pairs: List[Tuple[int, int]] = [
            (int(all_ids[i]), int(all_ids[j]))
            for i in range(len(all_ids))
            for j in range(i + 1, len(all_ids))
        ]
        self.num_all_pairs = len(self.all_sample_pairs)

        # MultiIndex and flat lookup used by count_coal_segments()
        self.all_pairs_MultiIndex = pd.MultiIndex.from_tuples(
            self.all_sample_pairs, names=["sample_1", "sample_2"]
        )
        # Maps (s1_int, s2_int) -> row index in the genome-wide count array
        self.all_pair_to_idx: Dict[Tuple[int, int], int] = {
            pair: k for k, pair in enumerate(self.all_sample_pairs)
        }

        # --- Build ancestry-aware pairs -----------------------------------
        # subtype_id = sample_id + rank / 10**d, where d = ceil(log10(K+1))
        # and K = number of shared ancestry labels.
        #
        # Three categories:
        #   (a) admixed-subtype x non-admixed   : (subtype_float, nonadmixed_int)
        #   (b) same-type admixed x admixed     : (subtype_float_i, subtype_float_j)
        #       e.g. (40.1, 50.1) — only windows where BOTH carry the same label count.
        #   (c) cross-type admixed x admixed    : (subtype_float_i_A, subtype_float_j_B), A≠B
        #       e.g. (40.1, 50.2) — only windows where i carries A and j carries B count.
        #       Both (i.A, j.B) and (i.B, j.A) are registered for every upper-triangle
        #       admixed pair (i, j) with i < j.
        nonadm = self.nonadmixed_sample_ids
        self.admixed_pairs: List[Tuple] = []

        # Category (a): one row per (admixed subtype, non-admixed sample) pair
        for adm_id in self.admixed_sample_ids:
            labels = self.admixture_labels[adm_id]
            for rank, _label in enumerate(labels, start=1):
                subtype_id = self._subtype_id(int(adm_id), rank)
                for nad_id in nonadm:
                    self.admixed_pairs.append((subtype_id, int(nad_id)))

        # Category (b): upper-triangle pairs of admixed individuals at the same rank.
        # (40.1, 50.1) means "both carry label 1": counts only from their intersection.
        adm_ids = self.admixed_sample_ids
        for i in range(len(adm_ids)):
            for j in range(i + 1, len(adm_ids)):
                labels = self.admixture_labels[adm_ids[i]]
                for rank, _label in enumerate(labels, start=1):
                    st_i = self._subtype_id(int(adm_ids[i]), rank)
                    st_j = self._subtype_id(int(adm_ids[j]), rank)
                    self.admixed_pairs.append((st_i, st_j))

        # Category (c): cross-type pairs for every upper-triangle admixed pair (i, j).
        # For each ordered label pair (A, B) with A≠B, register both (i.A, j.B) and
        # (i.B, j.A): counts accumulate only where i carries A and j carries B
        # (or vice versa) simultaneously.
        for i in range(len(adm_ids)):
            for j in range(i + 1, len(adm_ids)):
                labels = self.admixture_labels[adm_ids[i]]  # same for all admixed samples
                ranks = list(enumerate(labels, start=1))
                for rank_A, _label_A in ranks:
                    for rank_B, _label_B in ranks:
                        if rank_A == rank_B:
                            continue  # same-type already handled in (b)
                        st_i_A = self._subtype_id(int(adm_ids[i]), rank_A)
                        st_j_B = self._subtype_id(int(adm_ids[j]), rank_B)
                        self.admixed_pairs.append((st_i_A, st_j_B))

        self.num_admixed_pairs = len(self.admixed_pairs)

        self.admixed_pairs_MultiIndex = pd.MultiIndex.from_tuples(
            self.admixed_pairs, names=["sample_1", "sample_2"]
        )
        # Flat-index lookup for ancestry-aware pairs
        self.admixed_pair_to_idx: Dict[Tuple, int] = {
            pair: k for k, pair in enumerate(self.admixed_pairs)
        }

        # --- Stochastic sub-sampling mask (all integer samples × contigs) --
        self.samples_selected = pd.DataFrame(
            np.random.rand(self.num_samples, self.num_contigs) < self.fraction_samples,
            index=self.sample_ids,
            columns=self.contig_ids,
        )

    # ------------------------------------------------------------------ #
    #  Admixture initialisation
    # ------------------------------------------------------------------ #

    def _init_admixture(
        self,
        admixture_df: Optional[pd.DataFrame],
        all_sample_ids: np.ndarray,
    ) -> None:
        """Derive admixed/non-admixed sample metadata from one representative parquet.

        Called once during ``__init__`` with a single parquet loaded from
        ``admixture_dir``.  All per-contig files are assumed to share the same
        set of admixed sample IDs and ancestry labels; per-contig data is
        loaded on demand via ``_load_admixture_for_contig``.

        All admixed samples must carry the same set of ancestry labels.  A
        ``ValueError`` is raised if any sample has a different label set.

        Sets
        ----
        self.admixed_sample_ids : np.ndarray   (integer IDs)
        self.nonadmixed_sample_ids : np.ndarray (integer IDs)
        self.shared_labels : list
            Sorted list of ancestry labels shared by every admixed sample.
        self.admixture_labels : dict[int, list]
            admixed_id -> shared_labels  (same list for every admixed sample)
        self.label_rank_decimals : int
            Decimal digits needed to encode ranks 1..K unambiguously, i.e.
            ``ceil(log10(K + 1))``.  E.g. K=9 -> 1, K=10..99 -> 2.
        self.admixed_subtype_to_label : dict[float, str]
            subtype_id -> ancestry label string
        self.admixed_subtype_to_base : dict[float, int]
            subtype_id -> integer base sample ID
        """
        if admixture_df is None:
            self.admixed_sample_ids = np.array([], dtype=int)
            self.nonadmixed_sample_ids = np.sort(all_sample_ids).astype(int)
            self.shared_labels: List = []
            self.admixture_labels: Dict[int, List] = {}
            self.label_rank_decimals: int = 1
            self.admixed_subtype_to_label: Dict[float, str] = {}
            self.admixed_subtype_to_base: Dict[float, int] = {}
            return

        required_cols = {"sample_id", "start_pos", "end_pos", "ancestry_label"}
        if not required_cols.issubset(admixture_df.columns):
            raise ValueError(
                f"admixture parquet files must have columns {required_cols}; "
                f"got {set(admixture_df.columns)}"
            )

        admixture_df = admixture_df.copy()
        admixture_df["sample_id"] = admixture_df["sample_id"].astype(int)

        admixed_in_df = np.unique(admixture_df["sample_id"].values)

        # If sample_ids was provided, restrict admixed IDs to the intersection
        admixed_ids = np.intersect1d(admixed_in_df, all_sample_ids)
        self.admixed_sample_ids = np.sort(admixed_ids).astype(int)

        # Non-admixed = all requested samples that are NOT in the admixed set
        self.nonadmixed_sample_ids = np.sort(
            np.setdiff1d(all_sample_ids, self.admixed_sample_ids)
        ).astype(int)

        # Verify and collect shared label set across all admixed samples
        per_sample_labels: Dict[int, List] = {}
        for adm_id in self.admixed_sample_ids:
            labels = sorted(
                admixture_df.loc[
                    admixture_df["sample_id"] == adm_id, "ancestry_label"
                ].unique()
            )
            per_sample_labels[adm_id] = labels

        # All admixed samples must share the same label set
        label_sets = [frozenset(v) for v in per_sample_labels.values()]
        if len(set(label_sets)) > 1:
            raise ValueError(
                "All admixed samples must have the same set of ancestry labels. "
                "Differing label sets found:\n"
                + "\n".join(
                    f"  sample {sid}: {sorted(per_sample_labels[sid])}"
                    for sid in self.admixed_sample_ids
                )
            )

        self.shared_labels: List = sorted(label_sets[0]) if label_sets else []
        K = len(self.shared_labels)
        # Number of decimal digits required to represent ranks 1..K uniquely
        self.label_rank_decimals: int = max(1, math.ceil(math.log10(K + 1))) if K > 0 else 1

        self.admixture_labels: Dict[int, List] = {
            adm_id: self.shared_labels for adm_id in self.admixed_sample_ids
        }

        self.admixed_subtype_to_label: Dict[float, str] = {}
        self.admixed_subtype_to_base: Dict[float, int] = {}
        for adm_id in self.admixed_sample_ids:
            for rank, label in enumerate(self.shared_labels, start=1):
                subtype_id = self._subtype_id(int(adm_id), rank)
                self.admixed_subtype_to_label[subtype_id] = label
                self.admixed_subtype_to_base[subtype_id] = int(adm_id)

        print(
            f"Admixture: {len(self.admixed_sample_ids)} admixed sample(s), "
            f"{len(self.nonadmixed_sample_ids)} non-admixed sample(s), "
            f"{K} shared label(s): {self.shared_labels}."
        )

    # ------------------------------------------------------------------ #
    #  Subtype-ID encoding helper
    # ------------------------------------------------------------------ #

    def _subtype_id(self, sample_id: int, rank: int) -> float:
        """Encode ``sample_id.rank`` as a float with zero-padded rank.

        The rank is right-padded to ``self.label_rank_decimals`` decimal
        digits so that ranks never collide regardless of the base sample_id.

        Examples (label_rank_decimals=2, i.e. K in 10..99)
        -------
        sample 10, rank 3  -> 10.03
        sample 10, rank 13 -> 10.13
        sample 1,  rank 3  ->  1.03

        Parameters
        ----------
        sample_id : int
        rank : int
            1-based ancestry rank.

        Returns
        -------
        float
        """
        d = self.label_rank_decimals
        return round(sample_id + rank / (10 ** d), d + 10)

    # ------------------------------------------------------------------ #
    #  Private helpers
    # ------------------------------------------------------------------ #

    def _init_contigs(self, num_posterior_samples: int) -> None:
        """Populate contig/posterior-sample metadata from discovered files."""
        if num_posterior_samples == 1:
            self.num_posterior_samples = 1
            self.num_contigs = len(self.ts_files)
            self.contig_ids = [f.removesuffix(".trees") for f in self.ts_files]
            self.contig_lengths = np.array([
                tskit.load(os.path.join(self.ts_dir, cid + ".trees")).sequence_length
                for cid in tqdm(self.contig_ids, total=self.num_contigs,
                                desc="Reading contig lengths")
            ], dtype=np.uint64)
        else:
            self.num_posterior_samples = num_posterior_samples
            contig_ids = np.unique([
                re.sub(r"_[0-9]+\.trees$", "", f) for f in self.ts_files
            ])
            contigs_with_complete_replicates = []
            for cid in contig_ids:
                cid_tree_files = [f for f in self.ts_files if f.startswith(cid + "_")]
                num_found = len(cid_tree_files)
                if num_found == num_posterior_samples:
                    contigs_with_complete_replicates.append(cid)
                else:
                    print(
                        f"Warning: Contig '{cid}' has {num_found} posterior samples "
                        f"(expected {num_posterior_samples}). Skipping."
                    )

            self.contig_ids = np.array(contigs_with_complete_replicates)
            self.num_contigs = len(self.contig_ids)
            self.ts_files = [
                f for f in self.ts_files
                if any(f.startswith(cid + "_") for cid in self.contig_ids)
            ]
            print(
                f"Using {len(self.ts_files)} tree-sequence files across "
                f"{self.num_contigs} contigs with "
                f"{self.num_posterior_samples} posterior samples each."
            )
            self.contig_lengths = np.array([
                tskit.load(os.path.join(self.ts_dir, cid + "_0.trees")).sequence_length
                for cid in tqdm(self.contig_ids, total=self.num_contigs,
                                desc="Reading contig lengths")
            ], dtype=np.uint64)

            print(f"Total used genome length across contigs: {np.sum(self.contig_lengths)} bp")
            for cid, length in zip(self.contig_ids, self.contig_lengths):
                print(f"  Contig '{cid}': {length} bp")

    def _resolve_contig_id(self, ts_file: str) -> str:
        """Extract the contig ID from a tree-sequence filename."""
        if self.num_posterior_samples == 1:
            return ts_file.removesuffix(".trees")
        return re.sub(r"_[0-9]+\.trees$", "", ts_file)

    def _load_admixture_for_contig(self, contig_id: str) -> pd.DataFrame:
        """Load and return the admixture parquet for *contig_id*.

        Parameters
        ----------
        contig_id : str

        Returns
        -------
        pd.DataFrame
            The parquet table for this contig, with ``sample_id`` cast to
            ``int``.

        Raises
        ------
        FileNotFoundError
            If ``<admixture_dir>/<contig_id>.parquet`` does not exist.
        """
        path = os.path.join(self.admixture_dir, f"{contig_id}.parquet")
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Admixture parquet not found for contig '{contig_id}': {path}"
            )
        df = pd.read_parquet(path)
        df["sample_id"] = df["sample_id"].astype(int)
        return df

    def _ancestry_windows_for_contig(
        self, adm_id: int, df: pd.DataFrame, seq_length: float
    ) -> np.ndarray:
        """Return sorted genomic breakpoints for *adm_id* from a contig's df.

        Parameters
        ----------
        adm_id : int
        df : pd.DataFrame
            The already-loaded admixture parquet for this contig.
        seq_length : float

        Returns
        -------
        np.ndarray
            1-D sorted float array of breakpoints, always starting at 0 and
            ending at *seq_length*.
        """
        sub = df[df["sample_id"] == adm_id]
        # Collect all unique boundary positions for this sample on this contig.
        positions = np.unique(
            np.concatenate([sub["start_pos"].values, sub["end_pos"].values])
        )
        positions = positions[(positions >= 0) & (positions <= seq_length)]
        windows = np.unique(np.concatenate([[0.0], positions, [float(seq_length)]]))
        return windows

    def _ancestry_label_for_window(
        self, adm_id: int, df: pd.DataFrame, window_start: float, window_end: float
    ) -> Optional[str]:
        """Return the ancestry label covering the midpoint of a window.

        Parameters
        ----------
        adm_id : int
        df : pd.DataFrame
            The already-loaded admixture parquet for this contig.
        window_start : float
        window_end : float

        Returns
        -------
        str or None
            None if the midpoint falls in an unannotated gap.
        """
        mid = (window_start + window_end) / 2.0
        sub = df[
            (df["sample_id"] == adm_id)
            & (df["start_pos"] <= mid)
            & (df["end_pos"] > mid)
        ]
        if sub.empty:
            return None
        return sub.iloc[0]["ancestry_label"]

    def _intersection_windows_for_pair(
        self,
        adm_id_1: int,
        adm_id_2: int,
        label_1,
        label_2,
        df: pd.DataFrame,
        seq_length: float,
    ) -> np.ndarray:
        """Return breakpoints of genomic regions where adm_id_1 carries label_1 AND adm_id_2 carries label_2.

        Intersects the segments annotated with *label_1* for *adm_id_1* and
        the segments annotated with *label_2* for *adm_id_2* via a sweep-line
        algorithm and returns the result as a flat
        ``[lo_0, hi_0, lo_1, hi_1, ...]`` array for iteration when calling
        ``pair_coalescence_counts``.

        When ``label_1 == label_2`` this is equivalent to the former single-label
        behaviour (same-type intersection).  When ``label_1 != label_2`` it
        computes the cross-type intersection.

        Returns an empty array (length 0) when there is no overlapping region.
        """
        def _segments(adm_id: int, label) -> np.ndarray:
            # Filter rows belonging to this sample with the given label.
            # Cast label to the column's actual dtype to avoid type-mismatch
            # false negatives (e.g. label stored as int but passed as str).
            col_dtype = df["ancestry_label"].dtype
            try:
                label_cast = col_dtype.type(label)
            except (ValueError, TypeError):
                label_cast = label
            sub = df[(df["sample_id"] == adm_id) & (df["ancestry_label"] == label_cast)]
            if sub.empty:
                return np.empty((0, 2), dtype=np.float64)
            # Extract [start, end) intervals, clip to contig bounds, drop zero-length
            segs = sub[["start_pos", "end_pos"]].values.astype(np.float64)
            segs[:, 0] = np.clip(segs[:, 0], 0.0, seq_length)
            segs[:, 1] = np.clip(segs[:, 1], 0.0, seq_length)
            segs = segs[segs[:, 0] < segs[:, 1]]
            # Sort by start position for the sweep-line algorithm
            return segs[np.argsort(segs[:, 0])]

        segs1 = _segments(adm_id_1, label_1)
        segs2 = _segments(adm_id_2, label_2)

        # No intersection possible if either sample has no segments for this label
        if segs1.shape[0] == 0 or segs2.shape[0] == 0:
            return np.empty(0, dtype=np.float64)

        # Sweep-line intersection: advance the pointer of whichever interval ends first
        breakpoints: List[float] = []
        i, j = 0, 0
        while i < len(segs1) and j < len(segs2):
            lo = max(segs1[i, 0], segs2[j, 0])  # overlap start
            hi = min(segs1[i, 1], segs2[j, 1])  # overlap end
            if lo < hi:
                # Record this overlapping interval
                breakpoints.append(lo)
                breakpoints.append(hi)
            # Advance the pointer whose interval ends sooner
            if segs1[i, 1] < segs2[j, 1]:
                i += 1
            else:
                j += 1

        if not breakpoints:
            return np.empty(0, dtype=np.float64)

        # Merge any abutting or overlapping intervals produced by the sweep
        # (can arise if input segments themselves overlap or abut)
        merged: List[float] = []
        cur_lo, cur_hi = breakpoints[0], breakpoints[1]
        for k in range(2, len(breakpoints), 2):
            nlo, nhi = breakpoints[k], breakpoints[k + 1]
            if nlo <= cur_hi:
                cur_hi = max(cur_hi, nhi)
            else:
                merged.extend([cur_lo, cur_hi])
                cur_lo, cur_hi = nlo, nhi
        merged.extend([cur_lo, cur_hi])

        return np.array(merged, dtype=np.float64)

    # ------------------------------------------------------------------ #
    #  Counting coalescent segments — non-admixed pairs
    # ------------------------------------------------------------------ #

    def count_coal_segments_chunk(self, ts_file: str) -> np.ndarray:
        """Count coalescent segment spans in time bins for one tree sequence.

        Genome-wide computation across **all** samples (admixed and
        non-admixed), ignoring local ancestry entirely.  Admixed individuals
        are treated as ordinary integer-ID samples.

        Parameters
        ----------
        ts_file : str
            Filename (not full path) of the ``.trees`` file.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_all_pairs, num_timepoints)`` with dtype
            ``self.datatype``.
        """
        ts = tskit.load(os.path.join(self.ts_dir, ts_file))
        contig_id = self._resolve_contig_id(ts_file)

        # All samples (admixed + non-admixed) active on this contig
        active_mask = self.samples_selected.loc[self.sample_ids, contig_id]
        active_samples = active_mask[active_mask].index.values  # integer IDs

        coal_segments_total = np.zeros(
            (self.num_all_pairs, self.num_timepoints), dtype=self.datatype
        )

        if len(active_samples) < 2:
            return coal_segments_total

        # Simplify tree sequence to active samples only
        ts_simp = (
            ts.simplify(samples=active_samples)
            if len(active_samples) < ts.num_samples
            else ts
        )

        # Build local pair <-> global-index mapping
        local_pairs: List[Tuple[int, int]] = []
        global_pair_idxs: List[int] = []
        for i, s1 in enumerate(active_samples):
            for j in range(i + 1, len(active_samples)):
                s2 = active_samples[j]
                key = (int(s1), int(s2))
                if key in self.all_pair_to_idx:
                    local_pairs.append((i, j))
                    global_pair_idxs.append(self.all_pair_to_idx[key])

        if not local_pairs:
            return coal_segments_total

        global_pair_idx = np.array(global_pair_idxs)
        time_windows = np.linspace(0, self.max_time, self.num_timepoints + 1)

        coal_segments = ts_simp.pair_coalescence_counts(
            sample_sets=[[i] for i in range(len(active_samples))],
            indexes=local_pairs,
            span_normalise=False,
            time_windows=time_windows,
        ).astype(self.datatype)

        coal_segments_total[global_pair_idx, :] = coal_segments
        return coal_segments_total

    # ------------------------------------------------------------------ #
    #  Counting coalescent segments — admixed pairs
    # ------------------------------------------------------------------ #

    def count_coal_segments_chunk_admixed(self, ts_file: str) -> np.ndarray:
        """Count coalescent segments for all ancestry-aware pair categories on one contig.

        Three categories are handled for each active admixed individual:

        (a) Admixed-subtype x non-admixed
            Build genomic windows from all local-ancestry breakpoints of the
            admixed individual.  Call ``pair_coalescence_counts`` once against
            all active non-admixed samples, yielding shape
            ``(num_gwindows, num_active_nonadmixed, num_timepoints)``.
            For each label, find the window indices carrying that label
            (vectorised) and sum the corresponding slices into the output rows
            for ``(adm_subtype, nonadmixed_id)``.

        (b) Same-type admixed x admixed and (c) cross-type admixed x admixed
            For each upper-triangle pair of active admixed individuals
            ``(adm_id_1, adm_id_2)``, build the K² intersection window sets
            (one per ordered label pair ``(A, B)``).  Merge all non-empty
            intersection breakpoints into a single ``windows`` array and call
            ``pair_coalescence_counts`` once per pair.  Recover each subtype
            pair's contributing window indices via ``np.searchsorted`` and
            accumulate into the output rows.

        Results are stored in a standalone array indexed by
        ``self.admixed_pair_to_idx`` — independent from the genome-wide array.

        Parameters
        ----------
        ts_file : str

        Returns
        -------
        np.ndarray
            Array of shape ``(num_admixed_pairs, num_timepoints)``.
        """
        coal_segments_total = np.zeros(
            (self.num_admixed_pairs, self.num_timepoints), dtype=self.datatype
        )

        if self.admixture_dir is None or len(self.admixed_sample_ids) == 0:
            return coal_segments_total

        ts = tskit.load(os.path.join(self.ts_dir, ts_file))
        contig_id = self._resolve_contig_id(ts_file)
        seq_length = ts.sequence_length

        # Load this contig's admixture table once for all admixed samples
        admix_df = self._load_admixture_for_contig(contig_id)

        time_windows = np.linspace(0, self.max_time, self.num_timepoints + 1)

        # ---------------------------------------------------------------- #
        # (a) Admixed-subtype x non-admixed
        # One pair_coalescence_counts call per active admixed individual,
        # covering all non-admixed samples at once.  Window-to-label assignment
        # is vectorised: for each label we collect all window indices carrying
        # that label in a single pass and sum the corresponding rows.
        # ---------------------------------------------------------------- #
        nonadm = self.nonadmixed_sample_ids
        if len(nonadm) > 0:
            # Active non-admixed samples on this contig
            nonadm_active_mask = self.samples_selected.loc[nonadm, contig_id]
            active_nonadmixed = nonadm_active_mask[nonadm_active_mask].index.values

            for adm_id in self.admixed_sample_ids:
                # Skip admixed samples that are inactive on this contig
                if not self.samples_selected.loc[adm_id, contig_id]:
                    continue
                if len(active_nonadmixed) == 0:
                    continue

                # Build genomic breakpoints from all ancestry boundaries of adm_id
                genomic_windows = self._ancestry_windows_for_contig(
                    adm_id, admix_df, seq_length
                )
                num_gwindows = len(genomic_windows) - 1

                if num_gwindows == 0:
                    continue

                # Simplify ts to {adm_id} ∪ {active_nonadmixed}; local index
                # 0 = adm_id, 1..N = active_nonadmixed (in order)
                all_active = np.concatenate([[adm_id], active_nonadmixed]).astype(int)
                ts_simp = (
                    ts.simplify(samples=all_active)
                    if len(all_active) < ts.num_samples
                    else ts
                )
                pairs_local = [(0, j) for j in range(1, len(active_nonadmixed) + 1)]

                # Returns shape (num_gwindows, num_nonadmixed, num_timepoints)
                counts_windowed = ts_simp.pair_coalescence_counts(
                    sample_sets=[[i] for i in range(len(all_active))],
                    indexes=pairs_local,
                    windows=genomic_windows,
                    span_normalise=False,
                    time_windows=time_windows,
                ).astype(self.datatype)

                labels = self.admixture_labels[adm_id]  # shared sorted label list

                # Vectorised label assignment: for each window find its midpoint
                # label, then group window indices by label.
                win_mids = (genomic_windows[:-1] + genomic_windows[1:]) / 2.0
                # Build array of label strings parallel to windows (None for gaps)
                win_labels = np.array([
                    self._ancestry_label_for_window(adm_id, admix_df, genomic_windows[w], genomic_windows[w + 1])
                    for w in range(num_gwindows)
                ], dtype=object)

                for rank, label in enumerate(labels, start=1):
                    subtype_id = self._subtype_id(int(adm_id), rank)
                    # Indices of windows that carry this label
                    label_win_idxs = np.where(win_labels == label)[0]
                    if len(label_win_idxs) == 0:
                        continue

                    # Warn about unannotated gaps (windows with label None)
                    if rank == 1:
                        gap_idxs = np.where(win_labels == None)[0]  # noqa: E711
                        for w in gap_idxs:
                            print(
                                f"Warning: Unannotated gap for admixed sample {adm_id} on "
                                f"contig '{contig_id}' between positions "
                                f"{genomic_windows[w]} and {genomic_windows[w + 1]}. "
                                f"Skipping this window."
                            )

                    # Sum counts over all windows carrying this label:
                    # counts_windowed shape (num_gwindows, num_nonadmixed, num_timepoints)
                    # label_counts shape: (num_nonadmixed, num_timepoints)
                    label_counts = counts_windowed[label_win_idxs, :, :].sum(axis=0)

                    for n_idx, nad_id in enumerate(active_nonadmixed):
                        key = (subtype_id, int(nad_id))
                        row = self.admixed_pair_to_idx.get(key)
                        if row is None:
                            continue
                        coal_segments_total[row, :] += label_counts[n_idx]

        # ---------------------------------------------------------------- #
        # (b)+(c) Admixed x admixed — same-type and cross-type
        # For each upper-triangle pair of active admixed individuals, build all
        # K² intersection window sets (one per ordered label pair (A, B)).
        # Merge into a single windows array for one pair_coalescence_counts
        # call, then recover each subtype pair's window indices via searchsorted
        # and accumulate into the output rows.
        # ---------------------------------------------------------------- #
        adm_ids = self.admixed_sample_ids
        # Restrict to admixed samples active on this contig
        active_adm_mask = self.samples_selected.loc[adm_ids, contig_id]
        active_adm_ids = active_adm_mask[active_adm_mask].index.values

        for ii in range(len(active_adm_ids)):
            for jj in range(ii + 1, len(active_adm_ids)):
                adm_id_1 = int(active_adm_ids[ii])
                adm_id_2 = int(active_adm_ids[jj])

                labels = self.admixture_labels[adm_id_1]  # shared label list
                ranks = list(enumerate(labels, start=1))

                # Simplify ts to just these two individuals once per pair;
                # local index 0 = adm_id_1, 1 = adm_id_2
                ts_simp = ts.simplify(
                    samples=np.array([adm_id_1, adm_id_2], dtype=int)
                )
                pair_local = [(0, 1)]

                # Build K² intersection breakpoint arrays, one per (A, B) label pair.
                # inter_bps_map[(rank_A, rank_B)] = flat [lo_0,hi_0,...] or empty array
                inter_bps_map: Dict[Tuple[int, int], np.ndarray] = {}
                for rank_A, label_A in ranks:
                    for rank_B, label_B in ranks:
                        inter_bps_map[(rank_A, rank_B)] = self._intersection_windows_for_pair(
                            adm_id_1, adm_id_2, label_A, label_B, admix_df, seq_length
                        )

                # Merge all non-empty intersection breakpoints into one windows array.
                # np.unique sorts and de-duplicates; 0.0 and seq_length are always included.
                all_bps: List[float] = [0.0, float(seq_length)]
                for bps in inter_bps_map.values():
                    if len(bps) >= 2:
                        all_bps.extend(bps.tolist())
                full_windows = np.unique(all_bps)

                # Skip this pair entirely if no labelled intersections exist on this contig
                if len(full_windows) <= 2 and not any(
                    len(bps) >= 2 for bps in inter_bps_map.values()
                ):
                    continue

                # One pair_coalescence_counts call covering all K² subtype windows.
                # Returns shape (num_windows, 1, num_timepoints)
                counts_all = ts_simp.pair_coalescence_counts(
                    sample_sets=[[0], [1]],
                    indexes=pair_local,
                    windows=full_windows,
                    span_normalise=False,
                    time_windows=time_windows,
                ).astype(self.datatype)
                # Squeeze pair dimension: shape (num_windows, num_timepoints)
                counts_all = counts_all[:, 0, :]

                # Accumulate counts for each (rank_A, rank_B) subtype pair.
                # For each intersection interval [lo, hi], find its index in
                # full_windows via searchsorted and add the corresponding row.
                for rank_A, _label_A in ranks:
                    for rank_B, _label_B in ranks:
                        inter_bps = inter_bps_map[(rank_A, rank_B)]
                        if len(inter_bps) < 2:
                            continue

                        st_1 = self._subtype_id(adm_id_1, rank_A)
                        st_2 = self._subtype_id(adm_id_2, rank_B)
                        row = self.admixed_pair_to_idx.get((st_1, st_2))
                        if row is None:
                            continue

                        # Sum counts over all windows belonging to this intersection.
                        # Each interval [lo, hi] maps to one window index in full_windows.
                        for k in range(0, len(inter_bps), 2):
                            lo = inter_bps[k]
                            w = np.searchsorted(full_windows, lo)
                            coal_segments_total[row, :] += counts_all[w, :]

        return coal_segments_total

    # ------------------------------------------------------------------ #
    #  Aggregation across contigs
    # ------------------------------------------------------------------ #

    def count_coal_segments_all_chunks(
        self, ts_files: List[str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Accumulate coalescent-segment counts across multiple tree sequences.

        Returns two independent arrays:

        * ``genome_wide`` — shape ``(num_all_pairs, num_timepoints)`` from
          :meth:`count_coal_segments_chunk` (all integer-ID pairs, no ancestry).
        * ``ancestry_aware`` — shape ``(num_admixed_pairs, num_timepoints)``
          from :meth:`count_coal_segments_chunk_admixed` (subtype x non-admixed,
          ancestry masked).

        Parameters
        ----------
        ts_files : list[str]

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(genome_wide_total, ancestry_aware_total)``
        """
        genome_wide_total: Optional[np.ndarray] = None
        ancestry_aware_total: Optional[np.ndarray] = None

        for ts_file in tqdm(ts_files, desc="Counting coalescent segments"):
            seg = self.count_coal_segments_chunk(ts_file)
            seg_adm = self.count_coal_segments_chunk_admixed(ts_file)

            if genome_wide_total is None:
                genome_wide_total = seg
                ancestry_aware_total = seg_adm
            else:
                genome_wide_total += seg
                ancestry_aware_total += seg_adm  # type: ignore[operator]

        return genome_wide_total, ancestry_aware_total  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  PDF estimation — genome-wide
    # ------------------------------------------------------------------ #

    def compute_coal_time_pdf(
        self,
        num_threads: int = 1,
        data_output: Optional[str] = None,
    ) -> pd.DataFrame:
        """Genome-wide pairwise coalescence-time PDF, ignoring local ancestry.

        All samples (admixed and non-admixed) are treated as plain integer-ID
        samples.  Every unique pair is included.

        Parameters
        ----------
        num_threads : int, optional
            Number of parallel workers.  Defaults to 1.
        data_output : str, optional
            Directory for temporary chunk files (required when
            ``num_threads > 1``).

        Returns
        -------
        pd.DataFrame
            Indexed by ``(sample_1, sample_2)`` integer pairs with one column
            per time-bin left-edge.  Normalised by the total active genome
            length per pair (sum of contig lengths where both samples are
            active) times ``num_posterior_samples``.
        """
        num_threads = min(num_threads, len(self.ts_files))

        if num_threads == 1:
            genome_wide_total, _ = self.count_coal_segments_all_chunks(self.ts_files)
        else:
            genome_wide_total = self._count_parallel_genomewide(num_threads, data_output)

        time_bin_edges = np.linspace(0, self.max_time, self.num_timepoints + 1)

        pdf_df = pd.DataFrame(
            genome_wide_total,
            columns=time_bin_edges[:-1],
            index=self.all_pairs_MultiIndex,
        )

        total_segments = self._total_segments_per_pair_genomewide()
        pdf_df = pdf_df / (
            total_segments.reshape(-1, 1) * self.num_posterior_samples
        )
        return pdf_df

    # ------------------------------------------------------------------ #
    #  PDF estimation — ancestry-aware
    # ------------------------------------------------------------------ #

    def compute_coal_time_pdf_admixed(
        self,
        num_threads: int = 1,
        data_output: Optional[str] = None,
    ) -> pd.DataFrame:
        """Ancestry-aware pairwise coalescence-time PDF.

        Computes the PDF for three categories of pairs, all indexed by
        float subtype IDs of the form
        ``subtype_float_id = sample_id + rank / 10**d``
        where *d* = ``self.label_rank_decimals``:

        (a) Admixed-subtype x non-admixed — ``(subtype_float_id, nonadmixed_int_id)``.
            Counts accumulate only over genomic windows where the admixed
            individual carries the subtype's ancestry label.  Normalised by
            the total span of that label on contigs where both individuals
            are active.

        (b) Same-type admixed x admixed — ``(subtype_float_id_i, subtype_float_id_j)``
            with both subtypes sharing the same rank (e.g. ``(40.1, 50.1)``).
            Counts accumulate only over genomic windows where *both* individuals
            simultaneously carry that label.  Normalised by the total length
            of that same-label intersection on jointly-active contigs.

        (c) Cross-type admixed x admixed — ``(subtype_float_id_i_A, subtype_float_id_j_B)``
            with A != B (e.g. ``(40.1, 50.2)``).  Both orderings are included
            for every upper-triangle admixed pair ``(i, j)`` with ``i < j``.
            Counts accumulate only over genomic windows where individual *i*
            carries label A *and* individual *j* carries label B simultaneously.
            Normalised by the total length of that cross-label intersection on
            jointly-active contigs.

        Parameters
        ----------
        num_threads : int, optional
            Number of parallel workers.  Defaults to 1.
        data_output : str, optional
            Directory for temporary chunk files (required when
            ``num_threads > 1``).

        Returns
        -------
        pd.DataFrame
            Indexed by the union of all three pair categories above, with one
            column per time-bin left-edge.  Each row is normalised by the
            observed ancestry-masked genome length for that pair (intersection
            length on jointly-active contigs) times ``num_posterior_samples``,
            so that normalisation is correct even when ``fraction_samples < 1``.
            Returns an empty DataFrame when no admixed samples are present.
        """
        if self.admixture_dir is None or len(self.admixed_sample_ids) == 0:
            return pd.DataFrame(index=self.admixed_pairs_MultiIndex)

        num_threads = min(num_threads, len(self.ts_files))

        if num_threads == 1:
            _, ancestry_aware_total = self.count_coal_segments_all_chunks(self.ts_files)
        else:
            ancestry_aware_total = self._count_parallel_admixed(num_threads, data_output)

        time_bin_edges = np.linspace(0, self.max_time, self.num_timepoints + 1)

        pdf_df = pd.DataFrame(
            ancestry_aware_total,
            columns=time_bin_edges[:-1],
            index=self.admixed_pairs_MultiIndex,
        )

        total_segments = self._total_segments_per_pair_admixed()
        pdf_df = pdf_df / (
            total_segments.reshape(-1, 1) * self.num_posterior_samples
        )
        return pdf_df

    # ------------------------------------------------------------------ #
    #  PDF estimation — both at once
    # ------------------------------------------------------------------ #

    def compute_coal_time_pdfs(
        self,
        num_threads: int = 1,
        data_output: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Compute both the genome-wide and ancestry-aware PDFs in a single pass.

        Equivalent to calling :meth:`compute_coal_time_pdf` and
        :meth:`compute_coal_time_pdf_admixed` separately, but processes each
        ``.trees`` file only once, making it roughly twice as fast when both
        outputs are needed.

        Parameters
        ----------
        num_threads : int, optional
            Number of parallel workers.  Defaults to 1.
        data_output : str, optional
            Directory for temporary chunk files (required when
            ``num_threads > 1``).

        Returns
        -------
        pdf_gw : pd.DataFrame
            Genome-wide PDF — same as :meth:`compute_coal_time_pdf`.
        pdf_aa : pd.DataFrame
            Ancestry-aware PDF — same as :meth:`compute_coal_time_pdf_admixed`.
            Covers all three pair categories: admixed-subtype x non-admixed (a),
            same-type admixed x admixed (b), and cross-type admixed x admixed (c).
            See :meth:`compute_coal_time_pdf_admixed` for normalisation details.
        """
        num_threads = min(num_threads, len(self.ts_files))

        if num_threads == 1:
            genome_wide_total, ancestry_aware_total = self.count_coal_segments_all_chunks(
                self.ts_files
            )
        else:
            genome_wide_total, ancestry_aware_total = self._count_parallel_both(
                num_threads, data_output
            )

        time_bin_edges = np.linspace(0, self.max_time, self.num_timepoints + 1)

        # Genome-wide PDF
        pdf_gw = pd.DataFrame(
            genome_wide_total,
            columns=time_bin_edges[:-1],
            index=self.all_pairs_MultiIndex,
        )
        total_gw = self._total_segments_per_pair_genomewide()
        pdf_gw = pdf_gw / (total_gw.reshape(-1, 1) * self.num_posterior_samples)

        # Ancestry-aware PDF
        if self.admixture_dir is None or len(self.admixed_sample_ids) == 0:
            pdf_aa = pd.DataFrame(index=self.admixed_pairs_MultiIndex)
        else:
            pdf_aa = pd.DataFrame(
                ancestry_aware_total,
                columns=time_bin_edges[:-1],
                index=self.admixed_pairs_MultiIndex,
            )
            total_aa = self._total_segments_per_pair_admixed()
            pdf_aa = pdf_aa / (total_aa.reshape(-1, 1) * self.num_posterior_samples)

        return pdf_gw, pdf_aa

    # ------------------------------------------------------------------ #
    #  Parallel helpers
    # ------------------------------------------------------------------ #

    def _count_parallel_both(
        self,
        num_threads: int,
        data_output: Optional[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run ``count_coal_segments_all_chunks`` in parallel, saving both arrays.

        Each worker processes its chunk and saves both the genome-wide and
        ancestry-aware arrays to disk.  The main process aggregates them and
        returns both totals.
        """
        if data_output is None:
            raise ValueError(
                "data_output directory must be specified when using multiple threads."
            )
        os.makedirs(data_output, exist_ok=True)
        chunks = np.array_split(self.ts_files, num_threads)

        def _process_chunk(ts_files_chunk: list, chunk_id: int) -> None:
            gw, aa = self.count_coal_segments_all_chunks(ts_files_chunk)
            np.save(os.path.join(data_output, f"coal_gw_chunk_{chunk_id}.npy"), gw)
            np.save(os.path.join(data_output, f"coal_aa_chunk_{chunk_id}.npy"), aa)

        pool = Pool(nodes=num_threads)
        pool.map(
            lambda args: _process_chunk(*args),
            [(chunk, cid) for cid, chunk in enumerate(chunks)],
        )

        print("Aggregating results from parallel workers …")
        gw_total: Optional[np.ndarray] = None
        aa_total: Optional[np.ndarray] = None
        for chunk_id in tqdm(range(len(chunks))):
            gw_path = os.path.join(data_output, f"coal_gw_chunk_{chunk_id}.npy")
            aa_path = os.path.join(data_output, f"coal_aa_chunk_{chunk_id}.npy")
            gw_chunk = np.load(gw_path)
            aa_chunk = np.load(aa_path)
            gw_total = gw_chunk if gw_total is None else gw_total + gw_chunk
            aa_total = aa_chunk if aa_total is None else aa_total + aa_chunk  # type: ignore[operator]
            os.remove(gw_path)
            os.remove(aa_path)

        os.rmdir(data_output)
        return gw_total, aa_total  # type: ignore[return-value]

    def _count_parallel_genomewide(
        self,
        num_threads: int,
        data_output: Optional[str],
    ) -> np.ndarray:
        """Run genome-wide counting in parallel across contig chunks."""
        if data_output is None:
            raise ValueError(
                "data_output directory must be specified when using multiple threads."
            )
        os.makedirs(data_output, exist_ok=True)
        chunks = np.array_split(self.ts_files, num_threads)

        def _process_chunk(ts_files_chunk: list, chunk_id: int) -> None:
            gw, _ = self.count_coal_segments_all_chunks(ts_files_chunk)
            np.save(
                os.path.join(data_output, f"coal_gw_chunk_{chunk_id}.npy"), gw
            )

        pool = Pool(nodes=num_threads)
        pool.map(
            lambda args: _process_chunk(*args),
            [(chunk, cid) for cid, chunk in enumerate(chunks)],
        )

        print("Aggregating genome-wide results from parallel workers …")
        total: Optional[np.ndarray] = None
        for chunk_id in tqdm(range(len(chunks))):
            chunk_path = os.path.join(data_output, f"coal_gw_chunk_{chunk_id}.npy")
            chunk_data = np.load(chunk_path)
            total = chunk_data if total is None else total + chunk_data
            os.remove(chunk_path)

        os.rmdir(data_output)
        return total  # type: ignore[return-value]

    def _count_parallel_admixed(
        self,
        num_threads: int,
        data_output: Optional[str],
    ) -> np.ndarray:
        """Run ancestry-aware counting in parallel across contig chunks."""
        if data_output is None:
            raise ValueError(
                "data_output directory must be specified when using multiple threads."
            )
        os.makedirs(data_output, exist_ok=True)
        chunks = np.array_split(self.ts_files, num_threads)

        def _process_chunk(ts_files_chunk: list, chunk_id: int) -> None:
            _, aa = self.count_coal_segments_all_chunks(ts_files_chunk)
            np.save(
                os.path.join(data_output, f"coal_aa_chunk_{chunk_id}.npy"), aa
            )

        pool = Pool(nodes=num_threads)
        pool.map(
            lambda args: _process_chunk(*args),
            [(chunk, cid) for cid, chunk in enumerate(chunks)],
        )

        print("Aggregating ancestry-aware results from parallel workers …")
        total: Optional[np.ndarray] = None
        for chunk_id in tqdm(range(len(chunks))):
            chunk_path = os.path.join(data_output, f"coal_aa_chunk_{chunk_id}.npy")
            chunk_data = np.load(chunk_path)
            total = chunk_data if total is None else total + chunk_data
            os.remove(chunk_path)

        os.rmdir(data_output)
        return total  # type: ignore[return-value]

    # ------------------------------------------------------------------ #
    #  Normalisation helpers
    # ------------------------------------------------------------------ #

    def _total_segments_per_pair_genomewide(self) -> np.ndarray:
        """Total active genome length for every genome-wide pair.

        For each pair ``(s1, s2)`` the normaliser is the sum of lengths of
        contigs where both ``s1`` and ``s2`` are active (selected).  Local
        ancestry is ignored; admixed samples are treated as plain integers.

        Returns
        -------
        np.ndarray
            1-D array of length ``num_all_pairs``, dtype ``float64``.
        """
        all_ids = self.sample_ids  # sorted integer array
        sel = self.samples_selected.loc[all_ids].values  # (n_samples, n_contigs)
        n = len(all_ids)

        result = np.zeros(self.num_all_pairs, dtype=np.float64)
        k = 0
        for i in range(n):
            for j in range(i + 1, n):
                joint = sel[i] & sel[j]  # (n_contigs,) bool
                result[k] = float((joint * self.contig_lengths).sum())
                k += 1
        return result

    def _total_segments_per_pair_admixed(self) -> np.ndarray:
        """Total ancestry-masked genome length for every ancestry-aware pair.

        For each ``(subtype_id, nonadmixed_id)`` pair the normaliser is the
        total span of segments carrying the subtype's ancestry label on contigs
        where both the admixed individual and the non-admixed individual are
        active.

        For same-type and cross-type admixed x admixed pairs the normaliser is
        the total length of the cross-label intersection (positions where
        adm_id_1 carries label_A and adm_id_2 carries label_B simultaneously),
        summed over jointly-active contigs.

        Returns
        -------
        np.ndarray
            1-D array of length ``num_admixed_pairs``, dtype ``float64``.
        """
        result = np.zeros(self.num_admixed_pairs, dtype=np.float64)

        if self.admixture_dir is None or len(self.admixed_sample_ids) == 0:
            return result

        nonadm = self.nonadmixed_sample_ids

        for adm_id in tqdm(self.admixed_sample_ids, desc="Computing normalisers for admixed pairs"):
            labels = self.admixture_labels[adm_id]

            adm_active = self.samples_selected.loc[adm_id].values   # (num_contigs,)
            nonadm_active = self.samples_selected.loc[nonadm].values  # (n_nad, n_contigs)
            joint_active = adm_active[None, :] & nonadm_active       # (n_nad, n_contigs)

            for rank, label in enumerate(labels, start=1):
                subtype_id = self._subtype_id(int(adm_id), rank)

                # Per-contig span of this ancestry label for adm_id
                contig_span: Dict[str, float] = {}
                for cid in self.contig_ids:
                    df = self._load_admixture_for_contig(str(cid))
                    sub = df[
                        (df["sample_id"] == adm_id)
                        & (df["ancestry_label"] == label)
                    ]
                    contig_span[cid] = float(
                        (sub["end_pos"] - sub["start_pos"]).clip(lower=0).sum()
                    )

                ancestry_span_arr = np.array(
                    [contig_span[cid] for cid in self.contig_ids], dtype=np.float64
                )  # (num_contigs,)

                # Effective span per (non-admixed sample, contig)
                effective = joint_active * ancestry_span_arr[None, :]  # (n_nad, n_contigs)
                totals_adm = effective.sum(axis=1)                      # (n_nad,)

                for n_idx, nad_id in enumerate(nonadm):
                    key = (subtype_id, int(nad_id))
                    result[self.admixed_pair_to_idx[key]] = totals_adm[n_idx]

        # ---------------------------------------------------------------- #
        # (b)+(c) Same-type and cross-type admixed x admixed
        # Normaliser = total cross-label intersection length where adm_id_1
        # carries label_A and adm_id_2 carries label_B simultaneously,
        # summed over jointly-active contigs.
        #
        # Restructured for speed: iterate over contigs in the outer loop so
        # each parquet is loaded exactly once.  For each contig, pre-extract
        # sorted segment arrays for every (adm_id, label) combination, then
        # sweep over all upper-triangle (i, j) admixed pairs and all K²
        # label combinations without any further I/O.  Intersection *length*
        # is accumulated directly via the sweep-line — no breakpoint array is
        # materialised.
        # ---------------------------------------------------------------- #
        adm_ids = self.admixed_sample_ids
        labels = self.shared_labels
        ranks = list(enumerate(labels, start=1))

        # Pre-compute flat row indices for all (i, j, rank_A, rank_B) combinations
        # so the inner loop only does arithmetic, no dict lookups on the critical path.
        # row_map[(i, j, rank_A, rank_B)] = flat index into result, or None if absent.
        row_map: Dict[Tuple[int, int, int, int], Optional[int]] = {}
        for i in range(len(adm_ids)):
            for j in range(i + 1, len(adm_ids)):
                for rank_A, _ in ranks:
                    for rank_B, _ in ranks:
                        st_1 = self._subtype_id(int(adm_ids[i]), rank_A)
                        st_2 = self._subtype_id(int(adm_ids[j]), rank_B)
                        row_map[(i, j, rank_A, rank_B)] = self.admixed_pair_to_idx.get((st_1, st_2))

        # Accumulator: (i, j, rank_A, rank_B) -> total intersection length
        totals: Dict[Tuple[int, int, int, int], float] = {
            key: 0.0 for key, row in row_map.items() if row is not None
        }

        # Activity flags for all admixed samples across all contigs: (n_adm, n_contigs)
        adm_active_all = self.samples_selected.loc[adm_ids].values

        for c_idx, cid in enumerate(self.contig_ids):
            seq_length = float(self.contig_lengths[c_idx])
            df = self._load_admixture_for_contig(str(cid))
            col_dtype = df["ancestry_label"].dtype

            # Pre-extract sorted segment arrays for every (adm_id, label) on this contig.
            # segs_cache[(adm_id, label)] = (n_segs, 2) float64 array, sorted by start.
            segs_cache: Dict[Tuple[int, any], np.ndarray] = {}
            for adm_id in adm_ids:
                sub_adm = df[df["sample_id"] == adm_id]
                for _, label in ranks:
                    try:
                        label_cast = col_dtype.type(label)
                    except (ValueError, TypeError):
                        label_cast = label
                    sub = sub_adm[sub_adm["ancestry_label"] == label_cast]
                    if sub.empty:
                        segs_cache[(int(adm_id), label)] = np.empty((0, 2), dtype=np.float64)
                        continue
                    segs = sub[["start_pos", "end_pos"]].values.astype(np.float64)
                    segs[:, 0] = np.clip(segs[:, 0], 0.0, seq_length)
                    segs[:, 1] = np.clip(segs[:, 1], 0.0, seq_length)
                    segs = segs[segs[:, 0] < segs[:, 1]]
                    segs_cache[(int(adm_id), label)] = segs[np.argsort(segs[:, 0])]

            # Sweep over all upper-triangle admixed pairs active on this contig
            for i in tqdm(range(len(adm_ids)), desc="Computing normalisers for admixed pairs"):
                if not adm_active_all[i, c_idx]:
                    continue
                for j in range(i + 1, len(adm_ids)):
                    if not adm_active_all[j, c_idx]:
                        continue

                    adm_id_1 = int(adm_ids[i])
                    adm_id_2 = int(adm_ids[j])

                    for rank_A, label_A in ranks:
                        segs1 = segs_cache[(adm_id_1, label_A)]
                        if segs1.shape[0] == 0:
                            continue
                        for rank_B, label_B in ranks:
                            key = (i, j, rank_A, rank_B)
                            if key not in totals:
                                continue
                            segs2 = segs_cache[(adm_id_2, label_B)]
                            if segs2.shape[0] == 0:
                                continue

                            # Sweep-line intersection length (no breakpoint array needed)
                            length = 0.0
                            pi, pj = 0, 0
                            while pi < len(segs1) and pj < len(segs2):
                                lo = max(segs1[pi, 0], segs2[pj, 0])
                                hi = min(segs1[pi, 1], segs2[pj, 1])
                                if lo < hi:
                                    length += hi - lo
                                if segs1[pi, 1] < segs2[pj, 1]:
                                    pi += 1
                                else:
                                    pj += 1
                            totals[key] += length

        # Write accumulated totals into result
        for key, row in row_map.items():
            if row is not None:
                result[row] = totals[key]

        return result