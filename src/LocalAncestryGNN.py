#%%
"""
Local ancestry inference for admixed samples.

This module provides two directory-oriented painters that both walk every
``.trees`` file in a directory and write one ``.parquet`` segment table per
contig to a user-specified output directory (named ``<contig_id>.parquet``):

  * ``LocalAncestryInference`` — infers local ancestry per local tree using
    ``TreeSequence.genealogical_nearest_neighbours`` against reference
    sample panels, then merges consecutive same-label trees into segments.
    Optionally relabels segments below ``min_segment_size`` to match a
    neighbouring segment (probabilities and boundaries are left untouched).
  * ``TrueLocalAncestry`` — recovers ground-truth local ancestry from census
    nodes baked into the simulation by
    ``msprime.Demography.add_census(time=admix_time + 1)``.

Both classes expose ``infer_single`` (one tree sequence) and ``infer_all``
(every ``.trees`` file in the directory, optionally in parallel, mirroring
``CountCoalEvents.count_coal_spans_contigs_all_ts``).

Downstream of these painters:
  * ``AncestryPaintingComparator`` / ``compare_ancestry_paintings`` compare an
    inferred painting against ground truth at base-pair resolution.
  * ``plot_chromosome_painting`` visualises inferred (probability-aware) vs.
    true ancestry for a single sample.
"""

from __future__ import annotations
import matplotlib.pyplot as plt

import os
from pathlib import Path
from typing import Any, NamedTuple, Optional

import numpy as np
import pandas as pd
import tskit
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Local ancestry inference  (GNN-based)
# ─────────────────────────────────────────────────────────────────────────────

class LocalAncestryInference:
    """
    Infer local ancestry for admixed samples using
    ``TreeSequence.genealogical_nearest_neighbours`` on each local tree.

    For each tree the GNN proportion vector is computed for every focal
    sample across the reference ``ancestry_samples`` groups.  The group
    with the highest proportion (majority vote / argmax) is assigned as
    the ancestry label for that tree interval.

    Parameters
    ----------
    ts_dir : str | Path
        Directory containing one or more ``.trees`` files.
    admixed_sample_ids : list[int]
        Node IDs of the admixed (query) haplotypes — must be consistent
        across all tree sequences in the directory.
    ancestry_samples : dict[Any, list[int]]
        Mapping from ancestry label → list of reference sample node IDs.
        Must contain at least 2 groups; admixed IDs must not appear here.
        Must be consistent across all tree sequences in the directory.
    """

    def __init__(
        self,
        ts_dir: str | Path,
        admixed_sample_ids: list[int],
        ancestry_samples: dict[Any, list[int]],
    ) -> None:
        self.ts_dir = Path(ts_dir)
        self.admixed_sample_ids = admixed_sample_ids
        self.ancestry_samples = ancestry_samples
        self._validate()

        self.ts_files = sorted(self.ts_dir.glob("*.trees"))
        if not self.ts_files:
            raise FileNotFoundError(f"No .trees files found in: {self.ts_dir}")

    def _validate(self) -> None:
        if not self.ts_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.ts_dir}")
        if not self.admixed_sample_ids:
            raise ValueError("`admixed_sample_ids` must not be empty.")
        if len(self.ancestry_samples) < 2:
            raise ValueError("`ancestry_samples` must contain at least 2 groups.")
        all_reference = {s for grp in self.ancestry_samples.values() for s in grp}
        overlap = all_reference & set(self.admixed_sample_ids)
        if overlap:
            raise ValueError(
                f"Node IDs present in both admixed and reference sets: {overlap}"
            )

    def infer_single(
        self,
        ts_path: str | Path,
        min_segment_size: Optional[float] = None,
    ) -> pd.DataFrame:
        """
        Run inference on a single tree sequence and return a merged-segment
        DataFrame.

        For each local tree the raw GNN proportion vector is recorded for
        every focal sample, giving the probability of belonging to each
        reference ancestry group.  The winning label (``ancestry_label``) is
        still the group with the highest proportion.  When consecutive trees
        share the same winning label they are merged into one segment; the
        reported probabilities are the span-weighted averages of the
        per-tree GNN values across all merged trees.  If ``min_segment_size``
        is given, segments shorter than that span then have their
        ``ancestry_label`` overwritten to match a neighbouring segment —
        probability columns and segment boundaries are left untouched, so
        short segments are NOT re-merged into their neighbour; see
        ``_collapse_short_segments``.

        Parameters
        ----------
        ts_path : str | Path
            Path to a ``.trees`` file.
        min_segment_size : float, optional
            If given, segments shorter than this span (in base pairs) have
            their label relabeled to a neighbouring segment's label, without
            altering probability columns or segment boundaries; see
            ``_collapse_short_segments``. Default: no collapsing.

        Returns
        -------
        pd.DataFrame
            Columns:
              sample_id, start_pos, end_pos, ancestry_label,
              prob_<label> for each label in ancestry_samples
        """
        if min_segment_size is not None and min_segment_size <= 0:
            raise ValueError("`min_segment_size` must be positive.")

        ts = tskit.load(str(ts_path))

        labels    = list(self.ancestry_samples.keys())
        prob_cols = [f"prob_{lbl}" for lbl in labels]
        sample_sets = [self.ancestry_samples[lbl] for lbl in labels]
        rows: list[dict] = []

        for tree in ts.trees():
            start, end = tree.interval.left, tree.interval.right

            # Isolate this single tree so that genealogical_nearest_neighbours
            # returns local (not genome-wide averaged) GNN proportions.
            single_tree_ts = ts.keep_intervals(
                [[start, end]], simplify=False
            ).trim()

            # gnn shape: (n_focal, n_groups)
            gnn = single_tree_ts.genealogical_nearest_neighbours(
                focal=self.admixed_sample_ids,
                sample_sets=sample_sets,
            )

            for fi, sample_id in enumerate(self.admixed_sample_ids):
                props = gnn[fi]  # shape: (n_groups,)
                missing = props.sum() == 0
                label   = None if missing else labels[int(np.argmax(props))]

                row: dict = {
                    "sample_id":      sample_id,
                    "start_pos":      start,
                    "end_pos":        end,
                    "ancestry_label": label,
                }
                for col, prob in zip(prob_cols, props):
                    # Store NaN for missing-data trees so downstream code
                    # can distinguish "truly 0 probability" from "no data".
                    row[col] = float("nan") if missing else float(prob)
                rows.append(row)

        df = pd.DataFrame(
            rows,
            columns=["sample_id", "start_pos", "end_pos", "ancestry_label"] + prob_cols,
        )
        merged = _merge_segments_with_probs(df, self.admixed_sample_ids, prob_cols)

        if min_segment_size is not None:
            merged = _collapse_short_segments(
                merged, self.admixed_sample_ids, prob_cols, min_segment_size
            )
        return merged

    def infer_all(
        self,
        output_dir: str | Path,
        num_threads: int = 1,
        min_segment_size: Optional[float] = None,
    ) -> None:
        """
        Run GNN local ancestry inference across every ``.trees`` file in
        ``self.ts_dir`` and write one ``.parquet`` file per contig to
        ``output_dir``.

        Each output file is named ``<contig_id>.parquet`` where ``contig_id``
        is the source filename without the ``.trees`` suffix.  Files are
        written independently — no aggregation step is required and results
        are immediately usable as each worker finishes.

        Parameters
        ----------
        output_dir : str | Path
            Directory where parquet files are written.  Created if it does
            not exist.
        num_threads : int
            Number of parallel worker processes.  When 1 (default), processes
            files sequentially.  When > 1, uses
            ``pathos.multiprocessing.ProcessingPool`` with one file per worker.
        min_segment_size : float, optional
            Forwarded to :meth:`infer_single` for every tree sequence; see
            its docstring. Default: no collapsing.

        Returns
        -------
        None
            Results are written to disk; nothing is returned or held in memory.

        Notes
        -----
        * Memory usage per worker is proportional to one tree sequence.
          With ``num_threads=k``, peak memory is k × (one tree sequence).
        * To load a specific contig's results afterwards:
            ``pd.read_parquet("output_dir/chr1.parquet")``
        * To load all contigs lazily (e.g. with dask or pyarrow):
            ``pd.read_parquet("output_dir/")``
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        work_items = [
            (str(p), self.admixed_sample_ids, self.ancestry_samples,
             min_segment_size, str(output_dir))
            for p in self.ts_files
        ]

        if num_threads == 1:
            for item in tqdm(work_items, desc="Processing contigs"):
                _infer_and_save_contig(item)
        else:
            from pathos.multiprocessing import ProcessingPool

            pool = ProcessingPool(nodes=num_threads)
            try:
                # imap returns a lazy iterator — tqdm shows per-file progress
                # as each worker finishes rather than waiting for all to complete
                for _ in tqdm(
                    pool.imap(_infer_and_save_contig, work_items),
                    total=len(work_items),
                    desc=f"Processing contigs ({num_threads} threads)",
                ):
                    pass
            finally:
                pool.close()
                pool.join()
                pool.clear()


def _infer_and_save_contig(args: tuple) -> str:
    """
    Module-level worker: load one .trees file, run GNN inference, and
    write the result as a parquet file.  Returns the output path as a
    string so the caller can log progress.

    Must be module-level (not nested) so pathos/dill can pickle it.

    Parameters
    ----------
    args : tuple
        (ts_path, admixed_sample_ids, ancestry_samples, min_segment_size, output_dir)

    Returns
    -------
    str
        Path to the written parquet file.
    """
    ts_path, admixed_sample_ids, ancestry_samples, min_segment_size, output_dir = args
    ts_path    = Path(ts_path)
    output_dir = Path(output_dir)
    contig_id  = ts_path.stem   # filename without .trees suffix

    # Build a single-file inference instance so infer_single's logic
    # (including the optional min_segment_size collapsing) is reused as-is.
    inference = LocalAncestryInference(ts_path.parent, admixed_sample_ids, ancestry_samples)
    df = inference.infer_single(ts_path, min_segment_size=min_segment_size)

    out_path = output_dir / f"{contig_id}.parquet"
    df.to_parquet(out_path, index=False)
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth local ancestry  (census-node based)
# ─────────────────────────────────────────────────────────────────────────────

class TrueLocalAncestry:
    """
    Derive ground-truth local ancestry by reading the census nodes that
    ``msprime.Demography.add_census(time=admix_time + 1)`` bakes into the
    simulation.

    The census records one node per lineage at exactly ``admix_time + 1``,
    before any admixture mixing has occurred.  Each census node therefore
    carries the population of its pre-admixture source lineage.  Walking
    downward from a census node with ``tree.samples()`` identifies every
    admixed sample that inherited that genomic segment from that source
    population.

    This approach is preferable to the post-hoc ``split_edges`` strategy
    because the census nodes are produced by the simulator with correct
    population assignments from the outset — no edges are modified after
    simulation.

    Parameters
    ----------
    ts_dir : str | Path
        Directory containing one or more ``.trees`` files.  Each must have
        been simulated with ``demography.add_census(admix_time + 1)`` so
        that census nodes are present at that time.
    admixed_sample_ids : list[int]
        Node IDs of the admixed haplotypes — must be consistent across all
        tree sequences in the directory.
    admix_time : float
        The admixture time used in the demography.  Census nodes are
        expected at exactly ``admix_time + 1``.
    """

    def __init__(
        self,
        ts_dir: str | Path,
        admixed_sample_ids: list[int],
        admix_time: float,
    ) -> None:
        self.ts_dir = Path(ts_dir)
        if not self.ts_dir.is_dir():
            raise FileNotFoundError(f"Directory not found: {self.ts_dir}")
        if not admixed_sample_ids:
            raise ValueError("`admixed_sample_ids` must not be empty.")

        self.admixed_sample_ids = admixed_sample_ids
        self.admix_time = admix_time

        self.ts_files = sorted(self.ts_dir.glob("*.trees"))
        if not self.ts_files:
            raise FileNotFoundError(f"No .trees files found in: {self.ts_dir}")

    def _find_census_nodes(self, ts: tskit.TreeSequence) -> np.ndarray:
        """
        Return all node IDs whose time equals ``admix_time + 1``.

        Uses the vectorised ``ts.nodes_time`` array rather than a Python
        loop over individual node objects.
        """
        census_time = self.admix_time + 1
        return np.where(ts.nodes_time == census_time)[0]

    def infer_single(self, ts_path: str | Path) -> pd.DataFrame:
        """
        Recover ground-truth ancestry for a single tree sequence.

        Uses ``tskit.TableCollection.link_ancestors`` to map each admixed
        sample's genome directly onto its census-node ancestors — a
        vectorised C-level operation, avoiding a per-tree Python loop over
        ``tree.samples(cp_node)``.

        Parameters
        ----------
        ts_path : str | Path
            Path to a ``.trees`` file.

        Returns
        -------
        pd.DataFrame
            Columns: sample_id, start_pos, end_pos, ancestry_label
            ``ancestry_label`` is the integer population ID of the source
            lineage.  Use ``ts.population(id).metadata`` to map to names.
        """
        ts = tskit.load(str(ts_path))
        census_nodes = self._find_census_nodes(ts)
        if len(census_nodes) == 0:
            raise ValueError(
                f"No census nodes found at time {self.admix_time + 1} in {ts_path}. "
                "Ensure the demography was built with "
                f"demography.add_census(time={self.admix_time + 1}) before simulation."
            )

        tables = ts.dump_tables()
        ancestry = tables.link_ancestors(
            samples=self.admixed_sample_ids, ancestors=census_nodes
        )

        # Map each census node -> population ID using the vectorised
        # nodes_population array — avoids repeated ts.node(u) Python calls.
        census_node_population = {
            int(u): int(ts.nodes_population[u]) for u in census_nodes
        }

        df = pd.DataFrame({
            "sample_id":      ancestry.child,
            "start_pos":      ancestry.left,
            "end_pos":        ancestry.right,
            "ancestry_label": [census_node_population[p] for p in ancestry.parent],
        })

        if df.empty:
            raise ValueError(
                f"No ancestry segments found for {ts_path}. Check that "
                "`admixed_sample_ids` and census nodes are correct."
            )

        df = df.sort_values(["sample_id", "start_pos"]).reset_index(drop=True)
        return _merge_segments(df, self.admixed_sample_ids)

    def infer_all(self, output_dir: str | Path) -> None:
        """
        Recover ground-truth local ancestry across every ``.trees`` file in
        ``self.ts_dir`` and write one ``.parquet`` file per contig to
        ``output_dir``.

        Each output file is named ``<contig_id>.parquet`` where ``contig_id``
        is the source filename without the ``.trees`` suffix.  Processed
        sequentially — ``link_ancestors`` is already a fast, vectorised
        C-level call, so parallelising across files is unnecessary.

        Parameters
        ----------
        output_dir : str | Path
            Directory where parquet files are written.  Created if it does
            not exist.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for ts_path in tqdm(self.ts_files, desc="Processing contigs"):
            df = self.infer_single(ts_path)
            out_path = output_dir / f"{ts_path.stem}.parquet"
            df.to_parquet(out_path, index=False)

# ─────────────────────────────────────────────────────────────────────────────
# Shared segment utilities
# ─────────────────────────────────────────────────────────────────────────────

def _merge_segments(df: pd.DataFrame, sample_ids: list[int]) -> pd.DataFrame:
    """
    Collapse consecutive rows with identical (sample_id, ancestry_label)
    into a single contiguous segment.  Used by both inference and
    ground-truth painters after their per-tree loops.
    """
    merged: list[dict] = []

    for sample_id in sample_ids:
        sample_df = df[df["sample_id"] == sample_id].reset_index(drop=True)
        if sample_df.empty:
            continue

        seg_start = sample_df.loc[0, "start_pos"]
        seg_label = sample_df.loc[0, "ancestry_label"]

        for i in range(1, len(sample_df)):
            current_label = sample_df.loc[i, "ancestry_label"]
            if current_label != seg_label:
                merged.append({
                    "sample_id":      sample_id,
                    "start_pos":      seg_start,
                    "end_pos":        sample_df.loc[i - 1, "end_pos"],
                    "ancestry_label": seg_label,
                })
                seg_start = sample_df.loc[i, "start_pos"]
                seg_label = current_label

        merged.append({
            "sample_id":      sample_id,
            "start_pos":      seg_start,
            "end_pos":        sample_df.loc[len(sample_df) - 1, "end_pos"],
            "ancestry_label": seg_label,
        })

    return pd.DataFrame(
        merged, columns=["sample_id", "start_pos", "end_pos", "ancestry_label"]
    )


def _merge_segments_with_probs(
    df: pd.DataFrame,
    sample_ids: list[int],
    prob_cols: list[str],
) -> pd.DataFrame:
    """
    Collapse consecutive rows with identical (sample_id, ancestry_label)
    into a single contiguous segment, computing span-weighted averages of
    the GNN probability columns across the merged trees.

    Span-weighting ensures that a wide tree contributes proportionally
    more to the reported probability than a narrow one.

    Used exclusively by LocalAncestryInference after its per-tree loop.

    Parameters
    ----------
    df : pd.DataFrame
        Per-tree rows with columns:
        sample_id, start_pos, end_pos, ancestry_label, <prob_cols...>
    sample_ids : list[int]
    prob_cols : list[str]
        Names of the probability columns to average (e.g. ["prob_A", "prob_B"]).
    """
    fixed_cols = ["sample_id", "start_pos", "end_pos", "ancestry_label"]
    merged: list[dict] = []

    for sample_id in sample_ids:
        sample_df = df[df["sample_id"] == sample_id].reset_index(drop=True)
        if sample_df.empty:
            continue

        # Track the current open segment as a list of constituent tree row
        # indices so we can compute the span-weighted average at flush time.
        seg_rows: list[int] = [0]
        seg_label = sample_df.loc[0, "ancestry_label"]

        def _flush(row_indices: list[int], end_pos: float) -> dict:
            sub = sample_df.loc[row_indices]
            spans = sub["end_pos"].to_numpy() - sub["start_pos"].to_numpy()
            total_span = spans.sum()
            row: dict = {
                "sample_id":      sample_id,
                "start_pos":      sample_df.loc[row_indices[0], "start_pos"],
                "end_pos":        end_pos,
                "ancestry_label": seg_label,
            }
            for col in prob_cols:
                # Span-weighted mean: Σ(span_i * prob_i) / Σ(span_i)
                row[col] = float(np.dot(spans, sub[col].to_numpy()) / total_span)
            return row

        for i in range(1, len(sample_df)):
            current_label = sample_df.loc[i, "ancestry_label"]
            if current_label != seg_label:
                merged.append(_flush(seg_rows, sample_df.loc[i - 1, "end_pos"]))
                seg_rows  = [i]
                seg_label = current_label
            else:
                seg_rows.append(i)

        merged.append(_flush(seg_rows, sample_df.loc[len(sample_df) - 1, "end_pos"]))

    return pd.DataFrame(merged, columns=fixed_cols + prob_cols)


def _collapse_short_segments(
    df: pd.DataFrame,
    sample_ids: list[int],
    prob_cols: list[str],
    min_segment_size: float,
) -> pd.DataFrame:
    """
    Relabel segments shorter than ``min_segment_size`` to match a neighbour.

    Only ``ancestry_label`` is overwritten — segment boundaries
    (``start_pos``/``end_pos``) and the GNN probability columns are left
    untouched, so the reported probabilities always reflect the tree(s)
    that were actually observed in that interval, even after relabeling.
    Consequently rows are NOT re-merged: two adjacent rows that end up
    sharing a label after relabeling remain separate rows with their own
    original probabilities, rather than being combined into one wider
    segment with an averaged probability.

    For each sample's segment track (processed independently), a single
    left-to-right pass relabels any segment with span < min_segment_size to
    the label of its left neighbour; the first segment (if too short) is
    instead relabeled to its right neighbour, since it has no left
    neighbour to inherit from. Relabeling does not cascade back over
    already-visited segments — each segment inherits whatever its left
    neighbour's label currently is at the time it is visited.

    Parameters
    ----------
    df : pd.DataFrame
        Merged-segment rows with columns:
        sample_id, start_pos, end_pos, ancestry_label, <prob_cols...>
    sample_ids : list[int]
    prob_cols : list[str]
    min_segment_size : float
        Minimum acceptable segment span (in base pairs).

    Returns
    -------
    pd.DataFrame
        Same rows and probability values as the input, with
        ``ancestry_label`` updated in place for short segments.
    """
    fixed_cols = ["sample_id", "start_pos", "end_pos", "ancestry_label"]
    relabeled: list[dict] = []

    for sample_id in sample_ids:
        sample_df = df[df["sample_id"] == sample_id].reset_index(drop=True)
        if sample_df.empty:
            continue

        labels = sample_df["ancestry_label"].tolist()
        spans  = (sample_df["end_pos"] - sample_df["start_pos"]).to_numpy()

        for i in range(len(sample_df)):
            if spans[i] >= min_segment_size or len(sample_df) == 1:
                continue
            if i > 0:
                labels[i] = labels[i - 1]
            else:
                # First (and, since len > 1, not last) segment with no left
                # neighbour: borrow the label to its right instead.
                labels[i] = labels[i + 1]

        for i in range(len(sample_df)):
            row = sample_df.loc[i].to_dict()
            row["ancestry_label"] = labels[i]
            relabeled.append(row)

    return pd.DataFrame(relabeled, columns=fixed_cols + prob_cols)


# ─────────────────────────────────────────────────────────────────────────────
# Ancestry painting comparison
# ─────────────────────────────────────────────────────────────────────────────

class AncestryComparison(NamedTuple):
    """Return type of :func:`compare_ancestry_paintings`."""
    summary: pd.DataFrame
    """Per-sample accuracy statistics."""
    misassigned_segments: pd.DataFrame
    """Every genomic interval where inferred and true labels differ."""


class AncestryPaintingComparator:
    """
    Compare an inferred local ancestry painting against ground truth.

    Both DataFrames use the schema:
        sample_id, start_pos, end_pos, ancestry_label

    Because the two paintings will generally have different segment
    breakpoints, comparison is done at the base-pair level via a unified
    breakpoint sweep:

    1. Collect all start/end positions from both DataFrames into a sorted
       breakpoint grid.
    2. For each sub-interval in the grid, look up the label in each
       painting with a binary search (O(log n)).
    3. Classify the interval as correct or mis-assigned and accumulate
       statistics.
    4. Merge adjacent mis-assigned sub-intervals that share the same
       (inferred, true) label pair back into contiguous segments.

    Parameters
    ----------
    inferred : pd.DataFrame
        Output of :meth:`LocalAncestryInference.infer_single`.
    truth : pd.DataFrame
        Output of :meth:`TrueLocalAncestry.infer_single`.
        Labels must use the same vocabulary as ``inferred``; map integer
        population IDs to names beforehand if needed.
    """

    def __init__(self, inferred: pd.DataFrame, truth: pd.DataFrame) -> None:
        self._validate(inferred, "inferred")
        self._validate(truth, "truth")
        self.inferred = inferred
        self.truth = truth

    # ── public ────────────────────────────────────────────────────────────

    def compare(self) -> AncestryComparison:
        """Run the comparison and return summary + mis-assigned segments."""
        sample_ids = sorted(
            set(self.inferred["sample_id"]) | set(self.truth["sample_id"])
        )
        summary_rows: list[dict] = []
        mis_rows: list[dict] = []

        for sample_id in sample_ids:
            inf_s = self.inferred[self.inferred["sample_id"] == sample_id].reset_index(drop=True)
            tru_s = self.truth[self.truth["sample_id"] == sample_id].reset_index(drop=True)

            if inf_s.empty or tru_s.empty:
                continue

            sample_summary, sample_mis = self._compare_sample(
                sample_id, inf_s, tru_s
            )
            summary_rows.append(sample_summary)
            mis_rows.extend(sample_mis)

        summary_df = pd.DataFrame(
            summary_rows,
            columns=[
                "sample_id", "total_bp", "correct_bp", "misassigned_bp",
                "accuracy", "n_correct_segments", "n_mis_segments",
            ],
        )
        mis_df = self._merge_mis_segments(
            pd.DataFrame(
                mis_rows,
                columns=[
                    "sample_id", "start_pos", "end_pos", "span",
                    "inferred_label", "true_label",
                ],
            ) if mis_rows else pd.DataFrame(
                columns=[
                    "sample_id", "start_pos", "end_pos", "span",
                    "inferred_label", "true_label",
                ]
            )
        )
        return AncestryComparison(summary=summary_df, misassigned_segments=mis_df)

    # ── private ───────────────────────────────────────────────────────────

    @staticmethod
    def _validate(df: pd.DataFrame, name: str) -> None:
        required = {"sample_id", "start_pos", "end_pos", "ancestry_label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"`{name}` is missing columns: {missing}")

    @staticmethod
    def _sorted_breakpoints(a: pd.DataFrame, b: pd.DataFrame) -> list[float]:
        """Sorted union of all segment boundaries from two DataFrames."""
        pts: set[float] = set()
        for df in (a, b):
            pts.update(df["start_pos"].tolist())
            pts.update(df["end_pos"].tolist())
        return sorted(pts)

    @staticmethod
    def _label_at(df: pd.DataFrame, starts: np.ndarray, pos: float) -> Any:
        """
        O(log n) lookup of the ancestry label at genomic position ``pos``.
        Returns ``None`` when ``pos`` falls outside all segments.
        """
        idx = int(np.searchsorted(starts, pos, side="right")) - 1
        if idx < 0 or idx >= len(df):
            return None
        row = df.iloc[idx]
        if pos < row["start_pos"] or pos >= row["end_pos"]:
            return None
        return row["ancestry_label"]

    def _compare_sample(
        self,
        sample_id: int,
        inf_s: pd.DataFrame,
        tru_s: pd.DataFrame,
    ) -> tuple[dict, list[dict]]:
        """Return (summary_row_dict, list_of_mis_row_dicts) for one sample."""
        breakpoints = self._sorted_breakpoints(inf_s, tru_s)
        inf_starts  = inf_s["start_pos"].to_numpy()
        tru_starts  = tru_s["start_pos"].to_numpy()

        correct_bp     = 0.0
        misassigned_bp = 0.0
        n_correct      = 0
        n_mis          = 0
        mis_rows: list[dict] = []

        for left, right in zip(breakpoints[:-1], breakpoints[1:]):
            span      = right - left
            mid       = (left + right) / 2
            inf_label = self._label_at(inf_s, inf_starts, mid)
            tru_label = self._label_at(tru_s, tru_starts, mid)

            if inf_label is None or tru_label is None:
                continue

            if inf_label == tru_label:
                correct_bp += span
                n_correct  += 1
            else:
                misassigned_bp += span
                n_mis          += 1
                mis_rows.append({
                    "sample_id":      sample_id,
                    "start_pos":      left,
                    "end_pos":        right,
                    "span":           span,
                    "inferred_label": inf_label,
                    "true_label":     tru_label,
                })

        total_bp = correct_bp + misassigned_bp
        summary  = {
            "sample_id":           sample_id,
            "total_bp":            total_bp,
            "correct_bp":          correct_bp,
            "misassigned_bp":      misassigned_bp,
            "accuracy":            correct_bp / total_bp if total_bp > 0 else float("nan"),
            "n_correct_segments":  n_correct,
            "n_mis_segments":      n_mis,
        }
        return summary, mis_rows

    @staticmethod
    def _merge_mis_segments(df: pd.DataFrame) -> pd.DataFrame:
        """
        Merge adjacent mis-assigned sub-intervals that share the same
        (sample_id, inferred_label, true_label) back into one segment.
        These were only split by the breakpoint grid, not by a label change.
        """
        if df.empty:
            return df

        df = df.sort_values(["sample_id", "start_pos"]).reset_index(drop=True)
        merged: list[dict] = []
        seg = df.iloc[0].to_dict()

        for i in range(1, len(df)):
            row = df.iloc[i]
            if (
                row["sample_id"] == seg["sample_id"]
                and abs(row["start_pos"] - seg["end_pos"]) < 1e-9
                and row["inferred_label"] == seg["inferred_label"]
                and row["true_label"] == seg["true_label"]
            ):
                seg["end_pos"] = row["end_pos"]
                seg["span"]    = seg["end_pos"] - seg["start_pos"]
            else:
                merged.append(seg)
                seg = row.to_dict()

        merged.append(seg)
        return pd.DataFrame(
            merged,
            columns=[
                "sample_id", "start_pos", "end_pos", "span",
                "inferred_label", "true_label",
            ],
        ).reset_index(drop=True)


def compare_ancestry_paintings(
    inferred: pd.DataFrame,
    truth: pd.DataFrame,
) -> AncestryComparison:
    """Functional wrapper around :class:`AncestryPaintingComparator`."""
    return AncestryPaintingComparator(inferred, truth).compare()


# ─────────────────────────────────────────────────────────────────────────────
# Chromosome painting visualisation
# ─────────────────────────────────────────────────────────────────────────────

def plot_chromosome_painting(
    sample_id: int,
    inferred: pd.DataFrame,
    truth: pd.DataFrame,
    n_bins: int = 2000,
    figsize: tuple[int, int] = (16, 4),
) -> "matplotlib.figure.Figure":
    """
    Visualise inferred vs ground-truth local ancestry for a single sample
    using three stacked axes that share the same genomic x-axis.

    Top axis — Ground truth
        A single-row seaborn heatmap.  Each bin is painted with a solid
        colour representing its ancestry label (probability = 1).

    Middle axis — Inferred ancestry segments
        A single-row seaborn heatmap in the same style as the ground-truth
        axis, painted with the winning (argmax) inferred ``ancestry_label``
        for each bin.  This reflects the discrete segments actually
        assigned by :meth:`LocalAncestryInference.infer_single` (including
        any ``min_segment_size`` relabeling), independent of the GNN
        probabilities shown below.

    Bottom axis — Inferred (probability-aware)
        A stacked bar chart drawn with ``axvspan`` patches.  At each bin
        the vertical space [0, 1] is divided proportionally among ancestry
        labels according to their GNN probabilities, so a bin that is 70 %
        label A and 30 % label B will show 70 % of its height in colour A
        and the remaining 30 % in colour B.  Bins with missing data are
        filled with light grey.

    Mis-assigned bins (where the winning inferred label differs from the
    true label) are highlighted with a red border drawn on the ground-truth
    axis only.

    Parameters
    ----------
    sample_id : int
        The sample to visualise.  Must be present in both DataFrames.
    inferred : pd.DataFrame
        Output of :meth:`LocalAncestryInference.infer_single`.
        Must contain ``prob_<label>`` columns in addition to the standard
        ``sample_id, start_pos, end_pos, ancestry_label`` columns.
    truth : pd.DataFrame
        Output of :meth:`TrueLocalAncestry.infer_single`.
        Columns: sample_id, start_pos, end_pos, ancestry_label
    n_bins : int
        Number of equal-width genomic bins.  Default 2000.
    figsize : tuple[int, int]
        Matplotlib figure size.  Default (16, 4).

    Returns
    -------
    matplotlib.figure.Figure

    Example
    -------
    >>> fig = plot_chromosome_painting(42, admix_df, true_admix_df)
    >>> fig.savefig("sample_42.png", dpi=150, bbox_inches="tight")
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors
    import seaborn as sns

    # ── filter to this sample ─────────────────────────────────────────────
    inf_s = inferred[inferred["sample_id"] == sample_id].reset_index(drop=True)
    tru_s = truth[truth["sample_id"] == sample_id].reset_index(drop=True)

    if inf_s.empty or tru_s.empty:
        raise ValueError(f"sample_id {sample_id} not found in both DataFrames.")

    # Detect probability columns from the inferred DataFrame
    prob_cols = [c for c in inf_s.columns if c.startswith("prob_")]
    if not prob_cols:
        raise ValueError(
            "No probability columns (prob_<label>) found in `inferred`. "
            "Ensure LocalAncestryInference.infer_single() was called with the current version."
        )
    # Derive label names from the prob column names (strip "prob_" prefix)
    labels = [c[len("prob_"):] for c in prob_cols]

    seq_start = min(inf_s["start_pos"].min(), tru_s["start_pos"].min())
    seq_end   = max(inf_s["end_pos"].max(),   tru_s["end_pos"].max())

    # ── palette ───────────────────────────────────────────────────────────
    palette      = sns.color_palette("tab10", n_colors=len(labels))
    colour       = {lbl: palette[i] for i, lbl in enumerate(labels)}
    MISSING_CLR  = (0.88, 0.88, 0.88)
    MISSING_INT  = len(labels)   # sentinel for heatmap integer grid

    # ── bin grid ──────────────────────────────────────────────────────────
    bin_edges = np.linspace(seq_start, seq_end, n_bins + 1)
    bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_width = bin_edges[1] - bin_edges[0]

    # Precompute searchsorted index arrays for fast segment lookup
    inf_starts = inf_s["start_pos"].to_numpy()
    tru_starts = tru_s["start_pos"].to_numpy()

    def _seg_index(starts: np.ndarray, df: pd.DataFrame, mid: float) -> int | None:
        """Return the row index in df whose segment contains mid, or None."""
        idx = int(np.searchsorted(starts, mid, side="right")) - 1
        if idx < 0 or idx >= len(df):
            return None
        row = df.iloc[idx]
        if mid < row["start_pos"] or mid >= row["end_pos"]:
            return None
        return idx

    # ── rasterise ground truth → integer label grid ───────────────────────
    # truth ancestry_label values are integer population IDs; labels derived
    # from prob_ columns are strings.  Cast truth labels to str so both sides
    # of every lookup use the same type.
    label_to_int = {lbl: i for i, lbl in enumerate(labels)}
    tru_grid = np.full(n_bins, MISSING_INT, dtype=float)
    for b, mid in enumerate(bin_mids):
        idx = _seg_index(tru_starts, tru_s, mid)
        if idx is None:
            continue
        lbl = str(tru_s.iloc[idx]["ancestry_label"])
        if lbl in label_to_int:
            tru_grid[b] = label_to_int[lbl]

    # ── rasterise inferred → per-label probability grids ─────────────────
    # prob_grids[i] is a (n_bins,) array of probabilities for labels[i].
    # NaN entries mark missing-data bins.
    prob_grids = np.full((len(labels), n_bins), np.nan)
    inf_label_grid = np.full(n_bins, MISSING_INT, dtype=float)

    for b, mid in enumerate(bin_mids):
        idx = _seg_index(inf_starts, inf_s, mid)
        if idx is None:
            continue
        row = inf_s.iloc[idx]
        lbl = str(row["ancestry_label"])          # cast to str, same as truth
        if lbl != "None" and lbl in label_to_int:
            inf_label_grid[b] = label_to_int[lbl]
        for li, pc in enumerate(prob_cols):
            prob_grids[li, b] = row[pc]

    # ── mis-assignment mask ───────────────────────────────────────────────
    both_present  = (inf_label_grid != MISSING_INT) & (tru_grid != MISSING_INT)
    mismatch_mask = both_present & (inf_label_grid != tru_grid)

    # Merge contiguous mismatch runs into (start, end) genomic ranges
    mis_ranges: list[tuple[float, float]] = []
    in_run = False
    for b in range(n_bins):
        if mismatch_mask[b] and not in_run:
            run_start = bin_edges[b]
            in_run    = True
        elif not mismatch_mask[b] and in_run:
            mis_ranges.append((run_start, bin_edges[b]))
            in_run = False
    if in_run:
        mis_ranges.append((run_start, bin_edges[-1]))

    # ── build figure with three vertically stacked axes ────────────────────
    fig, (ax_true, ax_seg, ax_inf) = plt.subplots(
        3, 1,
        figsize=figsize,
        sharex=False,           # x-ticks are set manually on each
        gridspec_kw={"hspace": 0.08, "height_ratios": [1, 1, 2]},
    )

    # ── shared discrete colormap for the two solid-label heatmaps ─────────
    cmap_colors = [colour[lbl] for lbl in labels] + [MISSING_CLR]
    cmap = mcolors.ListedColormap(cmap_colors)
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(len(cmap_colors) + 1) - 0.5,
        ncolors=len(cmap_colors),
    )

    # ── top axis: ground truth heatmap ────────────────────────────────────
    sns.heatmap(
        tru_grid.reshape(1, -1),
        ax=ax_true,
        cmap=cmap,
        norm=norm,
        cbar=False,
        xticklabels=False,
        yticklabels=["True"],
        linewidths=0,
        rasterized=True,
    )
    ax_true.tick_params(axis="y", labelsize=9, rotation=0)
    ax_true.set_ylabel("")

    # ── middle axis: inferred ancestry segments heatmap ────────────────────
    # Winning (argmax) label per bin, same rendering as the truth axis, so
    # the discrete segment calls can be compared directly by eye before
    # looking at the probability-aware view below.
    sns.heatmap(
        inf_label_grid.reshape(1, -1),
        ax=ax_seg,
        cmap=cmap,
        norm=norm,
        cbar=False,
        xticklabels=False,
        yticklabels=["Inferred"],
        linewidths=0,
        rasterized=True,
    )
    ax_seg.tick_params(axis="y", labelsize=9, rotation=0)
    ax_seg.set_ylabel("")

    # ── bottom axis: inferred stacked-probability bars ────────────────────
    ax_inf.set_xlim(seq_start, seq_end)
    ax_inf.set_ylim(0, 1)

    for b in range(n_bins):
        x0 = bin_edges[b]
        x1 = bin_edges[b + 1]

        # Missing data bin — fill entirely with grey
        if np.all(np.isnan(prob_grids[:, b])):
            ax_inf.axvspan(x0, x1, ymin=0, ymax=1,
                           color=MISSING_CLR, linewidth=0)
            continue

        # Stack each label's probability band from bottom to top
        bottom = 0.0
        for li, lbl in enumerate(labels):
            prob = prob_grids[li, b]
            if np.isnan(prob) or prob <= 0:
                continue
            ax_inf.axvspan(
                x0, x1,
                ymin=bottom,
                ymax=bottom + prob,
                color=colour[lbl],
                linewidth=0,
            )
            bottom += prob

    ax_inf.set_yticks([0, 0.5, 1])
    ax_inf.set_yticklabels(["0", "0.5", "1"], fontsize=8)
    ax_inf.set_ylabel("GNN prob.", fontsize=8)
    ax_inf.spines[["top", "right"]].set_visible(False)

    # ── coordinate converter: genomic position → heatmap bin index ────────
    # ax_true and ax_seg are seaborn heatmaps whose x-axis spans [0, n_bins].
    # ax_inf uses real genomic coordinates [seq_start, seq_end].
    # All overlays and ticks must be expressed in each axis's own space.
    def _to_bin(pos: float) -> float:
        return (pos - seq_start) / (seq_end - seq_start) * n_bins

    # ── mis-assignment overlay (ground-truth axis only) ────────────────────
    # Use a blended transform: x in data (bin-index) coordinates,
    # y in axes-fraction coordinates (0=bottom, 1=top of the axis).
    # This makes the rectangle span the full row height regardless of
    # whatever ylim seaborn sets internally.
    from matplotlib.transforms import blended_transform_factory

    blend = blended_transform_factory(ax_true.transData, ax_true.transAxes)
    for p_start, p_end in mis_ranges:
        ax_true.add_patch(mpatches.Rectangle(
            (_to_bin(p_start), 0),
            _to_bin(p_end) - _to_bin(p_start), 1,
            linewidth=1.5, edgecolor="red", facecolor="none",
            zorder=5, clip_on=False,
            transform=blend,
        ))

    # ── x-axis ticks ──────────────────────────────────────────────────────
    mb_ticks = np.arange(
        np.floor(seq_start / 2e5) * 2e5,
        seq_end + 2e5,
        2e5,                    # every 0.2 Mb
    )
    mb_labels = [f"{t / 1e6:.1f}" for t in mb_ticks]

    # ax_true / ax_seg ticks in bin-index space (hidden labels — top axes)
    ax_true.set_xticks([_to_bin(t) for t in mb_ticks])
    ax_true.set_xticklabels([])
    ax_seg.set_xticks([_to_bin(t) for t in mb_ticks])
    ax_seg.set_xticklabels([])

    # ax_inf ticks in genomic space (shown labels — bottom axis)
    ax_inf.set_xticks(mb_ticks)
    ax_inf.set_xticklabels(mb_labels, fontsize=8)
    ax_inf.set_xlabel("Genomic position (Mb)", fontsize=9)

    # ── legend (right of figure) ──────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=colour[lbl], label=str(lbl))
        for lbl in labels
    ]
    legend_handles.append(
        mpatches.Patch(facecolor=MISSING_CLR, label="Missing")
    )
    legend_handles.append(
        mpatches.Patch(facecolor="none", edgecolor="red",
                       linewidth=1.2, label="Mis-assigned")
    )
    fig.legend(
        handles=legend_handles,
        loc="center right",
        bbox_to_anchor=(1.08, 0.5),
        frameon=True,
        fontsize=8,
        title="Ancestry",
        title_fontsize=8,
    )

    fig.suptitle(f"Local ancestry painting — sample {sample_id}",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    return fig

import msprime
def simulate_admix_ts(n=10, admix_time=100, sequence_length=5e6, recomb_rate=1e-8, Ne = 1e4, split_time=2000, random_seed=42):
    demography = msprime.Demography()
    demography.add_population(name="Ancestral", initial_size=Ne)
    demography.add_population(name="A", initial_size=Ne)
    demography.add_population(name="B", initial_size=Ne)
    demography.add_population(name="C", initial_size=Ne)
    demography.add_population_split(time=split_time, derived=["A", "B"], ancestral="Ancestral")
    demography.add_admixture(time=admix_time, derived="C", ancestral=["A", "B"], proportions=[0.5, 0.5])
    demography.add_census(time=admix_time + 1)
    demography.sort_events()
    ts = msprime.sim_ancestry(samples={"A": n, "B": n, "C": n}, demography=demography, sequence_length=sequence_length, recombination_rate=recomb_rate, random_seed=random_seed)
    return ts
# #%%
# trees_dir = "./"
# n = 10
# ancestry_samples = {
#     1.0: list(range(0, 2 * n)),
#     2.0: list(range(2 * n, 4 * n)),
# }
# admixed_sample_ids = list(range(4 * n, 6 * n))

# #%%
# min_segment_size = 1e4
# ts = simulate_admix_ts(random_seed=42)
# ts.dump("admix.trees")
# true_ancestry = TrueLocalAncestry(ts_dir=trees_dir, admixed_sample_ids=admixed_sample_ids, admix_time=100)
# true_ancestry_df = true_ancestry.infer_single("admix.trees")
# GNN_ancestry = LocalAncestryInference(ts_dir=trees_dir, admixed_sample_ids=admixed_sample_ids, ancestry_samples=ancestry_samples)
# GNN_ancestry_df = GNN_ancestry.infer_single("admix.trees", min_segment_size=min_segment_size)

# comparator = AncestryPaintingComparator(inferred=GNN_ancestry_df, truth=true_ancestry_df)
# summary, misinferred = comparator.compare()
# #%%
# fig = plot_chromosome_painting(sample_id=4*n, inferred=GNN_ancestry_df, truth=true_ancestry_df)
# # %%
# misinferred[misinferred["sample_id"] == 4*n]
# # %%
