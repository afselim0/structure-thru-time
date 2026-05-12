# %%
from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np
import pandas as pd
import tskit
from tqdm import tqdm

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
    ts : tskit.TreeSequence
    admixed_sample_ids : list[int]
        Node IDs of the admixed (query) haplotypes.
    ancestry_samples : dict[Any, list[int]]
        Mapping from ancestry label → list of reference sample node IDs.
        Must contain at least 2 groups; admixed IDs must not appear here.
    """

    def __init__(
        self,
        ts: tskit.TreeSequence,
        admixed_sample_ids: list[int],
        ancestry_samples: dict[Any, list[int]],
    ) -> None:
        self.ts = ts
        self.admixed_sample_ids = admixed_sample_ids
        self.ancestry_samples = ancestry_samples
        self._validate()

    def _validate(self) -> None:
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

    def infer(self) -> pd.DataFrame:
        """
        Run inference and return a merged-segment DataFrame.

        For each local tree the raw GNN proportion vector is recorded for
        every focal sample, giving the probability of belonging to each
        reference ancestry group.  The winning label (``ancestry_label``) is
        still the group with the highest proportion.  When consecutive trees
        share the same winning label they are merged into one segment; the
        reported probabilities are the span-weighted averages of the
        per-tree GNN values across all merged trees.

        Returns
        -------
        pd.DataFrame
            Columns:
              sample_id, start_pos, end_pos, ancestry_label,
              prob_<label> for each label in ancestry_samples
        """
        labels    = list(self.ancestry_samples.keys())
        prob_cols = [f"prob_{lbl}" for lbl in labels]
        sample_sets = [self.ancestry_samples[lbl] for lbl in labels]
        rows: list[dict] = []

        for tree in tqdm(self.ts.trees(), desc="Inferring local ancestry",
                         total=self.ts.num_trees):
            start, end = tree.interval.left, tree.interval.right

            # Isolate this single tree so that genealogical_nearest_neighbours
            # returns local (not genome-wide averaged) GNN proportions.
            single_tree_ts = self.ts.keep_intervals(
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
        return _merge_segments_with_probs(df, self.admixed_sample_ids, prob_cols)


def infer_local_ancestry(
    ts: tskit.TreeSequence,
    admixed_sample_ids: list[int],
    ancestry_samples: dict[Any, list[int]],
) -> pd.DataFrame:
    """Functional wrapper around :class:`LocalAncestryInference`."""
    return LocalAncestryInference(ts, admixed_sample_ids, ancestry_samples).infer()


# ─────────────────────────────────────────────────────────────────────────────
# Ground-truth local ancestry  (split_edges-based)
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
    ts : tskit.TreeSequence
        Must have been simulated with ``demography.add_census(admix_time + 1)``
        so that census nodes are present at that time.
    admixed_sample_ids : list[int]
        Node IDs of the admixed haplotypes.
    admix_time : float
        The admixture time used in the demography.  Census nodes are
        expected at exactly ``admix_time + 1``.
    """

    def __init__(
        self,
        ts: tskit.TreeSequence,
        admixed_sample_ids: list[int],
        admix_time: float,
    ) -> None:
        if not admixed_sample_ids:
            raise ValueError("`admixed_sample_ids` must not be empty.")
        self.ts = ts
        self.admixed_sample_ids = admixed_sample_ids
        self.admix_time = admix_time
        self._census_nodes = self._find_census_nodes()
        if not self._census_nodes:
            raise ValueError(
                f"No census nodes found at time {admix_time + 1}. "
                "Ensure the demography was built with "
                f"demography.add_census(time={admix_time + 1}) before simulation."
            )

    def _find_census_nodes(self) -> list[int]:
        """
        Return all node IDs whose time equals ``admix_time + 1``.

        Uses the vectorised ``ts.nodes_time`` array rather than a Python
        loop over individual node objects.
        """
        census_time = self.admix_time + 1
        return list(np.where(self.ts.nodes_time == census_time)[0])

    def infer(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            Columns: sample_id, start_pos, end_pos, ancestry_label
            ``ancestry_label`` is the integer population ID of the source
            lineage.  Use ``ts.population(id).metadata`` to map to names.
        """
        admixed_set = set(self.admixed_sample_ids)

        # Pre-fetch population for every census node using the vectorised
        # nodes_population array — avoids repeated ts.node(u) Python calls.
        census_populations: dict[int, int] = {
            u: int(self.ts.nodes_population[u])
            for u in self._census_nodes
        }

        rows: list[dict] = []

        for tree in tqdm(self.ts.trees(sample_lists=True),
                         desc="Reading ground-truth ancestry",
                         total=self.ts.num_trees):
            start, end = tree.interval.left, tree.interval.right
            sample_ancestry: dict[int, Any] = {s: None for s in self.admixed_sample_ids}

            for cp_node, cp_population in census_populations.items():
                # tree.samples(u) iterates sample descendants of u in this
                # tree — O(descendants), enabled by sample_lists=True.
                for descendant in tree.samples(cp_node):
                    if descendant in admixed_set:
                        sample_ancestry[descendant] = cp_population

            for sample_id in self.admixed_sample_ids:
                rows.append({
                    "sample_id":      sample_id,
                    "start_pos":      start,
                    "end_pos":        end,
                    "ancestry_label": sample_ancestry[sample_id],
                })

        df = pd.DataFrame(
            rows, columns=["sample_id", "start_pos", "end_pos", "ancestry_label"]
        )
        return _merge_segments(df, self.admixed_sample_ids)


def true_local_ancestry(
    ts: tskit.TreeSequence,
    admixed_sample_ids: list[int],
    admix_time: float,
) -> pd.DataFrame:
    """Functional wrapper around :class:`TrueLocalAncestry`."""
    return TrueLocalAncestry(ts, admixed_sample_ids, admix_time).infer()


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
        Output of :func:`infer_local_ancestry`.
    truth : pd.DataFrame
        Output of :func:`true_local_ancestry`.
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
    using two stacked axes that share the same genomic x-axis.
 
    Top axis — Ground truth
        A single-row seaborn heatmap.  Each bin is painted with a solid
        colour representing its ancestry label (probability = 1).
 
    Bottom axis — Inferred (probability-aware)
        A stacked bar chart drawn with ``axvspan`` patches.  At each bin
        the vertical space [0, 1] is divided proportionally among ancestry
        labels according to their GNN probabilities, so a bin that is 70 %
        label A and 30 % label B will show 70 % of its height in colour A
        and the remaining 30 % in colour B.  Bins with missing data are
        filled with light grey.
 
    Mis-assigned bins (where the winning inferred label differs from the
    true label) are highlighted with a red border drawn across both axes.
 
    Parameters
    ----------
    sample_id : int
        The sample to visualise.  Must be present in both DataFrames.
    inferred : pd.DataFrame
        Output of :func:`infer_local_ancestry`.
        Must contain ``prob_<label>`` columns in addition to the standard
        ``sample_id, start_pos, end_pos, ancestry_label`` columns.
    truth : pd.DataFrame
        Output of :func:`true_local_ancestry`.
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
            "Ensure infer_local_ancestry() was called with the current version."
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
 
    # ── build figure with two vertically stacked axes ─────────────────────
    fig, (ax_true, ax_inf) = plt.subplots(
        2, 1,
        figsize=figsize,
        sharex=False,           # x-ticks are set manually on both
        gridspec_kw={"hspace": 0.08, "height_ratios": [1, 2]},
    )
 
    # ── top axis: ground truth heatmap ────────────────────────────────────
    cmap_colors = [colour[lbl] for lbl in labels] + [MISSING_CLR]
    cmap = mcolors.ListedColormap(cmap_colors)
    norm = mcolors.BoundaryNorm(
        boundaries=np.arange(len(cmap_colors) + 1) - 0.5,
        ncolors=len(cmap_colors),
    )
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
    # ax_true is a seaborn heatmap whose x-axis spans [0, n_bins].
    # ax_inf uses real genomic coordinates [seq_start, seq_end].
    # All overlays and ticks must be expressed in each axis's own space.
    def _to_bin(pos: float) -> float:
        return (pos - seq_start) / (seq_end - seq_start) * n_bins
 
    # ── mis-assignment overlay (true axis only) ───────────────────────────
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
 
    # ax_true ticks in bin-index space (hidden labels — top axis)
    ax_true.set_xticks([_to_bin(t) for t in mb_ticks])
    ax_true.set_xticklabels([])
 
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
