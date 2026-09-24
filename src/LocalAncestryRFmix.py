#%%
"""
RFMix2 Local Ancestry Inference Wrapper
========================================
Runs RFMix2 for local ancestry inference from a tree sequence and returns
results in the same DataFrame format used by LocalAncestryInference:

    columns = [sample_id, start_pos, end_pos, ancestry_label]

ancestry_label is RFMix2's Viterbi/CRF decoding from the .msp.tsv output
(its single most likely ancestry call per window); forward-backward
probabilities (.fb.tsv) are not parsed or retained.

Accepts the same admixed_sample_ids (node IDs as ints) and ancestry_samples
(dict keyed by any type) as the GNN-based LocalAncestryInference. The tree
sequence is simplified down to just the required nodes (admixed +
reference), a VCF is written from it with explicit, deterministic
individual names ("ind_{individual_id}", using the *post-simplification*
individual IDs), and that VCF is fed to RFMix2. All intermediate files
(VCF, genetic map, RFMix2 output) are removed once inference completes,
unless ``keep_temp=True``.

Requirements
------------
* RFMix2 must be installed and available on PATH as ``rfmix``
* bcftools must be available on PATH (for VCF splitting and indexing)
* A genetic map file in the format: chrom  pos_bp  pos_cM

Example
-------
>>> rfmix_df = infer_local_ancestry_single_ts(
...     ts_path="contig_0.trees",
...     admixed_sample_ids=np.arange(40, 60).tolist(),   # node IDs
...     ancestry_samples={
...         1.0: np.arange(0, 20).tolist(),               # node IDs, float key
...         2.0: np.arange(20, 40).tolist(),
...     },
...     recombination_rate=1e-8,
... )
"""

from __future__ import annotations

import msprime
import logging
import os
import shutil
import subprocess
import tempfile
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tskit
from pathos.multiprocessing import ProcessPool
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Genetic map helper (kept here so a single import covers everything)
# ─────────────────────────────────────────────────────────────────────────────

def write_genetic_map(
    sequence_length: float,
    recombination_rate: float,
    chrom: str,
    output_path: str,
) -> None:
    """
    Write a two-point genetic map for RFMix2 assuming a constant
    recombination rate.

    Format: chrom  pos_bp  pos_cM  (tab-delimited, no header).
    RFMix2 interpolates linearly between provided points, so two anchor
    points are sufficient for a uniform rate.

    Parameters
    ----------
    sequence_length : float
        Length of the simulated sequence in base pairs.
    recombination_rate : float
        Per-bp per-generation rate (e.g. 1e-8).
    chrom : str
        Must match the CHROM column in the VCF (default "1").
    output_path : str
    """
    seq_len = int(sequence_length)
    with open(output_path, "w") as fh:
        fh.write(f"{chrom}\t0\t0.000000\n")
        fh.write(f"{chrom}\t{seq_len}\t{seq_len * recombination_rate * 100:.6f}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Tree sequence → VCF helpers
# ─────────────────────────────────────────────────────────────────────────────

def _node_to_vcf_name(ts: tskit.TreeSequence, node_id: int) -> str:
    """
    Convert a tskit node ID to the VCF sample name that will be written by
    ``write_ts_vcf`` for the individual owning that node: ``"ind_{individual_id}"``.

    Raises
    ------
    ValueError
        If the node is not associated with an individual (write_vcf requires
        individuals to combine haplotypes into phased multiploid samples).
    """
    individual_id = ts.node(node_id).individual
    if individual_id == tskit.NULL:
        raise ValueError(
            f"Node {node_id} is not associated with an individual; "
            f"write_vcf() cannot include it as a phased VCF sample."
        )
    return f"ind_{individual_id}"


def _unique_vcf_names(ts: tskit.TreeSequence, node_ids: list[int]) -> list[str]:
    """
    Return the ordered, deduplicated list of VCF individual names for a
    list of node IDs, preserving first-occurrence order.
    """
    seen: set[str] = set()
    result: list[str] = []
    for nid in node_ids:
        name = _node_to_vcf_name(ts, nid)
        if name not in seen:
            seen.add(name)
            result.append(name)
    return result


def _with_diploid_individuals(ts: tskit.TreeSequence) -> tskit.TreeSequence:
    """
    Inferred tree sequences (e.g. SINGER output) carry no individuals table.
    Pair consecutive sample nodes (2k, 2k+1) into diploid individual k, the
    convention of the simulated trees and of pop_labels' individual_id.
    """
    if ts.num_individuals > 0:
        return ts
    samples = ts.samples()
    if len(samples) % 2:
        raise ValueError(f"Cannot pair {len(samples)} sample nodes into diploid individuals.")
    tables = ts.dump_tables()
    individual = tables.nodes.individual
    for k in range(len(samples) // 2):
        tables.individuals.add_row()
        individual[samples[2 * k]] = individual[samples[2 * k + 1]] = k
    tables.nodes.individual = individual
    return tables.tree_sequence()


def write_ts_vcf(ts: tskit.TreeSequence, node_ids: list[int], output_path: str | Path, mutation_rate: float = None) -> None:
    """
    Write a VCF for exactly the individuals owning *node_ids*, using
    deterministic sample names ``"ind_{individual_id}"`` (rather than
    tskit's default positional ``tsk_0, tsk_1, ...`` naming, which depends
    on output order and is easily desynchronised from node IDs after
    simplification).

    Parameters
    ----------
    ts : tskit.TreeSequence
    node_ids : list[int]
        Node IDs whose owning individuals should be included as VCF samples.
    output_path : str | Path
    """
    individual_ids = sorted({ts.node(nid).individual for nid in node_ids})
    if tskit.NULL in individual_ids:
        raise ValueError(
            "One or more of the given node IDs are not associated with an "
            "individual; write_vcf() cannot include them as phased VCF samples."
        )
    individual_names = [f"ind_{iid}" for iid in individual_ids]
    if ts.num_sites == 0:
        if mutation_rate is None:
            raise ValueError(
                "Tree sequence has no sites and `mutation_rate` was not provided; "
                "RFMix2 needs variant sites to call ancestry from."
            )
        ts = msprime.sim_mutations(ts, rate=mutation_rate)
    with open(output_path, "w") as fh:
        ts.write_vcf(fh, individuals=individual_ids, individual_names=individual_names, allow_position_zero=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main class
# ─────────────────────────────────────────────────────────────────────────────

class RFMixInference:
    """
    Wrapper around RFMix2 that translates node-ID / any-key inputs into
    VCF sample names, runs the binary, parses the output, and returns a
    standardised local ancestry DataFrame compatible with the GNN pipeline.

    Parameters
    ----------
    vcf_path : str | Path
        VCF written from *ts* via :func:`write_ts_vcf`, so sample names
        follow the "ind_{individual_id}" convention.
    ts : tskit.TreeSequence
        The tree sequence *vcf_path* was written from (used only to
        translate node IDs to VCF sample names via individual ownership;
        node IDs must refer to nodes in this tree sequence).
    admixed_sample_ids : list[int]
        Node IDs of admixed haplotypes (same format as GNN inference).
    ancestry_samples : dict[Any, list[int]]
        label → list of reference node IDs (same format as GNN inference).
        Keys can be any type (int, float, str); they are stringified for
        use in the sample map and as ancestry_label values.
    recombination_rate : float
        Per-bp per-generation recombination rate (e.g. 1e-8).
        A two-point genetic map is generated from this value and
        *sequence_length*, used during inference, then deleted.
    output_dir : str | Path | None
        Directory for RFMix2 output files.  Temp dir used if None.
    keep_temp : bool
    rfmix_extra_args : list[str]
        Forwarded verbatim to the rfmix command line.
    """

    def __init__(
        self,
        vcf_path: str | Path,
        ts: tskit.TreeSequence,
        admixed_sample_ids: list[int],
        ancestry_samples: dict[Any, list[int]],
        recombination_rate: float,
        sequence_length: float,
        output_dir: str | Path | None = None,
        keep_temp: bool = False,
        rfmix_extra_args: list[str] | None = None,
    ) -> None:
        self.vcf_path           = Path(vcf_path)
        self._ts                = ts
        self.recombination_rate = recombination_rate
        self.sequence_length    = int(sequence_length)
        self.keep_temp          = keep_temp
        self.rfmix_extra_args   = rfmix_extra_args or []
        self._tmp_dir_obj       = None

        # ── normalise inputs ──────────────────────────────────────────────
        # Store original node IDs and keys exactly as passed by the caller,
        # so the output DataFrame uses the same IDs as the GNN pipeline.
        self._orig_admixed_node_ids: list[int] = admixed_sample_ids

        # Stringify all ancestry keys for use in sample map / column names
        self._labels: list[str] = [str(k) for k in ancestry_samples.keys()]
        self._ref_node_ids: dict[str, list[int]] = {
            str(k): v for k, v in ancestry_samples.items()
        }

        # ── node ID → VCF name translation ────────────────────────────────
        # Each individual appears as one VCF column ("ind_{individual_id}").
        # Multiple nodes from the same individual collapse to one VCF name.
        self._admixed_vcf_names: list[str] = _unique_vcf_names(
            ts, admixed_sample_ids
        )
        self._ref_vcf_names: dict[str, list[str]] = {
            lbl: _unique_vcf_names(ts, nodes)
            for lbl, nodes in self._ref_node_ids.items()
        }

        # Reverse map: VCF name → first node ID of that individual.
        # We iterate in order and only record the first time we see each
        # VCF name, so ind_20 → its first node ID, etc.
        self._vcf_to_node: dict[str, int] = {}
        for nid in admixed_sample_ids:
            vcf_name = _node_to_vcf_name(ts, nid)
            if vcf_name not in self._vcf_to_node:
                self._vcf_to_node[vcf_name] = nid

        if output_dir is None:
            self._tmp_dir_obj = tempfile.TemporaryDirectory(prefix="rfmix_")
            self.output_dir   = Path(self._tmp_dir_obj.name)
        else:
            self.output_dir = Path(output_dir)
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self._validate()

    # ── public ────────────────────────────────────────────────────────────

    def infer(self) -> pd.DataFrame:
        """Run the full pipeline and return the standardised DataFrame."""
        chromosome       = self._detect_chromosome(self.vcf_path)
        genetic_map_path = self.output_dir / "genetic_map.txt"
        try:
            write_genetic_map(
                sequence_length=self.sequence_length,
                recombination_rate=self.recombination_rate,
                chrom=chromosome,
                output_path=str(genetic_map_path),
            )
            sample_map_path    = self._write_sample_map()
            query_vcf, ref_vcf = self._split_vcf()
            out_basename       = self.output_dir / "rfmix_out"
            self._run_rfmix(query_vcf, ref_vcf, sample_map_path, out_basename,
                            genetic_map_path, chromosome)

            msp_path = Path(str(out_basename) + ".msp.tsv")

            pop_order = self._read_population_order(msp_path)
            msp_df    = self._parse_msp(msp_path, pop_order)
            return self._build_output(msp_df)
        finally:
            if genetic_map_path.exists():
                genetic_map_path.unlink()
            if not self.keep_temp and self._tmp_dir_obj is not None:
                self._tmp_dir_obj.cleanup()

    # ── private: validation & setup ───────────────────────────────────────

    def _validate(self) -> None:
        if not self.vcf_path.exists():
            raise FileNotFoundError(f"VCF not found: {self.vcf_path}")
        if self.recombination_rate <= 0:
            raise ValueError(f"`recombination_rate` must be positive, got {self.recombination_rate}")
        if len(self._labels) < 2:
            raise ValueError("`ancestry_samples` must contain at least 2 groups.")
        all_ref_vcf = {n for grp in self._ref_vcf_names.values() for n in grp}
        overlap = all_ref_vcf & set(self._admixed_vcf_names)
        if overlap:
            raise ValueError(
                f"VCF sample names present in both admixed and reference sets: {overlap}"
            )
        for tool in ("rfmix", "bcftools"):
            if shutil.which(tool) is None:
                raise EnvironmentError(
                    f"'{tool}' not found on PATH. "
                    f"Please install it before running RFMixInference."
                )

    def _write_sample_map(self) -> Path:
        """
        Write the RFMix2 sample map: <vcf_sample_name>\\t<label> per line.
        Uses stringified label keys and VCF individual names.
        """
        path = self.output_dir / "sample_map.txt"
        with open(path, "w") as fh:
            for label, vcf_names in self._ref_vcf_names.items():
                for name in vcf_names:
                    fh.write(f"{name}\t{label}\n")
        return path

    def _split_vcf(self) -> tuple[Path, Path]:
        """
        Split the input VCF into query and reference VCFs using bcftools.

        The input VCF may be uncompressed (.vcf) or bgzip-compressed
        (.vcf.gz).  Region filtering with -r requires a tabix index, so
        we skip it — the simulation already contains only one chromosome.
        If the input is uncompressed we pipe it through bcftools view
        without -r; bcftools handles plain .vcf natively.
        """
        all_ref_vcf = [n for grp in self._ref_vcf_names.values() for n in grp]

        query_vcf = self.output_dir / "query.vcf.gz"
        ref_vcf   = self.output_dir / "reference.vcf.gz"

        def _extract(names: list[str], out: Path) -> None:
            subprocess.run(
                [
                    "bcftools", "view",
                    "-s", ",".join(names),
                    "-O", "z",           # bgzip-compressed output
                    "-o", str(out),
                    str(self.vcf_path),
                ],
                check=True,
            )
            subprocess.run(["bcftools", "index", str(out)], check=True)

        _extract(self._admixed_vcf_names, query_vcf)
        _extract(all_ref_vcf, ref_vcf)
        return query_vcf, ref_vcf

    @staticmethod
    def _detect_chromosome(vcf_path: Path) -> str:
        """
        Return the CHROM value from the first data line of the VCF.
        Uses ``bcftools view -H -G`` to strip the header and genotypes,
        then reads just the first column of the first record.

        Raises
        ------
        ValueError
            If no data lines are found in the VCF.
        """
        result = subprocess.run(
            ["bcftools", "view", "-H", "-G", str(vcf_path)],
            capture_output=True, text=True, check=True,
        )
        for line in result.stdout.splitlines():
            line = line.strip()
            if line:
                chrom = line.split("\t", 1)[0]
                return chrom
        raise ValueError(f"No data lines found in VCF: {vcf_path}")

    def _run_rfmix(
        self,
        query_vcf: Path,
        ref_vcf: Path,
        sample_map: Path,
        out_basename: Path,
        genetic_map_path: Path,
        chromosome: str,
    ) -> None:
        cmd = [
            "rfmix",
            "-f", str(query_vcf),
            "-r", str(ref_vcf),
            "-m", str(sample_map),
            "-g", str(genetic_map_path),
            "-o", str(out_basename),
            f"--chromosome={chromosome}",
        ] + self.rfmix_extra_args
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"RFMix exited with status {result.returncode}.\n"
                f"Command: {' '.join(cmd)}\n"
                f"--- stderr ---\n{result.stderr.strip()}\n"
                f"--- stdout ---\n{result.stdout.strip()}"
            )

    # ── private: output parsing ───────────────────────────────────────────

    @staticmethod
    def _read_population_order(msp_path: Path) -> list[str]:
        """
        Read the population order from the .msp.tsv header comment line:
            #Subpopulation order/index: A=0\\tB=1
        Returns labels sorted by their integer index.
        """
        with open(msp_path) as fh:
            first_line = fh.readline().strip()
        if ":" not in first_line:
            raise ValueError(
                f"Cannot parse population order from .msp.tsv header: {first_line!r}"
            )
        order_str = first_line.split(":", 1)[1]
        pairs = [(int(idx), name.strip())
                 for token in order_str.split()
                 for name, idx in [token.rsplit("=", 1)]]
        pairs.sort()
        return [name for _, name in pairs]

    def _parse_msp(self, msp_path: Path, pop_order: list[str]) -> pd.DataFrame:
        """
        Parse .msp.tsv Viterbi output into a DataFrame with columns:
            start_pos, end_pos,
            <vcf_name>.0 (ancestry label for haplotype 0),
            <vcf_name>.1 (ancestry label for haplotype 1)

        Both haplotype calls are stored independently so they can be
        routed to their respective node IDs in _build_output.
        """
        with open(msp_path) as fh:
            fh.readline()
            header_line = fh.readline().strip().lstrip("#")
            col_names   = header_line.split("\t")

        msp = pd.read_csv(
            msp_path, sep="\t", comment="#",
            header=None, names=col_names,
        )

        int_to_label = {i: lbl for i, lbl in enumerate(pop_order)}

        rows = []
        for _, row in msp.iterrows():
            entry: dict[str, Any] = {
                "start_pos": int(row["spos"]),
                "end_pos":   int(row["epos"]),
            }
            for vcf_name in self._admixed_vcf_names:
                # Store each haplotype call independently under
                # "<vcf_name>.0" and "<vcf_name>.1" keys
                hap0_col = f"{vcf_name}.0"
                hap1_col = f"{vcf_name}.1"
                entry[hap0_col] = int_to_label.get(int(row[hap0_col])) if hap0_col in row.index else None
                entry[hap1_col] = int_to_label.get(int(row[hap1_col])) if hap1_col in row.index else None
            rows.append(entry)
        return pd.DataFrame(rows)

    # ── private: build standardised output ────────────────────────────────

    def _build_output(
        self,
        msp_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert per-haplotype Viterbi calls (msp_df) into the standardised
        long-format output.

        Each haplotype is routed to its own node ID:
          haplotype 0 (ind_20.0) → first  node ID (e.g. 40)
          haplotype 1 (ind_20.1) → second node ID (e.g. 41)

        Segment boundaries are clamped to [0, sequence_length].
        """
        # Build map: VCF name → [first_node, second_node] in order
        vcf_to_nodes: dict[str, list[int]] = {}
        for nid in self._orig_admixed_node_ids:
            vcf_name = _node_to_vcf_name(self._ts, nid)
            vcf_to_nodes.setdefault(vcf_name, []).append(nid)

        all_rows: list[dict] = []

        for vcf_name in self._admixed_vcf_names:
            nodes = vcf_to_nodes[vcf_name]   # [node_hap0, node_hap1]

            # Process each haplotype independently
            for hap_idx, hap_suffix in enumerate([".0", ".1"]):
                if hap_idx >= len(nodes):
                    continue          # haploid edge case
                node_id  = nodes[hap_idx]
                msp_col  = f"{vcf_name}{hap_suffix}"   # e.g. "tsk_20.0"

                seg_rows: list[dict] = [
                    {
                        "start_pos":      int(seg["start_pos"]),
                        "end_pos":        int(seg["end_pos"]),
                        "ancestry_label": seg[msp_col],
                    }
                    for _, seg in msp_df.iterrows()
                ]

                merged = _merge_rfmix_segments(seg_rows)

                if merged:
                    merged[0]["start_pos"] = 0
                    merged[-1]["end_pos"]  = self.sequence_length

                for seg in merged:
                    all_rows.append({
                        "sample_id":      node_id,
                        "start_pos":      seg["start_pos"],
                        "end_pos":        seg["end_pos"],
                        "ancestry_label": seg["ancestry_label"],
                    })

        return pd.DataFrame(
            all_rows,
            columns=["sample_id", "start_pos", "end_pos", "ancestry_label"],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Segment merging
# ─────────────────────────────────────────────────────────────────────────────

def _merge_rfmix_segments(seg_rows: list[dict]) -> list[dict]:
    """
    Merge consecutive rows with the same ancestry_label into one segment.
    """
    if not seg_rows:
        return []

    merged: list[dict] = []
    current = dict(seg_rows[0])

    for row in seg_rows[1:]:
        if row["ancestry_label"] == current["ancestry_label"]:
            current["end_pos"] = row["end_pos"]
        else:
            merged.append(current)
            current = dict(row)

    merged.append(current)
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# Single tree sequence
# ─────────────────────────────────────────────────────────────────────────────

def infer_local_ancestry_single_ts(
    ts_path: str | Path,
    admixed_sample_ids: list[int],
    ancestry_samples: dict[Any, list[int]],
    recombination_rate: float,
    mutation_rate: float | None = None,
    output_dir: str | Path | None = None,
    keep_temp: bool = False,
    rfmix_extra_args: list[str] | None = None,
) -> pd.DataFrame:
    """
    Run RFMix2 local ancestry inference on a single tree sequence.

    The tree sequence is simplified down to just the nodes needed
    (*admixed_sample_ids* plus every node in *ancestry_samples*) unless its
    sample set already equals that union, in which case it is used as-is.
    A VCF is then written from the (possibly simplified) tree sequence with
    explicit "ind_{individual_id}" sample names via :func:`write_ts_vcf`, and
    fed to RFMix2 through :class:`RFMixInference`. All intermediate files
    (VCF, genetic map, RFMix2 output) are removed once inference completes,
    unless *keep_temp* is True.

    Parameters
    ----------
    ts_path : str | Path
        Path to a ``.trees`` file.
    admixed_sample_ids : list[int]
        Node IDs of admixed haplotypes, in the *original* (pre-simplification)
        tree sequence's ID space. The returned DataFrame's ``sample_id``
        column uses these same original node IDs.
    ancestry_samples : dict[Any, list[int]]
        label → list of reference node IDs, in the original ID space.
    recombination_rate : float
        Per-bp per-generation recombination rate (e.g. 1e-8).
    mutation_rate : float | None
        Per-bp per-generation mutation rate (e.g. 1e-8), used to simulate
        mutations via ``msprime.sim_mutations`` only if *ts_path* has no
        sites (RFMix2 needs variant sites to call ancestry from). Required
        if the tree sequence has no sites; ignored otherwise.
    output_dir : str | Path | None
        Directory for RFMix2 output files. Temp dir used if None.
    keep_temp : bool
        If True, retain the intermediate VCF and RFMix2 output files in
        *output_dir* instead of deleting them.
    rfmix_extra_args : list[str] | None
        Forwarded verbatim to the rfmix command line.

    Returns
    -------
    pd.DataFrame
        columns = [sample_id, start_pos, end_pos, ancestry_label],
        with sample_id in the original node ID space.
    """
    ts_path = Path(ts_path)
    if not ts_path.exists():
        raise FileNotFoundError(f"Tree sequence not found: {ts_path}")

    ts = _with_diploid_individuals(tskit.load(str(ts_path)))

    all_reference = {s for grp in ancestry_samples.values() for s in grp}
    overlap = all_reference & set(admixed_sample_ids)
    if overlap:
        raise ValueError(
            f"Node IDs present in both admixed and reference sets: {overlap}"
        )

    # Simplify to just the required nodes, unless the tree sequence's
    # samples already match that set exactly (avoids a redundant copy).
    required_nodes = sorted(set(admixed_sample_ids) | all_reference)
    if sorted(ts.samples()) == required_nodes:
        node_map = np.arange(ts.num_nodes)
    else:
        ts, node_map = ts.simplify(samples=required_nodes, map_nodes=True)

    sim_admixed_ids = [int(node_map[nid]) for nid in admixed_sample_ids]
    sim_ancestry_samples = {
        lbl: [int(node_map[nid]) for nid in nodes]
        for lbl, nodes in ancestry_samples.items()
    }

    if output_dir is None:
        tmp_dir_obj = tempfile.TemporaryDirectory(prefix="rfmix_ts_")
        work_dir    = Path(tmp_dir_obj.name)
    else:
        tmp_dir_obj = None
        work_dir    = Path(output_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

    vcf_path = work_dir / f"{ts_path.stem}.vcf"
    try:
        sim_reference_ids = [nid for nodes in sim_ancestry_samples.values() for nid in nodes]
        write_ts_vcf(ts, sim_admixed_ids + sim_reference_ids, vcf_path, mutation_rate=mutation_rate)
        rfmix_df = RFMixInference(
            vcf_path=vcf_path,
            ts=ts,
            admixed_sample_ids=sim_admixed_ids,
            ancestry_samples=sim_ancestry_samples,
            recombination_rate=recombination_rate,
            sequence_length=ts.sequence_length,
            output_dir=work_dir,
            keep_temp=keep_temp,
            rfmix_extra_args=rfmix_extra_args,
        ).infer()
    finally:
        if not keep_temp:
            if vcf_path.exists():
                vcf_path.unlink()
            if tmp_dir_obj is not None:
                tmp_dir_obj.cleanup()

    # Translate sample_id back from simplified to original node ID space.
    sim_to_orig = dict(zip(sim_admixed_ids, admixed_sample_ids))
    rfmix_df["sample_id"] = rfmix_df["sample_id"].map(sim_to_orig)
    return rfmix_df


# ─────────────────────────────────────────────────────────────────────────────
# Directory of tree sequences
# ─────────────────────────────────────────────────────────────────────────────

def _batch_worker(args: tuple) -> tuple[str, bool]:
    """
    Top-level (picklable) worker function for multiprocessing.

    Parameters
    ----------
    args : tuple
        (ts_path, output_parquet_path, shared_kwargs)
        where shared_kwargs is a dict of keyword arguments forwarded
        verbatim to :func:`infer_local_ancestry_single_ts`.

    Returns
    -------
    (ts_stem, success) where ts_stem is the stem of the tree sequence file.
    """
    ts_path, output_parquet_path, shared_kwargs = args
    ts_path              = Path(ts_path)
    output_parquet_path  = Path(output_parquet_path)

    try:
        df = infer_local_ancestry_single_ts(ts_path=ts_path, **shared_kwargs)
        df.to_parquet(output_parquet_path, index=False)
        logger.info("✓ %s → %s", ts_path.name, output_parquet_path.name)
        return ts_path.stem, True
    except Exception:
        logger.error(
            "✗ Failed to process %s:\n%s",
            ts_path.name,
            traceback.format_exc(),
        )
        return ts_path.stem, False


def infer_local_ancestry_all_ts(
    ts_dir: str | Path,
    output_dir: str | Path,
    admixed_sample_ids: list[int],
    ancestry_samples: dict[Any, list[int]],
    recombination_rate: float,
    mutation_rate: float | None = None,
    keep_temp: bool = False,
    rfmix_extra_args: list[str] | None = None,
    n_workers: int | None = None,
) -> dict[str, bool]:
    """
    Run :func:`infer_local_ancestry_single_ts` on every ``.trees`` file in
    *ts_dir* and write each result as a Parquet file in *output_dir*.

    All RFMix parameters are shared across every tree sequence. Files are
    processed in parallel using :mod:`pathos.multiprocessing`, one ``rfmix``
    process per tree sequence. RFMix2's own ``--n-threads`` flag is not used:
    on this platform, passing it to RFMix2 segfaults the binary regardless
    of thread count (reproduced with values 1-12), so internal multithreading
    is avoided entirely in favor of this process-level parallelism.

    Parameters
    ----------
    ts_dir : str | Path
        Directory to scan for ``*.trees`` files.
    output_dir : str | Path
        Directory where output Parquet files are written.
        Each output file is named ``<ts_stem>.parquet``
        (e.g. ``contig_0.trees`` → ``contig_0.parquet``).
        The directory is created if it does not exist.
    admixed_sample_ids : list[int]
        Passed unchanged to :func:`infer_local_ancestry_single_ts`.
    ancestry_samples : dict[Any, list[int]]
        Passed unchanged to :func:`infer_local_ancestry_single_ts`.
    recombination_rate : float
        Per-bp per-generation recombination rate (e.g. 1e-8).
    mutation_rate : float | None
        Passed unchanged to :func:`infer_local_ancestry_single_ts`.
    keep_temp : bool
        Passed unchanged to :func:`infer_local_ancestry_single_ts`.
    rfmix_extra_args : list[str] | None
        Passed unchanged to :func:`infer_local_ancestry_single_ts`.
    n_workers : int | None
        Number of parallel worker processes. Defaults to
        ``min(len(ts_files), os.cpu_count())``.

    Returns
    -------
    dict[str, bool]
        Mapping of tree sequence stem → True (succeeded) / False (failed).
        Failed files are logged but do not raise exceptions.

    Examples
    --------
    >>> results = infer_local_ancestry_all_ts(
    ...     ts_dir="simulations/",
    ...     output_dir="lai_results/",
    ...     admixed_sample_ids=list(range(80, 120)),
    ...     ancestry_samples={1.0: list(range(0, 40)), 2.0: list(range(40, 80))},
    ...     recombination_rate=1e-8,
    ...     n_workers=12,
    ... )
    >>> failed = [k for k, v in results.items() if not v]
    """
    ts_dir     = Path(ts_dir)
    output_dir = Path(output_dir)

    if not ts_dir.is_dir():
        raise NotADirectoryError(f"ts_dir does not exist or is not a directory: {ts_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect tree sequence files; preserve a stable sort order for reproducibility
    ts_files: list[Path] = sorted(ts_dir.glob("*.trees"))

    if not ts_files:
        logger.warning("No .trees files found in %s", ts_dir)
        return {}

    logger.info("Found %d tree sequence file(s) in %s", len(ts_files), ts_dir)

    # Shared kwargs forwarded to infer_local_ancestry_single_ts for every ts.
    # output_dir is intentionally omitted so each worker uses a fresh temp dir.
    shared_kwargs: dict[str, Any] = dict(
        admixed_sample_ids=admixed_sample_ids,
        ancestry_samples=ancestry_samples,
        recombination_rate=recombination_rate,
        mutation_rate=mutation_rate,
        keep_temp=keep_temp,
        rfmix_extra_args=rfmix_extra_args,
    )

    # Build per-file argument tuples for the worker
    worker_args: list[tuple] = [
        (
            ts_path,
            output_dir / f"{ts_path.stem}.parquet",
            shared_kwargs,
        )
        for ts_path in ts_files
    ]

    n_workers = n_workers or min(len(ts_files), os.cpu_count())
    logger.info("Launching %d worker(s) for %d tree sequence(s)", n_workers, len(ts_files))

    results: dict[str, bool] = {}

    with ProcessPool(nodes=n_workers) as pool:
        imap_it = pool.imap(_batch_worker, worker_args)
        for stem, success in tqdm(imap_it, total=len(ts_files), desc="RFMix inference", unit="ts"):
            results[stem] = success

    n_ok   = sum(results.values())
    n_fail = len(results) - n_ok
    logger.info(
        "Batch complete: %d succeeded, %d failed.",
        n_ok, n_fail,
    )
    if n_fail:
        failed_names = [k for k, v in results.items() if not v]
        logger.warning("Failed tree sequences: %s", failed_names)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────────────────────
# #%%
# import msprime

# n = 10
# ancestry_samples = {
#     1.0: list(range(0, 2 * n)),
#     2.0: list(range(2 * n, 4 * n)),
# }
# admixed_sample_ids = list(range(4 * n, 5 * n))

# # admix.trees has no sites/mutations on disk; RFMix needs variant sites to
# # call ancestry from, so mutate a copy rather than the tree sequence itself.
# ts = tskit.load("admix.trees")
# ts = msprime.sim_mutations(ts, rate=1e-8, random_seed=1)
# ts.dump("admix_mutated.trees")

# rfmix_df = infer_local_ancestry_single_ts(
#     ts_path="admix_mutated.trees",
#     admixed_sample_ids=admixed_sample_ids,
#     ancestry_samples=ancestry_samples,
#     recombination_rate=1e-8,
#     keep_temp=False,
# )
# print(rfmix_df.shape)

# #%%
# from LocalAncestryGNN import TrueLocalAncestry, plot_chromosome_painting
# # admix.trees was simulated with demography.add_census(time=admix_time + 1);
# # admix_time=100 matches simulate_admix_ts()'s default in LocalAncestryGNN.py.
# true_ancestry = TrueLocalAncestry(
#     ts_dir=".", admixed_sample_ids=admixed_sample_ids, admix_time=100
# )
# true_df = true_ancestry.infer_single("admix.trees")

# # TrueLocalAncestry reports integer population IDs; ancestry_samples above
# # maps population A -> 1.0 and B -> 2.0, so the labels already coincide
# # numerically — just cast to match RFMix's float label strings ("1.0", "2.0").
# true_df["ancestry_label"] = true_df["ancestry_label"].astype(float)

# fig = plot_chromosome_painting(admixed_sample_ids[5], rfmix_df, true_df)
# fig.savefig("rfmix_vs_truth.png", dpi=150, bbox_inches="tight")
# print(rfmix_df.head(10))

# # %%
