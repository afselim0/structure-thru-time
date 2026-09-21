#%%
import argparse
import json
import pandas as pd
import demes
import msprime
import numpy as np
import os
from tqdm import tqdm
from time import time
from pathos.multiprocessing import Pool
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from LocalAncestryRFmix import *
from LocalAncestryGNN import LocalAncestryInference

# Genome-wide ancestry proportions, written alongside the per-contig LAI files
# in LAI/<method>/ and skipped when those per-contig files are enumerated.
TOTAL_ANCESTRY_PROPORTIONS_FILE = "total_ancestry_proportions.parquet"

# Provenance record of which exact samples formed each reference panel, written
# beside the proportions in LAI/<method>/. A .csv deliberately: every consumer of
# that directory enumerates "*.parquet", so this file is invisible to them.
REFERENCE_PANELS_FILE = "reference_panels.csv"


def compute_total_ancestry_proportions(lai_dir: str, admixed_sample_ids, reference_pops: list[str]) -> pd.DataFrame:
    """
    Aggregate per-contig local ancestry dataframes (as written by
    infer_local_ancestry_all_ts) into the genome-wide proportion of each
    reference population assigned to each admixed sample.

    Reads every "*.parquet" file in lai_dir, sums segment spans
    (end_pos - start_pos) per (sample_id, ancestry_label) across all
    contigs, then normalizes by each sample's total span.

    Parameters
    ----------
    lai_dir : str
        Directory containing per-contig LAI parquet files
        (columns: sample_id, start_pos, end_pos, ancestry_label).
    admixed_sample_ids : array-like of int
        Node IDs of admixed samples to report proportions for.
    reference_pops : list[str]
        Reference population labels; output has one "<pop>_prop" column
        per label, filled with 0.0 for samples with no assigned segments.

    Returns
    -------
    pd.DataFrame
        columns = [sample_id] + [f"{pop}_prop" for pop in reference_pops]
    """
    # Exclude this function's own output, which is written into the same
    # directory and would otherwise be re-read as if it were a contig on re-runs.
    lai_files = [
        f for f in os.listdir(lai_dir)
        if f.endswith(".parquet") and f != TOTAL_ANCESTRY_PROPORTIONS_FILE
    ]
    if not lai_files:
        raise FileNotFoundError(f"No .parquet files found in {lai_dir}")

    admixed_sample_ids = list(admixed_sample_ids)

    # Sum segment spans per (sample_id, ancestry_label) within each contig
    # via a vectorized groupby, then accumulate across contigs — avoids
    # per-sample/per-population boolean masking and scalar .loc updates.
    span_totals = None
    for file in tqdm(lai_files, desc="Aggregating ancestry proportions"):
        df = pd.read_parquet(
            os.path.join(lai_dir, file),
            columns=["sample_id", "start_pos", "end_pos", "ancestry_label"],
        )
        df = df[df["sample_id"].isin(admixed_sample_ids)]
        span = df["end_pos"] - df["start_pos"]
        contig_totals = span.groupby([df["sample_id"], df["ancestry_label"]]).sum()
        span_totals = contig_totals if span_totals is None else contig_totals.add(span_totals, fill_value=0.0)

    span_wide = span_totals.unstack("ancestry_label", fill_value=0.0)

    # Guard the reindex below against a label-set mismatch between what the LAI
    # files actually contain and the panels we were told to report. Without this
    # a mismatch yields an all-zero frame, zero row sums, and a silent NaN from
    # the division that survives all the way to the plots.
    observed = set(span_wide.columns)
    requested = set(reference_pops)
    if not (observed & requested):
        raise ValueError(
            f"None of the requested reference labels {sorted(requested)} appear in the local "
            f"ancestry calls in {lai_dir} (which contain {sorted(observed)}). Composite "
            "reference_pops are incompatible with LAI_method='True', whose ancestry labels are "
            "always atomic deme names."
        )
    if requested - observed:
        print(f"WARNING: reference labels with no assigned segments: {sorted(requested - observed)}")
    if observed - requested:
        print(f"WARNING: LAI files contain unrequested labels (dropped): {sorted(observed - requested)}")

    # Positional column assignment below is only correct because this reindex
    # forces the column order to match reference_pops.
    span_wide = span_wide.reindex(index=admixed_sample_ids, columns=reference_pops, fill_value=0.0)

    total_span = span_wide.sum(axis=1)
    if (total_span <= 0).any():
        bad = total_span.index[total_span <= 0].tolist()
        raise ValueError(
            f"{len(bad)} admixed sample(s) have zero total assigned ancestry span "
            f"(e.g. sample_id={bad[:5]}); their proportions would be NaN."
        )
    prop_df = span_wide.div(total_span, axis=0)
    prop_df.columns = [f"{pop}_prop" for pop in reference_pops]
    prop_df.insert(0, "sample_id", prop_df.index)
    return prop_df.reset_index(drop=True)


def resolve_reference_panels(
    pop_labels: pd.DataFrame,
    reference_spec: dict | None,
    admix_pops: list[str],
    pops: list[str] | None = None,
) -> dict[str, list[int]]:
    """Resolve the reference-panel spec into label -> haploid sample_ids.

    This is the one place where a reference population stops being a name and
    becomes a set of samples; everything downstream is string plumbing on the
    labels returned here.

    Three input forms, one output shape:

    * ``reference_spec is None`` -- default: one panel per non-admixed population
      in ``pop_labels`` (optionally restricted to ``pops``), each using all of
      that population's samples. This is the historical behaviour.
    * ``{"A": None, "B": None}`` -- one panel per named population, all samples.
      What a plain ``reference_pops: [A, B]`` list normalises to.
    * ``{"G": {"P": n, ...}}`` -- a COMPOSITE panel ``G`` built from the first
      ``n`` DIPLOID INDIVIDUALS of each source population ``P``. Both haplotypes
      of each chosen individual are included, so the panel holds
      ``2 * sum(n)`` haploid sample_ids. This is what lets panels be size-matched
      across populations of different sizes.

    Selection is deterministic and seedless by design: panels must be
    reproducible across reruns and comparable across LAI methods. "First n"
    means lowest ``individual_id`` -- keyed on the individual rather than on
    ``sample_id`` so that both haplotypes of an individual are always taken
    together, even if the rows are unsorted or an individual's haplotypes are
    not adjacent. Note this makes the choice SYSTEMATIC rather than random: if
    ``individual_id`` order correlates with anything (accession order, batch),
    so does the panel.

    Parameters
    ----------
    pop_labels : pd.DataFrame
        Needs columns ``sample_id`` (haploid; two rows per individual),
        ``population`` and ``individual_id``.
    reference_spec : dict or None
        Normalised spec as produced by the snakefile's ``_to_reference_spec``.
    admix_pops : list[str]
        Populations treated as admixed: excluded from the default, and rejected
        as composite sources.
    pops : list[str] or None
        Optional run-wide population restriction. Applied only to the ``None``
        default; explicit specs are validated in the snakefile.

    Returns
    -------
    dict[str, list[int]]
        Insertion-ordered label -> sorted haploid sample_ids.
    """
    required = {"sample_id", "population", "individual_id"}
    missing_cols = required - set(pop_labels.columns)
    if missing_cols:
        raise ValueError(f"pop_labels is missing required column(s): {sorted(missing_cols)}")

    if reference_spec is None:
        labels = [p for p in pop_labels["population"].unique() if p not in admix_pops]
        if pops is not None:
            labels = [p for p in labels if p in pops]
        print(f"reference_pops not specified; defaulting to: {labels}")
        reference_spec = {label: None for label in labels}

    known_pops = set(pop_labels["population"].unique())
    panels: dict[str, list[int]] = {}
    claimed: dict[str, str] = {}      # source population -> the label that took it

    for label, members in reference_spec.items():
        # A group with no member mapping means "all samples of the population
        # that shares the label's name" -- the legacy one-pop-one-panel form.
        sources = {label: None} if members is None else members

        sample_ids: list[int] = []
        for source, n_individuals in sources.items():
            if source not in known_pops:
                raise ValueError(
                    f"reference_pops['{label}'] draws from population {source!r}, which is not "
                    f"present in pop_labels (populations: {sorted(known_pops)})."
                )
            if source in admix_pops:
                raise ValueError(
                    f"reference_pops['{label}'] draws from {source!r}, which is listed in "
                    "admixed_pops; a reference panel cannot contain admixed samples."
                )
            if source in claimed:
                raise ValueError(
                    f"Source population {source!r} is claimed by both '{claimed[source]}' and "
                    f"'{label}'; reference panels must be disjoint. GNN takes an argmax over the "
                    "panels and RFmix assigns one label per individual, so overlapping panels are "
                    "undefined rather than merely redundant."
                )
            claimed[source] = label

            sub = pop_labels[pop_labels["population"] == source]

            if n_individuals is None:
                chosen = sub
            else:
                # np.sort on an integer column is numeric, so individual 2 comes
                # before individual 10 (a lexicographic sort would not).
                individuals = np.sort(sub["individual_id"].unique())
                if len(individuals) < n_individuals:
                    raise ValueError(
                        f"reference_pops['{label}'] requests {n_individuals} individuals from "
                        f"{source!r}, but only {len(individuals)} are present in pop_labels. "
                        "Lower the count, or drop it to use all of them."
                    )
                chosen = sub[sub["individual_id"].isin(individuals[:n_individuals])]

            sample_ids.extend(chosen["sample_id"].tolist())

        if not sample_ids:
            raise ValueError(f"reference_pops['{label}'] resolved to zero samples.")
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError(f"reference_pops['{label}'] resolved to duplicate sample_ids.")
        panels[label] = sorted(int(s) for s in sample_ids)

    if len(panels) < 2:
        raise ValueError(
            f"Local ancestry inference needs at least 2 reference panels; resolved "
            f"{len(panels)}: {list(panels)}."
        )
    return panels


def write_reference_panels_manifest(panels: dict[str, list[int]], pop_labels: pd.DataFrame,
                                    out_path: str) -> pd.DataFrame:
    """Record which samples formed each reference panel, for provenance.

    Long format, one row per haploid sample, so "is this balanced?" is a
    one-line groupby. Also records what the ``None`` default actually resolved
    to, which otherwise exists only in stdout.
    """
    cols = [c for c in ("sample_id", "population", "individual_id", "sample_name")
            if c in pop_labels.columns]
    frames = []
    for label, sample_ids in panels.items():
        rows = pop_labels[pop_labels["sample_id"].isin(sample_ids)][cols].copy()
        rows.insert(0, "ancestry_label", label)
        frames.append(rows)
    manifest = pd.concat(frames, ignore_index=True).rename(
        columns={"population": "source_population"}
    )
    manifest.to_csv(out_path, index=False)
    return manifest


def run_local_ancestry(
    LAI_method: str,
    input_path: str,
    LAI_path: str,
    admix_samples: list[int],
    reference_samples_dict: dict[str, list[int]],
    recomb_rate: float,
    mut_rate: float,
    num_threads: int,
) -> None:
    """
    Populate LAI_path with one per-contig local ancestry parquet file, using
    the requested inference method.

    RFmix : RFMix2 via src/LocalAncestryRFmix.py
    GNN   : genealogical nearest neighbours via src/LocalAncestryGNN.py
    True  : ground-truth calls already written by utils/simulate_trees.py
            (compute_true_local_ancestry) into LAI_path — nothing is inferred,
            the files are only validated to exist.

    Parameters mirror the inference entry points; recomb_rate/mut_rate are
    only consulted by RFmix.
    """
    if LAI_method == "RFmix":
        print(f"Running RFMix local ancestry inference on tree sequences in: {input_path}")
        results = infer_local_ancestry_all_ts(
            ts_dir=input_path,
            output_dir=LAI_path,
            admixed_sample_ids=admix_samples,
            ancestry_samples=reference_samples_dict,
            recombination_rate=recomb_rate,
            mutation_rate=mut_rate,
            n_workers=num_threads,
        )
        n_failed = sum(1 for success in results.values() if not success)
        print(f"Finished: {len(results) - n_failed}/{len(results)} tree sequences succeeded.")

    elif LAI_method == "GNN":
        print(f"Running GNN local ancestry inference on tree sequences in: {input_path}")
        inference = LocalAncestryInference(
            ts_dir=input_path,
            admixed_sample_ids=admix_samples,
            ancestry_samples=reference_samples_dict,
        )
        inference.infer_all(output_dir=LAI_path, num_threads=num_threads)
        print(f"Finished: wrote GNN local ancestry for {len(inference.ts_files)} tree sequences.")

    elif LAI_method == "True":
        # simulate_trees.py wrote the ground-truth calls here during simulation;
        # this branch only checks they are present and usable.
        true_files = os.listdir(LAI_path) if os.path.isdir(LAI_path) else []
        true_files = [
            f for f in true_files
            if f.endswith(".parquet") and f != TOTAL_ANCESTRY_PROPORTIONS_FILE
        ]
        if not true_files:
            raise FileNotFoundError(
                f"LAI_method='True' but no ground-truth local ancestry parquet files were found in {LAI_path}. "
                "These are written by utils/simulate_trees.py during simulation, and only for demographies "
                "with a positive admix_time in pop_labels.csv. Re-run the simulation, or use "
                "LAI_method='RFmix' or 'GNN'."
            )
        print(f"Using {len(true_files)} ground-truth local ancestry files from: {LAI_path}")

    else:
        raise ValueError(f"Unknown LAI_method={LAI_method!r}; expected one of ['RFmix', 'GNN', 'True'].")


def main():
    parser = argparse.ArgumentParser(description="Infer local ancestry (RFmix/GNN) or use ground-truth calls, for all tree sequences in a directory")
    parser.add_argument("--input_path", required=True, help="Path to tree sequence files")
    parser.add_argument("--LAI_path", required=True, help="Output directory for LAI dataframes")
    parser.add_argument("--pop_labels", required=True, help="Path to population labels parquet file")
    parser.add_argument("--r", type=float, default=1.2e-8, help="recombination_rate")
    parser.add_argument("--mu", type=float, default=1.2e-8, help="mutation_rate")
    parser.add_argument("--admix_pops", nargs='+',default=None, help="population labels to compute local ancestry for")
    # Two spellings of the same thing: a flat list of population names, or the
    # composite form carrying per-source individual counts. The composite cannot
    # be space-joined onto a command line, so it crosses as one JSON object.
    ref_group = parser.add_mutually_exclusive_group()
    ref_group.add_argument("--reference_pops", nargs='+', default=None, help="population labels to use as reference populations for local ancestry inference")
    ref_group.add_argument("--reference_spec", default=None, help="JSON object mapping reference-panel label -> {source_population: n_individuals}, or null for 'all samples of the population named by the label'. Counts are DIPLOID INDIVIDUALS; both haplotypes of each chosen individual enter the panel.")
    parser.add_argument("--LAI_method", choices=["RFmix", "GNN", "True"], default="RFmix", help="Local ancestry method: RFmix (RFMix2), GNN (genealogical nearest neighbours), or True (ground-truth calls written by simulate_trees.py)")
    parser.add_argument("--num_threads", type=int, default=1, help="number of tree sequences to process in parallel (one rfmix process each)")
    parser.add_argument("--pops", nargs='+', default=None, help="Optional restriction: population labels the whole analysis is scoped to (used to filter the default reference_pops when --reference_pops is not given).")


    args = parser.parse_args()
    input_path = args.input_path
    LAI_path = args.LAI_path
    pop_labels = pd.read_csv(args.pop_labels)
    recomb_rate = args.r
    mut_rate = args.mu
    admix_pops = args.admix_pops
    reference_pops = args.reference_pops
    num_threads = int(args.num_threads)

    admix_samples = pop_labels[pop_labels["population"].isin(admix_pops)]["sample_id"].values if admix_pops is not None else None

    if admix_samples is None or len(admix_samples) == 0:
        raise ValueError(f"No samples found for admix_pops={admix_pops} in {args.pop_labels}")

    # Normalise whichever spelling arrived into the one internal form, then
    # resolve it to actual samples. The None default (every non-admixed
    # population, all samples) lives inside the resolver so it is unit-testable.
    if args.reference_spec is not None:
        reference_spec = json.loads(args.reference_spec)
    elif reference_pops is not None:
        reference_spec = {p: None for p in reference_pops}
    else:
        reference_spec = None

    reference_samples_dict = resolve_reference_panels(
        pop_labels=pop_labels,
        reference_spec=reference_spec,
        admix_pops=admix_pops,
        pops=args.pops,
    )
    reference_pops = list(reference_samples_dict.keys())
    for label, ids in reference_samples_dict.items():
        print(f"  reference panel {label}: {len(ids)} haplotypes ({len(ids) // 2} individuals)")

    # Written before inference so it survives a later failure, and so a long
    # RFmix run can be checked for panel balance while it is still going.
    os.makedirs(LAI_path, exist_ok=True)
    write_reference_panels_manifest(
        reference_samples_dict, pop_labels, os.path.join(LAI_path, REFERENCE_PANELS_FILE)
    )
    print(f"Saved reference panel manifest to: {os.path.join(LAI_path, REFERENCE_PANELS_FILE)}")

    run_local_ancestry(
        LAI_method=args.LAI_method,
        input_path=input_path,
        LAI_path=LAI_path,
        admix_samples=admix_samples.tolist(),
        reference_samples_dict=reference_samples_dict,
        recomb_rate=recomb_rate,
        mut_rate=mut_rate,
        num_threads=num_threads,
    )

    print("Computing total ancestry proportions per sample")
    total_admix_prop = compute_total_ancestry_proportions(
        lai_dir=LAI_path,
        admixed_sample_ids=admix_samples,
        reference_pops=reference_pops,
    )
    # Written inside the method's own LAI directory (LAI/<method>/) so the
    # proportions stay attributable to the method that produced the calls.
    # This file is the sole hand-off to select_ancestry_samples.py, which does
    # the threshold-dependent sample selection in a separate rule — keeping
    # significant_ancestry_threshold out of this (expensive) step entirely.
    out_path = os.path.join(LAI_path, TOTAL_ANCESTRY_PROPORTIONS_FILE)
    total_admix_prop.to_parquet(out_path, index=False)
    print(f"Saved total ancestry proportions to: {out_path}")


if __name__ == "__main__":
    main()
