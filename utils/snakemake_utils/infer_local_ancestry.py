#%%
import argparse
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
    span_wide = span_wide.reindex(index=admixed_sample_ids, columns=reference_pops, fill_value=0.0)

    total_span = span_wide.sum(axis=1)
    prop_df = span_wide.div(total_span, axis=0)
    prop_df.columns = [f"{pop}_prop" for pop in reference_pops]
    prop_df.insert(0, "sample_id", prop_df.index)
    return prop_df.reset_index(drop=True)


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
    parser.add_argument("--reference_pops", nargs='+',default=None, help="population labels to use as reference populations for local ancestry inference")
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

    if reference_pops is None:
        # Default to every population in pop_labels.csv that isn't an admixed
        # population, optionally further restricted to --pops if given.
        reference_pops = [p for p in pop_labels["population"].unique() if p not in admix_pops]
        if args.pops is not None:
            reference_pops = [p for p in reference_pops if p in args.pops]
        print(f"--reference_pops not specified; defaulting to: {reference_pops}")

    reference_samples_dict = {pop_label: pop_labels[pop_labels["population"] == pop_label]["sample_id"].values for pop_label in reference_pops}
    if len(reference_samples_dict) == 0:
        raise ValueError(f"No reference populations resolved (reference_pops={reference_pops}) in {args.pop_labels}")

    run_local_ancestry(
        LAI_method=args.LAI_method,
        input_path=input_path,
        LAI_path=LAI_path,
        admix_samples=admix_samples.tolist(),
        reference_samples_dict={label: samples.tolist() for label, samples in reference_samples_dict.items()},
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
