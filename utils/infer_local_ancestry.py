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
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from LocalAncestryRFmix import *


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
    lai_files = [f for f in os.listdir(lai_dir) if f.endswith(".parquet")]
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


def main():
    parser = argparse.ArgumentParser(description="Infer local ancestry via RFMix2 for all tree sequences in a directory")
    parser.add_argument("--input_path", required=True, help="Path to tree sequence files")
    parser.add_argument("--LAI_path", required=True, help="Output directory for LAI dataframes")
    parser.add_argument("--ancestry_STT_path", required=True, help="Output directory for ancestry STT runs")
    parser.add_argument("--pop_labels", required=True, help="Path to population labels parquet file")
    parser.add_argument("--pop_colors", required=True, help="Path to population colors file")
    parser.add_argument("--r", type=float, default=1.2e-8, help="recombination_rate")
    parser.add_argument("--mu", type=float, default=1.2e-8, help="mutation_rate")
    parser.add_argument("--admix_pops", nargs='+',default=None, help="population labels to compute local ancestry for")
    parser.add_argument("--reference_pops", nargs='+',default=None, help="population labels to use as reference populations for local ancestry inference")
    parser.add_argument("--significant_ancestry_threshold", type=float, default=0.05, help="threshold for significant ancestry proportion to include in STT analysis")
    parser.add_argument("--num_threads", type=int, default=1, help="number of tree sequences to process in parallel (one rfmix process each)")
    parser.add_argument("--pops", nargs='+', default=None, help="Optional restriction: population labels the whole analysis is scoped to (used to filter the default reference_pops when --reference_pops is not given).")
   
    
    args = parser.parse_args()
    input_path = args.input_path
    LAI_path = args.LAI_path
    ancestry_STT_path = args.ancestry_STT_path
    pop_labels = pd.read_csv(args.pop_labels)
    pop_colors = pd.read_csv(args.pop_colors)
    recomb_rate = args.r
    mut_rate = args.mu
    admix_pops = args.admix_pops
    reference_pops = args.reference_pops
    significant_ancestry_threshold = args.significant_ancestry_threshold
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

    print(f"Running RFMix local ancestry inference on tree sequences in: {input_path}")
    results = infer_local_ancestry_all_ts(
        ts_dir=input_path,
        output_dir=LAI_path,
        admixed_sample_ids=admix_samples.tolist(),
        ancestry_samples={label: samples.tolist() for label, samples in reference_samples_dict.items()},
        recombination_rate=recomb_rate,
        mutation_rate=mut_rate,
        n_workers=num_threads,
    )

    n_failed = sum(1 for success in results.values() if not success)
    print(f"Finished: {len(results) - n_failed}/{len(results)} tree sequences succeeded.")

    print("Computing total ancestry proportions per sample")
    total_admix_prop = compute_total_ancestry_proportions(
        lai_dir=LAI_path,
        admixed_sample_ids=admix_samples,
        reference_pops=reference_pops,
    )
    out_path = os.path.join(os.path.dirname(os.path.normpath(LAI_path)), "total_ancestry_proportions.parquet")
    total_admix_prop.to_parquet(out_path, index=False)
    print(f"Saved total ancestry proportions to: {out_path}")

    for reference_pop in reference_pops:
        dirname = os.path.join(ancestry_STT_path, reference_pop)
        data_dir = os.path.join(dirname, "data")
        plots_dir = os.path.join(dirname, "plot")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Drop admixed samples that don't meet the ancestry threshold for
        # this reference population; all other samples (reference panels,
        # samples of other admixed pops, etc.) are kept as-is.
        insignificant_admix_samples = total_admix_prop[total_admix_prop[f"{reference_pop}_prop"] <= significant_ancestry_threshold]["sample_id"].values
        pop_labels_filtered = pop_labels[~pop_labels["sample_id"].isin(insignificant_admix_samples)]
        pop_labels_filtered["is_admixed"] = pop_labels_filtered["population"].isin(admix_pops)
        pop_labels_filtered.to_csv(os.path.join(data_dir, "pop_labels.csv"), index=False)
        pop_colors.to_csv(os.path.join(data_dir, "pop_colors.csv"), index=False)

        n_dropped = len(insignificant_admix_samples)
        print(f"{reference_pop}: kept {len(pop_labels_filtered)}/{len(pop_labels)} samples ({n_dropped} admixed samples below threshold={significant_ancestry_threshold})")


if __name__ == "__main__":
    main()
