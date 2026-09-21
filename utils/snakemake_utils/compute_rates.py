#%%
#!/usr/bin/env python3
import argparse
import os
import shutil
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from ComputeCoalescentRates import (
    adaptive_local_linear_filter,
    compute_coal_rate_weights,
    compute_coal_rates,
    slice_pdf_for_rates,
)
import numpy as np
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Compute coalescence rates from tree sequences")
    parser.add_argument("--pdf_path", required=True, help="Path to tree sequence directory")
    parser.add_argument("--output_path", required=True, help="Output directory for pdf data")
    parser.add_argument("--output_name", required=True, help="Output filename for pdf parquet file")
    parser.add_argument("--minimum_window_size", type=float, required=True, help="minimum window size for rate calculation")
    parser.add_argument("--maximum_window_size", type=float, required=True, help="maximum window size for rate calculation")
    parser.add_argument("--rates_min_time", type=float, required=True, help="Start of the interval rates are estimated on; minimum_window_size is the bandwidth here")
    parser.add_argument("--rates_max_time", type=float, required=True, help="End of the interval rates are estimated on; maximum_window_size is the bandwidth here")
    parser.add_argument("--per-haplotype", dest="per_haplotype", action="store_true", help="If set, per-sample-pair (haplotype) rates are returned. Otherwise rates are averaged over pairs of haploid samples from the same diploid individual to return per-individual-pair rates.")
    parser.add_argument("--pop_labels", type=str, default=None, help="Path to population labels csv (with sample_id and individual_id columns). Used to map haploid samples to diploid individuals when phased=False. If omitted, consecutive sample IDs (2k, 2k+1) are assumed to belong to individual k.")
    args = parser.parse_args()

    pdf_path = args.pdf_path
    output_path = args.output_path
    output_name = args.output_name
    minimum_window_size = args.minimum_window_size
    maximum_window_size = args.maximum_window_size
    per_haplotype = args.per_haplotype
    print(f"computing rates from estimates of the pdfs of pairwise coalescent times: {args.pdf_path}")
    pdf_df = pd.read_parquet(pdf_path)
    pdf_df.columns = pdf_df.columns.astype(float)
    coal_counts_dir = output_path + "/counts_dir" + "/" + os.path.basename(pdf_path).replace(".parquet", "")

    # Sample-to-individual mapping for averaging haplotype pairs into individuals
    # (only used when phased=False). Passing it explicitly avoids relying on the
    # consecutive sample-ID fallback, which is invalid for subsampled empirical data.
    sample_individal_ids = None
    if not per_haplotype and args.pop_labels is not None:
        sample_individal_ids = pd.read_csv(args.pop_labels)[["sample_id", "individual_id"]]

    # Widen the requested interval by the filter's boundary width so that every
    # timepoint we keep is an interior, symmetric-window row -- i.e. carries the
    # bandwidth the config asked for rather than an artefact of where the grid
    # stops. The pad is clipped off again before saving.
    padded_pdf, keep_mask = slice_pdf_for_rates(
        pdf_df, args.rates_min_time, args.rates_max_time,
        minimum_window_size, maximum_window_size,
    )

    # One filter, built once and shared, so the rates and their weights provably
    # use the same bandwidth schedule; each call would otherwise build its own.
    padded_times = np.asarray(padded_pdf.columns, dtype=float)
    filter_matrix = adaptive_local_linear_filter(
        minimum_window_size, maximum_window_size,
        len(padded_times), padded_times[1] - padded_times[0],
    )

    rates_df = compute_coal_rates(padded_pdf, minimum_window_size, maximum_window_size,
                                  per_haplotype=per_haplotype,
                                  sample_individal_ids=sample_individal_ids,
                                  filter_matrix=filter_matrix)

    # Loss weights accompany the rates: w = S_hat / (dt * sum_r Phi^2), derived
    # from the same filter and pdf, so they are written alongside them.
    weights_df = compute_coal_rate_weights(padded_pdf, minimum_window_size, maximum_window_size,
                                           per_haplotype=per_haplotype,
                                           sample_individal_ids=sample_individal_ids,
                                           filter_matrix=filter_matrix)

    # Clip the pad off BOTH with the same mask. The embeddings loss pairs each
    # rate with its weight by position, so any divergence here would be silent.
    rates_df = rates_df.loc[:, keep_mask]
    weights_df = weights_df.loc[:, keep_mask]
    if not rates_df.columns.equals(weights_df.columns):
        raise RuntimeError(
            "rates and weights ended up on different time grids after clipping; "
            "they must stay column-aligned for the embeddings loss."
        )
    print(f"rates over [{rates_df.columns[0]:g}, {rates_df.columns[-1]:g}] "
          f"({rates_df.shape[1]} timepoints), from a pdf of {pdf_df.shape[1]}")

    rates_df.columns = rates_df.columns.astype(str)
    rates_df.to_parquet(f"{output_path}/{output_name}")
    weights_df.columns = weights_df.columns.astype(str)
    weights_df.to_parquet(f"{output_path}/weights_{output_name}")

    # The per-contig coalescent counts are a bulky intermediate (hundreds of GB
    # for fine time grids) needed only to build the pdf, which now exists on
    # disk, so the directory is removed to reclaim the space. It is reproducible
    # by re-running compute_pdf.py, which recounts from the tree sequences.
    if os.path.isdir(coal_counts_dir):
        print(f"removing coalescent counts directory: {coal_counts_dir}")
        shutil.rmtree(coal_counts_dir)

if __name__ == "__main__":
    main()