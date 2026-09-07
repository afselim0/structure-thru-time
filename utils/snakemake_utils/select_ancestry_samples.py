#%%
import argparse
import os
import pandas as pd

# Genome-wide ancestry proportions written by infer_local_ancestry.py into
# LAI/<method>/. Kept in sync with the constant of the same name there.
TOTAL_ANCESTRY_PROPORTIONS_FILE = "total_ancestry_proportions.parquet"


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Select the samples entering each ancestry-specific STT run, by thresholding the "
            "genome-wide ancestry proportions produced by infer_local_ancestry.py. Cheap and "
            "independent of local ancestry inference itself, so re-thresholding never re-infers."
        )
    )
    parser.add_argument("--LAI_path", required=True, help="Directory holding the LAI outputs for this method (contains total_ancestry_proportions.parquet)")
    parser.add_argument("--ancestry_STT_path", required=True, help="Output directory for ancestry STT runs; one subdirectory per reference population")
    parser.add_argument("--pop_labels", required=True, help="Path to population labels file")
    parser.add_argument("--pop_colors", required=True, help="Path to population colors file")
    parser.add_argument("--admix_pops", nargs='+', required=True, help="Population labels treated as admixed")
    parser.add_argument("--significant_ancestry_threshold", type=float, default=0.05, help="threshold for significant ancestry proportion to include in CoaSTT analysis")

    args = parser.parse_args()
    ancestry_STT_path = args.ancestry_STT_path
    pop_labels = pd.read_csv(args.pop_labels)
    pop_colors = pd.read_csv(args.pop_colors)
    admix_pops = args.admix_pops
    significant_ancestry_threshold = args.significant_ancestry_threshold

    proportions_path = os.path.join(args.LAI_path, TOTAL_ANCESTRY_PROPORTIONS_FILE)
    if not os.path.isfile(proportions_path):
        raise FileNotFoundError(
            f"Ancestry proportions not found at {proportions_path}. This file is written by "
            "utils/infer_local_ancestry.py; run the local ancestry inference step first."
        )
    total_admix_prop = pd.read_parquet(proportions_path)

    # Recover the reference populations from the "<pop>_prop" columns rather
    # than re-deriving them from config: infer_local_ancestry.py may have
    # defaulted them internally, and this file records what it actually used.
    reference_pops = [c.removesuffix("_prop") for c in total_admix_prop.columns if c != "sample_id"]
    if not reference_pops:
        raise ValueError(f"No '<pop>_prop' columns found in {proportions_path}; cannot resolve reference populations.")

    for reference_pop in reference_pops:
        data_dir = os.path.join(ancestry_STT_path, reference_pop, "data")
        plots_dir = os.path.join(ancestry_STT_path, reference_pop, "plots")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(plots_dir, exist_ok=True)

        # Drop admixed samples that don't meet the ancestry threshold for
        # this reference population; all other samples (reference panels,
        # samples of other admixed pops, etc.) are kept as-is.
        insignificant_admix_samples = total_admix_prop[total_admix_prop[f"{reference_pop}_prop"] <= significant_ancestry_threshold]["sample_id"].values
        pop_labels_filtered = pop_labels[~pop_labels["sample_id"].isin(insignificant_admix_samples)].copy()
        pop_labels_filtered["is_admixed"] = pop_labels_filtered["population"].isin(admix_pops)
        pop_labels_filtered.to_csv(os.path.join(data_dir, "pop_labels.csv"), index=False)
        pop_colors.to_csv(os.path.join(data_dir, "pop_colors.csv"), index=False)

        n_dropped = len(insignificant_admix_samples)
        print(f"{reference_pop}: kept {len(pop_labels_filtered)}/{len(pop_labels)} samples ({n_dropped} admixed samples below threshold={significant_ancestry_threshold})")


if __name__ == "__main__":
    main()
