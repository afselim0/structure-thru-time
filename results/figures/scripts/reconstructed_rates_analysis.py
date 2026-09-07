#%%
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plotting_style import *

setup_plotting_style()

DATA_DIR = "../data"
OUTPUT_DIR = "../outputs"
SCAN_DIR = "../../simulations/outputs/complex_4_pops_d"
SUMMARY_CSV = f"{DATA_DIR}/reconstructed_rates_scan_d_offset_removed.csv"

D_VALUES = list(range(2, 11))

# the inverse-distance link maps the largest representable distance to a rate of
# 1/N_max, so every reconstructed rate carries that additive floor. Subtracting it
# (and clamping at zero) removes the floor before comparing against the true rates.
N_MAX = 1e6
RATE_OFFSET = 1.0 / N_MAX

NOISY_INPUT_FILE = "rates.minWS20.maxWS100.V1.FS1.0.maxT4000.nT1000.parquet"
SMOOTH_INPUT_FILE = "rates.minWS100.maxWS500.V1.FS1.0.maxT4000.nT1000.parquet"


def align_to_grid(reference_rates, target_columns):
    """Interpolate rates onto the time grid of the reconstructed/input rates.

    The true rates are stored on a slightly different grid (0..4000, spacing 4.004)
    than the estimated rates (0..3996, spacing 4.0), so they must be resampled
    rather than silently relabelled before any pointwise comparison.

    Args:
        reference_rates (pd.DataFrame): Rates indexed by pair, columns are times.
        target_columns (array-like): Time points to interpolate onto.

    Returns:
        pd.DataFrame: Rates on the target time grid, same row order as the input.
    """
    source_times = np.asarray(reference_rates.columns, dtype=float)
    target_times = np.asarray(target_columns, dtype=float)
    interpolated = np.vstack([
        np.interp(target_times, source_times, row) for row in reference_rates.values
    ])
    return pd.DataFrame(interpolated, index=reference_rates.index, columns=target_times)


def compute_RMSE(estimated_rates, reference_rates):
    """Root mean squared error between estimated and reference rates.

    RMSE = sqrt( 1/(n_pairs*n_timepoints) * sum_{i,j,t} (lambda_{ijt} - hat{lambda}_{ijt})^2 )

    Args:
        estimated_rates (pd.DataFrame): The estimated rates.
        reference_rates (pd.DataFrame): The reference rates (e.g. true rates).

    Returns:
        float: The RMSE, in units of the rates themselves.
    """
    check_alignment(estimated_rates, reference_rates)
    squared_error = (estimated_rates.values - reference_rates.values) ** 2
    return float(np.sqrt(np.mean(squared_error)))


def compute_KL(estimated_rates, reference_rates):
    """Average KL divergence between the coalescent time pdfs implied by the two rate sets.

    For a hazard lambda(t) with cumulative hazard Lambda(t), the implied density is
    f = lambda * exp(-Lambda). Integrating by parts, the KL between the true and
    estimated densities reduces to a functional of the hazards alone:

        KL = int f_true * ln(lambda_true / hat{lambda}) dt - int S_true * (lambda_true - hat{lambda}) dt

    Both terms are computed on the shared time grid with the correct dt. Where the
    true rate is exactly zero the density is zero, so the first term drops out and
    only the (non-negative) second term +S_true * hat{lambda} remains; this is handled
    with an explicit mask rather than by adding an epsilon inside the log, which would
    otherwise contribute a spurious 0 * ln(eps).

    Args:
        estimated_rates (pd.DataFrame): The estimated rates.
        reference_rates (pd.DataFrame): The reference rates (e.g. true rates).

    Returns:
        float: The KL divergence averaged over pairs (nats per pair).
    """
    check_alignment(estimated_rates, reference_rates)
    times = np.asarray(reference_rates.columns, dtype=float)
    dt = float(np.median(np.diff(times)))

    reference = reference_rates.values
    estimated = estimated_rates.values

    # survival and density implied by the reference rates on this grid
    cumulative_hazard = np.cumsum(reference, axis=1) * dt
    reference_survival = np.exp(-cumulative_hazard)
    reference_pdf = reference * reference_survival

    # log-ratio term, defined only where the reference rate (hence the density) is non-zero.
    # Where the reference is positive but the estimate has been clamped to zero the KL is
    # genuinely infinite, so refuse to report a finite number rather than silently flooring it.
    positive = reference > 0
    unsupported = positive & (estimated <= 0)
    if unsupported.any():
        raise ValueError(
            f"{unsupported.sum()} entries have a positive true rate but a zero estimated rate; "
            "the KL divergence is infinite there."
        )
    log_ratio = np.zeros_like(reference)
    log_ratio[positive] = np.log(reference[positive] / estimated[positive])
    term_1 = reference_pdf * log_ratio
    term_2 = -reference_survival * (reference - estimated)

    n_pairs = reference.shape[0]
    return float(np.sum(term_1 + term_2) * dt / n_pairs)


def check_alignment(estimated_rates, reference_rates):
    """Raise if two rate frames are not comparable pointwise."""
    if estimated_rates.shape != reference_rates.shape:
        raise ValueError(
            f"Shape mismatch: {estimated_rates.shape} vs {reference_rates.shape}."
        )
    if not np.array_equal(estimated_rates.index.values, reference_rates.index.values):
        raise ValueError("Estimated and reference rates must share the same pair index.")
    if not np.allclose(
        np.asarray(estimated_rates.columns, dtype=float),
        np.asarray(reference_rates.columns, dtype=float),
    ):
        raise ValueError("Estimated and reference rates must share the same time grid.")


def load_rates(path, target_columns):
    """Load a rates parquet and put it on the shared time grid and index naming."""
    rates = pd.read_parquet(path)
    rates.index.names = ["individual_1", "individual_2"]
    rates.columns = np.asarray(target_columns, dtype=float)
    return rates


def remove_link_offset(rates):
    """Subtract the 1/N_max floor imposed by the link function, clamping at zero.

    Applied only to reconstructed rates: the true rates and the input rates never
    passed through the link and so carry no such floor.

    Args:
        rates (pd.DataFrame): Reconstructed rates.

    Returns:
        pd.DataFrame: Rates with the offset removed and negatives set to zero.
    """
    return rates.clip(lower=RATE_OFFSET) - RATE_OFFSET


#%% build (or load) the summary table of RMSE / KL across d
if os.path.exists(SUMMARY_CSV):
    summary = pd.read_csv(SUMMARY_CSV)
    print(f"Loaded existing summary from {SUMMARY_CSV}")
else:
    true_rates_raw = pd.read_parquet(f"{SCAN_DIR}/noisy/data/rates_true.parquet")
    true_rates_raw.index.names = ["individual_1", "individual_2"]

    # the estimated-rate time grid is the common grid for every comparison
    grid = np.asarray(
        pd.read_parquet(f"{SCAN_DIR}/noisy/data/{NOISY_INPUT_FILE}").columns, dtype=float
    )
    true_rates = align_to_grid(true_rates_raw, grid)

    input_rates = {
        False: load_rates(f"{SCAN_DIR}/noisy/data/{NOISY_INPUT_FILE}", grid),
        True: load_rates(f"{SCAN_DIR}/smooth/data/{SMOOTH_INPUT_FILE}", grid),
    }

    records = []
    for smoothed_input, branch in [(False, "noisy"), (True, "smooth")]:
        for smoothed_embeddings, prefix in [(False, "reconstructed_rates"), (True, "smooth_reconstructed_rates")]:
            for d in D_VALUES:
                path = f"{SCAN_DIR}/{branch}/data/scan_d/{prefix}.d{d}.parquet"
                reconstructed = remove_link_offset(load_rates(path, grid))
                records.append({
                    "d": d,
                    "smoothed_input": smoothed_input,
                    "smoothed_embeddings": smoothed_embeddings,
                    "RMSE": compute_RMSE(reconstructed, true_rates),
                    "KL": compute_KL(reconstructed, true_rates),
                })

    summary = pd.DataFrame.from_records(records)
    summary.to_csv(SUMMARY_CSV, index=False)
    print(f"Wrote summary to {SUMMARY_CSV}")

    # baseline error of the input rates themselves, for the reference lines in the figures
    baseline = pd.DataFrame.from_records([
        {
            "smoothed_input": smoothed_input,
            "RMSE": compute_RMSE(input_rates[smoothed_input], true_rates),
            "KL": compute_KL(input_rates[smoothed_input], true_rates),
        }
        for smoothed_input in [False, True]
    ])
    baseline.to_csv(f"{DATA_DIR}/input_rates_baseline_error.csv", index=False)

baseline = pd.read_csv(f"{DATA_DIR}/input_rates_baseline_error.csv")
print(summary)
print(baseline)

#%% figures: one per embedding-smoothing setting, one panel per input type
RMSE_COLOR = okabe_ito[0]
KL_COLOR = okabe_ito[1]


def pad_axis(ax, curve_values, baseline_value, lower_pad):
    """Set y-limits spanning the curve and its baseline, with headroom below.

    The RMSE and KL baselines otherwise coincide visually on the twin axes; padding
    the two axes by different fractions separates the dotted lines.

    Args:
        ax (matplotlib.axes.Axes): Axis to rescale.
        curve_values (array-like): The plotted curve.
        baseline_value (float): The input-rate reference value.
        lower_pad (float): Padding below the minimum, as a fraction of the data range.
    """
    low = min(np.min(curve_values), baseline_value)
    high = max(np.max(curve_values), baseline_value)
    span = high - low
    ax.set_ylim(low - lower_pad * span, high + 0.08 * span)


def plot_scan(smoothed_embeddings, filename):
    """Two-panel figure of RMSE and KL against d, one panel per input rate type."""
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.4), sharex=True)

    for ax, smoothed_input in zip(axes, [False, True]):
        subset = summary[
            (summary["smoothed_input"] == smoothed_input)
            & (summary["smoothed_embeddings"] == smoothed_embeddings)
        ].sort_values("d")
        reference = baseline[baseline["smoothed_input"] == smoothed_input].iloc[0]

        # RMSE on the left axis, with the input-rate baseline as a dotted line
        line_rmse, = ax.plot(subset["d"], subset["RMSE"], "o-", color=RMSE_COLOR, markersize=4)
        base_rmse = ax.axhline(reference["RMSE"], color=RMSE_COLOR, linestyle=":", linewidth=1.5)
        ax.set_xlabel("$d$")
        ax.set_ylabel("RMSE", color=RMSE_COLOR)
        ax.tick_params(axis="y", colors=RMSE_COLOR)
        ax.set_xticks(D_VALUES)

        # KL on the right axis, same treatment in the contrasting colour
        ax_kl = ax.twinx()
        line_kl, = ax_kl.plot(subset["d"], subset["KL"], "s-", color=KL_COLOR, markersize=4)
        base_kl = ax_kl.axhline(reference["KL"], color=KL_COLOR, linestyle=":", linewidth=1.5)
        ax_kl.set_ylabel("KL divergence", color=KL_COLOR)
        ax_kl.tick_params(axis="y", colors=KL_COLOR)
        ax_kl.spines["right"].set_visible(True)
        ax_kl.spines["right"].set_color(KL_COLOR)
        ax.spines["left"].set_color(RMSE_COLOR)

        # the two baselines land on top of each other under the default twin scaling;
        # pad each axis by a different amount so both dotted lines stay separately readable
        pad_axis(ax, subset["RMSE"], reference["RMSE"], lower_pad=0.35)
        pad_axis(ax_kl, subset["KL"], reference["KL"], lower_pad=0.10)

        ax.set_title("Smooth input rates" if smoothed_input else "Noisy input rates")

    axes[0].legend(
        [line_rmse, line_kl, base_rmse, base_kl],
        ["RMSE (reconstructed)", "KL (reconstructed)", "RMSE (input rates)", "KL (input rates)"],
        loc="upper right", frameon=False, fontsize=8,
    )

    smoothing_label = "smoothed" if smoothed_embeddings else "unsmoothed"
    fig.suptitle(f"Reconstructed rates from {smoothing_label} embeddings", y=1.02)
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{filename}", bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUTPUT_DIR}/{filename}")


plot_scan(smoothed_embeddings=False, filename="reconstructed_rates_scan_d_unsmoothed_offset_removed.pdf")
plot_scan(smoothed_embeddings=True, filename="reconstructed_rates_scan_d_smoothed_offset_removed.pdf")
# %%
