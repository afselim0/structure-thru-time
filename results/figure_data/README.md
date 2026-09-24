# Prepared figure data

One subdirectory per figure, holding the tables that figure is drawn from. Nothing
here is edited by hand.

The split is: **preparation** reads raw run outputs from `results/outputs/` (or runs
its own simulation/optimization) and writes these tables; **plotting** reads only
these tables and draws. So a script in `results/figures/plotting_scripts/` never
touches `results/outputs/`, never runs the optimizer, and never Procrustes-aligns
anything — all of that has already happened here.

## Regenerating

```
bash utils/runs/prepare_all_figure_data.sh   # every figure
python utils/runs/prepare_figure_data/<figure_name>.py           # just one
```

Each figure's prep script is `utils/runs/prepare_figure_data/<figure_name>.py`, named
for the directory it writes. A figure whose pipeline run is not on this machine fails
with `missing run: <path>`; produce that run with the matching config under
`data/configs/` first.

## Conventions

- `.parquet` for the large numeric tables (zstd-compressed, full precision), `.csv`
  for the small label, colour and parameter tables.
- Every prep script that simulates or optimizes fixes its seeds and records them in
  its `params.csv`, so a figure can be reproduced exactly.
- `params.csv` also carries the plotting constants the figure needs but should not
  re-derive (time windows, `N_max`, chosen timepoints), as `parameter,value` rows.
- Times are in generations unless a column says otherwise.

## Tracking

The small `.csv` and `.yaml` tables — labels, colours, `params.csv` — are checked
in, since a figure cannot be drawn without them. The `.parquet` tables are not:
they are regenerable from `results/outputs/` by the matching prep script, so
`.gitignore` ignores `*.parquet` outright. Rerun the prep script for a figure
whose tables are missing.

## Contents

| Directory | Prep script | Holds |
| --- | --- | --- |
| `anomaly_zone/` | `anomaly_zone.py` | Probability of each four-taxon topology under the anomaly-zone demography, from seeded msprime replicates. |
| `arg_rates_comparison/` | `arg_rates_comparison.py` | Per-pair and mean within/between rates for split and sawtooth from the true model, true trees, tsinfer+tsdate, Relate and SINGER+POLEGON; the between/within mean-rate ratio and the time each inferred source reaches 0.5. |
| `avg_distance_vs_Ne/` | `avg_distance_vs_Ne.py` | Average embedded pairwise distance, hull radius and the link's inverse of 1/Ne, over an Ne grid for each link/p. |
| `avg_distance_vs_Nmin_n/` | `avg_distance_vs_Nmin_n.py` | Average embedded pairwise distance sweeping `N_min` and sweeping the sample count, at several Ne. |
| `badmix_ancestry/` | `badmix_ancestry.py` | Per-ancestry (B, C, D) embeddings for those three examples under GNN and RFmix, all aligned to one reference panel. |
| `badmix_WG/` | `badmix_WG.py` | Whole-genome embeddings for the three hard admixture examples, aligned to the `recent_bottleneck` frame, plus labels and the shared population palette. |
| `chimp_dataset/` | `chimp_dataset.py` | The chimpanzee dataset: sample origins and Natural Earth outlines for the map, the genotype PCA, and the per-subspecies Ne(t) and per-pair relative diversity, each as a mean curve plus every individual pair's trajectory. The trajectory tables are thinned in time; the mean curves are not. |
| `coal_rate_mig_rate/` | `coal_rate_mig_rate.py` | Within/between coalescence-rate trajectories at four Nm, and the plateau and rate constant over a finer migration-rate grid. |
| `complex_4_pops/` | `complex_4_pops.py` | Embeddings, estimated and reconstructed rates, and the p scan for the four-population example. |
| `complex_4_pops_migration/` | `complex_4_pops_migration.py` | The same example at three migration rates, aligned to the no-migration run, with the analytic rates and the 1/(2Ne) baseline. |
| `double_admix_GNN_RFmix/` | `double_admix_GNN_RFmix.py` | Per-ancestry embeddings, rates and shared rate limits for the two-admixture example under GNN and RFmix. |
| `double_admix_singer/` | `double_admix_singer.py` | The two-admixture example on SINGER trees under GNN and RFmix local ancestry: per-ancestry embeddings aligned onto the simulated-tree GNN frames, rates and shared rate limits. |
| `double_admix_true/` | `double_admix_true.py` | Whole-genome embeddings and rates plus the four true-ancestry embeddings for the same example. |
| `double_SS_2d/` | `double_SS_2d.py` | Smoothed embeddings, shifted estimated/reconstructed rates and each pair's within/between group type for the two-group 2D stepping stone. **Needs the smoothed (`l1`) run, which is not on disk.** |
| `GaussianvsCauchyKernels/` | `GaussianvsCauchyKernels.py` | The two kernels sampled on a distance grid, and the distance range each maps the "moderate similarity" band to. |
| `model_description/` | `model_description.py` | Sawtooth-model true rates, seeded embeddings through time, pairwise distances, and the focal pair the schematic follows. |
| `panmictic_diff_links/` | `panmictic_diff_links.py` | One panmictic embedding per link/p combination. |
| `sim_vs_singer/` | `sim_vs_singer.py` | Split, sawtooth, four-population and two-group 2D stepping-stone embeddings on simulated versus SINGER-inferred trees, SINGER aligned onto simulated per demography. |
| `split_different_links/` | `split_different_links.py` | Embeddings of separated populations of differing size, per link/p and at two `N_max`. |
| `SS_examples/` | `SS_examples.py` | The 1D and 2D stepping-stone embeddings, six per figure (2 values of p x 3 migration rates), aligned within each p. |
| `two_pop_migration/` | `two_pop_migration.py` | Embeddings for a two-population split under recent migration and under a soft split, at four Nm, plus the demes files they were built from. |
