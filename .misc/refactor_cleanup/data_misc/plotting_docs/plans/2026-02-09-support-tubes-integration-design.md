# Support Tubes Integration Design

## Goal

Move `support_tube.py` to `utils/`, wire it into the Snakemake pipeline as a separate rule, write documentation, and verify end-to-end with a smoke test.

## Decisions Made

- **File move**: `src/support_tube.py` -> `utils/support_tube.py`
- **Pipeline integration**: Separate Snakemake rule (`plot_support_tubes`), not added to default targets (opt-in only)
- **Geometry**: Both mesh surface and ring outlines included
- **Documentation**: New `docs/support-tubes.md`
- **Smoke test**: End-to-end run against `results/simulations/outputs/split/` (has existing embeddings + labels + colors)
- **Minimal footprint**: No changes to existing rules or `utils/plot_embeddings.py`

## Implementation Steps

### Step 1: Move `support_tube.py` to `utils/`

- `git mv src/support_tube.py utils/support_tube.py`
- Delete `src/Support_Tubes_README.md` (replaced by `docs/support-tubes.md`)

### Step 2: Update imports

**`src/PlotEmbeddings.py`** (two locations: `add_support_tube` and `add_support_rings`):
- Change `from src import support_tube` to resolve from repo root -> `from utils import support_tube`
- The lazy import already lives inside the method body, so we just change the import path

**`sandbox/convex_hull_sandbox.py`**:
- Already uses `sys.path.insert(0, str(REPO_ROOT))`, so changing `from src.support_tube import ...` (if any direct imports) or verifying the indirect import through `PlotEmbeddings` still works

### Step 3: Create `utils/plot_support_tubes.py` (CLI wrapper) 

Thin CLI script following the pattern of `utils/plot_embeddings.py`:

```
Arguments:
  --input           Path to embeddings parquet
  --colors          Path to pop_colors CSV
  --labels          Path to pop_labels CSV
  --output_path     Output directory
  --output_name     Output filename (HTML)
  --min_time        Minimum time for plotting (default: 0)

  # Tube params (with defaults matching DEFAULT_TUBE_PARAMS)
  --grid_res_xy     XY grid resolution (default: 256)
  --grid_res_u      Time-axis grid resolution (default: 256)
  --mass_frac       Fraction of population mass captured (default: 0.90)
  --sigma_u         Temporal smoothing (default: 1.0)
```

The script:
1. Reads inputs (embeddings, colors, labels parquet/CSV)
2. Creates `PlotEmbeddings` instance
3. Calls `init_plot(time_scale="log")`
4. Calls `add_sample_trajectories()` + `add_population_medians()`
5. Calls `add_support_tube()` and `add_support_rings()` with the CLI params
6. Calls `add_legend()`
7. Saves HTML output

### Step 4: Add Snakemake rule

In `results/empirical/snakefile`:

- Add config params for tube settings (with defaults)
- Define output path: `support_tubes_plot = os.path.join(plot_output_path, f"support_tubes{EMBEDDINGS_PLOT_TAG}.html")`
- Add `PLOT_SUPPORT_TUBES_SCRIPT = "../../utils/plot_support_tubes.py"`
- Add rule `plot_support_tubes` that depends on `embeddings_data`, `pop_colors`, `pop_labels`
- **Not** added to `BASE_TARGETS` or `COMB_TARGETS` (opt-in via `snakemake plot_support_tubes`)

### Step 5: Write documentation (`docs/support-tubes.md`)

Sections:
1. **What is a support tube** - Conceptual overview (cleaned up from existing README)
2. **Pipeline usage** - How to run via Snakemake, config params
3. **Parameter reference** - Table grouped by importance:
   - Primary: `grid_res_xy`, `grid_res_u`, `mass_frac`, `sigma_u`
   - Secondary: `tau`, `trim_center`, `min_points_per_slice`, `time_eps`
   - Advanced: `post_smooth_u/xy`, `close_r_u/xy`, ring binning params
4. **Architecture** - Where code lives, module responsibilities

### Step 6: End-to-end smoke test

Run the wrapper script directly against existing test data:

```bash
python utils/plot_support_tubes.py \
  --input results/simulations/outputs/split/data/embeddings_true.N01.parquet \
  --colors results/simulations/outputs/split/data/pop_colors.csv \
  --labels results/simulations/outputs/split/data/pop_labels.csv \
  --output_path results/simulations/outputs/split/plots \
  --output_name support_tubes_test.html
```

Verify:
- Script exits cleanly (no Python errors)
- Output HTML file exists and is non-trivial (> 10KB)
- Open the HTML in browser to visually confirm tubes render

## Files Changed

| File | Action |
|------|--------|
| `src/support_tube.py` | Move to `utils/support_tube.py` |
| `src/Support_Tubes_README.md` | Delete (replaced by docs) |
| `src/PlotEmbeddings.py` | Update import path (2 locations) |
| `sandbox/convex_hull_sandbox.py` | Verify imports still work |
| `utils/plot_support_tubes.py` | **New** - CLI wrapper |
| `results/empirical/snakefile` | Add rule + config params |
| `docs/support-tubes.md` | **New** - documentation |

## What We're NOT Changing

- `utils/plot_embeddings.py` - untouched
- Existing Snakemake rules - untouched
- `BASE_TARGETS` / `COMB_TARGETS` - untouched
- `src/plot_binned_mask_hulls_3d.py` - untouched
