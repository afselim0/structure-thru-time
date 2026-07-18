#!/usr/bin/env python3
"""Set up symlink directories and config files for low-res runs across all
complete SINGER posterior samples (indices 0–19, 389 segments each).

Creates:
  data/trees/chimps_lowres_{k}/              — 389 symlinks to all_chopped/*_{k}.trees
  data/empirical_runs/chimps_lowres/PS{k}.yaml  — config for each k
  data/empirical_runs/chimps_lowres/README.md   — explains the setup

Template: data/empirical_runs/chimps_PS0_lowres.yaml
Output results go to: results/empirical/DeManuel/chimps_lowres/PS{k}/
"""
import glob
import os
import re

SRC_DIR     = "/project/jnovembre/afselim/structure-thru-time/data/trees/DeManuel/all_chopped"
BASE        = "/project2/jnovembre/mianniravn/stt-0526/structure-thru-time"
TREES_BASE  = os.path.join(BASE, "data", "trees")
CONFIGS_DIR = os.path.join(BASE, "data", "empirical_runs", "chimps_lowres")
TEMPLATE    = os.path.join(BASE, "data", "empirical_runs", "chimps_PS0_lowres.yaml")

COMPLETE_INDICES = list(range(100))  # 0–99; coverage decreases with index (389→281 segments)

with open(TEMPLATE) as f:
    template_text = f.read()

os.makedirs(CONFIGS_DIR, exist_ok=True)

for k in COMPLETE_INDICES:
    # ── symlink directory ────────────────────────────────────────────────────
    dest_dir = os.path.join(TREES_BASE, f"chimps_lowres_{k}")
    os.makedirs(dest_dir, exist_ok=True)

    for old in glob.glob(os.path.join(dest_dir, "*.trees")):
        os.remove(old)

    src_files = [
        f for f in glob.glob(os.path.join(SRC_DIR, f"*_{k}.trees"))
        if re.search(rf"_{k}\.trees$", os.path.basename(f))
    ]
    for src in sorted(src_files):
        os.symlink(src, os.path.join(dest_dir, os.path.basename(src)))

    print(f"chimps_lowres_{k}: {len(src_files)} symlinks")

    # ── config file (text substitution to preserve comments/order) ──────────
    cfg_text = template_text
    cfg_text = re.sub(
        r"^output_path:.*$",
        f"output_path: ../results/empirical/DeManuel/chimps_lowres/PS{k}",
        cfg_text, flags=re.MULTILINE,
    )
    cfg_text = re.sub(
        r"^trees_dir:.*$",
        f"trees_dir: ../data/trees/chimps_lowres_{k}",
        cfg_text, flags=re.MULTILINE,
    )
    cfg_text = re.sub(
        r"^posterior_samp:.*$",
        f"posterior_samp: {k}",
        cfg_text, flags=re.MULTILINE,
    )

    out_cfg = os.path.join(CONFIGS_DIR, f"PS{k}.yaml")
    with open(out_cfg, "w") as f:
        f.write(cfg_text)

# ── README ───────────────────────────────────────────────────────────────────
readme = """\
# chimps_lowres — low-resolution CoaSTT runs across all 100 SINGER posterior samples

## What this is
Low-resolution (max_time=10000, n_timepoints=400) CoaSTT runs on all 100
SINGER posterior sample indices (0–99). Segment coverage decreases with index:

| Indices | Segments |
|---------|----------|
| 0–19    | 389      |
| 20–42   | 361      |
| 43–77, 87 | 328    |
| 78–86   | 313      |
| 88–95, 99 | 296    |
| 96–98   | 281      |

## How configs were generated
`utils/setup_chimps_lowres.py` reads `chimps_PS0_lowres.yaml` as a template and
produces `PS{k}.yaml` for k=0..99, varying:
  - `posterior_samp`  — posterior sample index (label only; pipeline uses trees_dir)
  - `trees_dir`       — ../data/trees/chimps_lowres_{k}/
  - `output_path`     — ../results/empirical/DeManuel/chimps_lowres/PS{k}/

Symlink directories `data/trees/chimps_lowres_{k}/` contain N symlinks each,
pointing to the corresponding `*_{k}.trees` files in the SINGER all_chopped dir.

## How to run
```bash
# 1. Create symlinks + configs (idempotent, safe to re-run)
python utils/setup_chimps_lowres.py

# 2. Submit SLURM array (jobs 0–19)
cd utils && sbatch slurm_chimps_lowres.sh
```

## Key parameters (from template)
| Parameter        | Value   |
|------------------|---------|
| max_time         | 10000   |
| n_timepoints     | 400     |
| n_timepoints_emb | 1000    |
| min_time         | 1000    |
| min_window_size  | 200     |
| N_min            | 100     |
| embedding_seed   | 0       |
| pops             | Western, Eastern, Central, NC |

## Output location
`results/empirical/DeManuel/chimps_lowres/PS{k}/`
"""
with open(os.path.join(CONFIGS_DIR, "README.md"), "w") as f:
    f.write(readme)

print(f"\n{len(COMPLETE_INDICES)} config files written to {CONFIGS_DIR}/")
print(f"README written to {CONFIGS_DIR}/README.md")
print("Next: cd utils && sbatch slurm_chimps_lowres.sh")
