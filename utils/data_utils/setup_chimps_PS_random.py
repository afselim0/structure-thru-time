#!/usr/bin/env python3
"""Recreate chimps_PS{k} symlink directories using 8 randomly selected posterior
sample indices drawn from the 20 complete ones (0–19, all with 389 segments).

Random seed: 42  →  selected = [1, 7, 9, 10, 13, 15, 16, 18]
  chimps_PS0 → posterior sample 1
  chimps_PS1 → posterior sample 7
  chimps_PS2 → posterior sample 9
  chimps_PS3 → posterior sample 10
  chimps_PS4 → posterior sample 13
  chimps_PS5 → posterior sample 15
  chimps_PS6 → posterior sample 16
  chimps_PS7 → posterior sample 18

Run this BEFORE submitting SLURM jobs, and after deleting old results.
"""
import glob
import os
import re
import numpy as np

SRC_DIR    = "/project/jnovembre/afselim/structure-thru-time/data/trees/DeManuel/all_chopped"
TREES_BASE = "/project2/jnovembre/mianniravn/stt-0526/structure-thru-time/data/trees"
N_PS       = 8

rng      = np.random.default_rng(42)
selected = sorted(rng.choice(20, size=N_PS, replace=False).tolist())

print("Posterior sample index selection (rng_seed=42, pool=0-19):")
for k, ps_idx in enumerate(selected):
    print(f"  chimps_PS{k} -> posterior sample {ps_idx}")
print()

for k, ps_idx in enumerate(selected):
    dest_dir = os.path.join(TREES_BASE, f"chimps_PS{k}")
    os.makedirs(dest_dir, exist_ok=True)

    # Remove all existing .trees symlinks
    old_links = glob.glob(os.path.join(dest_dir, "*.trees"))
    for link in old_links:
        os.remove(link)

    # Create new symlinks for the chosen posterior index
    src_files = sorted(glob.glob(os.path.join(SRC_DIR, f"*_{ps_idx}.trees")))
    # Filter: only files whose posterior index suffix exactly matches ps_idx
    exact = [f for f in src_files
             if re.search(rf"_{ps_idx}\.trees$", os.path.basename(f))]
    for src in exact:
        link_name = os.path.join(dest_dir, os.path.basename(src))
        os.symlink(src, link_name)

    print(f"chimps_PS{k}: {len(exact)} symlinks -> posterior sample {ps_idx}")

print("\nDone. Remember to delete old result directories before rerunning:")
print("  rm -rf ../results/empirical/DeManuel/chimps_PS{0..7}")
