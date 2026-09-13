#!/usr/bin/env bash
# STEP 7 -- verify the assembled inputs. THE GATE BEFORE INFERENCE.
#
#   bash drivers/7_verify.sh
#
# Needs step 6. Writes nothing; exits non-zero if any check fails.
#
# Do not skip this. Step 8 spends about a day of compute, and the failures this catches are
# the ones that are otherwise silent: the workflow asserts EXACT equality between hapmap
# length, ancestral FASTA length and chrom.sizes, and would fail hours in. An unphased input
# is worse still -- right length, right positions, right samples, simply wrong -- and one
# build once passed 13/13 checks while carrying 91.79% unphased genotypes.
#
# 14 checks per chromosome. See scripts/verify_inputs.py for what each one covers.
#
# 23-task array, ~10-20 min per task.
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"
submit_and_wait verify_all.sbatch
echo
echo "Step 7 complete -- inputs verified. Next: bash drivers/8_run_singer.sh"
