#!/usr/bin/env python3
"""Infer tree sequences from simulated tree sequences with tsinfer, Relate or SINGER.

One contig in, one contig out: for each simulated ``contig_{i}.trees`` the script
overlays mutations with a fixed seed (i + 42), exports a VCF into a per-contig work
directory, runs the chosen ARG inference method and writes ``contig_{i}.trees`` into
the output directory. Nothing else is left in the output directory except the
sentinel file, so downstream ``*.trees`` globs only ever see final tree sequences.

Command lines and the fixes they encode (genetic-map units for Relate, calling the
``singer`` binary directly, rebuilding mutation parents after POLEGON) are ported
from the rate-bias benchmark driver, which ran 800 contigs without a failure.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.machinery
import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import time

import msprime
import numpy as np
import tskit

script_dir = os.path.dirname(os.path.abspath(__file__))
# The pipeline's relative paths (configs, data, results) are all anchored at utils/,
# which is one level above this script since the move into snakemake_utils/.
utils_dir = os.path.dirname(script_dir)
os.chdir(utils_dir)

_SUFFIX = "_linux" if platform.system() == "Linux" else "_mac"
BIN_ROOT = os.path.join(utils_dir, "arg_infer_utils")
SINGER_DIR = os.path.join(BIN_ROOT, "singer" + _SUFFIX)
RELATE_DIR = os.path.join(BIN_ROOT, "relate" + _SUFFIX)
POLEGON_DIR = os.path.join(BIN_ROOT, "polegon_linux")

SINGER_POLAR = 0.99   # REF is the true ancestral allele in the simulated VCF


# ----------------------------------------------------------------- helpers
def run(cmd, cwd, log):
    """Run a command with its output appended to ``log``; fail loudly on non-zero exit."""
    cmd = [str(c) for c in cmd]
    with open(log, "a") as f:
        f.write(f"$ {' '.join(cmd)}\n")
        f.flush()
        res = subprocess.run(cmd, cwd=cwd, stdout=f, stderr=subprocess.STDOUT, text=True)
        f.write(f"  -> exit {res.returncode}\n")
    if res.returncode != 0:
        raise RuntimeError(f"command failed (exit {res.returncode}): {' '.join(cmd)}")


def mutate(ts, i, mu):
    """Overlay mutations with a per-contig fixed seed, then restrict to sites every
    method consumes identically: sites carrying more than one mutation (recurrent /
    multi-allelic; Relate's haps format is strictly biallelic and SINGER assumes
    biallelic sites) and a site at position 0 (VCF is 1-based) are dropped."""
    mts = msprime.sim_mutations(ts, rate=mu, random_seed=i + 42)
    n_per_site = np.bincount(mts.mutations_site, minlength=mts.num_sites)
    bad = np.flatnonzero((n_per_site != 1) | (mts.sites_position == 0))
    return mts.delete_sites(bad) if len(bad) else mts


def write_vcf(mts, path, contig_id):
    with open(path, "w") as f:
        mts.write_vcf(f, contig_id=contig_id)


def estimate_Ne(mts, mu):
    """Data-only diploid Ne from site diversity: pi = 4 Ne mu. This is what
    singer_master computes from the VCF and what Relate's --effectiveN (haploid,
    hence 2*Ne) needs; using one value for both keeps the methods comparable."""
    return float(mts.diversity(mode="site") / (4 * mu))


def finalize(src, dest, expect_samples):
    """Validate and move a finished tree sequence into place atomically."""
    ts = tskit.load(src)
    if ts.num_samples != expect_samples:
        raise RuntimeError(f"{src}: {ts.num_samples} samples, expected {expect_samples}")
    tmp = dest + ".tmp"
    ts.dump(tmp)
    os.replace(tmp, dest)
    return ts


# ----------------------------------------------------------------- methods
def infer_tsinfer(mts, i, wd, log, args):
    import tsdate
    import tsinfer
    vdata = tsinfer.SampleData.from_tree_sequence(mts)
    its = tsinfer.infer(vdata, recombination_rate=args.r, num_threads=1).simplify()
    dts = tsdate.variational_gamma(its, mutation_rate=args.mu)
    out = os.path.join(wd, f"contig_{i}.trees")
    dts.dump(out)
    return out


def infer_relate(mts, i, wd, log, args):
    cid = f"contig_{i}"
    vcf_prefix = os.path.join(wd, cid)
    write_vcf(mts, vcf_prefix + ".vcf", str(i))
    L = mts.sequence_length
    # Genetic map: uniform rate. Rate column is cM/Mb, Map column cumulative cM.
    with open(os.path.join(wd, f"{cid}.map"), "w") as f:
        f.write("Position.bp.\tRate.cM.Mb.\tMap.cM.\n")
        f.write(f"0\t{100 * args.r * 1e6}\t0\n")
        f.write(f"{L:.0f}\t0\t{100 * args.r * L}\n")
    run([os.path.join(RELATE_DIR, "RelateFileFormats"), "--mode", "ConvertFromVcf",
         "--input", vcf_prefix, "--haps", f"{cid}.haps", "--sample", f"{cid}.sample"], wd, log)
    # Relate writes into cwd; keep it inside the work dir.
    run([os.path.join(RELATE_DIR, "Relate"), "--mode", "All", "-m", args.mu,
         "--effectiveN", int(round(2 * estimate_Ne(mts, args.mu))),
         "--haps", f"{cid}.haps", "--sample", f"{cid}.sample", "--map", f"{cid}.map",
         "--seed", i + 1, "--output", cid], wd, log)
    run([os.path.join(RELATE_DIR, "RelateFileFormats"), "--mode", "ConvertToTreeSequence",
         "-i", cid, "-o", cid], wd, log)
    return os.path.join(wd, f"{cid}.trees")


def _polegon(singer_trees, i, wd, log, mu):
    """Re-date a SINGER draw with POLEGON (posterior-average ARG). The binary is run
    here rather than through polegon_master, which prints "polegon failed" and exits 0."""
    if platform.system() != "Linux":
        raise RuntimeError("POLEGON is only distributed for Linux; rerun with --no_polegon")
    in_prefix = os.path.join(wd, f"contig_{i}_pg")
    out_prefix = in_prefix + "_polegon"
    loader = importlib.machinery.SourceFileLoader(
        "polegon_master", os.path.join(POLEGON_DIR, "polegon_master"))
    spec = importlib.util.spec_from_loader("polegon_master", loader)
    pm = importlib.util.module_from_spec(spec)
    loader.exec_module(pm)
    old_ts = tskit.load(singer_trees)
    if old_ts.num_mutations == 0:
        raise RuntimeError("SINGER tree sequence carries no mutations; POLEGON needs them")
    pm.write_arg(old_ts, f"{in_prefix}_nodes.txt", f"{in_prefix}_branches.txt")
    pm.write_muts(old_ts, f"{in_prefix}_muts.txt")
    Ne = old_ts.diversity(mode="site") / 2 / mu   # polegon_master's own convention
    run([os.path.join(POLEGON_DIR, "polegon"), "-m", mu, "-Ne", Ne, "-burn_in", 100,
         "-num_samples", 100, "-thin", 10, "-scaling_rep", 3, "-max_step", 10,
         "-input", in_prefix, "-output", out_prefix], wd, log)
    # polegon_master.read_mutation never sets mutation parents, so a site carrying
    # more than one mutation is rejected with TSK_ERR_BAD_MUTATION_PARENT. Rebuild
    # with the standard sort + compute_mutation_parents so every mutation is kept.
    tables = pm.read_ARG(f"{in_prefix}_new_nodes.txt", f"{in_prefix}_branches.txt")
    pm.read_mutation(tables, f"{in_prefix}_muts.txt")
    tables.sort()
    tables.build_index()
    tables.compute_mutation_parents()
    out = f"{out_prefix}.trees"
    tables.tree_sequence().dump(out)
    return out


def infer_singer(mts, i, wd, log, args):
    cid = f"contig_{i}"
    prefix = os.path.join(wd, cid)
    write_vcf(mts, prefix + ".vcf", str(i))
    # n MCMC samples spaced by --thinning_interval cover the burn-in; only the last
    # one (index n-1) is converted and kept.
    n = int(np.ceil(args.burnin / args.thinning_interval))
    # The binary is called directly: singer_master's auto-debug loop retries up to
    # 100 seeds on failure, which would hide crashes.
    run([os.path.join(SINGER_DIR, "singer"), "-Ne", estimate_Ne(mts, args.mu),
         "-m", args.mu, "-r", args.r, "-input", prefix, "-output", prefix,
         "-start", 0, "-end", f"{mts.sequence_length:.0f}", "-polar", SINGER_POLAR,
         "-n", n, "-thin", args.thinning_interval, "-seed", i + 1], wd, log)
    last = n - 1
    run([sys.executable, os.path.join(SINGER_DIR, "convert_to_tskit"), "-input", prefix,
         "-output", prefix, "-start", last, "-end", n, "-step", 1], wd, log)
    out = f"{prefix}_{last}.trees"
    return _polegon(out, i, wd, log, args.mu) if args.polegon else out


METHODS = {"tsinfer": infer_tsinfer, "relate": infer_relate, "singer": infer_singer}


# ----------------------------------------------------------------- driver
def infer_contig(i, args):
    dest = os.path.join(args.output_path, f"contig_{i}.trees")
    if os.path.exists(dest):
        return f"contig_{i}: exists, skipped"
    root = os.path.dirname(os.path.abspath(args.output_path))
    # Absolute: the external binaries are run with cwd set to the work directory,
    # so every path handed to them has to resolve independently of that cwd.
    wd = os.path.join(root, "work", f"contig_{i}")
    log_dir = os.path.join(root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    shutil.rmtree(wd, ignore_errors=True)   # discard a previous failed attempt
    os.makedirs(wd)
    log = os.path.join(log_dir, f"contig_{i}.log")
    t0 = time.time()
    mts = mutate(tskit.load(os.path.join(args.sim_dir, f"contig_{i}.trees")), i, args.mu)
    ts = finalize(METHODS[args.method](mts, i, wd, log, args), dest, mts.num_samples)
    shutil.rmtree(wd)
    return (f"contig_{i}: {ts.num_trees} trees, {ts.num_samples} samples, "
            f"L={ts.sequence_length:.0f}, {ts.num_sites} sites, {(time.time() - t0) / 60:.1f} min")


def main():
    p = argparse.ArgumentParser(description="Infer tree sequences from simulated tree sequences")
    p.add_argument("--sim_dir", required=True, help="directory holding simulated contig_{i}.trees")
    p.add_argument("--output_path", required=True, help="directory receiving inferred contig_{i}.trees")
    p.add_argument("--output_name", required=True, help="sentinel file written into output_path when done")
    p.add_argument("--method", required=True, choices=tuple(METHODS))
    p.add_argument("--num_contigs", type=int, default=None,
                   help="infer contig_0 .. contig_{n-1} (default: all contig_*.trees in sim_dir)")
    p.add_argument("--mu", type=float, default=1.2e-8, help="mutation rate")
    p.add_argument("--r", type=float, default=1.2e-8, help="recombination rate")
    p.add_argument("--burnin", type=int, default=1000, help="SINGER: total burn-in iterations")
    p.add_argument("--thinning_interval", type=int, default=20, help="SINGER: -thin")
    p.add_argument("--polegon", dest="polegon", action="store_true", default=True,
                   help="SINGER: re-date the draw with POLEGON (default)")
    p.add_argument("--no_polegon", dest="polegon", action="store_false",
                   help="SINGER: keep the raw SINGER draw")
    p.add_argument("--num_threads", type=int, default=1, help="contigs inferred in parallel")
    args = p.parse_args()

    n = args.num_contigs
    if n is None:
        n = len([f for f in os.listdir(args.sim_dir)
                 if f.startswith("contig_") and f.endswith(".trees")])
    if n == 0:
        raise ValueError(f"no contig_*.trees found in {args.sim_dir}")
    os.makedirs(args.output_path, exist_ok=True)
    print(f"inferring {n} contigs from {args.sim_dir} with {args.method}"
          f"{'+polegon' if args.method == 'singer' and args.polegon else ''}", flush=True)

    t0 = time.time()
    if args.num_threads == 1:
        for i in range(n):
            print(infer_contig(i, args), flush=True)
    else:
        with concurrent.futures.ProcessPoolExecutor(args.num_threads) as ex:
            futures = [ex.submit(infer_contig, i, args) for i in range(n)]
            for fut in futures:
                print(fut.result(), flush=True)   # fail fast on the first exception
    took = time.time() - t0

    with open(os.path.join(args.output_path, args.output_name), "w") as f:
        f.write(f"inference took {took / 60:.2f} minutes")


if __name__ == "__main__":
    main()
