# Chimpanzee panel for SINGER

Builds the input files that [`singer-snakemake`](https://github.com/nspope/singer-snakemake)
needs to infer ancestral recombination graphs from the Published_Pan chimpanzee callset, and
runs the inference. The workflow clone lives at `../singer-snakemake`.

## The panel

| | |
|---|---|
| Samples | 54 chimpanzees (7 bonobos removed from the 61-sample callset) |
| Subspecies | *schweinfurthii* 19, *troglodytes* 15, *verus* 14, *ellioti* 5, plus one *verus* × *troglodytes* hybrid |
| Assembly | panTro6, 23 autosomes — `chr2A` and `chr2B` replace `chr2`, so 23 sequences rather than 22 |
| Source | raw GATK callset: unphased, unfiltered, variant-only, no published accessibility mask |

This is the sibling of [`../human_data/`](../human_data/), but starting from much rougher
material. The HGDP prep begins with a curated, phased, quality-filtered public release.
Everything that prep could download, this one has to build.

The hybrid (`SAMN01920541`) is deliberately kept. It gets its own `subspecies` value, so
`stratify-by: subspecies` produces a one-sample stratum: **every cross-coalescence pair
involving it is noise and should be ignored in plots.** The trade is worth it — it is the
panel's only *verus* × *troglodytes* individual, so keeping it means admixture between those
subspecies is measured rather than removed by construction.

## Running the pipeline

Nine steps, in order, from `data/chimp_data/`. Steps 1–7 block until their SLURM array
finishes, so they can be chained; steps 8 and 9 are described below.

```bash
conda activate CoaSTT_env
cd data/chimp_data

bash drivers/1_prep_vcf.sh         # split by chromosome, drop bonobos, apply site filters
bash drivers/2_genmap_index.sh     # genome-wide mappability index      (~40 min, once)
bash drivers/3_build_resources.sh  # recombination maps + ancestral FASTAs
bash drivers/4_build_masks.sh      # accessibility masks
bash drivers/5_phase.sh            # SHAPEIT5 phasing                   (longest prep step)
bash drivers/6_assemble.sh         # assemble SingerSnakemake_Input/
bash drivers/7_verify.sh           # per-chromosome invariant checks — the gate
bash drivers/8_run_singer.sh       # SINGER + POLEGON, one job per chromosome
bash drivers/9_extract_chunks.sh   # per-chunk .trees for CoaSTT        (minutes)
```

Steps 2 and 3 do not depend on step 1 and can be submitted at the same time. Step 4 needs
1 and 2; everything from 5 on is sequential. **Do not skip step 7** — it is the gate that
catches an unphased or mislabelled input before step 8 spends a day on it.

Step 8 submits one job per chromosome and returns immediately. It defaults to the dedicated
`jnovembre` partition; pass `--partition caslake` for the shared one.

```bash
bash drivers/8_run_singer.sh --chrom chr21          # a single chromosome
bash drivers/8_run_singer.sh --partition caslake    # shared partition
bash drivers/8_run_singer.sh -n                     # dry run: print, submit nothing
```

Everything runs on SLURM (`account=pi-jnovembre`). The login session is capped at an 8 GiB
cgroup, which is not enough for phasing, whole-chromosome `read_vcf`, or even the verification
script — all were OOM-killed there during development. Submit rather than running
interactively.

## How the raw VCF became these files

```
resources/vcf/full.vcf.gz                    28 GB, 61 samples, GenBank accession names
        │
        │  bcftools view -S chimps.txt -a  |  bcftools view -c 1
        ▼
   drop the 7 bonobos, then drop sites no chimpanzee carries
        │
        │  bcftools annotate --rename-chrs      CM009259.2 → chr21
        ▼
   ── site filters ────────────────────────────────── scripts/filter_sites.py
        │  GATK hard filters, allelic imbalance, depth, non-biallelic
        │  removed from the VCF, recorded in filter.txt
        ▼
   ── phasing ───────────────────────────────────── SHAPEIT5 phase_common
        ▼
   ── assembly ────────────────────────────────── scripts/assemble_inputs.py
        │  meta.csv in VCF order, shared resources linked in
        ▼
   SingerSnakemake_Input/chrN.{vcf.gz,tbi,filter.txt,meta.csv,mask.bed,hapmap,ancestral.fa.gz}
        │
        │  ── ARG inference ──────────────────── drivers/8_run_singer.sh
        ▼
   results/chimp/singer-snakemake_outputs/chrN/trees/chrN.<draw>.tsz
        │
        │  ── packaging ───────────────────── drivers/9_extract_chunks.sh
        ▼
   results/chimp/ts_chunks/chrN_<start>_<end>.trees      flat — what CoaSTT reads
```

Three resources are built separately, once per chromosome, and linked in:

```
scripts/build_hapmap.py           Stevison map:  hg18 → hg38 → panTro6
scripts/build_ancestral_fasta.py  hg38 bases projected through the panTro6→hg38 chain
scripts/build_mask.py             genmap mappability ∪ assembly gaps
```

## Site filters

Four families. The first three all target the same dominant failure mode in a non-model
assembly — **collapsed paralogues and mismapping**, which masquerade as high-diversity
regions, and that is precisely the signal CoaSTT reads.

<!-- FILTER_TABLE_START -->
Genome-wide across all 23 autosomes, from 49,368,272 called sites:

| filter | sites removed | % of input |
|---|---|---|
| GATK hard filters | 2,699,496 | 5.47% |
| Allelic imbalance | 3,304,436 | 6.69% |
| Depth | 813,363 | 1.65% |
| Non-biallelic SNPs | 2,833,183 | 5.74% |
| **Union** | **6,467,948** | **13.10%** |

42,900,324 sites survive. The union is far smaller than the sum of the parts — 9,650,478
against 6,467,948, so **3,182,530 sites are caught by more than one filter.** That overlap is
the reassuring result: independent criteria converging on the same sites is what one
underlying artefact looks like, whereas filters each removing different sites would suggest
they were cutting real variation.
<!-- FILTER_TABLE_END -->

- **GATK hard filters** — `QD<2 | FS>60 | MQ<40 | SOR>3 | ReadPosRankSum<-8 | MQRankSum<-12.5`.
  Best Practices thresholds, human-tuned, not re-derived for chimpanzee. Two are effectively
  inert here: `ReadPosRankSum` removes nothing and `MQRankSum` almost nothing; `QD` and `SOR`
  do the work.
- **Allelic imbalance** — pooled alternate-allele fraction across heterozygous calls outside
  `[0.25, 0.75]`. A true heterozygote sits near 0.5; a collapsed paralogue does not. The
  panel-wide distribution is healthy (median 0.4964 against a 0.5 expectation), so this cuts
  genuine outliers rather than the bulk.
- **Depth** — summed `FORMAT/DP` outside 0.25×–2× the chromosome median. Median rather than
  mean, because the mean is inflated by exactly the paralogue pileups the upper cut targets.
  Built on `FORMAT/DP`, never `INFO/DP`: the latter runs **8.7× higher** here, since it counts
  pre-filter reads across all 61 originally-called samples.
- **Non-biallelic SNPs** — anything that is not exactly one REF and one ALT SNP allele:
  multiallelics, indels, mixed records. SINGER builds haplotypes from biallelic SNPs only.

Filtering runs **before** phasing, and not by preference. SHAPEIT5 emits `GT` and nothing
else — it strips `AD`, `DP`, `GQ`, `PGT`, `PID`, `PL` and `PS` — so the depth and
allelic-imbalance filters cannot be computed on a phased VCF at all. It also measures better:
**3.59% switch error against 4.32%** for the reverse order, on the same chromosome with the
same metric.

### `filter.txt`

Every removed position is written to `chrN.filter.txt`, and the workflow uses it to scale
`adj_mu`. This file is load-bearing: a site removed but not recorded inflates the inferred
mutation rate, because the workflow then believes that sequence was surveyed and found
monomorphic.

Multiallelics are listed there **because** they are absent from the VCF. The workflow adds
multiallelics to `filtered_positions` itself, but only ones it can see — and it will see none.
There is no double-counting risk and a real under-counting one.

Positions in masked regions are the one exclusion: the workflow strips those from
`filtered_positions` anyway, so listing them is a no-op.

## Accessibility mask

`chrN.mask.bed` holds **INACCESSIBLE** intervals — storing the opposite polarity would invert
the analysis with no error raised anywhere. It is the union of two signals:

**Mappability**, from `genmap -K 36 -E 0`. genmap reports `1 / (times this k-mer occurs in the
assembly)`, so the threshold `< 1.0` means *this 36-mer is not unique* — the strictest
available criterion. The parameterisation was chosen by sweep, not assumed:

| setting | accessible |
|---|---|
| k=100, e=2 | 97.85% |
| k=50, e=0 | 97.29% |
| **k=36, e=0** | **94.97%** |

k=100 with 2 mismatches was rejected as degenerate: 97.85% of chr21 sat at *exactly*
mappability 1.0, and every threshold from `<0.1` to `<1.0` gave between 97.85% and 99.03%
accessible, so no threshold could rescue that track. k=36 exact is the strictest of the
settings tested, and among tracks that say roughly the same thing the one that errs toward
exclusion is preferable.

**Assembly gaps**, from the UCSC panTro6 gap track, keyed by RefSeq accession. In practice
this adds ~0 bp on top of mappability — runs of N have no unique k-mers, so mappability has
already excluded them. It is kept as an explicit guarantee rather than an inference.

<!-- MASK_TABLE_START -->
Genome-wide: **88.22% accessible** — 312,847,835 of 2,654,688,569 bp masked across 10,772,802
intervals. Per chromosome this ranges from 79.33% (chr19) to 90.94% (chr13).
<!-- MASK_TABLE_END -->

Two things are deliberately **not** in the mask:

- **RepeatMasker.** It covers 86.81% of chr21, and including it would leave ~13% accessible.
  Alu, MIR and L2 elements are roughly half of any primate genome and most of that sequence is
  uniquely mappable and perfectly callable. Unique 36-mers are the operational criterion: what
  matters is whether a read can be *placed*, not whether the sequence is annotated repetitive.
- **Excess heterozygosity.** The test needs enough called samples within a subspecies to be
  meaningful, and this panel is too small — *ellioti* has 5 individuals. A test that thin
  produces noise rather than paralogue evidence.

**The accessible fraction here is high compared with the human prep's ~74%, and that is a mask
measuring less, not cleaner data.** The HGDP mask combines mappability, read depth and excess
heterozygosity. This one can only ever have the mappability component: depth-based callability
is impossible on a variant-only callset, because depth can only be judged where something was
called, and the regions that matter most are those where nothing was.

## Recombination map

The source is the Stevison *et al.* *ellioti* map, built on hg18. Getting it into panTro6
coordinates and into the units `msprime.RateMap.read_hapmap` expects takes three steps:

1. **Lift hg18 → hg38 → panTro6** with UCSC chain files. The map's own hg18 chromosome column
   is what drives this, not the chimpanzee name — `chr2A` and `chr2B` both derive from human
   `chr2`, so keying on the chimpanzee name lifts 0% of intervals.
2. **Convert ρ/kb to cM/Mb** with a single global scalar (1.37424) across all chromosomes,
   never per chromosome. Small chromosomes genuinely recombine more per megabase because of
   the obligate crossover, and per-chromosome scaling would erase that real signal. The built
   maps confirm it survives:

   | chr1 | chr12 | chr19 | chr21 |
   |---|---|---|---|
   | 224 Mb, 1.00 cM/Mb | 131 Mb, 1.11 | 57 Mb, 1.28 | 33 Mb, 1.61 |

3. **Gap-fill and extend to the chromosome end.** Intervals the map could not estimate carry
   the string `na` rather than a number, and the workflow requires the map to extend past the
   last variant; unestimated and trailing regions take the chromosome mean.

Only the absolute scale is anchored on a human-derived figure; the landscape is
chimpanzee-derived.

## Ancestral states

Ancestral states come from **hg38 reference bases projected into panTro6** through the
`panTro6ToHg38` chain. Positions with no human alignment (2–3%) are left as `N` and fall
through to SINGER's own per-site polarisation.

The chain is parsed directly rather than shelled out to `liftOver`, for two reasons: it nets
in panTro6 coordinates — the space being filled — so every panTro6 base is covered at most
once by construction; and `liftOver`'s BED output silently drops the strand column unless the
input carries six, which would leave minus-strand bases uncomplemented with no error raised.

The build gates on **per-strand identity** against the panTro6 reference, requiring ≥90% where
~98% is expected, and refuses to write output otherwise. That gate caught a real bug: chain
`q` coordinates on minus-strand blocks are in reverse-complement space and must be converted
per block, and converting once per chain instead is correct for the first block and drifts
after. Minus-strand blocks are only 0.4% of the data, so a pooled identity read 98.3% and
would have passed while every minus-strand base was wrong. Measured: 29.4% identity before the
fix, 95.8% after.

**Why the human outgroup and not the Ensembl chimpanzee ancestor.** The EPO chimpanzee
ancestral sequence was the original choice and was rejected on measurement. It is reconstructed
against a *western* chimpanzee reference, which leaves a per-subspecies bias: *verus* appears
fixed-derived at 3.79% of sites against *troglodytes*' 0.46%. Keeping only EPO's
high-confidence calls makes it **worse**, taking the asymmetry from 8.17× to 13.24× — which is
diagnostic. If the asymmetry were reconstruction *uncertainty* it would concentrate in the
low-confidence calls and masking would reduce it; instead the low-confidence calls were
diluting it.

The human outgroup trades a **higher random** mispolarisation rate (6 My of divergence, so more
multiple hits) for a **near-zero systematic** one. For a joint ARG across four subspecies that
is the right direction: random error inflates apparent old mutations roughly uniformly and is
absorbed into a slightly distorted time axis, whereas systematic error manufactures
between-subspecies demographic differences that do not exist. A uniform distortion is
recoverable; an invented structural difference is not.

## Phasing

SHAPEIT5 `phase_common`, no reference panel and no genetic map, `--hmm-ne 10000`.

Beagle was tested and lost: **4.66% switch error against SHAPEIT5's 4.03%**, on the same
chromosome with the same metric, and worse in every subspecies. Beagle is ~9× faster, which is
not the binding constraint across 23 chromosomes.

`--hmm-ne` was swept at 10,000 / 25,000 / 30,000 and made no difference — a 0.06-point spread,
non-monotonic, against a 0.20-point swing between chromosomes at the *same* setting. It stays
at 10,000, and no "best" value should be quoted from that sweep.

Accuracy is measured against GATK `PGT`/`PID` read-backed phase blocks, the only independent
phase truth in a panel with no trios: **one switch error per ~16–17 kb, against a marginal tree
change every 0.146 kb**, so phase error is roughly 0.9% as frequent as tree transitions. Per
subspecies the rate tracks **sample size**, not data quality (*ellioti* n=5 worst at 5.4%,
*schweinfurthii* n=19 best at 2.3%), which is an argument against phasing subspecies
separately.

One caveat applies to every phasing number here: read-backed blocks are short, so this measures
*local* accuracy. It does not bound long-range switch error.

## Running SINGER

Each chromosome gets one SLURM job holding a whole node, and `snakemake --cores 64` schedules
every rule inside it as a local subprocess. SINGER is single-threaded and no rule declares
`threads:`, so one core per chunk means 64 chunks in parallel per chromosome.

This matters: singer-snakemake is written to run as `snakemake --cores=N` on a single machine,
and `run_polegon` is parameterised per (chunk, draw). Submitting each as its own SLURM job
means tens of thousands of submissions for a binary that runs in seconds, where scheduler
overhead exceeds the compute and throughput becomes submission-rate-bound rather than
CPU-bound.

**MCMC: 5,000 iterations — 20 draws thinned by 250.** SINGER's supplementary figure 10 puts the
burn-in requirement at roughly 4,000 iterations, so `singer-mcmc-burnin: 0.6` treats the
**first 12 draws as burn-in and uses the last 8** (draws 12–19, iterations 3,250–5,000).

Two consequences worth knowing:

- The burn-in fraction affects **only** the diagnostic statistics and plots. `merge_chunks`
  writes a tree sequence for every draw regardless, so nothing in the workflow stops a
  pre-convergence draw reaching CoaSTT. Step 9 enforces that separately, and on chain
  *position* rather than draw index: it keeps draws at iteration ≥ 4,000 and reads the last.
- Chunks are 2 Mb. SINGER's cost is roughly linear in chunk length in both memory and time,
  and at 3 Mb the measured peak was 4.00 GB per chunk against a node core's 3.93 GB share —
  the worst case exceeded its budget. Halving the chunk roughly halves the peak and compresses
  the runtime tail, which is what actually sets the makespan: a node's job finishes when its
  *slowest* chunk does, and at 3 Mb the slowest 10% of chunks consumed 46% of all CPU-hours.

## Output

```
results/chimp/
  singer-snakemake_outputs/    the workflow's own output: one tree sequence per draw,
    chr1/ chr2A/ … chr22/      diagnostic plots, per-chunk intermediates
  ts_chunks/                   what CoaSTT reads
    chr1_0_1994461.trees
    chr2A_1994461_3988922.trees
    …
```

`ts_chunks/` is **flat** — every chromosome's chunks in one directory, no subdirectories —
because `CountCoalEvents` globs `*.trees` and turns each filename into a contig id. Filenames
keep the `chr` prefix rather than a bare number: contig ids beginning with a digit are awkward
in Python and pandas, and `chr2A` and `chr2B` are not numbers at all.

Three details in that step are easy to get wrong, and the shared implementation
(`../human_data/scripts/extract_ts_chunks.py`, used here rather than duplicated) handles them:

- `merge_chunks` emits one whole-chromosome tree sequence per draw, not per-chunk files, so
  `ts_chunks/` is built by re-splitting the merged output — not by collecting the intermediates
  under `chunks/`.
- Chunk boundaries come from `<chrom>.chunks.p`, a pickled `RateMap` where rate 1.0 marks a
  kept chunk and 0.0 a skipped one. Its positions are **one-based** and need a shift of 1 to
  reach the tree sequence's zero-based coordinates.
- Splitting uses `keep_intervals(..., simplify=False)` rather than trimming, so each chunk
  keeps true chromosomal coordinates. The consequence is that every chunk reports the **full
  chromosome** `sequence_length` with data only inside its own interval — which is what CoaSTT
  expects, since it treats each file as an independent contig.

`drivers/prep.yaml` records every measured number, every decision, and every decision that was
later reversed along with the measurement that overturned it.
