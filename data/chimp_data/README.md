# Chimpanzee preparation for SINGER

Builds the input files that [`singer-snakemake`](https://github.com/nspope/singer-snakemake)
needs to infer ancestral recombination graphs from the Published_Pan chimpanzee callset
(54 chimpanzees, panTro6). Running the workflow itself happens elsewhere; the clone lives at
`../singer-snakemake`.

The sibling of [`../human_data/`](../human_data/), but starting from much rougher material.
The HGDP prep begins with a curated, phased, quality-filtered public release. This one begins
with a raw GATK callset: **unphased, unfiltered, no published accessibility mask, and
variant-only**. Everything the human prep could download, this one has to build.

## Reproducing this from a fresh clone

Everything is driven by the numbered scripts in `drivers/`. Run them in order; each is a thin
wrapper that submits the matching script in `scripts/` and waits for it, so the whole pipeline
is nine commands and no other knowledge.

```bash
conda activate CoaSTT_env
cd data/chimp_data

bash drivers/1_prep_vcf.sh         # split by chromosome, drop bonobos, site filters
bash drivers/2_genmap_index.sh     # whole-genome mappability index          (~40 min, once)
bash drivers/3_build_resources.sh  # recombination maps + ancestral FASTAs
bash drivers/4_build_masks.sh      # accessibility masks
bash drivers/5_phase.sh            # SHAPEIT5 phasing                        (longest step)
bash drivers/6_assemble.sh         # assemble SingerSnakemake_Input/
bash drivers/7_verify.sh           # 14 checks per chromosome -- THE GATE
bash drivers/8_run_singer.sh       # SINGER + POLEGON                        (~1 day)
bash drivers/9_extract_chunks.sh   # per-chunk .trees for CoaSTT             (minutes)
```

Steps 1-7 **block** until their SLURM array finishes, so `bash drivers/1_prep_vcf.sh && bash
drivers/2_genmap_index.sh && ...` is a valid way to run the whole thing unattended. Step 8
detaches and returns immediately -- monitor it with `tail -f logs/chimp.snakemake.log`.

Steps 2 and 3 do not depend on step 1 and can be submitted at the same time. Step 4 needs 1
and 2; everything from 5 on is sequential. **Do not skip step 7** -- it is the gate that
catches an unphased or mislabelled input before step 8 spends a day on it.

Useful variants:

```bash
bash drivers/8_run_singer.sh --pilot --chrom chr21   # smoke test: one chromosome, short MCMC
bash drivers/8_run_singer.sh -n                      # dry run: print the plan, submit nothing
bash drivers/8_run_singer.sh --config slurm_caslake  # shared partition instead of jnovembre
```

## Layout

```
README.md                   <- the only file at the top level

drivers/                    everything a person runs, and the parameters they run with
  1_prep_vcf.sh ... 9_extract_chunks.sh    the pipeline, in order
  singer_config.yaml        scientific parameters for singer-snakemake
  slurm_jnovembre.config    cluster settings: dedicated partition (the default)
  slurm_caslake.config      cluster settings: shared partition, higher throughput
  prep.yaml                 every measurement, decision, and reversal on record

scripts/                    the implementations the drivers call
resources/                  downloaded and derived reference data (~15 GB + a 28 GB VCF)
intermediates/              per-chromosome working files (~46 GB, regenerable by steps 1-5)
SingerSnakemake_Input/      the 161 files singer-snakemake reads
results/chimp/
  singer-snakemake_outputs/ SINGER + POLEGON output, one .tsz per MCMC replicate
  ts_chunks/                flat per-chunk .trees -- THIS is what CoaSTT reads
logs/                       run logs
```

### What lands in `intermediates/`

Working files, kept because regenerating them costs a full prep re-run, but safe to delete
if you need the space back -- nothing downstream of step 6 reads them.

| Path | Written by | What it is |
|---|---|---|
| `filtered/<chrom>.raw.vcf.gz` | step 1 | chromosome split out, bonobos dropped |
| `filtered/<chrom>.renamed.vcf.gz` | step 1 | same, with contigs renamed to UCSC names |
| `filtered/<chrom>.sitefiltered.vcf.gz` | step 1 | after the three site filters -- the phasing input |
| `filter_positions/<chrom>.txt` | steps 1, 4 | removed positions, restricted to accessible sequence |
| `<chrom>_het_flags.npy` | step 1 | excess-heterozygosity sites, consumed by the mask build |
| `<chrom>_map_{hg18,hg38,p6}.bed` | step 3 | the recombination map at each liftOver stage |
| `phased/<chrom>.vcf.gz` | step 5 | SHAPEIT5 output -- the most expensive artefact here |
| `rename_<chrom>.txt` | step 1 | the accession-to-UCSC mapping handed to `bcftools annotate` |

Everything runs on SLURM (`account=pi-jnovembre`). The login session is capped at an **8 GiB
cgroup**, which is not enough for Beagle, whole-chromosome `read_vcf`, or even the
verification script -- all three were OOM-killed there during development, so submit rather
than running interactively.

## What the panel is

| | |
|---|---|
| Samples | 54 chimpanzees (7 bonobos removed) |
| Subspecies | *schweinfurthii* 19, *troglodytes* 15, *verus* 14, *ellioti* 5, and one *verus* × *troglodytes* hybrid |
| Assembly | panTro6, 23 autosomes — `chr2A` and `chr2B` replace `chr2`, so 23 sequences, not 22 |
| Final sites | 44,346,959 across all autosomes |

The hybrid (`SAMN01920541`) is deliberately kept. It gets its own `subspecies` value, so
`stratify-by: subspecies` produces a one-sample stratum: **every cross-coalescence pair
involving it is noise and should be ignored in plots**. The trade is worth it — it is the
panel's only *verus* × *troglodytes* individual, so keeping it means admixture between those
subspecies is measured rather than removed by construction.

## How the raw VCF became these files

```
resources/vcf/full.vcf.gz                      28 GB, 61 samples, GenBank accession names
        │
        │  per chromosome: bcftools view -S chimps.txt -a  |  bcftools view -c 1
        ▼
   drop the 7 bonobos, then drop sites no chimpanzee carries
        │
        │  bcftools annotate --rename-chrs        CM009259.2 → chr21
        ▼
   ── site filters ──────────────────────────────────────────── scripts/filter_sites.py
        │  GATK hard filters, allelic imbalance, depth       →  intermediates/filter_positions/chrN.txt
        ▼
   ── phasing ──────────────────────────────────────────────── SHAPEIT5 phase_common
        │  --hmm-ne 10000, no reference panel, no genetic map
        ▼
   ── assembly ─────────────────────────────────────────────── scripts/assemble_inputs.py
           merge the multiallelic records back in, write meta.csv, link shared resources
        ▼
   SingerSnakemake_Input/chrN.{vcf.gz,tbi,filter.txt,meta.csv,mask.bed,hapmap,ancestral.fa.gz}
        │
        │  ── ARG inference ──────────────────────── drivers/8_run_singer.sh
        │     SINGER + POLEGON via singer-snakemake, one SLURM job per rule
        ▼
   results/chimp/singer-snakemake_outputs/chrN/trees/chrN.<rep>.tsz   one per MCMC replicate
        │
        │  ── packaging ─────────────────────────────── drivers/9_extract_chunks.sh
        │     take the final replicate, re-split into per-chunk tree sequences
        ▼
   results/chimp/ts_chunks/chrN_<start>_<end>.trees               flat, what CoaSTT reads
```

Built separately, once per chromosome, and symlinked in:

```
scripts/build_hapmap.py           Stevison map:  hg18 → hg38 → panTro6
scripts/build_ancestral_fasta.py  hg38 bases projected through the panTro6→hg38 chain
scripts/build_mask.py             genmap mappability ∪ assembly gaps ∪ excess-het regions
```

### Filtering comes before phasing

Not a preference — a requirement, for two independent reasons.

**SHAPEIT5 emits `GT` and nothing else.** It strips `AD`, `DP`, `GQ`, `PGT`, `PID`, `PL` and
`PS`. The depth filter needs `FORMAT/DP` and the allelic-imbalance filter needs `FORMAT/AD`,
so after phasing they are impossible to compute, not merely awkward.

**Phasing on unfiltered data is measurably worse.** Same chromosome, same metric, same samples:

| | Switch error |
|---|---|
| phase → filter | 4.32% |
| **filter → phase** | **3.59%** |

A 17% relative improvement, better in every subspecies.

### The three site filters

All three target the same failure mode: **collapsed paralogues and mismapping**, which
masquerade as high-diversity regions — precisely the signal CoaSTT reads.

| Filter | Sites removed | % |
|---|---|---|
| GATK hard filters | 2,699,496 | 5.47% |
| Allelic imbalance | 3,304,436 | 6.69% |
| Depth | 813,363 | 1.65% |
| **Union** | **5,021,313** | **10.17%** |

The union is far smaller than the sum, because the three overlap heavily — which is the
reassuring result. Three independent criteria converging on the same sites is what one
underlying artefact looks like; three filters each removing different sites would suggest they
were cutting real variation.

- **GATK hard filters** — `QD<2 | FS>60 | MQ<40 | SOR>3 | ReadPosRankSum<-8 | MQRankSum<-12.5`.
  Best Practices thresholds, human-tuned, not re-derived for chimpanzee. Two are effectively
  inert here: `ReadPosRankSum` removes nothing and `MQRankSum` almost nothing. `QD` and `SOR`
  do the work.
- **Allelic imbalance** — pooled alternate-allele fraction across heterozygous calls outside
  `[0.25, 0.75]`. The distribution is healthy (median 0.4964 against a 0.5 expectation), so
  this cuts genuine outliers rather than the bulk.
- **Depth** — summed `FORMAT/DP` outside 0.25×–2× the median. Median rather than mean, because
  the mean is inflated by exactly the paralogue pileups the upper cut targets. Built on
  `FORMAT/DP`, never `INFO/DP`: the latter runs **8.7× higher** here, since it counts
  pre-filter reads across all 61 originally-called samples.

### `filter.txt`, and why it matters

`chrN.filter.txt` lists the **2,729,120** removed positions that fall in accessible sequence.
The workflow uses it to scale `adj_mu`: a site removed but not recorded would inflate the
inferred mutation rate, because the workflow would believe that sequence was surveyed and
found monomorphic.

Two things deliberately stay out of it. Positions in masked regions are excluded because the
workflow strips inaccessible positions from `filtered_positions` anyway. Multiallelic sites
are excluded because the workflow routes those there itself — listing them would double-count.

That second point drives the assembly step. SHAPEIT5 drops multiallelic records, but they must
still be **present** in the final VCF so the workflow can count them. So the assembly merges
them back, unphased, after phasing. They are discarded before any haplotype is built; they
exist purely to be counted. Omitting them would inflate μ by roughly 8%.

## Decisions worth knowing about

### Human outgroup, not the Ensembl chimpanzee ancestor

Ancestral states come from hg38 reference bases projected into panTro6. The Ensembl EPO
chimpanzee ancestral sequence was the original choice and was **rejected on measurement**.

EPO's ancestor is reconstructed against a *western* chimpanzee reference, and that leaves a
per-subspecies bias: *verus* appears fixed-derived at 3.79% of sites against *troglodytes*' 0.46%.
The obvious remedy — keeping only EPO's high-confidence calls — makes it **worse**, taking the
asymmetry from 8.17× to 13.24×. That is diagnostic: if the asymmetry were reconstruction
*uncertainty* it would concentrate in the low-confidence calls and masking would reduce it.
Instead the low-confidence calls were diluting it. The bias lives in the high-confidence calls,
where no confidence filter can reach.

The human outgroup trades a **higher random** mispolarisation rate (6 My of divergence, so more
multiple hits) for a **near-zero systematic** one. For a joint ARG across four subspecies that is
the right direction: random error inflates apparent old mutations roughly uniformly and is
absorbed into a slightly distorted time axis, whereas systematic error manufactures
between-subspecies demographic differences that do not exist. A uniform distortion is
recoverable; an invented structural difference is not.

Positions with no human alignment (2–3%) are left as `N` and fall through to SINGER's own
per-site polarisation.

### SHAPEIT5, not Beagle

Beagle was tested and lost: **4.66% switch error against SHAPEIT5's 4.03%**, on the same
chromosome with the same metric, and worse in every subspecies. Beagle is ~9× faster, which is
not the binding constraint across 23 chromosomes.

`--hmm-ne` was swept at 10,000 / 25,000 / 30,000 and made no difference — a 0.06-point spread,
non-monotonic, against a 0.20-point swing between chromosomes at the *same* setting. It stays at
10,000, and no "best" value should be quoted from that sweep.

### The mask is mappability-only

**The accessible fraction is 79–91% per chromosome, against the human prep's ~74%. That is not
cleaner data — it is a mask that measures less.** Anyone comparing retention between the two
preps needs to know this.

Two components are missing and neither is recoverable. **Depth-based callability is impossible**
because the callset is variant-only: depth can only be judged where something was called, and
the regions that matter most are those where nothing was. And **RepeatMasker was deliberately
dropped** — it covers 87% of the assembly, and most Alu/MIR/L2 sequence is uniquely mappable and
perfectly callable. Unique *k*-mers are the operational criterion: what matters is whether a read
can be *placed*, not whether the sequence is annotated as repetitive.

What remains is `genmap` mappability (k=36, exact matching) unioned with assembly gaps and with
excess-heterozygosity regions. Excess heterozygosity is computed **within subspecies, never
pooled** — the panel is deeply structured, and pooling produces heterozygote *deficit* by the
Wahlund effect, which is the wrong direction entirely.

### One global scalar for the recombination map

The Stevison *ellioti* map is normalised to cM/Mb with a single scalar across all chromosomes,
never per chromosome. Small chromosomes genuinely recombine more per megabase because of the
obligate crossover, and per-chromosome scaling would erase that. The built maps confirm the
signal survives:

| chr1 | chr12 | chr19 | chr21 |
|---|---|---|---|
| 224 Mb, 1.00 cM/Mb | 131 Mb, 1.11 | 57 Mb, 1.28 | 33 Mb, 1.61 |

Only the absolute scale is anchored on a human-derived figure; the landscape is chimpanzee-derived.

## How good is the phasing?

**One switch error per ~16–17 kb, against a marginal tree change every 0.146 kb** — so phase
error is roughly 0.9% as frequent as tree transitions. A switch lands inside a stretch already
spanned by ~115 distinct marginal trees, which is why it is noise here rather than a dominant
error term. Treat that as an order-of-magnitude bound: one switch can force more than one
spurious recombination, but there is an order of magnitude of headroom.

The underlying rate is 3.59% of adjacent heterozygous pairs, measured against GATK `PGT`/`PID`
read-backed phase blocks — the only independent phase truth in a panel with no trios. Per
subspecies the rate tracks **sample size**, not data quality (*ellioti* n=5 worst at 5.4%,
*schweinfurthii* n=19 best at 2.3%), which is an argument against phasing subspecies separately.

**One caveat applies to every phasing number here.** Read-backed blocks are short, so this
measures *local* accuracy. It does not bound long-range switch error, and `--hmm-ne` plausibly
matters more there than this metric can see.

## What is verified

`bash drivers/7_verify.sh` runs 14 checks per chromosome. **All 23 currently pass.**
Each covers a failure that is otherwise silent or surfaces only hours into a SINGER run:

- hapmap length == ancestral FASTA length == `chrom.sizes`, **exactly** — the workflow asserts
  equality, not sufficiency
- ancestral FASTA is a single record of the right length
- hapmap positions strictly increasing, cumulative cM non-decreasing, no negative rates, mean
  rate in 0.3–4.0 cM/Mb
- mask intervals in bounds and well-formed; accessible fraction in [0.50, 0.99], a band chosen
  to catch **inverted polarity** (an inverted mask lands near 5%)
- no duplicate positions
- **all biallelic genotypes phased** — this check exists because an unphased input is fatal and
  invisible to every other check; one build once passed everything while 92% unphased
- `meta.csv` row order matches the VCF sample order element by element (the workflow zips by
  index, with no ID matching)
- no bonobo samples present

The ancestral build has its own gate: it compares each projected human base against the panTro6
reference **separately for plus- and minus-strand chain blocks**, requiring ≥90% identity when
~98% is expected. This caught a real bug — minus-strand blocks are 0.4% of the data, so a pooled
identity read 98.3% and would have passed while every minus-strand base was wrong.

## Sources

| What | Where | Becomes |
|---|---|---|
| Published_Pan callset | GATK, panTro6, 61 samples | `resources/vcf/full.vcf.gz` |
| Sample metadata | `sample_data.csv` | `resources/samples/subspecies.tsv` |
| hg38 + `panTro6ToHg38` chain | UCSC | `resources/panTro6/ancestral/chr{N}.ancestral.fa.gz` |
| Stevison *ellioti* map | `lstevison/great-ape-recombination` | `resources/panTro6/hapmap/chr{N}.hapmap` |
| panTro6 reference + gap track | UCSC goldenPath and hub | `resources/panTro6/mask/chr{N}.mask.bed` |

`prep.yaml` records every measured number, every decision, and every decision that was later
reversed along with the measurement that overturned it.

## Traps in this data

Each of these fails silently. Most were found the hard way.

- **Three naming schemes for one chromosome.** The VCF uses GenBank (`CM009259.2`), UCSC hub
  tracks use RefSeq (`NC_036900.1`), our inputs use `chr21`. A `bcftools` target file keyed on
  the wrong one matches nothing **and reports no error** — a filtering step once removed 0 of
  681,688 sites this way and declared success.
- **`chr2A` and `chr2B` both come from human `chr2`.** The recombination map's hg18 column says
  `chr2` for both; using the chimpanzee name lifts 0% of intervals.
- **The map's minus-strand token is `minus1`, not `minus`** — 7.4% of rows genome-wide. chr21
  has none, so a chr21-only pilot never surfaces it.
- **`filter != 0` in the map means the rate is the string `na`** — an unestimated interval to
  gap-fill, not a flagged value.
- **Map coordinates are megabases as decimals.** Multiply by 1e6, or the map silently covers the
  first ~33 bp of the chromosome.
- **Chain `q` coordinates on minus-strand blocks are in reverse-complement space** and must be
  converted per block. Converting once per chain is right for the first block and drifts after.
- **`mask.bed` holds INACCESSIBLE intervals.** Storing the opposite polarity inverts the analysis
  with no error.
- **A killed Beagle still writes a valid-looking 567-byte VCF and index.** Check record counts,
  never exit status.
- **Memory scales with chromosome length, not variant count** — the per-base ancestral array
  dominates. 8 GB per task is enough for the largest chromosome.

## Running SINGER, and what comes out

```bash
bash drivers/8_run_singer.sh
```

This launches a detached snakemake that submits **one SLURM job per rule** and returns
immediately. Scientific parameters come from `drivers/singer_config.yaml`; cluster settings
from `drivers/slurm_jnovembre.config` (the default) or `drivers/slurm_caslake.config`.

`singer_config.yaml` sets `mutation-rate: 1.2e-8` (chimpanzee trio-derived, not the human
value), `stratify-by: subspecies`, and keeps unpolarised sites. The MCMC budget is
`singer-mcmc-samples: 100` with `singer-mcmc-thin: 50` -- **5,000 iterations per chunk** --
and `singer-mcmc-burnin: 0.8`, which keeps only the last 20 draws (80-99, i.e. iterations
4,050-5,000). Same settings as the human prep.

Run state goes to a scratch workdir **outside the repo** (`$SCRATCH/singer_work/chimp`).
That state is disposable: which chunks are finished is recorded by the `.singer.done`
sentinels under `results/`, not by the workdir, so deleting it loses no completed work and a
restart picks up where the last one stopped.

A short test run is worth doing first:

```bash
bash drivers/8_run_singer.sh --pilot --chrom chr21
```

`--pilot` cuts the number of MCMC samples but keeps the shared thinning interval, so a pilot
draw comes from the same chain behaviour as a production one rather than a different regime.

### Why one job per rule, and not an sbatch array

The first genome-wide run used a 23-task array with a flat `--mem=56G`, which forced every
rule to share one memory number. That number was sized from `run_singer`'s per-chunk cost
(2.4-4.5 GB) and did not cover `merge_chunks`, whose peak scales with **total chromosome
size** rather than chunk size. Six of 23 chromosomes were OOM-killed there, each after 12-16
hours of completed inference.

The margin was nil rather than merely tight: **chr2B survived at 57.1 GB while chr5 died at
56.6 GB**. Same limit, opposite outcomes. The flat number was not separating well-behaved
chromosomes from badly-behaved ones -- it was a coin flip on which merge jobs happened to be
in flight when memory ran out.

Per-rule submission fixes this at the root. `merge_chunks` gets its own allocation (72 GB,
escalating to 112 and 152 on retry) without inflating the ~958 `run_singer` jobs, and
`retries: 2` means a chromosome that needs more memory gets it automatically instead of
failing permanently. The retry is cheap: `merge_chunks` is parameterised per replicate and
the SINGER output it consumes survives on disk, so a re-run redoes minutes of merging rather
than the hours of inference behind it.

### Burn-in, and why 5,000 iterations

SINGER's supplementary figure 10 puts the burn-in requirement at roughly **4,000 iterations**.
An earlier run at thin 100 x 16 samples -- 1,600 iterations -- was abandoned after 22 of 23
chromosomes had started, because **every draw it produced fell inside burn-in**: the chain had
not converged, so none of its tree sequences were usable no matter how many completed.

The current setting produces 100 draws at thin 50, and `singer-mcmc-burnin: 0.8` treats the
first 80 as burn-in. Only draws 80-99 are used.

Two traps follow from that, both of which have teeth:

- **`singer-mcmc-burnin` does not reduce what gets built.** It is applied only to statistics
  and diagnostic plots; `merge_chunks` iterates over all `singer-mcmc-samples`, so the
  workflow constructs, dates and stores **all 100 draws**. Budget disk accordingly -- the
  abandoned 16-draw run already held 399 GB.
- **Nothing upstream stops a pre-convergence draw reaching CoaSTT.** Step 9 is where that is
  caught: it reads the burn-in floor from `drivers/singer_config.yaml` and refuses any
  chromosome whose best draw is below it.

### Timing

Timing is **not linear in the MCMC budget**: each chunk costs a fixed ARG-construction phase
plus a term linear in iterations, so tripling the iteration count does not triple the runtime.

`run_singer` requests **20 h**, sized from measured per-chunk times on the human run at these
exact MCMC settings (2,971 completed jobs):

| p50 | p90 | p99 | max |
|---|---|---|---|
| 1.71 h | 3.36 h | 5.65 h | **17.15 h** |

A 12 h limit would have killed 12 of those 2,971 jobs (0.40%); 18 h and above kill none, and
20 h clears the observed maximum with ~17% margin.

Resist raising this "to be safe". An oversized walltime is not free -- SLURM backfills short
jobs into scheduling gaps, so a job asking for far more than it uses waits longer for headroom
it never touches. Those numbers are from n=216 humans; this panel is n=54, so ARG construction
should be faster here, while 3 Mb chunks against 2 Mb push the other way. **Re-measure once
real chunks have run** rather than trusting the transfer.

`jnovembre` has 192 CPUs and runs one chunk per core; `slurm_caslake` has 9,120 and can hold
far more in flight, at the cost of queueing behind other users.

### Check the replicates before extracting

An OOM or crash takes out whichever `merge_chunks` jobs were in flight, and those need not be
the last ones. On the first run **chr8 came back missing replicates 1, 6 and 15 while keeping
14** -- interior gaps, not a truncation.

This matters because step 9 takes the *highest-numbered* replicate per chromosome. A
chromosome missing its top draw would silently contribute a different, less burnt-in
posterior sample than its neighbours, and nothing downstream could tell: the file is valid,
the chunk count is right, and CoaSTT has no way to know one contig came from elsewhere in the
chain.

Step 9 refuses to build a mixed-replicate dataset and names the chromosomes to re-merge, so
this cannot pass silently. To inspect by hand:

```bash
for c in results/chimp/singer-snakemake_outputs/*/; do
  echo "$(basename $c): $(ls $c/trees/*.tsz 2>/dev/null | sed 's/.*\.\([0-9]*\)\.tsz/\1/' | sort -n | tr '\n' ' ')"
done
```

Results land in two directories, with different consumers:

```
results/chimp/
  singer-snakemake_outputs/       <- singer-snakemake writes here
    chr1/  chr2A/  ...  chr22/       trees/ plots/ stats/ chunks/ per chromosome
  ts_chunks/                      <- CoaSTT reads here
    chr1_0_2952691.trees
    chr2A_2952691_5905382.trees
    ...
```

`results/chimp/` is named for the sample set, matching `data/human_data/results/<SET>/`. The
chimpanzee panel is a single set, so there is one such directory; the human side has eight.

`singer-snakemake_outputs/` holds the full per-chromosome output: one tree sequence per MCMC
replicate, diagnostic plots, and the per-chunk intermediates.

`ts_chunks/` is what CoaSTT consumes. It is **flat** — every chromosome's chunks in one
directory, no subdirectories — because `CountCoalEvents` globs `*.trees` and turns each
filename into a contig id. Build it with:

```bash
bash drivers/9_extract_chunks.sh               # the production run
bash drivers/9_extract_chunks.sh chimp_pilot   # a pilot run, or any other results/<name>
```

It refuses to run if the chromosomes disagree on which replicate is their latest -- see
[Check the replicates before extracting](#check-the-replicates-before-extracting).

Three details in that step are easy to get wrong, and the shared implementation
(`../human_data/scripts/extract_ts_chunks.py`, used here rather than duplicated) handles them:

- `merge_chunks` emits **one whole-chromosome tree sequence per replicate**, not per-chunk
  files, so `ts_chunks/` is built by re-splitting the merged output — not by collecting the
  intermediates in `chunks/`.
- Chunk boundaries come from `<chrom>.chunks.p`, a pickled `RateMap` where rate 1.0 marks a
  kept chunk and 0.0 a skipped one. Its positions are **one-based**, so they need a shift of 1
  to reach the tree sequence's zero-based coordinates. Only kept chunks are written; skipped
  ones have no SINGER output.
- Splitting uses `keep_intervals(..., simplify=False)` rather than trimming, so each chunk
  keeps true chromosomal coordinates — which is what makes the start and end in the filename
  meaningful. The consequence is that every chunk reports the **full chromosome**
  `sequence_length` with data only inside its own interval. That is what CoaSTT expects, since
  it treats each file as an independent contig.

Filenames are `<chrom>_<start>_<end>.trees`, keeping the `chr` prefix rather than a bare
number: contig ids beginning with a digit are awkward in Python and pandas, and `chr2A` and
`chr2B` are not numbers at all.

Tree sequences come from the **final MCMC replicate** — the highest index in
`<chrom>/trees/<chrom>.<rep>.tsz`, which is the most burnt-in posterior draw.
