# Chimpanzee preparation for SINGER

Builds the input files that [`singer-snakemake`](https://github.com/nspope/singer-snakemake)
needs to infer ancestral recombination graphs from the Published_Pan chimpanzee callset
(54 chimpanzees, panTro6). Running the workflow itself happens elsewhere; the clone lives at
`../singer-snakemake`.

The sibling of [`../human_data/`](../human_data/), but starting from much rougher material.
The HGDP prep begins with a curated, phased, quality-filtered public release. This one begins
with a raw GATK callset: **unphased, unfiltered, no published accessibility mask, and
variant-only**. Everything the human prep could download, this one has to build.

**Status: inputs complete, ARG inference in progress.** All 23 autosomes are built and pass
verification (14/14 checks each). The genome-wide SINGER run is under way; see
[Running SINGER](#running-singer-and-what-comes-out) for the production command and the
memory caveat that is currently biting it.

## Full VCF to SINGER trees, start to finish

Nine commands. Run them in this order from `data/chimp_data/`; each waits on the one before it.
Steps 1-7 build the inputs, step 8 runs the ARGs, step 9 packages them for CoaSTT.

```bash
conda activate CoaSTT_env

# --- inputs: raw callset -> SingerSnakemake_Input/ -------------------------------
sbatch scripts/prep_vcf.sbatch          # 1. split by chromosome, drop bonobos, site filters
sbatch scripts/genmap_index.sbatch      # 2. whole-genome mappability index      (~40 min, once)
sbatch scripts/build_resources.sbatch   # 3. recombination maps + ancestral FASTAs
sbatch scripts/build_mask_all.sbatch    # 4. accessibility masks, restrict filter.txt
sbatch scripts/phase_all.sbatch         # 5. SHAPEIT5 phasing
sbatch scripts/assemble_all.sbatch      # 6. assemble the final input directory
sbatch scripts/verify_all.sbatch        # 7. 14 checks per chromosome -- gate before step 8

# --- ARGs: SingerSnakemake_Input/ -> results/singer-snakemake_outputs/ -----------
sbatch scripts/run_singer_genome.sbatch # 8. SINGER + POLEGON, one array task per chromosome

# --- packaging: merged trees -> results/ts_chunks/ -------------------------------
bash   scripts/extract_chunks.sh        # 9. flat per-chunk .trees for CoaSTT (runs in minutes)
```

Steps 2 and 3 are independent of step 1 and can be submitted at the same time; step 4 needs
step 2, and everything from step 5 on is strictly sequential. **Do not skip step 7** -- it is
the gate that catches an unphased or mislabelled input before step 8 spends a day on it.

Each step is a 23-task array (one task per autosome) except step 2, which is a single
whole-genome job, and step 9, which is a short serial script. Steps are idempotent: each skips
work whose output already exists, so a partially-failed array can simply be resubmitted.

Everything runs on SLURM (`account=pi-jnovembre`). Steps 1-7 use `partition=caslake`; step 8
uses `partition=jnovembre`. The login session is capped at an **8 GiB cgroup**, which is not
enough for Beagle, whole-chromosome `read_vcf`, or even the verification script -- all three
were OOM-killed there during development, so submit rather than running interactively.

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
        │  GATK hard filters, allelic imbalance, depth       →  work/filter_positions/chrN.txt
        ▼
   ── phasing ──────────────────────────────────────────────── SHAPEIT5 phase_common
        │  --hmm-ne 10000, no reference panel, no genetic map
        ▼
   ── assembly ─────────────────────────────────────────────── scripts/assemble_inputs.py
           merge the multiallelic records back in, write meta.csv, link shared resources
        ▼
   SingerSnakemake_Input/chrN.{vcf.gz,tbi,filter.txt,meta.csv,mask.bed,hapmap,ancestral.fa.gz}
        │
        │  ── ARG inference ──────────────────────── scripts/run_singer_genome.sbatch
        │     SINGER + POLEGON via singer-snakemake, one array task per chromosome
        ▼
   results/singer-snakemake_outputs/chrN/trees/chrN.<rep>.tsz     one per MCMC replicate
        │
        │  ── packaging ─────────────────────────────── scripts/extract_chunks.sh
        │     take the final replicate, re-split into per-chunk tree sequences
        ▼
   results/ts_chunks/chrN_<start>_<end>.trees                     flat, what CoaSTT reads
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

`sbatch scripts/verify_all.sbatch` runs 14 checks per chromosome. **All 23 currently pass.**
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
sbatch scripts/run_singer_genome.sbatch
```

**Submit the array; do not run `snakemake` directly.** One snakemake over all 23 chromosomes
serialises them, and 23 concurrent snakemakes in one directory race on `.snakemake/iocache`
even with `--nolock`. The array solves both: one task per chromosome, each with its own
symlink-farm workdir under `runs/work/<chrom>/` pointing back at the shared clone, and a
per-chromosome config generated by `sed` from `singer_config.yaml`.

`singer_config.yaml` sets `mutation-rate: 1.2e-8` (chimpanzee trio-derived, not the human
value), `stratify-by: subspecies`, and keeps unpolarised sites. The MCMC budget is
`singer-mcmc-samples: 16` with `singer-mcmc-thin: 100` and `singer-mcmc-resumes: 1000`
-- 1,600 iterations per chunk.

A short test run is worth doing first:

```bash
sbatch scripts/run_singer_smoke.sbatch    # chr21 only, cut MCMC budget, ~30 min
```

`singer_config_smoke.yaml` is the same config restricted to chr21 with the budget cut (10
samples, thin 10, 20 resumes) and smaller chunks, so it exercises the whole pipeline quickly.

### Sizing the run, and the memory trap in it

Timing is **not linear in the MCMC budget**: there is a fixed setup cost of about 7.7 minutes
per chunk before the first iteration, then ~7.2 s/iteration. At 1,600 iterations that is
~3.3 h per chunk, and the genome is ~958 chunks.

Task size is a packing problem, not a speed problem. The `jnovembre` partition has a hard
ceiling of 192 CPUs (3 x 64), and `run_singer` declares no `threads:`, so snakemake runs one
chunk per core. Smaller tasks pack better:

| CPUs/task | Concurrent tasks | Makespan |
|---|---|---|
| **16** | **12** | **~20 h** |
| 24 | 8 | 23 h |
| 48 | 4 | 27 h |

16 CPUs/task is the current setting, within 20% of the 16.7 h theoretical floor.

**The memory limit is the live problem, and 56 GB is too low.** The `merge_chunks` rule --
the final step, which merges per-chunk output into one tree sequence per replicate -- peaks
far higher than SINGER itself. Its footprint scales with **total chromosome size**, not with
per-chunk size, so the 56 GB figure derived from SINGER's per-chunk cost (2.4-4.5 GB x 16
concurrent) never covered it. Measured: chr2A **succeeded at 50.9 GB**, 9% under the limit;
chr8, chr10 and chr11 were OOM-killed at ~57 GB.

An OOM here is recoverable and cheap, because `merge_chunks` is parameterised per replicate
(`{chrom}.{rep}.tsz`) -- one independent job per draw. SINGER and POLEGON outputs survive on
disk, so a rerun redoes only the missing replicates, in minutes rather than the ~12 h of
inference. Raise `--mem` and resubmit; snakemake skips everything already done.

**Check for interior gaps before extracting.** A kill takes out whichever merge jobs were in
flight, which need not be the last ones: chr8 came back missing replicates 1, 6 and 15 while
keeping 14. Step 9 takes the *highest-numbered* replicate per chromosome, so a chromosome
missing its top draw silently contributes a different, less burnt-in draw than its neighbours.
List what is present before packaging:

```bash
for c in results/singer-snakemake_outputs/*/; do
  echo "$(basename $c): $(ls $c/trees/*.tsz 2>/dev/null | sed 's/.*\.\([0-9]*\)\.tsz/\1/' | sort -n | tr '\n' ' ')"
done
```

Results land in two directories, with different consumers:

```
results/
  singer-snakemake_outputs/       <- singer-snakemake writes here
    chr1/  chr2A/  ...  chr22/       trees/ plots/ stats/ chunks/ per chromosome
  ts_chunks/                      <- CoaSTT reads here
    chr1_0_2952691.trees
    chr2A_2952691_5905382.trees
    ...
```

`singer-snakemake_outputs/` holds the full per-chromosome output: one tree sequence per MCMC
replicate, diagnostic plots, and the per-chunk intermediates.

`ts_chunks/` is what CoaSTT consumes. It is **flat** — every chromosome's chunks in one
directory, no subdirectories — because `CountCoalEvents` globs `*.trees` and turns each
filename into a contig id. Build it with:

```bash
bash scripts/extract_chunks.sh                      # the production run
bash scripts/extract_chunks.sh results/chr21_smoke  # or any other output tree
```

It accepts either the singer output directory itself or the run root containing it, so both
forms above work. Run it only once every chromosome has its final replicate -- see the gap
check in [Sizing the run](#sizing-the-run-and-the-memory-trap-in-it).

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
