# HGDP preparation for SINGER

Builds the input files that [`singer-snakemake`](https://github.com/nspope/singer-snakemake)
needs to infer ancestral recombination graphs from the statistically phased HGDP callset
(Bergström et al. 2020 — 929 genomes, GRCh38).

Two stages: download and derive shared per-chromosome resources, then cut those into 8
ready-to-run input directories, one per sample set. Then run the workflow — see *Running
SINGER* below. `singer-snakemake` is a git submodule at `../singer-snakemake`, pinned to the
commit these inputs were built and tested against; `git submodule update --init` after cloning.

## The whole pipeline, start to finish

Four commands take you from the published HGDP callset to tree sequences CoaSTT can read.
Each stage is idempotent — it skips work whose output already exists — so an interrupted run
resumes rather than restarting.

```bash
conda activate CoaSTT_env

# 1. download raw data + derive the shared per-chromosome resources   (a few hours)
bash download.sh

# 2. cut those into 8 per-set input directories                       (~15 min per set)
bash subset.sh --slurm-partition jnovembre --slurm-account pi-jnovembre

# 3. infer ARGs with SINGER                                           (~1 day, whole genome)
bash runs/run_singer.sh ALL --profile caslake_profile

# 4. cut the final posterior draw into CoaSTT's input directory       (minutes)
bash runs/extract_chunks.sh ALL
```

What each stage actually does, in order:

**1. `download.sh`** — seven steps, then verification:

| step | script | produces |
|---|---|---|
| 1 | *(curl)* | `resources/raw/` — statphase VCF, mask, metadata, ancestral tarball |
| 2 | `bcftools index -n` | integrity check: 929 samples against the published index |
| 3 | `make_ancestral.py` | `grch38/ancestral/chr{N}.ancestral.fa.gz` + `chrom.sizes` |
| 4 | `make_masks.py` | `grch38/mask/chr{N}.mask.bed` — **complemented** to inaccessible |
| 5 | `make_hapmaps.py` | `grch38/hapmap/chr{N}.hapmap` from the deCODE map |
| 6 | `make_metadata.py`, `make_sample_sets.py` | `hgdp/metadata.tsv`, `hgdp/sets/*.csv` |
| 7 | `split_statphase.sh` | `hgdp/statphase/chr{N}.bcf` — GT-only, per chromosome |
| — | `verify_resources.py` | 200 checks (see *Verification*) |

The deCODE map must be placed by hand first — see the note below.

**2. `subset.sh`** — runs `make_set_inputs.sh` per set (locally via `xargs -P`, or as an
8-task SLURM array), then `verify_sets.py`. Per set, per chromosome:

```
bcftools view -S <set>.csv -a -c 1 -v snps   →  chr{N}.vcf.gz   (+ index)
ln -s  mask.bed / hapmap / ancestral.fa.gz   →  shared, symlinked
make_sample_masks.py                         →  chr{N}.sample.mask.bed
write_set_metadata.py                        →  chr{N}.meta.csv  (in VCF order — order matters)
```

**3. `runs/run_singer.sh`** — one `snakemake` call against
`../singer-snakemake/workflow/Snakefile`, with `singer_config.yaml` as the configfile and only
`input-dir`, `output-dir` and `chromosomes` overridden per run. Writes
`results/<SET>/singer-snakemake_outputs/chr{N}/` with `trees/ plots/ stats/ chunks/`.

**4. `runs/extract_chunks.sh`** — calls `extract_ts_chunks.py`, which takes the highest-numbered
replicate (the most burnt-in posterior draw) for each chromosome, cuts it at the chunk
boundaries in `chr{N}.chunks.p`, and writes `results/<SET>/ts_chunks/chr{N}_{start}_{end}.trees`
— flat, all chromosomes together, which is what CoaSTT globs.

### Before you start: one file must be fetched by hand

science.org serves its supplements behind bot protection and returns 403 to every scripted
request, so the deCODE recombination map cannot be downloaded automatically. Grab *Data S3*
from [doi:10.1126/science.aau1043](https://www.science.org/doi/10.1126/science.aau1043) and put
it at `resources/raw/aau1043_datas3.gz` (gzip it if it arrives uncompressed). `download.sh`
stops with these instructions if it is missing.

### Variations

```bash
bash subset.sh                       # 4 sets at a time locally; --jobs 8 widens it
bash runs/run_singer.sh --chrom chr21 ALL --profile caslake_profile   # one chromosome
bash runs/run_singer.sh --chrom chr22 --profile caslake_profile       # all 8 sets
bash runs/run_singer.sh --pilot --chrom chr22 OCEANIA                 # smoke test
bash runs/extract_chunks.sh                                           # every finished set
```

## Sources

| What | Where | Becomes |
|---|---|---|
| Phased VCF + index | Sanger `hgdp_wgs.20190516/statphase/` | `resources/hgdp/statphase/chr{N}.bcf` (GT-only) |
| Accessibility mask | Sanger `…/accessibility-mask/` | `resources/grch38/mask/chr{N}.mask.bed` |
| Sample metadata | Sanger `…/metadata/` | `resources/hgdp/metadata.tsv` |
| Ancestral alleles | Ensembl release 113, EPO | `resources/grch38/ancestral/chr{N}.ancestral.fa.gz` |
| Recombination map | deCODE 2019, Data S3 (manual) | `resources/grch38/hapmap/chr{N}.hapmap` |

Sizes and SHA-256 for every download are in `prep.yaml`. Sanger publishes no checksums, so the
VCF is validated by agreement between its index and its data instead.

## What each input directory contains

`singer-snakemake` globs for `{chrom}.vcf.gz` and locates everything else by substituting on
that filename, so all five files share a stem and sit together:

```
SingerSnakemake_SampleSets/<SET>/input/
  chr{N}.vcf.gz(.tbi)       set-specific: subset, trimmed to segregating SNPs
  chr{N}.meta.csv           set-specific: sample metadata, in VCF order
  chr{N}.sample.mask.bed    set-specific: missing genotypes to impute (see below)
  chr{N}.mask.bed        →  symlink, shared
  chr{N}.hapmap          →  symlink, shared
  chr{N}.ancestral.fa.gz →  symlink, shared
```

The three shared files are byte-identical across sets, hence symlinks rather than eight copies.

`bcftools view -a -c 1 -v snps` trims alleles nobody in the subset carries and drops what has
become monomorphic, but does *not* enforce biallelic — multiallelic sites survive into the VCF
and the workflow removes them itself (323 of them on OCEANIA/chr22), counting them toward the
mutation-rate adjustment.

## The 8 sample sets

| set | n | per-population cap |
|---|---|---|
| OCEANIA | 28 | — |
| AMERICA | 61 | — |
| MIDDLE_EAST | 100 | 25 |
| AFRICA | 104 | — |
| CENTRAL_SOUTH_ASIA | 130 | 15 |
| EUROPE | 155 | — |
| EAST_ASIA | 168 | 10 |
| ALL | 216 | 4 (all 54 populations) |

Seven are regional. Three of those are capped per population to hold down SINGER's cost, which
rises steeply with sample count: MIDDLE_EAST 161→100, CENTRAL_SOUTH_ASIA 197→130, EAST_ASIA
223→168, for 929→746 across the seven. A cap only ever removes samples, so populations already
at or below it are untouched — Uygur (10) under CSA's cap of 15, and the fifteen EAST_ASIA
populations already at ≤10. Where a cap binds, the highest-coverage samples are kept, ties
broken by sample ID.

ALL is the 4 highest-coverage samples from each of the 54 populations. It is drawn from the
full metadata rather than the capped regional pools, so capping a region does not change who
represents it in ALL.

## Missing genotypes

This is the one substantive transformation applied to the genotype data, so it's worth
understanding.

SINGER's wrapper drops any site with even a single missing genotype. That's a lot of sites —
per-site union over samples, across all 22 autosomes, measured on the *uncapped* sets:

| set | n | affected | | set | n | affected |
|---|---|---|---|---|---|---|
| OCEANIA | 28 | 6.4% | | EUROPE | 155 | 27.9% |
| AMERICA | 61 | 12.3% | | MIDDLE_EAST | 161 | 28.1% |
| ALL | 216 | 19.2% | | CENTRAL_SOUTH_ASIA | 197 | 35.3% |
| AFRICA | 104 | 23.7% | | EAST_ASIA | 223 | 41.3% |

The three capped sets are now smaller than the n shown, so their true fractions are somewhat
lower — the statistic is a per-site union over samples, so it falls with sample count. The
figures are kept as measured because they are what motivated the imputation scheme.

Most of the loss is sites missing in one or two samples, so we split by severity:

- **≤10% of alleles missing** → `chr{N}.sample.mask.bed` marks each (site, sample), and the
  workflow fills them by binomial draw from the observed allele frequency at that site.
- **>10%** → left missing, dropped as before, counted toward the mutation rate.

Over accessible sequence the discarded fraction (where over 10% of samples have missing variant) is only 0.6–2.4% of sites, so nearly all of the
loss is recovered and the spread across sets largely closes.

Two things make this safe. A binomial draw is unbiased in expectation — it widens the frequency
spectrum through added variance rather than shifting it, unlike majority-allele fill. And the
mask does double duty: the workflow also uses it to clip imputed ancestry back out of the final
tree sequence and to exclude those spans from dating, so imputed calls inform topology without
passing as observed data.

The assumption is that missingness is independent of genotype within a site. Missingness here
is site-driven rather than genotype-driven, which makes that reasonable, but it is an
assumption. Note also that the callset contains no half-missing genotypes (`./.` only), so
imputation can never overwrite an observed allele — `make_sample_masks.py` aborts if one ever
appears.

## Design decisions

**The mask is stored complemented.** The published HGDP mask is *positive* — its intervals are
the accessible regions. The workflow treats every interval it reads as inaccessible, so storing
the published polarity would invert the analysis with no error. After complementing, 74.11% of
autosomal sequence is accessible, ranging from 48.08% (chr22) to 82.09% (chr3). chr22 is lowest
because its acrocentric p-arm is unmappable satellite — biology, not a bug.

**No `filter.txt`.** Sites removed by HGDP's own quality filters before phasing can't be
recovered from the statphase VCF; getting them would mean streaming ~250 GB of full callsets to
recover 0.46% of sites. The workflow derives its own filtered-position list from our VCFs
anyway, which for these subsets is much the larger effect. Writing one now would also
double-count the >10%-missing sites, which the workflow already handles.

**Unpolarised sites are kept.** Supplying the ancestral FASTA overrides the `polarised` config
flag. Ensembl's high- and low-confidence calls are used identically; `N`, `-` and `.` fall
through to unpolarised and are handed to SINGER's per-site polarisation rather than discarded.

**deCODE 2019 sex-averaged** was chosen over stdpopsim's `DeCodeSexAveraged_GRCh38` (which is
the older Kong et al. 2010 map lifted over). The published map starts inside each chromosome and ends before its end,
so each file gets a leading row at position 1 and a terminal row at full chromosome length. The
leading interval reads as missing data, so the first chunk is skipped.

That skip is a small part of a larger pattern. On chr22, 18 of 53 chunks are dropped, almost
all of them in the acrocentric p-arm where the sequence is inaccessible *and* deCODE has no
markers — the `zero recombination rate` and `masked sequence > 0.75` criteria fire on the same
18 chunks, and `min-sites: 500` adds nothing independent. Expect roughly two-thirds of chunks
retained on chr22, and better on the large chromosomes, which have no comparable unmappable
arm.

## Verification

Two scripts, both run automatically at the end of the stage that produces what they check, and
both safe to re-run at any time:

```bash
python scripts/verify_resources.py --resources resources   # 200 checks, shared resources
python scripts/verify_sets.py --root .                     # 2320 checks, all 8 sets
python scripts/test_max_masked_samples.py --root .         # the config-trap regression test
```

Worth re-running whenever anything is rebuilt, since each covers a failure that is either silent
or very late:

- hapmap length == ancestral FASTA length == `chrom.sizes`, **exactly**. The workflow asserts
  equality rather than sufficiency, and would otherwise fail hours into a run.
- Each ancestral FASTA is a single record matching `chrom.sizes`.
- Cumulative cM non-decreasing, positions strictly increasing, mean rate 0.5–3 cM/Mb.
- Mask accessible fraction within 40–95% — this is the check that catches inverted polarity.
- `meta.csv` order matches `bcftools query -l` element by element. The workflow pairs metadata
  to samples by row index with no ID matching, so a wrong order mislabels every sample
  silently.
- Sample-mask sanity, including that the observed maximum stays under `max-masked-samples`.

## Running SINGER

```bash
# one set, one chromosome
bash runs/run_singer.sh --chrom chr21 ALL --profile caslake_profile

# one set, whole genome (omit --chrom)
bash runs/run_singer.sh ALL --profile caslake_profile

# all 8 sets, one chromosome
bash runs/run_singer.sh --chrom chr22 --profile caslake_profile

# quick smoke test: fewer MCMC samples, writes to <SET>_pilot/
bash runs/run_singer.sh --pilot --chrom chr22 OCEANIA

# then cut the final posterior draw into CoaSTT's input directory
bash runs/extract_chunks.sh ALL
```

Output per set:

```
results/<SET>/
  singer-snakemake_outputs/     chr1/ ... chr22/, each with trees/ plots/ stats/ chunks/
  ts_chunks/                    chr21_10519382_12519382.trees ...   <- CoaSTT reads this
```

`ts_chunks/` is flat — every chromosome's chunks in one directory, no subdirectories — because
`CountCoalEvents` globs `*.trees` and turns each filename into a contig id. The files are the
final posterior draw cut at the chunk boundaries SINGER actually ran on, so they keep true
chromosomal coordinates, and only kept chunks appear.

### Configuration

**`singer_config.yaml` is the single source of truth for every scientific parameter.** There are
no per-set config files. `run_singer.sh` overrides only `input-dir`, `output-dir` and
`chromosomes` via `--config`, which snakemake applies after `--configfile`. Change a rate or an
MCMC length in that one file and every subsequent run picks it up.

Current settings: 2 Mb chunks, `min-sites: 1500`, **16 MCMC samples × 100 thinning = 1600
iterations**. Note SINGER's chunk log prints one line per *retained sample*, not per iteration —
"iteration 15" means all 16 samples are done, with the 100 thinning steps happening internally.

Two documented exceptions, both applied by the launcher rather than a second config file:

- **ALL is stratified by `region`, not `population`.** It holds 4 samples from each of 54
  populations, so stratifying by population gives 1,485 cross-coalescence pairs each backed by
  just 8 haplotypes — too sparse to estimate. The 7 regions give 28 pairs with real support.
  Every other set covers a single region, where `population` is the informative level.
- **`--pilot` cuts the number of MCMC samples** but keeps the thinning interval, so a pilot draw
  comes from the same chain behaviour as a production one. It writes to `<SET>_pilot/` so it
  cannot overwrite production output.

### Cluster profiles

Two, selected with `--profile`:

- **`caslake_profile/`** (recommended) — the shared partition: 9,120 CPUs, 190 nodes. Every rule
  is single-threaded, so each job asks for **one CPU** and memory sized from measured `MaxRSS`.
  Small requests backfill into gaps that large ones wait behind, which matters on a queue shared
  with other users.

  | rule | measured max | requested |
  |---|---|---|
  | `run_singer` | 5.8 h / 5.4 GB | 10 h / 8 GB |
  | `run_polegon` | 1.2 min / 346 MB | 30 min / 1 GB |
  | `chunk_chromosomes` | ~2 min / ~35 MB (chr22) | 4 h / 8 GB |

  `chunk_chromosomes` keeps generous memory deliberately: it is the one rule whose cost scales
  with chromosome length, building a full-chromosome numpy array from the ancestral FASTA plus
  the genotype matrix, and chr1 is 5× chr21.

  **⚠ caslake's QOS caps a single job at 36 h**, even though the partition advertises
  `MaxTime=UNLIMITED`. Exceed it and jobs sit in `PENDING` forever with reason
  `QOSMaxWallDurationPerJobLimit` — not slow, unschedulable. Check with
  `sacctmgr show qos caslake format=MaxWall`.

- **`slurm_profile/`** — the dedicated `jnovembre` partition, 192 CPUs. Blanket 16 GB per job
  and a 12 h walltime we control. Simpler, but the coarse request wastes memory and the job
  count is capped by a much smaller partition.

### Run state

Each set gets its own workdir under `runs/work/<SET>/`. Snakemake keeps run state in
`<workdir>/.snakemake`, and concurrent runs sharing a directory race on `.snakemake/iocache` —
`--nolock` is not sufficient. The launcher handles this.

Two related gotchas, both learned the hard way:

- **Do not delete or move a config file while a run is in flight.** Snakemake's SLURM executor
  re-reads `--configfile` inside *every* submitted job, not just the driver, so removing it
  mid-run fails every job that starts afterwards.
- **A killed driver leaves a stale lock.** Clear it with the same invocation plus `--unlock`
  before relaunching.

### Scale

For ALL (n=216) at 2 Mb chunks and 1600 iterations: **1,327 kept chunks genome-wide** (of 1,518
created; retention is 87–94% on the large chromosomes and drops to ~59% on chr22, whose
acrocentric p-arm is unmappable). Roughly **4 h per chunk** end to end and **~5,300 CPU-h** for
the genome, which is about a day of wall-clock at 500 concurrent jobs.

Note the job count snakemake prints at startup is a large underestimate: `chunk_chromosomes` is
a checkpoint, so per-chunk and per-replicate jobs only join the DAG after it runs. A whole-genome
ALL run starts by reporting 21 jobs and expands to ~24,000.

Expect one benign warning per run: `Failed to solve scheduling problem with ILP solver, falling
back to greedy scheduler`. It affects job ordering only, not results.

## Known issues

**`max-masked-samples` is very important and defaults to a value that breaks this setup.** At the
default of `0`, any position covered by a mask is marked globally inaccessible — silently
deleting the sites the mask exists to rescue. `singer_config.yaml` sets it to 999,
`verify_sets.py` asserts it exceeds the observed maximum, and
`scripts/test_max_masked_samples.py` is a permanent regression test covering both branches.
Don't copy this config without that value.

**Stock tskit, not the pinned fork.** `singer-snakemake/requirements.txt` pins
`nspope/tskit@nsp-paircoal-partial-missing`; `CoaSTT_env` has stock tskit 1.0.3. The fork
corrects how pair-coalescence rates are *normalised* under partial missingness: stock divides by
all sample pairs including unobserved ones, the fork by pairs actually observed, which biases
stock's rates low (~17% on a test with isolated samples).

This affects singer-snakemake's own diagnostic output — `plots/pair-coalescence-rates.png` and
the `stats/*.coalrate.p` files — so treat those as biased low where chunks carry isolated
samples. It does **not** reach CoaSTT's rates: `src/CountCoalEvents.py` calls
`pair_coalescence_counts(..., span_normalise=False)` and does its own normalisation, so the
divergent code path is never taken.

**Sample masks include inaccessible positions — confirmed.** `make_sample_masks.py` streams the
whole VCF without consulting the accessibility mask, so it marks sites that will be discarded
anyway. Measured on chr22: 10,886 of 17,184 mask lines for OCEANIA (63.3%) and 374,253 of
575,394 for EAST_ASIA (65.0%).

The output is unaffected — those sites are dropped at `filter_accessible` whether or not we
mask them — but roughly two-thirds of the 3.8 GB of masks is wasted disk and parse time. The fix
is to intersect against `resources/grch38/mask/` inside `make_sample_masks.py`; worth doing only
when the masks are being regenerated for another reason. Re-test with:

```bash
bedtools intersect -a SingerSnakemake_SampleSets/<SET>/input/chr22.sample.mask.bed \
                   -b resources/grch38/mask/chr22.mask.bed -u | wc -l
```