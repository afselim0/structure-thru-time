# HGDP preparation for SINGER

Builds the input files that [`singer-snakemake`](https://github.com/nspope/singer-snakemake)
needs to infer ancestral recombination graphs from the statistically phased HGDP callset
(Bergström et al. 2020 — 929 genomes, GRCh38).

Two stages: download and derive shared per-chromosome resources, then cut those into 8
ready-to-run input directories, one per sample set. Then run the workflow — see *Running
SINGER* below. `singer-snakemake` is a git submodule at `../singer-snakemake`, pinned to the
commit these inputs were built and tested against; `git submodule update --init` after cloning.

## Quick start

```bash
conda activate CoaSTT_env

bash download.sh    # raw downloads, shared resources, verification (a few hours)
bash subset.sh      # the 8 input directories (minutes per set)
```

`subset.sh` runs 4 sets at a time locally. `--jobs 8` widens that; `--slurm-partition` and
`--slurm-account` together submit a single 8-task array instead.

**One file needs fetching by hand.** science.org serves supplements behind bot protection and
returns 403 to every scripted request, so the deCODE recombination map can't be automated. Grab
*Data S3* from [doi:10.1126/science.aau1043](https://www.science.org/doi/10.1126/science.aau1043)
and drop it at `resources/raw/aau1043_datas3.gz`. `download.sh` stops with these instructions
if it's missing.

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

Seven take every sample in a region: AFRICA (104), EUROPE (155), CENTRAL_SOUTH_ASIA (197),
EAST_ASIA (223), OCEANIA (28), AMERICA (61), MIDDLE_EAST (161).

ALL (216) is the 4 highest-coverage samples from each of the 54 populations, ties broken by
sample ID.

## Missing genotypes

This is the one substantive transformation applied to the genotype data, so it's worth
understanding.

SINGER's wrapper drops any site with even a single missing genotype. That's a lot of sites —
per-site union over samples, across all 22 autosomes:

| set | n | affected | | set | n | affected |
|---|---|---|---|---|---|---|
| OCEANIA | 28 | 6.4% | | EUROPE | 155 | 27.9% |
| AMERICA | 61 | 12.3% | | MIDDLE_EAST | 161 | 28.1% |
| ALL | 216 | 19.2% | | CENTRAL_SOUTH_ASIA | 197 | 35.3% |
| AFRICA | 104 | 23.7% | | EAST_ASIA | 223 | 41.3% |

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

Everything needed to run the workflow on these inputs lives here:

```bash
bash runs/run_chr22.sh              # all 8 sets, chr22
bash runs/run_chr22.sh OCEANIA      # one set
```

- `slurm_profile/` — the SLURM profile. Its `mem_mb: 16000` default is deliberate:
  `chunk_chromosomes` builds a full-chromosome numpy array from the ancestral FASTA (~1 GB for
  chr1) on top of the genotype matrix, so the 4000 MB that upstream's `SLURM.md` suggests will
  OOM. No rule declares its own `resources:`, so every job inherits this.
- `runs/chr22_configs/<SET>.yaml` — one config per set, generated from `singer_config.yaml` with
  `input-dir`, `output-dir` and `chromosomes` overridden.
- `runs/work/<SET>/` — **each set needs its own workdir.** Snakemake keeps run state in
  `<workdir>/.snakemake`, and concurrent runs sharing a directory race on `.snakemake/iocache`;
  `--nolock` is not sufficient. The launcher handles this.

A pilot at `singer-mcmc-samples: 10` completes chr22/OCEANIA in about 9 minutes and is the
fastest way to confirm a change end to end. Note the job count snakemake prints at startup is an
underestimate: `chunk_chromosomes` is a checkpoint, so the per-chunk and per-replicate jobs are
only added to the DAG once it has run — the pilot starts by reporting 44 jobs and finishes at
429. Production runs use 100 replicates, roughly 10x that again.

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