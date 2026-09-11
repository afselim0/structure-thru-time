# HGDP data preparation for SINGER

Everything upstream of ARG inference on the statistically phased HGDP dataset (Bergström et
al. 2020, 929 diverse genomes, GRCh38): download the source data, derive the shared
per-chromosome inputs, and build 8 ready-to-run input directories for
[`singer-snakemake`](https://github.com/nspope/singer-snakemake) (cloned at `../singer-snakemake`).
Running the workflow itself is not part of this.

## Running

Needs `conda activate CoaSTT_env`.

```bash
bash download.sh                 # raw downloads + shared resources + verification
bash subset.sh                   # the 8 input directories (local, 4 at a time)
bash subset.sh --jobs 8
bash subset.sh --slurm-partition jnovembre --slurm-account pi-jnovembre   # one 8-task array
```

Both are idempotent: steps are skipped when their derived artefacts exist, so an interrupted
run resumes. `download.sh` takes a few hours, dominated by the 6.3 GB VCF download and the
per-chromosome split; `subset.sh` is minutes per set, which is why local is the default.

**One file must be fetched by hand.** science.org serves its supplements behind bot
protection and returns HTTP 403 to every automated request, so the deCODE map cannot be
downloaded by script. Get *Data S3* from
[doi:10.1126/science.aau1043](https://www.science.org/doi/10.1126/science.aau1043) and place it
at `resources/raw/aau1043_datas3.gz` (gzip it if it arrives uncompressed). `download.sh` keeps
the fetch line commented out and fails with these instructions if the file is absent.

## Downloads

| Source | URL | Derived artefact |
|---|---|---|
| HGDP statphase VCF + index | `ngs.sanger.ac.uk/production/hgdp/hgdp_wgs.20190516/statphase/` | `resources/hgdp/statphase/chr{N}.bcf` — GT-only, per chromosome |
| Accessibility mask | `…/accessibility-mask/` | `resources/grch38/mask/chr{N}.mask.bed` — **complemented** |
| Sample metadata | `…/metadata/hgdp_wgs.20190516.metadata.txt` | `resources/hgdp/metadata.tsv`, `resources/hgdp/sets/*.csv` |
| Ensembl EPO ancestral alleles (release 113) | `ftp.ensembl.org/pub/release-113/fasta/ancestral_alleles/` | `resources/grch38/ancestral/chr{N}.ancestral.fa.gz`, `resources/grch38/chrom.sizes` |
| deCODE 2019 sex-averaged map (manual) | doi:10.1126/science.aau1043, Data S3 | `resources/grch38/hapmap/chr{N}.hapmap` |

Sizes and sha256 for each are recorded in `prep.yaml`. Sanger publishes no checksums, so the
VCF's integrity is established by agreement between its bgzip index and its data.

## What is verified

`scripts/verify_resources.py` (run at the end of `download.sh`, 200 checks) and
`scripts/verify_sets.py` (run at the end of `subset.sh`). These are invariants worth re-running
whenever the resources are rebuilt, because each covers a failure that is silent or very late:

- hapmap `sequence_length` == ancestral FASTA length == `chrom.sizes`, **exactly** — the
  workflow asserts equality, not sufficiency, and would otherwise fail hours into a run
- each ancestral FASTA is a single record whose length matches `chrom.sizes`
- cumulative cM non-decreasing, positions strictly increasing, mean rate 0.5–3 cM/Mb
- mask accessible fraction within 40–95%, catching inverted polarity
- mask intervals within chromosome bounds
- per set: `meta.csv` order matches `bcftools query -l` element by element, row counts match,
  all five per-chromosome files present with symlinks resolving, no duplicate positions

## The 8 sample sets

Seven regional sets take every sample in the region: `AFRICA` (104), `EUROPE` (155),
`CENTRAL_SOUTH_ASIA` (197), `EAST_ASIA` (223), `OCEANIA` (28), `AMERICA` (61), `MIDDLE_EAST`
(161).

`ALL` (216) is the **4 highest-coverage samples from each of the 54 populations**, ties broken
by sample id. SINGER's cost is quadratic in sample count, so all 929 is prohibitive, and an
unweighted subsample would be dominated by the densely sampled regions. Every population has
at least 4 samples, so the quota is met exactly.

Each `SingerSnakemake_SampleSets/<set>/input/` holds, per chromosome:

```
chr{N}.vcf.gz(.tbi)      subset, re-trimmed to segregating biallelic sites
chr{N}.mask.bed          -> symlink to the shared inaccessible-interval mask
chr{N}.hapmap            -> symlink to the shared recombination map
chr{N}.ancestral.fa.gz   -> symlink to the shared ancestral FASTA
chr{N}.meta.csv          sample metadata IN VCF ORDER
```

The workflow globs `{chrom}.vcf.gz` and finds every other input by string substitution on that
path, so all five share a stem and sit in one directory. The three shared files are identical
across sets, hence symlinks rather than 8 copies.

## Decisions

**The mask is stored complemented.** The published HGDP mask is a *positive* mask — its
intervals are the accessible regions. `chunk_chromosomes.py` does `bitmask[a:b] = True` for
every interval it reads, where `True` means *inaccessible*, so the file must list the regions
to exclude. Storing the published polarity would invert the analysis silently. Measured
accessible fraction after complementing is 74.11% of autosomal sequence, ranging 48.08%
(chr22) to 82.09% (chr3) — chr22 is lowest because its entire acrocentric p-arm
(0–10,661,594) is unmappable satellite, which is biology rather than a polarity error.

**No `filter.txt`.** Sites removed by HGDP's quality filters before phasing aren't recoverable
from the statphase VCF, and getting them would mean streaming ~250 GB of full callsets. The
workflow still derives `filtered_positions` from our own VCFs (multiallelic sites and sites
with missing genotypes), which for these subsets is much the larger effect.

This stays true under the imputation scheme below, and writing one would now be actively wrong.
`chunk_chromosomes.py:317-321` already appends every site failing `filter_nonmissing` to
`filtered_positions`, so the >10%-missing sites we deliberately leave unimputed are counted
toward the mutation-rate adjustment automatically. Listing them in a `filter.txt` as well would
double-count them.

**Lightly-missing genotypes are imputed; heavily-missing sites are dropped.** A substantial
minority of sites that are accessible and segregating in the subset carry at least one missing
genotype. Genome-wide over all 22 autosomes:

| set | n | missing | | set | n | missing |
|---|---|---|---|---|---|---|
| OCEANIA | 28 | 6.4% | | EUROPE | 155 | 27.9% |
| AMERICA | 61 | 12.3% | | MIDDLE_EAST | 161 | 28.1% |
| ALL | 216 | 19.2% | | CENTRAL_SOUTH_ASIA | 197 | 35.3% |
| AFRICA | 104 | 23.7% | | EAST_ASIA | 223 | 41.3% |

It is a per-site union over samples, so it rises with set size — but size does not order the
sets. EAST_ASIA (n=223) is highest at 41.3% while ALL (n=216) is less than half that at almost
the same sample count, because ALL spans 54 populations and so has far more segregating sites
(24.6M vs 14.3M) to spread the same missing calls across. Treat it as a property of set size ×
diversity, never of the dataset. It also varies by chromosome: chr22 reads 35–45% higher than
genome-wide, so single-chromosome pilots overstate it.

`filter_nonmissing` has zero tolerance, so every one of those sites would be dropped outright.
For EAST_ASIA that is over 40% of otherwise-usable sites, and most of the loss is sites missing
in only one or two samples, so we split by how much is missing:

- **≤10% of alleles missing → imputed.** We emit `chr{N}.sample.mask.bed` marking each missing
  (site, sample), and the workflow fills them by binomial draw from the observed allele
  frequency at that site.
- **>10% → left missing**, dropped by `filter_nonmissing` and added to `filtered_positions`
  automatically, which scales `adj_mu`. Measured over accessible sequence this tail is only
  0.6–2.4% of sites, so nearly all of the loss is recovered.

There are no half-missing genotypes in this callset — the only missing form is `./.` — so the
allele and sample denominators coincide and imputation can never overwrite an observed allele.
`make_sample_masks.py` aborts if one is ever found.

The mask is not only an imputation trigger: `merge_chunks.py` feeds it to
`remove_partial_ancestry`, which clips the imputed ancestry back out of the output tree
sequence, and `run_polegon.py` excludes those spans from dating. Imputed calls therefore inform
topology without masquerading as real data in what CoaSTT measures.

**⚠ `max-masked-samples` is load-bearing.** It defaults to `0`, and at that value *any* position
covered by a mask is marked globally inaccessible — silently deleting the sites the mask exists
to rescue. `singer_config.yaml` sets it to 999; `verify_sets.py` asserts it exceeds the observed
maximum, and `scripts/test_max_masked_samples.py` is a permanent regression test proving both
branches.

On the statistics: a binomial draw is **unbiased in expectation** — expected AC after imputation
is `AC_obs × 2n/(2n−m)`, the unbiased estimate of what complete data would have given. It
broadens the SFS through added variance (some singletons become doubletons and vice versa)
rather than shifting it. Majority-allele fill, which the earlier attempt in `.misc/human_data`
used, *does* shift toward the major allele, and that is the substantive reason to prefer this.
The assumption being made is **MCAR-within-site**, since the frequency is estimated from the
non-missing calls at that site; missingness here is site-driven rather than genotype-driven,
which makes that plausible. Neither approach can manufacture haplotype sharing, which is the
mechanism by which panel imputation would inject false coalescent structure into exactly what
CoaSTT measures.

**No `omit.txt`.** Considered for CpG sites and rejected: `adj_mu` has no term for omitted
variants — there is an explicit `FIXME` in `chunk_chromosomes.py` noting the bias is only
acceptable when omitted variants are few — and CpG transitions are 15–25% of human SNVs.

**Unpolarised sites are kept.** The ancestral FASTA overrides the `polarised` config flag
entirely. `np.char.upper` is applied to it, so Ensembl's high-confidence (uppercase) and
low-confidence (lowercase) calls are used identically; `N`, `-` and `.` fall through to
unpolarised and are handed to SINGER's own per-site polarisation rather than discarded.

**Ensembl 113 pinned**, because Ensembl is mid-migration and the `current_*` symlinks are
unstable; it is also the release used across comparable human ARG work.

**deCODE 2019 sex-averaged** chosen over stdpopsim's `DeCodeSexAveraged_GRCh38` (which is Kong
et al. 2010 lifted over) and over Pálsson et al. 2024 (1 Mb resolution, NCO map). The map
starts inside each chromosome and ends before its end, so each file gets a leading row at
position 1 and a terminal row at the full length; the leading interval is read as missing data,
which makes SINGER skip one chunk per chromosome start — expected, and visible in the logs.

**Stock tskit, not the pinned fork.** `singer-snakemake/requirements.txt` pins
`nspope/tskit@nsp-paircoal-partial-missing`; `CoaSTT_env` uses stock tskit 1.0.3. That fork is
not cosmetic — it corrects pair-coalescence rates under partial missingness — so this is a
known deviation to revisit before trusting rate estimates from these ARGs.
