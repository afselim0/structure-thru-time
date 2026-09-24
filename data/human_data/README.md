# HGDP preparation for SINGER

Builds the input files that [`singer-snakemake`](https://github.com/nspope/singer-snakemake)
needs to infer ancestral recombination graphs from the statistically phased HGDP callset
(Bergström et al. 2020 — 929 genomes, GRCh38).

Two stages: download and derive shared per-chromosome resources, then cut those into 8
ready-to-run input directories, one per sample set. Then run the workflow — see *Running
SINGER* below. `singer-snakemake` is a git submodule at `../singer-snakemake`, pinned to the
commit these inputs were built and tested against; `git submodule update --init` after cloning.

> **A local fork of the submodule exists but is NOT in use.** Branch
> `coastt-batched-polegon` (`dbd1509`) makes `run_polegon` date all draws of a chunk in one
> job instead of one job per draw. It is correct and tested, but the submodule is pinned to
> upstream `1d6b370` because the same throughput problem is solved more simply by keeping the
> draw count low (see `singer-mcmc-samples` in `drivers/singer_config.yaml`). Revisit the fork
> if a future run needs many draws.

## Layout

```
README.md                        this file — the only file at the top level
drivers/                         run these in order; the numbers are the pipeline
  1_download.sh                    fetch + derive the shared per-chromosome resources
  2_subset.sh                      cut them into the 8 per-set input directories
  3_run_singer.sh                  infer ARGs with SINGER, date them with POLEGON
  4_extract_chunks.sh              cut the posterior draw into CoaSTT's input directory
  _common.sh                       shared helpers, sourced by the four above
  singer_config.yaml               ALL scientific parameters, one source of truth
  slurm_caslake.config             cluster settings: shared partition
  slurm_jnovembre.config           cluster settings: dedicated partition (default)
  prep.yaml                        recorded decisions, provenance, known issues
scripts/                         the implementation each driver calls
  download.sh, subset.sh           steps 1-2
  run_singer.sh, extract_chunks.sh steps 3-4
  make_*.py, verify_*.py, ...      the per-stage work
resources/                       downloaded + derived shared data          (gitignored)
SingerSnakemake_SampleSets/<SET>/  per-set workflow input, flat            (gitignored)
results/<SET>/
  singer-snakemake_outputs/        the workflow's own output tree          (gitignored)
  ts_chunks/                       CoaSTT's input directory                (gitignored)
logs/                            run logs                                  (gitignored)
```

Snakemake run state is deliberately **not** here — it lives under `$SCRATCH` (see *Run state*).

## The whole pipeline, start to finish

**Run the numbered drivers in order.** Each one documents at the top what it needs, what it
writes and how long it takes, and ends by naming the next. Every stage is idempotent — it skips
work whose output already exists — so an interrupted run resumes rather than restarting.

```bash
conda activate CoaSTT_env

bash drivers/1_download.sh                      # a few hours, ~15 GB
bash drivers/2_subset.sh                        # ~15 min per set
bash drivers/3_run_singer.sh ALL                # ~14 h wall clock, whole genome
bash drivers/4_extract_chunks.sh ALL            # minutes
```

Steps 1 and 2 are the same for everybody and do not touch the cluster; they run once, not once
per sample set. Step 3 is the only one that submits SLURM jobs, and the only one that takes a
cluster config:

```bash
bash drivers/3_run_singer.sh ALL --config slurm_caslake      # shared partition
bash drivers/3_run_singer.sh ALL --config slurm_jnovembre    # dedicated partition (default)
```

**To run this on your own cluster**, copy one of the `drivers/slurm_*.config` files, change the
account and partition in it, and pass its name to step 3. Nothing else changes. See *Adapting
this to your own cluster*.

Step 3 does not block — it launches a detached snakemake and returns, so monitor it with
`tail -f logs/<SET>.snakemake.log` and `squeue -u $USER`, then run step 4 once it is done.
Steps 1, 2 and 4 do block.

What each stage actually does, in order:

**1. `drivers/1_download.sh`** — seven steps, then verification:

| step | script | produces |
|---|---|---|
| 1 | *(curl)* | `resources/raw/` — statphase VCF, mask, metadata, ancestral tarball |
| 2 | `bcftools index -n` | integrity check: 929 samples against the published index |
| 3 | `make_ancestral.py` | `grch38/ancestral/chr{N}.ancestral.fa.gz` + `chrom.sizes` |
| 4 | `make_masks.py` | `grch38/mask/chr{N}.mask.bed` — **complemented** to inaccessible |
| 5 | `make_hapmaps.py` | `grch38/hapmap/chr{N}.hapmap` from the deCODE map |
| 6 | `make_metadata.py`, `make_sample_sets.py`, `HGDP_color_palette.py` | `hgdp/metadata.tsv`, `hgdp/sets/*.csv`, `population_labels_colors/palette.{csv,png}` |
| 7 | `split_statphase.sh` | `hgdp/statphase/chr{N}.bcf` — GT-only, per chromosome |
| — | `verify_resources.py` | 200 checks (see *Verification*) |

The deCODE map must be placed by hand first — see the note below.

**2. `drivers/2_subset.sh`** — runs `make_set_inputs.sh` per set (locally via `xargs -P`, or as an
8-task SLURM array), then `verify_sets.py`. Per set, per chromosome:

```
bcftools view -S <set>.csv -a -c 1 -v snps   →  chr{N}.vcf.gz   (+ index)
ln -s  mask.bed / hapmap / ancestral.fa.gz   →  shared, symlinked
make_sample_masks.py                         →  chr{N}.sample.mask.bed
write_set_metadata.py                        →  chr{N}.meta.csv  (in VCF order — order matters)
```

**3. `drivers/3_run_singer.sh`** — one `snakemake` call against
`../singer-snakemake/workflow/Snakefile`, with `drivers/singer_config.yaml` as the configfile
and only `input-dir`, `output-dir` and `chromosomes` overridden per run. Writes
`results/<SET>/singer-snakemake_outputs/chr{N}/` with `trees/ plots/ stats/ chunks/`. Takes
`--config`, `--chrom`, `--pilot` and `-n`; with no set named it launches all 8 at once.

**4. `drivers/4_extract_chunks.sh`** — calls `extract_ts_chunks.py`, which takes the
highest-numbered replicate (the most burnt-in posterior draw) for each chromosome, cuts it at
the chunk boundaries in `chr{N}.chunks.p`, and writes
`results/<SET>/ts_chunks/chr{N}_{start}_{end}.trees` — flat, all chromosomes together, which is
what CoaSTT globs.

It writes **one** draw per chunk: CoaSTT treats each `.trees` file as an independent contig, so
several draws of the same region would enter that region into the likelihood twice. And it
refuses two situations that would otherwise pass silently — a draw from inside the burn-in
implied by `drivers/singer_config.yaml`, and chromosomes disagreeing on which replicate is
their latest (a killed merge job), which would mix posterior samples of different ages.

**5. `scripts/make_labels.sh`** — writes the files CoaSTT labels and colours a run with, into
`resources/population_labels_colors/`:

```
HGDP_color_palette.py   →  global_palette.{csv,png}      region hue + population ramp
                        →  regional_palette.{csv,png}    discrete within each region
make_sample_sets.py     →  <SET>_pop_labels.csv          one row per HAPLOTYPE
                        →  <SET>_pop_colors.csv          population,color
```

`pop_labels.csv` carries `sample_id` (the tree sequence's own sample/node index),
`individual_id`, `HGDP_sample_id`, `HGDP_individual_id`, `population` and `region`. Row order
is the sample order of the tree sequences, taken from each set's `chr1.meta.csv` — the VCF
order SINGER saw, not the sorted set definition. A tree sequence pairs individual *i* with
nodes *2i* and *2i+1*, so the two haplotype rows of an individual are consecutive. When
`results/<SET>/ts_chunks/` exists the order is asserted against a real tree sequence, because a
wrong order would mislabel every sample rather than raise.

There are TWO palettes, because a plot of all seven regions and a plot of one region need
different things from colour. `global_palette` is two-level -- seven region anchors, then a
within-region ramp ordered along the region's principal geographic axis -- so a colour says
which region a population is in; `ALL_pop_colors.csv` uses it. `regional_palette` instead
gives each region its own max-separation palette drawn from the whole gamut, since a
single-region plot has no other region to distinguish from and only needs its populations
told apart; the seven `<REGION>_pop_colors.csv` use it. Minimum separation within a region
goes from dE 0.3-3.6 (global) to dE 8.8-37.1 (regional). The same population therefore has a
different colour in ALL than in its own region's set, by design. Both assign colours in
geographic order, so a legend still reads west-to-east or south-to-north.
The global anchors alternate dark and light on purpose — protanopia and deuteranopia collapse
the red-green axis, so lightness, not hue, is what separates the regions for those viewers.
East Asia's 18 populations exceed what one hue can hold, so it carries a second anchor
internally (`REGION_SUBSPLITS`, cut at 33°N); this is palette-internal and the `region` column
stays `EAST_ASIA` throughout. The script re-checks every anchor on each run and fails if an
edit breaks the floor of ΔE 6. Both PNGs are always regenerated, including on a failing
palette, so a retune can be inspected.

### Before you start: one file must be fetched by hand

science.org serves its supplements behind bot protection and returns 403 to every scripted
request, so the deCODE recombination map cannot be downloaded automatically. Grab *Data S3*
from [doi:10.1126/science.aau1043](https://www.science.org/doi/10.1126/science.aau1043) and put
it at `resources/raw/aau1043_datas3.gz` (gzip it if it arrives uncompressed). `drivers/1_download.sh`
stops with these instructions if it is missing.

### Variations

```bash
bash drivers/2_subset.sh --jobs 8                        # all 8 sets at once, locally
bash drivers/2_subset.sh --slurm-partition jnovembre \
                         --slurm-account pi-jnovembre    # as one 8-task SLURM array

bash drivers/3_run_singer.sh ALL -n                      # dry run: the plan, submit nothing
bash drivers/3_run_singer.sh ALL --chrom chr21           # one chromosome
bash drivers/3_run_singer.sh OCEANIA --pilot --chrom chr22   # smoke test, reduced MCMC
bash drivers/3_run_singer.sh                             # all 8 sets at once

bash drivers/4_extract_chunks.sh                         # every finished set
```

Start with `--pilot --chrom chr22` on a small set the first time you run this anywhere new: it
exercises every rule in the workflow in minutes rather than hours.

`--config` selects the cluster settings by the basename of a `drivers/*.config` file;
`slurm_jnovembre` is the default if omitted. See *Cluster profiles* below, and *Adapting this to
your own cluster* if you are not on midway3.

## Sources

| What | Where | Becomes |
|---|---|---|
| Phased VCF + index | Sanger `hgdp_wgs.20190516/statphase/` | `resources/hgdp/statphase/chr{N}.bcf` (GT-only) |
| Accessibility mask | Sanger `…/accessibility-mask/` | `resources/grch38/mask/chr{N}.mask.bed` |
| Sample metadata | Sanger `…/metadata/` | `resources/hgdp/metadata.tsv` |
| Ancestral alleles | Ensembl release 113, EPO | `resources/grch38/ancestral/chr{N}.ancestral.fa.gz` |
| Recombination map | deCODE 2019, Data S3 (manual) | `resources/grch38/hapmap/chr{N}.hapmap` |

Sizes and SHA-256 for every download are in `drivers/prep.yaml`. Sanger publishes no checksums, so the
VCF is validated by agreement between its index and its data instead.

## What each input directory contains

`singer-snakemake` globs for `{chrom}.vcf.gz` and locates everything else by substituting on
that filename, so all five files share a stem and sit together:

```
SingerSnakemake_SampleSets/<SET>/
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
bash drivers/3_run_singer.sh ALL --chrom chr21 --config slurm_caslake

# one set, whole genome (omit --chrom)
bash drivers/3_run_singer.sh ALL --config slurm_caslake

# all 8 sets, one chromosome
bash drivers/3_run_singer.sh --chrom chr22 --config slurm_caslake

# quick smoke test: fewer MCMC samples, writes to <SET>_pilot/
bash drivers/3_run_singer.sh OCEANIA --pilot --chrom chr22

# then cut the final posterior draw into CoaSTT's input directory
bash drivers/4_extract_chunks.sh ALL
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

**`drivers/singer_config.yaml` is the single source of truth for every scientific parameter.** There are
no per-set config files. `run_singer.sh` overrides only `input-dir`, `output-dir` and
`chromosomes` via `--config`, which snakemake applies after `--configfile`. Change a rate or an
MCMC length in that one file and every subsequent run picks it up.

Current settings: 2 Mb chunks, `min-sites: 1500`, **20 MCMC samples × 250 thinning = 5,000
iterations**, with `singer-mcmc-burnin: 0.6`. SINGER's supplementary figure 10 puts the burn-in
requirement at roughly 4,000 iterations, so the draw CoaSTT consumes — the last one, at
iteration 5,000 — is well past it. The burn-in fraction drops the first 12 draws from the
*statistics and diagnostic plots*, which are then built from iterations 3,250–5,000.

**Why 20 draws at thin 250 rather than 100 at thin 50.** Both give 5,000 iterations, and since
extraction feeds CoaSTT the highest draw, that draw is at iteration 5,000 either way. But
POLEGON runs once per *(chunk, draw)* and `merge_chunks` once per *(chromosome, draw)*, so the
draw count multiplies everything after SINGER:

| | n=100 / thin 50 | n=20 / thin 250 |
|---|---|---|
| SINGER CPU-h | 6,634 | 6,634 *(identical)* |
| POLEGON jobs | 133,700 | **26,740** |
| merge jobs | 2,200 | **440** |
| tree storage | 298 GB | **60 GB** |
| wall clock after SINGER | ~38 h | **~7.5 h** |

At n=100 a whole-genome set is 133,700 POLEGON jobs of ~10 s each — pathological for SLURM,
where submission and teardown cost as much as the work, so throughput is capped by the job
submission rate rather than the partition (measured: ~3,700 jobs/h). Draw spacing is also
better at thin 250: retained draws are less autocorrelated.

**A SINGER chunk log contains two different `Iteration:` counters**, and reading the first as
MCMC is how an earlier run was set 3× too short:

- The **first** sequence counts **ARG construction** — SINGER threads in one haplotype at a
  time. It runs to `2 × samples − 1` (431 for this set's 216 diploids) and is fixed by sample
  size, not by any MCMC setting. It is not burn-in.
- The **second**, restarting at `Iteration: 0` after `rescaled ARG length`, is one entry per
  **retained draw**. That is the MCMC.

**`singer-mcmc-burnin` only affects statistics and diagnostic plots.** `merge_chunks` writes a
tree sequence for *every* draw, pre-convergence ones included, so nothing in the workflow stops
an unconverged draw reaching CoaSTT. `extract_ts_chunks.py` enforces the floor instead.

Two documented exceptions, both applied by the launcher rather than a second config file:

- **ALL is stratified by `region`, not `population`.** It holds 4 samples from each of 54
  populations, so stratifying by population gives 1,485 cross-coalescence pairs each backed by
  just 8 haplotypes — too sparse to estimate. The 7 regions give 28 pairs with real support.
  Every other set covers a single region, where `population` is the informative level.
- **`--pilot` cuts the number of MCMC samples** but keeps the thinning interval, so a pilot draw
  comes from the same chain behaviour as a production one. It writes to `<SET>_pilot/` so it
  cannot overwrite production output.

### Cluster profiles

Pick one with `--profile`. Both request **one CPU per job** — every rule in this workflow is
single-threaded, so that is what the work actually is, not a compromise — and both size memory
and walltime from measured usage rather than guesses:

| rule | measured max | requested |
|---|---|---|
| `run_singer` | 5.8 h / 5.4 GB | 10–12 h / 8 GB |
| `run_polegon` | 1.2 min / 346 MB | 30 min / 1 GB |
| `chunk_chromosomes` | ~2 min / ~35 MB (chr22) | 4 h / 8 GB |
| everything else | ~200 MB | 4 GB |

`chunk_chromosomes` keeps generous memory deliberately: it is the one rule whose cost scales
with chromosome length, building a full-chromosome numpy array from the ancestral FASTA plus the
genotype matrix, and chr1 is 5× chr21. Everything else runs on fixed-size 2 Mb chunks, so its
cost does not grow with the chromosome.

**`drivers/slurm_caslake.config` — shared, high throughput.** 9,120 CPUs over 190 nodes, but
shared with the rest of the cluster, so jobs queue. Small requests backfill into gaps that large
ones wait behind, which is exactly why the per-rule sizing matters here. `jobs: 500` concurrent.

**`drivers/slurm_jnovembre.config` — dedicated, predictable.** The `jnovembre` partition, 192
CPUs over 3 nodes. Nothing else competes, so jobs start immediately, but throughput is capped by
the partition. `jobs: 190`. This is the default.

Rule of thumb: use `slurm_caslake` for a whole-genome run, `slurm_jnovembre` for a single
chromosome or when you want results without queue variance.

Both files are snakemake *profile bodies*. Snakemake's `--profile` takes a directory containing
a file named `config.yaml`, so the launcher copies the one you select into a
temporary directory under that name. Selecting one by basename is all the launcher needs:
`--config slurm_caslake`. To drive snakemake by hand, stage it yourself:

```bash
mkdir -p /tmp/prof && cp drivers/slurm_caslake.config /tmp/prof/config.yaml
snakemake ... --profile /tmp/prof
```

### Adapting this to your own cluster

Copy either file to `drivers/slurm_<yourcluster>.config` — the launcher will then accept
`--config slurm_<yourcluster>` — and change three things:

```yaml
default-resources:
    slurm_account: <your-account>       # sacctmgr show assoc user=$USER format=account
    slurm_partition: <your-partition>   # sinfo -s
```

```yaml
jobs: <n>        # concurrent jobs. Cap near the CPUs you can reasonably occupy;
                 # on a shared partition, well below its total.
```

Then check your walltime ceiling before trusting `runtime:`:

```bash
sinfo -p <partition> -o "%P %l"                          # partition limit
sacctmgr show qos <qos> format=Name,MaxWall              # QOS limit -- often lower
```

**The QOS limit is the one that bites.** caslake's partition advertises `MaxTime=UNLIMITED`
while its QOS caps a single job at 36 h; jnovembre's QOS allows 48 h. Asking for more than the
QOS allows does not slow a job down — it makes it **unschedulable**, sitting in `PENDING`
forever with reason `QOSMaxWallDurationPerJobLimit`. That is silent unless you look.

If your cluster has no SLURM, drop `--profile` and pass `--cores N` instead; snakemake will run
the same DAG locally.

Sanity-check any profile before committing a long run to it:

```bash
bash drivers/3_run_singer.sh OCEANIA --pilot --chrom chr22 --config slurm_<yourcluster>
```

### Run state

Each set gets its own workdir, **outside the repository** — by default
`$SCRATCH/singer_work/<SET>/` (set `WORKDIR_ROOT=...` to put it elsewhere). Snakemake keeps run
state in `<workdir>/.snakemake`, and concurrent runs sharing a directory race on
`.snakemake/iocache` — `--nolock` is not sufficient. The launcher handles this.

**This state is disposable.** Which chunks are finished is recorded by the `.singer.done`
sentinel files under `results/<SET>/singer-snakemake_outputs/`, not by the workdir, so deleting
a workdir loses no completed work: the next run rebuilds its DAG from `results/` and queues only
what is genuinely unfinished. That makes a fresh workdir the first thing to try when a driver is
behaving strangely — it is cheaper and safer than trying to repair accumulated state.

Two related gotchas, both learned the hard way:

- **Do not delete or move a config file while a run is in flight.** Snakemake's SLURM executor
  re-reads `--configfile` inside *every* submitted job, not just the driver, so removing it
  mid-run fails every job that starts afterwards.
- **A killed driver leaves a stale lock.** Clear it with the same invocation plus `--unlock`
  before relaunching.

### Scale

For ALL (n=216) at 2 Mb chunks: **1,327 kept chunks genome-wide** (of 1,518 created; retention
is 87–94% on the large chromosomes and drops to ~59% on chr22, whose acrocentric p-arm is
unmappable).

Cost splits cleanly into a fixed part and a per-iteration part. Both were measured over 553
completed chunks and reconcile with wall clock to within 1%:

| | measured |
|---|---|
| ARG construction | **23 min** per chunk, fixed by sample size |
| MCMC | **3.3 s** per iteration |

So a chunk costs `23 min + 3.3 s × iterations`. At the current 5,000 iterations that is **~4.96 h
per chunk** and **~6,700 CPU-h** genome-wide — about 14 h of wall clock at 500 concurrent jobs,
before queueing. Storage is ~60 GB.

Memory is measured too: p50 1.90 GB, p90 3.41, p99 4.93, max 6.81 GB over 576 jobs. `run_singer`
requests 12 GB and escalates to 20 and 28 GB on retry, so a chunk that needs more gets it
without the other 99% paying for the headroom.

Note the job count snakemake prints at startup is a large underestimate: `chunk_chromosomes` is
a checkpoint, so per-chunk and per-replicate jobs only join the DAG after it runs. A whole-genome
ALL run starts by reporting 21 jobs and expands to ~24,000.

Expect one benign warning per run: `Failed to solve scheduling problem with ILP solver, falling
back to greedy scheduler`. It affects job ordering only, not results.

## Known issues

**A SLURM node failure can poison every subsequent job submission — guarded, but know the
symptom.** `snakemake-executor-plugin-slurm` 2.8.0 reacts to a `NODE_FAIL` by recording the
failed node so later jobs avoid it. It reads the node from `sacct`, which reports
`"None assigned"` for a job that died before SLURM placed it anywhere. The executor stores that
string verbatim and puts it in every later submission:

```
sbatch ... --exclude=None assigned ...
sbatch: error: Batch job submission failed: Invalid node name specified
```

which the executor reports, confusingly, as `Unable to open file assigned`. It is a latch: one
node failure and nothing submits again for the life of the driver. Observed on a whole-genome
run — a single `NODE_FAIL` was followed by **8,522 consecutive failed submissions**, with SINGER
limping along on already-queued jobs while POLEGON produced nothing at all.

`scripts/slurm_executor_patch.py` validates node names before they enter the exclusion set, and
`scripts/sitecustomize.py` applies it automatically because `scripts/run_singer.sh` puts
`scripts/` on `PYTHONPATH`. **Launch through `drivers/3_run_singer.sh`** — invoking `snakemake`
directly skips the guard.

The guard is deliberately not a patch to site-packages: that would be lost the moment the conda
environment is rebuilt from `environment.yml`, so a fresh clone would hit the bug again. Drop it
once upstream validates the `sacct` output itself.

To check a suspect run:

```bash
grep -c "Unable to open file assigned" logs/<SET>.snakemake.log   # should be 0
```



**`max-masked-samples` is very important and defaults to a value that breaks this setup.** At the
default of `0`, any position covered by a mask is marked globally inaccessible — silently
deleting the sites the mask exists to rescue. `drivers/singer_config.yaml` sets it to 999,
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
bedtools intersect -a SingerSnakemake_SampleSets/<SET>/chr22.sample.mask.bed \
                   -b resources/grch38/mask/chr22.mask.bed -u | wc -l
```