# Prepping chimps for SINGER + POLEGON

- **Goal:** Starting from a raw vcf of chimps, prepare the input files needed to run the singer-snakemake pipeline.
- **Dataset**:
	- 54 chimp WGS (7 bonobos are filtered out in this run)
	- *schweinfurthii (Eastern):* 19, *troglodytes (Central):* 15, *verus (Western):* 14, *ellioti (Nigeria-Cameron):* 5, *verus × troglodytes hybrid:* 1
	- panTro6, 23 autosomes
	- Source: raw GATK callset: unphased, unfiltered, variant-only, no published accessibility mask
- **Files prepared for each chromosome:**
	- **<chromosome_name>.vcf.gz:** Phased + filtered vcf for that chromosome
		- This is the main input to singer
	- **<chromosome_name>.filter.txt:** Coordinates of unreliable variant calls that were filtered out
		- These SNPs are already removed from the VCF. Declaring them allows POLEGON to calculate an adjustment to mutation rates
	- **<chromosome_name>.mask.bed:** Start and end coordinates of regions that were not reliably sequenced
		- These intervals do not inform the topology or the dates. In the inferred tree sequences, samples within these intervals have no ancestors, as no local trees are inferred for these intervals. Chunks with more than 50% masking are not passed to singer.
	- **<chromosome_name>.ancestral.fasta:** The ancestral state of each variant in the vcf
		- Singer sets the "polarized" parameter to "0.99" for sites where the ancestral state is with high confidence, and "0.5" otherwise. "0.5" allows singer more flexibility in flipping the polarization at that site
	- **<chromosome_name>.hapmap:** [Recombination map](https://github.com/lstevison/great-ape-recombination/tree/master/Final%20Chimp%20Map)

## Running the pipeline

Assumes you cloned the repo and created `CoaSTT_env` (`conda env create -f environment.yml`). All paths below are relative to `data/chimp_data/`, and every command is run from there.

**1. Put the raw VCF in place.** This is the only input you supply:

```
data/chimp_data/resources/vcf/full.vcf.gz       # bgzipped, multi-sample, all chromosomes
data/chimp_data/resources/vcf/full.vcf.gz.tbi   # index: bcftools index -t full.vcf.gz
```

The name and location are hard-coded (`scripts/prep_vcf.sbatch`). The callset may be unphased, unfiltered and variant-only. It must carry `FORMAT/AD` and `FORMAT/DP` — the allelic-imbalance and depth filters read them, and phasing strips them, so they cannot be recovered later.

Also check `resources/samples/chimps.txt` (the samples to keep — one ID per line; this is where bonobos are dropped) and `resources/samples/subspecies.tsv` (used by `stratify-by` in the SINGER config).

**2. Run the drivers in order.** Each blocks until its SLURM job finishes, so they can be chained:

```bash
bash drivers/1_prep_vcf.sh        # split by chromosome, drop bonobos, apply site filters
bash drivers/2_genmap_index.sh    # genome-wide mappability index  (independent of step 1)
bash drivers/3_build_resources.sh # recombination maps + ancestral FASTAs
bash drivers/4_build_masks.sh     # accessibility masks
bash drivers/5_phase.sh           # SHAPEIT5
bash drivers/6_assemble.sh        # build SingerSnakemake_Input/
bash drivers/7_verify.sh          # 15 checks per chromosome -- the gate before inference
bash drivers/8_run_singer.sh      # one SLURM job per chromosome (days)
bash drivers/9_extract_chunks.sh  # cut ARGs into results/chimp/ts_chunks/ for CoaSTT
```

Steps 1 and 2 are independent and can be submitted together. **Do not skip step 7** — it is the last point at which a malformed input is caught before committing days of compute.

Step 9 loads a whole chromosome at a time and needs ~3 GB free. Login nodes cap a user session at 8 GB total (a running IDE can hold most of it), so if it is OOM-killed, run it per chromosome:

```bash
for c in chr1 chr2A chr2B chr3 chr4 chr5 chr6 chr7 chr8 chr9 chr10 chr11 chr12 \
         chr13 chr14 chr15 chr16 chr17 chr18 chr19 chr20 chr21 chr22; do
    python ../human_data/scripts/extract_ts_chunks.py \
        --singer-output results/chimp/singer-snakemake_outputs \
        --out-dir results/chimp/ts_chunks \
        --config drivers/singer_config.yaml --chromosomes $c
done
```

**3. Reclaim disk when done.** The per-draw SINGER files are the bulk of the output (428 GB here). After step 9 has succeeded, drop the burn-in draws — dry-run by default:

```bash
python scripts/prune_chunk_draws.py --singer-output results/chimp/singer-snakemake_outputs \
    --ts-chunks results/chimp/ts_chunks --keep-from 12 --expect-chunks 1334   # add --apply
```

### Changing SINGER parameters

Everything tunable lives in `drivers/singer_config.yaml`; no script needs editing. The keys you are most likely to change:

| key | effect |
| --- | --- |
| `max-chunk-size` / `min-chunk-size` | chunk length. Smaller chunks finish faster and parallelise better, but there are more of them |
| `max-missing` / `min-sites` | when a chunk is discarded (>50% masked, or <1,000 polymorphic sites) |
| `singer-mcmc-samples` / `-thin` | chain length = samples x thin. Currently 20 x 250 = 5,000 iterations |
| `singer-mcmc-burnin` | fraction of draws treated as burn-in for the pipeline's own stats and plots |
| `mutation-rate` | per-base per-generation; `recombination-rate` is ignored because a hapmap is supplied |
| `stratify-by` | column in `resources/samples/subspecies.tsv` used to group samples in the diagnostics |

Note `singer-mcmc-burnin` affects only the pipeline's statistics and plots — `merge_chunks` writes a tree sequence for **every** draw. The burn-in that matters for downstream analysis is enforced in step 9, which refuses any draw below ~4,000 MCMC iterations.

### SLURM settings

Step 8 submits one plain sbatch job per chromosome; there is no snakemake executor involved. To adapt it to another cluster, edit the `#SBATCH` header of `scripts/run_singer_chrom.sbatch` — account, partition, QOS, `--cpus-per-task` (16 here, which is also how many chunks run in parallel), `--time` and `--mem`. The partition can also be overridden per invocation:

```bash
bash drivers/8_run_singer.sh --partition jnovembre   # default: caslake
bash drivers/8_run_singer.sh --chrom chr21           # one chromosome
bash drivers/8_run_singer.sh -n                      # dry run
```

Two sizing traps worth inheriting: walltime must cover the **tail** of the chunk-time distribution rather than the median (the slowest chunk observed here took ~17 h against a ~0.9 h median), and `merge_chunks` memory scales with **chromosome length**, not chunk size — it needs ~60-120 GB and is the step that OOMs if sized from SINGER's per-chunk footprint.

## How the raw VCF became these files

### 0. Prepare data-independent files
#### a) Build Accessibility mask
- This step does not see the VCFs at all. It builds an accessibility mask by a mappability index with the annotated gaps of the PanTro6 assembly
1. **Mappability index:** Run `genmap -K 36 -E 0` on PanTro6 assembly. genmap reports `1 / (times this k-mer occurs in the assembly)`, so the threshold `< 1.0` means _this 36-mer is not unique.
2. **Assembly gaps**: from the UCSC panTro6 gap track. Doesn't add much on top of the mappability mask, but still kept.
- Genome-wide: **88.22% accessible.** 312,847,835 of 2,654,688,569 bp masked across 10,772,802 intervals. Per chromosome this ranges from 79.33% (chr19) to 90.94% (chr13).
- per-chromosome masked regions written to  chrN.mask.bed. Singer does not infer local trees for masked intervals. The output tree sequences will have trees whose samples have no ancestors within masked regions.
#### b) Recombination map
- The source is the [Stevison _et al._](https://academic.oup.com/mbe/article/33/4/928/2579342) _ellioti_ map (LD-based map)
- Availible map had to be brought to PanTro6 coordinates and the units of `msprime.RateMap.read_hapmap`
	1. **Lift:** hg18 → hg38 → panTro6** with UCSC chain files 
	2. **Convert ρ/kb to cM/Mb** with a single global scalar across all chromosomes to target genome-wide average cM/Mb of 1.1 cM/Mb (human average. Human genome-wide averages actually run 1.0-1.3 cM/Mb, so the choice is kind of arbitrary and could be changed later).
		- After the lift-over, genome-wide ρ/kb was measured at 0.80044, matching 0.8 reported by Stevison et al. 
	3. **Gap-fill and extend to the chromosome end.** Intervals the map could not estimate carry the string `NA` rather than a number, and the workflow requires the map to extend past the last variant. unestimated and trailing regions take the chromosome mean.
#### c) Ancestral fasta
- Ancestral states come from **hg38 reference** bases projected into panTro6 through the `panTro6ToHg38` chain. Positions with no human alignment (2–3%) are left as `N` and fall through to SINGER's own per-site polarization (had to take care of lift-over landing on a minus strand, where the reverse-compliment has to be taken).
- **Why the human outgroup and not the Ensembl chimpanzee ancestor.** The EPO chimpanzee ancestral sequence was the original choice and was rejected on measurement. It is reconstructed against a _western_ chimpanzee reference, which leaves a per-subspecies bias: _verus_ appears fixed for the derived allele at 3.79% of sites against _troglodytes_' 0.46%. Keeping only EPO's high-confidence calls makes it **worse**, taking the asymmetry from 8.17× to 13.24×. The human outgroup trades a higher random mispolarization rate (6 My of divergence, so more multiple hits) for a near-zero systematic one. This observation was noted by DeManuel et al. as well, where they also used the human-derived ancestral states for some of their analyses.
- Singer has a higher chance of repolarizing sites where the ancestral state is not called with high confidence. Can also repolarize those called with high confidence, but less likely to.
### 1. Remove Bonobos and drop monomorphic sites + rename chromosomes
### 2. Apply site filters 
- **GATK hard filters**: `QD<2 | FS>60 | MQ<40 | SOR>3 | ReadPosRankSum<-8 | MQRankSum<-12.5`. Best Practices thresholds, human-tuned, not re-derived for chimpanzee. Two are effectively inert here: `ReadPosRankSum` removes nothing and `MQRankSum` almost nothing; `QD` and `SOR` do most of the work.
- **Allelic imbalance:** pooled alternate-allele fraction across heterozygous calls outside `[0.25, 0.75]`. A true heterozygote sits near 0.5; a collapsed paralogue does not. The panel-wide distribution is healthy (median 0.4964 against a 0.5 expectation)
- **Depth:** summed `FORMAT/DP` outside 0.25×–2× the chromosome median
- **Non-biallelic SNPs:** anything that is not exactly one REF and one ALT SNP allele: multiallelics, indels, mixed records

| filter             | sites removed | % of unfiltered sites |
| ------------------ | ------------- | --------------------- |
| GATK hard filters  | 2,699,496     | 5.47%                 |
| Allelic imbalance  | 3,304,436     | 6.69%                 |
| Depth              | 813,363       | 1.65%                 |
| Non-biallelic SNPs | 2,833,183     | 5.74%                 |
| **Union**          | **6,467,948** | **13.10%**            |
**42,900,324 sites survive.** Every filtered position is written to chrN.filter.txt to adjust mutation rates during the POLEGON phase.
### 3. Phase (SHAPEIT5) 
- used `phase_common --hmm-ne 10000` of SHAPEIT5, with no reference panel and no genetic map
- `--hmm-ne` was swept at 10,000 / 25,000 / 30,000 and made no difference
- Beagle was tested performed slightly worse than SHAPEIT: 4.66% switch error against SHAPEIT5's 4.03%
- Accuracy is measured against GATK `PGT`/`PID` read-backed phase blocks, the only independent phase truth in a panel with no trios: **one switch error per ~16–17 kb, against a marginal tree change every 0.146 kb**, so phase error is roughly 0.9% as frequent as tree transitions. Read-backed blocks are short, so this measures _local_ accuracy. It does not bound long-range switch error.
### 4. Run Singer
- Using the [snakemake](https://github.com/nspope/singer-snakemake) pipeline built by Nate Pope and others, and the files prepped above. 
- 5,000 MCMC total iterations. 20 posterior draws thinned by 250 iterations. The first 12 draws (3250 iterations) are burn-in, and the last 8 are used to compute the analysis stats in the pipeline 
- Chromosomes are chopped into 2 Mb chunks. Any chunk with more than 50% masked sites or less than 1,000 polymorphic sites is discarded (these parameters can be changed from a config file)
- A single SLURM job for each chromosome with 16 cpus (~4 Gb per cpu). Singer run in parallel on each cpu, so 16 chunks are run at a time.
- Run time:

| chrom     | chunks    | CPU-h      | Actual hours | hours/chunk |
| --------- | --------- | ---------- | ------------ | ----------- |
| chr1      | 112       | 966        | 60.4         | 0.54        |
| chr2A     | 55        | 556        | 34.7         | 0.63        |
| chr2B     | 65        | 908        | 56.7         | 0.87        |
| chr3      | 99        | 916        | 57.3         | 0.58        |
| chr4      | 94        | 1,352      | 84.5         | 0.90        |
| chr5      | 80        | 911        | 56.9         | 0.71        |
| chr6      | 84        | 838        | 52.4         | 0.62        |
| chr7      | 78        | 844        | 52.8         | 0.68        |
| chr8      | 72        | 867        | 54.2         | 0.75        |
| chr9      | 56        | 964        | 60.2         | 1.07        |
| chr10     | 65        | 841        | 52.6         | 0.81        |
| chr11     | 66        | 869        | 54.3         | 0.82        |
| chr12     | 66        | 641        | 40.0         | 0.61        |
| chr13     | 48        | 526        | 32.9         | 0.69        |
| chr14     | 44        | 538        | 33.6         | 0.76        |
| chr15     | 41        | 565        | 35.3         | 0.86        |
| chr16     | 37        | 603        | 37.7         | 1.02        |
| chr17     | 39        | 550        | 34.4         | 0.88        |
| chr18     | 38        | 484        | 30.3         | 0.80        |
| chr19     | 29        | 544        | 34.0         | 1.17        |
| chr20     | 32        | 523        | 32.7         | 1.02        |
| chr21     | 17        | 294        | 18.4         | 1.08        |
| chr22     | 17        | 290        | 18.1         | 1.06        |
| **TOTAL** | **1,334** | **16,389** | **1,024.4**  | **0.77**    |
