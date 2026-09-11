"""
Write `chr{N}.sample.mask.bed` for one sample set: the per-sample missing-genotype mask.

singer-snakemake drops any site with a single missing genotype (`filter_nonmissing` is
`counts.sum(axis=1) == ploidy * samples.size`, zero tolerance). Genome-wide that costs 6.4%
(OCEANIA) to 41.3% (EAST_ASIA) of otherwise-usable sites, and most of it is sites missing in
only one or two samples.

This marks the lightly-missing ones so the workflow imputes them instead:

  <= MAX_MISSING_FRACTION of alleles missing -> write a mask line per missing sample, and
     `chunk_chromosomes.py` fills them by binomial draw from the observed allele frequency
  >  MAX_MISSING_FRACTION                    -> write nothing; the site stays missing, is
     dropped by `filter_nonmissing`, and is added to `filtered_positions` automatically
     (chunk_chromosomes.py:317-321), which scales the per-chunk mutation rate

The mask is not only an imputation trigger. `merge_chunks.py` feeds it to
`remove_partial_ancestry`, which clips the imputed ancestry back out of the output tree
sequence, and `run_polegon.py` passes it to `effective_edge_spans` so imputed spans are
excluded from dating. So imputed calls inform topology without masquerading as real data in
what CoaSTT ultimately measures.

Format, set by `parse_sample_bedmask` in singer-snakemake's utils.py: exactly four
whitespace-separated columns, `chrom start end sample_name`, with column 4 resolved against
the VCF's own sample order. Coordinates are 0-based half-open, so a single site at VCF POS is
`POS-1  POS` -- the 1 bp form the bundled `example.sample.mask.bed.complex` uses.

Two properties of this callset that the logic relies on, both verified and both asserted here:

  * There are no half-missing genotypes -- the only missing form is `./.`. So a missing sample
    always contributes exactly 2 missing alleles, the allele- and sample-fraction denominators
    coincide, and imputation can never overwrite an observed allele. `--strict-half-missing`
    (default on) aborts if one is ever found, since it would silently break that equivalence.
  * Multiallelic sites exist (ALT indices up to 3 are present). They are skipped entirely:
    `filter_biallelic` discards them regardless, so masking them would be wasted lines and
    would inflate `samplemask_coverage` against the `max-masked-samples` guard.
"""

import argparse
import json
import os
import subprocess
import sys

AUTOSOMES = [f"chr{i}" for i in range(1, 23)]

# Sites missing more than this fraction of their alleles are left missing rather than imputed.
MAX_MISSING_FRACTION = 0.10


def vcf_samples(path):
    """Sample names in VCF order. Same helper as write_set_metadata.py, same reason: the
    mask's fourth column is resolved positionally against this order by parse_sample_bedmask."""
    result = subprocess.run(
        ["bcftools", "query", "-l", path],
        check=True, capture_output=True, text=True,
    )
    return [line for line in result.stdout.splitlines() if line]


def write_mask(vcf, chrom, samples, out_path, max_fraction, strict_half_missing):
    """
    Stream one chromosome's genotypes and write its mask. Returns a summary dict.

    Streaming rather than loading: these VCFs run to millions of sites x hundreds of samples,
    and only one site is needed at a time.
    """
    n_samples = len(samples)
    max_missing_samples = int(max_fraction * n_samples)

    query = subprocess.Popen(
        ["bcftools", "query", "-f", "%POS[\t%GT]\n", vcf],
        stdout=subprocess.PIPE, text=True, bufsize=1 << 20,
    )

    sites_total = sites_clean = sites_imputed = sites_dropped = 0
    sites_multiallelic = 0
    bed_lines = 0
    worst_masked_samples = 0

    with open(out_path, "w") as out:
        for line in query.stdout:
            fields = line.rstrip("\n").split("\t")
            position, calls = fields[0], fields[1:]
            sites_total += 1

            missing = []
            multiallelic = False
            for index, call in enumerate(calls):
                # Genotypes are phased `a|b`, or `./.` when missing. Anything with an allele
                # index above 1 is multiallelic and will be dropped by filter_biallelic.
                if "." in call:
                    if strict_half_missing and call not in ("./.", ".|."):
                        raise SystemExit(
                            f"{chrom}: half-missing genotype {call!r} at POS {position} for "
                            f"sample {samples[index]}. The allele- and sample-fraction "
                            f"denominators are no longer equivalent and the mask would "
                            f"convert an observed allele into an imputed one. Re-examine the "
                            f"threshold before proceeding (see this script's docstring)."
                        )
                    missing.append(index)
                elif "2" in call or "3" in call:
                    multiallelic = True
                    break

            if multiallelic:
                sites_multiallelic += 1
                continue
            if not missing:
                sites_clean += 1
                continue
            if len(missing) > max_missing_samples:
                sites_dropped += 1
                continue

            start = int(position) - 1
            for index in missing:
                out.write(f"{chrom}\t{start}\t{position}\t{samples[index]}\n")
            bed_lines += len(missing)
            sites_imputed += 1
            worst_masked_samples = max(worst_masked_samples, len(missing))

    query.stdout.close()
    if query.wait() != 0:
        raise RuntimeError(f"bcftools query failed for {vcf}")

    return {
        "chromosome": chrom,
        "samples": n_samples,
        "max_missing_fraction": max_fraction,
        "max_missing_samples_allowed": max_missing_samples,
        "sites_total": sites_total,
        "sites_no_missing": sites_clean,
        "sites_multiallelic_skipped": sites_multiallelic,
        "sites_imputed": sites_imputed,
        "sites_left_missing": sites_dropped,
        "bed_lines": bed_lines,
        "max_masked_samples_at_a_site": worst_masked_samples,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True)
    parser.add_argument(
        "--max-missing-fraction", type=float, default=MAX_MISSING_FRACTION,
        help="sites missing more than this fraction of alleles are left missing",
    )
    parser.add_argument(
        "--no-strict-half-missing", dest="strict_half_missing", action="store_false",
        help="permit half-missing genotypes instead of aborting (see docstring)",
    )
    parser.add_argument("--chromosomes", nargs="*", default=AUTOSOMES)
    args = parser.parse_args()

    reference_order = None
    summaries = {}

    for chrom in args.chromosomes:
        vcf = os.path.join(args.input_dir, f"{chrom}.vcf.gz")
        assert os.path.exists(vcf), f"Missing {vcf}"

        samples = vcf_samples(vcf)
        if reference_order is None:
            reference_order = samples
        else:
            assert samples == reference_order, (
                f"{chrom} sample order differs from the first chromosome; mask sample names "
                f"are resolved against each VCF's own order, so this must be consistent"
            )

        out_path = os.path.join(args.input_dir, f"{chrom}.sample.mask.bed")
        summary = write_mask(
            vcf, chrom, samples, out_path,
            args.max_missing_fraction, args.strict_half_missing,
        )
        summaries[chrom] = summary

        with open(os.path.join(args.input_dir, f"{chrom}.sample.mask.json"), "w") as handle:
            json.dump(summary, handle, indent=2)

        imputed_pct = 100 * summary["sites_imputed"] / max(summary["sites_total"], 1)
        dropped_pct = 100 * summary["sites_left_missing"] / max(summary["sites_total"], 1)
        print(
            f"  {chrom}: {summary['sites_imputed']:,} imputed ({imputed_pct:.2f}%), "
            f"{summary['sites_left_missing']:,} left missing ({dropped_pct:.2f}%), "
            f"{summary['bed_lines']:,} mask lines, "
            f"max {summary['max_masked_samples_at_a_site']} masked samples at a site"
        )

    worst = max(s["max_masked_samples_at_a_site"] for s in summaries.values())
    print(
        f"  -> max masked samples at any site across all chromosomes: {worst}. "
        f"`max-masked-samples` in singer_config.yaml MUST exceed this, or every masked "
        f"position is marked globally inaccessible (its default of 0 does exactly that)."
    )


if __name__ == "__main__":
    main()
