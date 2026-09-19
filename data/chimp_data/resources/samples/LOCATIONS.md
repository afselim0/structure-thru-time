# Sample locations

Coordinates for the 54 panel samples, pulled from the de Manuel *et al.* 2016 supplement
(`aag2602-de-manuel-sm-data-s1.xlsx`, sheet `Ark1`, 65 rows with `Latitude`/`Longitude`).

**The IDs do not match.** The VCF uses accessions (`SAMEA4374763`, `SAMN01920517`, `PD_0257`);
the supplement uses individual names (`A991_Berta`, `Akwaya_Jean`). The bridge is two steps:

```
accession  --[ EBI BioSamples ]-->  individual name  --[ token match ]-->  supplement row
```

1. `https://www.ebi.ac.uk/biosamples/samples/<accession>.json` returns the animal's name
   (`Berta`, `Akwaya-Jean`) and sometimes a country. Resolves 50/54; the 4 `PD_*` IDs are
   internal and have no public record.
2. Match on shared name tokens, splitting on any non-alphanumeric character. Not exact
   string match: the supplement prefixes studbook codes (`A991_Berta`, `12348_Luky`) and
   separators differ (`Akwaya-Jean` vs `Akwaya_Jean`). Numeric/studbook tokens are excluded
   from matching so a code alone can never make a match.

**Five names are ambiguous** — three Cindys and two Julies, in different subspecies. BioSamples
supplies the disambiguator (`Cindy_verus`, `Cindy_schwein`, `Julie_A959`), which is resolved
against subspecies. Those rows are flagged `subspecies_disambiguated` in `match_basis`;
check them first if a location looks wrong.

## Coverage

| | individuals |
| --- | --- |
| with lat/lon | **38 / 54** |
| matched, but supplement lat/lon blank | 11 (6 of them Central/*troglodytes*) |
| country only, no coordinates | 1 (Mgbadolite, DRC) |
| no BioSample record (`PD_*`) | 4 |

Subspecies agrees between the supplement and our own labels for 48/50 matches. The two
exceptions are hybrid-annotation differences, not mismatched individuals: `Mgbadolite`
(supplement Pts/Ptt hybrid, ours schweinfurthii) and `Donald` (supplement verus, ours
verus x troglodytes).

## Caveats

- Coordinates are **inferred origin**, not sampling points. Most of these animals were
  sampled in sanctuaries and zoos; two records carry `Captive Born` in the country field.
- They are coarse — mostly whole or half degrees, with the supplement's own uncertainty
  column (`uncertainty_deg`, typically 4 degrees) kept in `sample_locations.csv`.
- Fine for colouring a map by broad origin; not for spatial inference.

## Files

- `sample_locations.csv` — one row per individual, with `demanuel_shortname` and
  `match_basis` so every join is auditable.
- `pop_labels.csv` — the CoaSTT input; carries `latitude`, `longitude`, `origin_country`,
  `origin_country_2` per haplotype (76/108 populated).
