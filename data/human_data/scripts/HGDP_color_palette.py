"""
Build the per-population colour palette for the 54 HGDP populations.

Writes two palettes into `resources/population_labels_colors/`, each as a CSV plus an
always-regenerated PNG with one row per region and one labelled box per population:

    global_palette.{csv,png}     seven region hues, populations ramped within each
    regional_palette.{csv,png}   seven independent palettes, discrete within each region

They answer different questions, so a population's colour differs between them. The global
palette is for ALL, where all seven regions share a plot and the colour must say which
region a population belongs to. The regional palettes are for the single-region runs, where
no other region is present and the colour only has to tell populations apart -- so each
region spends the whole gamut on its own populations. See REGIONAL POOL below.

THE GLOBAL PALETTE is two-level. Each region gets an anchor hue from REGION_ANCHORS below --
those seven colours are the identity axis and are the thing that has to survive colour
blindness, so they are checked against the Machado-Oliveira-Fernandes CVD simulation for
every pair. Within a region, populations vary in lightness and chroma around that anchor
along a *geographic* axis, so that neighbouring populations get neighbouring colours and
the ramp reads as a gradient across the region rather than as an arbitrary ordering.

The geographic axis is the first principal component of the region's (longitude, latitude)
after scaling longitude by cos(mean latitude), i.e. the direction of greatest spread in
approximate great-circle kilometres. Populations project onto it and are ranked by that
projection; the rank drives the ramp. Regions of one or two populations skip the PCA.
Several HGDP populations share a coordinate exactly (Druze/Palestinian, Brahui/Balochi),
so ties in the projection are broken by population name, keeping the output deterministic.

Within-region contrast is deliberately lower than between-region contrast: two populations
in the same region SHOULD look alike, because that similarity is the geographic signal.
So the CVD check is a hard gate on the seven anchors and informational within a region.

REGION_ANCHORS is meant to be edited. Change a hex, re-run, look at the PNG.
"""

import argparse
import colorsys
import csv
import itertools
import math
import os

# --- the palette's one tunable: seven region anchors -------------------------------
# Edit these. Everything else is derived. Chosen to be mutually distinguishable under
# protanopia and deuteranopia, which every run re-verifies.
REGION_ANCHORS = {
    "AFRICA":             "#843A15",  # deep brick
    "MIDDLE_EAST":        "#CDAF18",  # gold
    "EUROPE":             "#518E0B",  # olive green
    "CENTRAL_SOUTH_ASIA": "#3ACBB0",  # light teal
    "EAST_ASIA":          "#4A7FE8",  # blue
    "OCEANIA":            "#7E1D8B",  # deep purple
    "AMERICA":            "#CE4695",  # pink
}
# These seven alternate dark/light on purpose. Protanopia and deuteranopia collapse the
# red-green axis, so a pure warm-to-cool hue sweep -- red, orange, yellow, green -- is
# unreadable however pretty it looks in full colour. The lightness staircase is what
# carries the separation for those viewers, and the hue sweep is what makes the regions
# read continentally for everyone else. Worst anchor pair: dE 13.5 under protan/deutan,
# against a floor of 6 and a target of 8. Keep that staircase if you retune the hues.

# Region order, top to bottom in the PNG. Roughly the out-of-Africa path.
REGION_ORDER = [
    "AFRICA", "MIDDLE_EAST", "EUROPE", "CENTRAL_SOUTH_ASIA",
    "EAST_ASIA", "OCEANIA", "AMERICA",
]

# Regions split into two hue groups internally, by a latitude cut.
#
# A single hue can only hold so many populations. The ramp's whole lightness range is
# about dE 30 end to end, so n populations get roughly 30/(n-1) between neighbours: fine
# at n=7, but EAST_ASIA's 18 come out at dE 1.5, far below the ~8 needed to tell two
# colours apart. Separating 18 at dE 8 would need dE 136 end to end and the entire sRGB
# gamut offers about 100, so no amount of tuning one hue can fix it. (A hue arc across
# the region was measured too: it reached dE 2.2 at best while pushing cross-region
# collisions from 78 to 113. It is not the answer.)
#
# So EAST_ASIA gets a second anchor and its populations ramp within whichever group they
# fall in, halving n per hue. The cut at 33degN splits it 9/9 into Siberia/Mongolia/Japan/
# northern China against southern China and Southeast Asia -- geographically coherent, and
# NorthernHan/Han land on the sides their names imply.
#
# This is PALETTE-INTERNAL ONLY. The `region` column stays EAST_ASIA for every population,
# the PNG keeps one East Asia row, and nothing downstream sees a new region. It is two
# hues inside one region, not eight regions.
REGION_SUBSPLITS = {
    "EAST_ASIA": {
        "latitude_cut": 33.0,
        "north_anchor": "#4A7FE8",  # the region anchor; keeps the north as it was
        "south_anchor": "#4A4AE8",  # +20deg of hue: dE 8.8 from the north, 14.4 from any
                                    # other region's anchor
    },
}

# How far the within-region ramp travels from the anchor, in HLS units. The ramp runs
# from (anchor darkened, saturated) to (anchor lightened, desaturated). Widening these
# makes a region's populations easier to tell apart but weakens the region's own identity.
LIGHTNESS_SPAN = 0.34
SATURATION_SPAN = 0.26

# Hard limits on the ramp. Without a floor the dark end of a region whose anchor is
# already dark goes to near-black, and without a ceiling the light end of a light anchor
# washes out to near-white; both lose the region's hue, which is the one thing the ramp
# must not do. The ramp is also re-centred inside these limits (see region_ramp), so a
# region keeps its full span instead of being clipped flat against one end.
LIGHTNESS_MIN = 0.26
LIGHTNESS_MAX = 0.78

# Population label size in palette.png. Labels that would overrun their box are measured
# and scaled down individually, so this is the size everything else renders at.
LABEL_FONTSIZE = 12


# CVD thresholds, OKLab Delta E x100, per the data-viz palette checks.
CVD_TARGET = 8.0
CVD_FLOOR = 6.0


# --- colour conversion -------------------------------------------------------------
def hex_to_rgb(value):
    value = value.strip().lstrip("#")
    return tuple(int(value[i:i + 2], 16) / 255 for i in (0, 2, 4))


def rgb_to_hex(rgb):
    return "#" + "".join(f"{round(max(0.0, min(1.0, c)) * 255):02X}" for c in rgb)


def srgb_to_linear(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def oklab(rgb):
    r, g, b = (srgb_to_linear(c) for c in rgb)
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (
        0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
        1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
        0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s,
    )


# Machado, Oliveira & Fernandes (2009), severity 1.0, applied in linear RGB.
MACHADO = {
    "protan": ((0.152286, 1.052583, -0.204868),
               (0.114503, 0.786281, 0.099216),
               (-0.003882, -0.048116, 1.051998)),
    "deutan": ((0.367322, 0.860646, -0.227968),
               (0.280085, 0.672501, 0.047413),
               (-0.011820, 0.042940, 0.968881)),
    "tritan": ((1.255528, -0.076749, -0.178779),
               (-0.078411, 0.930809, 0.147602),
               (0.004733, 0.691367, 0.303900)),
}


def simulate_cvd(rgb, kind):
    r, g, b = (srgb_to_linear(c) for c in rgb)
    rows = MACHADO[kind]
    out = []
    for row in rows:
        v = row[0] * r + row[1] * g + row[2] * b
        v = max(0.0, min(1.0, v))
        # Back to sRGB so oklab() -- which linearises again -- sees the right value.
        out.append(12.92 * v if v <= 0.0031308 else 1.055 * v ** (1 / 2.4) - 0.055)
    return tuple(out)


def delta_e(hex_a, hex_b, kind=None):
    """Euclidean distance in OKLab, x100. kind=None is unsimulated vision."""
    a, b = hex_to_rgb(hex_a), hex_to_rgb(hex_b)
    if kind:
        a, b = simulate_cvd(a, kind), simulate_cvd(b, kind)
    pa, pb = oklab(a), oklab(b)
    return 100 * math.dist(pa, pb)


# --- geography ---------------------------------------------------------------------
def geographic_order(populations):
    """Rank populations along the principal axis of their coordinates.

    `populations` is a list of (name, longitude, latitude). Returns the names ordered
    along the direction of greatest geographic spread, ties broken by name.
    """
    if len(populations) <= 2:
        return [name for name, _, _ in sorted(populations)]

    mean_lat = sum(lat for _, _, lat in populations) / len(populations)
    scale = math.cos(math.radians(mean_lat))
    points = [(lon * scale, lat) for _, lon, lat in populations]
    cx = sum(x for x, _ in points) / len(points)
    cy = sum(y for _, y in points) / len(points)

    # First principal axis via the 2x2 covariance eigenvector -- closed form, no numpy.
    sxx = sum((x - cx) ** 2 for x, _ in points) / len(points)
    syy = sum((y - cy) ** 2 for _, y in points) / len(points)
    sxy = sum((x - cx) * (y - cy) for x, y in points) / len(points)
    theta = 0.5 * math.atan2(2 * sxy, sxx - syy)
    ux, uy = math.cos(theta), math.sin(theta)

    # Orient the axis north/east-ward so the ramp direction does not flip between runs
    # on a sign choice that atan2 makes arbitrarily.
    if (uy < 0) or (uy == 0 and ux < 0):
        ux, uy = -ux, -uy

    projected = [
        (( x - cx) * ux + (y - cy) * uy, name)
        for (x, y), (name, _, _) in zip(points, populations)
    ]
    return [name for _, name in sorted(projected)]


# --- palette construction ----------------------------------------------------------
def region_ramp(anchor_hex, n):
    """n colours around `anchor_hex`, dark/saturated -> light/desaturated."""
    h, l, s = colorsys.rgb_to_hls(*hex_to_rgb(anchor_hex))
    if n == 1:
        return [rgb_to_hex(colorsys.hls_to_rgb(h, l, s))]

    # Centre the ramp so it fits between LIGHTNESS_MIN/MAX without clipping. An anchor
    # near either limit would otherwise have half its ramp flattened against that limit,
    # collapsing several populations onto one colour.
    half = LIGHTNESS_SPAN / 2
    centre = min(LIGHTNESS_MAX - half, max(LIGHTNESS_MIN + half, l))

    colors = []
    for i in range(n):
        t = i / (n - 1) - 0.5                      # -0.5 .. +0.5
        li = min(LIGHTNESS_MAX, max(LIGHTNESS_MIN, centre + t * LIGHTNESS_SPAN))
        si = min(1.00, max(0.22, s - t * SATURATION_SPAN))
        colors.append(rgb_to_hex(colorsys.hls_to_rgb(h, li, si)))
    return colors


def build_palette(metadata_path):
    """Return a list of dicts, one per population, in REGION_ORDER then ramp order."""
    coords = {}
    with open(metadata_path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            coords[(row["region"], row["population"])] = (
                float(row["longitude"]), float(row["latitude"])
            )

    by_region = {}
    for (region, population), (lon, lat) in coords.items():
        by_region.setdefault(region, []).append((population, lon, lat))

    missing = set(by_region) - set(REGION_ANCHORS)
    assert not missing, f"No anchor colour for region(s): {sorted(missing)}"
    unseen = set(REGION_ANCHORS) - set(by_region)
    assert not unseen, f"Anchor given for region(s) absent from metadata: {sorted(unseen)}"
    assert set(REGION_ORDER) == set(REGION_ANCHORS), "REGION_ORDER must list every anchor"

    records = []
    for region in REGION_ORDER:
        populations = by_region[region]
        lookup = {name: (lon, lat) for name, lon, lat in populations}

        # A split region ramps each half against its own anchor; everything else is one
        # group against the region anchor. Either way `region` below stays the real region.
        split = REGION_SUBSPLITS.get(region)
        if split:
            groups = [
                ([p for p in populations if p[2] >= split["latitude_cut"]],
                 split["north_anchor"]),
                ([p for p in populations if p[2] < split["latitude_cut"]],
                 split["south_anchor"]),
            ]
            for members, _ in groups:
                assert members, (
                    f"{region}: latitude_cut {split['latitude_cut']} leaves a group empty"
                )
        else:
            groups = [(populations, REGION_ANCHORS[region])]

        rank = 0
        for members, anchor in groups:
            ordered = geographic_order(members)
            for population, color in zip(ordered, region_ramp(anchor, len(ordered))):
                lon, lat = lookup[population]
                records.append({
                    "population": population,
                    "region": region,
                    "color": color,
                    "region_anchor": anchor,
                    "geographic_rank": rank,
                    "longitude": lon,
                    "latitude": lat,
                })
                rank += 1
    return records


# --- checks ------------------------------------------------------------------------
# --- regional palettes -------------------------------------------------------------
# Seven independent palettes, one per region, for the runs that plot a single region on
# its own. There are no other regions on those plots, so nothing has to be reserved for
# region identity and each palette may use the whole gamut: colours are chosen to be as
# far apart as possible instead of as similar as possible.
#
# That inverts the global palette's goal, which is why these are two separate files and
# not two views of one. A population deliberately has a DIFFERENT colour in the two --
# Han is one of eighteen blues in the global palette and a fully distinct colour in the
# East Asia one -- because the two palettes answer different questions.
#
# Assignment still follows geographic_order, so a legend reads west-to-east or
# south-to-north as before; it is the colours that are now discrete rather than a ramp.
REGIONAL_POOL_HUE_STEP = 6          # degrees
REGIONAL_POOL_LIGHTNESS = range(26, 82, 5)   # HLS lightness, percent
REGIONAL_POOL_SATURATION = range(40, 101, 12)
REGIONAL_LIGHTNESS_BAND = (0.42, 0.80)       # OKLCH L, the readable band
REGIONAL_CHROMA_FLOOR = 0.075                # below this a hue reads as grey
REGIONAL_NORMAL_FLOOR = 12.0        # min OKLab dE under unsimulated vision
REGIONAL_RESTARTS = 400             # greedy restarts; deterministic via REGIONAL_SEED
REGIONAL_SEED = 1


def _regional_pool():
    """Candidate colours inside the readable lightness band, with their CVD coordinates."""
    import numpy as np

    def to_linear(a):
        return np.where(a <= 0.04045, a / 12.92, ((a + 0.055) / 1.055) ** 2.4)

    def to_srgb(a):
        a = np.clip(a, 0, 1)
        return np.where(a <= 0.0031308, 12.92 * a, 1.055 * a ** (1 / 2.4) - 0.055)

    lms = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                    [0.2119034982, 0.6806995451, 0.1073969566],
                    [0.0883024619, 0.2817188376, 0.6299787005]])
    lab = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                    [1.9779984951, -2.4285922050, 0.4505937099],
                    [0.0259040371, 0.7827717662, -0.8086757660]])

    def to_oklab(rgb):
        return np.cbrt(to_linear(rgb) @ lms.T) @ lab.T

    def simulate(rgb, kind):
        return to_srgb(np.clip(to_linear(rgb) @ np.array(MACHADO[kind]).T, 0, 1))

    rgb = np.array([
        colorsys.hls_to_rgb(hue / 360.0, l / 100.0, s / 100.0)
        for hue in range(0, 360, REGIONAL_POOL_HUE_STEP)
        for l in REGIONAL_POOL_LIGHTNESS
        for s in REGIONAL_POOL_SATURATION
    ])
    coords = to_oklab(rgb)
    lightness = coords[:, 0]
    chroma = np.hypot(coords[:, 1], coords[:, 2])
    low, high = REGIONAL_LIGHTNESS_BAND
    rgb = rgb[(lightness >= low) & (lightness <= high) & (chroma >= REGIONAL_CHROMA_FLOOR)]

    def pairwise(points):
        squares = (points ** 2).sum(1)
        return 100 * np.sqrt(np.maximum(
            squares[:, None] + squares[None, :] - 2 * points @ points.T, 0
        ))

    cvd = np.minimum(pairwise(to_oklab(simulate(rgb, "protan"))),
                     pairwise(to_oklab(simulate(rgb, "deutan"))))
    normal = pairwise(to_oklab(rgb))
    np.fill_diagonal(cvd, np.inf)
    np.fill_diagonal(normal, np.inf)
    return rgb, cvd, normal


def max_separated(n, pool):
    """n colours from `pool` maximising the smallest pairwise CVD separation."""
    import numpy as np

    rgb, cvd, normal = pool
    size = len(rgb)
    assert n <= size, f"cannot draw {n} distinct colours from a pool of {size}"
    rng = np.random.default_rng(REGIONAL_SEED)

    def score_against(chosen):
        # Distance to the nearest already-chosen colour, with candidates that are too
        # close under NORMAL vision ruled out: full-colour readers must see them apart
        # too, not just CVD viewers.
        best = cvd[:, chosen].min(axis=1)
        best = np.where(normal[:, chosen].min(axis=1) >= REGIONAL_NORMAL_FLOOR, best, -1.0)
        best[chosen] = -1.0             # never pick the same colour twice
        return best

    best_set, best_score = None, -1.0
    for _ in range(REGIONAL_RESTARTS):
        chosen = [int(rng.integers(size))]
        while len(chosen) < n:
            chosen.append(int(np.argmax(score_against(chosen))))

        improved = True
        while improved:                 # swap each slot for the best available colour
            improved = False
            for i in range(n):
                others = [chosen[j] for j in range(n) if j != i]
                scores = score_against(others)
                candidate = int(np.argmax(scores))
                if scores[candidate] > cvd[chosen[i], others].min() + 1e-9:
                    chosen[i] = candidate
                    improved = True

        score = min(cvd[a, b] for a, b in itertools.combinations(chosen, 2))
        if score > best_score:
            best_score, best_set = score, list(chosen)

    return [rgb_to_hex(tuple(rgb[i])) for i in best_set], best_score


def build_regional_palettes(metadata_path):
    """One max-separation palette per region, assigned in geographic order."""
    coords = {}
    with open(metadata_path) as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            coords[(row["region"], row["population"])] = (
                float(row["longitude"]), float(row["latitude"])
            )

    by_region = {}
    for (region, population), (lon, lat) in coords.items():
        by_region.setdefault(region, []).append((population, lon, lat))

    pool = _regional_pool()
    records, separations = [], {}
    for region in REGION_ORDER:
        populations = by_region[region]
        ordered = geographic_order(populations)
        colors, score = max_separated(len(ordered), pool)
        separations[region] = score
        lookup = {name: (lon, lat) for name, lon, lat in populations}
        for rank, (population, color) in enumerate(zip(ordered, colors)):
            lon, lat = lookup[population]
            records.append({
                "population": population,
                "region": region,
                "color": color,
                "region_anchor": "",
                "geographic_rank": rank,
                "longitude": lon,
                "latitude": lat,
            })
    return records, separations


def check_regional(records, separations):
    """Report each region's internal separation. True if every region clears the target."""
    ok = True
    print("Regional palettes -- minimum pairwise separation within each region")
    print("(OKLab dE x100 under protan/deutan; these palettes are used one region at a "
          "time, so only within-region separation matters):")
    for region in REGION_ORDER:
        n = sum(1 for r in records if r["region"] == region)
        score = separations[region]
        if score < CVD_FLOOR:
            verdict = "FAIL"
            ok = False
        elif score < CVD_TARGET:
            verdict = "WARN"
        else:
            verdict = "ok"
        print(f"  {region:<20} n={n:<3} min dE {score:5.1f}  {verdict}")
    return ok


def check_palette(records):
    """Hard-gate the seven anchors under CVD; report within-region spread. True if OK."""
    ok = True

    # A split region contributes both of its anchors, so a sub-anchor that collides with
    # another region -- or with its own other half -- is caught by the same gate.
    anchors = []
    for region in REGION_ORDER:
        split = REGION_SUBSPLITS.get(region)
        if split:
            anchors.append((f"{region}(N)", split["north_anchor"]))
            anchors.append((f"{region}(S)", split["south_anchor"]))
        else:
            anchors.append((region, REGION_ANCHORS[region]))

    print("Region anchors, worst CVD separation (OKLab dE x100, min over protan/deutan):")
    worst = None
    for i in range(len(anchors)):
        for j in range(i + 1, len(anchors)):
            (ra, ca), (rb, cb) = anchors[i], anchors[j]
            d = min(delta_e(ca, cb, "protan"), delta_e(ca, cb, "deutan"))
            if worst is None or d < worst[0]:
                worst = (d, ra, rb)
            if d < CVD_FLOOR:
                print(f"  FAIL {ra} vs {rb}: {d:.1f} < {CVD_FLOOR}")
                ok = False
            elif d < CVD_TARGET:
                print(f"  WARN {ra} vs {rb}: {d:.1f} < {CVD_TARGET}")
    print(f"  worst pair: {worst[1]} vs {worst[2]} at dE {worst[0]:.1f} "
          f"(floor {CVD_FLOOR}, target {CVD_TARGET})")

    print("\nWithin-region separation of consecutive populations (informational -- "
          "similarity here is the geographic signal):")
    for region in REGION_ORDER:
        in_region = [r for r in records if r["region"] == region]
        # A split region is reported per hue group: its populations ramp within a group,
        # so a step measured across the N/S boundary would not describe anything.
        anchors = sorted({r["region_anchor"] for r in in_region},
                         key=lambda a: [r["region_anchor"] for r in in_region].index(a))
        for anchor in anchors:
            colors = [r["color"] for r in in_region if r["region_anchor"] == anchor]
            label = region if len(anchors) == 1 else f"{region} [{anchor}]"
            if len(colors) < 2:
                print(f"  {label:<28} n=1")
                continue
            steps = [
                min(delta_e(colors[i], colors[i + 1], "protan"),
                    delta_e(colors[i], colors[i + 1], "deutan"))
                for i in range(len(colors) - 1)
            ]
            end = min(delta_e(colors[0], colors[-1], "protan"),
                      delta_e(colors[0], colors[-1], "deutan"))
            print(f"  {label:<28} n={len(colors):<3} min step dE {min(steps):5.1f}  "
                  f"end-to-end dE {end:5.1f}")

    return ok


# --- rendering ---------------------------------------------------------------------
def wrap_label(population, limit=9):
    """Break a long CamelCase population name across lines, e.g. BantuSouthAfrica.

    Labels are horizontal, so the limit is what fits across a box rather than down it.
    """
    if len(population) <= limit:
        return population
    # Split before each capital that follows a lower-case letter.
    parts, current = [], population[0]
    for previous, character in zip(population, population[1:]):
        if character.isupper() and previous.islower():
            parts.append(current)
            current = character
        else:
            current += character
    parts.append(current)

    lines, line = [], ""
    for part in parts:
        if line and len(line) + len(part) > limit:
            lines.append(line)
            line = part
        else:
            line += part
    if line:
        lines.append(line)
    return "\n".join(lines)


def render_png(records, path, title="HGDP population colour palette"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    rows = [[r for r in records if r["region"] == region] for region in REGION_ORDER]
    widest = max(len(row) for row in rows)

    # Horizontal labels need a box wider than it is tall, and the widest region (EAST_ASIA,
    # 18 populations) sets the figure width for everyone.
    box_w, box_h, gap = 1.0, 0.72, 0.08
    label_column = 2.6

    fig_w = 1.30 * widest + label_column + 0.6
    fig_h = 1.02 * len(rows) + 0.9
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    labels = []
    for y, (region, row) in enumerate(zip(REGION_ORDER, rows)):
        top = len(rows) - y
        ax.text(-0.3, top - box_h / 2, region.replace("_", " ").title(),
                ha="right", va="center", fontsize=15, fontweight="bold")
        for x, record in enumerate(row):
            ax.add_patch(Rectangle(
                (x + gap / 2, top - box_h), box_w - gap, box_h,
                facecolor=record["color"], edgecolor="white", linewidth=1.2,
            ))
            # Label ink follows the box's luminance, so it stays legible either way.
            lum = oklab(hex_to_rgb(record["color"]))[0]
            labels.append(ax.text(
                x + box_w / 2, top - box_h / 2, wrap_label(record["population"]),
                ha="center", va="center", fontsize=LABEL_FONTSIZE, linespacing=1.1,
                color="#111111" if lum > 0.62 else "#FFFFFF",
            ))

    ax.set_xlim(-label_column, widest)
    ax.set_ylim(0.05, len(rows) + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=16, pad=16)

    # Shrink any label that still overruns its box. A character count cannot predict this:
    # glyph widths differ, and a single long word ("Bougainville") has nowhere to wrap. So
    # measure what was actually rendered and scale that label down until it fits.
    fig.canvas.draw()
    limit = (box_w - gap) * 0.94
    for label in labels:
        extent = label.get_window_extent(fig.canvas.get_renderer())
        width = ax.transData.inverted().transform(
            [(0, 0), (extent.width, 0)]
        )
        width = width[1][0] - width[0][0]
        if width > limit:
            label.set_fontsize(LABEL_FONTSIZE * limit / width)

    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", required=True, help="resources/hgdp/metadata.tsv")
    parser.add_argument("--outdir", required=True,
                        help="resources/population_labels_colors")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    columns = ["population", "region", "color", "region_anchor",
               "geographic_rank", "longitude", "latitude"]

    def write(records, stem, title):
        csv_path = os.path.join(args.outdir, f"{stem}.csv")
        with open(csv_path, "w", newline="") as out:
            writer = csv.DictWriter(out, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
        png_path = os.path.join(args.outdir, f"{stem}.png")
        render_png(records, png_path, title=title)
        return csv_path, png_path

    # Global: seven region hues, populations ramped inside each. For ALL, where every
    # region is on the plot together and region identity is what the colours must carry.
    global_records = build_palette(args.metadata)
    global_paths = write(global_records, "global_palette",
                         "HGDP global palette — region hue, population ramp")
    global_ok = check_palette(global_records)

    print()

    # Regional: seven independent max-separation palettes, for the single-region runs
    # where only telling populations apart matters.
    regional_records, separations = build_regional_palettes(args.metadata)
    regional_paths = write(regional_records, "regional_palette",
                           "HGDP regional palettes — discrete within each region")
    regional_ok = check_regional(regional_records, separations)

    print()
    for stem, (csv_path, png_path) in [("global", global_paths),
                                       ("regional", regional_paths)]:
        print(f"wrote {csv_path}")
        print(f"wrote {png_path}")

    if not global_ok:
        raise SystemExit("Global palette failed the colour-blindness gate on the anchors")
    if not regional_ok:
        raise SystemExit("A regional palette fell below the colour-blindness floor")


if __name__ == "__main__":
    main()
