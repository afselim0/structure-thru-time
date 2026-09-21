"""
Figure data: the chimpanzee dataset and what CoaSTT measures on it.

Panels this feeds:
  A  map of equatorial Africa with sample origins  -> africa_outlines.parquet, sample_locations.csv
  B  genotype PCA over the run's tree sequences    -> pca.csv
  C  Ne(t) per subspecies                          -> effective_size_mean.parquet,
                                                      effective_size_pairs.parquet
  D  relative diversity rho_AB(t) per pair         -> relative_divergence_mean.parquet,
                                                      relative_divergence_pairs.parquet, pair_colors.csv

Panels B, C and D read the chimp run under results/outputs/chimps/no_admix/ (the deep
run: pdf_maxT 400000, pdf_bin 40, windows 250-4000, matching
data/configs/app_configs/chimps.yaml).
Panel A does not depend on the run, so its tables and params.csv are written first and
the script then stops with a "missing run" message if the run is absent.

C and D are built by the same two functions the pipeline's own diagnostic plots use,
so the figure's bold curves and plots/effective_size*.png / plots/relative_div*_summary.png
are the same numbers. Note those two modules import plotting_style, and so matplotlib,
at module scope: that is an import, not drawing, and is not a convention violation here.

The country outlines are fetched once from Natural Earth (110m admin-0) and stored as a
plain table of lon/lat vertices, so the plotting script needs no GIS stack -- neither
cartopy nor geopandas is in environment.yml. The shapefile is parsed here with struct
rather than a reader library for the same reason.
"""
import io
import os
import ssl
import struct
import sys
import urllib.request
import zipfile

import certifi
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import (OUTPUTS_DIR, REPO_ROOT, add_src_to_path, find_output_file,
                     out_dir, report, require, write_table)

add_src_to_path()
# The pipeline's own aggregation functions, reused so the figure cannot drift from
# the diagnostics. RATE_FLOOR is the estimator's zero-floor: a rate at or below it is
# not a measurement, and 1/(2*1e-15) = 5e14 would set the axis on its own.
from plot_effective_size import RATE_FLOOR, compute_effective_sizes
from plot_relative_div import compute_relative_divergence
from OptimizationFunctions import procrustes_single_time

FIGURE = "chimp_dataset"

# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------
CHIMP_RUN_DIR = os.path.join(OUTPUTS_DIR, "chimps", "no_admix", "data")
SAMPLES_DIR = os.path.join(REPO_ROOT, "data", "chimp_data", "resources", "samples")
POP_COLORS_FILE = os.path.join(SAMPLES_DIR, "pop_colors.csv")
POP_LABELS_FILE = os.path.join(SAMPLES_DIR, "pop_labels.csv")

# These tokens pick the deep run out of the run directory, which also still holds a
# shallower one.
MIN_WINDOW_SIZE, MAX_WINDOW_SIZE = 250, 4000
PDF_MAX_TIME, PDF_BIN_SIZE = 400000, 40

# Natural Earth layers, as used by utils/data_utils/pop_locs_process.py. The
# coastline comes from the coarse 110m cultural layer (country polygons, of which
# only the outer coast is drawn); the rivers need the detailed 10m physical layer
# to carry the named courses at the scale this map is drawn.
NATURAL_EARTH_BASE = "https://naturalearth.s3.amazonaws.com"
# The land layer, not the country layer: a country polygon carries its inland
# borders as well as its coast, so drawing its edges draws the political map. This
# one is the landmass alone, which is all this figure wants.
NATURAL_EARTH_LAYER = "ne_110m_land"
RIVERS_LAYER = "ne_10m_rivers_lake_centerlines"
NATURAL_EARTH_LAYERS = {
    NATURAL_EARTH_LAYER: "110m_physical",
    RIVERS_LAYER: "10m_physical",
}
# Cached beside the other chimp resources so a rerun without network still works.
OUTLINES_CACHE = os.path.join(SAMPLES_DIR, f"{NATURAL_EARTH_LAYER}.zip")

# The rivers that separate the subspecies. Natural Earth splits a river into
# several named parts, so every part carrying the name is kept.
#   Sanaga  -- ellioti / troglodytes. The best supported of the river boundaries,
#              though not a hard wall: Mitchell et al. 2015 put the split at
#              200-250 kya with ~1 migrant per generation since.
#   Ubangi  -- troglodytes / schweinfurthii.
#   Congo   -- chimpanzees / bonobos. A species-level boundary, drawn as context
#              for the southern edge of the range rather than as a subspecies one.
#   Niger   -- proposed as a finer boundary near the verus / ellioti split, but
#              the evidence is weaker and more contested than for the Dahomey Gap,
#              so it is drawn in the same muted style and called out as such.
# A segment is kept only if it is at least this fraction of the longest segment
# of the same river. A fixed length would not work: the Sanaga and Ubangi are
# genuinely short rivers (3.6 and 4.3 deg end to end) and any absolute threshold
# that removes the Niger's fragments would remove them entirely. The fraction has
# to stay low enough to keep the Congo's upper reaches (3.5-4.7 deg) against its
# 16.9 deg lower course, while still dropping the Niger's distributaries.
MIN_RIVER_SEGMENT_FRACTION = 0.15
# How close two segment ends must be to count as joined, in degrees. Natural
# Earth's own reaches meet to within 0.03 deg, so this is loose enough to absorb
# rounding without bridging genuinely separate watercourses.
RIVER_JOIN_TOLERANCE_DEG = 0.1

# Natural Earth splits a long river across several named records, and the Congo's
# upper course is named Lualaba, so both names are collected under one river here:
# records 734/736 (Lualaba, Congo) chain into 738 (Lualaba) and then 740 (Congo),
# which is the westward run to the Atlantic. Keeping only the records named
# "Congo" left a gap in the middle of the river.
BOUNDARY_RIVERS = {
    "Sanaga": "ellioti / troglodytes",
    "Ubangi": "troglodytes / schweinfurthii",
    "Congo": "chimpanzees / bonobos (species level)",
    "Lualaba": "chimpanzees / bonobos (species level)",
    "Niger": "verus / ellioti (contested)",
}
# Names that are the same watercourse, so the fragment filter judges their
# segments against the whole river rather than against each record separately.
RIVER_ALIASES = {"Lualaba": "Congo"}

# The Dahomey Gap: the widest and oldest of the four barriers, and the reason
# western chimpanzees are the most distinct lineage. It is a belt of dry savanna
# reaching the coast and breaking the rainforest through Ghana, Togo and Benin --
# an area, not a line.
#
# It is taken from the RESOLVE Ecoregions 2017 layer (Dinerstein et al. 2017, An
# Ecoregion-Based Approach to Protecting Half the Terrestrial Realm, BioScience
# 67:534-545), which revises the WWF Terrestrial Ecoregions of the World. The Gap
# is the stretch of the "Guinean forest-savanna" ecoregion (ECO_ID 44, the WWF
# AT0707 Guinean forest-savanna mosaic) that reaches the sea: elsewhere the same
# ecoregion runs inland, north of the forest belt, and is not a barrier between
# these subspecies. Clipping to the longitudes where it meets the coast is what
# isolates the Gap from the rest of the belt.
#
# Using the real ecoregion rather than a drawn box also fixes a correctness bug:
# a rectangle spilled into the Gulf of Guinea, whereas every vertex of this
# polygon is on land.
ECOREGIONS_URL = "https://storage.googleapis.com/teow2016/Ecoregions2017.zip"
ECOREGIONS_LAYER = "Ecoregions2017"
ECOREGIONS_CACHE = os.path.join(SAMPLES_DIR, "Ecoregions2017.zip")
DAHOMEY_ECOREGION = "Guinean forest-savanna"
# Longitudes over which that ecoregion reaches the coast (measured from the layer
# itself: its southern limit runs 5.0-6.3 N between -2 and +3 E, against 7-8 N on
# either side, which is the Gap).
DAHOMEY_LON_RANGE = (-2.0, 3.6)

# Range polygons from the IUCN Red List of Threatened Species -- the same source
# Steux et al. 2025 (Hum Popul Genet Genom 5(2):0004) use for their Figure 1:
#   Humle T, Maisels F, Oates JF, Plumptre A, Williamson EA. Pan troglodytes.
#     IUCN Red List 2016: e.T15933A17964454
#   Fruth B, Hickey JR, Andre C, Furuichi T, Hart J, Hart T, et al. Pan paniscus.
#     IUCN Red List 2016: e.T15932A17964305
# The Red List's spatial data is behind a free account and a terms agreement, so
# it cannot be fetched here and is not redistributable in the repo. The path is
# read from the environment and the ranges are simply left out when it is unset --
# every other panel still builds. Point CHIMP_IUCN_RANGES at the unpacked
# shapefile (the .shp; its .dbf must sit beside it).
# The Red List exports one zip per assessment, each holding a data_0 shapefile:
# the Pan troglodytes download carries one record per subspecies, the Pan paniscus
# one a single species record. They sit with the other downloaded source archives
# in resources/samples/, not in the figure's own directory -- that directory is
# this script's output, and anything needed to rebuild it has to survive wiping it.
IUCN_RANGE_ARCHIVES = tuple(
    os.path.join(SAMPLES_DIR, name) for name in
    ("redlist_species_chimp_range.zip", "redlist_species_bonobo_range.zip"))
IUCN_LAYER = "data_0"

# Red List taxa mapped onto this figure's population names. A row names the taxon
# with SCI_NAME plus, for a subspecies, a separate SUBSPECIES epithet, so the two
# are joined before lookup. Bonobo is included although no bonobo was sequenced
# here: the Congo river bounds the chimpanzee range to the south, and showing what
# is on the far side is what makes that boundary legible (and is where de Manuel
# et al.'s admixture signal implies the barrier was not always closed).
BONOBO_POPULATION = "Bonobo"
IUCN_SUBSPECIES = {
    "Pan troglodytes verus": "Western",
    "Pan troglodytes ellioti": "NC",
    "Pan troglodytes vellerosus": "NC",     # the older name for ellioti
    "Pan troglodytes troglodytes": "Central",
    "Pan troglodytes schweinfurthii": "Eastern",
    "Pan paniscus": BONOBO_POPULATION,
}
# Bonobo has no sample and so no row in pop_colors.csv. Purple: distinct from
# every subspecies colour (blue, red, green, orange) and from the navy rivers, so
# the one non-chimpanzee range on the map reads as a different kind of thing.
BONOBO_COLOR = "#7B4EA8"

# Bounding box of the map panel: the chimpanzee range, from Senegal to Tanzania,
# with a few degrees of padding around the outermost sample. Vertices outside it
# are still kept, so the plotting script can clip as it likes.
# The latitude span sets the map's height, since the panel is equal-aspect and
# spans the figure's width. The top reaches 18 N so the Niger's northern bend
# through the inland delta is shown whole -- it peaks at 17.1 N, and a 14 N bound
# cut 37% of the river, leaving it running off the frame. The bottom stays at
# -13 rather than being trimmed to buy that height back: the southernmost sample
# sits at exactly -12, so a -12 bound would clip its marker against the frame.
MAP_BOUNDS = dict(lon_min=-18.0, lon_max=38.0, lat_min=-13.0, lat_max=18.0)

# Individuals whose recorded origin does not survive review, and which are
# therefore not plotted. The coordinates in sample_locations.csv are inferred
# origins from the de Manuel supplement, not sampling points, and most are rounded
# to whole or half degrees (21 of 38), so a point sitting just outside its IUCN
# range is expected. These are the cases that are wrong rather than coarse:
#
#   (0, 20)  -- a round-number placeholder shared by three individuals from two
#               different countries, which lands across the Congo INSIDE the
#               bonobo range, where no chimpanzee occurs.
#   SAMEA4374798 -- recorded DRC, coordinate falls in Zambia, 4.2 deg outside the
#               eastern range and south of the Congo.
#   SAMEA4374789 -- recorded DRC, coordinate falls in Uganda.
#
# The remaining excluded-by-absence individuals (no coordinates at all) are
# dropped by sample_locations() as before.
UNRELIABLE_LOCATIONS = {
    "SAMEA4374791": "placeholder coordinate (0, 20), inside the bonobo range",
    "SAMEA4374796": "placeholder coordinate (0, 20); recorded Zambia, plots in DRC",
    "SAMN01920531": "placeholder coordinate (0, 20), inside the bonobo range",
    "SAMEA4374798": "recorded DRC, coordinate plots in Zambia, outside the range",
    "SAMEA4374789": "recorded DRC, coordinate plots in Uganda",
    "SAMEA4374795": ("whole-degree approximation of \"DRC-South\": falls on the far "
                     "bank of the Congo, 2.4 deg outside the eastern range and 2 deg "
                     "west of every other eastern sample"),
}
# Note on the country check: it is weak for large countries. SAMEA4374795 is
# recorded as DRC and its coordinate is inside DRC, but DRC spans 18 degrees of
# latitude, so agreement there says almost nothing. Its own record gives the
# region only as "DRC-South", which is what the whole-degree coordinate encodes.

# de Manuel et al. 2016 assume a 25-year generation time for chimpanzees.
GENERATION_TIME = 25
# chimps.yaml: the pipeline's own plots start here, and 1/(2*lambda) below it is not
# an Ne -- almost no pair has coalesced that recently.
MIN_TIME = 50.0

# The hybrid is a single W x C individual: it belongs to no subspecies, so it has no
# within-population pair and is excluded from C and D. It stays in the PCA (B), where
# it is a real sample and where it falls is the point.
HYBRID_POPULATION = "W x C"
POPULATION_ORDER = ["Western", "NC", "Central", "Eastern"]

# Individual-trajectory thinning (see thin_time_index).
THIN_CROSSOVER = 2000.0
THIN_N_LOG = 300

# Panel D marks where each pair's mean rho rises through this level.
RHO_CROSSING_LEVEL = 0.5

# Panel C's mean curves peak at ~1.6e6 in the first few hundred generations, where
# almost no pair has coalesced. Capping here rather than cropping the time axis keeps
# C and D on one shared x axis; it clips 0.05% of points, all at t <= 240.
NE_YMAX_CAP = 5e5

# Panel D colours one line per POPULATION PAIR, which is a different encoding from
# the per-population colours in pop_colors.csv, so the pairs take the categorical
# fallback instead. Mirrors okabe_ito in results/figures/plotting_style.py; prep
# scripts must not import that module (it applies the matplotlib style on import),
# and six of the eight are enough for the six pairs.
# Panel D's six population pairs, grouped into three colour families (see main).
# Greens for the NC pairs, purples for the remaining Western pairs, and a single
# amber for Eastern-Central.
PAIR_COLORS = {
    frozenset(("Eastern", "Central")): "#E8A33D",   # amber, on its own
    frozenset(("NC", "Western")):      "#0B5B3B",   # greens, NC-Western darkest
    frozenset(("NC", "Eastern")):      "#2E9E6B",
    frozenset(("NC", "Central")):      "#7FD1A8",
    frozenset(("Western", "Eastern")): "#4A2E83",   # purples, Western-Eastern darker
    frozenset(("Western", "Central")): "#9B7FD4",
}

OKABE_ITO = [
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#009E73",  # green
    "#E69F00",  # orange
    "#CC79A7",  # pink
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#000000",  # black
]

# ---------------------------------------------------------------------------
# Panel A: map outlines + sample origins
# ---------------------------------------------------------------------------
def _read_dbf(data, wanted_fields):
    """Field values of every record in a dBase III table, for the named fields."""
    n_records, header_len, record_len = struct.unpack("<IHH", data[4:12])

    fields, offset = [], 32
    while data[offset] != 0x0D:          # 0x0D terminates the field descriptor array
        name = data[offset:offset + 11].split(b"\0")[0].decode("latin1")
        fields.append((name, data[offset + 16]))
        offset += 32

    records = []
    for i in range(n_records):
        record = data[header_len + i * record_len: header_len + (i + 1) * record_len]
        position, row = 1, {}         # byte 0 is the deletion flag
        for name, length in fields:
            if name in wanted_fields:
                # Natural Earth pads with NULs rather than spaces
                row[name] = record[position:position + length].decode("latin1").replace("\0", "").strip()
            position += length
        records.append(row)
    return records


def _read_shp_polygons(data):
    """Rings (lists of (lon, lat)) of every polygon record, in file order."""
    shapes, offset = [], 100          # the 100-byte file header precedes the records
    while offset < len(data):
        _, content_len = struct.unpack(">ii", data[offset:offset + 8])
        offset += 8
        body = data[offset:offset + content_len * 2]   # content length is in 16-bit words
        offset += content_len * 2

        shape_type, = struct.unpack("<i", body[0:4])
        if shape_type != 5:           # 5 = polygon; null shapes carry no geometry
            shapes.append([])
            continue

        n_parts, n_points = struct.unpack("<ii", body[36:44])
        parts = struct.unpack(f"<{n_parts}i", body[44:44 + 4 * n_parts])
        points_at = 44 + 4 * n_parts
        coords = struct.unpack(f"<{2 * n_points}d", body[points_at:points_at + 16 * n_points])

        rings = []
        for k, start in enumerate(parts):
            end = parts[k + 1] if k + 1 < n_parts else n_points
            rings.append([(coords[2 * i], coords[2 * i + 1]) for i in range(start, end)])
        shapes.append(rings)
    return shapes


def _cached_download(url, cache, what):
    """The bytes at `url`, from `cache` if it is there and from the network once."""
    if os.path.exists(cache):
        with open(cache, "rb") as handle:
            return handle.read()

    print(f"downloading {what} from {url}")
    # This cluster's python does not see the system CA bundle (the same reason the
    # kaleido Chrome fetch needs SSL_CERT_FILE), so verify against certifi's.
    context = ssl.create_default_context(cafile=certifi.where())
    with urllib.request.urlopen(url, timeout=600, context=context) as response:
        payload = response.read()
    with open(cache, "wb") as handle:
        handle.write(payload)
    return payload


def _natural_earth_bytes(layer=NATURAL_EARTH_LAYER):
    """The zip for one Natural Earth layer, from the local cache or the network."""
    return _cached_download(
        f"{NATURAL_EARTH_BASE}/{NATURAL_EARTH_LAYERS[layer]}/{layer}.zip",
        os.path.join(SAMPLES_DIR, f"{layer}.zip"), layer)


def _read_shp_polylines(data):
    """Parts (lists of (lon, lat)) of every polyline record, in file order."""
    shapes, offset = [], 100
    while offset < len(data):
        _, content_len = struct.unpack(">ii", data[offset:offset + 8])
        offset += 8
        body = data[offset:offset + content_len * 2]
        offset += content_len * 2

        shape_type, = struct.unpack("<i", body[0:4])
        if shape_type != 3:           # 3 = polyline; null shapes carry no geometry
            shapes.append([])
            continue

        n_parts, n_points = struct.unpack("<ii", body[36:44])
        parts = struct.unpack(f"<{n_parts}i", body[44:44 + 4 * n_parts])
        points_at = 44 + 4 * n_parts
        coords = struct.unpack(f"<{2 * n_points}d", body[points_at:points_at + 16 * n_points])

        segments = []
        for k, start in enumerate(parts):
            end = parts[k + 1] if k + 1 < n_parts else n_points
            segments.append([(coords[2 * i], coords[2 * i + 1]) for i in range(start, end)])
        shapes.append(segments)
    return shapes


def africa_outlines():
    """
    One row per vertex of the landmass outline, as
    (polygon_id, order, longitude, latitude).

    This is the coastline only: the land layer carries no internal borders, so
    nothing political is drawn. Rings are kept whole and identified by polygon_id
    so the plotting script can fill each as a patch. Any ring that does not reach
    the map's bounding box is dropped -- the layer is global, and the figure only
    ever shows equatorial Africa.
    """
    archive = zipfile.ZipFile(io.BytesIO(_natural_earth_bytes()))
    geometries = _read_shp_polygons(archive.read(f"{NATURAL_EARTH_LAYER}.shp"))

    rows = []
    polygon_id = 0
    for rings in geometries:
        for ring in rings:
            lon, lat = np.array(ring).T
            if (lon.max() < MAP_BOUNDS["lon_min"] or lon.min() > MAP_BOUNDS["lon_max"]
                    or lat.max() < MAP_BOUNDS["lat_min"] or lat.min() > MAP_BOUNDS["lat_max"]):
                continue
            rows.append(pd.DataFrame({
                "polygon_id": polygon_id,
                "order": np.arange(len(ring)),
                "longitude": lon,
                "latitude": lat,
            }))
            polygon_id += 1

    if not rows:
        raise ValueError("no land polygons overlap the map bounds")
    return pd.concat(rows, ignore_index=True)


def sample_locations(pop_labels):
    """
    One row per individual that has an origin, with its population and colour.

    pop_labels is per haplotype (two rows per individual), so it is collapsed to
    individuals first. Individuals without coordinates -- the four internal PD_*
    IDs, the captive-born samples and the animals whose supplement row has blank
    lat/lon (see data/chimp_data/resources/samples/LOCATIONS.md) -- cannot be put on
    a map and are dropped here, so the panel never has to decide what to do with a
    NaN. The counts of what was dropped go into params.csv for the caption.
    """
    individuals = (pop_labels
                   .drop_duplicates("individual_id")
                   .loc[:, ["individual_id", "individual_name", "population",
                            "latitude", "longitude", "origin_country"]])
    located = individuals.dropna(subset=["latitude", "longitude"]).copy()

    # Only origins that survive review are mapped (see UNRELIABLE_LOCATIONS).
    unreliable = located["individual_name"].isin(UNRELIABLE_LOCATIONS)
    for name in located.loc[unreliable, "individual_name"]:
        print(f"  dropping {name}: {UNRELIABLE_LOCATIONS[name]}")
    located = located[~unreliable]

    # Several animals share an origin exactly (the supplement's coordinates are
    # coarse -- whole or half degrees), so identical points would overplot into one
    # marker. The count travels with the point and the panel scales the marker by it.
    grouped = (located
               .groupby(["population", "latitude", "longitude"], as_index=False)
               .agg(n_individuals=("individual_id", "size"),
                    individual_names=("individual_name", lambda s: ";".join(sorted(s))),
                    origin_country=("origin_country", "first")))
    return located, grouped


def boundary_rivers():
    """
    One row per vertex of each river that separates two of the populations.

    Natural Earth breaks a river into several named parts, so every part carrying
    the name is kept and identified by segment_id; the plotting script draws each
    segment as its own line rather than joining them.
    """
    archive = zipfile.ZipFile(io.BytesIO(_natural_earth_bytes(RIVERS_LAYER)))
    attributes = _read_dbf(archive.read(f"{RIVERS_LAYER}.dbf"), {"name"})
    geometries = _read_shp_polylines(archive.read(f"{RIVERS_LAYER}.shp"))

    # Natural Earth stores a river as its whole drainage network, so a name picks
    # up tributaries, delta distributaries and (for the Congo) a detached stretch
    # of the upper Lualaba 7 degrees east of the main channel. Drawn as they come,
    # those short pieces read as stray marks floating beside the river. Only the
    # segments making up the main course are kept, judged against the longest
    # segment of the same river: that separates the three main Niger reaches
    # (7.6-12.8 deg) from its eight fragments (0.09-3.9 deg) and the Congo's main
    # channel from its detached Lualaba stub, while leaving the single-segment
    # Sanaga and Ubangi untouched.
    candidates = []
    for record, segments in zip(attributes, geometries):
        name = record.get("name")
        if name not in BOUNDARY_RIVERS:
            continue
        name = RIVER_ALIASES.get(name, name)
        for segment in segments:
            points = np.array(segment)
            length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
            candidates.append((name, points, length))

    # A short segment is kept if it BRIDGES two segments that are themselves
    # kept: the Niger's 0.85 deg piece from (4.66, 10.61) to (4.63, 9.86) joins
    # its two longest reaches, and dropping it on length alone left a visible gap
    # mid-river. So the long segments are chosen first, then any short one whose
    # two ends both touch the growing network is added back, repeatedly, until
    # nothing more connects.
    longest = {}
    for name, _, length in candidates:
        longest[name] = max(longest.get(name, 0.0), length)

    keep = [i for i, (name, _, length) in enumerate(candidates)
            if length >= MIN_RIVER_SEGMENT_FRACTION * longest[name]]
    kept_set = set(keep)
    added = True
    while added:
        added = False
        endpoints = {}
        for i in kept_set:
            name, points, _ = candidates[i]
            endpoints.setdefault(name, []).extend([points[0], points[-1]])
        for i, (name, points, _) in enumerate(candidates):
            if i in kept_set or name not in endpoints:
                continue
            ends = np.array(endpoints[name])
            touches = sum(
                np.min(np.linalg.norm(ends - end, axis=1)) <= RIVER_JOIN_TOLERANCE_DEG
                for end in (points[0], points[-1]))
            if touches >= 2:          # bridges the network at both ends
                kept_set.add(i)
                added = True

    rows, segment_id = [], 0
    for index, (name, points, length) in enumerate(candidates):
        if index not in kept_set:
            continue
        rows.append(pd.DataFrame({
            "river": name,
            "separates": BOUNDARY_RIVERS[name],
            "segment_id": segment_id,
            "order": np.arange(len(points)),
            "longitude": points[:, 0],
            "latitude": points[:, 1],
        }))
        segment_id += 1

    if not rows:
        raise ValueError("no boundary rivers found in the Natural Earth layer")
    kept = pd.concat(rows, ignore_index=True)
    dropped = len(candidates) - kept["segment_id"].nunique()
    print(f"  rivers: kept {kept['segment_id'].nunique()} main segments, "
          f"dropped {dropped} tributary/distributary fragments")
    return kept


def dahomey_gap():
    """
    The Dahomey Gap as a polygon, from the RESOLVE Ecoregions 2017 layer.

    The Guinean forest-savanna ecoregion runs right across West Africa; only the
    stretch that reaches the coast is the Gap, so the ring is clipped to
    DAHOMEY_LON_RANGE. The clip is done on the ring's own vertices and closed
    along the cut longitudes, which keeps the coastline edge exactly as the
    ecoregion draws it -- that is what stops the polygon spilling into the sea.

    Returns one row per vertex (order, longitude, latitude).
    """
    archive = zipfile.ZipFile(io.BytesIO(_cached_download(
        ECOREGIONS_URL, ECOREGIONS_CACHE, "RESOLVE Ecoregions 2017")))
    attributes = _read_dbf(archive.read(f"{ECOREGIONS_LAYER}.dbf"), {"ECO_NAME"})
    index = next((i for i, record in enumerate(attributes)
                  if record.get("ECO_NAME") == DAHOMEY_ECOREGION), None)
    if index is None:
        raise ValueError(f"{DAHOMEY_ECOREGION!r} not found in {ECOREGIONS_LAYER}")

    # The layer is 240 MB, so the record is seeked to through the .shx index
    # rather than parsing every shape.
    shx = archive.read(f"{ECOREGIONS_LAYER}.shx")
    offset, content_len = struct.unpack(">ii", shx[100 + index * 8: 108 + index * 8])
    with archive.open(f"{ECOREGIONS_LAYER}.shp") as handle:
        handle.seek(offset * 2 + 8)
        body = handle.read(content_len * 2)

    n_parts, n_points = struct.unpack("<ii", body[36:44])
    # The parts offsets are not guaranteed to be in increasing order.
    parts = sorted(struct.unpack(f"<{n_parts}i", body[44:44 + 4 * n_parts]))
    points_at = 44 + 4 * n_parts
    coords = np.array(struct.unpack(f"<{2 * n_points}d",
                                    body[points_at:points_at + 16 * n_points])
                      ).reshape(-1, 2)

    rings = []
    for k, start in enumerate(parts):
        end = parts[k + 1] if k + 1 < n_parts else n_points
        if end - start >= 4:
            rings.append(coords[start:end])
    ring = max(rings, key=len)          # the mainland belt, not the outliers

    low, high = DAHOMEY_LON_RANGE
    inside = (ring[:, 0] >= low) & (ring[:, 0] <= high)
    clipped = ring[inside]
    if len(clipped) < 4:
        raise ValueError("the Dahomey clip kept too few vertices")

    return pd.DataFrame({
        "order": np.arange(len(clipped)),
        "longitude": clipped[:, 0],
        "latitude": clipped[:, 1],
    })


def subspecies_ranges():
    """
    Range polygons per subspecies, from the IUCN Red List exports.

    The source is the same one Steux et al. 2025 use for their Figure 1:
      Humle T, Maisels F, Oates JF, Plumptre A, Williamson EA. Pan troglodytes.
        IUCN Red List 2016: e.T15933A17964454
      Fruth B, Hickey JR, Andre C, Furuichi T, Hart J, Hart T, et al. Pan paniscus.
        IUCN Red List 2016: e.T15932A17964305

    Each download is a zip holding a data_0 shapefile: the chimpanzee one has a
    record per subspecies, the bonobo one a single species record. A row names its
    taxon as SCI_NAME ("Pan troglodytes") plus, for a subspecies, a separate
    SUBSPECIES epithet ("verus"), so the two are joined before lookup.

    The archives are not redistributable, so they are read in place and never
    unpacked into the tracked tables; the derived polygons are .parquet, which is
    gitignored. Returns None when they are absent, leaving the map without range
    shading and every other panel unaffected.
    """
    archives = list(IUCN_RANGE_ARCHIVES)
    missing = [os.path.basename(a) for a in archives if not os.path.exists(a)]
    if missing:
        print(f"IUCN range archives not found ({', '.join(missing)}): "
              f"skipping the subspecies ranges")
        return None

    rows, polygon_id = [], 0
    for path in archives:
        archive = zipfile.ZipFile(path)
        attributes = _read_dbf(archive.read(f"{IUCN_LAYER}.dbf"),
                               {"SCI_NAME", "SUBSPECIES"})
        geometries = _read_shp_polygons(archive.read(f"{IUCN_LAYER}.shp"))

        for record, rings in zip(attributes, geometries):
            taxon = " ".join(part for part in (record.get("SCI_NAME"),
                                               record.get("SUBSPECIES")) if part)
            population = IUCN_SUBSPECIES.get(taxon.strip())
            if population is None:
                continue
            for ring in rings:
                lon, lat = np.array(ring).T
                rows.append(pd.DataFrame({
                    "population": population,
                    "taxon": taxon,
                    "polygon_id": polygon_id,
                    "order": np.arange(len(ring)),
                    "longitude": lon,
                    "latitude": lat,
                }))
                polygon_id += 1

    if not rows:
        raise ValueError(
            f"no Pan polygons matched in {IUCN_RANGE_ARCHIVES}; "
            f"expected one of {sorted(IUCN_SUBSPECIES)}")
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Panels C and D: the rate-derived trajectories
# ---------------------------------------------------------------------------
def thin_time_index(times, min_time=MIN_TIME, crossover=THIN_CROSSOVER,
                    n_log=THIN_N_LOG):
    """
    Column indices of the grid the individual-trajectory tables are stored on.

    Every native timepoint from min_time up to `crossover` is kept, then `n_log`
    log-spaced ones above it. The source grid is uniform at 40 generations, so at the
    recent end it is already coarser in log space (0.18 dex between t=80 and t=120)
    than any log grid could be -- thinning there would drop exactly where the curves
    move fastest. Above the crossover the curves are smooth and the axis is log, so
    the remaining gaps are at most 0.014 dex, invisible in a 2.4 in panel.

    This turns panel D's 1001 x 10000 table from ~80 MB into ~2 MB. The mean curves
    are not thinned: they are only a handful of rows and they are what the figure
    actually asserts.
    """
    times = np.asarray(times, dtype=float)
    below = np.flatnonzero((times >= min_time) & (times < crossover))
    above = np.searchsorted(times, np.geomspace(crossover, times[-1], n_log))
    above = above[above < len(times)]
    return np.unique(np.concatenate([below, above]))


def _trajectory_frame(values, times, time_index, key_frame):
    """A (n_traces x n_thinned_times) table with its identifying columns in front."""
    thinned = pd.DataFrame(values[:, time_index].astype(np.float32),
                           columns=[f"{t:g}" for t in times[time_index]])
    return pd.concat([key_frame.reset_index(drop=True), thinned], axis=1)


def effective_size_tables(rates_df, pop_labels, times, time_index):
    """
    Panel C: the mean Ne(t) per subspecies, and every within-pair trace.

    The mean is compute_effective_sizes' 1/(2*mean(lambda)): the rates are averaged
    over pairs BEFORE the reciprocal. 1/x is convex, so mean(1/lambda) >= 1/mean(lambda)
    (Jensen), and the two disagree badly exactly where lambda is small -- measured on
    this run, Central at 80 generations is 1.6e6 one way and 1.4e14 the other, the
    difference being pairs sitting on the estimator's floor. A consequence worth
    knowing when reading the panel: the bold line is NOT the average of the thin lines
    drawn under it, and sits below the visual centre of its bundle by construction.

    compute_effective_sizes skips any population with no within-pairs, which is how
    the single hybrid individual drops out without being named here.

    The traces are the same quantity for one pair, 1/(2*lambda_i), with floored pairs
    masked to NaN for the same reason the mean masks floored timepoints.
    """
    effective_sizes, _ = compute_effective_sizes(rates_df, pop_labels,
                                                 per_haplotype=False)
    keep = times >= MIN_TIME
    mean_ne = pd.DataFrame({pop: effective_sizes[pop][keep]
                            for pop in POPULATION_ORDER if pop in effective_sizes},
                           index=pd.Index(times[keep], name="time"))

    individuals = pop_labels.drop_duplicates("individual_id")
    rows, keys = [], []
    for pop in POPULATION_ORDER:
        ids = set(individuals.loc[individuals["population"] == pop, "individual_id"])
        pairs = [p for p in rates_df.index.values if p[0] in ids and p[1] in ids]
        if not pairs:
            continue
        rates = rates_df.loc[pairs].values
        with np.errstate(divide="ignore", invalid="ignore"):
            ne = 1.0 / (2.0 * np.where(rates <= RATE_FLOOR, np.nan, rates))
        rows.append(ne)
        keys.append(pd.DataFrame({"population": pop,
                                  "individual_1": [p[0] for p in pairs],
                                  "individual_2": [p[1] for p in pairs]}))

    pair_ne = _trajectory_frame(np.vstack(rows), times, time_index,
                                pd.concat(keys, ignore_index=True))
    return mean_ne, pair_ne


def relative_divergence_tables(rates_df, pop_labels, times, time_index):
    """
    Panel D: the mean rho_AB(t) per population pair, and every between-pair trace.

    rho_AB = mean_between / mean_within, where mean_within pools the two populations'
    own within-pair means. compute_relative_divergence returns all pairs of the five
    populations; the four involving the hybrid have rho entirely NaN (it has no
    within-pairs, so its mean_within is NaN), so keeping only the six pairs among the
    subspecies drops exactly the degenerate ones.

    An individual trace is that pair's own between-population rate over the POOLED
    mean within rate -- the same denominator as the mean line, as plot_relative_div
    draws it -- so the traces and the mean are the same ratio and the mean line sits
    inside its own bundle rather than beside it.
    """
    results = compute_relative_divergence(rates_df, pop_labels,
                                          id_col="individual_id")
    # Population order follows the labels file, not the alphabet, so the pair keys
    # are taken as returned rather than rebuilt.
    pairs = [key for key in results
             if key[0] in POPULATION_ORDER and key[1] in POPULATION_ORDER]

    keep = times >= MIN_TIME
    mean_rho = pd.DataFrame({f"{a}-{b}": results[(a, b)]["rho"][keep]
                             for a, b in pairs},
                            index=pd.Index(times[keep], name="time"))

    rows, keys = [], []
    for pop_a, pop_b in pairs:
        result = results[(pop_a, pop_b)]
        between = result["between_pairs"]
        with np.errstate(divide="ignore", invalid="ignore"):
            rho = rates_df.loc[between].values / result["mean_within"]
        rows.append(rho)
        keys.append(pd.DataFrame({"population_1": pop_a, "population_2": pop_b,
                                  "individual_1": [p[0] for p in between],
                                  "individual_2": [p[1] for p in between]}))

    pair_rho = _trajectory_frame(np.vstack(rows), times, time_index,
                                 pd.concat(keys, ignore_index=True))
    return mean_rho, pair_rho, pairs


def crossing_times(mean_rho, level=RHO_CROSSING_LEVEL):
    """
    When each pair's mean rho(t) first rises through `level`, in generations.

    The crossing is the point where two subspecies have gone from largely
    distinct to half as coalescent with each other as within themselves, so it is
    a compact summary of how deep each pair's separation runs.

    Interpolated in log-log between the two bracketing timepoints, since the grid
    is coarse in log time exactly where these curves are steep. Only the FIRST
    upward crossing counts: several curves wobble back and forth across the level
    (Central-NC crosses three times), and the later crossings are estimator noise
    on an essentially flat curve rather than anything about the demography.
    """
    times = mean_rho.index.to_numpy(dtype=float)
    rows = []
    for pair in mean_rho.columns:
        values = mean_rho[pair].to_numpy(dtype=float)
        usable = np.isfinite(values)
        up = np.flatnonzero(usable[:-1] & usable[1:]
                            & (values[:-1] < level) & (values[1:] >= level))
        if len(up) == 0:
            print(f"  {pair}: never reaches rho = {level}; no crossing drawn")
            continue
        i = up[0]
        log_t = np.log10(times[i:i + 2])
        log_v = np.log10(values[i:i + 2])
        crossing = 10 ** (log_t[0] + (np.log10(level) - log_v[0])
                          * (log_t[1] - log_t[0]) / (log_v[1] - log_v[0]))
        rows.append({"pair_label": pair, "level": level,
                     "crossing_generations": crossing,
                     "crossing_years": crossing * GENERATION_TIME,
                     "n_crossings": len(up)})
    return pd.DataFrame(rows)


def mean_line_limits(frame, pad=2.0, top_cap=None):
    """
    (ymin, ymax) for a log panel, taken from the MEAN lines only and padded.

    Both C and D draw hundreds of traces that run orders of magnitude past their
    means -- D's reach 5e-12 against a mean minimum of 7.9e-4, C's reach 8.4e8
    against a mean maximum of 1.6e6 -- so limits autoscaled to the traces would
    compress every mean line into a sliver. This is the rule plot_relative_div
    already uses for its own panels, lifted out.
    """
    values = frame.values.astype(float)
    finite = values[np.isfinite(values) & (values > 0)]
    low, high = finite.min() / pad, finite.max() * pad
    return low, min(high, top_cap) if top_cap is not None else high


def align_pca_to_map(pca, pop_labels):
    """
    Rotate, reflect and scale the PCA into the map's longitude/latitude frame.

    A PCA's axes are arbitrary up to rotation and sign, so the raw PC1/PC2 frame
    carries no geographic meaning. Aligning it onto the sampling coordinates puts
    the two panels in one frame: the inset can then be read directly against the
    map above it, and how well the alignment fits is itself the result -- PC1
    tracks longitude at r = 0.90 across these samples.

    The rotation is the repo's own procrustes_single_time, estimated from the 38
    individuals that have both a PCA position and an origin (the rest have no
    coordinates -- see LOCATIONS.md). Reflection is allowed: a PCA axis has no
    inherent sign, so forbidding it would only make the fit worse for no reason.
    A uniform scale is fitted afterwards, since PC units and degrees differ by a
    factor of ~65 and a rotation alone cannot bridge that.

    Both frames are centred on the located individuals, so the aligned coordinates
    are offsets in degrees about that centroid rather than absolute positions: the
    inset is a rotated PCA, not a claim about where any animal lives.
    """
    per_individual = (pca
                      .merge(pop_labels[["sample_id", "individual_id"]],
                             on="sample_id", how="left")
                      .groupby("individual_id", as_index=False)[["PC1", "PC2"]]
                      .mean())
    origins = (pop_labels
               .drop_duplicates("individual_id")
               .loc[:, ["individual_id", "longitude", "latitude"]])
    merged = per_individual.merge(origins, on="individual_id", how="left")
    located = merged.dropna(subset=["longitude", "latitude"])

    reference = located[["longitude", "latitude"]].values
    subject = located[["PC1", "PC2"]].values
    _, rotation = procrustes_single_time(reference, subject, allow_reflection=True)

    # The scale that best matches the rotated PCA to the map, both centred.
    reference_centred = reference - reference.mean(axis=0)
    rotated_located = (subject - subject.mean(axis=0)) @ rotation
    scale = ((rotated_located * reference_centred).sum()
             / (rotated_located * rotated_located).sum())

    # Applied to every individual, including those with no origin of their own.
    rotated = (merged[["PC1", "PC2"]].values - subject.mean(axis=0)) @ rotation * scale
    aligned = merged[["individual_id"]].copy()
    aligned["PC1_aligned"] = rotated[:, 0]
    aligned["PC2_aligned"] = rotated[:, 1]

    residual = rotated_located * scale - reference_centred
    fit = {
        "pca_align_n_individuals": len(located),
        "pca_align_scale": scale,
        "pca_align_reflected": float(np.linalg.det(rotation) < 0),
        "pca_align_r_longitude": np.corrcoef(rotated_located[:, 0] * scale,
                                             reference_centred[:, 0])[0, 1],
        "pca_align_r_latitude": np.corrcoef(rotated_located[:, 1] * scale,
                                            reference_centred[:, 1])[0, 1],
        "pca_align_fit": 1.0 - (residual ** 2).sum() / (reference_centred ** 2).sum(),
    }
    return aligned, fit


# ---------------------------------------------------------------------------
def main():
    destination = out_dir(FIGURE)
    written = []

    pop_colors = pd.read_csv(require(POP_COLORS_FILE, "chimp population colours"))
    pop_labels = pd.read_csv(require(POP_LABELS_FILE, "chimp population labels"))
    pop_labels = pop_labels.loc[:, ~pop_labels.columns.str.startswith("Unnamed")]

    # -- A: map ------------------------------------------------------------
    outlines = africa_outlines()
    located, grouped = sample_locations(pop_labels)

    # params.csv is written here, before the run-dependent panels, and rewritten with
    # their numbers below. Panel A's map bounds live in it, so writing it only at the
    # end would leave the map undrawable until the run finishes.
    map_params = [
        {"parameter": "generation_time", "value": GENERATION_TIME},
        {"parameter": "n_individuals", "value": pop_labels["individual_id"].nunique()},
        {"parameter": "n_located_individuals", "value": len(located)},
        {"parameter": "n_map_points", "value": len(grouped)},
        {"parameter": "min_time", "value": MIN_TIME},
        {"parameter": "max_time", "value": PDF_MAX_TIME},
        {"parameter": "pdf_bin_size", "value": PDF_BIN_SIZE},
        {"parameter": "min_window_size", "value": MIN_WINDOW_SIZE},
        {"parameter": "max_window_size", "value": MAX_WINDOW_SIZE},
    ] + [{"parameter": f"map_{key}", "value": value}
         for key, value in MAP_BOUNDS.items()]

    rivers = boundary_rivers()
    ranges = subspecies_ranges()
    dahomey = dahomey_gap()

    written += [
        write_table(outlines, os.path.join(destination, "africa_outlines.parquet")),
        write_table(rivers, os.path.join(destination, "boundary_rivers.parquet")),
        write_table(dahomey, os.path.join(destination, "dahomey_gap.parquet")),
        write_table(grouped, os.path.join(destination, "sample_locations.csv"), index=False),
        write_table(pop_colors, os.path.join(destination, "pop_colors.csv"), index=False),
        write_table(pd.DataFrame([{"population": BONOBO_POPULATION,
                                   "color": BONOBO_COLOR}]),
                    os.path.join(destination, "bonobo_color.csv"), index=False),
        write_table(pop_labels, os.path.join(destination, "pop_labels.csv"), index=False),
        write_table(pd.DataFrame(map_params),
                    os.path.join(destination, "params.csv"), index=False),
    ]
    if ranges is not None:
        written.append(write_table(
            ranges, os.path.join(destination, "subspecies_ranges.parquet")))

    # -- B: PCA ------------------------------------------------------------
    # PCA is a genotype GRM and carries no time tag, so nothing time-like selects it.
    pca = pd.read_csv(find_output_file(CHIMP_RUN_DIR, "pca"))
    # The inset is drawn in the map's frame, so the alignment happens here: it is a
    # computation over the data, and the plotting script only draws what it finds.
    aligned_pca, pca_fit = align_pca_to_map(pca, pop_labels)
    written += [
        write_table(pca, os.path.join(destination, "pca.csv"), index=False),
        write_table(aligned_pca, os.path.join(destination, "pca_aligned.csv"),
                    index=False),
    ]

    # -- C and D: the rate trajectories ------------------------------------
    rates_df = pd.read_parquet(find_output_file(
        CHIMP_RUN_DIR, "rates", minWS=MIN_WINDOW_SIZE, maxWS=MAX_WINDOW_SIZE,
        pdf_maxT=PDF_MAX_TIME, pdf_bin=PDF_BIN_SIZE))

    # Restrict the labels to the individuals the rate table actually covers, as
    # plot_effective_size.main does, so a stale label row cannot invent a population.
    active = np.unique([s for pair in rates_df.index.values for s in pair])
    run_labels = pop_labels[pop_labels["individual_id"].isin(active)]

    times = rates_df.columns.astype(float).values
    time_index = thin_time_index(times)

    mean_ne, pair_ne = effective_size_tables(rates_df, run_labels, times, time_index)
    mean_rho, pair_rho, pairs = relative_divergence_tables(rates_df, run_labels,
                                                           times, time_index)

    # A pair is not a population, so its colour cannot come from pop_colors.csv.
    # The six pairs are grouped by which subspecies they involve, so the panel
    # reads as three families rather than six unrelated lines:
    #   Eastern-Central            its own colour (the two that still exchange migrants)
    #   the three NC pairs         one hue family, NC-Western darkest
    #   the two remaining Western  a second hue family
    # Within a family the shades are ordered deliberately, so a reader can see at
    # a glance which pairs share a population.
    pair_colors = pd.DataFrame([
        {"population_1": a, "population_2": b, "pair_label": f"{a}-{b}",
         "color": PAIR_COLORS[frozenset((a, b))]}
        for a, b in pairs])

    crossings = crossing_times(mean_rho)
    ne_ymin, ne_ymax = mean_line_limits(mean_ne, top_cap=NE_YMAX_CAP)
    rho_ymin, rho_ymax = mean_line_limits(mean_rho)

    written += [
        write_table(mean_ne, os.path.join(destination, "effective_size_mean.parquet")),
        write_table(pair_ne, os.path.join(destination, "effective_size_pairs.parquet"),
                    index=False),
        write_table(mean_rho, os.path.join(destination, "relative_divergence_mean.parquet")),
        write_table(pair_rho, os.path.join(destination, "relative_divergence_pairs.parquet"),
                    index=False),
        write_table(pair_colors, os.path.join(destination, "pair_colors.csv"), index=False),
        write_table(crossings, os.path.join(destination, "rho_crossings.csv"), index=False),
    ]

    params = pd.DataFrame(map_params + [
        {"parameter": "n_pairs", "value": len(rates_df)},
        {"parameter": "n_within_pairs", "value": len(pair_ne)},
        {"parameter": "n_between_pairs", "value": len(pair_rho)},
        {"parameter": "n_thinned_timepoints", "value": len(time_index)},
        {"parameter": "ne_ymin", "value": ne_ymin},
        {"parameter": "ne_ymax", "value": ne_ymax},
        {"parameter": "rho_ymin", "value": rho_ymin},
        {"parameter": "rho_ymax", "value": rho_ymax},
    ] + [{"parameter": key, "value": value} for key, value in pca_fit.items()])
    written.append(write_table(params, os.path.join(destination, "params.csv"),
                               index=False))

    report(written)


if __name__ == "__main__":
    main()
