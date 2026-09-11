"""Build the two map assets the input-data summary draws its regions on.

``tools/input_data_summary.py`` draws two maps, at either of two levels. It
needs a boundary for every country the model carries and for every bidding zone
inside them, small enough to keep in the repository and readable with nothing
but the standard library. This makes those files, once, by hand. No build runs
them.

Two sources, doing two different jobs. **Natural Earth** states what the land
is, and every border drawn comes from it. The **ENTSO-E bidding-zone layer**
states only which zone a piece of that land belongs to.

That division is not tidiness, it is the only honest arrangement available.
The ENTSO-E outline of Norway is 31.4% sea -- 464,167 square kilometres against
320,834 of land, a median 13 km offshore and up to 88, which is territorial
waters with the fjords and the skerry belt filled in. And the two sources
disagree about coastlines in both directions: 7.1% of Denmark's land lies
outside the ENTSO-E envelope, Copenhagen sits in a hole in the source's own
DK_2, and Bornholm is inside no Danish zone at all. So the operation is
assignment and never intersection -- an intersection would delete Copenhagen.

The zone identifiers are read, not invented. The ENTSO-E source spells zones
``DE_LU``, ``DK_1``, ``NO_1``, ``SE_1``; the model spells the same places
``DE00``, ``DKW1``, ``NOS0``, ``SE01``. The translation comes from the ``area
codes`` sheet of ``src_files/data_files/transferdata_TYNDP2020.xlsx``, which is
tracked and names all of them. Which Natural Earth *map units* make up a country
is hardcoded instead, in ``COUNTRY_UNITS``: that is a decision rather than data
that drifts, and every automatic alternative reaches for something else's
territory -- see the comment there.

Every zone the crosswalk names is written, not only the ones a scenario happens
to build, so adding Italy or Portugal to a config later needs no new asset. The
countries around them are written too, as context to draw in grey, and the
report tool can switch them off.

How a zone gets its shape
-------------------------
Fourteen of the seventeen model countries have one zone, and that zone is simply
the country. For Norway, Sweden and Denmark, each Natural Earth land polygon is
either wholly inside one zone -- in which case it is emitted untouched -- or
inside none, and goes to the nearest zone of its own country, or it spans zones
and has to be cut. Exactly two polygons in the whole dataset span: the Norwegian
and the Swedish mainland. Every Danish island, Gotland, Oland and Lofoten keep
their source outline exactly.

The cut is the one thing here resolved on a grid. Both sides of a seam are
traced from the same grid, walking cell edges, so the boundary between two zones
is shared vertex for vertex and a country's zones add up to the country.

What it cannot see
------------------
Whether the boundaries are current. They are a snapshot of two datasets; a zone
that splits after they were published stays whole here, and nothing in this file
would notice.

Whether a zone is where the model thinks it is. The crosswalk is a name-to-name
table, so an error in it produces a map that is wrong and confident.

Which zone a piece of coast really belongs to, where the two sources disagree
about where the coast is. Land no zone claims goes to the nearest one, which is
right for Bornholm and is a guess everywhere else it happens.

Where exactly the divider inside Norway and inside Sweden runs. It is the
ENTSO-E line resampled onto a 0.02 degree grid, not a surveyed boundary, so it
is accurate to about a grid cell -- enough to draw at 39 pixels to the degree,
not enough to measure with.

That a country's zones still add up to it *after* simplification. They do
before: the seam is shared exactly. Douglas-Peucker then runs per ring, and the
same seam in two rings can simplify differently, by up to the tolerance. The
report strokes every zone with a wider white edge, so it does not show; the
figure the tool prints is measured before simplification.

Usage:
    python tools/prepare_zone_geometry.py [--natural-earth FILE] [--source FILE]
        [--crosswalk FILE] [--out-dir DIR] [--tolerance DEG] [--min-ring DEG]
        [--split-resolution DEG]

The Natural Earth file is ``ne_50m_admin_0_map_units`` from
https://github.com/nvkelso/natural-earth-vector (the geojson/ folder). The
ENTSO-E layer is the Mopo extract; see ATTRIBUTION.md beside the assets.

Exit 0 when both files were written, 1 when a source could not be read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_NATURAL_EARTH = _REPO_ROOT / "example_maps" / "ne_50m_admin_0_map_units.geojson"
DEFAULT_SOURCE = _REPO_ROOT / "example_maps" / "entsoe_shp" / "entsoe_bid.geojson"
DEFAULT_CROSSWALK = _REPO_ROOT / "src_files" / "data_files" / "transferdata_TYNDP2020.xlsx"

DEFAULT_OUT_DIR = _REPO_ROOT / "tools" / "data"
COUNTRY_ASSET_NAME = "country_shapes.geojson"
ZONE_ASSET_NAME = "zone_shapes.geojson"

CROSSWALK_SHEET = "area codes"
CROSSWALK_HEADER = "transparency platform"

#: Coordinates are degrees, and the figure is about 39 px per degree at the
#: report's width and dpi, so a 0.05 degree tolerance moves a boundary by about
#: two pixels -- under what the reader can see, and a twentieth of the file size.
DEFAULT_TOLERANCE = 0.05

#: A ring smaller than this in both directions is an islet that renders as one
#: pixel. Dropping them is most of the saving and none of the shape.
DEFAULT_MIN_RING = 0.1

#: Three decimals is about 100 m. The tolerance above is already far coarser.
COORD_DECIMALS = 3

#: Grid step for the two landmasses that span zones. Half the tolerance above,
#: so the staircase it leaves is finer than the simplification that follows it
#: and about half a pixel in the report.
DEFAULT_SPLIT_STEP = 0.02

#: Which Natural Earth map units make up each model country, by ``SU_A3``.
#:
#: Hardcoded on purpose, and the one table in this file that is. Which map units
#: make up Germany is a decision, not data that drifts, and the alternatives all
#: reach for something else's territory: ``ISO_A2_EH == "ES"`` brings the Canary
#: Islands, ``"PT"`` the Azores at 31 W, ``"NO"`` Jan Mayen at 9 W;
#: ``ADM0_A3 == "FRA"`` brings French Guiana and ``"NOR"`` brings Svalbard. The
#: maps autoscale, so any one of them silently shrinks Europe. Zone *names* do
#: drift, and those stay data-driven -- see ``read_crosswalk``.
#:
#: UK00 is Great Britain, without Northern Ireland: PECD carries that separately
#: as UKNI and the model does not build it.
COUNTRY_UNITS: Dict[str, Tuple[str, ...]] = {
    "AT": ("AUT",),
    "BE": ("BFR", "BWR", "BCR"),
    "CH": ("CHE",),
    "DE": ("DEU", "LUX"),
    "DK": ("DNK",),
    "EE": ("EST",),
    "ES": ("ESP",),
    "FI": ("FIN",),
    "FR": ("FXX",),
    "LT": ("LTU",),
    "LV": ("LVA",),
    "NL": ("NLD",),
    "NO": ("NOR",),
    "PL": ("POL",),
    "PT": ("PRX",),
    "SE": ("SWE",),
    "UK": ("ENG", "SCT", "WLS"),
}

#: The ring of countries drawn in grey around the modelled ones. The set is the
#: one the ENTSO-E layer carries beyond the model's own zones, plus Ireland and
#: Northern Ireland -- without those two, Great Britain sits in an empty sea and
#: reads as though the model stopped at the coast.
NEIGHBOUR_UNITS: Dict[str, Tuple[str, ...]] = {
    "BG": ("BGR",),
    "CZ": ("CZE",),
    "GR": ("GRC",),
    "HR": ("HRV",),
    "HU": ("HUN",),
    "IE": ("IRL",),
    "IT": ("ITA",),
    "NIR": ("NIR",),
    "RO": ("ROU",),
    "RS": ("SRS", "SRV"),
    "SI": ("SVN",),
    "SK": ("SVK",),
}

#: Second line of defence, in degrees ``(west, south, east, north)``. A ring
#: wholly outside it is dropped and counted. The unit table above keeps out most
#: far-flung territory, but the Canary Islands are inside the ``ESP`` unit itself
#: and cannot be excluded by unit at all. The window is the extent of the map the
#: report already draws, with a margin: Ireland reaches 10.4 W and Crete 34.9 N.
KEEP_BBOX = (-11.0, 34.0, 33.0, 72.0)


def read_crosswalk(path: Path) -> Dict[str, str]:
    """``{'DE_LU': 'DE00', 'NO_1': 'NOS0', ...}`` from the tracked sheet.

    The sheet is a block floating in an otherwise empty grid -- the header sits
    several rows down and its two columns are unnamed -- so the pair is found by
    looking for the header text rather than by position.
    """
    frame = pd.read_excel(path, sheet_name=CROSSWALK_SHEET, header=None)
    left = start = None
    for row in range(len(frame)):
        for col in range(frame.shape[1] - 1):
            if str(frame.iat[row, col]).strip().lower() == CROSSWALK_HEADER:
                left, start = col, row + 1
                break
        if left is not None:
            break
    if left is None:
        raise ValueError(
            f"{path.name}: no {CROSSWALK_HEADER!r} header in sheet {CROSSWALK_SHEET!r}"
        )

    mapping: Dict[str, str] = {}
    for row in range(start, len(frame)):
        source = frame.iat[row, left]
        model = frame.iat[row, left + 1]
        if pd.isna(source) or pd.isna(model):
            continue
        mapping[str(source).strip()] = str(model).strip()
    return mapping


def douglas_peucker(
    points: Sequence[Tuple[float, float]], tolerance: float
) -> List[Tuple[float, float]]:
    """Drop the vertices that move the line by less than ``tolerance``.

    Iterative rather than recursive: a Norwegian ring runs to thousands of
    points and the recursive form can reach Python's stack limit on one.
    """
    if len(points) < 3:
        return list(points)

    keep = [False] * len(points)
    keep[0] = keep[-1] = True
    stack = [(0, len(points) - 1)]

    while stack:
        first, last = stack.pop()
        if last <= first + 1:
            continue
        x1, y1 = points[first]
        x2, y2 = points[last]
        dx, dy = x2 - x1, y2 - y1
        span = math.hypot(dx, dy)

        worst, at = -1.0, first
        for i in range(first + 1, last):
            x0, y0 = points[i]
            if span:
                distance = abs(dy * x0 - dx * y0 + x2 * y1 - y2 * x1) / span
            else:
                distance = math.hypot(x0 - x1, y0 - y1)
            if distance > worst:
                worst, at = distance, i

        if worst > tolerance:
            keep[at] = True
            stack.append((first, at))
            stack.append((at, last))

    return [point for point, wanted in zip(points, keep) if wanted]


def simplify_polygons(coordinates, tolerance: float, min_ring: float) -> List:
    """One zone's polygons, simplified and rounded.

    A ring that simplification takes below four points has no area left to draw
    and is dropped; the first point is copied onto the last so the ring stays
    closed, which rounding can otherwise break by a millimetre.
    """
    out: List = []
    for polygon in coordinates:
        rings: List = []
        for ring in polygon:
            xs = [point[0] for point in ring]
            ys = [point[1] for point in ring]
            if (max(xs) - min(xs)) < min_ring and (max(ys) - min(ys)) < min_ring:
                continue
            simplified = douglas_peucker(
                [(float(point[0]), float(point[1])) for point in ring], tolerance
            )
            if len(simplified) < 4:
                continue
            rounded = [[round(x, COORD_DECIMALS), round(y, COORD_DECIMALS)]
                       for x, y in simplified]
            rounded[-1] = list(rounded[0])
            rings.append(rounded)
        if rings:
            out.append(rings)
    return out


def as_multipolygon(geometry) -> List:
    """Polygon and MultiPolygon differ by one level of nesting; equalise them."""
    if geometry["type"] == "Polygon":
        return [geometry["coordinates"]]
    if geometry["type"] == "MultiPolygon":
        return geometry["coordinates"]
    raise ValueError(f"unsupported geometry {geometry['type']!r}")


def ring_outside_window(ring) -> bool:
    """True when a ring's bounding box misses the map window entirely."""
    west, south, east, north = KEEP_BBOX
    xs = [point[0] for point in ring]
    ys = [point[1] for point in ring]
    return max(xs) < west or min(xs) > east or max(ys) < south or min(ys) > north


def load_map_units(path: Path, wanted: Sequence[str]) -> Tuple[Dict[str, List], Dict[str, int]]:
    """``({SU_A3: [polygon, ...]}, {SU_A3: rings dropped})`` from Natural Earth.

    Only the units asked for, so that the drop report is about this map rather
    than about the two hundred other countries in the file. What was dropped is
    returned rather than swallowed: a silent drop is how Great Britain went
    missing from the first version of this asset, and the window rule is exactly
    the kind of thing that would do it again.
    """
    keep = set(wanted)
    data = json.loads(path.read_text(encoding="utf-8"))
    units: Dict[str, List] = {}
    dropped: Dict[str, int] = {}
    for feature in data.get("features", []):
        unit = str(feature.get("properties", {}).get("SU_A3", "")).strip()
        if unit not in keep:
            continue
        for polygon in as_multipolygon(feature["geometry"]):
            if ring_outside_window(polygon[0]):
                dropped[unit] = dropped.get(unit, 0) + 1
                continue
            units.setdefault(unit, []).append(polygon)
    return units, dropped


def units_for(units: Dict[str, List], wanted: Sequence[str]) -> List:
    """One area's Natural Earth polygons, dissolved into a single outline.

    Four of the seventeen model countries are more than one map unit -- Germany
    is Germany and Luxembourg, the United Kingdom three, Belgium three regions
    with Brussels held as a hole inside Flanders. Drawn undissolved they show a
    white hairline along a border the model does not have.
    """
    polygons = [polygon for unit in wanted for polygon in units.get(unit, [])]
    if not polygons:
        return []
    return dissolve_polygons(polygons)


def ring_area(ring) -> float:
    """Signed shoelace area in square degrees. Sign is the winding direction."""
    total = 0.0
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        total += x1 * y2 - x2 * y1
    return total / 2.0


def dissolve(rings) -> List[List]:
    """The outline of rings that share their internal borders vertex for vertex.

    Every directed edge along a border between two of the rings appears twice --
    once each way -- so cancelling each edge against its exact reverse leaves the
    outline and nothing else. What survives is stitched head to tail.

    This works because Natural Earth shares its vertices: on the borders this
    tool dissolves, every shared edge has an exact reverse twin (30 of 30 on
    Germany-Luxembourg, 142 of 142 across the three British units). The ENTSO-E
    layer does not -- six of 4,636 along the four Swedish zones -- which is why
    the geometry is taken from Natural Earth and only the zone labels from
    ENTSO-E.
    """
    edges: Dict[Tuple[Tuple[float, float], Tuple[float, float]], int] = {}
    for ring in rings:
        points = [(float(p[0]), float(p[1])) for p in ring]
        if points[0] != points[-1]:
            points.append(points[0])
        for a, b in zip(points[:-1], points[1:]):
            if a == b:
                continue
            edges[(a, b)] = edges.get((a, b), 0) + 1

    for a, b in list(edges):
        shared = min(edges.get((a, b), 0), edges.get((b, a), 0))
        if not shared:
            continue
        for edge in ((a, b), (b, a)):
            edges[edge] -= shared
            if not edges[edge]:
                del edges[edge]

    outgoing: Dict[Tuple[float, float], List[Tuple[float, float]]] = {}
    for (a, b), count in edges.items():
        outgoing.setdefault(a, []).extend([b] * count)

    out: List[List] = []
    while outgoing:
        start = next(iter(outgoing))
        ring = [start]
        here = start
        while True:
            following = outgoing.get(here)
            if not following:
                raise ValueError(
                    "dissolve: the outline dead-ends -- these rings do not share "
                    "their borders exactly, so there is no honest union"
                )
            step = following.pop()
            if not following:
                del outgoing[here]
            ring.append(step)
            here = step
            if here == start:
                break
        out.append([[x, y] for x, y in ring])
    return out


def point_in_ring(point, ring) -> bool:
    """Even-odd crossing test for one point against one ring."""
    x, y = point
    inside = False
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        if (y1 > y) != (y2 > y) and x < (x2 - x1) * (y - y1) / (y2 - y1) + x1:
            inside = not inside
    return inside


def nest_rings(rings) -> List[List]:
    """Rings sorted into polygons, each an outer ring followed by its holes.

    Depth by containment rather than by winding direction: a source is free to
    wind its rings either way, and this file has already dissolved some of them,
    but a ring inside an odd number of others is a hole under any convention.
    """
    areas = [abs(ring_area(ring)) for ring in rings]
    containers: List[List[int]] = []
    for index, ring in enumerate(rings):
        containers.append([
            other for other, ring_other in enumerate(rings)
            if other != index and point_in_ring(ring[0], ring_other)
        ])

    polygons: Dict[int, List] = {}
    holes: List[Tuple[int, int]] = []
    for index, inside in enumerate(containers):
        if len(inside) % 2 == 0:
            polygons[index] = [rings[index]]
        else:
            parent = min(inside, key=lambda other: areas[other])
            holes.append((parent, index))
    for parent, hole in holes:
        polygons.setdefault(parent, [rings[parent]]).append(rings[hole])
    return [polygons[index] for index in sorted(polygons)]


def dissolve_polygons(polygons) -> List:
    """One outline from several polygons, holes included.

    Holes take part in the cancellation rather than being carried through: an
    enclave held as a hole in one unit and as the outer ring of another --
    Brussels inside Flanders is the case here -- cancels away, which is the
    right answer and is not reachable by treating holes separately.
    """
    rings = [ring for polygon in polygons for ring in polygon]
    if not rings:
        return []
    return nest_rings(dissolve(rings))


# ---------------------------------------------------------------------------
# Which zone a piece of land belongs to
#
# Natural Earth says what the land is; the ENTSO-E layer says only which zone a
# piece of it belongs to. That is assignment, never intersection: the two
# sources disagree about coastlines in both directions -- 7.1% of Denmark's land
# lies outside the ENTSO-E envelope, and Copenhagen sits in a hole in the
# source's own DK_2 -- so an intersection would delete real land.
# ---------------------------------------------------------------------------


def _crossings(px: np.ndarray, py: np.ndarray, ring, chunk: int = 2048) -> np.ndarray:
    """How many edges of ``ring`` a ray from each point crosses."""
    edge = np.asarray(ring, dtype=float)
    x1, y1, x2, y2 = edge[:-1, 0], edge[:-1, 1], edge[1:, 0], edge[1:, 1]
    total = np.zeros(px.shape, dtype=np.int64)
    for start in range(0, len(x1), chunk):
        ax1, ay1 = x1[start:start + chunk], y1[start:start + chunk]
        ax2, ay2 = x2[start:start + chunk], y2[start:start + chunk]
        straddles = (ay1[None, :] > py[:, None]) != (ay2[None, :] > py[:, None])
        if not straddles.any():
            continue
        with np.errstate(divide="ignore", invalid="ignore"):
            at = ((ax2 - ax1)[None, :] * (py[:, None] - ay1[None, :])
                  / (ay2 - ay1)[None, :] + ax1[None, :])
        total += (straddles & (px[:, None] < at)).sum(axis=1)
    return total


def point_in_polygons(points: np.ndarray, polygons) -> np.ndarray:
    """Inside any of these polygons, each with its own holes subtracted.

    Even-odd within one polygon, then OR across polygons -- not one even-odd
    sweep over every ring of the multipolygon. Where two parts of a multipolygon
    overlap, the single sweep reports "outside" for a point inside both, and
    these sources do overlap in places.
    """
    px, py = points[:, 0], points[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    for polygon in polygons:
        count = np.zeros(len(points), dtype=np.int64)
        for ring in polygon:
            count += _crossings(px, py, ring)
        inside |= (count % 2 == 1)
    return inside


def rings_cross(ring_a, ring_b, chunk: int = 2048) -> bool:
    """True when an edge of one ring properly crosses an edge of the other.

    The guard on "wholly inside one zone". Natural Earth's Swedish mainland
    averages 20 to 50 km between vertices, so a polygon whose every vertex sits
    in one zone can still bulge across the divider between two of them -- and a
    vertex test would then skip the very branch that would have split it.
    Touching at a point is not a crossing, which keeps a shared border from
    counting as one.
    """
    a = np.asarray(ring_a, dtype=float)
    b = np.asarray(ring_b, dtype=float)
    ax1, ay1, ax2, ay2 = a[:-1, 0], a[:-1, 1], a[1:, 0], a[1:, 1]
    west, east = min(ax1.min(), ax2.min()), max(ax1.max(), ax2.max())
    south, north = min(ay1.min(), ay2.min()), max(ay1.max(), ay2.max())

    bx1, by1, bx2, by2 = b[:-1, 0], b[:-1, 1], b[1:, 0], b[1:, 1]
    near = ((np.minimum(bx1, bx2) <= east) & (np.maximum(bx1, bx2) >= west)
            & (np.minimum(by1, by2) <= north) & (np.maximum(by1, by2) >= south))
    if not near.any():
        return False
    bx1, by1, bx2, by2 = bx1[near], by1[near], bx2[near], by2[near]

    def side(x1, y1, x2, y2, x, y):
        return np.sign((x2 - x1) * (y - y1) - (y2 - y1) * (x - x1))

    for start in range(0, len(ax1), chunk):
        sx1 = ax1[start:start + chunk][:, None]
        sy1 = ay1[start:start + chunk][:, None]
        sx2 = ax2[start:start + chunk][:, None]
        sy2 = ay2[start:start + chunk][:, None]
        d1 = side(sx1, sy1, sx2, sy2, bx1[None, :], by1[None, :])
        d2 = side(sx1, sy1, sx2, sy2, bx2[None, :], by2[None, :])
        d3 = side(bx1[None, :], by1[None, :], bx2[None, :], by2[None, :], sx1, sy1)
        d4 = side(bx1[None, :], by1[None, :], bx2[None, :], by2[None, :], sx2, sy2)
        if bool(((d1 * d2 < 0) & (d3 * d4 < 0)).any()):
            return True
    return False


def zones_touching(polygon, zone_polygons: Dict[str, List]) -> List[str]:
    """Every zone this land polygon has any of, by containment or by crossing."""
    vertices = np.asarray(polygon[0], dtype=float)
    touching = []
    for zone in sorted(zone_polygons):
        parts = zone_polygons[zone]
        if point_in_polygons(vertices, parts).any():
            touching.append(zone)
            continue
        if any(rings_cross(polygon[0], ring) for part in parts for ring in part):
            touching.append(zone)
    return touching


def nearest_zone(polygon, zone_polygons: Dict[str, List]) -> str:
    """The closest zone, for land no zone contains.

    Restricted to one country's zones by the caller, and that restriction is
    load-bearing: Bornholm's nearest model zone of any country is SE04, 35 km
    away, against DKE1's 136 km. Unrestricted, Bornholm becomes Swedish and the
    map is wrong and confident. Longitude is scaled by the cosine of the
    latitude so a degree means the same in both directions.
    """
    here = np.asarray(polygon[0], dtype=float)
    scale = math.cos(math.radians(float(here[:, 1].mean())))
    best, chosen = math.inf, ""
    for zone in sorted(zone_polygons):
        for part in zone_polygons[zone]:
            for ring in part:
                there = np.asarray(ring, dtype=float)
                dx = (here[:, 0][:, None] - there[:, 0][None, :]) * scale
                dy = here[:, 1][:, None] - there[:, 1][None, :]
                distance = float(np.min(dx * dx + dy * dy))
                if distance < best:
                    best, chosen = distance, zone
    return chosen


def assign_polygons(
    polygons, zone_polygons: Dict[str, List]
) -> Tuple[Dict[str, List], List, List[str]]:
    """``({zone: [polygon, ...]}, [polygons that span zones], [notes])``."""
    placed: Dict[str, List] = {}
    spanning: List = []
    notes: List[str] = []
    for polygon in polygons:
        touching = zones_touching(polygon, zone_polygons)
        if len(touching) > 1:
            spanning.append(polygon)
        elif len(touching) == 1:
            placed.setdefault(touching[0], []).append(polygon)
        else:
            zone = nearest_zone(polygon, zone_polygons)
            placed.setdefault(zone, []).append(polygon)
            notes.append(f"no zone contains a {len(polygon[0])}-point island; "
                         f"placed in {zone}, its nearest")
    return placed, spanning, notes


# ---------------------------------------------------------------------------
# Splitting the polygons that span zones
# ---------------------------------------------------------------------------


def rasterise(polygon, x0: float, y0: float, step: float, nx: int, ny: int) -> np.ndarray:
    """A polygon filled onto a grid, by scanline rather than per-point testing.

    One pass per raster row over every edge at once. Per-point point-in-polygon
    on the same grid takes minutes; this takes a fiftieth of a second on
    Norway's three-million-cell grid.
    """
    edges = []
    for ring in polygon:
        ring_array = np.asarray(ring, dtype=float)
        edges.append(np.column_stack([
            ring_array[:-1, 0], ring_array[:-1, 1],
            ring_array[1:, 0], ring_array[1:, 1],
        ]))
    if not edges:
        return np.zeros((ny, nx), dtype=bool)
    edge = np.vstack(edges)
    x1, y1, x2, y2 = edge[:, 0], edge[:, 1], edge[:, 2], edge[:, 3]

    mask = np.zeros((ny, nx), dtype=bool)
    for row in range(ny):
        y = y0 + (row + 0.5) * step
        straddles = (y1 > y) != (y2 > y)
        if not straddles.any():
            continue
        at = np.sort((x2[straddles] - x1[straddles]) * (y - y1[straddles])
                     / (y2[straddles] - y1[straddles]) + x1[straddles])
        for pair in range(0, len(at) - 1, 2):
            first = int(math.ceil((at[pair] - x0) / step - 0.5))
            last = int(math.floor((at[pair + 1] - x0) / step - 0.5))
            if last < 0 or first >= nx or last < first:
                continue
            mask[row, max(first, 0):min(last, nx - 1) + 1] = True
    return mask


def label_grid(land: np.ndarray, zone_masks: Sequence[np.ndarray]) -> np.ndarray:
    """A zone index per land cell, -1 off the land, nothing left unlabelled.

    Land the ENTSO-E layer puts in no zone at all is taken by whichever zone
    reaches it first, growing one cell at a time. About one cell in 150 of the
    Norwegian and Swedish mainlands needs it, nearly all of it on the coast,
    which is where the two sources disagree.
    """
    labels = np.full(land.shape, -1, dtype=np.int16)
    for index, mask in enumerate(zone_masks):
        labels[land & mask & (labels < 0)] = index

    # Growth is one cell a step through land only, so the bound is the width
    # plus the height. It stops earlier as soon as a round changes nothing,
    # which is the real terminator; land unreachable from any zone -- a piece
    # the raster has severed from the rest -- stays unlabelled and shows up in
    # the coverage figure the tool prints.
    for _ in range(sum(land.shape) + 2):
        if not (land & (labels < 0)).any():
            return labels
        grown = labels.copy()
        for into, source in (
            (np.s_[1:, :], np.s_[:-1, :]),
            (np.s_[:-1, :], np.s_[1:, :]),
            (np.s_[:, 1:], np.s_[:, :-1]),
            (np.s_[:, :-1], np.s_[:, 1:]),
        ):
            take = (grown[into] < 0) & (labels[source] >= 0)
            grown[into] = np.where(take, labels[source], grown[into])
        grown[~land] = -1
        if np.array_equal(grown, labels):
            return labels
        labels = grown
    return labels


#: Which way to turn first where four cells meet diagonally and the boundary
#: could go either way. Turning as tightly as possible keeps the two rings
#: separate instead of joining them into a figure eight.
_RIGHT_OF = {(1, 0): (0, -1), (0, -1): (-1, 0), (-1, 0): (0, 1), (0, 1): (1, 0)}


def trace_labels(labels: np.ndarray, x0: float, y0: float, step: float) -> Dict[int, List]:
    """Each label back to polygons, by walking cell edges rather than contouring.

    Every boundary between two labels is emitted twice -- once for each side,
    reversed -- so the seam between two zones is shared vertex for vertex by
    construction, and dissolving a country's zones returns its coastline
    exactly. Contouring cannot promise that: it samples cell centres, so every
    region comes out inset by half a cell, it leaves polylines open where a
    region meets the edge of the array, and traced per label it shares nothing.
    """
    out: Dict[int, List] = {}
    height, width = labels.shape
    for value in sorted(set(int(v) for v in np.unique(labels)) - {-1}):
        mask = labels == value
        padded = np.zeros((height + 2, width + 2), dtype=bool)
        padded[1:-1, 1:-1] = mask

        outgoing: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
        rows, cols = np.nonzero(mask)
        for row, col in zip(rows.tolist(), cols.tolist()):
            if not padded[row, col + 1]:
                outgoing.setdefault((col, row), []).append((col + 1, row))
            if not padded[row + 1, col + 2]:
                outgoing.setdefault((col + 1, row), []).append((col + 1, row + 1))
            if not padded[row + 2, col + 1]:
                outgoing.setdefault((col + 1, row + 1), []).append((col, row + 1))
            if not padded[row + 1, col]:
                outgoing.setdefault((col, row + 1), []).append((col, row))

        rings: List[List] = []
        while outgoing:
            start = next(iter(outgoing))
            lattice = [start]
            here, heading = start, None
            while True:
                choices = outgoing.get(here)
                if not choices:
                    raise ValueError("trace_labels: a boundary dead-ends")
                if heading is not None and len(choices) > 1:
                    turn = _RIGHT_OF[heading]
                    preferred = (here[0] + turn[0], here[1] + turn[1])
                    step_to = preferred if preferred in choices else choices[-1]
                    choices.remove(step_to)
                else:
                    step_to = choices.pop()
                if not choices:
                    del outgoing[here]
                heading = (step_to[0] - here[0], step_to[1] - here[1])
                lattice.append(step_to)
                here = step_to
                if here == start:
                    break
            rings.append([[x0 + i * step, y0 + j * step] for i, j in lattice])
        out[value] = nest_rings(rings)
    return out


def split_polygon_by_zone(
    polygon, zone_polygons: Dict[str, List], step: float
) -> Dict[str, List]:
    """One land polygon cut into the zones that share it.

    The only part of this file resolved on a grid rather than exactly, and it
    runs for exactly two polygons in the whole dataset: the Norwegian and the
    Swedish mainland. Everything else -- every Danish island, Gotland, Oland,
    Lofoten -- keeps its Natural Earth outline untouched.
    """
    xs = [point[0] for ring in polygon for point in ring]
    ys = [point[1] for ring in polygon for point in ring]
    x0 = math.floor(min(xs) / step) * step - step
    y0 = math.floor(min(ys) / step) * step - step
    nx = int(math.ceil((max(xs) - x0) / step)) + 2
    ny = int(math.ceil((max(ys) - y0) / step)) + 2

    land = rasterise(polygon, x0, y0, step, nx, ny)
    names = sorted(zone_polygons)
    masks = []
    for name in names:
        mask = np.zeros((ny, nx), dtype=bool)
        for part in zone_polygons[name]:
            mask |= rasterise(part, x0, y0, step, nx, ny)
        masks.append(mask)

    traced = trace_labels(label_grid(land, masks), x0, y0, step)
    return {names[value]: polygons for value, polygons in traced.items()}


ATTRIBUTION = '''# Where `country_shapes.geojson` and `zone_shapes.geojson` come from

Generated by `tools/prepare_zone_geometry.py`. Do not edit them by hand -- rerun
the tool instead.

The two sources do two different jobs, and they are licensed differently, so
what follows is split the same way.

## Every border drawn: Natural Earth

Natural Earth, `ne_50m_admin_0_map_units`, from
<https://github.com/nvkelso/natural-earth-vector>.

Natural Earth is in the **public domain**; the credit is courtesy rather than
obligation. Both assets take all of their geometry from it.

Map units per model country: Austria `AUT`; Belgium `BFR`+`BWR`+`BCR`;
Switzerland `CHE`; Germany `DEU`+`LUX`; Denmark `DNK`; Estonia `EST`; Spain
`ESP`; Finland `FIN`; France `FXX`; Lithuania `LTU`; Latvia `LVA`; Netherlands
`NLD`; Norway `NOR`; Poland `POL`; Portugal `PRX`; Sweden `SWE`; United Kingdom
`ENG`+`SCT`+`WLS`. Great Britain excludes Northern Ireland, which PECD carries
separately as `UKNI` and the model does not build; it is drawn as context
instead, with Ireland.

## Which zone a piece of land belongs to: Mopo

EPRI Europe DAC, & Porras Cabrera, A. (2026). *Mopo: Pan-European Dataset for
Energy System Planning* (Version v0.5) [Dataset]. Zenodo.
<https://doi.org/10.5281/zenodo.21278830>

Licensed **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>).

Changes made, which the licence requires be stated:

- The ENTSO-E bidding-zone layer is no longer used as a source of outlines. It
  is used to decide which zone each Natural Earth land polygon belongs to, and
  its zone identifiers were translated to the model's own spellings using the
  `area codes` sheet of `src_files/data_files/transferdata_TYNDP2020.xlsx`.
- `country_shapes.geojson` contains **no geometry derived from this work**.
- In `zone_shapes.geojson`, the only geometry derived from it is the divider
  inside the Norwegian mainland and inside the Swedish mainland. **Those two
  boundaries were moved**: they were resampled onto a 0.02 degree grid and
  re-traced, which shifts them by up to half a cell. No other boundary in either
  file comes from this work.
- Land that no bidding zone contains -- Bornholm, and stretches of coast where
  the two sources disagree -- was assigned to the nearest zone of the same
  country rather than left out.
- Boundaries were then simplified with Douglas-Peucker, coordinates rounded to
  three decimal places, and rings below the minimum size dropped.

## The result

Both files are derivative works of the above and are distributed under the
repository's licence, **CC BY-NC-SA 4.0**, which CC BY 4.0 permits.
'''


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--natural-earth", "--gb", dest="natural_earth", type=Path,
                        default=DEFAULT_NATURAL_EARTH,
                        help="Natural Earth ne_50m_admin_0_map_units GeoJSON -- every "
                             "border drawn comes from this file")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="ENTSO-E bidding zone GeoJSON, which says only which zone "
                             "a piece of land belongs to")
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK,
                        help="workbook carrying the 'area codes' sheet")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                        help="where to write both assets")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help=f"simplification tolerance in degrees (default {DEFAULT_TOLERANCE})")
    parser.add_argument("--min-ring", type=float, default=DEFAULT_MIN_RING,
                        help=f"drop rings smaller than this in degrees (default {DEFAULT_MIN_RING})")
    parser.add_argument("--split-resolution", type=float, default=DEFAULT_SPLIT_STEP,
                        help="grid step in degrees for the two landmasses that span "
                             f"zones (default {DEFAULT_SPLIT_STEP})")
    return parser


def polygons_area(polygons) -> float:
    """Area in square degrees, holes subtracted. For comparing, not for reporting."""
    total = 0.0
    for polygon in polygons:
        if not polygon:
            continue
        total += abs(ring_area(polygon[0]))
        total -= sum(abs(ring_area(ring)) for ring in polygon[1:])
    return total


def area_features(table: Dict[str, List], role: str, tolerance: float,
                  min_ring: float) -> List[Dict]:
    """One GeoJSON feature per area, simplified."""
    features = []
    for key in sorted(table):
        simplified = simplify_polygons(table[key], tolerance, min_ring)
        if not simplified:
            continue
        features.append({
            "type": "Feature",
            "properties": {"zone": key, "role": role},
            "geometry": {"type": "MultiPolygon", "coordinates": simplified},
        })
    return features


def write_asset(path: Path, features: List[Dict], tolerance: float,
                min_ring: float) -> Tuple[int, int]:
    """``(model features, points)`` after writing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "type": "FeatureCollection",
        "properties": {
            "generated_by": "tools/prepare_zone_geometry.py",
            "attribution": "see ATTRIBUTION.md beside this file",
            "tolerance_degrees": tolerance,
            "min_ring_degrees": min_ring,
        },
        "features": features,
    }, separators=(",", ":")), encoding="utf-8")
    model = sum(1 for f in features if f["properties"]["role"] == "model")
    points = sum(len(ring)
                 for feature in features
                 for polygon in feature["geometry"]["coordinates"]
                 for ring in polygon)
    return model, points


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    for label, path, hint in (
        ("Natural Earth map units", args.natural_earth,
         "Pass --natural-earth with ne_50m_admin_0_map_units.geojson; see the module docstring."),
        ("ENTSO-E bidding zones", args.source,
         "Pass --source, or put the Mopo ENTSO-E extract in example_maps/."),
        ("Crosswalk workbook", args.crosswalk, ""),
    ):
        if not path.exists():
            print(f"{label} not found: {path}")
            if hint:
                print(hint)
            return 1

    try:
        crosswalk = read_crosswalk(args.crosswalk)
    except Exception as error:                    # noqa: BLE001 -- reported, not raised
        print(f"Could not read the crosswalk: {error}")
        return 1

    wanted = ([unit for units_ in COUNTRY_UNITS.values() for unit in units_]
              + [unit for units_ in NEIGHBOUR_UNITS.values() for unit in units_])
    units, dropped = load_map_units(args.natural_earth, wanted)

    source = json.loads(args.source.read_text(encoding="utf-8"))
    zone_parts: Dict[str, List] = {}
    for feature in source.get("features", []):
        name = str(feature["properties"].get("zoneName", "")).strip()
        target = crosswalk.get(name)
        if target:
            zone_parts.setdefault(target, []).extend(as_multipolygon(feature["geometry"]))

    absent = sorted({unit for wanted in COUNTRY_UNITS.values() for unit in wanted
                     if unit not in units}
                    | {unit for wanted in NEIGHBOUR_UNITS.values() for unit in wanted
                       if unit not in units})
    if absent:
        print(f"Natural Earth has no map unit {', '.join(absent)} -- "
              "is this ne_50m_admin_0_map_units?")
        return 1

    country_land = {country: units_for(units, wanted)
                    for country, wanted in COUNTRY_UNITS.items()}
    neighbour_land = {name: units_for(units, wanted)
                      for name, wanted in NEIGHBOUR_UNITS.items()}

    zone_land: Dict[str, List] = {}
    notes: List[str] = []
    for country in sorted(country_land):
        land = country_land[country]
        zones = sorted({zone for zone in crosswalk.values() if zone[:2] == country})
        if len(zones) == 1:
            zone_land[zones[0]] = land
            continue
        if not zones:
            notes.append(f"{country}: the crosswalk names no zone for it")
            continue

        # Great Britain is the case the other way round: the crosswalk names
        # UK00 and the ENTSO-E layer has no feature for it, which is fine while
        # the country has only one zone and would be reported if it did not.
        parts = {zone: zone_parts[zone] for zone in zones if zone in zone_parts}
        if len(parts) < len(zones):
            notes.append(f"{country}: the ENTSO-E layer has no shape for "
                         f"{', '.join(sorted(set(zones) - set(parts)))}, so its land "
                         "is shared out among the zones that do")
        if not parts:
            zone_land[zones[0]] = land
            continue
        placed, spanning, said = assign_polygons(land, parts)
        notes.extend(f"{country}: {line}" for line in said)
        for polygon in spanning:
            pieces = split_polygon_by_zone(polygon, parts, args.split_resolution)
            for zone, cut in pieces.items():
                placed.setdefault(zone, []).extend(cut)
            notes.append(f"{country}: a {len(polygon[0])}-point landmass spans "
                         f"{len(zones)} zones and was split on a "
                         f"{args.split_resolution} degree grid")
        for zone, pieces in placed.items():
            zone_land.setdefault(zone, []).extend(pieces)

    country_features = (area_features(country_land, "model", args.tolerance, args.min_ring)
                        + area_features(neighbour_land, "neighbour", args.tolerance,
                                        args.min_ring))
    zone_features = (area_features(zone_land, "model", args.tolerance, args.min_ring)
                     + area_features(neighbour_land, "neighbour", args.tolerance,
                                     args.min_ring))

    country_path = args.out_dir / COUNTRY_ASSET_NAME
    zone_path = args.out_dir / ZONE_ASSET_NAME
    countries, country_points = write_asset(country_path, country_features,
                                            args.tolerance, args.min_ring)
    zones, zone_points = write_asset(zone_path, zone_features, args.tolerance, args.min_ring)
    (args.out_dir / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")

    print(f"Wrote {country_path.relative_to(_REPO_ROOT)}")
    print(f"  {countries} countr(ies), {len(country_features) - countries} neighbour(s), "
          f"{country_points:,} point(s), {country_path.stat().st_size / 1024:.1f} KB")
    print(f"Wrote {zone_path.relative_to(_REPO_ROOT)}")
    print(f"  {zones} zone(s), {len(zone_features) - zones} neighbour(s), "
          f"{zone_points:,} point(s), {zone_path.stat().st_size / 1024:.1f} KB")

    if dropped:
        listed = ", ".join(f"{unit} x{count}" for unit, count in sorted(dropped.items()))
        print(f"  outside the map window, dropped: {listed}")
    for line in notes:
        print(f"  {line}")

    # A country's zones cover its land and no more. Exact for every country that
    # needed no splitting; off by up to half a grid cell where one did, because
    # there the coastline is the traced grid rather than the source's own line.
    worst, worst_country = 0.0, ""
    for country, land in country_land.items():
        zones_here = [zone for zone in zone_land if zone[:2] == country]
        if not zones_here:
            continue
        covered = sum(polygons_area(zone_land[zone]) for zone in zones_here)
        whole = polygons_area(land)
        drift = abs(covered - whole) / whole if whole else 0.0
        if drift > worst:
            worst, worst_country = drift, country
    print(f"  zones cover their country's land to within {worst * 100:.3f}% "
          f"(worst: {worst_country or 'none'})")

    unplaced = sorted(set(crosswalk.values()) - set(zone_land))
    if unplaced:
        print(f"  no geometry for {len(unplaced)}: {', '.join(unplaced)}")
    else:
        print("  every zone the crosswalk names has a shape")
    return 0


if __name__ == "__main__":
    sys.exit(main())
