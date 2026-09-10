"""Build the map asset the input-data summary draws its regions on.

``tools/input_data_summary.py`` draws two maps. It needs a boundary for every
bidding zone the model carries, small enough to keep in the repository and
readable with nothing but the standard library. This makes that file, once, by
hand. No build runs it.

The zone identifiers are the interesting part. The ENTSO-E source spells zones
``DE_LU``, ``DK_1``, ``NO_1``, ``SE_1``; the model spells the same places
``DE00``, ``DKW1``, ``NOS0``, ``SE01``. The translation is not invented here --
it is read from the ``area codes`` sheet of
``src_files/data_files/transferdata_TYNDP2020.xlsx``, which is tracked and names
all of them. A crosswalk hardcoded in this file would be one more thing to keep
in step with the data.

Every zone the crosswalk names is written, not only the ones a scenario happens
to build, so adding Italy or Portugal to a config later needs no new asset. The
countries around them are written too, as context to draw in grey, and the
report tool can switch them off.

What it cannot see
------------------
Whether the boundaries are current. They are a snapshot of one dataset; a zone
that splits after it was published stays whole here, and nothing in this file
would notice.

Whether a zone is where the model thinks it is. The crosswalk is a name-to-name
table, so an error in it produces a map that is wrong and confident.

It does not dissolve zones into countries. Adjacent zones in this source do not
share vertices -- of the 4,636 directed edges along the four Swedish zones, six
have an exact reverse twin -- so a union without a geometry library would have
to guess where the seam is. The report tool draws zone polygons at both levels
instead and colours them by area, which needs no union and shows the zone
structure the model actually solves.

Usage:
    python tools/prepare_zone_geometry.py [--source FILE] [--gb FILE]
        [--crosswalk FILE] [--out FILE] [--tolerance DEG] [--min-ring DEG]

The Great Britain outline is a second source. Pass --gb with the Natural Earth
``ne_50m_admin_0_map_units`` GeoJSON, from
https://github.com/nvkelso/natural-earth-vector (the geojson/ folder), and UK00
gets a shape. Without it the tool still writes a file, and reports UK00 as the
one zone it could not place.

Exit 0 when a file was written, 1 when a source could not be read.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_SOURCE = _REPO_ROOT / "example_maps" / "entsoe_shp" / "entsoe_bid.geojson"
DEFAULT_CROSSWALK = _REPO_ROOT / "src_files" / "data_files" / "transferdata_TYNDP2020.xlsx"
DEFAULT_OUT = _REPO_ROOT / "tools" / "data" / "zone_shapes.geojson"

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

#: Natural Earth splits the United Kingdom into four map units. UK00 is Great
#: Britain: PECD carries Northern Ireland separately as UKNI, so including it
#: would draw a zone the model does not have.
GB_UNITS = frozenset({"ENG", "SCT", "WLS"})
GB_UNIT_NAMES = frozenset({"England", "Scotland", "Wales"})

#: Context countries the ENTSO-E layer does not carry, taken from the same
#: Natural Earth file as Great Britain. Ireland and Northern Ireland are the
#: only conspicuous hole in the ring -- without them Great Britain sits in an
#: empty sea and reads as though the model stopped at the coast.
EXTRA_NEIGHBOUR_UNITS = frozenset({"IRL", "NIR"})


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


def load_natural_earth(path: Path) -> Tuple[List, Dict[str, List]]:
    """``(Great Britain, {extra neighbour: parts})`` from one Natural Earth read.

    England, Scotland and Wales stay three polygons rather than one merged
    shape, for the same reason the zones do: without a geometry library there is
    no honest union. They share a fill, so the seams do not show.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    gb: List = []
    extra: Dict[str, List] = {}
    for feature in data.get("features", []):
        properties = feature.get("properties", {})
        unit = str(properties.get("SU_A3", "")).strip()
        name = str(properties.get("NAME", "")).strip()
        if unit in GB_UNITS or name in GB_UNIT_NAMES:
            gb.extend(as_multipolygon(feature["geometry"]))
        elif unit in EXTRA_NEIGHBOUR_UNITS:
            extra.setdefault(unit, []).extend(as_multipolygon(feature["geometry"]))
    return gb, extra


ATTRIBUTION = '''# Where `zone_shapes.geojson` comes from

Generated by `tools/prepare_zone_geometry.py`. Do not edit it by hand -- rerun
the tool instead.

## Bidding zone boundaries

EPRI Europe DAC, & Porras Cabrera, A. (2026). *Mopo: Pan-European Dataset for
Energy System Planning* (Version v0.5) [Dataset]. Zenodo.
<https://doi.org/10.5281/zenodo.21278830>

Licensed **CC BY 4.0** (<https://creativecommons.org/licenses/by/4.0/>).

Changes made, which the licence requires be stated: the ENTSO-E bidding-zone
layer was subset to the zones this model carries plus a ring of neighbouring
countries; zone identifiers were translated to the model's own spellings using
the `area codes` sheet of `src_files/data_files/transferdata_TYNDP2020.xlsx`;
boundaries were simplified with Douglas-Peucker and coordinates rounded to three
decimal places; rings below the minimum size were dropped. No boundary was moved
deliberately.

## Great Britain

Natural Earth, `ne_50m_admin_0_map_units`, from
<https://github.com/nvkelso/natural-earth-vector>.

Natural Earth is in the **public domain**; the credit above is courtesy rather
than obligation. Great Britain is the England, Scotland and Wales map units --
Northern Ireland is excluded because PECD carries it separately as `UKNI` and
this model does not build it.

## The result

This file is a derivative of the above and is distributed under the
repository's licence, **CC BY-NC-SA 4.0**, which CC BY 4.0 permits.
'''


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE,
                        help="ENTSO-E bidding zone GeoJSON (default: the example_maps clone)")
    parser.add_argument("--gb", type=Path, default=None,
                        help="Natural Earth ne_50m_admin_0_map_units GeoJSON, for UK00")
    parser.add_argument("--crosswalk", type=Path, default=DEFAULT_CROSSWALK,
                        help="workbook carrying the 'area codes' sheet")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="where to write the asset")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE,
                        help=f"simplification tolerance in degrees (default {DEFAULT_TOLERANCE})")
    parser.add_argument("--min-ring", type=float, default=DEFAULT_MIN_RING,
                        help=f"drop rings smaller than this in degrees (default {DEFAULT_MIN_RING})")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    if not args.source.exists():
        print(f"Source not found: {args.source}")
        print("Pass --source, or put the Mopo ENTSO-E extract in example_maps/.")
        return 1
    if not args.crosswalk.exists():
        print(f"Crosswalk workbook not found: {args.crosswalk}")
        return 1

    try:
        crosswalk = read_crosswalk(args.crosswalk)
    except Exception as error:                    # noqa: BLE001 -- reported, not raised
        print(f"Could not read the crosswalk: {error}")
        return 1

    source = json.loads(args.source.read_text(encoding="utf-8"))

    # One model zone can be several source zones -- NOS0 is NO_1, NO_2 and NO_5 --
    # so parts accumulate per model zone instead of replacing each other.
    model_parts: Dict[str, List] = {}
    neighbour_parts: Dict[str, List] = {}
    for feature in source.get("features", []):
        name = str(feature["properties"].get("zoneName", "")).strip()
        if not name:
            continue
        target = crosswalk.get(name)
        bucket = model_parts if target else neighbour_parts
        bucket.setdefault(target or name, []).extend(as_multipolygon(feature["geometry"]))

    if args.gb:
        if not args.gb.exists():
            print(f"Great Britain source not found: {args.gb}")
            return 1
        parts, extra = load_natural_earth(args.gb)
        if not parts:
            print(f"No England/Scotland/Wales map units in {args.gb.name} -- "
                  "is it ne_50m_admin_0_map_units?")
            return 1
        model_parts[crosswalk.get("GB", "UK00")] = parts
        for unit, polygons in extra.items():
            neighbour_parts.setdefault(unit, []).extend(polygons)

    unplaced = sorted(set(crosswalk.values()) - set(model_parts))

    features = []
    for role, bucket in (("model", model_parts), ("neighbour", neighbour_parts)):
        for key in sorted(bucket):
            simplified = simplify_polygons(bucket[key], args.tolerance, args.min_ring)
            if not simplified:
                continue
            properties = {"zone": key, "role": role}
            if role == "model":
                properties["country"] = key[:2]
            features.append({
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "MultiPolygon", "coordinates": simplified},
            })

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "type": "FeatureCollection",
        "properties": {
            "generated_by": "tools/prepare_zone_geometry.py",
            "attribution": "see ATTRIBUTION.md beside this file",
            "tolerance_degrees": args.tolerance,
            "min_ring_degrees": args.min_ring,
        },
        "features": features,
    }
    args.out.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    (args.out.parent / "ATTRIBUTION.md").write_text(ATTRIBUTION, encoding="utf-8")

    model_count = sum(1 for f in features if f["properties"]["role"] == "model")
    points = sum(len(ring)
                 for feature in features
                 for polygon in feature["geometry"]["coordinates"]
                 for ring in polygon)

    print(f"Wrote {args.out.relative_to(_REPO_ROOT)}")
    print(f"  {model_count} model zone(s), {len(features) - model_count} neighbour(s), "
          f"{points:,} point(s), {args.out.stat().st_size / 1024:.1f} KB")
    if unplaced:
        print(f"  no geometry for {len(unplaced)}: {', '.join(unplaced)}")
        if "UK00" in unplaced:
            print("  UK00 needs --gb with the Natural Earth ne_50m_admin_0_map_units "
                  "GeoJSON; see the module docstring.")
    else:
        print("  every zone the crosswalk names has a shape")
    return 0


if __name__ == "__main__":
    sys.exit(main())
