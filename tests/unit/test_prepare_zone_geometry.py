"""tools/prepare_zone_geometry.py -- the parts that fail silently.

Like the summary tool, this is not here because tools belong in the suite. It
is here because three of its mistakes would produce a map that renders without
complaint and is wrong:

- a simplified ring that no longer closes draws a wedge to the first vertex,
  which at these tolerances looks like a coastline;
- a zone the crosswalk names but the source has no shape for would vanish from
  the map rather than be reported, which is how Great Britain went missing from
  the first asset;
- the crosswalk sheet is a block floating in an empty grid, so reading it by
  position works until someone inserts a row.
"""

import json
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import prepare_zone_geometry as prep  # noqa: E402


def square(size: float = 1.0, jitter: float = 0.0):
    """A closed ring with collinear points that simplification should remove."""
    return [
        [0.0, 0.0], [size / 2, jitter], [size, 0.0],
        [size, size], [0.0, size], [0.0, 0.0],
    ]


class TestSimplificationKeepsARingDrawable:
    def test_a_ring_stays_closed(self):
        out = prep.simplify_polygons([[square()]], tolerance=0.05, min_ring=0.0)
        ring = out[0][0]
        assert ring[0] == ring[-1]

    def test_a_ring_keeps_enough_points_to_have_an_area(self):
        out = prep.simplify_polygons([[square()]], tolerance=0.05, min_ring=0.0)
        assert len(out[0][0]) >= 4

    def test_a_collinear_point_is_dropped(self):
        out = prep.simplify_polygons([[square(jitter=0.0)]], tolerance=0.05, min_ring=0.0)
        assert len(out[0][0]) == 5      # four corners, first repeated

    def test_a_point_that_moves_the_line_is_kept(self):
        out = prep.simplify_polygons([[square(jitter=0.5)]], tolerance=0.05, min_ring=0.0)
        assert len(out[0][0]) == 6

    def test_an_islet_below_the_minimum_is_dropped(self):
        out = prep.simplify_polygons([[square(size=0.01)]], tolerance=0.001, min_ring=0.1)
        assert out == []

    def test_a_degenerate_ring_is_dropped_rather_than_drawn(self):
        out = prep.simplify_polygons([[[[0.0, 0.0], [1.0, 1.0], [0.0, 0.0]]]],
                                     tolerance=0.05, min_ring=0.0)
        assert out == []


class TestDouglasPeuckerHandlesTheRealShapes:
    def test_a_long_ring_does_not_exhaust_the_stack(self):
        """Norwegian rings run to thousands of points; the recursive form broke."""
        points = [(i * 0.001, (i % 7) * 0.0001) for i in range(20000)]
        out = prep.douglas_peucker(points, tolerance=0.01)
        assert out[0] == points[0] and out[-1] == points[-1]

    def test_the_endpoints_are_never_dropped(self):
        points = [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]
        assert prep.douglas_peucker(points, tolerance=10.0) == [(0.0, 0.0), (2.0, 0.0)]


class TestGreatBritainExcludesNorthernIreland:
    """UK00 is Great Britain; PECD carries Northern Ireland separately as UKNI."""

    def _natural_earth(self, tmp_path: Path) -> Path:
        features = [
            {"properties": {"SU_A3": unit, "NAME": unit},
             "geometry": {"type": "Polygon", "coordinates": [square()]}}
            for unit in ("ENG", "SCT", "WLS", "NIR", "IRL")
        ]
        path = tmp_path / "ne.geojson"
        path.write_text(json.dumps({"features": features}), encoding="utf-8")
        return path

    def test_three_units_make_great_britain(self, tmp_path):
        gb, _ = prep.load_natural_earth(self._natural_earth(tmp_path))
        assert len(gb) == 3

    def test_northern_ireland_is_context_not_uk00(self, tmp_path):
        gb, extra = prep.load_natural_earth(self._natural_earth(tmp_path))
        assert len(gb) == 3
        assert set(extra) == {"NIR", "IRL"}


class TestTheAssetCoversEveryZoneItNames:
    """A zone with no shape is reported, not dropped -- that is the GB lesson."""

    def test_the_shipped_asset_places_every_model_zone(self):
        asset = Path(prep.DEFAULT_OUT)
        if not asset.exists():
            pytest.skip("asset not built; run tools/prepare_zone_geometry.py")
        data = json.loads(asset.read_text(encoding="utf-8"))
        placed = {f["properties"]["zone"] for f in data["features"]
                  if f["properties"].get("role") == "model"}
        expected = {
            "AT00", "BE00", "CH00", "DE00", "DKE1", "DKW1", "EE00", "ES00",
            "FI00", "FR00", "LT00", "LV00", "NL00", "NOM1", "NON1", "NOS0",
            "PL00", "SE01", "SE02", "SE03", "SE04", "UK00",
        }
        assert expected <= placed, f"no geometry for {sorted(expected - placed)}"

    def test_every_ring_in_the_shipped_asset_is_closed(self):
        asset = Path(prep.DEFAULT_OUT)
        if not asset.exists():
            pytest.skip("asset not built; run tools/prepare_zone_geometry.py")
        data = json.loads(asset.read_text(encoding="utf-8"))
        for feature in data["features"]:
            for polygon in feature["geometry"]["coordinates"]:
                for ring in polygon:
                    assert ring[0] == ring[-1], feature["properties"]["zone"]
                    assert len(ring) >= 4, feature["properties"]["zone"]
