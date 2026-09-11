"""tools/prepare_zone_geometry.py -- the parts that fail silently.

Like the summary tool, this is not here because tools belong in the suite. It
is here because these mistakes produce a map that renders without complaint and
is wrong:

- a simplified ring that no longer closes draws a wedge to the first vertex,
  which at these tolerances looks like a coastline;
- a zone the crosswalk names but the sources have no shape for would vanish from
  the map rather than be reported, which is how Great Britain went missing from
  the first asset;
- the crosswalk sheet is a block floating in an empty grid, so reading it by
  position works until someone inserts a row;
- land assigned to the wrong zone -- Bornholm's nearest zone of any country is
  Swedish, 35 km away against DKE1's 136 km, so a nearest-zone rule that is not
  held to one country hands it to Sweden and says nothing;
- land dropped rather than assigned. The zones of a country cover its whole
  coastline or the map has a hole in it, and the only thing that proves the
  split shared its seams is that the parts still add up to the whole;
- a polygon that spans a zone divider between two of its vertices. Natural Earth
  puts 20 to 50 km between vertices, so testing containment on vertices alone
  skips the very branch that would have split it.
"""

import json
import math
import sys
from pathlib import Path

import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import numpy as np  # noqa: E402
import prepare_zone_geometry as prep  # noqa: E402


def square(size: float = 1.0, jitter: float = 0.0):
    """A closed ring with collinear points that simplification should remove."""
    return [
        [0.0, 0.0], [size / 2, jitter], [size, 0.0],
        [size, size], [0.0, size], [0.0, 0.0],
    ]


def box(west: float, south: float, east: float, north: float):
    """A closed rectangular ring, counter-clockwise."""
    return [[west, south], [east, south], [east, north], [west, north], [west, south]]


def ring_km2(ring) -> float:
    """Spherical polygon area, so a country split across latitudes still adds up."""
    radius = 6371.0087714
    total = 0.0
    for (x1, y1), (x2, y2) in zip(ring[:-1], ring[1:]):
        total += math.radians(x2 - x1) * (
            2 + math.sin(math.radians(y1)) + math.sin(math.radians(y2))
        )
    return abs(total * radius * radius / 2.0)


def polygons_km2(polygons) -> float:
    return sum(ring_km2(p[0]) - sum(ring_km2(r) for r in p[1:]) for p in polygons if p)


def shipped(name: str):
    path = prep.DEFAULT_OUT_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not built; run tools/prepare_zone_geometry.py")
    return json.loads(path.read_text(encoding="utf-8"))


def model_areas(asset) -> dict:
    return {f["properties"]["zone"]: f["geometry"]["coordinates"]
            for f in asset["features"] if f["properties"].get("role") == "model"}


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


class TestDissolveIsExactWhereTheSourceSharesItsVertices:
    """Four model countries are more than one Natural Earth map unit.

    Drawn undissolved they show a white hairline along a border the model does
    not have -- through Germany where Luxembourg is, and twice through Belgium.
    """

    def test_two_squares_sharing_an_edge_become_one(self):
        out = prep.dissolve([box(0, 0, 1, 1), box(1, 0, 2, 1)])
        assert len(out) == 1
        assert abs(abs(prep.ring_area(out[0])) - 2.0) < 1e-12

    def test_an_enclave_held_as_a_hole_cancels_with_its_own_outline(self):
        """Brussels is a hole in Flanders and a unit of its own; both must go."""
        outer = box(0, 0, 3, 3)
        hole = box(1, 1, 2, 2)[::-1]
        out = prep.dissolve_polygons([[outer, hole], [box(1, 1, 2, 2)]])
        assert len(out) == 1 and len(out[0]) == 1
        assert abs(polygons_km2(out) - polygons_km2([[outer]])) < 1e-6

    def test_a_true_hole_survives(self):
        """Cancelling must not eat an enclave nothing else fills, such as Vatican City."""
        out = prep.dissolve_polygons([[box(0, 0, 3, 3), box(1, 1, 2, 2)[::-1]]])
        assert len(out) == 1 and len(out[0]) == 2

    def test_rings_that_do_not_share_a_border_are_kept_apart(self):
        out = prep.dissolve([box(0, 0, 1, 1), box(5, 5, 6, 6)])
        assert len(out) == 2

    def test_a_border_drawn_with_different_vertices_does_not_merge(self):
        """Why the geometry is not taken from the ENTSO-E layer.

        Six of the 4,636 directed edges along the four Swedish zones there have
        an exact reverse twin, against 274 of 274 on the same border in Natural
        Earth. Nothing cancels, so the two stay two -- the honest answer, and a
        useless one for drawing a country.
        """
        neighbour = [[1.0, 0.1], [2.0, 0.1], [2.0, 0.9], [1.0, 0.9], [1.0, 0.1]]
        assert len(prep.dissolve([box(0, 0, 1, 1), neighbour])) == 2


class TestRingsCrossCatchesWhatVerticesMiss:
    def test_a_bulge_between_two_vertices_is_seen(self):
        coarse = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0], [0.0, 0.0]]
        divider = box(4.0, -5.0, 6.0, 5.0)
        assert prep.rings_cross(coarse, divider)

    def test_a_ring_wholly_inside_another_does_not_cross_it(self):
        assert not prep.rings_cross(box(1, 1, 2, 2), box(0, 0, 10, 10))

    def test_rings_far_apart_do_not_cross(self):
        assert not prep.rings_cross(box(0, 0, 1, 1), box(50, 50, 51, 51))


class TestTheScanlineRasteriserFillsWhatPointInPolygonWouldSay:
    """The rasteriser is a shortcut for speed; it has to agree with the slow way."""

    def _awkward(self):
        outer = [[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [1.0, 1.0], [1.0, 2.0],
                 [3.0, 2.0], [3.0, 3.0], [0.0, 3.0], [0.0, 0.0]]
        hole = [[0.2, 0.2], [0.2, 0.8], [0.8, 0.8], [0.8, 0.2], [0.2, 0.2]]
        return [outer, hole]

    def test_it_agrees_with_point_in_polygon(self):
        step, nx, ny = 0.05, 64, 64
        mask = prep.rasterise(self._awkward(), -0.1, -0.1, step, nx, ny)
        xs = -0.1 + (np.arange(nx) + 0.5) * step
        ys = -0.1 + (np.arange(ny) + 0.5) * step
        grid = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
        naive = prep.point_in_polygons(grid, [self._awkward()]).reshape(ny, nx)
        assert np.array_equal(mask, naive)

    def test_a_hole_is_not_filled(self):
        mask = prep.rasterise(self._awkward(), -0.1, -0.1, 0.05, 64, 64)
        inside_hole = prep.rasterise([self._awkward()[1]], -0.1, -0.1, 0.05, 64, 64)
        assert not (mask & inside_hole).any()


class TestTracingSharesTheSeamBetweenTwoZones:
    """Two zones traced from one grid must agree about the line between them.

    Traced independently -- which is what contouring each label would do -- they
    do not, and the map gets a sliver of background between two neighbours.
    """

    def _two_zones(self):
        labels = np.zeros((20, 20), dtype=np.int16)
        labels[:, 10:] = 1
        return prep.trace_labels(labels, 0.0, 0.0, 0.1)

    def test_both_zones_are_traced(self):
        assert set(self._two_zones()) == {0, 1}

    def test_the_shared_edge_has_the_same_vertices_in_both(self):
        traced = self._two_zones()
        left = {tuple(p) for p in traced[0][0][0] if abs(p[0] - 1.0) < 1e-9}
        right = {tuple(p) for p in traced[1][0][0] if abs(p[0] - 1.0) < 1e-9}
        assert left and left == right

    def test_the_two_pieces_add_up_to_the_whole(self):
        traced = self._two_zones()
        assert abs(sum(abs(prep.ring_area(v[0][0])) for v in traced.values()) - 4.0) < 1e-9

    def test_a_hole_in_a_region_stays_a_hole(self):
        labels = np.zeros((20, 20), dtype=np.int16)
        labels[8:12, 8:12] = -1
        traced = prep.trace_labels(labels, 0.0, 0.0, 0.1)
        assert len(traced[0][0]) == 2


class TestLabellingLeavesNoLandBehind:
    def test_land_no_zone_claims_is_grown_into_rather_than_dropped(self):
        """One corner cell of a 10x10 country has to reach the far corner.

        Eighteen steps away, so a bound of max(width, height) leaves a third of
        the country unlabelled and silently missing from the map.
        """
        land = np.ones((10, 10), dtype=bool)
        zone = np.zeros((10, 10), dtype=bool)
        zone[0, 0] = True
        labels = prep.label_grid(land, [zone])
        assert (labels[land] >= 0).all()

    def test_the_nearer_zone_takes_the_unclaimed_ground(self):
        land = np.ones((3, 9), dtype=bool)
        left = np.zeros((3, 9), dtype=bool); left[:, 0] = True
        right = np.zeros((3, 9), dtype=bool); right[:, 8] = True
        labels = prep.label_grid(land, [left, right])
        assert labels[1, 1] == 0 and labels[1, 7] == 1

    def test_a_cell_off_the_land_is_never_labelled(self):
        land = np.zeros((6, 6), dtype=bool)
        land[2:4, 2:4] = True
        zone = np.ones((6, 6), dtype=bool)
        labels = prep.label_grid(land, [zone])
        assert (labels[~land] == -1).all()


class TestTheAssetsCoverEveryAreaTheyName:
    """A zone with no shape is reported, not dropped -- that is the GB lesson."""

    ZONES = {
        "AT00", "BE00", "CH00", "DE00", "DKE1", "DKW1", "EE00", "ES00",
        "FI00", "FR00", "LT00", "LV00", "NL00", "NOM1", "NON1", "NOS0",
        "PL00", "PT00", "SE01", "SE02", "SE03", "SE04", "UK00",
    }

    def test_the_zone_asset_places_every_model_zone(self):
        placed = set(model_areas(shipped(prep.ZONE_ASSET_NAME)))
        assert self.ZONES <= placed, f"no geometry for {sorted(self.ZONES - placed)}"

    def test_the_country_asset_places_every_model_country(self):
        placed = set(model_areas(shipped(prep.COUNTRY_ASSET_NAME)))
        assert set(prep.COUNTRY_UNITS) <= placed

    def test_the_two_assets_carry_the_same_countries(self):
        zones = {zone[:2] for zone in model_areas(shipped(prep.ZONE_ASSET_NAME))}
        assert zones == set(model_areas(shipped(prep.COUNTRY_ASSET_NAME)))

    def test_the_two_assets_carry_the_same_neighbour_ring(self):
        def ring(name):
            return {f["properties"]["zone"] for f in shipped(name)["features"]
                    if f["properties"].get("role") == "neighbour"}
        assert ring(prep.ZONE_ASSET_NAME) == ring(prep.COUNTRY_ASSET_NAME)

    @pytest.mark.parametrize("name", ["zone_shapes.geojson", "country_shapes.geojson"])
    def test_every_ring_is_closed(self, name):
        for feature in shipped(name)["features"]:
            for polygon in feature["geometry"]["coordinates"]:
                for ring in polygon:
                    assert ring[0] == ring[-1], feature["properties"]["zone"]
                    assert len(ring) >= 4, feature["properties"]["zone"]


class TestACountryIsCoveredByItsZones:
    """The invariant the split exists to satisfy.

    If the zones of a country do not add up to it, either the split lost land or
    its seams were guessed rather than shared -- and both draw a map with a gap
    in it that no reader would think to question.
    """

    def test_every_country_is_covered_by_its_zones(self):
        zones = model_areas(shipped(prep.ZONE_ASSET_NAME))
        countries = model_areas(shipped(prep.COUNTRY_ASSET_NAME))
        for country, geometry in countries.items():
            whole = polygons_km2(geometry)
            covered = sum(polygons_km2(g) for z, g in zones.items() if z[:2] == country)
            assert abs(covered - whole) < 0.01 * whole, (
                f"{country}: zones cover {covered:,.0f} km2 of {whole:,.0f} km2"
            )

    def test_a_country_that_needed_no_split_matches_exactly(self):
        """Only Norway and Sweden are cut on a grid; the other fifteen are not."""
        zones = model_areas(shipped(prep.ZONE_ASSET_NAME))
        countries = model_areas(shipped(prep.COUNTRY_ASSET_NAME))
        for country in ("FI", "DE", "PL", "FR", "ES", "UK", "DK"):
            covered = sum(polygons_km2(g) for z, g in zones.items() if z[:2] == country)
            assert covered == pytest.approx(polygons_km2(countries[country]), rel=1e-9)


class TestNorwayIsLandRatherThanSea:
    """The ENTSO-E bidding-zone outline of Norway is 31.4% sea.

    464,167 km2 against 320,834 km2 of land, a median 13 km offshore and up to
    88 km: territorial waters with the fjords and the skerry belt filled in.
    Every other country in that layer is land-accurate, so this number is the
    one thing that says the geometry still comes from the national source.
    """

    def test_norway_is_about_its_land_area(self):
        countries = model_areas(shipped(prep.COUNTRY_ASSET_NAME))
        assert 290_000 < polygons_km2(countries["NO"]) < 340_000

    def test_norway_does_not_reach_as_far_out_to_sea(self):
        countries = model_areas(shipped(prep.COUNTRY_ASSET_NAME))
        east = max(p[0] for poly in countries["NO"] for ring in poly for p in ring)
        assert east < 31.5, "the ENTSO-E envelope reached 31.76 E; the coastline does not"


class TestFarFlungTerritoryStaysOffTheMap:
    """The maps autoscale, so one stray island silently shrinks Europe.

    Jan Mayen is a Norwegian map unit at 9 W, the Azores are Portuguese at 31 W,
    and the Canary Islands are inside the Spanish unit itself -- that last one
    cannot be excluded by unit at all, which is what the window rule is for.
    """

    def test_the_asset_stays_within_the_window(self):
        west, south, east, north = prep.KEEP_BBOX
        for name in (prep.ZONE_ASSET_NAME, prep.COUNTRY_ASSET_NAME):
            for feature in shipped(name)["features"]:
                for polygon in feature["geometry"]["coordinates"]:
                    for ring in polygon:
                        assert min(p[0] for p in ring) > west, feature["properties"]["zone"]
                        assert min(p[1] for p in ring) > south, feature["properties"]["zone"]

    def test_a_ring_outside_the_window_is_seen(self):
        assert prep.ring_outside_window(box(-18.2, 27.6, -13.4, 29.5))

    def test_a_ring_inside_the_window_is_not(self):
        assert not prep.ring_outside_window(box(4.8, 58.0, 31.0, 71.1))

    def test_norway_is_the_mainland_and_not_jan_mayen(self):
        assert prep.COUNTRY_UNITS["NO"] == ("NOR",)

    def test_great_britain_excludes_northern_ireland(self):
        """PECD carries Northern Ireland separately as UKNI; the model has no such zone."""
        assert "NIR" not in prep.COUNTRY_UNITS["UK"]
        assert prep.NEIGHBOUR_UNITS["NIR"] == ("NIR",)


class TestLandNoZoneClaimsGoesToItsOwnCountry:
    """Bornholm is inside no ENTSO-E zone at all, and Sweden is much nearer."""

    def test_nearest_zone_picks_the_closer_of_the_candidates(self):
        near = {"A": [[box(0.0, 0.0, 1.0, 1.0)]], "B": [[box(8.0, 0.0, 9.0, 1.0)]]}
        assert prep.nearest_zone([box(2.0, 0.0, 3.0, 1.0)], near) == "A"

    def test_bornholm_is_danish(self):
        zones = model_areas(shipped(prep.ZONE_ASSET_NAME))

        def holds_bornholm(geometry):
            return any(14.5 < min(p[0] for p in poly[0]) < 15.3
                       and 54.9 < min(p[1] for p in poly[0]) < 55.3
                       for poly in geometry)

        assert holds_bornholm(zones["DKE1"])
        assert not holds_bornholm(zones["SE04"])
