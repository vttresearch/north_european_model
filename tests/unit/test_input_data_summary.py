"""tools/input_data_summary.py -- the arithmetic a reader could never check.

Tools are not normally in this suite (see tests/README.md): they print a report
a person reads, and a wrong number there is usually visible. Three of this
tool's conventions are not, and each of them was wrong in at least one draft of
the design:

- a transfer corridor is written twice, once per direction, and 15 of the 44 in
  the shipped scenarios differ between the two. Summing them double-counts;
  taking one makes the answer depend on which row the builder wrote first. Both
  produce a plausible number.
- a country's capacity factor has to be weighted by each zone's own installed
  capacity. An unweighted mean lets a 200 MW zone move the answer as much as a
  12 GW one, and still looks like a capacity factor.
- ``isActive`` is 1 on every row of every shipped scenario, so an unfiltered sum
  is right today and silently wrong the first time a unit is retired.

The reader-visible parts -- which figures exist, what the prose says -- are not
tested here. They are checked by running the tool, which is what the tool is
for.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import input_data_summary as summary  # noqa: E402


def make_workbook(**overrides) -> summary.Workbook:
    """A Workbook with empty sheets, for a test to fill in only what it uses."""
    empty = pd.DataFrame()
    fields = dict(
        path=Path("inputData.xlsx"), scenario="test", year="2030",
        p_gnu_io=empty, p_unit=empty, p_gn=empty, p_gnn=empty, boundary=empty,
        flow_unit=empty, unit_unittype=empty, n_emission=empty,
        emission_price=empty, grids=[], nodes=[], unittypes=[],
    )
    fields.update(overrides)
    return summary.Workbook(**fields)


class TestTransferIsTheMeanOfItsTwoDirections:
    """A corridor written 2800 one way and 4300 the other is not 7100, nor 2800."""

    @pytest.fixture
    def asymmetric(self):
        return make_workbook(p_gnn=pd.DataFrame({
            "grid": ["elec", "elec"],
            "from_node": ["BE00_elec", "FR00_elec"],
            "to_node": ["FR00_elec", "BE00_elec"],
            "transferCap": [2800.0, 4300.0],
        }))

    def test_each_side_reports_the_mean(self, asymmetric):
        per_area, _, _ = summary.transfer_by_area(asymmetric, zones=False)
        assert per_area.loc["BE", "cross_border_MW"] == pytest.approx(3550.0)
        assert per_area.loc["FR", "cross_border_MW"] == pytest.approx(3550.0)

    def test_the_asymmetry_is_counted_and_not_hidden(self, asymmetric):
        _, _, facts = summary.transfer_by_area(asymmetric, zones=False)
        assert facts["pairs"] == 1
        assert facts["asymmetric"] == 1

    def test_a_symmetric_corridor_is_not_counted_as_asymmetric(self):
        workbook = make_workbook(p_gnn=pd.DataFrame({
            "grid": ["elec", "elec"],
            "from_node": ["BE00_elec", "FR00_elec"],
            "to_node": ["FR00_elec", "BE00_elec"],
            "transferCap": [3000.0, 3000.0],
        }))
        _, _, facts = summary.transfer_by_area(workbook, zones=False)
        assert facts["asymmetric"] == 0

    def test_a_link_inside_one_country_is_not_cross_border(self):
        """SE01-SE02 must not inflate Sweden's border capacity."""
        workbook = make_workbook(p_gnn=pd.DataFrame({
            "grid": ["elec", "elec"],
            "from_node": ["SE01_elec", "SE02_elec"],
            "to_node": ["SE02_elec", "SE01_elec"],
            "transferCap": [6000.0, 6000.0],
        }))
        per_area, _, _ = summary.transfer_by_area(workbook, zones=False)
        assert per_area.loc["SE", "cross_border_MW"] == 0.0
        # 6000 one way and 6000 the other is one 6000 MW corridor, and both of
        # its ends are Sweden -- so it is Sweden's once, not twice.
        assert per_area.loc["SE", "inter_zonal_MW"] == pytest.approx(6000.0)


class TestCapacityFactorIsWeightedByCapacity:
    """A country's capacity factor is its fleet's, not the average of its zones'."""

    def test_the_big_zone_dominates(self):
        # 10 GW at 0.40 and 0.2 GW at 0.10: weighted 0.394, unweighted 0.25.
        workbook = make_workbook(
            p_gnu_io=pd.DataFrame({
                "grid": ["elec", "elec"],
                "node": ["SE02_elec", "SE04_elec"],
                "unit": ["SE02_windOnshore", "SE04_windOnshore"],
                "input_output": ["output", "output"],
                "capacity": [10_000.0, 200.0],
            }),
            flow_unit=pd.DataFrame({
                "flow": ["onshore", "onshore"],
                "unit": ["SE02_windOnshore", "SE04_windOnshore"],
            }),
        )
        timeseries = summary.Timeseries(node_cf_mean=pd.DataFrame({
            "flow": ["onshore", "onshore"],
            "node": ["SE02_elec", "SE04_elec"],
            "cf": [0.40, 0.10],
        }))
        result = summary.capacity_weighted_cf(timeseries, workbook, zones=False)
        assert result.loc[result["area"] == "SE", "cf"].iloc[0] == pytest.approx(0.3941, abs=1e-3)

    def test_a_node_with_no_capacity_cannot_vote(self):
        """A capacity-factor series exists for nodes carrying no units of that flow."""
        workbook = make_workbook(
            p_gnu_io=pd.DataFrame({
                "grid": ["elec"], "node": ["FI00_elec"], "unit": ["FI00_windOnshore"],
                "input_output": ["output"], "capacity": [5000.0],
            }),
            flow_unit=pd.DataFrame({"flow": ["onshore"], "unit": ["FI00_windOnshore"]}),
        )
        timeseries = summary.Timeseries(node_cf_mean=pd.DataFrame({
            "flow": ["onshore", "onshore"],
            "node": ["FI00_elec", "EE00_elec"],
            "cf": [0.35, 0.99],
        }))
        result = summary.capacity_weighted_cf(timeseries, workbook, zones=False)
        assert set(result["area"]) == {"FI"}
        assert result["cf"].iloc[0] == pytest.approx(0.35)


class TestRetiredUnitsAreExcluded:
    def test_an_inactive_unit_adds_no_capacity(self):
        workbook = make_workbook(p_gnu_io=pd.DataFrame({
            "grid": ["elec", "elec"],
            "node": ["FI00_elec", "FI00_elec"],
            "unit": ["FI00_live", "FI00_retired"],
            "input_output": ["output", "output"],
            "capacity": [1000.0, 9999.0],
            "isActive": [1, 0],
        }))
        rows = summary.classify_capacity(workbook).rows
        assert rows["capacity"].sum() == pytest.approx(1000.0)

    def test_a_build_without_the_column_counts_every_row(self):
        """isActive is dropped when no row sets it -- absent means active."""
        workbook = make_workbook(p_gnu_io=pd.DataFrame({
            "grid": ["elec"], "node": ["FI00_elec"], "unit": ["FI00_live"],
            "input_output": ["output"], "capacity": [1000.0],
        }))
        assert summary.classify_capacity(workbook).rows["capacity"].sum() == pytest.approx(1000.0)


class TestEpsIsAZeroAndAnythingElseIsLoud:
    def test_eps_becomes_zero(self):
        result = summary.numeric_with_eps(pd.Series(["Eps", 5.0, "eps"]))
        assert list(result) == [0.0, 5.0, 0.0]

    def test_a_blank_stays_missing(self):
        """NaN in this column means 'a timeseries carries the value instead'."""
        assert summary.numeric_with_eps(pd.Series([None, 3.0])).isna().iloc[0]

    def test_an_unknown_word_raises_rather_than_becoming_missing(self):
        """errors='coerce' would turn a new GAMS literal into the same NaN that
        already means 'a timeseries carries this value', and the storage total
        would quietly grow a node."""
        with pytest.raises(ValueError):
            summary.numeric_with_eps(pd.Series(["undf", 1.0]))


class TestOnlyTheMarkedSheetsSkipARow:
    """The five parameter sheets carry a GDXXRW marker row; the others do not.

    Skipping a row on every sheet ate the first unit and the first flow mapping
    in an early run, which showed up only as a country's solar quietly missing
    from the net-load curve.
    """

    def test_marked_sheets_are_listed(self):
        assert "p_gnu_io" in summary.MARKER_ROW_SHEETS
        assert "flowUnit" not in summary.MARKER_ROW_SHEETS
        assert "unitUnittype" not in summary.MARKER_ROW_SHEETS

    def test_an_unmarked_sheet_keeps_its_first_row(self, tmp_path):
        path = tmp_path / "book.xlsx"
        with pd.ExcelWriter(path) as writer:
            pd.DataFrame({"flow": ["PV", "onshore"],
                          "unit": ["AT00_solarPV", "AT00_windOnshore"]}).to_excel(
                writer, sheet_name="flowUnit", index=False)
        rows = summary.read_bb_sheet(pd.ExcelFile(path), "flowUnit")
        assert list(rows["unit"]) == ["AT00_solarPV", "AT00_windOnshore"]


class TestAnOptionalColumnIsNeverIndexedDirectly:
    def test_a_missing_column_reads_as_its_default(self):
        frame = pd.DataFrame({"node": ["FI00_elec"]})
        assert summary.col_or(frame, "transferLoss", 0.0).iloc[0] == 0.0

    def test_a_present_column_is_returned_unchanged(self):
        frame = pd.DataFrame({"transferLoss": [0.02]})
        assert summary.col_or(frame, "transferLoss", 0.0).iloc[0] == pytest.approx(0.02)


class TestAnInternalCorridorIsCountedOnce:
    """Both ends of SE01-SE02 are Sweden, so recording it per end doubles it."""

    def test_sweden_gets_its_ring_once(self):
        workbook = make_workbook(p_gnn=pd.DataFrame({
            "grid": ["elec", "elec", "elec", "elec"],
            "from_node": ["SE01_elec", "SE02_elec", "SE02_elec", "SE03_elec"],
            "to_node": ["SE02_elec", "SE01_elec", "SE03_elec", "SE02_elec"],
            "transferCap": [3300.0, 3300.0, 7300.0, 7300.0],
        }))
        per_area, _, _ = summary.transfer_by_area(workbook, zones=False)
        assert per_area.loc["SE", "inter_zonal_MW"] == pytest.approx(10600.0)

    def test_at_zone_level_both_ends_are_different_areas(self):
        workbook = make_workbook(p_gnn=pd.DataFrame({
            "grid": ["elec", "elec"],
            "from_node": ["SE01_elec", "SE02_elec"],
            "to_node": ["SE02_elec", "SE01_elec"],
            "transferCap": [3300.0, 3300.0],
        }))
        per_area, _, _ = summary.transfer_by_area(workbook, zones=True)
        assert per_area.loc["SE01", "cross_border_MW"] == pytest.approx(3300.0)
        assert per_area.loc["SE02", "cross_border_MW"] == pytest.approx(3300.0)


class TestStorageEnergyIsPowerTimesItsRatio:
    """The report once said this energy did not exist. It is 1.06 TWh."""

    def _workbook(self, ratio=4.0, per_state=1.0, active=1):
        return make_workbook(
            p_gnu_io=pd.DataFrame({
                "grid": ["battery4h"],
                "node": ["DE00_battery4h"],
                "unit": ["DE00_BatteryDisch4h"],
                "input_output": ["input"],
                "isActive": [active],
                "capacity": [116821.4],
                "upperLimitCapacityRatio": [ratio],
            }),
            p_gn=pd.DataFrame({
                "grid": ["battery4h"],
                "node": ["DE00_battery4h"],
                "energyStoredPerUnitOfState": [per_state],
            }),
        )

    def test_energy_is_capacity_times_hours(self):
        table, facts = summary.storage_energy_from_ratio(self._workbook(), zones=False)
        assert table["MWh"].sum() == pytest.approx(116821.4 * 4.0)
        assert facts["by_grid"]["battery4h"]["hours"] == pytest.approx(4.0)
        assert facts["total_TWh"] == pytest.approx(116821.4 * 4.0 * 1e-6)

    def test_the_state_conversion_is_applied_not_assumed(self):
        """v_state is only MWh when energyStoredPerUnitOfState says it is."""
        table, _ = summary.storage_energy_from_ratio(self._workbook(per_state=0.5), zones=False)
        assert table["MWh"].sum() == pytest.approx(116821.4 * 4.0 * 0.5)

    def test_a_node_that_cannot_be_converted_is_named_not_guessed(self):
        """Zero is "not set" in this project, so the ratio is not hours here."""
        table, facts = summary.storage_energy_from_ratio(self._workbook(per_state=0.0), zones=False)
        assert table.empty
        assert facts["unconvertible"] == ["DE00_battery4h"]
        assert facts["total_TWh"] == 0.0

    def test_a_retired_storage_adds_nothing(self):
        _, facts = summary.storage_energy_from_ratio(self._workbook(active=0), zones=False)
        assert facts["total_TWh"] == 0.0

    def test_a_label_spanning_two_durations_claims_neither(self):
        """heatStor, heatStorL and heatStorXL are one label and 10, 100, 450 hours."""
        facts = {"by_grid": {
            "heatStor": {"hours_seen": [10.0]},
            "heatStorL": {"hours_seen": [100.0]},
            "battery4h": {"hours_seen": [4.0]},
        }}
        durations = summary.duration_by_label(facts)
        assert durations["battery"] == pytest.approx(4.0)
        assert "heat storage" not in durations


class TestTheNoiseFloorComesFromTheNumberOfYears:
    """A round 0.3 sits below the level 35 points can resolve, which is 0.33."""

    def test_thirty_five_years(self):
        assert summary.significance_floor(35) == pytest.approx(0.334, abs=0.002)

    def test_fewer_years_need_a_stronger_correlation(self):
        assert summary.significance_floor(10) > summary.significance_floor(35)

    def test_too_few_points_to_say(self):
        assert summary.significance_floor(4) is None


class TestAMissingMapAssetDegradesRatherThanFails:
    """A report without geometry is still a report, and says why it has no map."""

    def test_an_absent_asset_says_what_it_wanted(self, tmp_path):
        shapes = summary.load_zone_shapes(tmp_path / "nothing.geojson")
        assert not shapes.available
        assert "prepare_zone_geometry" in shapes.missing

    def test_an_unreadable_asset_is_reported_not_raised(self, tmp_path):
        broken = tmp_path / "zone_shapes.geojson"
        broken.write_text("{not json", encoding="utf-8")
        shapes = summary.load_zone_shapes(broken)
        assert not shapes.available
        assert shapes.missing


class TestCountryGeometryIsTheUnionOfItsZones:
    """SE is drawn from SE01..SE04, because the source has no dissolved outline."""

    def _shapes(self):
        ring = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]
        return summary.ZoneShapes(by_zone={
            "SE01": [[ring]], "SE02": [[ring]], "FI00": [[ring]], "PT00": [[ring]],
        })

    def test_a_country_collects_every_zone_it_has(self):
        out = summary.shapes_for_areas(self._shapes(), ["SE", "FI"], zones=False)
        assert len(out["SE"]) == 2
        assert len(out["FI"]) == 1

    def test_a_zone_level_map_keeps_them_apart(self):
        out = summary.shapes_for_areas(self._shapes(), ["SE01", "SE02"], zones=True)
        assert set(out) == {"SE01", "SE02"}

    def test_a_zone_this_build_does_not_report_becomes_context(self):
        """PT00 is in the asset and in no shipped scenario; it must still be drawn."""
        context = summary.context_polygons(self._shapes(), ["SE", "FI"], zones=False)
        assert len(context) == 1


class TestResidualDemandByTimescale:
    """Storage of a given duration removes a swing of that period, and no other.

    The windows have to be nested -- each a whole multiple of the one before --
    or a longer window can cancel *less* than a shorter one and the difference
    between them, which is what the figure draws, comes out negative. A calendar
    month of 730 hours against a week of 168 does exactly that.
    """

    def _windows(self):
        import numpy as np
        return [1] + [w for _, w in summary.DURATION_WINDOWS]

    def _chain(self, values):
        return [summary._residue(values, w) for w in self._windows()]

    def test_a_flat_deficit_survives_every_window(self):
        import numpy as np
        chain = self._chain(np.full(8760, 10.0))
        assert chain == pytest.approx([87600.0] * len(chain))

    def test_a_daily_swing_is_gone_after_a_day(self):
        import numpy as np
        day = np.tile(np.r_[np.full(12, 10.0), np.full(12, -10.0)], 365)
        chain = self._chain(day)
        assert chain[0] == pytest.approx(43800.0)
        assert chain[1] == pytest.approx(0.0)

    def test_a_seasonal_swing_survives_a_week_and_not_a_year(self):
        import numpy as np
        half = np.r_[np.full(4380, 10.0), np.full(4380, -10.0)]
        chain = self._chain(half)
        assert chain[2] > 0.0            # still there after a week
        assert chain[-1] == pytest.approx(0.0)

    def test_every_window_cancels_at_least_as_much_as_a_shorter_one(self):
        """The property the nesting exists to guarantee."""
        import numpy as np
        rng = np.random.default_rng(0)
        for values in (rng.normal(5, 20, 8760),
                       rng.normal(-5, 40, 8760),
                       np.r_[np.full(4380, 10.0), np.full(4380, -10.0)]):
            chain = self._chain(values)
            for shorter, longer in zip(chain, chain[1:]):
                assert longer <= shorter + 1e-6

    def test_the_windows_are_nested(self):
        windows = [w for _, w in summary.DURATION_WINDOWS if w is not None]
        for finer, coarser in zip(windows, windows[1:]):
            assert coarser % finer == 0, f"{coarser} is not a multiple of {finer}"

    def test_the_last_window_is_the_whole_series(self):
        import numpy as np
        assert summary.DURATION_WINDOWS[-1][1] is None
        assert summary._residue(np.r_[np.full(10, 1.0), np.full(10, -1.0)], None) == 0.0
