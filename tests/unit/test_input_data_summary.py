"""tools/input_data_summary.py -- the arithmetic a reader could never check.

Tools are not normally in this suite (see tests/README.md): they print a report
a person reads, and a wrong number there is usually visible. Five of this
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
- a reservoir bounded by two seasonal series has no single size. Its nameplate
  ceiling is twice its usable volume in the shipped scenarios, and both are
  believable numbers printed in TWh next to the word storage.
- the depth a store needs is not monotone in the window it is measured over,
  though on every real 8760-hour series it has been. The figure stacks the
  differences, so a rare inversion would draw a negative bar.

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


class TestHydroUsableVolumeIsNotItsNameplate:
    """The report printed 149 TWh of seasonal reservoir. 70 TWh of it can move."""

    def _records(self, upward, downward, branches=("f01",)):
        rows = []
        for branch in branches:
            for hour, (up, down) in enumerate(zip(upward, downward)):
                rows.append(("reservoir", "SE01_reservoir", "upwardLimit", branch,
                             f"t{hour:06d}", up))
                rows.append(("reservoir", "SE01_reservoir", "downwardLimit", branch,
                             f"t{hour:06d}", down))
        return pd.DataFrame(rows, columns=["grid", "node", "param_gnBoundaryTypes",
                                           "f", "t", "value"])

    def test_the_three_measures_are_ordered(self):
        table, _ = summary._storage_limit_measures(
            self._records([100.0, 80.0, 60.0], [40.0, 20.0, 0.0]))
        row = table.iloc[0]
        assert row["usable_mean_MWh"] == pytest.approx(60.0)      # 60, 60, 60
        assert row["usable_envelope_MWh"] == pytest.approx(100.0)  # 100 - 0
        assert row["nameplate_MWh"] == pytest.approx(100.0)
        assert (row["usable_mean_MWh"] <= row["usable_envelope_MWh"]
                <= row["nameplate_MWh"])

    def test_a_seasonal_floor_is_not_usable_volume(self):
        """Ignoring downwardLimit was most of the difference between 149 and 70."""
        table, _ = summary._storage_limit_measures(
            self._records([100.0, 100.0], [90.0, 90.0]))
        assert table.iloc[0]["usable_mean_MWh"] == pytest.approx(10.0)
        assert table.iloc[0]["nameplate_MWh"] == pytest.approx(100.0)

    def test_one_forecast_branch_is_read_and_named(self):
        """The branches differ by up to 113 GWh an hour; blending them is silent."""
        table, branch = summary._storage_limit_measures(
            self._records([100.0, 100.0], [0.0, 0.0], branches=("f01", "f02", "f03")))
        assert branch == "f01"
        assert len(table) == 1
        assert table.iloc[0]["usable_mean_MWh"] == pytest.approx(100.0)

    def test_a_ceiling_with_no_floor_is_usable_to_the_bottom(self):
        rows = self._records([100.0], [0.0])
        table, _ = summary._storage_limit_measures(
            rows[rows["param_gnBoundaryTypes"] == "upwardLimit"])
        assert table.iloc[0]["usable_mean_MWh"] == pytest.approx(100.0)


class TestStorageGroupsAreNested:
    """Two groups, the second containing the first, and heat storage in neither."""

    def _workbook(self, battery="battery4h"):
        """Two stores charged and discharged on elec, one on dheat, one with inflow.

        Written as the builder writes it: hydro states its size as an
        upwardLimit in MWh, battery and heat state theirs as a duration per MW.
        """
        node = f"DE00_{battery}"
        rows = [
            # grid,      node,            unit,              side,     MW,    ratio
            (battery,    node,            "DE00_Charge",     "output", 100.0, None),
            (battery,    node,            "DE00_Disch",      "input",  100.0, 4.0),
            ("elec",     "DE00_elec",     "DE00_Charge",     "input",  100.0, None),
            ("elec",     "DE00_elec",     "DE00_Disch",      "output", 100.0, None),
            ("psClosed", "DE00_psClosed", "DE00_Pump",       "output",  50.0, None),
            ("psClosed", "DE00_psClosed", "DE00_Turb",       "input",   50.0, None),
            ("elec",     "DE00_elec",     "DE00_Pump",       "input",   50.0, None),
            ("elec",     "DE00_elec",     "DE00_Turb",       "output",  50.0, None),
            ("psOpen",   "NOS0_psOpen",   "NOS0_Turb",       "input",  200.0, None),
            ("elec",     "NOS0_elec",     "NOS0_Turb",       "output", 200.0, None),
            ("heatStor", "DE00_heatStor", "DE00_HeatCharge", "output",  10.0, None),
            ("heatStor", "DE00_heatStor", "DE00_HeatDisch",  "input",   10.0, 10.0),
            ("dheat",    "DE00_dheat",    "DE00_HeatCharge", "input",   10.0, None),
            ("dheat",    "DE00_dheat",    "DE00_HeatDisch",  "output",  10.0, None),
        ]
        return make_workbook(
            p_gn=pd.DataFrame({
                "grid": [battery, "psClosed", "psOpen", "heatStor", "dheat"],
                "node": [node, "DE00_psClosed", "NOS0_psOpen", "DE00_heatStor", "DE00_dheat"],
                "energyStoredPerUnitOfState": [1.0, 1.0, 1.0, 1.0, 0.0],
            }),
            p_gnu_io=pd.DataFrame({
                "grid": [r[0] for r in rows],
                "node": [r[1] for r in rows],
                "unit": [r[2] for r in rows],
                "input_output": [r[3] for r in rows],
                "capacity": [r[4] for r in rows],
                "upperLimitCapacityRatio": [r[5] for r in rows],
            }),
            boundary=pd.DataFrame({
                "grid": ["psClosed", "psClosed", "psOpen", "psOpen", "dheat"],
                "node": ["DE00_psClosed", "DE00_psClosed", "NOS0_psOpen", "NOS0_psOpen",
                         "DE00_dheat"],
                "param_gnBoundaryTypes": ["upwardLimit", "downwardLimit", "upwardLimit",
                                          "downwardLimit", "balancePenalty"],
                "useConstant": [1, 1, 1, 1, 1],
                "constant": [1000.0, 0.0, 5000.0, 0.0, 4000.0],
                "useTimeseries": [0, 0, 0, 0, 0],
            }),
        )

    def test_each_grid_lands_where_its_physics_puts_it(self):
        groups = summary.storage_groups_by_grid(
            self._workbook(), summary.HYDRO_INFLOW_GRIDS_FALLBACK)
        assert groups["battery4h"] == "elec_literal"
        assert groups["psClosed"] == "elec_literal"
        assert groups["psOpen"] == "elec_practical"
        assert groups["heatStor"] is None

    def test_a_renamed_battery_is_still_a_battery(self):
        """The TYNDP scenarios call it battery4h; Observed Trends calls it battery."""
        for name in ("battery", "battery4h", "battery8h"):
            groups = summary.storage_groups_by_grid(
                self._workbook(battery=name), summary.HYDRO_INFLOW_GRIDS_FALLBACK)
            assert groups[name] == "elec_literal"

    def test_a_grid_that_holds_nothing_is_in_no_group(self):
        groups = summary.storage_groups_by_grid(
            self._workbook(), summary.HYDRO_INFLOW_GRIDS_FALLBACK)
        assert "dheat" not in groups

    def test_the_groups_nest(self):
        inventory = summary.storage_inventory(
            self._workbook(), summary.Timeseries(), zones=False,
            inflow_grids=summary.HYDRO_INFLOW_GRIDS_FALLBACK)
        totals = inventory.by_group
        for measure in ("nameplate_TWh", "usable_mean_TWh", "usable_envelope_TWh"):
            assert totals.loc["elec_literal", measure] <= totals.loc["elec_practical", measure]

    def test_every_group_gets_a_row_even_with_nothing_in_it(self):
        """A build with no closed-loop pumped hydro prints a zero, not a gap."""
        inventory = summary.storage_inventory(
            self._workbook(), summary.Timeseries(), zones=False,
            inflow_grids=summary.HYDRO_INFLOW_GRIDS_FALLBACK)
        assert list(inventory.by_group.index) == [k for k, _, _ in summary.STORAGE_GROUPS]


class TestAStorageGridIsNotGuessedToBePumpedStorage:
    """Both halves of a live mislabelling: dheat was a state, and states fell to psOpen."""

    def test_an_unknown_storage_grid_is_named_not_assumed(self):
        assert summary.storage_label("caes") == summary.OTHER_LABEL
        assert summary.storage_label("psClosed") == "pumped storage"

    def test_a_grid_with_inflow_is_hydro_wherever_it_is_reported(self):
        """psOpen read as pumped storage here and hydro in the capacity table."""
        assert summary.technology_label("psOpen", {"psOpen"}) == "hydro"
        assert summary.technology_label("psClosed", {"psOpen"}) == "pumped storage"

    def test_a_balance_penalty_does_not_make_a_carrier_a_store(self):
        """Every dheat node carries balancePenalty and maxSpill; none holds energy."""
        workbook = make_workbook(
            boundary=pd.DataFrame({
                "grid": ["dheat", "battery"],
                "node": ["DE00_dheat", "DE00_battery"],
                "param_gnBoundaryTypes": ["balancePenalty", "downwardLimit"],
            }),
            p_gn=pd.DataFrame({
                "grid": ["dheat", "battery"],
                "node": ["DE00_dheat", "DE00_battery"],
                "energyStoredPerUnitOfState": [0.0, 1.0],
            }),
        )
        assert summary.state_grids(workbook) == {"battery"}


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


class TestAnAreaCollectsEveryFeatureThatMapsToIt:
    """Each level has its own asset, and the same lookup serves both.

    A two-letter country code is its own ``country_of``, so a country feature
    keyed ``SE`` and a zone feature keyed ``SE01`` both land under ``SE`` on a
    country map. The tool no longer draws a country from its zones' outlines --
    the country asset carries the dissolved border -- but the lookup that finds
    them has not changed, and this is what pins it.
    """

    RING = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]]

    def _zone_level(self):
        return summary.ZoneShapes(by_zone={
            "SE01": [[self.RING]], "SE02": [[self.RING]],
            "FI00": [[self.RING]], "PT00": [[self.RING]],
        })

    def _country_level(self):
        return summary.ZoneShapes(by_zone={
            "SE": [[self.RING]], "FI": [[self.RING]], "PT": [[self.RING]],
        })

    def test_a_country_is_one_feature(self):
        out = summary.shapes_for_areas(self._country_level(), ["SE", "FI"], zones=False)
        assert len(out["SE"]) == 1 and len(out["FI"]) == 1

    def test_a_zone_level_map_keeps_the_zones_apart(self):
        out = summary.shapes_for_areas(self._zone_level(), ["SE01", "SE02"], zones=True)
        assert set(out) == {"SE01", "SE02"}

    def test_an_area_this_build_does_not_report_becomes_context(self):
        """PT is in the asset and in no shipped scenario; it must still be drawn."""
        context = summary.context_polygons(self._country_level(), ["SE", "FI"], zones=False)
        assert len(context) == 1

    def test_both_assets_are_named(self):
        """The maps use both at once: zone fills, country borders over them."""
        assert summary.ZONE_ASSET.name == "zone_shapes.geojson"
        assert summary.COUNTRY_ASSET.name == "country_shapes.geojson"

    def test_both_assets_are_where_the_names_say(self):
        """A half-done rename of the folder would not fail anywhere else.

        ``load_zone_shapes`` returns empty for a file it cannot read, so the
        report keeps building and simply draws no maps. Nothing else notices.
        """
        for asset in (summary.ZONE_ASSET, summary.COUNTRY_ASSET):
            assert asset.parent.name == "maps"
            assert asset.is_file(), f"{asset} is committed and must be readable"


class TestRequiredStorageDepth:
    """How deep a store must be to flatten a swing, window by window.

    The quantity is an energy in TWh, which is the point: the unservable energy
    per year this replaced could never be held against an installed volume,
    however the caption was worded.

    The windows have to be nested -- each a whole multiple of the one before --
    because the figure stacks the increase from one to the next. A calendar
    month of 730 hours against a week of 168 cuts across it and that difference
    comes out negative.
    """

    def _windows(self):
        return [w for _, w in summary.DURATION_WINDOWS]

    def _chain(self, values):
        return [summary.storage_depth(values, w) for w in self._windows()]

    def _timeseries(self, values):
        import numpy as np
        return summary.Timeseries(
            years=[1990], areas=["DE"], hours_per_year=len(values),
            hourly={"demand_elec": np.asarray(values, dtype="float64").reshape(-1, 1)},
        )

    def test_a_flat_series_needs_no_storage(self):
        """Nothing has to move, so nothing has to be held."""
        import numpy as np
        assert self._chain(np.full(8760, 10.0)) == pytest.approx([0.0] * 5)

    def test_a_daily_swing_needs_the_same_store_at_every_window(self):
        """Twelve hours of surplus is 120 MWh whether you look at a day or a year."""
        import numpy as np
        day = np.tile(np.r_[np.full(12, 10.0), np.full(12, -10.0)], 365)
        assert self._chain(day) == pytest.approx([120.0] * 5)

    def test_a_seasonal_swing_needs_a_season(self):
        import numpy as np
        half = np.r_[np.full(4380, 10.0), np.full(4380, -10.0)]
        chain = self._chain(half)
        assert chain[-1] == pytest.approx(43800.0)
        assert chain[1] < chain[-1] / 100.0        # a week reaches almost none of it

    def test_a_constant_offset_changes_nothing(self):
        """Firm generation supplies the block mean; the store supplies the rest."""
        import numpy as np
        rng = np.random.default_rng(3)
        values = rng.normal(0.0, 20.0, 8760)
        assert self._chain(values + 5000.0) == pytest.approx(self._chain(values))

    def test_a_sine_needs_its_half_cycle_integral(self):
        """A sin(2*pi*t/P) integrates to A*P/pi between its turning points."""
        import numpy as np
        period, amplitude = 24, 10.0
        wave = amplitude * np.sin(2 * np.pi * np.arange(8760) / period)
        full = summary.storage_depth(wave, period)
        assert full == pytest.approx(amplitude * period / np.pi, rel=0.03)
        assert summary.storage_depth(wave, period // 4) < full

    def test_the_ragged_last_block_is_still_a_block(self):
        """8760 is not a multiple of 168, and the leftover hours are real."""
        import numpy as np
        values = np.zeros(200)
        values[168:184] = 10.0                      # entirely inside the tail block
        assert summary.storage_depth(values, 168) > 0.0

    def test_the_raw_depth_can_fall_as_the_window_grows(self):
        """Why depth_by_timescale accumulates before it differences.

        A longer block subtracts a different mean, which tilts the running sum
        and can shrink its range. It has never been seen on a real 8760-hour
        series, which is exactly why the guard looks removable.
        """
        values = [-1.0, -1.0, 0.0, 0.0, 2.0, -1.0, -1.0, 2.0]
        assert summary.storage_depth(values, 4) == pytest.approx(3.0)
        assert summary.storage_depth(values, 8) == pytest.approx(2.0)

    def test_the_chain_never_goes_backwards(self):
        """The property the figure depends on, asserted where the figure reads it."""
        import numpy as np
        rng = np.random.default_rng(0)
        for values in (rng.normal(5, 20, 8760),
                       np.cumsum(rng.normal(0, 5, 8760)),
                       np.r_[np.full(4380, 10.0), np.full(4380, -10.0)]):
            table = summary.depth_by_timescale(
                summary.netload_by_area(self._timeseries(values)), self._timeseries(values))
            assert (table["by_area"].to_numpy() >= -1e-9).all()

    def test_the_windows_are_nested(self):
        windows = [w for _, w in summary.DURATION_WINDOWS if w is not None]
        for finer, coarser in zip(windows, windows[1:]):
            assert coarser % finer == 0, f"{coarser} is not a multiple of {finer}"

    def test_the_last_window_is_the_whole_series(self):
        assert summary.DURATION_WINDOWS[-1][1] is None


class TestTheResidualPutsHydroInflowOnTheSupplySide:
    """Norway's inflow is three times its demand. Left out, it read as a deficit."""

    def _timeseries(self):
        import numpy as np
        column = lambda v: np.full((24, 1), v, dtype="float64")
        return summary.Timeseries(
            years=[1990], areas=["NO"], hours_per_year=24,
            hourly={"demand_elec": column(100.0), "vre": column(20.0),
                    "inflow_hydro": column(50.0)},
        )

    def test_net_load_leaves_hydro_for_firm_capacity_to_cover(self):
        """Chapter 7 asks what firm capacity must cover, and hydro is dispatchable."""
        assert summary.netload_by_area(self._timeseries())["NO"][0] == pytest.approx(80.0)

    def test_the_storage_residual_counts_the_inflow(self):
        assert summary.residual_after_inflow(self._timeseries())["NO"][0] == pytest.approx(30.0)


class TestDemandCountsBothRoutesWithoutDoubleCounting:
    """A constant influx and a timeseries are alternatives, never a sum.

    `p_gn`'s `influx` is "overridden by time series if provided", and the model
    gates it on `not gn_influxTs(grid, node)` in six places. Adding them would
    double-count every node carrying both, in every table at once, and the
    result would look entirely plausible. Industrial steam is 495 TWh/yr of
    constant in the shipped scenarios -- more than district heat -- so reading
    only the `ts_influx` families misses a carrier whole.
    """

    def _steam_workbook(self, influx=(-1000.0, -500.0)):
        return make_workbook(
            p_gn=pd.DataFrame({
                "grid": ["steam", "steam"],
                "node": ["FI00_steam_industry", "SE03_steam_industry"],
                "influx": list(influx),
            }),
        )

    def test_a_constant_becomes_annual_energy(self):
        out = summary.constant_demand(self._steam_workbook(), "steam", zones=False)
        expected = 1000.0 * summary.HOURS_PER_YEAR * summary.MWH_TO_TWH
        assert out.set_index("area").loc["FI", "mean"] == pytest.approx(expected)

    def test_demand_is_positive_though_influx_is_negative(self):
        out = summary.constant_demand(self._steam_workbook(), "steam", zones=False)
        assert (out["mean"] > 0).all()

    def test_an_area_written_as_exactly_zero_is_not_an_area_with_demand(self):
        """Two Danish steam nodes are written 0.00; they are not steam areas."""
        workbook = self._steam_workbook(influx=(-1000.0, 0.0))
        out = summary.constant_demand(workbook, "steam", zones=False)
        assert set(out["area"]) == {"FI"}

    def test_inflow_at_one_node_never_cancels_demand_at_another(self):
        """Only the outflux is summed. A net would be a number that is neither."""
        workbook = self._steam_workbook(influx=(-1000.0, 900.0))
        out = summary.constant_demand(workbook, "steam", zones=False)
        expected = 1000.0 * summary.HOURS_PER_YEAR * summary.MWH_TO_TWH
        assert out["mean"].sum() == pytest.approx(expected)

    def test_a_timeseries_wins_and_is_not_added_to(self):
        workbook = make_workbook(p_gn=pd.DataFrame({
            "grid": ["elec"], "node": ["FI00_elec"], "influx": [-1000.0],
        }))
        timeseries = summary.Timeseries(
            years=[1995], areas=["FI"], hours_per_year=8760,
            annual=pd.DataFrame({"key": ["demand_elec"], "area": ["FI"],
                                 "year": [1995], "TWh": [80.0]}),
        )
        frame, is_constant = summary.carrier_demand(workbook, timeseries, "elec", zones=False)
        assert not is_constant
        assert frame.set_index("area").loc["FI", "mean"] == pytest.approx(80.0)

    def test_the_constant_survives_a_run_that_read_no_gdx(self):
        """It is a workbook fact, so no GAMS install is needed to report it."""
        nothing = summary.Timeseries(skipped="gamsapi is not importable")
        frame, is_constant = summary.carrier_demand(
            self._steam_workbook(), nothing, "steam", zones=False)
        assert is_constant and not frame.empty


class TestTheCarrierMapColoursWhatIsModelled:
    """Tinting only what was demanded drew a worse map than tinting what exists.

    Hydrogen has nodes in 22 bidding zones of the TYNDP scenarios and demand in
    none, so the carrier those scenarios were built to study tinted nothing and
    the map came out identical to Observed Trends, which has no hydrogen at all.
    Which carriers have no sink is a fact about demand and is listed as one.
    """

    def _presence(self, index=("FI00",), **states):
        return pd.DataFrame(states, index=list(index)).reindex(
            columns=[g for g, _, _ in summary.CARRIERS], fill_value=summary.PRESENCE_ABSENT)

    def _modelled(self, presence, area):
        return [c for c in summary.MAP_CARRIERS
                if presence.loc[area, c] != summary.PRESENCE_ABSENT]

    def test_a_node_with_no_demand_still_tints(self):
        presence = self._presence(elec=[summary.PRESENCE_DEMAND], H2=[summary.PRESENCE_NODE])
        assert self._modelled(presence, "FI00") == ["elec", "H2"]

    def test_a_carrier_that_is_not_there_does_not_tint(self):
        presence = self._presence(elec=[summary.PRESENCE_DEMAND])
        assert self._modelled(presence, "FI00") == ["elec"]

    def test_a_carrier_demanded_nowhere_is_named_beside_the_key(self):
        """This was the map's title, where a caveat wore the clothes of a finding."""
        presence = self._presence(index=("FI00", "SE01"),
                                  elec=[summary.PRESENCE_DEMAND] * 2,
                                  H2=[summary.PRESENCE_NODE] * 2)
        assert summary._carriers_without_demand(presence) == ["hydrogen"]

    def test_a_carrier_demanded_somewhere_is_not_named(self):
        presence = self._presence(index=("FI00", "SE01"),
                                  elec=[summary.PRESENCE_DEMAND] * 2,
                                  H2=[summary.PRESENCE_DEMAND, summary.PRESENCE_NODE])
        assert summary._carriers_without_demand(presence) == []

    def test_steam_is_in_the_report_and_off_the_map(self):
        assert "steam" in [g for g, _, _ in summary.CARRIERS]
        assert "steam" not in summary.MAP_CARRIERS

    def test_the_palette_covers_every_combination_the_map_can_draw(self):
        """_blend_for falls back to a grey, which would hide a missing entry."""
        from itertools import combinations
        for size in range(1, len(summary.MAP_CARRIERS) + 1):
            for combination in combinations(summary.MAP_CARRIERS, size):
                assert frozenset(combination) in summary.CARRIER_BLEND

    def test_the_four_fills_that_are_drawn_are_four_different_colours(self):
        """Every area has electricity, so these four are the map's whole palette."""
        drawn = [("elec",), ("elec", "dheat"), ("elec", "H2"), ("elec", "dheat", "H2")]
        fills = [summary._blend_for(c) for c in drawn]
        assert len(set(fills)) == len(drawn)
        assert summary.CARRIER_BLEND_UNKNOWN not in fills


class TestACountryIsItsMostDevelopedZone:
    """One Finnish zone with district heat makes Finland a district-heat country."""

    def _zones(self, **states):
        return pd.DataFrame(states, index=["FI00", "FI01", "SE01"]).reindex(
            columns=[g for g, _, _ in summary.CARRIERS], fill_value=summary.PRESENCE_ABSENT)

    def test_demand_in_one_zone_carries_the_country(self):
        rolled = summary.presence_by_country(self._zones(
            dheat=[summary.PRESENCE_DEMAND, summary.PRESENCE_ABSENT, summary.PRESENCE_ABSENT]))
        assert rolled.loc["FI", "dheat"] == summary.PRESENCE_DEMAND
        assert rolled.loc["SE", "dheat"] == summary.PRESENCE_ABSENT

    def test_a_node_without_demand_does_not_outrank_one_with_it(self):
        rolled = summary.presence_by_country(self._zones(
            dheat=[summary.PRESENCE_NODE, summary.PRESENCE_DEMAND, summary.PRESENCE_ABSENT]))
        assert rolled.loc["FI", "dheat"] == summary.PRESENCE_DEMAND

    def test_the_country_count_never_exceeds_the_zone_count(self):
        zones = self._zones(elec=[summary.PRESENCE_DEMAND] * 3)
        rolled = summary.presence_by_country(zones)
        assert len(rolled) == 2 and len(zones) == 3
        assert list(rolled.columns) == list(zones.columns)


class TestTheSummaryHasATopLevelEntryPoint:
    """build_input_summary.py sits beside build_input_data.py and does nothing else.

    The wrapper exists so that building a folder and reading it are one command
    each. What it must not become is a second command line: an argument handled
    there and not in the tool would work from the root and fail from tools/.
    """

    def test_it_passes_the_arguments_through_and_returns_the_exit_code(self, monkeypatch):
        import build_input_summary

        seen = {}

        def fake_main(argv):
            seen["argv"] = argv
            return 2

        monkeypatch.setattr(build_input_summary.input_data_summary, "main", fake_main)
        assert build_input_summary.main(["input_OT2030", "--zones"]) == 2
        assert seen["argv"] == ["input_OT2030", "--zones"]

    def test_it_wraps_the_one_tool_and_not_a_copy_of_it(self):
        """Two module objects would mean two sets of constants to keep in step."""
        import build_input_summary

        assert build_input_summary.input_data_summary is summary

    def test_the_help_description_names_no_file(self):
        """argparse prints the real program name; a filename in the description
        as well would be the wrong one whenever the wrapper is what was run."""
        description = summary.build_arg_parser().description
        assert ".py" not in description
        assert description.startswith("what is in one built input-data folder")
