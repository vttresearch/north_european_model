"""A constant influx for grids no processor covers.

A grid with an annual demand but no timeseries processor gets a flat rate rather
than nothing. What is pinned here is the **unit conversion**: TWh/year to MWh/h,
negative because demand is a withdrawal. It is arithmetic with one right answer,
so the values are pinned exactly (pinning case 2 in tests/README.md).

The rate is always per 8760 hours, and now visibly so -- there is no window to
get it wrong against. This used to write the same number into every hour of the
configured window and ship it as a GDX, which ``changes.inc`` then collapsed back
into ``p_gn('influx')``. Nothing was ever being detected there: the value is a
constant by construction, so it is contributed as one.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

from src.source_data.source_data_contributions import CONTRIBUTION_KEYS
from src.timeseries.timeseries_inputs import TimeseriesPipelineInputs
from src.timeseries.timeseries_pipeline import TimeseriesPipeline
from tests._common.fixtures import FakeLogger, make_config

HOURS_PER_YEAR = 8760


def _pipeline(demands: pd.DataFrame, *, logger=None, **config_overrides):
    """A pipeline built only far enough to call its private helpers.

    __init__ pulls df_demanddata off the source pipeline and otherwise just
    stores what it is given, so a SimpleNamespace stands in and no disk, cache
    or GAMS is involved.
    """
    return TimeseriesPipeline(
        TimeseriesPipelineInputs(
            config=make_config(**config_overrides),
            input_folder=".",
            output_folder=".",
            cache_manager=None,
            source_data_pipeline=SimpleNamespace(df_demanddata=demands),
            logger=logger or FakeLogger(),
        )
    )


def _demands(*rows) -> pd.DataFrame:
    """One default row when none are given, so the frame always has columns."""
    return pd.DataFrame(
        [
            {"grid": "elec", "node": "FI_elec", "twh/year": 10.0, **row}
            for row in (rows or ({},))
        ]
    )


def _influx(frame, grids, **kwargs):
    return _pipeline(frame, **kwargs)._influx_for_grids_without_a_processor(frame, grids)


class TestTheConversion:
    def test_twh_per_year_becomes_negative_mwh_per_hour(self):
        # 1 TWh/year = 1e6 MWh spread over 8760 h, withdrawn -> negative.
        out = _influx(_demands({"twh/year": 1.0}), {"elec"})
        assert out["influx"].iloc[0] == round(1e6 / HOURS_PER_YEAR * -1, 2)

    @pytest.mark.parametrize("twh", [0.5, 1.0, 10.0, 123.456])
    def test_the_conversion_scales_linearly(self, twh):
        out = _influx(_demands({"twh/year": twh}), {"elec"})
        assert out["influx"].iloc[0] == round(twh * 1e6 / HOURS_PER_YEAR * -1, 2)

    def test_demand_is_negative(self):
        out = _influx(_demands({"twh/year": 5.0}), {"elec"})
        assert (out["influx"] < 0).all()

    def test_zero_demand_stays_zero(self):
        out = _influx(_demands({"twh/year": 0.0}), {"elec"})
        assert (out["influx"] == 0).all()

    def test_the_rate_does_not_depend_on_the_window(self):
        """The nominal year is 8760 hours whatever bb_timeseries_length says.

        On the 5-year config, dividing by the window instead would scale every
        unprocessed demand by 1/5 and look merely "a bit low".
        """
        frame = _demands({"twh/year": 10.0})
        one_year = _influx(frame, {"elec"}, bb_timeseries_length=365)
        five_years = _influx(frame, {"elec"}, bb_timeseries_length=365 * 5)

        assert one_year["influx"].iloc[0] == five_years["influx"].iloc[0]


class TestTheContribution:
    def test_it_is_a_nodedata_contribution(self):
        """One row per node, keyed the way df_nodedata is keyed.

        The key is what decides whether the influx lands on the node's existing
        workbook row or adds a new one, so it is the contract, not a detail.
        """
        out = _influx(_demands(), {"elec"})

        assert set(CONTRIBUTION_KEYS["nodedata"]) <= set(out.columns)
        assert "influx" in out.columns

    def test_one_row_per_node(self):
        frame = _demands({"node": "FI_elec"}, {"node": "SE_elec"})
        out = _influx(frame, {"elec"})

        assert len(out) == 2

    def test_each_node_gets_its_own_rate(self):
        frame = _demands(
            {"node": "FI_elec", "twh/year": 10.0},
            {"node": "SE_elec", "twh/year": 20.0},
        )
        out = _influx(frame, {"elec"}).set_index("node")["influx"]

        # Each node against its own conversion rather than against the other:
        # values are rounded to 2 dp, so round(20e6/8760) is 2283.11 while twice
        # round(10e6/8760) is 2283.10. The ratio is not exactly 2 and should not
        # be asserted to be.
        assert out["FI_elec"] == round(10.0 * 1e6 / HOURS_PER_YEAR * -1, 2)
        assert out["SE_elec"] == round(20.0 * 1e6 / HOURS_PER_YEAR * -1, 2)


class TestSelection:
    def test_only_the_requested_grids_are_generated(self):
        frame = _demands({"grid": "elec"}, {"grid": "heat", "node": "FI_heat"})
        out = _influx(frame, {"heat"})
        assert set(out["grid"]) == {"heat"}

    def test_grid_matching_ignores_case(self):
        # The set is lowercased upstream; source data casing is the user's.
        out = _influx(_demands({"grid": "Elec"}), {"elec"})
        assert len(out) == 1

    def test_no_matching_grid_gives_an_empty_frame_with_the_right_columns(self):
        # An empty frame with no columns would break the stacking downstream.
        out = _influx(_demands(), {"nothing_matches"})
        assert out.empty
        assert list(out.columns) == ["grid", "node", "influx"]


class TestMissingInput:
    @pytest.mark.parametrize("missing", ["grid", "node", "twh/year"])
    def test_a_missing_required_column_warns_and_returns_the_right_shape(self, missing):
        # Error policy: after logger init, log and continue with a safe default.
        logger = FakeLogger()
        frame = _demands().drop(columns=[missing])
        out = _influx(frame, {"elec"}, logger=logger)

        assert out.empty
        assert list(out.columns) == ["grid", "node", "influx"]
        logger.assert_logged(missing, level="warn")

    def test_a_row_with_an_uncomputable_demand_is_skipped_not_fatal(self):
        logger = FakeLogger()
        frame = _demands(
            {"node": "FI_elec", "twh/year": "not a number"},
            {"node": "SE_elec", "twh/year": 10.0},
        )
        out = _influx(frame, {"elec"}, logger=logger)

        # The good row still gets its rate.
        assert set(out["node"]) == {"SE_elec"}
