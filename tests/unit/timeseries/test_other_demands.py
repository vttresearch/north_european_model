"""Flat demand profiles for grids no processor covers.

A grid with an annual demand but no timeseries processor gets a flat hourly
profile instead of nothing. Two things about it are worth pinning.

The **unit conversion** is the contract: TWh/year to MWh/h, negative because
demand is a withdrawal. It is arithmetic with one right answer, so the values
are pinned exactly (pinning case 2 in tests/README.md).

The **rate is always per 8760 hours**, whatever ``bb_timeseries_length`` says.
A 5-year window does not spread one year's demand across five; it repeats the
same hourly rate. Easy to get backwards, and a factor-of-five error in demand
would be visible only as an implausible result far downstream.
"""

from types import SimpleNamespace

import pandas as pd
import pytest

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


class TestTheConversion:
    def test_twh_per_year_becomes_negative_mwh_per_hour(self):
        # 1 TWh/year = 1e6 MWh spread over 8760 h, withdrawn -> negative.
        out = _pipeline(_demands({"twh/year": 1.0}))._create_other_demands(
            _demands({"twh/year": 1.0}), {"elec"}
        )
        assert out["value"].iloc[0] == round(1e6 / HOURS_PER_YEAR * -1, 2)

    @pytest.mark.parametrize("twh", [0.5, 1.0, 10.0, 123.456])
    def test_the_conversion_scales_linearly(self, twh):
        frame = _demands({"twh/year": twh})
        out = _pipeline(frame)._create_other_demands(frame, {"elec"})
        assert out["value"].iloc[0] == round(twh * 1e6 / HOURS_PER_YEAR * -1, 2)

    def test_demand_is_negative(self):
        frame = _demands({"twh/year": 5.0})
        out = _pipeline(frame)._create_other_demands(frame, {"elec"})
        assert (out["value"] < 0).all()

    def test_zero_demand_stays_zero(self):
        frame = _demands({"twh/year": 0.0})
        out = _pipeline(frame)._create_other_demands(frame, {"elec"})
        assert (out["value"] == 0).all()


class TestTheWindow:
    def test_one_row_per_hour_of_the_configured_window(self):
        frame = _demands()
        out = _pipeline(frame, bb_timeseries_length=3)._create_other_demands(frame, {"elec"})
        assert len(out) == 3 * 24
        assert out["t"].iloc[0] == "t000001"
        assert out["t"].iloc[-1] == "t000072"

    def test_the_hourly_rate_does_not_change_with_the_window_length(self):
        """A longer window repeats the rate; it does not divide the year across it.

        Getting this backwards on the 5-year config would scale every
        unprocessed demand by 1/5 and look merely "a bit low".
        """
        frame = _demands({"twh/year": 10.0})
        one_year = _pipeline(frame, bb_timeseries_length=365)._create_other_demands(
            frame, {"elec"}
        )
        five_years = _pipeline(frame, bb_timeseries_length=365 * 5)._create_other_demands(
            frame, {"elec"}
        )

        assert one_year["value"].iloc[0] == five_years["value"].iloc[0]
        assert len(five_years) == 5 * len(one_year)

    def test_every_row_carries_the_realized_weather_branch(self):
        frame = _demands()
        out = _pipeline(frame)._create_other_demands(frame, {"elec"})
        assert set(out["f"]) == {"f00"}

    def test_t_labels_are_unique_and_ordered(self):
        # Duplicate or shuffled labels would corrupt the parameter silently.
        frame = _demands()
        out = _pipeline(frame, bb_timeseries_length=2)._create_other_demands(frame, {"elec"})
        assert out["t"].is_unique
        assert list(out["t"]) == sorted(out["t"])


class TestSelection:
    def test_only_the_requested_grids_are_generated(self):
        frame = _demands({"grid": "elec"}, {"grid": "heat", "node": "FI_heat"})
        out = _pipeline(frame, bb_timeseries_length=1)._create_other_demands(frame, {"heat"})
        assert set(out["grid"]) == {"heat"}

    def test_grid_matching_ignores_case(self):
        # The set is lowercased upstream; source data casing is the user's.
        frame = _demands({"grid": "Elec"})
        out = _pipeline(frame, bb_timeseries_length=1)._create_other_demands(frame, {"elec"})
        assert len(out) == 24

    def test_each_node_gets_its_own_profile(self):
        frame = _demands(
            {"node": "FI_elec", "twh/year": 10.0},
            {"node": "SE_elec", "twh/year": 20.0},
        )
        out = _pipeline(frame, bb_timeseries_length=1)._create_other_demands(frame, {"elec"})

        assert len(out) == 2 * 24
        per_node = out.groupby("node")["value"].first()
        # Each node against its own conversion rather than against the other:
        # values are rounded to 2 dp, so round(20e6/8760) is 2283.11 while twice
        # round(10e6/8760) is 2283.10. The ratio is not exactly 2 and should not
        # be asserted to be.
        assert per_node["FI_elec"] == round(10.0 * 1e6 / HOURS_PER_YEAR * -1, 2)
        assert per_node["SE_elec"] == round(20.0 * 1e6 / HOURS_PER_YEAR * -1, 2)

    def test_no_matching_grid_gives_an_empty_frame_with_the_right_columns(self):
        # An empty frame with no columns would break the concat downstream.
        frame = _demands()
        out = _pipeline(frame)._create_other_demands(frame, {"nothing_matches"})
        assert out.empty
        assert list(out.columns) == ["grid", "node", "f", "t", "value"]


class TestMissingInput:
    @pytest.mark.parametrize("missing", ["grid", "node", "twh/year"])
    def test_a_missing_required_column_warns_and_returns_the_right_shape(self, missing):
        # Error policy: after logger init, log and continue with a safe default.
        logger = FakeLogger()
        frame = _demands().drop(columns=[missing])
        out = _pipeline(frame, logger=logger)._create_other_demands(frame, {"elec"})

        assert out.empty
        assert list(out.columns) == ["grid", "node", "f", "t", "value"]
        logger.assert_logged(missing, level="warn")

    def test_a_row_with_an_uncomputable_demand_is_skipped_not_fatal(self):
        logger = FakeLogger()
        frame = _demands(
            {"node": "FI_elec", "twh/year": "not a number"},
            {"node": "SE_elec", "twh/year": 10.0},
        )
        out = _pipeline(frame, bb_timeseries_length=1, logger=logger)._create_other_demands(
            frame, {"elec"}
        )

        # The good row still gets its profile.
        assert set(out["node"]) == {"SE_elec"}
