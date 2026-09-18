"""Forecast branches are quantiles across the realized climate windows.

The rule this module pins: at every t, forecast branch fNN is a quantile of the
realized f00 values at that same t, across the windows
``split_timeseries_to_climate_windows`` writes. So a forecast t and a realized t
name the same hour, whatever the start date, the length and the leap years
inside the window.

The branches used to be statistics of a nominal Jan-1 calendar year, mapped onto
the window by hour-of-year. That dropped Dec 31 from every leap year, so each
New Year inside a window was a step no realized year makes -- 15% of level in the
90% district-heat branch -- and a window longer than a year repeated its first
winter instead of carrying on. A ramp makes both visible: it is continuous
everywhere, so any step in its forecast is the method's own.
"""

import numpy as np
import pandas as pd
import pytest

from src.timeseries.timeseries_helpers import (
    _nan_quantiles,
    calculate_climatological_forecasts,
    split_timeseries_to_climate_windows,
)

DIMS = ["grid", "node", "f", "t"]
QUANTILES = {"f01": 0.5, "f02": 0.1, "f03": 0.9}

#: Five years with a leap year in the middle, so windows cross Feb 29 2016 and
#: the New Years either side of it.
HOURS = pd.date_range("2014-01-01", "2018-12-31 23:00", freq="h")


def _frame(values_by_node) -> pd.DataFrame:
    return pd.concat(
        [
            pd.DataFrame({"grid": "elec", "node": node, "time": HOURS, "value": values})
            for node, values in values_by_node.items()
        ],
        ignore_index=True,
    )


def _ramp() -> pd.DataFrame:
    """Hours since the record starts: continuous across every New Year and Feb 29."""
    ramp = np.arange(len(HOURS), dtype=float)
    return _frame({"a": ramp, "b": 1000.0 + ramp})


def _forecasts(df, *, start, days, years):
    return calculate_climatological_forecasts(
        df,
        bb_parameter_dimensions=DIMS,
        forecast_quantiles=QUANTILES,
        bb_ts_start=start,
        bb_ts_length=days,
        valid_climate_years=years,
        round_precision=None,
    )


def _branch(out, node, f):
    rows = out[(out["node"] == node) & (out["f"] == f)]
    return rows.set_index("t")["value"].sort_index()


class TestTheBranchesAreQuantilesOfTheRealizedWindows:
    def test_forecast_t_is_the_quantile_of_realized_t_across_the_windows(self):
        rng = np.random.default_rng(7)
        df = _frame({"a": rng.normal(size=len(HOURS)), "b": rng.normal(size=len(HOURS))})
        years = [2014, 2015, 2016, 2017]

        forecast = _forecasts(df, start="02-01", days=365, years=years)
        realized = pd.concat(
            split_timeseries_to_climate_windows(
                df, bb_parameter_dimensions=DIMS, bb_ts_start="02-01",
                bb_ts_length=365, valid_climate_years=years,
            ).values(),
            ignore_index=True,
        )

        realized["t"] = realized["t"].astype(str)
        forecast["t"] = forecast["t"].astype(str)
        for f, q in QUANTILES.items():
            expected = (
                realized.groupby(["node", "t"])["value"].quantile(q).rename("expected").reset_index()
            )
            both = expected.merge(forecast[forecast["f"] == f], on=["node", "t"], how="left")
            assert len(both) == 2 * 365 * 24
            # pandas interpolates by a different formula; they part at ~1e-17.
            np.testing.assert_allclose(both["value"], both["expected"], rtol=1e-12, atol=1e-12)

    def test_every_series_t_and_branch_gets_exactly_one_row(self):
        out = _forecasts(_ramp(), start="02-01", days=365, years=[2014, 2015])
        assert len(out) == 2 * 365 * 24 * len(QUANTILES)
        assert not out.duplicated(["node", "t", "f"]).any()
        assert list(out.columns) == DIMS + ["value"]


class TestNoStepThatTheRealizedWindowsDoNotHave:
    """27 Dec for 410 days: two New Years inside the window, and Feb 29 2016."""

    YEARS = [2014, 2015, 2016]

    def test_a_continuous_record_gives_a_continuous_forecast(self):
        out = _forecasts(_ramp(), start="12-27", days=410, years=self.YEARS)
        # Interpolated branches carry float noise of ~1e-12; a lost day on this
        # ramp would be a step of 24, and a repeated winter one of -8760.
        for f in QUANTILES:
            steps = np.diff(_branch(out, "a", f).to_numpy())
            np.testing.assert_allclose(steps, 1.0, atol=1e-9, err_msg=f)

    def test_the_second_winter_is_the_record_a_year_on_not_a_copy(self):
        # t000121 is Jan 1 and t008881 is 8760 hours later. A nominal calendar
        # year repeated onto the window gave both the same value.
        median = _branch(_forecasts(_ramp(), start="12-27", days=410, years=self.YEARS), "a", "f01")
        assert median["t008881"] - median["t000121"] == 8760.0


class TestMissingValues:
    def test_an_hour_no_window_has_data_for_is_nan(self):
        df = _ramp()
        df.loc[df["time"] == pd.Timestamp("2015-06-01 12:00"), "value"] = np.nan
        out = _forecasts(df, start="01-01", days=365, years=[2015])
        assert _branch(out, "a", "f01").isna().sum() == 1

    def test_a_gap_in_one_window_is_left_out_of_that_hour_s_sample(self):
        # Two windows, one missing the hour: the median is the other window's value.
        df = _ramp()
        gap = pd.Timestamp("2015-06-01 12:00")
        df.loc[df["time"] == gap, "value"] = np.nan
        out = _forecasts(df, start="01-01", days=365, years=[2014, 2015])

        t = f"t{int((gap - pd.Timestamp('2015-01-01')).total_seconds() // 3600) + 1:06d}"
        other = df.loc[(df["node"] == "a") & (df["time"] == gap - pd.DateOffset(years=1)), "value"]
        assert _branch(out, "a", "f01")[t] == other.item()


class TestNanQuantiles:
    def test_agrees_with_numpy(self):
        rng = np.random.default_rng(3)
        samples = rng.normal(size=(7, 4, 50))
        samples[rng.random(samples.shape) < 0.2] = np.nan
        samples[:, 0, 0] = np.nan  # a column with no value at all

        got = _nan_quantiles(samples, [0.1, 0.5, 0.9])
        with pytest.warns(RuntimeWarning):
            expected = np.nanquantile(samples, [0.1, 0.5, 0.9], axis=0)

        np.testing.assert_allclose(got, expected, rtol=1e-12, equal_nan=True)
        assert np.isnan(got[:, 0, 0]).all()

    def test_no_windows_at_all_is_all_nan(self):
        out = _nan_quantiles(np.empty((0, 2, 3)), [0.5])
        assert out.shape == (1, 2, 3) and np.isnan(out).all()
