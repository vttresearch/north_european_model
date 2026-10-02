"""Forecast branches: the shape of the climate years, the energy their quantile asks for.

The rules this module pins:

- **Shape.** At every t, branch fNN of a series is a quantile of the realized f00
  values at that same t, across the windows ``split_timeseries_to_climate_windows``
  writes. So a forecast t and a realized t name the same hour, whatever the
  start date, the length and the leap years inside the window.
- **Energy.** Which quantile is solved per series: the branch carries, over its
  own length, Q_p + (mean - Q_0.5) * (1 - |1 - 2p|) of the climate years' energy
  over that length. So 0.5 is the mean for every series, however skewed, and 0
  and 1 are the years' own lowest and highest.

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
    _solve_per_hour_quantiles,
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


def _skewed(seed=11) -> pd.DataFrame:
    """'wind': skewed hourly values whose level changes year by year; 'flat': symmetric."""
    rng = np.random.default_rng(seed)
    level = pd.Series(rng.uniform(0.6, 1.6, size=5), index=range(2014, 2019))
    wind = rng.exponential(1.0, size=len(HOURS)) * level[HOURS.year].to_numpy()
    flat = 10.0 + rng.normal(size=len(HOURS)) * level[HOURS.year].to_numpy()
    return _frame({"wind": wind, "flat": flat})


def _forecasts(df, *, start="01-01", days=365, years=(2014, 2015, 2016, 2017), quantiles=None,
               branch_days=None):
    return calculate_climatological_forecasts(
        df,
        bb_parameter_dimensions=DIMS,
        energy_quantiles=quantiles or QUANTILES,
        bb_ts_start=start,
        bb_ts_length=days,
        valid_climate_years=list(years),
        branch_days=branch_days,
        round_precision=None,
    )


def _branch(out, node, f):
    rows = out[(out["node"] == node) & (out["f"] == f)]
    return rows.set_index("t")["value"].sort_index()


def _windows(df, node, *, start="01-01", days=365, years=(2014, 2015, 2016, 2017)):
    """The realized windows of one series, as (window, hour)."""
    realized = split_timeseries_to_climate_windows(
        df, bb_parameter_dimensions=DIMS, bb_ts_start=start,
        bb_ts_length=days, valid_climate_years=list(years),
    )
    return np.stack([
        frame[frame["node"] == node].sort_values("t")["value"].to_numpy()
        for frame in realized.values()
    ])


def _rule(sums, p):
    """The energy a branch at p carries, along axis 0: the module's rule written out."""
    lift = (sums.mean(0) - np.quantile(sums, 0.5, axis=0)) * (1 - abs(1 - 2 * p))
    return np.quantile(sums, p, axis=0) + lift


def _window_sums(daily, days):
    """Sums over `days` from every start day, wrapping. daily: (..., D)."""
    n = daily.shape[-1]
    return np.stack([np.take(daily, range(d, d + days), axis=-1, mode="wrap").sum(-1)
                     for d in range(n)], axis=-1)


class TestEveryHourIsAQuantileOfThatHourAcrossTheWindows:
    def test_each_series_takes_one_quantile_for_all_its_hours(self):
        df = _skewed()
        years = [2014, 2015, 2016, 2017]
        result = _forecasts(df, start="02-01", years=years)
        q = result.per_hour_quantiles.set_index("node")

        for node in ("wind", "flat"):
            windows = _windows(df, node, start="02-01", years=years)
            for f in QUANTILES:
                expected = np.quantile(windows, q.loc[node, f], axis=0)
                np.testing.assert_allclose(
                    _branch(result.frame, node, f).to_numpy(), expected, rtol=1e-12, atol=1e-12,
                    err_msg=f"{node} {f}",
                )

    def test_every_series_t_and_branch_gets_exactly_one_row(self):
        out = _forecasts(_ramp(), start="02-01", years=[2014, 2015]).frame
        assert len(out) == 2 * 365 * 24 * len(QUANTILES)
        assert not out.duplicated(["node", "t", "f"]).any()
        assert list(out.columns) == DIMS + ["value"]


class TestTheEnergyIsWhatTheQuantileAsks:
    def test_half_is_the_mean_of_every_series_however_skewed(self):
        df = _skewed()
        result = _forecasts(df)
        for node in ("wind", "flat"):
            mean = _windows(df, node).sum(-1).mean()
            assert _branch(result.frame, node, "f01").sum() == pytest.approx(mean, rel=1e-9)

        # Which per-hour quantile that takes depends on the series: a per-hour
        # median of the skewed one carries less than its mean.
        q = result.per_hour_quantiles.set_index("node")["f01"]
        assert q["wind"] > 0.55
        assert q["flat"] == pytest.approx(0.5, abs=0.05)

    def test_another_value_is_the_years_own_quantile_lifted_less_and_less(self):
        df = _skewed()
        result = _forecasts(df)
        totals = _windows(df, "wind").sum(-1)
        for f, p in (("f02", 0.1), ("f03", 0.9)):
            assert _branch(result.frame, "wind", f).sum() == pytest.approx(_rule(totals, p), rel=1e-9)

    def test_zero_is_the_lowest_climate_year(self):
        # Levels 0, 10, 11, 11.5 a year: the lowest year is lowest at every hour,
        # so energy quantile 0 is that year, hour by hour.
        levels = {2014: 0.0, 2015: 10.0, 2016: 11.0, 2017: 11.5}
        df = _frame({"a": np.array([levels.get(y, 0.0) for y in HOURS.year]) + 1.0})
        result = _forecasts(df, quantiles={"f01": 0.5, "f02": 0.0}, days=31)
        assert (_branch(result.frame, "a", "f02") == 1.0).all()
        assert result.unreachable == {"f01": [], "f02": []}

    def test_a_branch_s_energy_is_measured_over_its_own_length(self):
        # Over 3-day windows from every start day, wrapping as Backbone's
        # circulation does: the branch's 3-day energies, summed, are what the
        # years' 3-day energies ask for, summed.
        df = _skewed()
        result = _forecasts(df, quantiles={"f01": 0.5, "f02": 0.1}, branch_days={"f02": 3})

        daily = _windows(df, "wind").reshape(4, 365, 24).sum(-1)
        realized = _window_sums(daily, 3)                                  # window, start day
        asked = _rule(realized, 0.1)
        branch = _window_sums(_branch(result.frame, "wind", "f02").to_numpy().reshape(365, 24).sum(-1), 3)
        assert branch.sum() == pytest.approx(asked.sum(), rel=1e-9)

        # And it is not the whole-window answer: a short branch is a different branch.
        whole = _forecasts(df, quantiles={"f01": 0.5, "f02": 0.1})
        assert _branch(whole.frame, "wind", "f02").sum() != pytest.approx(
            _branch(result.frame, "wind", "f02").sum(), rel=1e-3)

    def test_a_low_value_is_the_hard_direction_for_demand_too(self):
        # Demand is a negative influx: a low energy quantile is more demand.
        df = _skewed()
        df["value"] = -df["value"]
        result = _forecasts(df)
        mean = _windows(df, "wind").sum(-1).mean()
        assert _branch(result.frame, "wind", "f02").sum() < mean < _branch(result.frame, "wind", "f03").sum()

    def test_windows_that_are_alike_give_the_series_itself(self):
        hour_of_year = ((HOURS.dayofyear - 1) * 24 + HOURS.hour).to_numpy()
        values = np.sin(hour_of_year / 50.0)
        df = _frame({"a": values})
        result = _forecasts(df, years=[2014, 2015, 2017])  # no leap day inside a window
        np.testing.assert_allclose(_branch(result.frame, "a", "f02").to_numpy(), values[:8760], atol=1e-12)
        assert result.per_hour_quantiles.loc[0, "f02"] == 0.1
        assert result.unreachable == {f: [] for f in QUANTILES}

    def test_a_target_no_per_hour_quantile_can_carry_takes_the_nearest_and_says_so(self):
        # Three windows of two hours. Quantile 0 carries 1 + 2 = 3 and quantile 1
        # carries 5 + 6 = 11; a target outside that takes the nearest end.
        windows = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])[:, None, :]
        ordered = np.sort(windows, axis=0)
        n_valid = np.full((1, 2), 3)
        for target, end in ((2.0, 0.0), (12.0, 1.0)):
            q, beyond = _solve_per_hour_quantiles(ordered, n_valid, np.array([target]), fallback=0.5)
            assert q[0] == end and beyond[0]
        q, beyond = _solve_per_hour_quantiles(ordered, n_valid, np.array([7.0]), fallback=0.5)
        assert q[0] == pytest.approx(0.5) and not beyond[0]


class TestNoStepThatTheRealizedWindowsDoNotHave:
    """27 Dec for 410 days: two New Years inside the window, and Feb 29 2016."""

    YEARS = [2014, 2015, 2016]

    def test_a_continuous_record_gives_a_continuous_forecast(self):
        out = _forecasts(_ramp(), start="12-27", days=410, years=self.YEARS).frame
        # Interpolated branches carry float noise of ~1e-12; a lost day on this
        # ramp would be a step of 24, and a repeated winter one of -8760.
        for f in QUANTILES:
            steps = np.diff(_branch(out, "a", f).to_numpy())
            np.testing.assert_allclose(steps, 1.0, atol=1e-9, err_msg=f)

    def test_the_second_winter_is_the_record_a_year_on_not_a_copy(self):
        # t000121 is Jan 1 and t008881 is 8760 hours later. A nominal calendar
        # year repeated onto the window gave both the same value.
        out = _forecasts(_ramp(), start="12-27", days=410, years=self.YEARS).frame
        central = _branch(out, "a", "f01")
        assert central["t008881"] - central["t000121"] == pytest.approx(8760.0, abs=1e-6)


class TestMissingValues:
    def test_an_hour_no_window_has_data_for_is_nan(self):
        df = _ramp()
        df.loc[df["time"] == pd.Timestamp("2015-06-01 12:00"), "value"] = np.nan
        out = _forecasts(df, years=[2015]).frame
        assert _branch(out, "a", "f01").isna().sum() == 1

    def test_a_gap_in_one_window_is_left_out_of_that_hour_s_sample(self):
        # Two windows, one missing the hour: the branch is the other window's value.
        df = _ramp()
        gap = pd.Timestamp("2015-06-01 12:00")
        df.loc[df["time"] == gap, "value"] = np.nan
        out = _forecasts(df, years=[2014, 2015]).frame

        t = f"t{int((gap - pd.Timestamp('2015-01-01')).total_seconds() // 3600) + 1:06d}"
        other = df.loc[(df["node"] == "a") & (df["time"] == gap - pd.DateOffset(years=1)), "value"]
        assert _branch(out, "a", "f01")[t] == other.item()

    def test_a_gap_does_not_read_as_a_dry_year(self):
        # A month missing from one window: its energy target treats the month as
        # the mean of the windows that have it, so the central branch stays at the
        # mean of the full windows rather than dropping by the missing month.
        df = _skewed()
        gap = (df["node"] == "wind") & (df["time"] >= "2015-03-01") & (df["time"] < "2015-04-01")
        full_mean = _windows(df, "wind").sum(-1).mean()
        df.loc[gap, "value"] = np.nan
        branch = _branch(_forecasts(df).frame, "wind", "f01")
        assert branch.sum() == pytest.approx(full_mean, rel=0.02)


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
