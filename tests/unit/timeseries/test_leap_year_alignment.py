"""Leap years, and why they are currently harmless.

Electricity demand arrives on a standardised 365-day calendar; temperature,
hydro and PECD arrive on the real one. So in a leap year the sources genuinely
disagree about how many days exist, and ``elec_demand_TYNDP2024`` reconciles
that by shifting its profile and duplicating the last day.

None of it shows in practice, and the reason is worth being precise about,
because it is not the window. A 365-day window runs Jan 1 to Dec 30 23:00, and a
series that skipped Feb 29 runs out a day early -- 8736 rows against 8760. The
alignment holds only because the electricity processor expands its standardised
year onto the real calendar first. That expansion is load-bearing, not tidying.

Verified against the real build: all seven parameters produce exactly 8760 rows
per group in leap year 1984.

What the window does contain is the *last* day: a leap year's Dec 31 falls
outside a 365-day window, uniformly for every series. That containment depends
on the window being no longer than a year.
Past 365 days a window spans the leap day itself, and any series lacking it runs
a day ahead of the others for the whole remainder -- silently, because t-labels
are assigned by row position. ``config_OT2030-continuous5y.ini`` uses
``bb_timeseries_length = 365*5``.
"""

import pandas as pd
import pytest

from src.timeseries.timeseries_helpers import split_timeseries_to_climate_windows

DIMS = ["grid", "node", "f", "t"]


def _series(times, node):
    """Value encodes day-of-year, so a one-day offset is readable off the value."""
    return pd.DataFrame(
        {
            "grid": "elec",
            "node": node,
            "time": times,
            "value": [float(t.dayofyear) for t in times],
        }
    )


def _split(df, *, start, days, year):
    return split_timeseries_to_climate_windows(
        df,
        bb_parameter_dimensions=DIMS,
        bb_ts_start=start,
        bb_ts_length=days,
        valid_climate_years=[year],
    )[year]


def _without_leap_day(times):
    return times[~((times.month == 2) & (times.day == 29))]


LEAP_YEAR_HOURS = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")


class TestAYearLongWindowOnlyPartlyContainsIt:
    def test_a_leap_year_is_truncated_to_365_days(self):
        out = _split(_series(LEAP_YEAR_HOURS, "n"), start="01-01", days=365, year=2016)
        assert len(LEAP_YEAR_HOURS) == 8784
        assert len(out) == 365 * 24

    def test_the_dropped_hours_are_the_last_day_not_the_leap_day(self):
        """Feb 29 is inside the window; Dec 31 is what falls off.

        Worth stating because it is the opposite of the intuitive reading --
        the leap day is modelled, the last day of the year is not.
        """
        out = _split(_series(LEAP_YEAR_HOURS, "n"), start="01-01", days=365, year=2016)
        days_present = set(out["value"])
        assert 60.0 in days_present      # Feb 29 is day 60 of a leap year
        assert 366.0 not in days_present  # Dec 31 is day 366, and is dropped

    def test_a_series_lacking_the_leap_day_is_24_hours_short(self):
        """So the window alone does NOT contain the problem.

        The window runs Jan 1 to Dec 30 23:00. A series on the real calendar
        fills it exactly; one that skipped Feb 29 runs out a day early and
        contributes 8736 rows. Uneven coverage, in a plain 365-day window.

        This is what makes the expansion in elec_demand_TYNDP2024 load-bearing
        rather than tidying: without it, the standardised electricity calendar
        would be 24 hours short of temperature and hydro in every leap year.
        """
        with_leap = _series(LEAP_YEAR_HOURS, "real_calendar")
        without = _series(_without_leap_day(LEAP_YEAR_HOURS), "standard_calendar")
        out = _split(
            pd.concat([with_leap, without], ignore_index=True),
            start="01-01", days=365, year=2016,
        )

        counts = dict(out.groupby("node", observed=True).size())
        assert counts["real_calendar"] == 365 * 24
        assert counts["standard_calendar"] == 365 * 24 - 24

    def test_and_runs_a_day_ahead_from_the_leap_day_onward(self):
        """Length is only half of it; the meaning drifts too.

        Having skipped Feb 29, the standardised series is a day further into the
        year at any given row position -- so t008000 is Nov 30 for one and
        Dec 1 for the other.
        """
        with_leap = _series(LEAP_YEAR_HOURS, "real_calendar")
        without = _series(_without_leap_day(LEAP_YEAR_HOURS), "standard_calendar")
        out = _split(
            pd.concat([with_leap, without], ignore_index=True),
            start="01-01", days=365, year=2016,
        )

        late = out[out["t"] == "t008000"]
        values = dict(zip(late["node"], late["value"]))
        assert values["standard_calendar"] == values["real_calendar"] + 1


class TestLongerWindowsLoseTheContainment:
    """``bb_timeseries_length = 365*5`` puts a window across the leap day."""

    SPAN = pd.date_range("2015-12-01", "2016-12-31 23:00", freq="h")

    def _both(self):
        return pd.concat(
            [
                _series(self.SPAN, "real_calendar"),
                _series(_without_leap_day(self.SPAN), "standard_calendar"),
            ],
            ignore_index=True,
        )

    def test_the_two_series_end_up_with_different_row_counts(self):
        out = _split(self._both(), start="12-01", days=380, year=2015)
        counts = dict(out.groupby("node", observed=True).size())
        assert counts["real_calendar"] - counts["standard_calendar"] == 24

    def test_and_diverge_from_the_leap_day_onward(self):
        # Everything before Feb 29 still agrees; everything after is offset.
        out = _split(self._both(), start="12-01", days=380, year=2015)
        real = out[out["node"] == "real_calendar"].reset_index(drop=True)
        standard = out[out["node"] == "standard_calendar"].reset_index(drop=True)

        shared = min(len(real), len(standard))
        disagreements = [
            i for i in range(shared) if real["value"][i] != standard["value"][i]
        ]
        assert disagreements, "expected the two calendars to diverge"
        assert real["value"][disagreements[0]] == 60.0   # Feb 29

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "a window longer than 365 days spans the leap day, and a series "
            "lacking it runs 24 h ahead of the others for the remainder"
        ),
    )
    def test_series_should_stay_aligned_across_a_leap_day(self):
        """Open, and known: electricity data comes in standard years.

        The reconciliation exists (elec_demand_TYNDP2024 shifts its profile and
        duplicates the last day), so the built parameters currently agree --
        all seven produce 8760 rows per group in leap year 1984. What is missing
        is anything that *checks* it. A processor added later, or a source that
        changes its calendar, breaks the alignment with no signal at all,
        because t-labels come from row position rather than from the timestamp.

        Where the check belongs was decided in phase 5: the verifier, not the
        pipeline. Whether a processor's calendar lines up with the others is a
        property of the processor and its source, so the answer is identical for
        every build -- and users generate 20-50 input folders per scenario
        sweep, where ~16 s each buys nothing new.
        """
        out = _split(self._both(), start="12-01", days=380, year=2015)
        counts = set(out.groupby("node", observed=True).size())
        assert len(counts) == 1
