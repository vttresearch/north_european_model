"""Mapping TYNDP's 365-day calendar onto real years, including leap years.

The source data is a standardised 365-day year: every year has the same 8760
rows regardless of what the calendar actually did. Turning that into real
timestamps is the highest bug-risk transform in the processors, because a
one-day error propagates through the whole remainder of the year and looks
entirely plausible in a plot.

The documented strategy (docstring at :335-348) inserts the leap day by shifting
rather than interpolating:

    std day 1-59    (Jan 1 - Feb 28)  ->  same date
    std day 60      (Mar 1 std)       ->  Feb 29
    std day 61-365  (Mar 2 - Dec 31)  ->  Mar 1 - Dec 30
    std day 365                       ->  ALSO duplicated to Dec 31

Assertions here are structural wherever possible -- hour counts, ordering,
uniqueness, which real date a given standard date lands on -- rather than
pinned demand values, since the values are the fixture's business.
"""

import importlib.util

import pandas as pd
import pytest

from tests._common.fixtures import FakeLogger

_spec = importlib.util.spec_from_file_location(
    "elec_demand_TYNDP2024", "src/timeseries/processors/elec_demand_TYNDP2024.py"
)
_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_module)
ElecDemand = getattr(_module, "elec_demand_TYNDP2024")


def _processor(start_year=2014, end_year=2014):
    return ElecDemand(
        input_folder=".",
        country_codes=["FI"],
        start_year=start_year,
        end_year=end_year,
        df_annual_demands=pd.DataFrame(),
        scenario_year=2030,
        demand_grid="elec",
        logger=FakeLogger(),
    )


def _standard_year(year: int) -> pd.DataFrame:
    """One row per hour of a standardised 365-day year, value = day-of-year."""
    rows = []
    for day_of_year, date in enumerate(
        pd.date_range("2001-01-01", "2001-12-31", freq="D"), start=1
    ):
        for hour in range(24):
            rows.append(
                {
                    "year": year,
                    "month": date.month,
                    "day": date.day,
                    "hour": hour,
                    "FI": float(day_of_year),
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def common_year():
    return _processor(2014, 2014).process_datetime_index(_standard_year(2014))


@pytest.fixture(scope="module")
def leap_year():
    return _processor(2016, 2016).process_datetime_index(_standard_year(2016))


class TestCommonYear:
    def test_produces_one_row_per_hour(self, common_year):
        assert len(common_year) == 8760

    def test_dates_map_straight_through(self, common_year):
        # value == standard day-of-year, so this checks the mapping, not the data
        assert common_year.loc["2014-01-01 00:00", "FI"] == 1.0
        assert common_year.loc["2014-03-01 00:00", "FI"] == 60.0
        assert common_year.loc["2014-12-31 23:00", "FI"] == 365.0

    def test_the_index_is_ordered_and_unique(self, common_year):
        assert common_year.index.is_monotonic_increasing
        assert common_year.index.is_unique


class TestLeapYear:
    def test_produces_one_row_per_hour_of_a_366_day_year(self, leap_year):
        assert len(leap_year) == 8784

    def test_february_28_is_unaffected(self, leap_year):
        # Everything before the insertion point maps 1:1.
        assert leap_year.loc["2016-02-28 00:00", "FI"] == 59.0

    def test_february_29_takes_the_standard_march_1_data(self, leap_year):
        """The inserted day, filled by shifting rather than interpolating.

        Standard day 60 is Mar 1; in a leap year that offset from Jan 1 lands on
        Feb 29, so the leap day gets Mar 1's profile.
        """
        assert leap_year.loc["2016-02-29 00:00", "FI"] == 60.0

    def test_march_1_takes_the_standard_march_2_data(self, leap_year):
        # Everything after the insertion is shifted one day later.
        assert leap_year.loc["2016-03-01 00:00", "FI"] == 61.0

    def test_december_30_takes_the_last_standard_day(self, leap_year):
        assert leap_year.loc["2016-12-30 00:00", "FI"] == 365.0

    def test_december_31_is_duplicated_from_it(self, leap_year):
        """The shift leaves one day short at the end, so day 365 is reused.

        Without this the year would stop on Dec 30 and the last 24 hours would
        be empty -- which the final reindex would turn into NaN rather than an
        error.
        """
        assert leap_year.loc["2016-12-31 00:00", "FI"] == 365.0

    def test_the_index_is_ordered_and_unique(self, leap_year):
        # Chronological order is what the shifting strategy exists to preserve.
        assert leap_year.index.is_monotonic_increasing
        assert leap_year.index.is_unique

    def test_no_hour_of_the_year_is_missing(self, leap_year):
        expected = pd.date_range("2016-01-01", "2016-12-31 23:00", freq="h")
        assert leap_year.index.equals(expected)

    def test_every_hour_carries_data(self, leap_year):
        # A gap here would become a NaN row and, downstream, a zero demand hour.
        assert not leap_year["FI"].isna().any()


class TestCoverageIsAlwaysComplete:
    """Why the climate-window slicing never sees an uneven group from here.

    The transform ends with a reindex onto a full hourly range, so a country
    missing hours in the source still comes out with one row per hour -- NaN
    rather than absent. That keeps every group the same length, which is what
    the row-position t-labelling downstream depends on.
    """

    def test_missing_source_hours_become_na_rows_rather_than_absent_ones(self):
        frame = _standard_year(2014)
        frame = frame.drop(frame.index[100:124])          # remove a whole day

        out = _processor(2014, 2014).process_datetime_index(frame)

        assert len(out) == 8760                # length preserved
        assert out["FI"].isna().any()          # the gap is visible as NaN

    def test_a_multi_year_range_covers_every_hour_of_every_year(self):
        frames = pd.concat([_standard_year(2015), _standard_year(2016)], ignore_index=True)
        out = _processor(2015, 2016).process_datetime_index(frames)

        assert len(out) == 8760 + 8784
        assert out.index.is_unique
        assert out.index.is_monotonic_increasing

    def test_duplicate_timestamps_are_reported_and_the_first_kept(self):
        logger = FakeLogger()
        processor = _processor(2014, 2014)
        processor.logger = logger
        frame = _standard_year(2014)
        duplicated = pd.concat([frame, frame.head(1)], ignore_index=True)

        out = processor.process_datetime_index(duplicated)

        assert out.index.is_unique
        logger.assert_logged("duplicate timestamps", level="warn")


class TestRejections:
    def test_an_empty_frame_is_rejected(self):
        # Raised, not logged: this runs inside the processor, before
        # ProcessorRunner's contract check, and an empty input is a caller bug.
        with pytest.raises(ValueError, match="empty"):
            _processor().process_datetime_index(pd.DataFrame())
