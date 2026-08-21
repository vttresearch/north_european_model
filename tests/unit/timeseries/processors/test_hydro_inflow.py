"""Inflow is energy per period converted to average power, plus the week-53 rule.

The PECD weekly file carries 53 weeks for most zone-years. The extra anchor at
Dec 28 used to read position 51 -- week 52 -- so the real week-53 figure was
never used and week 52 was repeated one day after its own anchor. For 246 of the
422 zone-years that carry a week 53, those are different numbers.

The anchor cannot simply be dropped when week 53 is blank. Week 52 sits at
Dec 27 12:00 and the next year's week 1 at Jan 4 12:00, a 192 h gap, while the
hourly interpolation reaches 84 h from each side. Without something at Dec 28
there is a 24 h hole, and the GDX gate turns a hole into zeros.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.timeseries.processors.hydro_inflow_MAF2019 import hydro_inflow_MAF2019
from tests._common.fixtures import FakeLogger
from tests._common.processor_contract import assert_processor_conforms

WEEKLY_HEADER = "Cumulated inflow into reservoirs per week in GWh"
PS_HEADER = "Cumulated NATURAL inflow into the pump-storage reservoirs per week in GWh"
DAILY_HEADER = "Run of River Hydro Generation in GWh per day"

START_YEAR = 2014
END_YEAR = 2015

#: Distinct per week so a wrong row is visible rather than coincidentally equal.
def weekly_value(week: int) -> float:
    return 100.0 + week


def write_inputs(
    folder: Path,
    zones=("XX00",),
    weeks_per_year: int = 53,
    blank_week53_years=(),
    skip_weeks=(),
) -> Path:
    """Write synthetic weekly and daily PECD files into `folder`.

    `skip_weeks` omits those week *rows* entirely, which is how SE01 1991 and
    SE02 1985 look in the real file.
    """
    folder.mkdir(parents=True, exist_ok=True)

    weekly_rows = []
    for zone in zones:
        for year in range(START_YEAR, END_YEAR + 1):
            for week in range(1, weeks_per_year + 1):
                if week in skip_weeks:
                    continue
                blank = week == 53 and year in blank_week53_years
                weekly_rows.append({
                    "zone": zone, "week": week, "year": year,
                    WEEKLY_HEADER: None if blank else weekly_value(week),
                    PS_HEADER: 0.0,
                })
    pd.DataFrame(weekly_rows).to_csv(
        folder / "PECD-hydro-weekly-inflows.csv", index=False
    )

    daily_rows = []
    for zone in zones:
        for year in range(START_YEAR, END_YEAR + 1):
            for day in range(1, 366):
                daily_rows.append({
                    "zone": zone, "Day": day, "week": (day - 1) // 7 + 1,
                    "year": year, DAILY_HEADER: 24.0,
                })
    pd.DataFrame(daily_rows).to_csv(
        folder / "PECD-hydro-daily-ror-generation.csv", index=False
    )
    return folder


def make_processor(folder: Path, zones=("XX00",)) -> tuple[hydro_inflow_MAF2019, FakeLogger]:
    logger = FakeLogger()
    processor = hydro_inflow_MAF2019(
        input_folder=str(folder),
        country_codes=list(zones),
        start_year=START_YEAR,
        end_year=END_YEAR,
        logger=logger,
    )
    return processor, logger


def weekly_frame(folder: Path) -> pd.DataFrame:
    return pd.read_csv(folder / "PECD-hydro-weekly-inflows.csv")


def at(df: pd.DataFrame, column: str, when: pd.Timestamp) -> float:
    return float(df.loc[when, column])


class TestUnitConversion:
    def test_weekly_gwh_becomes_average_mw(self, tmp_path):
        """GWh over a week -> MWh/h: times 1000, divided by 168. No capacity involved."""
        folder = write_inputs(tmp_path / "ts")
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))
        # Week 1 of START_YEAR is anchored at Jan 4 12:00.
        got = at(out, "XX00_reservoir", pd.Timestamp(START_YEAR, 1, 4, 12))
        assert got == pytest.approx(1000 * weekly_value(1) / 168)

    def test_daily_gwh_becomes_average_mw(self, tmp_path):
        folder = write_inputs(tmp_path / "ts")
        processor, _ = make_processor(folder)
        daily = pd.read_csv(folder / "PECD-hydro-daily-ror-generation.csv")
        out = processor._process_ror_inflows(daily)
        got = at(out, "XX00_ror", pd.Timestamp(START_YEAR, 6, 1, 12))
        assert got == pytest.approx(1000 * 24.0 / 24)


class TestWeek53:
    def test_the_dec_28_anchor_carries_week_53_not_week_52(self, tmp_path):
        folder = write_inputs(tmp_path / "ts")
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))

        got = at(out, "XX00_reservoir", pd.Timestamp(START_YEAR, 12, 28, 12))
        assert got == pytest.approx(1000 * weekly_value(53) / 168)
        assert got != pytest.approx(1000 * weekly_value(52) / 168)

    def test_week_52_keeps_its_own_anchor(self, tmp_path):
        """Dec 27 is week 52's anchor; the week-53 fix must not disturb it."""
        folder = write_inputs(tmp_path / "ts")
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))
        got = at(out, "XX00_reservoir", pd.Timestamp(START_YEAR, 12, 27, 12))
        assert got == pytest.approx(1000 * weekly_value(52) / 168)

    def test_a_blank_week_53_is_interpolated_across_the_boundary(self, tmp_path):
        """Not zero, and not a repeat of week 52.

        A blank cell stays blank all the way to complete_native_grid, which is
        what lets it be told apart from a recorded zero. The processor used to
        fillna(0) on read, which turned every source gap into a convincing value
        and made this distinction impossible.
        """
        folder = write_inputs(tmp_path / "ts", blank_week53_years=(START_YEAR,))
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))

        got = at(out, "XX00_reservoir", pd.Timestamp(START_YEAR, 12, 28, 12))
        week52 = 1000 * weekly_value(52) / 168
        week1_next = 1000 * weekly_value(1) / 168
        # Dec 28 is one day along an eight-day span from Dec 27 to Jan 4.
        expected = week52 + (week1_next - week52) / 8

        assert got != 0.0
        assert got == pytest.approx(expected)

    def test_a_blank_week_53_at_the_very_end_carries_week_52_forward(self, tmp_path):
        """A trailing single-slot gap has nothing to interpolate towards.

        It is still a single-slot gap, so it is still repaired rather than
        escalated -- by carrying the previous week, which is what the old code did
        unconditionally and is defensible for exactly one step.
        """
        folder = write_inputs(tmp_path / "ts", blank_week53_years=(END_YEAR,))
        processor, logger = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))

        got = at(out, "XX00_reservoir", pd.Timestamp(END_YEAR, 12, 28, 12))
        assert got == pytest.approx(1000 * weekly_value(52) / 168)
        logger.assert_clean()

    def test_a_year_with_no_week_53_row_is_treated_like_a_blank_one(self, tmp_path):
        """An absent row and an empty cell are the same situation, so same handling.

        A 52-week year got no Dec 28 anchor at all, leaving the 192 h boundary gap
        that the 84 h interpolation cannot bridge -- a 24 h hole the GDX gate turns
        into zero inflow. Five zone-years of the shipped data and the default
        country list are affected.
        """
        folder = write_inputs(tmp_path / "ts", weeks_per_year=52)
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))

        boundary = out.loc[
            pd.Timestamp(START_YEAR, 12, 20):pd.Timestamp(END_YEAR, 1, 10),
            "XX00_reservoir",
        ]
        assert not boundary.isna().any()

        week52 = 1000 * weekly_value(52) / 168
        week1_next = 1000 * weekly_value(1) / 168
        got = at(out, "XX00_reservoir", pd.Timestamp(START_YEAR, 12, 28, 12))
        assert got == pytest.approx(week52 + (week1_next - week52) / 8)


class TestWeeklyGridIsCompletedFirst:
    """Complete the weekly series, then cast it to hourly -- not the reverse.

    At weekly resolution a missing week is one step from its neighbours. Scattered
    onto an hourly index it is 168 steps, and whether it gets bridged depends on
    an interpolation limit. Filling first makes the hourly pass mechanical.
    """

    def test_a_missing_interior_week_does_not_shift_the_rest_of_the_year(self, tmp_path):
        """Anchors come from the week number, not the row's position.

        SE01 1991 has no week 7 and SE02 1985 no weeks 6 or 7. Placing rows by
        position pulled every later week one or two weeks early for the whole
        rest of the year -- right values, wrong hours, which no value check sees.
        """
        folder = write_inputs(tmp_path / "ts", skip_weeks=(7,))
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))

        # Week 8 belongs at Jan 4 + 7 weeks, regardless of week 7 being absent.
        week8_at = pd.Timestamp(START_YEAR, 1, 4, 12) + pd.Timedelta(7 * 7, unit="D")
        assert at(out, "XX00_reservoir", week8_at) == pytest.approx(
            1000 * weekly_value(8) / 168
        )

    def test_the_missing_interior_week_is_interpolated_not_left_empty(self, tmp_path):
        folder = write_inputs(tmp_path / "ts", skip_weeks=(7,))
        processor, logger = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))

        week7_at = pd.Timestamp(START_YEAR, 1, 4, 12) + pd.Timedelta(6 * 7, unit="D")
        got = at(out, "XX00_reservoir", week7_at)
        expected = (1000 * weekly_value(6) / 168 + 1000 * weekly_value(8) / 168) / 2
        assert got == pytest.approx(expected)
        logger.assert_logged("single-week gap(s) interpolated", level="info")

    def test_the_report_says_how_much_of_the_series_was_filled(self, tmp_path):
        """A missing week means something different in a 20 TWh catchment than a 0.2 TWh one."""
        folder = write_inputs(tmp_path / "ts", skip_weeks=(7,))
        processor, logger = make_processor(folder)
        processor._process_reservoir_inflows(weekly_frame(folder))
        assert logger.matching("TWh/year", level="info")

    def test_a_complete_grid_is_not_reported_at_all(self, tmp_path):
        folder = write_inputs(tmp_path / "ts")
        processor, logger = make_processor(folder)
        processor._process_reservoir_inflows(weekly_frame(folder))
        logger.assert_not_logged("filled by interpolation")
        logger.assert_clean()


class TestNoHolesAcrossTheYearBoundary:
    """The anchor exists to keep the 192 h boundary gap bridgeable."""

    @pytest.mark.parametrize("blank_years", [(), (START_YEAR,)])
    def test_the_hourly_series_has_no_gap_at_new_year(self, tmp_path, blank_years):
        folder = write_inputs(tmp_path / "ts", blank_week53_years=blank_years)
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))

        window = out.loc[
            pd.Timestamp(START_YEAR, 12, 20):pd.Timestamp(END_YEAR, 1, 10),
            "XX00_reservoir",
        ]
        assert not window.isna().any(), "a hole here becomes zeros at the GDX gate"

    def test_the_whole_series_has_no_missing_hours(self, tmp_path):
        folder = write_inputs(tmp_path / "ts")
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))
        assert not out["XX00_reservoir"].isna().any()


class TestOutputContract:
    def test_the_output_meets_the_processor_contract(self, tmp_path):
        folder = write_inputs(tmp_path / "ts")
        assert_processor_conforms(
            hydro_inflow_MAF2019,
            dimensions=["grid", "node", "f", "t"],
            input_folder=str(folder),
            country_codes=["XX00"],
            start_year=START_YEAR,
            end_year=END_YEAR,
        )

    def test_inflow_is_declared_non_negative(self):
        assert hydro_inflow_MAF2019.value_sign == "non_negative"

    def test_it_needs_no_source_data(self):
        """Unlike the storage limits, this one is a pure unit conversion."""
        assert hydro_inflow_MAF2019.requires_source_data == ()
