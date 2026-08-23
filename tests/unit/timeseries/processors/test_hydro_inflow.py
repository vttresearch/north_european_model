"""Inflow is energy per period converted to average power, on a grid with no seams.

Week 53 is not read. The year has 52 whole weeks and a remainder of one or two
days, and the PECD file says different things about that remainder depending on
the zone: ten of the twenty-eight reservoir zones repeat week 52 verbatim, while
AT00 reports the remainder day itself -- 0.136 to 0.145 of its week 52 in all 36
years. Dividing the latter by 168 like a whole week put a one-day cliff at the
year change, collapsing AT00 inflow from 51.7 to 14.2 MWh/h and back up to 68 by
Jan 4, every year. Dropping the cell costs at most 0.33% of one node's annual
inflow and leaves the year change interpolated at the same weekly resolution as
everything else.

What that fixes is the anchor geometry, not the levels. How far apart week 52
and week 1 are is the source's own weather, so the tests here pin the geometry
and leave the size of the change to `_report_year_change_outliers`.
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

def weekly_value(week: int) -> float:
    """A seasonal shape: up to midsummer, back down, ending near where it began.

    Two properties matter and pull against each other. Every week must be
    distinct, so that a value landing on the wrong week is visible rather than
    coincidentally equal -- hence the tiny per-week tilt. And the year has to be
    roughly cyclic, because a monotone ramp would make week 52 -> week 1 the
    largest change in the series by construction, which no catchment does and
    which would make the year-change tests assert the fixture rather than the code.
    """
    return 100.0 + min(week, 53 - week) + week * 0.001


def write_inputs(
    folder: Path,
    zones=("XX00",),
    weeks_per_year: int = 53,
    skip_weeks=(),
    skip_days=(),
    week53_value: float | None = None,
    years: tuple[int, int] = (START_YEAR, END_YEAR),
    ror_value: float = 24.0,
) -> Path:
    """Write synthetic weekly and daily PECD files into `folder`.

    `skip_weeks` and `skip_days` omit those *rows* entirely, which is how SE01
    1991 and SE02 1985 look in the real file. `week53_value` overrides what the
    remainder cell says, so a test can prove it is not read whatever it holds.
    """
    folder.mkdir(parents=True, exist_ok=True)
    first, last = years

    weekly_rows = []
    for zone in zones:
        for year in range(first, last + 1):
            for week in range(1, weeks_per_year + 1):
                if week in skip_weeks:
                    continue
                value = weekly_value(week)
                if week == 53 and week53_value is not None:
                    value = week53_value
                weekly_rows.append({
                    "zone": zone, "week": week, "year": year,
                    WEEKLY_HEADER: value,
                    PS_HEADER: 0.0,
                })
    pd.DataFrame(weekly_rows).to_csv(
        folder / "PECD-hydro-weekly-inflows.csv", index=False
    )

    daily_rows = []
    for zone in zones:
        for year in range(first, last + 1):
            for day in range(1, 366):
                if day in skip_days:
                    continue
                daily_rows.append({
                    "zone": zone, "Day": day, "week": (day - 1) // 7 + 1,
                    "year": year, DAILY_HEADER: ror_value,
                })
    pd.DataFrame(daily_rows).to_csv(
        folder / "PECD-hydro-daily-ror-generation.csv", index=False
    )
    return folder


def nodedata(*nodes: str) -> pd.DataFrame:
    """A merged-nodedata frame naming the hydro nodes the model has."""
    return pd.DataFrame({
        "country": [n.split("_")[0] for n in nodes],
        "grid": [n.split("_", 1)[1] for n in nodes],
        "node": list(nodes),
        "upwardlimit": pd.array([1_000_000.0] * len(nodes), dtype="Float64"),
    })


def all_hydro_nodes(zones=("XX00",)) -> pd.DataFrame:
    return nodedata(*(f"{z}_{g}" for z in zones for g in ("reservoir", "psOpen", "ror")))


def make_processor(
    folder: Path, zones=("XX00",), df_nodedata=None
) -> tuple[hydro_inflow_MAF2019, FakeLogger]:
    logger = FakeLogger()
    processor = hydro_inflow_MAF2019(
        input_folder=str(folder),
        country_codes=list(zones),
        start_year=START_YEAR,
        end_year=END_YEAR,
        df_nodedata=all_hydro_nodes(zones) if df_nodedata is None else df_nodedata,
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


class TestWeek53IsNotRead:
    def test_a_week_53_value_does_not_reach_the_series(self, tmp_path):
        """The remainder cell is dropped, whatever it says.

        Written here as a value far from its neighbours, which is what AT00's
        one-day remainder amounts to after the /168 conversion. If it were still
        read, Dec 28 would sit near it instead of on the Dec 27 -> Jan 4 line.
        """
        folder = write_inputs(tmp_path / "ts", week53_value=1.0)
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))

        assert at(out, "XX00_reservoir", pd.Timestamp(START_YEAR, 12, 28, 12)) == pytest.approx(
            expected_year_change(pd.Timestamp(START_YEAR, 12, 28, 12))
        )

    def test_week_52_keeps_its_own_anchor(self, tmp_path):
        folder = write_inputs(tmp_path / "ts")
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))
        got = at(out, "XX00_reservoir", pd.Timestamp(START_YEAR, 12, 27, 12))
        assert got == pytest.approx(1000 * weekly_value(52) / 168)

    def test_a_year_shipped_without_week_53_gives_the_same_answer(self, tmp_path):
        """An absent remainder row and a present one must be indistinguishable."""
        with_53 = write_inputs(tmp_path / "a")
        without = write_inputs(tmp_path / "b", weeks_per_year=52)
        a = make_processor(with_53)[0]._process_reservoir_inflows(weekly_frame(with_53))
        b = make_processor(without)[0]._process_reservoir_inflows(weekly_frame(without))
        pd.testing.assert_series_equal(a["XX00_reservoir"], b["XX00_reservoir"])


def expected_year_change(when: pd.Timestamp) -> float:
    """Where the Dec 27 -> Jan 4 straight line sits at `when`.

    Both anchors are noon, so the span is 192 h in a common year. START_YEAR is
    2014, so no leap-year offset applies here.
    """
    start = pd.Timestamp(START_YEAR, 12, 27, 12)
    end = pd.Timestamp(END_YEAR, 1, 4, 12)
    week52 = 1000 * weekly_value(52) / 168
    week1 = 1000 * weekly_value(1) / 168
    share = (when - start) / (end - start)
    return week52 + (week1 - week52) * share


class TestYearChangeGeometry:
    """The year change is one straight weekly span, not a steeper one.

    This is the invariant the week-53 removal buys, and it is deliberately about
    shape rather than size: how far apart week 52 and week 1 are is weather.
    """

    @pytest.mark.parametrize("day,hour", [(27, 18), (29, 0), (31, 12), (1, 6), (4, 0)])
    def test_every_hour_of_the_span_is_on_the_straight_line(self, tmp_path, day, hour):
        folder = write_inputs(tmp_path / "ts")
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))

        year = START_YEAR if day > 5 else END_YEAR
        month = 12 if day > 5 else 1
        when = pd.Timestamp(year, month, day, hour)
        assert at(out, "XX00_reservoir", when) == pytest.approx(expected_year_change(when))

    def test_no_hour_at_the_year_change_steps_more_than_a_weekly_slope(self, tmp_path):
        """A 192 h span carrying one week's change is gentler than a 168 h one."""
        folder = write_inputs(tmp_path / "ts")
        processor, _ = make_processor(folder)
        series = processor._process_reservoir_inflows(weekly_frame(folder))["XX00_reservoir"]

        steps = series.diff().abs()
        year_change = steps.loc[
            pd.Timestamp(START_YEAR, 12, 27, 12):pd.Timestamp(END_YEAR, 1, 4, 12)
        ]
        interior = steps.loc[
            pd.Timestamp(START_YEAR, 2, 1):pd.Timestamp(START_YEAR, 11, 30)
        ]
        assert year_change.max() <= interior.max() + 1e-9


class TestSeriesEnds:
    """Linear interpolation does not extrapolate, so the ends need saying.

    Nothing downstream would catch a hole here: find_time_axis_defects checks
    that the rows exist rather than what is in them, and prepare_values_for_gdx
    turns a blank into a zero without a word.
    """

    def test_the_series_starts_on_january_1_not_on_the_first_anchor(self, tmp_path):
        folder = write_inputs(tmp_path / "ts")
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))

        stub = out.loc[
            pd.Timestamp(START_YEAR, 1, 1):pd.Timestamp(START_YEAR, 1, 4, 12), "XX00_reservoir"
        ]
        assert not stub.isna().any()
        assert stub.iloc[0] == pytest.approx(1000 * weekly_value(1) / 168)

    def test_the_series_reaches_december_31_of_a_leap_end_year(self, tmp_path):
        """Jan 4 + 357 d is Dec 26 in a leap year, leaving a 131 h stub."""
        folder = write_inputs(tmp_path / "ts", years=(2015, 2016))
        logger = FakeLogger()
        processor = hydro_inflow_MAF2019(
            input_folder=str(folder), country_codes=["XX00"],
            start_year=2015, end_year=2016,
            df_nodedata=all_hydro_nodes(), logger=logger,
        )
        out = processor._process_reservoir_inflows(weekly_frame(folder))

        stub = out.loc[pd.Timestamp(2016, 12, 26, 12):, "XX00_reservoir"]
        assert not stub.isna().any()
        assert out.index[-1] == pd.Timestamp(2016, 12, 31, 23)
        assert stub.iloc[-1] == pytest.approx(1000 * weekly_value(52) / 168)


class TestRefusedRunsAreNotBridged:
    def test_a_two_week_gap_stays_a_hole_at_hourly_resolution(self, tmp_path):
        """complete_native_grid refuses to invent it; the hourly cast must not either.

        The old limit-based interpolation left a ragged partial ramp into the gap.
        Filling only between adjacent anchors that both have a value leaves the
        whole span empty, which is what "this needs a decision" should look like.
        """
        folder = write_inputs(tmp_path / "ts", skip_weeks=(20, 21))
        processor, logger = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))

        first_missing = pd.Timestamp(START_YEAR, 1, 4, 12) + pd.Timedelta(19 * 7, unit="D")
        last_missing = pd.Timestamp(START_YEAR, 1, 4, 12) + pd.Timedelta(20 * 7, unit="D")
        gap = out.loc[first_missing:last_missing, "XX00_reservoir"]
        assert gap.isna().all()
        logger.assert_logged("no usable value", level="warn")


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
    """A hole here becomes zeros at the GDX gate, so there must not be one."""

    @pytest.mark.parametrize("weeks", [53, 52])
    def test_the_hourly_series_has_no_gap_at_new_year(self, tmp_path, weeks):
        folder = write_inputs(tmp_path / "ts", weeks_per_year=weeks)
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))

        window = out.loc[
            pd.Timestamp(START_YEAR, 12, 20):pd.Timestamp(END_YEAR, 1, 10),
            "XX00_reservoir",
        ]
        assert not window.isna().any()

    def test_the_whole_series_has_no_missing_hours(self, tmp_path):
        folder = write_inputs(tmp_path / "ts")
        processor, _ = make_processor(folder)
        out = processor._process_reservoir_inflows(weekly_frame(folder))
        assert not out["XX00_reservoir"].isna().any()


class TestRunOfRiverAnchors:
    """Days are placed by their Day number, like weeks are by their week number."""

    def test_a_missing_day_does_not_shift_the_rest_of_the_year(self, tmp_path):
        """Row position was the wrong key here too.

        The shipped file is a clean 1..365 block in all 1548 zone-years, so this
        guards a data refresh rather than a live defect.
        """
        folder = write_inputs(tmp_path / "ts", skip_days=(100,))
        processor, _ = make_processor(folder)
        daily = pd.read_csv(folder / "PECD-hydro-daily-ror-generation.csv")
        out = processor._process_ror_inflows(daily)

        # Day 200 belongs at Jan 1 + 199 days whether or not day 100 is there.
        when = pd.Timestamp(START_YEAR, 1, 1, 12) + pd.Timedelta(199, unit="D")
        assert at(out, "XX00_ror", when) == pytest.approx(1000 * 24.0 / 24)
        assert not out["XX00_ror"].isna().any()


class TestCoverageReport:
    def test_a_node_the_model_does_not_have_is_never_mentioned(self, tmp_path):
        """nodedata says what exists; the rest is not this processor's business."""
        folder = write_inputs(tmp_path / "ts")
        processor, logger = make_processor(folder, df_nodedata=nodedata("XX00_reservoir"))
        result = processor.process()

        assert set(result["node"].unique()) == {"XX00_reservoir"}
        logger.assert_not_logged("XX00_psOpen")
        logger.assert_not_logged("XX00_ror")

    def test_a_node_in_the_model_with_no_source_data_is_reported(self, tmp_path):
        """The psOpen column is all zeros here, which is how PECD says "none"."""
        folder = write_inputs(tmp_path / "ts")
        processor, logger = make_processor(folder, df_nodedata=nodedata("XX00_psOpen"))
        processor.process()
        logger.assert_logged("XX00_psOpen", level="info")

    def test_what_was_built_is_named(self, tmp_path):
        folder = write_inputs(tmp_path / "ts")
        processor, logger = make_processor(folder)
        processor.process()
        logger.assert_logged("Inflow built for", level="info")

    def test_unreadable_nodedata_builds_everything_and_says_so(self, tmp_path):
        """Failing open: a malformed workbook must not delete the hydro fleet."""
        folder = write_inputs(tmp_path / "ts")
        processor, logger = make_processor(folder, df_nodedata=pd.DataFrame())
        result = processor.process()
        assert "XX00_reservoir" in set(result["node"].unique())
        logger.assert_logged("cannot be determined", level="warn")


class TestYearChangeOutlierReport:
    def test_a_gross_outlier_is_named_and_left_alone(self, tmp_path):
        folder = write_inputs(tmp_path / "ts")
        processor, logger = make_processor(folder)
        weekly = weekly_frame(folder)
        # Make week 1 of END_YEAR far from week 52 of START_YEAR. The synthetic
        # weeks step by 1.0, so anything much larger is a gross outlier.
        mask = (weekly["week"] == 1) & (weekly["year"] == END_YEAR)
        weekly.loc[mask, WEEKLY_HEADER] = weekly_value(1) + 500.0
        out = processor._process_reservoir_inflows(weekly)

        logger.assert_logged("week 52 to week 1", level="info")
        # Reported, not smoothed: week 1 still carries what the source said.
        assert at(out, "XX00_reservoir", pd.Timestamp(END_YEAR, 1, 4, 12)) == pytest.approx(
            1000 * (weekly_value(1) + 500.0) / 168
        )

    def test_an_ordinary_year_change_is_not_reported(self, tmp_path):
        folder = write_inputs(tmp_path / "ts")
        processor, logger = make_processor(folder)
        processor._process_reservoir_inflows(weekly_frame(folder))
        logger.assert_not_logged("week 52 to week 1")


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
            df_nodedata=all_hydro_nodes(),
        )

    def test_inflow_is_declared_non_negative(self):
        assert hydro_inflow_MAF2019.value_sign == "non_negative"

    def test_it_reads_nodedata_to_learn_which_nodes_exist(self):
        assert hydro_inflow_MAF2019.requires_source_data == ('nodedata',)
