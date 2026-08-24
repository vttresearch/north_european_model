"""Electricity demand from TYNDP profiles, and the cache that is trusted instead of re-read.

Three things are pinned here, and none had a test before.

**The cache.** Reading the workbook takes over a minute, so every run after the
first trusts a parquet file instead. That trust is only worth anything if
something says what was checked when it was written, and if the trust breaks when
it should -- a new workbook, a widened country set, a changed contract. The cache
layer had no test at all, which is how a stale file kept a data-loss bug alive for
three months: the 2040 cache was short of its first climate year and nothing
looked at it.

**Which climate years are usable.** Coverage is genuinely ragged -- several
countries stop before the others do -- so a shortfall only matters against what a
run asks for. A country missing 2017 is nothing to a 1982-2016 build and is a
whole year of fabricated zeros to a 1982-2017 one. Both halves are asserted.

**That a zero is not a value this processor may produce.** Every way it can fail
ends as a NaN column that the GDX gate turns into a plausible zero, and unlike
district heating there is not even a summer it could be confused with.
``docs/elec-demand-timeseries.md`` has the details.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.timeseries.processors.elec_demand_TYNDP2024 import elec_demand_TYNDP2024
from tests._common.fixtures import FakeLogger
from tests._common.processor_contract import assert_processor_conforms

START_YEAR = 2014
END_YEAR = 2015

#: The one workbook every scenario year reads.
WORKBOOK = "elec_2030_National_Trends.xlsx"
CACHE = "elec_2030_National_Trends.parquet"

#: Hours in one standardised source year. Named here too, so a test that builds a
#: deliberately short sheet says what it is short of.
STANDARD_HOURS = 8760


def standard_calendar() -> pd.DataFrame:
    """The 365-day grid every sheet carries: month 0-indexed, day, hour 0-23."""
    days = pd.date_range("2001-01-01", "2001-12-31", freq="D")
    return pd.DataFrame({
        "month": np.repeat(days.month.to_numpy() - 1, 24),
        "day": np.repeat(days.day.to_numpy(), 24),
        "hour": np.tile(np.arange(24), len(days)),
    })


CALENDAR = standard_calendar()


def swinging(scale: float = 1.0) -> np.ndarray:
    """A profile with a real annual swing, so normalisation means something.

    A flat series would satisfy the normal-year assertions trivially; this one
    only satisfies them if the scaling is actually done across the whole climate
    range rather than per year.
    """
    day = np.repeat(np.arange(1, 366), 24)
    return scale * (100.0 - 30.0 * np.cos(2 * np.pi * (day - 15) / 365.0))


#: 2014 runs 10% above 2015, so neither year equals the table figure and the mean
#: of the two does.
DEFAULT_YEARS = {START_YEAR: swinging(1.1), END_YEAR: swinging(0.9)}


def write_workbook(
    folder: Path,
    sheets=None,
    *,
    header_row=7,
    calendar=CALENDAR,
    date_column=True,
    extra_columns=None,
    filename=WORKBOOK,
) -> Path:
    """Write a synthetic TYNDP workbook and return the folder.

    ``sheets`` maps a sheet name to ``{year: values}``. A year mapped to None gets
    a column of blanks -- the shape a country that stops early actually has --
    while a year simply absent from the mapping gets no column at all. Both occur
    in the real workbooks and the processor treats them the same way.
    """
    folder.mkdir(parents=True, exist_ok=True)
    if sheets is None:
        sheets = {"FI00": dict(DEFAULT_YEARS)}

    path = folder / filename
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, years in sheets.items():
            frame = pd.DataFrame(index=range(len(calendar)))
            if date_column:
                frame["Date"] = "n/a"
            for column in calendar.columns:
                frame[column] = calendar[column].to_numpy()
            for column, values in (extra_columns or {}).items():
                frame[column] = values
            for year, values in years.items():
                if values is None:
                    frame[year] = np.nan
                elif np.isscalar(values):
                    frame[year] = float(values)
                else:
                    frame[year] = np.asarray(values, dtype=float)[: len(calendar)]
            frame.to_excel(writer, sheet_name=name, startrow=header_row, index=False)
    return folder


def demands(*rows) -> pd.DataFrame:
    """Demand rows as SourceDataPipeline hands them over: already grid-filtered."""
    frame = pd.DataFrame(list(rows))
    frame.insert(1, "grid", "elec")
    return frame


def row(country, node, twh=10.0, share=0.3) -> dict:
    return {"country": country, "node": node, "twh/year": twh, "constant_share": share}


def make_processor(folder: Path, demand_rows: pd.DataFrame, *, countries=None, **overrides):
    logger = overrides.pop("logger", FakeLogger())
    kwargs = dict(
        input_folder=str(folder),
        country_codes=countries if countries is not None else ["FI00"],
        start_year=START_YEAR,
        end_year=END_YEAR,
        df_annual_demands=demand_rows,
        demand_grid="elec",
        rounding_precision=0,
        logger=logger,
    )
    kwargs.update(overrides)
    return elec_demand_TYNDP2024(**kwargs), logger


def build(folder: Path, demand_rows: pd.DataFrame, **overrides):
    """Run the processor and return (wide MWh frame indexed by time, logger)."""
    processor, logger = make_processor(folder, demand_rows, **overrides)
    result = processor.run_processor().main_result
    if result.empty:
        return result, logger
    wide = result.pivot(index="time", columns="node", values="value") * -1
    return wide, logger


def receipt_of(folder: Path, filename=CACHE) -> dict:
    metadata = pq.read_schema(folder / filename).metadata
    return json.loads(metadata[elec_demand_TYNDP2024.RECEIPT_KEY].decode())


@pytest.fixture(scope="module")
def clean_folder(tmp_path_factory):
    """A workbook with two countries, and the cache already built from it."""
    folder = write_workbook(
        tmp_path_factory.mktemp("tyndp"),
        {"FI00": dict(DEFAULT_YEARS), "PL00": dict(DEFAULT_YEARS)},
    )
    build(folder, demands(row("FI00", "FI00_elec"), row("PL00", "PL00_elec")),
          countries=["FI00", "PL00"])
    return folder


class TestTheCacheIsBuiltAndReused:
    def test_the_first_run_writes_a_cache_carrying_a_receipt(self, tmp_path):
        folder = write_workbook(tmp_path)
        build(folder, demands(row("FI00", "FI00_elec")))

        assert (folder / CACHE).exists()
        receipt = receipt_of(folder)
        assert receipt["contract_version"] == elec_demand_TYNDP2024.PARQUET_CONTRACT_VERSION
        assert receipt["source_name"] == WORKBOOK
        assert receipt["source_bytes"] == (folder / WORKBOOK).stat().st_size

    def test_the_second_run_does_not_reopen_the_workbook(self, tmp_path):
        folder = write_workbook(tmp_path)
        build(folder, demands(row("FI00", "FI00_elec")))

        _, logger = build(folder, demands(row("FI00", "FI00_elec")))
        logger.assert_logged("Using parquet cache")
        logger.assert_not_logged("Reading the whole workbook")

    def test_the_receipt_records_which_years_each_country_has(self, tmp_path):
        folder = write_workbook(
            tmp_path,
            {"FI00": {START_YEAR: swinging(), END_YEAR: swinging(), 2016: None}},
        )
        build(folder, demands(row("FI00", "FI00_elec")))

        coverage = receipt_of(folder)["countries"]["FI00"]
        assert coverage["complete"] == [START_YEAR, END_YEAR]
        assert coverage["empty"] == [2016]
        assert coverage["partial"] == {}

    def test_the_first_year_column_is_not_dropped(self, tmp_path):
        """The defect fixed in 6d11ba4, pinned as a property rather than a slice.

        A positional ``columns[4:]`` took one year column too few and silently
        lost the earliest climate year from every sheet. Year columns are now
        taken as the complement of the calendar columns, which makes that
        impossible; what this asserts is the property the bug violated.
        """
        folder = write_workbook(
            tmp_path,
            {"FI00": {2013: swinging(), START_YEAR: swinging(), END_YEAR: swinging()}},
        )
        build(folder, demands(row("FI00", "FI00_elec")))

        assert receipt_of(folder)["countries"]["FI00"]["complete"] == [
            2013, START_YEAR, END_YEAR
        ]


class TestTheReceiptDecidesWhenToRebuild:
    """The cache is trusted, so what breaks that trust is the whole contract."""

    def _reason(self, folder, **overrides):
        processor, _ = make_processor(folder, demands(row("FI00", "FI00_elec")), **overrides)
        return processor._rebuild_reason()

    def test_a_fresh_cache_is_trusted(self, tmp_path):
        folder = write_workbook(tmp_path)
        build(folder, demands(row("FI00", "FI00_elec")))
        assert self._reason(folder) is None

    def test_a_cache_without_a_receipt_is_not_trusted(self, tmp_path):
        folder = write_workbook(tmp_path)
        build(folder, demands(row("FI00", "FI00_elec")))
        # Rewrite it the way pandas would, losing the metadata -- which is
        # exactly the state every cache built before the receipt existed is in.
        pd.read_parquet(folder / CACHE).to_parquet(folder / CACHE, index=False)

        assert "no readable receipt" in self._reason(folder)

    def test_a_receipt_from_another_contract_version_is_not_trusted(self, tmp_path, monkeypatch):
        folder = write_workbook(tmp_path)
        build(folder, demands(row("FI00", "FI00_elec")))
        monkeypatch.setattr(elec_demand_TYNDP2024, "PARQUET_CONTRACT_VERSION", 99)

        assert "contract version" in self._reason(folder)

    def test_a_modified_workbook_is_not_trusted(self, tmp_path):
        folder = write_workbook(tmp_path)
        build(folder, demands(row("FI00", "FI00_elec")))
        stat = (folder / WORKBOOK).stat()
        os.utime(folder / WORKBOOK, ns=(stat.st_atime_ns, stat.st_mtime_ns + 10**9))

        assert "modified" in self._reason(folder)

    def test_a_workbook_of_a_different_size_is_not_trusted(self, tmp_path):
        folder = write_workbook(tmp_path)
        build(folder, demands(row("FI00", "FI00_elec")))
        # A second country makes the file bigger while leaving it valid, which is
        # what a re-download of a revised workbook looks like.
        write_workbook(folder, {"FI00": dict(DEFAULT_YEARS), "PL00": dict(DEFAULT_YEARS)})

        assert "bytes" in self._reason(folder)

    def test_an_edited_processor_is_not_trusted(self, tmp_path):
        folder = write_workbook(tmp_path)
        build(folder, demands(row("FI00", "FI00_elec")))
        cache = folder / CACHE
        stat = cache.stat()
        os.utime(cache, ns=(stat.st_atime_ns, stat.st_mtime_ns - 10 * 365 * 86400 * 10**9))

        assert "this processor has been edited" in self._reason(folder)

    def test_widening_the_allowed_countries_rebuilds(self, tmp_path, monkeypatch):
        """The stub-config case: a narrow build must not satisfy a wider one.

        The cache is keyed on ALLOWED_COUNTRIES rather than on the config's
        country_codes, so a run that asks for one country still builds the whole
        set. What has to be caught is the tuple itself growing.
        """
        folder = write_workbook(
            tmp_path, {"FI00": dict(DEFAULT_YEARS), "PL00": dict(DEFAULT_YEARS)}
        )
        monkeypatch.setattr(elec_demand_TYNDP2024, "ALLOWED_COUNTRIES", ("FI00",))
        build(folder, demands(row("FI00", "FI00_elec")))
        assert self._reason(folder) is None

        monkeypatch.setattr(elec_demand_TYNDP2024, "ALLOWED_COUNTRIES", ("FI00", "PL00"))
        assert "PL00" in self._reason(folder)

    def test_a_country_the_workbook_lacks_does_not_rebuild_forever(self, tmp_path, monkeypatch):
        """An absence that is recorded is accounted for, so it settles."""
        folder = write_workbook(tmp_path)
        monkeypatch.setattr(elec_demand_TYNDP2024, "ALLOWED_COUNTRIES", ("FI00", "ZZ00"))
        build(folder, demands(row("FI00", "FI00_elec")))

        assert receipt_of(folder)["sheets_rejected"]["ZZ00"]
        assert self._reason(folder) is None


class TestTheWorkbookNeedNotBePresent:
    def test_a_trusted_cache_is_used_without_the_workbook(self, tmp_path):
        folder = write_workbook(tmp_path)
        build(folder, demands(row("FI00", "FI00_elec")))
        (folder / WORKBOOK).unlink()

        wide, logger = build(folder, demands(row("FI00", "FI00_elec")))
        logger.assert_logged("freshness could not be confirmed", level="info")
        logger.assert_no_errors()
        assert wide["FI00_elec"].notna().all()

    def test_neither_present_is_reported_rather_than_raised(self, tmp_path):
        wide, logger = build(tmp_path, demands(row("FI00", "FI00_elec")))

        logger.assert_logged("is not present", level="warn")
        assert wide["FI00_elec"].isna().all()

    def test_an_unreadable_cache_without_a_workbook_is_reported(self, tmp_path):
        folder = write_workbook(tmp_path)
        build(folder, demands(row("FI00", "FI00_elec")))
        (folder / WORKBOOK).unlink()
        (folder / CACHE).write_bytes(b"not a parquet file")

        _, logger = build(folder, demands(row("FI00", "FI00_elec")))
        logger.assert_logged("is not present", level="warn")


class TestClimateYearCoverage:
    """Ragged coverage is data. A gap inside the requested range is a defect."""

    def test_a_missing_requested_year_is_named_not_silently_zeroed(self, tmp_path):
        # The real NT2040 case: the cache is short of its first climate year.
        folder = write_workbook(tmp_path, {"FI00": {END_YEAR: swinging()}})
        wide, logger = build(folder, demands(row("FI00", "FI00_elec")))

        logger.assert_logged(f"no complete profile for climate year(s) {START_YEAR}",
                             level="warn")
        assert wide["FI00_elec"].isna().all()

    def test_a_missing_requested_year_reaches_the_zero_alarm(self, tmp_path):
        folder = write_workbook(tmp_path, {"FI00": {END_YEAR: swinging()}})
        _, logger = build(folder, demands(row("FI00", "FI00_elec")))

        logger.assert_logged("has no demand in", level="error")

    def test_a_blank_requested_year_counts_as_missing(self, tmp_path):
        # The column exists and holds nothing, which is how the real workbooks
        # spell a country that stops early.
        folder = write_workbook(tmp_path, {"FI00": {START_YEAR: None, END_YEAR: swinging()}})
        _, logger = build(folder, demands(row("FI00", "FI00_elec")))

        logger.assert_logged(f"climate year(s) {START_YEAR}", level="warn")

    def test_a_year_of_zeros_counts_as_missing(self, tmp_path):
        # Electricity demand is never zero, so a column of zeros is a gap
        # dressed as data.
        folder = write_workbook(tmp_path, {"FI00": {START_YEAR: 0.0, END_YEAR: swinging()}})
        _, logger = build(folder, demands(row("FI00", "FI00_elec")))

        logger.assert_logged(f"climate year(s) {START_YEAR}", level="warn")

    def test_coverage_outside_the_requested_range_is_not_mentioned(self, tmp_path):
        # PL00 stops before FI00 does, exactly as AT00 does in the real workbook.
        # The run asks for neither of the years it lacks.
        folder = write_workbook(tmp_path, {
            "FI00": {START_YEAR: swinging(1.1), END_YEAR: swinging(0.9), 2016: swinging()},
            "PL00": {START_YEAR: swinging(1.1), END_YEAR: swinging(0.9), 2016: None},
        })
        wide, logger = build(
            folder,
            demands(row("FI00", "FI00_elec"), row("PL00", "PL00_elec")),
            countries=["FI00", "PL00"],
        )

        # Not assert_clean: this builds a fresh cache, and a synthetic workbook
        # legitimately warns that it carries only two of the allowed countries.
        logger.assert_not_logged("2016")
        logger.assert_not_logged("no complete profile")
        logger.assert_no_errors()
        assert wide["PL00_elec"].notna().all()

    def test_a_partial_year_is_warned_about_when_the_cache_is_written(self, tmp_path):
        short = swinging()
        short[:100] = np.nan
        folder = write_workbook(tmp_path, {"FI00": {START_YEAR: short, END_YEAR: swinging()}})
        _, logger = build(folder, demands(row("FI00", "FI00_elec")))

        logger.assert_logged("some but not all", level="warn")
        assert receipt_of(folder)["countries"]["FI00"]["partial"] == {str(START_YEAR): 8660}

    def test_only_complete_years_are_counted_when_normalising(self, tmp_path):
        """The year count is the index's, which is why the gate above must hold.

        A part-year counted as a whole one is what inflated every year of a
        country in the district heating processor. Here the shortfall removes the
        country instead, so the count can state the intended rule directly.
        """
        folder = write_workbook(tmp_path, {"FI00": dict(DEFAULT_YEARS)})
        wide, _ = build(folder, demands(row("FI00", "FI00_elec", twh=10.0, share=0.0)))

        annual = wide["FI00_elec"].groupby(wide.index.year).sum()
        assert annual.mean() == pytest.approx(10.0 * 1e6, rel=1e-9)


class TestNormalisation:
    """The normal-year contract, which is the processor's whole reason for shape."""

    def test_the_multi_year_mean_matches_the_table_but_no_single_year_does(self, clean_folder):
        wide, _ = build(clean_folder, demands(row("FI00", "FI00_elec", twh=10.0, share=0.0)))
        annual = wide["FI00_elec"].groupby(wide.index.year).sum()

        assert annual.mean() == pytest.approx(10.0 * 1e6, rel=1e-9)
        assert annual.min() < 10.0 * 1e6 < annual.max()

    def test_the_bigger_year_gets_more_energy(self, clean_folder):
        wide, _ = build(clean_folder, demands(row("FI00", "FI00_elec")))
        annual = wide["FI00_elec"].groupby(wide.index.year).sum()

        assert annual.loc[START_YEAR] > annual.loc[END_YEAR]

    def test_the_run_says_how_wide_the_climate_spread_is(self, clean_folder):
        _, logger = build(clean_folder, demands(row("FI00", "FI00_elec")))
        logger.assert_logged("individual climate years range", level="info")

    def test_the_flat_share_sets_the_minimum_hour(self, clean_folder):
        wide, _ = build(clean_folder, demands(row("FI00", "FI00_elec", twh=10.0, share=1.0)))
        expected = 10.0 * 1e6 / elec_demand_TYNDP2024.HOURS_PER_YEAR

        assert wide["FI00_elec"].min() == pytest.approx(expected, rel=1e-9)
        assert wide["FI00_elec"].max() == pytest.approx(expected, rel=1e-9)

    def test_several_nodes_share_one_country_profile(self, clean_folder):
        wide, _ = build(
            clean_folder,
            demands(row("FI00", "FI00_elec", twh=10.0), row("FI00", "FI00_elec_north", twh=5.0)),
        )
        ratio = wide["FI00_elec"] / wide["FI00_elec_north"]

        assert ratio.std() == pytest.approx(0.0, abs=1e-9)
        assert ratio.iloc[0] == pytest.approx(2.0, rel=1e-9)


class TestUnusableSheets:
    """Nothing in this class may raise. The rule is: build what you can, say the rest."""

    def _rejection(self, folder, sheet="FI00"):
        """Run a build and return the reason the sheet was refused, plus the log.

        Read off the log rather than the receipt: when the only sheet is unusable
        no cache is written at all, and the reason still has to reach the user.
        The receipt's copy of it is asserted separately, where a good sheet keeps
        the write alive.
        """
        _, logger = build(folder, demands(row("FI00", "FI00_elec")))
        lines = logger.matching("could not be read:")
        return (lines[0] if lines else ""), logger

    def test_a_sheet_without_a_date_column_is_named(self, tmp_path):
        folder = write_workbook(tmp_path, date_column=False)
        reason, logger = self._rejection(folder)

        assert "no 'Date' column" in reason
        logger.assert_logged("could not be read:", level="warn")

    def test_a_header_on_the_wrong_row_is_named(self, tmp_path):
        folder = write_workbook(tmp_path, header_row=3)
        reason, _ = self._rejection(folder)

        assert reason

    def test_a_one_indexed_month_column_is_named(self, tmp_path):
        calendar = CALENDAR.copy()
        calendar["month"] = calendar["month"] + 1
        folder = write_workbook(tmp_path, calendar=calendar)
        reason, _ = self._rejection(folder)

        assert "'month' column runs 1..12" in reason

    def test_a_one_indexed_hour_column_is_named(self, tmp_path):
        calendar = CALENDAR.copy()
        calendar["hour"] = calendar["hour"] + 1
        folder = write_workbook(tmp_path, calendar=calendar)
        reason, _ = self._rejection(folder)

        assert "'hour' column runs 1..24" in reason

    def test_a_short_year_is_named(self, tmp_path):
        folder = write_workbook(tmp_path, calendar=CALENDAR.iloc[:-1])
        reason, _ = self._rejection(folder)

        assert f"8759 rows where a standardised year has {STANDARD_HOURS}" in reason

    def test_a_repeated_hour_is_named(self, tmp_path):
        calendar = CALENDAR.copy()
        calendar.iloc[-1] = calendar.iloc[-2]
        folder = write_workbook(tmp_path, calendar=calendar)
        reason, _ = self._rejection(folder)

        assert "repeats or skips hours" in reason

    def test_a_column_that_is_neither_calendar_nor_year_is_named(self, tmp_path):
        folder = write_workbook(tmp_path, extra_columns={"Comment": "x"})
        reason, _ = self._rejection(folder)

        assert "neither a calendar column nor a year" in reason

    def test_sheets_disagreeing_about_the_calendar_are_named(self, tmp_path):
        shuffled = CALENDAR.iloc[::-1].reset_index(drop=True)
        folder = write_workbook(tmp_path, {"FI00": dict(DEFAULT_YEARS)})
        write_workbook(folder, {"PL00": dict(DEFAULT_YEARS)}, calendar=shuffled,
                       filename="second.xlsx")
        # One workbook, two sheets, one of them out of order.
        with pd.ExcelWriter(folder / WORKBOOK, engine="openpyxl") as writer:
            for name, cal in (("FI00", CALENDAR), ("PL00", shuffled)):
                frame = pd.DataFrame({"Date": "n/a"}, index=range(len(cal)))
                for column in cal.columns:
                    frame[column] = cal[column].to_numpy()
                for year, values in DEFAULT_YEARS.items():
                    frame[year] = values
                frame.to_excel(writer, sheet_name=name, startrow=7, index=False)

        _, logger = build(folder, demands(row("FI00", "FI00_elec"), row("PL00", "PL00_elec")),
                          countries=["FI00", "PL00"])
        assert "different hour ordering" in receipt_of(folder)["sheets_rejected"]["PL00"]
        logger.assert_logged("PL00_elec", level="warn")

    def test_a_negative_value_is_reported(self, tmp_path):
        values = swinging()
        values[5] = -1.0
        folder = write_workbook(tmp_path, {"FI00": {START_YEAR: values, END_YEAR: swinging()}})
        _, logger = build(folder, demands(row("FI00", "FI00_elec")))

        logger.assert_logged("negative demand value", level="warn")

    def test_no_usable_sheet_at_all_is_reported_rather_than_raised(self, tmp_path):
        folder = write_workbook(tmp_path, date_column=False)
        wide, logger = build(folder, demands(row("FI00", "FI00_elec")))

        logger.assert_logged("No usable country sheet", level="warn")
        assert wide["FI00_elec"].isna().all()

    def test_a_rejected_sheet_is_recorded_when_another_one_survives(self, tmp_path):
        # Two sheets from two writes: only the second carries the flaw, so the
        # cache is still written and the receipt has somewhere to record it.
        folder = write_workbook(tmp_path, {"FI00": dict(DEFAULT_YEARS)})
        broken = pd.DataFrame({"Date": "n/a"}, index=range(len(CALENDAR)))
        for column in CALENDAR.columns:
            broken[column] = CALENDAR[column].to_numpy()
        broken["Comment"] = "x"
        for year, values in DEFAULT_YEARS.items():
            broken[year] = values
        with pd.ExcelWriter(folder / WORKBOOK, engine="openpyxl", mode="a") as writer:
            broken.to_excel(writer, sheet_name="PL00", startrow=7, index=False)

        build(folder, demands(row("FI00", "FI00_elec")))

        assert "neither a calendar column nor a year" in (
            receipt_of(folder)["sheets_rejected"]["PL00"]
        )


class TestCountrySetTolerance:
    def test_a_country_outside_the_allowed_tuple_names_the_tuple(self, clean_folder):
        _, logger = build(clean_folder, demands(row("ZZ00", "ZZ00_elec")),
                          countries=["ZZ00"])

        logger.assert_logged("ALLOWED_COUNTRIES", level="warn")

    def test_the_country_code_is_matched_whatever_its_case(self, clean_folder):
        wide, logger = build(clean_folder, demands(row("fi00", "FI00_elec")),
                             countries=["fi00"])

        logger.assert_clean()
        assert wide["FI00_elec"].notna().all()

    def test_a_configured_country_with_no_demand_rows_is_information(self, clean_folder):
        _, logger = build(clean_folder, demands(row("FI00", "FI00_elec")),
                          countries=["FI00", "PL00"])

        logger.assert_logged("No elec demand rows for 1 configured", level="info")
        logger.assert_clean()

    def test_no_country_has_demand_rows_at_all(self, clean_folder):
        result, logger = build(clean_folder, demands(row("ZZ00", "ZZ00_elec")),
                               countries=["FI00"])

        logger.assert_logged("Nothing to build", level="warn")
        assert result.empty


class TestDemandRows:
    def test_a_non_numeric_twh_costs_one_node_and_names_it(self, clean_folder):
        wide, logger = build(
            clean_folder,
            demands(row("FI00", "FI00_elec", twh="n/a"), row("PL00", "PL00_elec")),
            countries=["FI00", "PL00"],
        )

        logger.assert_logged("not a number", level="warn")
        assert wide["FI00_elec"].isna().all()
        assert wide["PL00_elec"].notna().all()

    def test_a_negative_twh_is_refused_rather_than_flipping_into_supply(self, clean_folder):
        wide, logger = build(clean_folder, demands(row("FI00", "FI00_elec", twh=-5.0)))

        logger.assert_logged("negative", level="warn")
        assert wide["FI00_elec"].isna().all()

    def test_a_constant_share_out_of_range_costs_one_node_not_every_node(self, clean_folder):
        wide, logger = build(
            clean_folder,
            demands(row("FI00", "FI00_elec", share=1.5), row("PL00", "PL00_elec")),
            countries=["FI00", "PL00"],
        )

        logger.assert_logged("not between 0 and 1", level="warn")
        assert wide["FI00_elec"].isna().all()
        assert wide["PL00_elec"].notna().all()

    def test_a_missing_constant_share_means_no_flat_part(self, clean_folder):
        wide, _ = build(clean_folder, demands(row("FI00", "FI00_elec", share=None)))
        flat = 10.0 * 1e6 / elec_demand_TYNDP2024.HOURS_PER_YEAR

        assert wide["FI00_elec"].min() < flat

    def test_two_rows_naming_the_same_node_keep_the_first(self, clean_folder):
        wide, logger = build(
            clean_folder,
            demands(row("FI00", "shared", twh=10.0), row("PL00", "shared", twh=99.0)),
            countries=["FI00", "PL00"],
        )

        logger.assert_logged("named by more than one", level="warn")
        annual = wide["shared"].groupby(wide.index.year).sum()
        assert annual.mean() == pytest.approx(10.0 * 1e6, rel=1e-3)

    def test_a_table_without_a_twh_column_says_so_once(self, clean_folder):
        rows = demands(row("FI00", "FI00_elec")).drop(columns=["twh/year"])
        result, logger = build(clean_folder, rows)

        logger.assert_logged("no 'twh/year' column", level="error")
        assert result.empty


class TestZeroHours:
    """Electricity demand does not stop. Every zero here is a symptom."""

    def test_a_clean_build_alarms_about_nothing(self, clean_folder):
        _, logger = build(clean_folder, demands(row("FI00", "FI00_elec")))

        logger.assert_not_logged("has no demand in")
        logger.assert_clean()

    def test_a_node_with_no_profile_is_alarmed_about_once_not_per_hour(self, clean_folder):
        _, logger = build(clean_folder, demands(row("ZZ00", "ZZ00_elec")),
                          countries=["ZZ00"])

        assert len(logger.matching("has no demand in")) == 1

    def test_a_node_too_small_to_survive_rounding_is_caught(self, clean_folder):
        # 1e-6 TWh/year over 8760 h is well under half a MWh/h, so every hour of
        # it rounds to nothing.
        _, logger = build(clean_folder, demands(row("FI00", "tiny", twh=1e-6)))

        logger.assert_logged("has no demand in", level="error")

    def test_the_same_node_survives_when_rounding_is_finer(self, clean_folder):
        _, logger = build(clean_folder, demands(row("FI00", "tiny", twh=1e-6)),
                          rounding_precision=6)

        logger.assert_not_logged("has no demand in")

    def test_a_profile_that_crosses_zero_warns_instead_of_erroring(self, tmp_path):
        """A negative source has to pass through zero. That is data, not a gap.

        Grading it an error would be permanent: the crossing is a property of the
        workbook, so every future build would inherit `has_errors` and the full
        rerun that follows it, for something nobody can fix by editing a config.
        """
        values = swinging()
        values[10:14] = [-5.0, 0.0, -5.0, 0.0]
        folder = write_workbook(tmp_path, {"FI00": {START_YEAR: values, END_YEAR: swinging()}})
        # share=0, as every shipped electricity node has it: with a flat term the
        # crossing would be lifted off zero and there would be nothing to grade.
        _, logger = build(folder, demands(row("FI00", "FI00_elec", share=0.0)))

        logger.assert_logged("crosses zero on the way", level="warn")
        logger.assert_no_errors()

    def test_a_node_with_no_data_still_errors(self, clean_folder):
        # The distinction only excuses a crossing; an absent profile is still the
        # fabricated zero the alarm exists for.
        _, logger = build(clean_folder, demands(row("ZZ00", "ZZ00_elec")),
                          countries=["ZZ00"])

        logger.assert_logged("has no demand in", level="error")


class TestOutputContract:
    def test_it_meets_the_processor_contract(self, clean_folder):
        assert_processor_conforms(
            elec_demand_TYNDP2024,
            dimensions=["grid", "node", "f", "t"],
            input_folder=str(clean_folder),
            country_codes=["FI00"],
            start_year=START_YEAR,
            end_year=END_YEAR,
            df_annual_demands=demands(row("FI00", "FI00_elec")),
            demand_grid="elec",
        )

    def test_every_value_is_at_or_below_zero(self, clean_folder):
        processor, _ = make_processor(clean_folder, demands(row("FI00", "FI00_elec")))
        result = processor.run_processor().main_result

        assert (result["value"].dropna() <= 0).all()

    def test_the_grid_column_comes_from_the_spec(self, clean_folder):
        processor, _ = make_processor(
            clean_folder, demands(row("FI00", "FI00_elec")), demand_grid="power"
        )
        result = processor.run_processor().main_result

        assert set(result["grid"]) == {"power"}


class TestDeclarations:
    def test_it_declares_the_sign_of_what_it_produces(self):
        assert elec_demand_TYNDP2024.value_sign == "non_positive"

    def test_it_needs_no_source_data(self):
        # Electricity nodes carry no nodedata rows, so there is no frame to ask
        # for; the demand table arrives via demand_grid instead.
        assert elec_demand_TYNDP2024.requires_source_data == ()

    def test_every_scenario_year_reads_the_same_workbook(self):
        """The 2040 workbook is deliberately unused -- see the constant's comment.

        Pinned because the alternative failure is silent: a scenario_year branch
        creeping back would send 2040 runs to profiles carrying negative demand,
        which reaches Backbone as free generation.
        """
        rows = demands(row("FI00", "FI00_elec"))
        for scenario_year in (2030, 2035, 2040, 2050):
            processor, _ = make_processor(Path("."), rows, scenario_year=scenario_year)
            assert processor.input_file.endswith(
                elec_demand_TYNDP2024.PROFILE_WORKBOOK
            ), f"scenario_year={scenario_year} chose {processor.input_file}"

    def test_the_flat_divisor_matches_the_district_heating_one(self):
        # docs/elec-demand-timeseries.md states these move together or not at
        # all, and nothing else would notice if one of them drifted.
        from src.timeseries.processors.DH_demand_fromTemperature import (
            DH_demand_fromTemperature,
        )

        assert (elec_demand_TYNDP2024.HOURS_PER_YEAR
                == DH_demand_fromTemperature.HOURS_PER_YEAR)
