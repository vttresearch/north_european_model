"""Heat demand from one temperature series per country, and the zeros it must never ship.

Two things are worth pinning here, and neither had a test before.

The first is what ``twh/year`` means. It is a weather-normalised *normal year*
figure, so the built demand matches it as a multi-year mean while no single
climate year does. A per-year normalisation would satisfy half of that and
delete the inter-annual variation the processor exists to produce, so both
halves are asserted together.

The second is that a zero is not a value this processor may produce. Every way
it can fail -- a country the temperature file has never heard of, a hole in that
file, a ``twh/year`` that is not a number -- ends as a NaN column that the GDX
gate turns into a plausible zero. District heating is the worst parameter for
that, because a large part of every year already sits at zero *weather-driven*
demand and only the flat share keeps those hours off the floor; a fabricated zero
and a modelled summer look identical downstream. So the alarm is checked as
carefully as the arithmetic. ``docs/dh-demand-timeseries.md`` has the per-country
numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.timeseries.processors.DH_demand_fromTemperature import DH_demand_fromTemperature
from tests._common.fixtures import FakeLogger
from tests._common.processor_contract import assert_processor_conforms

START_YEAR = 2014
END_YEAR = 2015

#: Two days before START_YEAR, so a fixture can shorten the warm-up on purpose.
DEFAULT_FIRST = pd.Timestamp(f"{START_YEAR - 1}-12-30")
DEFAULT_LAST = pd.Timestamp(f"{END_YEAR}-12-31 23:00")


def cold_and_mild(index: pd.DatetimeIndex) -> np.ndarray:
    """A temperature series with a real seasonal swing and one colder year.

    Sinusoidal about 8 C so summer genuinely crosses the balance point, with
    2014 two degrees colder than 2015 -- which is what makes the normal-year
    assertions mean something rather than passing on a flat series.
    """
    day = index.dayofyear.to_numpy()
    seasonal = 8.0 - 12.0 * np.cos(2 * np.pi * (day - 15) / 365.0)
    return seasonal + np.where(index.year.to_numpy() == START_YEAR, -2.0, 0.0)


def write_temperature(
    folder: Path,
    columns=("FI",),
    *,
    temps=None,
    first=DEFAULT_FIRST,
    last=DEFAULT_LAST,
    drop_hours=(),
    blank_hours=(),
    duplicate_hours=(),
    bad_timestamps=(),
    time_header="Time",
) -> Path:
    """Write a synthetic Temperature.csv and return the folder.

    ``drop_hours`` removes whole rows -- the defect the coverage check is for.
    ``blank_hours`` keeps the row and empties the value, which is the defect
    ``complete_native_grid`` is for. The two are not the same and the processor
    treats them differently, so the helper can produce either.
    """
    folder.mkdir(parents=True, exist_ok=True)
    index = pd.date_range(first, last, freq="60min")
    values = cold_and_mild(index) if temps is None else np.asarray(temps, dtype=float)
    if values.ndim == 0:
        values = np.full(len(index), float(values))

    df = pd.DataFrame({col: values.copy() for col in columns}, index=index)
    for hour in blank_hours:
        df.loc[pd.Timestamp(hour), :] = np.nan
    df = df.drop(index=[pd.Timestamp(h) for h in drop_hours])

    df = df.reset_index(names=time_header)
    for hour in duplicate_hours:
        df = pd.concat([df, df[df[time_header] == pd.Timestamp(hour)]], ignore_index=True)
    df[time_header] = df[time_header].astype(str)
    for hour, text in bad_timestamps:
        df.loc[df[time_header] == str(pd.Timestamp(hour)), time_header] = text

    df.to_csv(folder / "Temperature.csv", index=False)
    return folder


def demands(*rows) -> pd.DataFrame:
    """Demand rows as SourceDataPipeline hands them over: already grid-filtered."""
    frame = pd.DataFrame(list(rows))
    frame.insert(1, "grid", "dheat")
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
        demand_grid="dheat",
        rounding_precision=0,
        logger=logger,
    )
    kwargs.update(overrides)
    return DH_demand_fromTemperature(**kwargs), logger


def build(folder: Path, demand_rows: pd.DataFrame, **overrides):
    """Run the processor and return (wide MWh frame indexed by time, logger)."""
    processor, logger = make_processor(folder, demand_rows, **overrides)
    result = processor.run_processor().main_result
    if result.empty:
        return result, logger
    wide = result.pivot(index="time", columns="node", values="value") * -1
    return wide, logger


@pytest.fixture(scope="module")
def clean_folder(tmp_path_factory):
    return write_temperature(tmp_path_factory.mktemp("temperature"), columns=("FI", "PL"))


class TestProfileShape:
    def test_a_warm_hour_leaves_only_the_flat_share(self, tmp_path):
        # 17 C for one week of an otherwise cold year: those hours sit exactly on
        # the balance point, so their weather term is zero and only B remains.
        index = pd.date_range(DEFAULT_FIRST, DEFAULT_LAST, freq="60min")
        warm = (index >= pd.Timestamp(f"{START_YEAR}-07-01")) & (
            index < pd.Timestamp(f"{START_YEAR}-07-08")
        )
        folder = write_temperature(tmp_path, temps=np.where(warm, 17.0, 5.0))
        wide, _ = build(folder, demands(row("FI00", "FI00_dheat", twh=10.0, share=0.3)))

        settled = wide.loc[f"{START_YEAR}-07-03":f"{START_YEAR}-07-07", "FI00_dheat"]
        assert settled.to_numpy() == pytest.approx(10.0 * 1e6 * 0.3 / 8760)

    def test_the_profile_is_the_balance_point_minus_the_smoothed_temperature(self, tmp_path):
        folder = write_temperature(tmp_path, temps=7.0)
        processor, _ = make_processor(folder, demands(row("FI00", "FI00_dheat")))
        grid = processor._read_temperature_grid()
        profile = processor.get_temperature_profile(["FI00"], grid)["FI00"]
        assert profile.eq(DH_demand_fromTemperature.BALANCE_POINT_C - 7.0).all()

    def test_the_window_is_trailing_so_a_step_takes_a_day_to_work_through(self, tmp_path):
        index = pd.date_range(DEFAULT_FIRST, DEFAULT_LAST, freq="60min")
        step = pd.Timestamp(f"{START_YEAR}-06-01 00:00")
        temps = np.where(index < step, 0.0, 10.0)
        folder = write_temperature(tmp_path, temps=temps)
        processor, _ = make_processor(folder, demands(row("FI00", "FI00_dheat")))
        profile = processor.get_temperature_profile(
            ["FI00"], processor._read_temperature_grid()
        )["FI00"]

        # One hour after the step only 1/24 of the window has changed.
        assert profile.loc[step - pd.Timedelta(1, unit="h")] == pytest.approx(17.0)
        assert profile.loc[step] == pytest.approx(17.0 - 10.0 / 24)
        # A full window later it has caught up entirely.
        assert profile.loc[step + pd.Timedelta(23, unit="h")] == pytest.approx(7.0)

    def test_the_first_output_hour_already_has_a_full_window(self, tmp_path):
        """The warm-up hours are read, not merely required.

        Warm-up is 0 C and the modelled range 20 C. A full window at the first
        output hour is 23 warm-up hours plus that hour, mean 20/24, giving a
        profile just under the balance point. A partial window would see 20 C
        alone and clip to zero, so the two are easy to tell apart.
        """
        index = pd.date_range(DEFAULT_FIRST, DEFAULT_LAST, freq="60min")
        temps = np.where(index < pd.Timestamp(f"{START_YEAR}-01-01"), 0.0, 20.0)
        folder = write_temperature(tmp_path, temps=temps)
        processor, _ = make_processor(folder, demands(row("FI00", "FI00_dheat")))
        profile = processor.get_temperature_profile(
            ["FI00"], processor._read_temperature_grid()
        )["FI00"]
        assert profile.iloc[0] == pytest.approx(17.0 - 20.0 / 24)


class TestNormalisation:
    """The normal-year contract, which is the processor's whole reason for shape."""

    def test_the_multi_year_mean_matches_the_table_but_no_single_year_does(self, clean_folder):
        wide, _ = build(clean_folder, demands(row("FI00", "FI00_dheat", twh=10.0)))
        per_year = wide["FI00_dheat"].groupby(wide.index.year).sum() / 1e6

        assert per_year.mean() == pytest.approx(10.0, rel=1e-3)
        # Both halves together: a per-year normalisation would pass the line
        # above and fail this one.
        assert per_year.min() < 9.9 < 10.1 < per_year.max()

    def test_the_colder_year_gets_more_energy(self, clean_folder):
        wide, _ = build(clean_folder, demands(row("FI00", "FI00_dheat")))
        per_year = wide["FI00_dheat"].groupby(wide.index.year).sum()
        assert per_year.loc[START_YEAR] > per_year.loc[END_YEAR]

    def test_the_run_says_how_wide_the_climate_spread_is(self, clean_folder):
        _, logger = build(clean_folder, demands(row("FI00", "FI00_dheat")))
        logger.assert_logged("normal-year twh/year", level="info")

    def test_the_flat_share_sets_the_minimum_hour(self, clean_folder):
        wide, _ = build(clean_folder, demands(row("FI00", "FI00_dheat", twh=10.0, share=0.3)))
        assert wide["FI00_dheat"].min() == pytest.approx(10.0 * 1e6 * 0.3 / 8760)

    def test_a_leap_year_receives_the_extra_hours_of_flat_share(self, tmp_path):
        """Pinning the nominal 8760, so changing it has to be deliberate.

        The flat term is annual/8760 per hour regardless of how many hours the
        year has, so 2016 gets 24 hours more of it than nominal -- about 0.02% of
        the total. Left alone because elec_demand_TYNDP2024 divides by the same
        constant and the two have to move together.
        """
        folder = write_temperature(
            tmp_path, temps=5.0,
            first=pd.Timestamp("2014-12-30"), last=pd.Timestamp("2016-12-31 23:00"),
        )
        wide, _ = build(
            folder, demands(row("FI00", "FI00_dheat", twh=10.0, share=1.0)),
            start_year=2015, end_year=2016,
        )
        per_year = wide["FI00_dheat"].groupby(wide.index.year).sum()
        assert per_year.loc[2015] == pytest.approx(10.0 * 1e6)
        assert per_year.loc[2016] == pytest.approx(10.0 * 1e6 * 8784 / 8760)


class TestMissingTemperatureData:
    """Nothing in this class may raise. The rule is: build what you can, say the rest."""

    def test_a_country_with_no_column_is_reported_and_the_others_still_build(self, clean_folder):
        wide, logger = build(
            clean_folder,
            demands(row("FI00", "FI00_dheat"), row("UK00", "UK00_dheat")),
            countries=["FI00", "UK00"],
        )
        # Written for the case that prompted the check: the temperature file
        # named the column GB while the config asked for UK00.
        logger.assert_logged("no 'UK' column", level="warn")
        logger.assert_logged("UK00_dheat", level="error")
        assert wide["UK00_dheat"].isna().all()
        assert wide["FI00_dheat"].notna().all()
        assert wide["FI00_dheat"].min() > 0

    def test_a_single_blank_hour_is_filled_and_said_so(self, tmp_path):
        folder = write_temperature(tmp_path, blank_hours=[f"{START_YEAR}-03-05 04:00"])
        wide, logger = build(folder, demands(row("FI00", "FI00_dheat")))
        logger.assert_logged("filled 1 single missing hour", level="info")
        assert wide["FI00_dheat"].notna().all()

    def test_a_run_of_blank_hours_leaves_the_country_unbuilt(self, tmp_path):
        folder = write_temperature(
            tmp_path,
            blank_hours=[f"{START_YEAR}-03-05 04:00", f"{START_YEAR}-03-05 05:00"],
        )
        wide, logger = build(folder, demands(row("FI00", "FI00_dheat")))
        logger.assert_logged("longest run 2 hour(s)", level="warn")
        assert wide["FI00_dheat"].isna().all()

    def test_a_dropped_row_is_reported_rather_than_shifting_the_calendar(self, tmp_path):
        # t-labels come from row position downstream, so an absent row would pull
        # every later hour one label earlier if it went unnoticed.
        folder = write_temperature(tmp_path, drop_hours=[f"{START_YEAR}-07-01 12:00"])
        _, logger = build(folder, demands(row("FI00", "FI00_dheat")))
        logger.assert_logged("are not in the file", level="warn")

    def test_the_other_countries_still_build_when_one_row_is_dropped(self, tmp_path):
        folder = write_temperature(
            tmp_path, columns=("FI", "PL"), drop_hours=[f"{START_YEAR}-07-01 12:00"]
        )
        wide, _ = build(
            folder,
            demands(row("FI00", "FI00_dheat"), row("PL00", "PL00_dheat")),
            countries=["FI00", "PL00"],
        )
        # The hole is shared, so both are refused -- but the run reaches the end
        # and returns a frame rather than raising.
        assert set(wide.columns) == {"FI00_dheat", "PL00_dheat"}

    def test_a_file_starting_at_the_first_modelled_hour_reports_the_missing_warmup(self, tmp_path):
        folder = write_temperature(tmp_path, first=pd.Timestamp(f"{START_YEAR}-01-01"))
        wide, logger = build(folder, demands(row("FI00", "FI00_dheat")))
        logger.assert_logged("are not in the file", level="warn")
        logger.assert_logged("24-hour mean cannot start without them", level="warn")
        assert wide["FI00_dheat"].isna().all()

    def test_a_file_ending_early_is_reported(self, tmp_path):
        folder = write_temperature(tmp_path, last=pd.Timestamp(f"{END_YEAR}-06-30 23:00"))
        _, logger = build(folder, demands(row("FI00", "FI00_dheat")))
        logger.assert_logged("are not in the file", level="warn")

    def test_a_duplicated_timestamp_keeps_the_first_row(self, tmp_path):
        folder = write_temperature(tmp_path, duplicate_hours=[f"{START_YEAR}-04-01 00:00"])
        wide, logger = build(folder, demands(row("FI00", "FI00_dheat")))
        logger.assert_logged("appear more than once", level="warn")
        assert wide["FI00_dheat"].notna().all()

    def test_an_unparseable_timestamp_is_dropped_and_named(self, tmp_path):
        folder = write_temperature(
            tmp_path, bad_timestamps=[(f"{START_YEAR}-08-08 08:00", "not a date")]
        )
        _, logger = build(folder, demands(row("FI00", "FI00_dheat")))
        logger.assert_logged("not a date", level="warn")

    def test_the_time_column_is_found_whatever_its_case(self, tmp_path):
        folder = write_temperature(tmp_path, time_header="time")
        wide, logger = build(folder, demands(row("FI00", "FI00_dheat")))
        logger.assert_not_logged("no 'Time' column")
        assert wide["FI00_dheat"].notna().all()

    def test_a_missing_file_is_reported_rather_than_raised(self, tmp_path):
        result, logger = build(tmp_path / "nothing-here", demands(row("FI00", "FI00_dheat")))
        logger.assert_logged("Unable to open", level="error")
        assert result.empty or result["FI00_dheat"].isna().all()

    def test_a_country_never_below_the_balance_point_is_named_not_silently_flat(self, tmp_path):
        """Reachable for a Mediterranean zone, and it would otherwise ship a
        featureless flat series that looks like modelled demand."""
        folder = write_temperature(tmp_path, temps=20.0)
        wide, logger = build(folder, demands(row("FI00", "FI00_dheat")))
        logger.assert_logged("never falls below 17.0 C", level="warn")
        assert wide["FI00_dheat"].isna().all()

    def test_zero_degrees_is_a_temperature_not_a_gap(self, tmp_path):
        """The one place this caller diverges from the hydro ones.

        ``complete_native_grid`` treats a zero as a dropped value for reservoir
        levels. Here a zero is 0 C -- an ordinary December -- and both of its zero
        flags are therefore off. If either were on, a series of freezing hours
        would be read as missing data and the whole country refused.
        """
        index = pd.date_range(DEFAULT_FIRST, DEFAULT_LAST, freq="60min")
        temps = np.full(len(index), 5.0)
        # A lone zero between non-zero neighbours: exactly the shape
        # isolated_zero_is_missing is designed to catch, and wrong here.
        temps[500] = 0.0
        # And a run of them, which zero_is_missing would catch.
        temps[1000:1100] = 0.0
        folder = write_temperature(tmp_path, temps=temps)
        wide, logger = build(folder, demands(row("FI00", "FI00_dheat")))

        logger.assert_not_logged("no temperature")
        logger.assert_not_logged("filled")
        assert wide["FI00_dheat"].notna().all()


class TestDemandRows:
    def test_a_non_numeric_twh_costs_one_node_and_names_it(self, clean_folder):
        wide, logger = build(
            clean_folder,
            demands(row("FI00", "FI00_dheat_A"), row("FI00", "FI00_dheat_B", twh="lots")),
        )
        logger.assert_logged("FI00_dheat_B", level="warn")
        logger.assert_logged("not a number", level="warn")
        assert wide["FI00_dheat_A"].notna().all()
        assert wide["FI00_dheat_B"].isna().all()

    def test_a_negative_twh_is_refused_rather_than_flipping_the_node_into_supply(self, clean_folder):
        wide, logger = build(
            clean_folder,
            demands(row("FI00", "FI00_dheat_A"), row("FI00", "FI00_dheat_B", twh=-5.0)),
        )
        logger.assert_logged("negative", level="warn")
        assert wide["FI00_dheat_B"].isna().all()

    def test_a_constant_share_out_of_range_costs_one_node_not_every_node(self, clean_folder):
        """It used to raise, which cost the whole processor its GDX."""
        wide, logger = build(
            clean_folder,
            demands(row("FI00", "FI00_dheat_A"), row("FI00", "FI00_dheat_B", share=1.4)),
        )
        logger.assert_logged("not between 0 and 1", level="warn")
        assert wide["FI00_dheat_A"].notna().all()
        assert wide["FI00_dheat_B"].isna().all()

    def test_a_missing_constant_share_means_no_flat_part(self, clean_folder):
        wide, _ = build(
            clean_folder, demands({"country": "FI00", "node": "FI00_dheat", "twh/year": 10.0})
        )
        assert wide["FI00_dheat"].min() == pytest.approx(0.0)

    def test_two_rows_naming_the_same_node_keep_the_first_and_name_both_countries(self, clean_folder):
        # The shape a wrong country cell makes: two rows claiming one node.
        # Silently keeping the last is how such a row goes unnoticed.
        wide, logger = build(
            clean_folder,
            demands(row("FI00", "FI00_dheat", twh=10.0), row("PL00", "FI00_dheat", twh=99.0)),
            countries=["FI00", "PL00"],
        )
        logger.assert_logged("named by more than one", level="warn")
        logger.assert_logged("PL00", level="warn")
        per_year = wide["FI00_dheat"].groupby(wide.index.year).sum() / 1e6
        assert per_year.mean() == pytest.approx(10.0, rel=1e-3)

    def test_the_country_match_folds_case(self, clean_folder):
        """The source-side whitelist folds case and keeps the workbook's spelling.

        So a cell reading 'fi00' arrives here as 'fi00'. It used to be dropped by
        a case-sensitive filter before the case-insensitive matcher downstream
        ever saw it.
        """
        wide, _ = build(clean_folder, demands(row("fi00", "FI00_dheat")))
        assert "FI00_dheat" in wide.columns
        assert wide["FI00_dheat"].notna().all()

    def test_a_table_without_a_twh_column_says_so_once(self, clean_folder):
        result, logger = build(
            clean_folder, pd.DataFrame([{"country": "FI00", "grid": "dheat", "node": "FI00_dheat"}])
        )
        logger.assert_logged("no 'twh/year' column", level="error")
        assert result.empty


class TestCountrySetTolerance:
    """Adding, splitting or removing a country code must never raise."""

    def test_a_split_country_gives_each_half_the_same_shape(self, tmp_path):
        # EE00 becomes EE01 and EE02. Both resolve to the file's 'EE' column by
        # the first-two-letters rule, which is why the rule is a rule and not a
        # lookup table.
        folder = write_temperature(tmp_path, columns=("EE",))
        wide, logger = build(
            folder,
            demands(row("EE01", "EE01_dheat", twh=3.0), row("EE02", "EE02_dheat", twh=1.0)),
            countries=["EE01", "EE02"],
        )
        logger.assert_clean()
        ratio = wide["EE01_dheat"] / wide["EE02_dheat"]
        assert ratio.round(6).nunique() == 1
        assert ratio.iloc[0] == pytest.approx(3.0)

    def test_a_configured_country_with_no_demand_rows_is_not_mentioned(self, clean_folder):
        """Spain has no district heating. Saying so every run only worries people.

        The workbooks are the statement of what the model contains, so a country
        with no rows is an answer rather than an absence -- see "What a build
        says" in docs/timeseries.md.
        """
        _, logger = build(
            clean_folder, demands(row("FI00", "FI00_dheat")), countries=["FI00", "SE01", "NOS0"]
        )
        logger.assert_not_logged("SE01")
        logger.assert_not_logged("NOS0")
        logger.assert_clean()

    def test_no_country_has_demand_rows_at_all(self, clean_folder):
        result, logger = build(
            clean_folder, demands(row("XX00", "XX00_dheat")), countries=["FI00"]
        )
        logger.assert_logged("Nothing to build", level="warn")
        assert result.empty


class TestZeroHours:
    """No hour of a heat network consumes nothing. Every zero here is a symptom."""

    def test_a_clean_build_alarms_about_nothing(self, clean_folder):
        _, logger = build(
            clean_folder,
            demands(row("FI00", "FI00_dheat"), row("PL00", "PL00_dheat")),
            countries=["FI00", "PL00"],
        )
        logger.assert_clean()

    def test_a_zero_constant_share_empties_the_summer_and_says_which_node(self, clean_folder):
        wide, logger = build(clean_folder, demands(row("FI00", "FI00_dheat", share=0.0)))
        logger.assert_logged("FI00_dheat", level="error")
        logger.assert_logged("constant_share is 0", level="error")
        assert (wide["FI00_dheat"] == 0).any()

    def test_a_node_with_no_temperature_data_is_alarmed_about_once_not_per_hour(self, clean_folder):
        _, logger = build(
            clean_folder,
            demands(row("FI00", "FI00_dheat"), row("UK00", "UK00_dheat")),
            countries=["FI00", "UK00"],
        )
        assert len(logger.matching("UK00_dheat", level="error")) == 1

    def test_a_node_too_small_to_survive_rounding_is_caught(self, clean_folder):
        """The case a naive `== 0` check would miss.

        ProcessorRunner rounds to whole MWh *after* the processor returns, so a
        node whose every hour is below 0.5 MWh/h leaves here as a real number and
        reaches GAMS as nothing at all.
        """
        # 0.004 TWh/yr all flat: 0.0046 MWh/h, well under half a unit.
        _, logger = build(clean_folder, demands(row("FI00", "FI00_dheat", twh=0.004, share=1.0)))
        logger.assert_logged("has no demand in", level="error")

    def test_the_same_node_survives_when_rounding_is_finer(self, clean_folder):
        _, logger = build(
            clean_folder,
            demands(row("FI00", "FI00_dheat", twh=0.004, share=1.0)),
            rounding_precision=5,
        )
        logger.assert_not_logged("has no demand in")


class TestOutputContract:
    def test_it_meets_the_processor_contract(self, clean_folder):
        assert_processor_conforms(
            DH_demand_fromTemperature,
            dimensions=["grid", "node", "f", "t"],
            input_folder=str(clean_folder),
            country_codes=["FI00"],
            start_year=START_YEAR,
            end_year=END_YEAR,
            df_annual_demands=demands(row("FI00", "FI00_dheat")),
            demand_grid="dheat",
        )

    def test_every_value_is_at_or_below_zero(self, clean_folder):
        processor, _ = make_processor(clean_folder, demands(row("FI00", "FI00_dheat")))
        result = processor.run_processor().main_result
        assert (result["value"].dropna() <= 0).all()

    def test_the_grid_column_comes_from_the_spec(self, clean_folder):
        processor, _ = make_processor(
            clean_folder, demands(row("FI00", "FI00_dheat")), demand_grid="districtheat"
        )
        result = processor.run_processor().main_result
        assert set(result["grid"]) == {"districtheat"}


class TestDeclarations:
    def test_it_declares_the_sign_of_what_it_produces(self):
        assert DH_demand_fromTemperature.value_sign == "non_positive"

    def test_it_needs_no_source_data(self):
        # The node cross-check lives in SourceDataPipeline, where it covers every
        # grid rather than only dheat, so this processor asks for no frame.
        assert DH_demand_fromTemperature.requires_source_data == ()
