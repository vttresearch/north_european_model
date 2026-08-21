"""Reservoir bounds are fill ratios scaled by the node's own size.

The size used to come from a CSV of the processor's own that duplicated
``nodedata.upwardLimit`` exactly -- the CSV in GWh, nodedata in MWh, the same
numbers maintained twice with nothing able to tell if they drifted. It now comes
from nodedata, which is what these tests pin: the arithmetic, and what happens
when a country code and the hydro data do not line up.

That last part is not hypothetical tidiness. Splitting a country into regions
produces codes PECD has never heard of, and the failure it used to cause was one
missing lookup discarding the GDX for *every* country.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import pandas as pd
import pytest

from src.timeseries.processors.hydro_storage_limits_MAF2019 import (
    hydro_storage_limits_MAF2019,
)
from tests._common.fixtures import FakeLogger
from tests._common.processor_contract import assert_processor_conforms

MIN_HEADER = "Minimum Reservoir levels at beginning of each week (ratio) 0<=x<=1.0"
MAX_HEADER = "Maximum Reservoir level at beginning of each week (ratio) 0<=x<=1.0"
THIRD_HEADER = "Reservoir levels at beginning of each week (ratio) 0<=x<=1.0"

START_YEAR = 2014
END_YEAR = 2015

#: A run has to cover at least the climate range, so both bounds get 52 weeks.
MIN_RATIO = 0.2
MAX_RATIO = 0.8


def write_levels(
    folder: Path,
    zones: dict[str, dict],
) -> Path:
    """Write a synthetic weekly levels CSV.

    `zones` maps a zone code to a dict of options:
        years           -- iterable of years to emit (default START_YEAR..END_YEAR)
        na_years        -- years whose ratios are written blank
        blank_weeks     -- weeks written blank in every year
        zero_min_weeks  -- weeks whose downwardLimit is written as 0
        zero_max_weeks  -- weeks whose upwardLimit is written as 0
        min/max         -- the ratio values to use
    """
    rows = []
    for zone, opts in zones.items():
        years = opts.get("years", range(START_YEAR, END_YEAR + 1))
        na_years = set(opts.get("na_years", ()))
        blank_weeks = set(opts.get("blank_weeks", ()))
        zero_min = set(opts.get("zero_min_weeks", ()))
        zero_max = set(opts.get("zero_max_weeks", ()))
        lo = opts.get("min", MIN_RATIO)
        hi = opts.get("max", MAX_RATIO)
        for year in years:
            for week in range(1, 53):
                blank = year in na_years or week in blank_weeks
                low = None if blank else (0.0 if week in zero_min else lo)
                high = None if blank else (0.0 if week in zero_max else hi)
                rows.append({
                    "zone": zone,
                    "week": week,
                    "year": year,
                    MIN_HEADER: low,
                    MAX_HEADER: high,
                    THIRD_HEADER: None,
                })
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "PECD-hydro-weekly-reservoir-levels.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def nodedata(**sizes: float) -> pd.DataFrame:
    """A merged-nodedata frame carrying `upwardLimit` for `zone_grid` nodes."""
    return pd.DataFrame({
        "country": [n.split("_")[0] for n in sizes],
        "grid": [n.split("_", 1)[1] for n in sizes],
        "node": list(sizes),
        "upwardlimit": pd.array(list(sizes.values()), dtype="Float64"),
    })


def run(tmp_path: Path, zones: dict, df_nodedata: pd.DataFrame,
        countries: list[str] | None = None) -> tuple[pd.DataFrame, FakeLogger]:
    folder = tmp_path / "timeseries"
    write_levels(folder, zones)
    logger = FakeLogger()
    processor = hydro_storage_limits_MAF2019(
        input_folder=str(folder),
        country_codes=countries if countries is not None else sorted(zones),
        start_year=START_YEAR,
        end_year=END_YEAR,
        df_nodedata=df_nodedata,
        logger=logger,
    )
    return processor.process(), logger


class TestScaling:
    def test_bounds_are_the_ratio_times_the_node_size(self, tmp_path):
        """The whole point: MWh out, no unit conversion on the way.

        nodedata.upwardLimit is already MWh, so there is no factor of 1000 here.
        There used to be one, because the CSV this replaced held GWh.
        """
        result, logger = run(
            tmp_path,
            {"XX00": {}},
            nodedata(XX00_reservoir=1_000_000.0),
        )
        logger.assert_clean()

        upper = result[result["param_gnBoundaryTypes"] == "upwardLimit"]["value"]
        lower = result[result["param_gnBoundaryTypes"] == "downwardLimit"]["value"]
        assert upper.max() == pytest.approx(MAX_RATIO * 1_000_000.0)
        assert lower.max() == pytest.approx(MIN_RATIO * 1_000_000.0)

    def test_two_nodes_of_different_size_scale_independently(self, tmp_path):
        result, _ = run(
            tmp_path,
            {"XX00": {}, "YY00": {}},
            nodedata(XX00_reservoir=1_000_000.0, YY00_reservoir=250_000.0),
        )
        upper = result[result["param_gnBoundaryTypes"] == "upwardLimit"]
        by_node = upper.groupby("node")["value"].max()
        assert by_node["XX00_reservoir"] == pytest.approx(MAX_RATIO * 1_000_000.0)
        assert by_node["YY00_reservoir"] == pytest.approx(MAX_RATIO * 250_000.0)

    def test_the_grid_comes_from_the_node_suffix(self, tmp_path):
        result, _ = run(tmp_path, {"XX00": {}}, nodedata(XX00_reservoir=1_000_000.0))
        assert set(result["grid"].unique()) == {"reservoir"}


class TestBlankYearsOutsideTheClimateRange:
    """A year the run does not use must not disqualify the country.

    AT00's 2017 rows are entirely blank, and the completeness test ran over the
    country's whole history, so AT00 was dropped from 1982-2016 runs on the
    strength of rows those runs never touch.
    """

    def test_a_blank_year_after_the_range_is_ignored(self, tmp_path):
        result, logger = run(
            tmp_path,
            {"XX00": {"years": range(START_YEAR, END_YEAR + 2),
                      "na_years": [END_YEAR + 1]}},
            nodedata(XX00_reservoir=1_000_000.0),
        )
        assert "XX00_reservoir" in set(result["node"])
        logger.assert_clean()

    def test_a_blank_year_before_the_range_is_ignored(self, tmp_path):
        result, logger = run(
            tmp_path,
            {"XX00": {"years": range(START_YEAR - 1, END_YEAR + 1),
                      "na_years": [START_YEAR - 1]}},
            nodedata(XX00_reservoir=1_000_000.0),
        )
        assert "XX00_reservoir" in set(result["node"])
        logger.assert_clean()

    def test_a_blank_later_year_no_longer_disqualifies_the_country(self, tmp_path):
        """The weekly pattern is identical across years, so one good year is enough.

        The old all-or-nothing gate refused a country if any row anywhere was
        blank, which is how AT00 lost its seasonal profile to a 2017 block that
        1982-2016 runs never read.
        """
        result, logger = run(
            tmp_path,
            {"XX00": {"na_years": [END_YEAR]}},
            nodedata(XX00_reservoir=1_000_000.0),
        )
        assert "XX00_reservoir" in set(result["node"])
        logger.assert_clean()


class TestPatternGaps:
    """Gaps in the pattern that actually gets used."""

    def test_a_single_missing_week_is_repaired(self, tmp_path):
        result, logger = run(
            tmp_path,
            {"XX00": {"blank_weeks": (7,)}},
            nodedata(XX00_reservoir=1_000_000.0),
        )
        assert "XX00_reservoir" in set(result["node"])
        logger.assert_logged("single-week gap(s) in the level pattern", level="info")
        logger.assert_clean()

    def test_a_run_of_missing_weeks_is_refused_and_named(self, tmp_path):
        """A hole here is not one bad week -- it is the same week in all 35 years."""
        result, logger = run(
            tmp_path,
            {"XX00": {"blank_weeks": (7, 8, 9)}},
            nodedata(XX00_reservoir=1_000_000.0),
        )
        assert result.empty or "XX00_reservoir" not in set(result["node"])
        logger.assert_logged("XX00_reservoir", level="warn")
        logger.assert_logged("run of 3 week(s)", level="warn")


class TestZerosInTheLevelPattern:
    """A recorded zero is a hole in both bounds, and the source proves it.

    `downwardLimit` of zero is meaningful in principle -- the reservoir may run
    dry -- but the shipped data never says it. Its only two zero runs are in
    SE04, in a column whose smallest non-zero is 0.00044 and which puts eleven
    weeks below 0.01. A column with that vocabulary writes 0.0004, not 0.
    """

    @pytest.mark.parametrize("bound,option", [
        ("downwardLimit", "zero_min_weeks"),
        ("upwardLimit", "zero_max_weeks"),
    ])
    def test_an_isolated_zero_is_repaired(self, tmp_path, bound, option):
        result, logger = run(
            tmp_path, {"XX00": {option: (15,)}}, nodedata(XX00_reservoir=1_000_000.0)
        )
        values = result[result["param_gnBoundaryTypes"] == bound]["value"]
        assert not (values == 0).any()
        logger.assert_logged("single-week gap(s) in the level pattern", level="info")

    def test_a_run_of_zeros_is_refused_and_named(self, tmp_path):
        """Refused rather than invented: the pattern repeats into every year."""
        result, logger = run(
            tmp_path,
            {"XX00": {"zero_min_weeks": (46, 47)}},
            nodedata(XX00_reservoir=1_000_000.0),
        )
        assert result.empty or "XX00_reservoir" not in set(result["node"])
        logger.assert_logged("run of 2 week(s)", level="warn")
        logger.assert_logged("downwardLimit", level="warn")

    def test_a_stated_exception_lets_a_run_through(self, tmp_path):
        """How SE04's weeks 46-47 are handled: a decision recorded, not a silence."""
        folder = tmp_path / "timeseries"
        write_levels(folder, {"XX00": {"zero_min_weeks": (46, 47)}})
        logger = FakeLogger()
        processor = hydro_storage_limits_MAF2019(
            input_folder=str(folder), country_codes=["XX00"],
            start_year=START_YEAR, end_year=END_YEAR,
            df_nodedata=nodedata(XX00_reservoir=1_000_000.0), logger=logger,
        )
        processor.ACCEPTED_LONG_RUNS = {
            "XX00_reservoir downwardLimit": "synthetic case for this test"
        }
        result = processor.process()

        assert "XX00_reservoir" in set(result["node"])
        lower = result[result["param_gnBoundaryTypes"] == "downwardLimit"]["value"]
        assert not (lower == 0).any()
        logger.assert_logged("Accepted rather than escalated", level="info")
        logger.assert_clean()

    def test_the_real_exception_names_its_reason(self):
        """An entry is a decision with a justification, not a silencer."""
        for node, reason in hydro_storage_limits_MAF2019.ACCEPTED_LONG_RUNS.items():
            assert len(reason) > 40, f"{node} needs a reason, not a shrug"


class TestCountrySetTolerance:
    """Adding or removing a country code must never raise."""

    def test_a_missing_size_skips_one_node_and_keeps_the_others(self, tmp_path):
        """The reported failure: this used to be a KeyError that lost every country.

        It escaped to ProcessorRunner, which catches at whole-processor level, so
        one zone without a size meant no GDX at all.
        """
        result, logger = run(
            tmp_path,
            {"XX00": {}, "YY00": {}},
            nodedata(XX00_reservoir=1_000_000.0),  # YY00 has no size
        )
        assert "XX00_reservoir" in set(result["node"])
        assert "YY00_reservoir" not in set(result["node"])
        logger.assert_logged("YY00_reservoir", level="warn")

    def test_a_zero_size_is_reported_rather_than_silently_dropped(self, tmp_path):
        result, logger = run(
            tmp_path,
            {"XX00": {}, "YY00": {}},
            nodedata(XX00_reservoir=1_000_000.0, YY00_reservoir=0.0),
        )
        assert "YY00_reservoir" not in set(result["node"])
        logger.assert_logged("YY00_reservoir", level="warn")

    def test_a_new_country_code_with_no_hydro_data_is_not_an_error(self, tmp_path):
        """FI00 split into FI01/FI02 produces codes PECD has never seen."""
        result, logger = run(
            tmp_path,
            {"XX00": {}},
            nodedata(XX00_reservoir=1_000_000.0, FI01_reservoir=500_000.0),
            countries=["XX00", "FI01"],
        )
        assert "XX00_reservoir" in set(result["node"])
        logger.assert_clean()
        logger.assert_logged("FI01_reservoir", level="info")
        logger.assert_logged("no weekly level data", level="info")

    def test_a_country_with_neither_data_nor_a_node_says_nothing_about_it(self, tmp_path):
        """Most countries in a run have no reservoir; naming them all is noise."""
        result, logger = run(
            tmp_path,
            {"XX00": {}},
            nodedata(XX00_reservoir=1_000_000.0),
            countries=["XX00", "EE00"],
        )
        assert "XX00_reservoir" in set(result["node"])
        logger.assert_not_logged("EE00")

    def test_no_sizes_at_all_returns_empty_rather_than_raising(self, tmp_path):
        result, logger = run(tmp_path, {"XX00": {}}, nodedata())
        assert result.empty
        logger.assert_logged("no reservoir sizes", level="warn")

    def test_pumped_storage_without_level_data_is_named_once(self, tmp_path):
        """PECD has no weekly levels for psOpen outside Norway, nor psClosed anywhere.

        Those nodes are constant-bounded by construction. Saying so beats leaving
        their absence from the time series looking like a bug.
        """
        _, logger = run(
            tmp_path,
            {"XX00": {}},
            nodedata(XX00_reservoir=1_000_000.0, XX00_psOpen=5.0, XX00_psClosed=7.0),
            countries=["XX00"],
        )
        logger.assert_logged("XX00_psOpen", level="info")
        logger.assert_logged("XX00_psClosed", level="info")
        logger.assert_clean()


class TestRatioRange:
    def test_a_ratio_above_one_is_reported(self, tmp_path):
        """`ratio * size` assumes a fraction of a full reservoir. Nothing else checks it."""
        _, logger = run(
            tmp_path,
            {"XX00": {"max": 1.4}},
            nodedata(XX00_reservoir=1_000_000.0),
        )
        logger.assert_logged("outside 0..1", level="warn")

    def test_a_negative_ratio_is_reported(self, tmp_path):
        _, logger = run(
            tmp_path,
            {"XX00": {"min": -0.1}},
            nodedata(XX00_reservoir=1_000_000.0),
        )
        logger.assert_logged("outside 0..1", level="warn")


class TestNodedataDtypes:
    """nodedata arrives under source-data conventions, not timeseries ones."""

    def test_an_all_na_upward_limit_column_is_object_and_tolerated(self, tmp_path):
        """standardize_df_dtypes leaves an all-NA column as object, never Float64.

        Assuming Float64 here would crash on exactly the frame that says "nobody
        has set this yet".
        """
        df = nodedata(XX00_reservoir=1_000_000.0)
        df["upwardlimit"] = pd.Series([pd.NA], dtype="object")
        result, logger = run(tmp_path, {"XX00": {}}, df)
        assert result.empty
        logger.assert_logged("no reservoir sizes", level="warn")

    def test_a_missing_upward_limit_column_does_not_raise(self, tmp_path):
        df = nodedata(XX00_reservoir=1_000_000.0).drop(columns=["upwardlimit"])
        result, logger = run(tmp_path, {"XX00": {}}, df)
        assert result.empty
        logger.assert_logged("no reservoir sizes", level="warn")

    def test_a_non_numeric_size_is_treated_as_missing(self, tmp_path):
        df = nodedata(XX00_reservoir=1_000_000.0, YY00_reservoir=1.0)
        df["upwardlimit"] = pd.Series([1_000_000.0, "unknown"], dtype="object")
        result, logger = run(tmp_path, {"XX00": {}, "YY00": {}}, df)
        assert "XX00_reservoir" in set(result["node"])
        logger.assert_logged("YY00_reservoir", level="warn")


class TestOutputContract:
    def test_the_output_meets_the_processor_contract(self, tmp_path):
        folder = tmp_path / "timeseries"
        write_levels(folder, {"XX00": {}})
        assert_processor_conforms(
            hydro_storage_limits_MAF2019,
            dimensions=["grid", "node", "param_gnBoundaryTypes", "f", "t"],
            input_folder=str(folder),
            country_codes=["XX00"],
            start_year=START_YEAR,
            end_year=END_YEAR,
            df_nodedata=nodedata(XX00_reservoir=1_000_000.0),
        )

    def test_the_secondary_result_lists_node_and_boundary_type(self, tmp_path):
        folder = tmp_path / "timeseries"
        write_levels(folder, {"XX00": {}})
        processor = hydro_storage_limits_MAF2019(
            input_folder=str(folder),
            country_codes=["XX00"],
            start_year=START_YEAR,
            end_year=END_YEAR,
            df_nodedata=nodedata(XX00_reservoir=1_000_000.0),
            logger=FakeLogger(),
        )
        processor.run_processor()
        secondary = processor.secondary_result
        assert list(secondary.columns) == ["node", "param_gnBoundaryTypes", "average_value"]
        assert set(secondary["param_gnBoundaryTypes"]) == {"upwardLimit", "downwardLimit"}


class TestDeclarations:
    def test_it_declares_the_source_data_it_needs(self):
        """Without this the cache cannot know a nodedata edit should rerun it."""
        assert hydro_storage_limits_MAF2019.requires_source_data == ("nodedata",)

    def test_stored_energy_is_declared_non_negative(self):
        assert hydro_storage_limits_MAF2019.value_sign == "non_negative"
