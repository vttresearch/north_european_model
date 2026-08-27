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

import numpy as np
import pandas as pd
import pytest

from src.source_data.source_data_contributions import CONTRIBUTION_KEYS
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
        min_profile     -- callable week -> downwardLimit, overriding `min`
        max_profile     -- callable week -> upwardLimit, overriding `max`
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
        min_profile = opts.get("min_profile")
        max_profile = opts.get("max_profile")
        for year in years:
            for week in range(1, 53):
                blank = year in na_years or week in blank_weeks
                lo_w = min_profile(week) if min_profile else lo
                hi_w = max_profile(week) if max_profile else hi
                low = None if blank else (0.0 if week in zero_min else lo_w)
                high = None if blank else (0.0 if week in zero_max else hi_w)
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
        logger.assert_logged("Gaps interpolated at 1 node(s)", level="info")
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
        logger.assert_logged("Gaps interpolated at 1 node(s)", level="info")

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
        """How SE04's weeks 46-47 are handled: a decision recorded, not a silence.

        Recorded in ACCEPTED_LONG_RUNS, that is -- the run itself only counts it.
        """
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
        logger.assert_logged("Gaps interpolated at 1 node(s)", level="info")
        logger.assert_not_logged("synthetic case for this test")
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

        YY00 is in nodedata here, with no usable size. That is the case worth a
        warning -- a real node whose workbook row contradicts itself -- and it is
        a different thing from the node simply not existing, below.
        """
        sizes = nodedata(XX00_reservoir=1_000_000.0, YY00_reservoir=1.0)
        sizes.loc[sizes["node"] == "YY00_reservoir", "upwardlimit"] = pd.NA

        result, logger = run(tmp_path, {"XX00": {}, "YY00": {}}, sizes)
        assert "XX00_reservoir" in set(result["node"])
        assert "YY00_reservoir" not in set(result["node"])
        logger.assert_logged("YY00_reservoir", level="warn")

    def test_a_node_the_model_does_not_have_is_never_mentioned(self, tmp_path):
        """nodedata says what exists, so its absence is not an absence to describe.

        This used to be guessed from whether PECD happened to carry ratios for the
        zone, which answers a different question and warned about nodes no run
        ever had.
        """
        result, logger = run(
            tmp_path,
            {"XX00": {}, "YY00": {}},
            nodedata(XX00_reservoir=1_000_000.0),  # YY00 is not in the model
        )
        assert "YY00_reservoir" not in set(result["node"])
        logger.assert_not_logged("YY00_reservoir")

    def test_a_zero_size_is_reported_rather_than_silently_dropped(self, tmp_path):
        result, logger = run(
            tmp_path,
            {"XX00": {}, "YY00": {}},
            nodedata(XX00_reservoir=1_000_000.0, YY00_reservoir=0.0),
        )
        assert "YY00_reservoir" not in set(result["node"])
        logger.assert_logged("YY00_reservoir", level="warn")

    def test_a_new_country_code_with_no_hydro_data_is_not_an_error(self, tmp_path):
        """FI00 split into FI01/FI02 produces codes PECD has never seen.

        Not reported either: the node keeps the constant bounds nodedata gives
        it, which is a complete answer. A node left with *no* bound by either
        route is caught in the workbook builder, which can see both.
        """
        result, logger = run(
            tmp_path,
            {"XX00": {}},
            nodedata(XX00_reservoir=1_000_000.0, FI01_reservoir=500_000.0),
            countries=["XX00", "FI01"],
        )
        assert "XX00_reservoir" in set(result["node"])
        assert "FI01_reservoir" not in set(result["node"])
        logger.assert_clean()
        logger.assert_not_logged("FI01_reservoir")

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

    def test_pumped_storage_without_level_data_is_not_mentioned(self, tmp_path):
        """PECD has no weekly levels for psOpen outside Norway, nor psClosed anywhere.

        Those nodes are constant-bounded by construction and are the same ones
        every run, so the build says nothing: which they are is in docs/hydro.md,
        and a node left with no bound at all is caught by the workbook builder,
        which can see the constants this processor cannot.
        """
        _, logger = run(
            tmp_path,
            {"XX00": {}},
            nodedata(XX00_reservoir=1_000_000.0, XX00_psOpen=5.0, XX00_psClosed=7.0),
            countries=["XX00"],
        )
        logger.assert_not_logged("XX00_psOpen")
        logger.assert_not_logged("XX00_psClosed")
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

    def _contribution(self, tmp_path):
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
        return processor.frames["boundarydata"]

    def test_it_states_which_boundaries_came_from_a_series(self, tmp_path):
        """The one thing about its own output this processor has to say.

        Nothing downstream can work it out: p_gnBoundaryProperties needs
        useTimeseries rather than the node's nodedata constant, and while
        changes.inc turns that flag off again for a series that proves flat,
        nothing ever turns it on.
        """
        contribution = self._contribution(tmp_path)

        assert set(contribution["param_gnboundarytypes"]) == {"upwardLimit", "downwardLimit"}
        assert set(contribution["usetimeseries"]) == {1}

    def test_the_contribution_is_keyed_the_way_the_boundary_table_is(self, tmp_path):
        # Without the key the merge has nothing to match on, and the flag would
        # be dropped at the contribution gate rather than reaching the workbook.
        contribution = self._contribution(tmp_path)

        assert set(CONTRIBUTION_KEYS["boundarydata"]) <= set(contribution.columns)
        assert set(contribution["grid"]) == {"reservoir"}

    def test_it_says_nothing_about_the_constant(self, tmp_path):
        """The flag is the claim; the number stays the workbook's.

        A constant here would be a second opinion on the reservoir size the
        processor read *from* nodedata to build the series in the first place.
        """
        contribution = self._contribution(tmp_path)

        assert "constant" not in contribution.columns


class TestDeclarations:
    def test_it_declares_the_source_data_it_needs(self):
        """Without this the cache cannot know a nodedata edit should rerun it."""
        assert hydro_storage_limits_MAF2019.requires_source_data == ("nodedata",)

    def test_stored_energy_is_declared_non_negative(self):
        assert hydro_storage_limits_MAF2019.value_sign == "non_negative"


#: A sawtooth: fill climbs all year and drops back at the year change. Every
#: week is distinct, and the wrap is the largest step in the series -- which is
#: what a climatological pattern replicated per calendar year actually does.
def rising(base: float, step: float):
    return lambda week: base + step * week


def weekly_anchors(result: pd.DataFrame, node: str, boundary_type: str) -> pd.Series:
    """The weekly anchor values behind the hourly output, in week order."""
    wide = result.pivot_table(
        index="time", columns=["node", "param_gnBoundaryTypes"], values="value"
    )[(node, boundary_type)]
    idx = [
        pd.Timestamp(year, 1, 4, 12) + pd.Timedelta(7 * k, unit="D")
        for year in range(START_YEAR, END_YEAR + 1)
        for k in range(52)
    ]
    idx = [t for t in idx if t in wide.index]
    return pd.Series(wide.reindex(idx).values, index=pd.DatetimeIndex(idx))


class TestYearChangeIsBounded:
    """The pattern is replicated per calendar year, so week 52 wraps onto week 1.

    Nothing used to verify what that wrap looked like. The rule now is that the
    step into week 1 must be no larger than one the profile already makes inside
    the year, measured as the 95th percentile so a single outlying week cannot
    license a seam as large as itself.
    """

    def test_a_flat_pattern_needs_no_blend(self, tmp_path):
        result, logger = run(tmp_path, {"XX00": {}}, nodedata(XX00_reservoir=1_000_000.0))
        assert not result.empty
        logger.assert_not_logged("blended")

    def test_a_sawtooth_is_blended_and_counted(self, tmp_path):
        """Counted, not explained: the reasoning is in docs/hydro.md and stays there."""
        result, logger = run(
            tmp_path,
            {"XX00": {"min_profile": rising(0.2, 0.004), "max_profile": rising(0.5, 0.004)}},
            nodedata(XX00_reservoir=1_000_000.0),
        )
        logger.assert_logged("year-change tail blended in 2 pattern(s)", level="info")

    def test_week_one_is_never_moved(self, tmp_path):
        """The model's year starts there, so it is the value that is trusted."""
        result, _ = run(
            tmp_path,
            {"XX00": {"min_profile": rising(0.2, 0.004), "max_profile": rising(0.5, 0.004)}},
            nodedata(XX00_reservoir=1_000_000.0),
        )
        anchors = weekly_anchors(result, "XX00_reservoir", "downwardLimit")
        week_one = anchors.loc[pd.Timestamp(START_YEAR, 1, 4, 12)]
        assert week_one == pytest.approx((0.2 + 0.004 * 1) * 1_000_000.0)

    def test_the_seam_ends_within_an_ordinary_weekly_step(self, tmp_path):
        result, _ = run(
            tmp_path,
            {"XX00": {"min_profile": rising(0.2, 0.004), "max_profile": rising(0.5, 0.004)}},
            nodedata(XX00_reservoir=1_000_000.0),
        )
        for boundary_type in ("downwardLimit", "upwardLimit"):
            anchors = weekly_anchors(result, "XX00_reservoir", boundary_type)
            by_year = {y: g for y, g in anchors.groupby(anchors.index.year)}
            interior = np.concatenate(
                [np.abs(np.diff(g.values)) for g in by_year.values() if len(g) > 1]
            )
            p95 = float(np.percentile(interior, 95))
            years = sorted(by_year)
            seam = max(
                abs(by_year[b].iloc[0] - by_year[a].iloc[-1])
                for a, b in zip(years, years[1:])
            )
            assert seam <= p95 * 1.000001, f"{boundary_type}: {seam} > {p95}"

    def test_only_the_tail_is_rewritten(self, tmp_path):
        """The blend is the shortest that works, so the rest of the year stands."""
        result, _ = run(
            tmp_path,
            {"XX00": {"min_profile": rising(0.2, 0.004), "max_profile": rising(0.5, 0.004)}},
            nodedata(XX00_reservoir=1_000_000.0),
        )
        anchors = weekly_anchors(result, "XX00_reservoir", "downwardLimit")
        midyear = anchors.loc[pd.Timestamp(START_YEAR, 1, 4, 12) + pd.Timedelta(7 * 25, unit="D")]
        assert midyear == pytest.approx((0.2 + 0.004 * 26) * 1_000_000.0)
