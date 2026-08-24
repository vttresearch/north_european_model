"""Reading a folder of PECD capacity factors that the user assembled themselves.

Everything hard about this processor comes from one fact: **a PECD download is a
set of about ten choices, and none of them is written into the CSV body.** Two
files from different downloads carry the same columns on the same hourly index
and differ only in their values, so nothing inside the data can tell them apart.
The file name is the only record, which is why the tests below spend so much of
their length on names.

Three behaviours follow from that and are pinned here:

- a folder holding two downloads is **warned about, not refused** -- the files
  are well-formed and which download is right depends on the scenario being
  modelled, which this processor has no way to know;
- a folder in which two files cover the same hour **is** refused, because the
  winner would be decided by directory order;
- and the warning has to fire on the *non-overlapping* case too, which is the
  one that otherwise ships: 4.1 for one year and 4.2 for the next is a
  perfectly continuous series with a step in the middle of it.

The zone choice is the other half. PECD's wind zones are finer than this model's
nodes, so a code like ``FR00`` is resolved by prefix and the highest-output
candidate wins. That is a modelling decision rather than a lookup, and the test
that matters is that it is taken over the whole configured window rather than
off whichever file sorted first. ``docs/vre-timeseries.md`` has the numbers.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.timeseries.processors.base_processor import SourceDataError
from src.timeseries.processors.VRE_PECD import VRE_PECD
from tests._common.fixtures import FakeLogger
from tests._common.processor_contract import assert_processor_conforms

#: A PECD comment block, trimmed to the keys the processor reads. The real files
#: carry about fifty lines of provenance above the header row.
HEADER_BLOCK = """\
# General
## Title
### {title}
## Unit
### {unit}
## Temporal extent
## Begin date
### {begin}
## End date
### {end}
## Temporal resolution
### 1 hour
#
"""


def pecd_name(
    year: int,
    *,
    technology: str = "WON",
    detail: str = "NA---",
    spatial: str = "PEON",
    variant: str = "NA-",
    hub_height: str = "30",
    regridding: str = "NA---",
    physical_model: str = "PhM02",
    version: str = "PECD4.1",
) -> str:
    """A PECD file name: 22 underscore-separated fields, S/E in slots 9 and 10.

    The shipped 4.1 and the 4.2 sample differ in exactly four of them --
    ``variant``, ``regridding``, ``physical_model``, ``version`` -- and in
    nothing else, which is the whole difficulty.
    """
    return "_".join([
        "H", "ERA5", "ECMW", "T639", technology, detail, "Pecd", spatial,
        f"S{year}01010000", f"E{year}12312300", "CFR", "TIM", "01h", variant,
        "noc", "org", hub_height, "NA---", regridding, physical_model,
        version, "fv1",
    ]) + ".csv"


def write_pecd(
    folder: Path,
    year: int,
    zones: dict[str, float | np.ndarray],
    *,
    name: str | None = None,
    title: str = "Wind Power Onshore",
    unit: str = "MW/MW",
    begin: str | None = None,
    end: str | None = None,
    **name_fields,
) -> Path:
    """Write one synthetic PECD year and return its path.

    ``zones`` maps a column name to either a constant capacity factor or a
    full-length array. A ``None`` value writes the column empty, which is how a
    zone that exists in the header but carries no data reaches the processor --
    the case that would otherwise merely lose the highest-sum comparison rather
    than being excluded from it.
    """
    folder.mkdir(parents=True, exist_ok=True)
    index = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="60min")

    frame = pd.DataFrame(index=index)
    for zone, value in zones.items():
        if value is None:
            frame[zone] = np.nan
        elif np.isscalar(value):
            frame[zone] = float(value)
        else:
            frame[zone] = np.asarray(value, dtype=float)

    path = folder / (name or pecd_name(year, **name_fields))
    block = HEADER_BLOCK.format(
        title=title,
        unit=unit,
        begin=begin or f"{year}-01-01-0000",
        end=end or f"{year}-12-31-2300",
    )
    body = frame.to_csv(index_label="Date", lineterminator="\n")
    path.write_text(block + body, encoding="utf-8")
    return path


def build(folder: Path, *, codes=("FI00",), start=2013, end=2013,
          flow="onshore", cutoff=0.01, precision=5):
    """Run the processor over a folder and return (result, logger)."""
    logger = FakeLogger()
    processor = VRE_PECD(
        input_folder=str(folder),
        country_codes=list(codes),
        start_year=start,
        end_year=end,
        attached_grid="elec",
        custom_column_value={"flow": flow},
        rounding_precision=precision,
        cutoff_below=cutoff,
        scaling_factor=1,
        logger=logger,
    )
    return processor.process(), logger


class TestTheFileNameIsTheOnlyRecord:
    """The ten download choices reach the processor through the name or not at all."""

    @pytest.mark.parametrize(
        "version, variant, regridding, physical_model",
        [
            ("PECD4.1", "NA-", "NA---", "PhM02"),
            ("PECD4.2", "COM", "ReGrB", "PhM04"),
        ],
    )
    def test_both_shipped_shapes_parse(self, tmp_path, version, variant,
                                       regridding, physical_model):
        path = write_pecd(
            tmp_path, 2013, {"FI00": 0.3},
            version=version, variant=variant,
            regridding=regridding, physical_model=physical_model,
        )
        processor = VRE_PECD(
            input_folder=str(tmp_path), country_codes=["FI00"],
            start_year=2013, end_year=2013, attached_grid="elec",
            logger=FakeLogger(),
        )
        fields, reason = processor._parse_filename(str(path))

        assert reason is None
        assert fields["technology"] == "WON"
        assert fields["pecd_version"] == version
        assert fields["technology_variant"] == variant
        assert fields["regridding"] == regridding
        assert fields["physical_model"] == physical_model
        assert fields["window_start"] == "S201301010000"

    def test_the_selection_is_logged_every_run(self, tmp_path):
        """A build is only reproducible from its log if the log says what went in."""
        write_pecd(tmp_path, 2013, {"FI00": 0.3},
                   version="PECD4.2", variant="COM", physical_model="PhM04")
        _, logger = build(tmp_path)

        logger.assert_logged("PECD onshore", level="info")
        logger.assert_logged("PECD4.2")
        logger.assert_logged("variant COM")

    def test_a_name_that_does_not_parse_is_skipped_and_said_so(self, tmp_path):
        """Not included on the grounds that it might be fine.

        "Include it to be safe" is exactly what lets a file from another dataset
        into the folder: a name this cannot read is a file whose provenance is
        unknown, and there is nothing else to consult.
        """
        write_pecd(tmp_path, 2013, {"FI00": 0.3}, name="capacity_factors_2013.csv")
        result, logger = build(tmp_path)

        logger.assert_logged("capacity_factors_2013.csv", level="warn")
        logger.assert_logged("provenance is unknown", level="warn")
        assert result.empty

    def test_a_header_that_contradicts_the_name_is_reported(self, tmp_path):
        write_pecd(tmp_path, 2013, {"FI00": 0.3}, begin="2011-01-01-0000")
        _, logger = build(tmp_path)

        logger.assert_logged("Begin date", level="warn")

    def test_a_unit_that_is_not_a_ratio_is_reported(self, tmp_path):
        write_pecd(tmp_path, 2013, {"FI00": 0.3}, unit="MWh")
        _, logger = build(tmp_path)

        logger.assert_logged("MW/MW", level="warn")


class TestOneFolderTwoDownloads:
    """Warned about, never refused -- and the warning has to reach both shapes."""

    def test_files_from_two_downloads_are_warned_about(self, tmp_path):
        write_pecd(tmp_path, 2012, {"FI00": 0.30})
        write_pecd(tmp_path, 2013, {"FI00": 0.45},
                   version="PECD4.2", variant="COM",
                   regridding="ReGrB", physical_model="PhM04")
        _, logger = build(tmp_path, start=2012, end=2013)

        logger.assert_logged("2 different PECD downloads", level="warn")
        logger.assert_logged("pecd_version", level="warn")

    def test_a_non_overlapping_blend_still_builds(self, tmp_path):
        """The case that otherwise ships.

        4.1 for one year and 4.2 for the next overlap on no hour at all, so
        nothing is refused and the series is continuous and complete. It is also
        joined from two different downloads with a step in the middle, and the
        warning is the only thing that says so -- on the real files that step is
        tens of percent.
        """
        write_pecd(tmp_path, 2012, {"FI00": 0.30})
        write_pecd(tmp_path, 2013, {"FI00": 0.45},
                   version="PECD4.2", variant="COM",
                   regridding="ReGrB", physical_model="PhM04")
        result, logger = build(tmp_path, start=2012, end=2013)

        logger.assert_logged("2 different PECD downloads", level="warn")
        assert not result.empty
        assert result["value"].notna().all()

        # Both halves are present and the step survives into the output, which
        # is what makes the warning worth reading.
        per_year = result.groupby(result["time"].dt.year)["value"].mean()
        assert per_year[2012] == pytest.approx(0.30, abs=1e-4)
        assert per_year[2013] == pytest.approx(0.45, abs=1e-4)

    def test_one_download_over_several_years_is_not_warned_about(self, tmp_path):
        """The ordinary case: 35 files, one download, nothing to say."""
        for year in (2012, 2013, 2014):
            write_pecd(tmp_path, year, {"FI00": 0.3})
        _, logger = build(tmp_path, start=2012, end=2014)

        logger.assert_not_logged("different PECD downloads")

    def test_the_differing_fields_are_named(self, tmp_path):
        """Two downloads can differ in one field of twenty-two."""
        write_pecd(tmp_path, 2012, {"FI00": 0.30})
        write_pecd(tmp_path, 2013, {"FI00": 0.45}, hub_height="100")
        _, logger = build(tmp_path, start=2012, end=2013)

        logger.assert_logged("hub_height", level="warn")


class TestOverlappingFilesAreRefused:
    """The one refusal, and it is a different thing from a blend."""

    def test_two_files_covering_the_same_hours_stop_the_processor(self, tmp_path):
        write_pecd(tmp_path, 2013, {"FI00": 0.30})
        write_pecd(tmp_path, 2013, {"FI00": 0.45},
                   version="PECD4.2", variant="COM", physical_model="PhM04")

        with pytest.raises(SourceDataError, match="cover the same hours"):
            build(tmp_path)

    def test_the_overlapping_files_and_span_are_named(self, tmp_path):
        write_pecd(tmp_path, 2013, {"FI00": 0.30})
        write_pecd(tmp_path, 2013, {"FI00": 0.45}, hub_height="100")

        logger = FakeLogger()
        processor = VRE_PECD(
            input_folder=str(tmp_path), country_codes=["FI00"],
            start_year=2013, end_year=2013, attached_grid="elec",
            logger=logger,
        )
        with pytest.raises(SourceDataError):
            processor.process()

        logger.assert_logged("2013-01-01", level="error")
        logger.assert_logged("No GDX output will be written", level="error")

    def test_consecutive_years_do_not_count_as_overlapping(self, tmp_path):
        """A year ends at 23:00 on 31 December and the next starts at 00:00."""
        write_pecd(tmp_path, 2012, {"FI00": 0.3})
        write_pecd(tmp_path, 2013, {"FI00": 0.3})
        result, logger = build(tmp_path, start=2012, end=2013)

        logger.assert_not_logged("cover the same hours")
        assert not result.empty


class TestTheTechnologyMustMatchTheSpec:
    def test_pv_files_under_an_onshore_spec_are_reported(self, tmp_path):
        """The folder and the flow are configured separately, two lines apart."""
        write_pecd(tmp_path, 2013, {"FI00": 0.15},
                   technology="SPV", detail="0000m", spatial="SZON",
                   title="Solar PV Power")
        _, logger = build(tmp_path, flow="onshore")

        logger.assert_logged("'SPV'", level="warn")
        logger.assert_logged("flow 'onshore'", level="warn")

    def test_matching_technology_says_nothing(self, tmp_path):
        write_pecd(tmp_path, 2013, {"FI00": 0.3})
        _, logger = build(tmp_path, flow="onshore")

        logger.assert_not_logged("Check input_sub_folder")


class TestChoosingAZone:
    """Highest output wins -- decided over the window, not off the first file."""

    def test_the_winner_is_decided_over_the_whole_window(self, tmp_path):
        """The bug this replaces.

        ``FR01`` beats ``FR02`` in the first year and loses over the three, so
        reading the mapping off the first file gives a different answer from
        reading it off the data. It also meant the choice moved whenever
        ``bb_timeseries_start`` moved, with nothing said about it.
        """
        write_pecd(tmp_path, 2012, {"FR01": 0.40, "FR02": 0.10})
        write_pecd(tmp_path, 2013, {"FR01": 0.10, "FR02": 0.40})
        write_pecd(tmp_path, 2014, {"FR01": 0.10, "FR02": 0.40})

        result, logger = build(tmp_path, codes=["FR00"], start=2012, end=2014)

        logger.assert_logged("FR00->FR02", level="info")
        assert result["value"].mean() == pytest.approx(0.30, abs=1e-3)

    def test_the_choice_and_its_size_are_reported(self, tmp_path):
        write_pecd(tmp_path, 2013, {"FR01": 0.40, "FR02": 0.20, "FR03": 0.15})
        _, logger = build(tmp_path, codes=["FR00"])

        logger.assert_logged("take the best of several PECD zones", level="info")
        logger.assert_logged("FR00->FR01", level="info")

    def test_a_candidate_with_no_values_cannot_win_by_summing_to_zero(self, tmp_path):
        """An all-NaN column sums to 0.0, so it only *loses* the comparison.

        That is indistinguishable from a zone that is genuinely calm, and a
        partially empty column is biased down by the same mechanism.
        """
        write_pecd(tmp_path, 2013, {"FR01": None, "FR02": 0.25})
        result, logger = build(tmp_path, codes=["FR00"])

        logger.assert_logged("Empty candidate column(s) ignored", level="info")
        logger.assert_logged("FR01", level="info")
        assert result["value"].mean() == pytest.approx(0.25, abs=1e-3)

    def test_a_code_whose_candidates_are_all_empty_is_not_built(self, tmp_path):
        write_pecd(tmp_path, 2013, {"FR01": None, "FR02": None})
        result, logger = build(tmp_path, codes=["FR00"])

        logger.assert_logged("hold no values", level="warn")
        assert result.empty

    def test_a_two_letter_match_is_warned_about(self, tmp_path):
        """The tier that can hand a node another zone's weather."""
        write_pecd(tmp_path, 2013, {"FR99": 0.3})
        _, logger = build(tmp_path, codes=["FR00"])

        logger.assert_logged("first two letters", level="warn")

    def test_zones_no_prefix_can_reach_are_named(self, tmp_path):
        """A prefix is arithmetic: 'FR0' cannot see FR10.

        Nothing else in the pipeline can notice, because the column is simply
        never looked at.
        """
        write_pecd(tmp_path, 2013, {"FR01": 0.3, "FR10": 0.9})
        _, logger = build(tmp_path, codes=["FR00"])

        logger.assert_logged("unreachable by prefix matching", level="info")

    def test_another_nodes_zone_is_not_called_orphaned(self, tmp_path):
        """``SE02`` is its own node, not a zone ``SE01`` is missing out on."""
        write_pecd(tmp_path, 2013, {"SE01": 0.3, "SE02": 0.4})
        _, logger = build(tmp_path, codes=["SE01", "SE02"])

        logger.assert_not_logged("unreachable by prefix matching")


class TestSayingWhatWasNotBuilt:
    def test_a_code_with_no_column_is_named(self, tmp_path):
        """The zero that matters here is a whole series, not an hour.

        A unit on that node can never generate for the entire run, which
        downstream is indistinguishable from a unit nobody asked for.
        """
        write_pecd(tmp_path, 2013, {"FI00": 0.3})
        result, logger = build(tmp_path, codes=["FI00", "AT00"])

        logger.assert_logged("AT00", level="warn")
        logger.assert_logged("cannot generate", level="warn")
        assert set(result["node"]) == {"FI00_elec"}

    def test_what_was_built_is_named_too(self, tmp_path):
        write_pecd(tmp_path, 2013, {"FI00": 0.3})
        _, logger = build(tmp_path, codes=["FI00"])

        logger.assert_logged("Capacity factors built for 1", level="info")


class TestIsolatedDropouts:
    """One flat hour between two ordinary ones -- reported, never repaired."""

    @staticmethod
    def _series(year: int, dropouts: dict[int, tuple[float, float]]) -> np.ndarray:
        """A steady profile with (value, neighbour) pairs punched into it."""
        index = pd.date_range(f"{year}-01-01", f"{year}-12-31 23:00", freq="60min")
        values = np.full(len(index), 0.30)
        for hour, (value, neighbour) in dropouts.items():
            values[hour - 1] = neighbour
            values[hour] = value
            values[hour + 1] = neighbour
        return values

    def test_a_flat_hour_between_two_windy_ones_is_reported(self, tmp_path):
        write_pecd(tmp_path, 2013,
                   {"FI00": self._series(2013, {100: (0.0, 0.5)})})
        _, logger = build(tmp_path)

        logger.assert_logged("isolated empty hour", level="warn")
        logger.assert_logged("Values unchanged", level="warn")

    def test_a_calm_spell_is_not_reported(self, tmp_path):
        """The reason the magnitude test exists.

        The source rounds to five decimals, so a genuine calm spell produces
        runs of values a hair above zero. A rule that ignored magnitude would
        fire on hundreds of them every build and teach people to skip warnings.
        """
        index = pd.date_range("2013-01-01", "2013-12-31 23:00", freq="60min")
        values = np.full(len(index), 0.30)
        values[100:120] = [0.004, 0.0, 0.003, 0.0, 0.002, 0.0, 0.001, 0.0,
                           0.002, 0.0, 0.003, 0.0, 0.001, 0.0, 0.004, 0.0,
                           0.002, 0.0, 0.003, 0.0]
        write_pecd(tmp_path, 2013, {"FI00": values})
        _, logger = build(tmp_path)

        logger.assert_not_logged("isolated empty hour")

    def test_the_values_reach_the_output_unchanged(self, tmp_path):
        write_pecd(tmp_path, 2013,
                   {"FI00": self._series(2013, {100: (0.0, 0.5)})})
        result, _ = build(tmp_path)

        at_dropout = result.loc[
            result["time"] == pd.Timestamp("2013-01-05 04:00"), "value"
        ]
        assert float(at_dropout.iloc[0]) == 0.0

    def test_the_threshold_follows_cutoff_below(self, tmp_path):
        """``cutoff_below`` is the user's to set, so it cannot be a literal.

        The second dropout sits at 0.02: above the shipped 0.01 cutoff, so it is
        an ordinary small value and not a dropout at all -- and below a 0.05
        cutoff, which zeroes it before GAMS ever sees it. A hard-coded 0.01
        would be quietly wrong for exactly the user who tuned the parameter.
        """
        write_pecd(tmp_path, 2013, {
            "FI00": self._series(2013, {100: (0.0, 0.6), 200: (0.02, 0.6)}),
        })

        _, lenient = build(tmp_path, cutoff=0.01)
        _, strict = build(tmp_path, cutoff=0.05)

        assert "1 isolated empty hour" in lenient.matching("isolated empty hour")[0]
        assert "2 isolated empty hour" in strict.matching("isolated empty hour")[0]


class TestTheOutputContract:
    def test_a_pecd_4_2_folder_goes_end_to_end(self, tmp_path):
        """No branch anywhere reads the version; either format is just data."""
        write_pecd(tmp_path, 2013, {"FI00": 0.3, "AT01": 0.25, "AT02": 0.20},
                   version="PECD4.2", variant="COM",
                   regridding="ReGrB", physical_model="PhM04",
                   title="Wind Power Onshore - Onshore Existing technologies")

        frame = assert_processor_conforms(
            VRE_PECD,
            dimensions=["flow", "node", "f", "t"],
            input_folder=str(tmp_path),
            country_codes=["FI00", "AT00"],
            start_year=2013,
            end_year=2013,
            attached_grid="elec",
            custom_column_value={"flow": "onshore"},
            logger=FakeLogger(),
        )

        assert set(frame["node"]) == {"FI00_elec", "AT00_elec"}
        assert set(frame["flow"]) == {"onshore"}
        assert frame["value"].between(0.0, 1.0).all()

    def test_no_scaling_factor_means_no_scaling(self, tmp_path):
        """The default lives in the processor, not only in ``config_reader``.

        Anything constructed without going through the config used to arrive
        with ``scaling_factor=None``, and ``None != 1`` is true, so it entered
        the scaling branch and multiplied the mean by ``None``.
        """
        write_pecd(tmp_path, 2013, {"FI00": 0.3})

        logger = FakeLogger()
        processor = VRE_PECD(
            input_folder=str(tmp_path), country_codes=["FI00"],
            start_year=2013, end_year=2013, attached_grid="elec",
            logger=logger,
        )
        result = processor.process()

        assert result["value"].mean() == pytest.approx(0.3, abs=1e-4)

    def test_an_empty_folder_is_a_warning_not_a_crash(self, tmp_path):
        result, logger = build(tmp_path)

        logger.assert_logged("No CSV files found", level="warn")
        assert list(result.columns) == ["flow", "node", "time", "value"]

    def test_a_missing_folder_is_a_warning_not_a_crash(self, tmp_path):
        result, logger = build(tmp_path / "absent")

        logger.assert_logged("does not exist", level="warn")
        assert result.empty
