"""Boundary 5: what a timeseries processor is allowed to hand back.

Processors are the pipeline's plugin surface -- users write their own -- so
every rejection path gets a misbehaving fake processor that triggers exactly it.

Each test asserts the **contract**, not the log line: no GDX file on disk and an
empty ``ProcessorRunResult``.  Checking only the message would have passed even
while the empty-DataFrame branch promised "No GDX output will be written" and
then carried straight on.

``ProcessorRunner`` loads processors by path from their own spec, so the fakes
are ordinary ``.py`` files in ``tmp_path``. No monkeypatching involved.
"""

import pandas as pd
import pytest

from tests._common.processor_contract import hourly_frame, run_fake_processor

# A well-formed two-row frame, used as the base that each misbehaving case breaks.
GOOD = (
    'pd.DataFrame({'
    '"grid": ["elec", "elec"], '
    '"node": ["FI00_elec", "FI00_elec"], '
    '"time": pd.to_datetime(["2014-01-01 00:00", "2014-01-01 01:00"]), '
    '"value": [1.0, 2.0]})'
)


class TestReturnShape:
    @pytest.mark.parametrize(
        "main_result",
        ['"a string"', "42", "None", "[1, 2, 3]", '{"grid": ["elec"]}'],
        ids=["str", "int", "None", "list", "dict"],
    )
    def test_anything_that_is_not_a_dataframe_is_rejected(self, tmp_path, main_result):
        run = run_fake_processor(tmp_path, main_result)
        run.logger.assert_logged("expected pd.DataFrame", level="error")
        run.assert_no_gdx_written()

    def test_an_empty_dataframe_is_rejected_and_actually_stops(self, tmp_path):
        """Regression: the message said "No GDX output will be written"...

        ...and then execution fell through into the curing block and onwards
        towards the writers. The promise is now kept.
        """
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": [], "node": [], "time": [], "value": []})',
        )
        run.logger.assert_logged("empty DataFrame", level="warn")
        run.assert_no_gdx_written()

    def test_an_emptiness_the_processor_ordered_is_not_a_warning(self, tmp_path):
        """`nothing_to_build` says the processor meant it and has said why.

        The runner cannot tell an emptiness that was ordered -- no unit uses this
        flow, so VRE_PECD builds nothing -- from one that is a failure, and
        warning on both is the noise "What a build says" exists to stop. The
        processor can tell, so it decides.
        """
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": [], "node": [], "time": [], "value": []})',
            nothing_to_build="True",
        )
        run.logger.assert_logged("empty DataFrame", level="info")
        run.logger.assert_no_errors()
        assert not run.logger.warnings

    def test_and_it_still_writes_no_gdx(self, tmp_path):
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": [], "node": [], "time": [], "value": []})',
            nothing_to_build="True",
        )
        run.assert_no_gdx_written()

    def test_missing_a_required_column_is_rejected(self, tmp_path):
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": ["elec"], '
            '"time": pd.to_datetime(["2014-01-01"]), "value": [1.0]})',
        )
        run.logger.assert_logged("unexpected columns", level="error")
        run.assert_no_gdx_written()

    def test_an_extra_column_is_rejected(self, tmp_path):
        # "nothing more, nothing less" -- an unexpected column usually means the
        # processor is emitting a different parameter than the spec declares.
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": ["elec"], "node": ["FI00_elec"], '
            '"extra": ["x"], "time": pd.to_datetime(["2014-01-01"]), '
            '"value": [1.0]})',
        )
        run.logger.assert_logged("unexpected columns", level="error")
        run.assert_no_gdx_written()

    def test_duplicate_dimension_time_rows_are_rejected(self, tmp_path):
        # Duplicates corrupt t-label assignment silently, which is why this is
        # an error rather than a de-duplication.
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": ["elec", "elec"], '
            '"node": ["FI00_elec", "FI00_elec"], '
            '"time": pd.to_datetime(["2014-01-01", "2014-01-01"]), '
            '"value": [1.0, 2.0]})',
        )
        run.logger.assert_logged("duplicate rows", level="error")
        run.assert_no_gdx_written()


class TestTimeAxis:
    """t-labels come from row position, so the row positions have to be right.

    ``split_timeseries_to_climate_windows`` gives row n of a group the label
    ``t{n+1}``. That is exact when the group holds one row per hour of the
    window, and undetectably wrong otherwise: the values that land on each label
    are all perfectly plausible, merely attached to the wrong hour.

    None of these four is caught by looking at values, which is why they are
    errors rather than warnings, and why they stop the GDX write.
    """

    def test_a_gap_is_rejected(self, tmp_path):
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": "elec", "node": "FI00_elec", '
            '"time": pd.date_range("2014-01-01", periods=48, freq="h").delete(10), '
            '"value": 1.0})',
        )
        run.logger.assert_logged("gap", level="error")
        run.assert_no_gdx_written()

    def test_the_gap_message_says_where_and_why(self, tmp_path):
        """The error text is the whole spec of this rule for most people.

        Someone adding a processor reads the message, not tests/README.md, so
        it has to name the hour and say what actually goes wrong -- otherwise
        "gap in the time axis" reads like a tidiness complaint.
        """
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": "elec", "node": "FI00_elec", '
            '"time": pd.date_range("2014-01-01", periods=48, freq="h").delete(10), '
            '"value": 1.0})',
        )
        message = run.logger.matching("gap")[0]
        assert "2014-01-01 11:00" in message
        assert "one label earlier" in message

    def test_sub_hourly_rows_are_rejected(self, tmp_path):
        """The case the old duplicate check could not see.

        00:00 and 00:15 are distinct timestamps, so ``duplicated()`` passed
        them, and row-position labelling then handed the quarter-hour the next
        model hour's label.
        """
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": "elec", "node": "FI00_elec", '
            '"time": pd.to_datetime(["2014-01-01 00:00", "2014-01-01 00:15", '
            '"2014-01-01 01:00"]), "value": 1.0})',
        )
        run.logger.assert_logged("duplicate rows", level="error")
        run.assert_no_gdx_written()

    def test_groups_covering_different_spans_are_rejected(self, tmp_path):
        """Every step is one hour and the frame is still fatal.

        Two nodes, each internally flawless, a day apart. They disagree about
        which real hour ``t000001`` names -- and for a model whose value is
        largely the correlation between countries, that is not a small error.
        """
        run = run_fake_processor(
            tmp_path,
            "pd.concat(["
            'pd.DataFrame({"grid": "elec", "node": "FI00_elec", '
            '"time": pd.date_range("2014-01-01", periods=48, freq="h"), "value": 1.0}), '
            'pd.DataFrame({"grid": "elec", "node": "SE00_elec", '
            '"time": pd.date_range("2014-01-02", periods=48, freq="h"), "value": 1.0})'
            "], ignore_index=True)",
        )
        run.logger.assert_logged("do not cover the same hours", level="error")
        run.assert_no_gdx_written()

    def test_a_missing_timestamp_is_rejected(self, tmp_path):
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": "elec", "node": "FI00_elec", '
            '"time": pd.to_datetime(["2014-01-01 00:00", None, "2014-01-01 02:00"]), '
            '"value": 1.0})',
        )
        run.logger.assert_logged("missing timestamp", level="error")
        run.assert_no_gdx_written()

    @pytest.mark.gams
    def test_a_clean_axis_still_writes_gdx(self, tmp_path):
        """The control. A gate that rejects everything would pass every test above.

        Two nodes over the full 48-hour window ``make_config`` declares, so this
        also pins that a complete frame draws no short-window warning.
        """
        run = run_fake_processor(
            tmp_path, hourly_frame(nodes=("FI00_elec", "SE00_elec"))
        )

        assert run.gdx_files, "a well-formed processor must still produce GDX"
        run.logger.assert_no_errors()
        assert not run.logger.matching("do not cover the full")


class TestDeclarationsOfIntent:
    """A processor says what its output should look like; the runner checks it.

    These replace the idea of a committed "this processor was checked" record.
    Such a record can only say that something passed once, against data the
    reader does not have -- and VRE_PECD reads whatever CSVs are in a
    config-supplied folder, so its inputs can change completely without a
    filename changing. A declaration claims nothing about the past, is checked
    against the data actually being processed, and cannot go stale.

    Breaches are warnings: an out-of-range value may be a real feature of the
    source data, where a broken time axis cannot be.
    """

    def _declaring(self, tmp_path, declaration, values="1.0"):
        # Whole numbers throughout: the fake spec rounds to 0 decimals, so a
        # fractional out-of-range value would be rounded back into range before
        # the check ever sees it -- which is correct, since the check judges
        # what actually gets written, but makes for a confusing test.
        return run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": "elec", "node": "FI00_elec", '
            '"time": pd.date_range("2014-01-01", periods=48, freq="h"), '
            f'"value": {values}}})',
            class_body=declaration,
        )

    def test_a_value_above_the_declared_maximum_warns(self, tmp_path):
        run = self._declaring(
            tmp_path, "value_range = (0.0, 1.0)", values="[1.0] * 47 + [5.0]"
        )
        run.logger.assert_logged("at most 1.0", level="warn")

    def test_a_value_below_the_declared_minimum_warns(self, tmp_path):
        run = self._declaring(
            tmp_path, "value_range = (0.0, 1.0)", values="[-5.0] + [1.0] * 47"
        )
        run.logger.assert_logged("at least 0.0", level="warn")

    def test_values_inside_the_declared_range_say_nothing(self, tmp_path):
        run = self._declaring(tmp_path, "value_range = (0.0, 1.0)", values="0.5")
        assert not run.logger.matching("value_range")
        assert not run.logger.matching("at most")
        assert not run.logger.matching("at least")

    def test_a_declared_sign_is_checked(self, tmp_path):
        run = self._declaring(
            tmp_path, 'value_sign = "non_negative"', values="[-1.0] + [1.0] * 47"
        )
        run.logger.assert_logged("non-negative", level="warn")

    def test_declaring_nothing_asserts_nothing(self, tmp_path):
        """The defaults have to be inert, or adding a processor means adding
        declarations before it will run quietly."""
        run = self._declaring(tmp_path, "pass", values="-99999.0")
        run.logger.assert_no_errors()
        assert not run.logger.matching("declares")

    def test_a_breach_does_not_stop_the_run(self, tmp_path):
        # Content, not form: the value may be right and the declaration stale.
        run = self._declaring(
            tmp_path, "value_range = (0.0, 1.0)", values="[1.0] * 47 + [1.5]"
        )
        run.logger.assert_no_errors()
        run.logger.assert_logged("Processing completed")

    def test_a_malformed_declaration_is_reported_not_obeyed(self, tmp_path):
        """A processor author who writes the attribute wrongly should hear about
        it, rather than get silence that reads like approval."""
        run = self._declaring(tmp_path, "value_range = 1.0", values="500.0")
        run.logger.assert_logged("not a (minimum, maximum) pair", level="warn")
        run.logger.assert_no_errors()

    def test_an_unknown_sign_is_reported_not_obeyed(self, tmp_path):
        run = self._declaring(tmp_path, 'value_sign = "positive"', values="1.0")
        run.logger.assert_logged("not one of", level="warn")

    def test_gaps_in_value_do_not_trip_the_range_check(self, tmp_path):
        """NaN means "no data" until the GDX gate, and must not read as 0."""
        run = self._declaring(
            tmp_path, "value_range = (0.5, 1.0)", values="[np.nan] * 24 + [0.7] * 24"
        )
        assert not run.logger.matching("at least")


class TestDimensionValues:
    @pytest.mark.parametrize(
        "bad", ["None", "np.nan", "pd.NA"], ids=["None", "np.nan", "pd.NA"]
    )
    def test_a_missing_dimension_value_is_rejected(self, tmp_path, bad):
        """Dimension values become GAMS set elements, so a blank one is a broken key.

        Previously ``fill_all_na`` turned these into the empty string and the
        run continued, silently adding a node named "" to the model.
        """
        run = run_fake_processor(
            tmp_path,
            f'pd.DataFrame({{"grid": ["elec", "elec"], '
            f'"node": ["FI00_elec", {bad}], '
            f'"time": pd.to_datetime(["2014-01-01 00:00", "2014-01-01 01:00"]), '
            f'"value": [1.0, 2.0]}})',
        )
        run.logger.assert_logged("missing 'node'", level="error")
        run.assert_no_gdx_written()

    def test_the_error_names_the_offending_dimension(self, tmp_path):
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": [None], "node": ["FI00_elec"], '
            '"time": pd.to_datetime(["2014-01-01"]), "value": [1.0]})',
        )
        run.logger.assert_logged("missing 'grid'", level="error")


class TestValueColumn:
    def test_a_non_numeric_value_column_is_rejected(self, tmp_path):
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": ["elec"], "node": ["FI00_elec"], '
            '"time": pd.to_datetime(["2014-01-01"]), "value": ["not a number"]})',
        )
        run.logger.assert_logged("non-numeric 'value'", level="error")
        run.assert_no_gdx_written()

    def test_missing_values_are_NOT_rejected(self, tmp_path):
        """NaN in `value` means "no data" and is legal at this boundary.

        It stays NaN until the GDX gate converts it to 0 and reports the count.
        Rejecting it here would force processors to invent zeros, which is
        precisely the information loss this design is trying to avoid -- and it
        would bias the climatological quantiles computed further downstream.
        """
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": ["elec", "elec"], '
            '"node": ["FI00_elec", "FI00_elec"], '
            '"time": pd.to_datetime(["2014-01-01 00:00", "2014-01-01 01:00"]), '
            '"value": [1.0, float("nan")]})',
        )
        run.logger.assert_not_logged("non-numeric 'value'")
        run.logger.assert_not_logged("missing 'node'")


class TestTimeColumn:
    def test_a_convertible_time_column_is_accepted_but_warned_about(self, tmp_path):
        # Forgiving rather than strict: this used to be coerced silently, so
        # processors had no way to learn they were off-contract.
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": ["elec", "elec"], '
            '"node": ["FI00_elec", "FI00_elec"], '
            '"time": ["2014-01-01 00:00", "2014-01-01 01:00"], '
            '"value": [1.0, 2.0]})',
        )
        run.logger.assert_logged("converted to datetime", level="warn")

    def test_an_unconvertible_time_column_is_rejected(self, tmp_path):
        run = run_fake_processor(
            tmp_path,
            'pd.DataFrame({"grid": ["elec", "elec"], '
            '"node": ["FI00_elec", "FI00_elec"], '
            '"time": ["not a date", "also not a date"], '
            '"value": [1.0, 2.0]})',
        )
        run.logger.assert_logged("could not be converted", level="error")
        run.assert_no_gdx_written()


class TestModuleLevelFailures:
    def test_a_processor_that_raises_is_caught(self, tmp_path):
        run = run_fake_processor(
            tmp_path, GOOD, body='raise RuntimeError("processor exploded")'
        )
        run.logger.assert_logged("raised an exception", level="warn")
        run.assert_no_gdx_written()

    def test_a_module_without_a_matching_class_is_rejected(self, tmp_path):
        # The module file and the class inside it must share a name.
        run = run_fake_processor(
            tmp_path,
            GOOD,
            name="Expected",
            raw_source="class SomethingElse:\n    pass\n",
        )
        run.logger.assert_logged("missing a class named", level="warn")
        run.assert_no_gdx_written()


class TestHashIsAlwaysUpdated:
    @pytest.mark.parametrize(
        "main_result",
        ['"a string"', 'pd.DataFrame()', GOOD],
        ids=["not-a-frame", "empty", "valid"],
    )
    def test_the_processor_hash_is_recorded_on_every_exit_path(self, tmp_path, main_result):
        """Otherwise a broken processor re-runs forever.

        The cache decides what to re-run by comparing processor file hashes. A
        rejection path that skipped the hash update would leave the processor
        permanently "changed", so every subsequent build would re-run it and
        fail again.
        """
        run = run_fake_processor(tmp_path, main_result)
        assert "FakeProcessor" in run.cache_manager.processor_hashes
