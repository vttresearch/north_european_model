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

from tests._common.processor_contract import run_fake_processor

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
