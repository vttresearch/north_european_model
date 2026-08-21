"""What ``BaseProcessor.read_input_csv`` refuses, and why it refuses instead of repairing.

This is where the reported bug actually came from: a helper UI started writing
the US thousands format into a processor's input CSV. In a comma-delimited file
that is not a formatting nuisance -- the comma **is** the delimiter, so the row
gains a field:

    node,value,unit
    FI,1,000.0,MW

pandas does not complain. It decides the file must have an index column, absorbs
the extra field, and returns a frame in which ``node`` is ``1``, ``value`` is
``'000.0'`` and ``unit`` is ``'MW'``. Every column has shifted and the node label
is gone. ``prepare_values_for_gdx`` rejects only *blank* dimension values and
``ProcessorRunner`` checks dimensions only for NA, so a numeric node label
travels all the way into the GDX as a set element. Nothing else in the pipeline
can see it, which is why the check has to happen at the read.

``index_col=False`` is what makes it visible: pandas then keeps the columns
aligned and emits a ``ParserWarning`` instead of silently re-interpreting the
file, and the helper turns that warning into a refusal.

Refuse, not repair
------------------
The source-workbook gate blanks a bad cell and carries on, because a hand-edited
sheet makes isolated typos. A generated file does not. One malformed number in
one means the producer changed format, so blanking would not rescue a stray cell
-- it would manufacture a column of zeros indistinguishable from real data.
Writing no GDX is the only honest outcome, and it is what
``find_time_axis_defects`` already does for the same reason.
"""

import pandas as pd
import pytest

from src.timeseries.processors.base_processor import BaseProcessor, SourceDataError
from tests._common.fixtures import FakeLogger


class _Reader(BaseProcessor):
    """Minimal concrete processor: the readers are what is under test."""

    def process(self):  # pragma: no cover - never called
        raise NotImplementedError


def _reader():
    logger = FakeLogger()
    return _Reader(logger=logger), logger


def _csv(tmp_path, text, name="input.csv"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


CLEAN = "node,value,unit\nFI,1000,MW\nSE,500,MW\n"


class TestACleanFileIsUntouched:
    def test_it_reads(self, tmp_path):
        reader, logger = _reader()
        df = reader.read_input_csv(_csv(tmp_path, CLEAN))
        assert list(df["node"]) == ["FI", "SE"]
        assert list(df["value"]) == [1000, 500]

    def test_it_says_nothing(self, tmp_path):
        reader, logger = _reader()
        reader.read_input_csv(_csv(tmp_path, CLEAN))
        logger.assert_clean()


class TestTheThousandsSeparator:
    def test_quoted_is_refused(self, tmp_path):
        # Quoted, the field count is right and the value survives as a string --
        # so this is caught by the numeric check rather than the field count.
        text = 'node,value,unit\nFI,"1,000.0",MW\nSE,500,MW\n'
        reader, logger = _reader()
        with pytest.raises(SourceDataError):
            reader.read_input_csv(_csv(tmp_path, text))
        logger.assert_logged("malformed number", level="error")

    def test_unquoted_is_refused(self, tmp_path):
        # The case that used to pass silently with every column shifted.
        text = "node,value,unit\nFI,1,000.0,MW\nSE,500,MW\n"
        reader, logger = _reader()
        with pytest.raises(SourceDataError):
            reader.read_input_csv(_csv(tmp_path, text))
        logger.assert_logged("Inconsistent number of fields", level="error")

    def test_unquoted_is_refused_even_when_the_shift_is_invisible(self, tmp_path):
        # With only two declared columns the extra field is absorbed as an index
        # and the frame looks entirely healthy: node=1, value=0.0.
        text = "node,value\nFI,1,000.0\nSE,500\n"
        reader, logger = _reader()
        with pytest.raises(SourceDataError):
            reader.read_input_csv(_csv(tmp_path, text))
        logger.assert_logged("Inconsistent number of fields", level="error")

    def test_the_message_explains_the_likely_cause(self, tmp_path):
        # A field-count error is unreadable without saying what usually causes it.
        text = "node,value\nFI,1,000.0\nSE,500\n"
        reader, logger = _reader()
        with pytest.raises(SourceDataError):
            reader.read_input_csv(_csv(tmp_path, text))
        logger.assert_logged("thousands separator", level="error")


class TestOtherMalformedNumbers:
    @pytest.mark.parametrize("value", ['"1 000"', '"100 MW"', '"5%"', '"1.000,5"'])
    def test_refused(self, tmp_path, value):
        text = f"node,value\nFI,{value}\nSE,500\n"
        reader, logger = _reader()
        with pytest.raises(SourceDataError):
            reader.read_input_csv(_csv(tmp_path, text))
        logger.assert_logged("malformed number", level="error")

    def test_an_excel_error_value_is_refused(self, tmp_path):
        text = "node,value\nFI,#REF!\nSE,500\n"
        reader, logger = _reader()
        with pytest.raises(SourceDataError):
            reader.read_input_csv(_csv(tmp_path, text))
        logger.assert_logged("Excel error value", level="error")


class TestMissingValueMarkers:
    """pandas knows the common ones; a bespoke marker has to be declared."""

    def test_the_markers_pandas_knows_need_no_declaration(self, tmp_path):
        text = "node,value\nFI,NA\nSE,500\n"
        reader, logger = _reader()
        df = reader.read_input_csv(_csv(tmp_path, text))
        assert pd.isna(df.loc[0, "value"])
        logger.assert_clean()

    def test_a_declared_marker_is_accepted(self, tmp_path):
        # 'n.a.' is not in pandas' default list, so the processor states it.
        text = "node,value\nFI,n.a.\nSE,500\n"
        reader, logger = _reader()
        df = reader.read_input_csv(_csv(tmp_path, text), na_values=["n.a."])
        assert pd.isna(df.loc[0, "value"])
        logger.assert_clean()


class TestTheRefusalStopsTheProcessor:
    """A logged error alone would not stop a GDX being written."""

    def test_it_raises_rather_than_returning_a_frame(self, tmp_path):
        text = 'node,value\nFI,"1,000.0"\nSE,500\n'
        reader, _ = _reader()
        with pytest.raises(SourceDataError):
            reader.read_input_csv(_csv(tmp_path, text))

    def test_the_error_is_recorded_at_error_level(self, tmp_path):
        # ProcessorRunner downgrades an escaping exception to a warning, so the
        # reader must have logged at error level itself for has_errors -- and
        # therefore the full-rerun flag -- to be set.
        text = 'node,value\nFI,"1,000.0"\nSE,500\n'
        reader, logger = _reader()
        with pytest.raises(SourceDataError):
            reader.read_input_csv(_csv(tmp_path, text))
        assert logger.has_errors


class TestIndexColIsNotNegotiable:
    def test_passing_index_col_is_rejected(self, tmp_path):
        # Allowing it back would reinstate the silent column shift this whole
        # helper exists to catch, so it fails loudly rather than being ignored.
        reader, _ = _reader()
        with pytest.raises(TypeError, match="index_col"):
            reader.read_input_csv(_csv(tmp_path, CLEAN), index_col=0)


class TestTheExcelReader:
    """Same numeric rule, no field-count check -- a sheet has cells, not delimiters."""

    def _write(self, tmp_path, value):
        path = tmp_path / "book.xlsx"
        pd.DataFrame({"node": ["FI", "SE"], "value": [value, 500.0]}).to_excel(
            path, index=False
        )
        return path

    def test_a_clean_sheet_reads(self, tmp_path):
        reader, logger = _reader()
        df = reader.read_input_excel(self._write(tmp_path, 1000.0))
        assert list(df["node"]) == ["FI", "SE"]
        logger.assert_clean()

    def test_a_malformed_number_is_refused(self, tmp_path):
        reader, logger = _reader()
        with pytest.raises(SourceDataError):
            reader.read_input_excel(self._write(tmp_path, "1,000.0"))
        logger.assert_logged("malformed number", level="error")
