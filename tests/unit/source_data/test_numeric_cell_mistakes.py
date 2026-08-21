"""What a mistyped number does to a sheet.

A user put ``1,000.0`` into a cell and the build crashed. The crash turned out to
be the fortunate case. ``standardize_df_dtypes`` types a column numeric only when
``pd.to_numeric`` introduces no new NA, so **one** unparseable cell leaves the
whole column ``object`` -- and around a dozen places downstream branch on exactly
that dtype. Nothing was logged, so the usual outcome was not a traceback but a
quietly different model:

- ``filter_nonzero_numeric_rows`` stopped seeing the column as numeric and
  dropped **every row of the sheet**;
- ``normalize_dataframe``'s ``_output1`` rename only fires on ``Float64``
  columns, so ``capacity_output1`` stayed unrenamed and the capacity was never
  read at all;
- ``merge_row_by_row``'s ``add`` and ``multiply`` concatenated strings.

``read_input_excels`` now runs ``utils.gate_xlsx_frame`` over every sheet it
reads, which reports the cell and blanks it. The column is then typed as it was
always meant to be, and the value reads as "not set" -- which the pipeline
already knows how to carry, and which is the only honest reading of a cell
nobody can interpret. ``1.000`` is a thousand to a German author and one to an
English one, and the cell says nothing about which.

Level is ``error`` rather than ``warn`` on purpose: ``logger.has_errors`` feeds
``workflow_run_successfully``, so the build still produces its output but is
marked failed and re-runs fully next time. A warning would let a wrong capacity
through green.

The sibling file is ``test_dimension_string_mistakes.py``, which covers the
mistakes that fork a node instead of breaking a number.
"""

import pandas as pd
import pytest

import src.utils as utils
import src.source_data.source_data_loader as loader
from tests._common.fixtures import FakeLogger


def _gated(rows, *, source="units.xlsx:unitdata"):
    """Put rows through the gate the way read_input_excels does."""
    logger = FakeLogger()
    df = utils.gate_xlsx_frame(pd.DataFrame(list(rows)), source, logger)
    return df, logger


def _row(capacity, country="FI", column="capacity"):
    return {"country": country, "grid": "elec", "method": "replace", column: capacity}




class TestTheSheetSurvives:
    """The cascade, as the properties it used to break."""

    def test_the_other_rows_are_not_dropped(self):
        # Before the gate this kept 0 of 3: the poisoned column was no longer
        # numeric, so no row had any numeric value left to be non-zero.
        df, _ = _gated([_row("1,000.0"), _row(500.0, "SE"), _row(250.0, "NO")])
        kept = loader.filter_nonzero_numeric_rows(loader.normalize_dataframe(df, "u", FakeLogger()))
        assert set(kept["country"]) == {"SE", "NO"}

    def test_the_column_is_still_numeric(self):
        df, _ = _gated([_row("1,000.0"), _row(500.0, "SE"), _row(250.0, "NO")])
        out = loader.normalize_dataframe(df, "u", FakeLogger())
        assert pd.api.types.is_numeric_dtype(out["capacity"])

    def test_the_output1_suffix_is_still_stripped(self):
        # The rename only considers Float64 columns, so a poisoned column kept
        # its suffix and create_p_gnu_io never found it.
        df, _ = _gated([
            _row("1,000.0", column="capacity_output1"),
            _row(500.0, "SE", column="capacity_output1"),
        ])
        out = loader.normalize_dataframe(df, "u", FakeLogger())
        assert "capacity" in out.columns
        assert "capacity_output1" not in out.columns

    def test_the_good_values_are_untouched(self):
        df, _ = _gated([_row("1,000.0"), _row(500.0, "SE"), _row(250.0, "NO")])
        out = loader.normalize_dataframe(df, "u", FakeLogger())
        assert list(out.loc[out["country"] == "SE", "capacity"]) == [500.0]


class TestItIsReported:
    def test_an_error_is_logged(self):
        _, logger = _gated([_row("1,000.0"), _row(500.0, "SE")])
        assert logger.error_count == 1

    def test_the_message_names_the_sheet_and_the_column(self):
        _, logger = _gated([_row("1,000.0"), _row(500.0, "SE")], source="units.xlsx:unitdata")
        message = logger.errors[0]
        assert "units.xlsx:unitdata" in message
        assert "capacity" in message

    def test_the_message_shows_the_offending_value(self):
        # Without the value the reader has to go hunting through the workbook.
        _, logger = _gated([_row("1,000.0"), _row(500.0, "SE")])
        logger.assert_logged("1,000.0", level="error")

    def test_a_clean_sheet_says_nothing(self):
        _, logger = _gated([_row(1000.0), _row(500.0, "SE")])
        logger.assert_no_errors()


class TestWhatCountsAsAMalformedNumber:
    """The rule is *starts with a digit after any sign or currency symbol*.

    "Contains a digit" was tried first and was wrong: it blanked ``chp1`` out of
    an identifier column, which is the damage the gate exists to prevent.
    """

    @pytest.mark.parametrize(
        "value",
        [
            "1,000.0",       # US thousands -- the reported case
            "1 000",         # space as group separator
            "1 000",    # non-breaking space, which is what Excel pastes
            "1'000",         # Swiss
            "12,345,678",
            "1.000,5",       # grouped and comma-decimal together
            "1_000",
            "100 MW",        # a number wearing its unit
            "100MW",
            "5%",
            "€100",     # leading currency symbol
            "(500)",         # accounting negative
            "−5",       # U+2212 minus, not a hyphen
        ],
    )
    def test_these_are_blanked_and_reported(self, value):
        df, logger = _gated([_row(value), _row(500.0, "SE")])
        assert pd.isna(df.loc[0, "capacity"])
        assert logger.error_count == 1

    @pytest.mark.parametrize(
        "column_values",
        [
            ["1", "chp1", "chp2"],    # identifiers that merely contain digits
            ["1", "dh1", "dh2"],      # node suffixes
            ["1", "chp", "wind"],     # a text column holding a numeric-looking label
            ["chp", "wind", "pv"],
            ["   ", 500.0, 250.0],    # whitespace is emptiness, not a bad number
            ["-", 500.0, 250.0],      # a marker pandas does not know
            ["n.a.", 500.0, 250.0],
            ["1,5", "2,5", "3,5"],    # uniformly comma-decimal: a format, not a typo
        ],
    )
    def test_these_are_left_alone(self, column_values):
        rows = [_row(v, country=c) for v, c in zip(column_values, ["FI", "SE", "NO"])]
        df, logger = _gated(rows)
        assert list(df["capacity"]) == column_values
        logger.assert_no_errors()


class TestOddColumnShapesDoNotCrashTheGate:
    """The gate runs on every sheet, so it has to survive every column shape.

    An ``object`` column holding no strings at all is the one that bit: pandas'
    ``.str`` accessor raises ``AttributeError: Can only use .str accessor with
    string values!`` on a column of pure floats, bools or datetimes, and dtype
    alone does not distinguish that from a column of text. Real sheets produce
    the shape whenever a frame is built with ``dtype=object``, and it is the same
    obligation the all-NA rule places on every consumer: handle it, or reject it
    with a message, but never crash.
    """

    @pytest.mark.parametrize(
        "values",
        [
            [1.0, 2.0],                       # floats parked in an object column
            [1, 2],                           # ints
            [True, False],                    # bools
            [pd.Timestamp("2030-01-01"), pd.Timestamp("2031-01-01")],
            [pd.NA, pd.NA],                   # the assume-nothing column
            [1.0, pd.NA],
        ],
    )
    def test_it_neither_crashes_nor_complains(self, values):
        logger = FakeLogger()
        df = pd.DataFrame({"capacity": pd.Series(values, dtype=object)})

        out = utils.gate_xlsx_frame(df, "units.xlsx:unitdata", logger)

        assert len(out) == len(values)
        logger.assert_no_errors()

    def test_an_empty_frame_is_fine(self):
        logger = FakeLogger()
        out = utils.gate_xlsx_frame(pd.DataFrame(), "units.xlsx:unitdata", logger)
        assert out.empty
        logger.assert_no_errors()


class TestExcelErrorValues:
    """``#REF!`` is not a failed number -- it is a broken formula.

    It gets its own check because it is equally wrong in a text column, where no
    numeric rule would look. ``#REF!`` in particular is what Excel writes when
    someone deletes a column another sheet referred to, so it marks a workbook
    that has quietly lost a reference.
    """

    @pytest.mark.parametrize("value", ["#REF!", "#DIV/0!", "#VALUE!", "#N/A", "#NAME?"])
    def test_reported_in_a_numeric_column(self, value):
        df, logger = _gated([_row(value), _row(500.0, "SE")])
        assert pd.isna(df.loc[0, "capacity"])
        assert logger.error_count == 1

    def test_reported_in_a_text_column_too(self):
        df, logger = _gated([
            {"country": "#REF!", "grid": "elec", "method": "replace"},
            {"country": "SE", "grid": "elec", "method": "replace"},
        ])
        assert pd.isna(df.loc[0, "country"])
        assert logger.error_count == 1

    def test_an_ordinary_hash_value_is_not_an_error(self):
        # '# of units' is a person writing a note, not Excel reporting a failure.
        _, logger = _gated([
            {"country": "FI", "grid": "elec", "method": "replace", "note_text": "# of units"},
            {"country": "SE", "grid": "elec", "method": "replace", "note_text": "two"},
        ])
        logger.assert_no_errors()
