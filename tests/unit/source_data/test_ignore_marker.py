"""``##`` marks something in a workbook as the author's, not the model's.

One marker, two placements: in a **column header** it ignores the column, in a
**data row** it ignores the row. The symmetry is the point -- there is one thing
to learn, and it has to be typed on purpose, so nothing is skipped by accident.

Why the column half had to exist
--------------------------------
Authors keep a helper table beside the real one: intermediate sums, a formula
being built, a column of pasted values to compare against. Those columns are not
input, but nothing said so, and the numeric gate added alongside this would have
reported every half-finished formula and every ``#DIV/0!`` in them as broken
input data. A ``note`` column was the only thing ever excluded, by exact name --
which does not survive an author wanting a second one, since pandas renames the
duplicate to ``note.1``.

Why the column drop happens first
---------------------------------
Before the numeric gate, so scratch work is never validated. Before the
blank-header handling, so a spacer column between the real table and the helper
table has nothing named to its right once the helpers are gone, and reads as the
end of the sheet rather than a hole in it.
"""

import pandas as pd
import pytest

import src.source_data.source_data_loader as loader
from tests._common.fixtures import FakeLogger


def _sheet(tmp_path, frame, *, name="unitdata", file_name="book.xlsx"):
    path = tmp_path / file_name
    frame.to_excel(path, sheet_name=name, index=False)
    return path


def _read(tmp_path, frame, *, name="unitdata"):
    _sheet(tmp_path, frame, name=name)
    logger = FakeLogger()
    frames = loader.read_input_excels(tmp_path, ["book.xlsx"], name, logger)
    return (frames[0] if frames else pd.DataFrame()), logger


def _model_columns(df):
    """Columns the pipeline will actually work with."""
    return [c for c in df.columns if not str(c).startswith("_")]


class TestAMarkedColumnIsIgnored:
    def test_it_does_not_reach_the_frame(self, tmp_path):
        frame = pd.DataFrame({
            "country": ["FI", "SE"],
            "capacity": [1000.0, 500.0],
            "## scratch": ["wip", "wip"],
        })
        df, _ = _read(tmp_path, frame)
        assert _model_columns(df) == ["country", "capacity"]

    def test_the_real_columns_are_untouched(self, tmp_path):
        frame = pd.DataFrame({
            "country": ["FI", "SE"],
            "capacity": [1000.0, 500.0],
            "## scratch": ["wip", "wip"],
        })
        df, _ = _read(tmp_path, frame)
        assert list(df["capacity"]) == [1000.0, 500.0]

    def test_several_marked_columns_all_go(self, tmp_path):
        # The case a single hardcoded 'note' column could not handle: pandas
        # renames repeats to note.1, note.2, and only the first ever matched.
        frame = pd.DataFrame({
            "country": ["FI"],
            "capacity": [1000.0],
            "## note": ["a"],
            "## note 2": ["b"],
            "## note 3": ["c"],
        })
        df, _ = _read(tmp_path, frame)
        assert _model_columns(df) == ["country", "capacity"]

    def test_it_is_silent(self, tmp_path):
        # A declared intention is not a problem to report.
        frame = pd.DataFrame({"country": ["FI"], "capacity": [1000.0], "## x": ["s"]})
        _, logger = _read(tmp_path, frame)
        logger.assert_clean()

    def test_leading_whitespace_is_forgiven(self, tmp_path):
        # The commonest spreadsheet accident, and the row rule strips too.
        frame = pd.DataFrame({"country": ["FI"], "capacity": [1000.0], "  ## x": ["s"]})
        df, _ = _read(tmp_path, frame)
        assert _model_columns(df) == ["country", "capacity"]


class TestItProtectsScratchWorkFromTheNumericGate:
    """The reason the marker had to land with the gate rather than after it."""

    def test_a_malformed_number_in_a_marked_column_is_not_reported(self, tmp_path):
        frame = pd.DataFrame({
            "country": ["FI", "SE"],
            "capacity": [1000.0, 500.0],
            "## working": ["1,000.0", 2000.0],
        })
        _, logger = _read(tmp_path, frame)
        logger.assert_no_errors()

    def test_a_broken_formula_in_a_marked_column_is_not_reported(self, tmp_path):
        frame = pd.DataFrame({
            "country": ["FI", "SE"],
            "capacity": [1000.0, 500.0],
            "## working": ["#DIV/0!", 2000.0],
        })
        _, logger = _read(tmp_path, frame)
        logger.assert_no_errors()

    def test_but_the_real_columns_are_still_checked(self, tmp_path):
        # Marking scratch work must not turn the gate off for the sheet.
        frame = pd.DataFrame({
            "country": ["FI", "SE"],
            "capacity": ["1,000.0", 500.0],
            "## working": ["whatever", 1.0],
        })
        _, logger = _read(tmp_path, frame)
        logger.assert_logged("capacity", level="error")


class TestTheHelperTableLayoutThisWasBuiltFor:
    """A real sheet: main table, a blank spacer column, then headerless scratch.

    Authors leave the helper columns' headers empty, so pandas names them
    ``Unnamed: 5``, ``Unnamed: 6`` and so on. Only one cell needs the marker --
    the first helper column that *has* a header. Everything headerless to its
    right is already dropped as an empty title, and the spacer between the two
    tables goes the same way.

    That composition is why the marker is applied before the blank-header
    handling rather than after: once the marked column is gone, nothing named
    survives to the spacer's right, so the spacer reads as the end of the sheet
    instead of a hole in the middle of one.
    """

    def _book(self, tmp_path):
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "unitdata"
        ws.append(["country", "capacity", "method", None, "## helper", None, None])
        ws.append(["FI", 1000, "replace", None, "=A1*2", "1,000.0", "#DIV/0!"])
        ws.append(["SE", 500, "replace", None, "scratch", "2 000", "#REF!"])
        path = tmp_path / "book.xlsx"
        wb.save(path)
        return path

    def _read(self, tmp_path):
        self._book(tmp_path)
        logger = FakeLogger()
        frames = loader.read_input_excels(tmp_path, ["book.xlsx"], "unitdata", logger)
        return frames[0], logger

    def test_only_the_real_columns_survive(self, tmp_path):
        df, _ = self._read(tmp_path)
        assert _model_columns(df) == ["country", "capacity", "method"]

    def test_both_data_rows_survive(self, tmp_path):
        # The '#DIV/0!' and '#REF!' in the scratch area must not delete rows.
        df, _ = self._read(tmp_path)
        assert len(df) == 2

    def test_the_scratch_area_produces_no_complaints(self, tmp_path):
        # It holds a thousands separator and two broken formulas -- every single
        # thing the gate reports, all of it deliberate working material.
        _, logger = self._read(tmp_path)
        logger.assert_clean()


class TestAMarkedRowIsIgnored:
    """The other half of the same marker."""

    def test_the_row_does_not_reach_the_frame(self, tmp_path):
        frame = pd.DataFrame({
            "country": ["## section heading", "FI", "SE"],
            "capacity": [None, 1000.0, 500.0],
        })
        df, _ = _read(tmp_path, frame)
        assert list(df["country"]) == ["FI", "SE"]

    def test_the_marker_may_sit_in_any_column(self, tmp_path):
        frame = pd.DataFrame({
            "country": ["FI", "SE"],
            "capacity": [1000.0, 500.0],
            "method": ["## not this one", "replace"],
        })
        df, _ = _read(tmp_path, frame)
        assert list(df["country"]) == ["SE"]

    def test_it_is_silent(self, tmp_path):
        frame = pd.DataFrame({"country": ["## x", "FI"], "capacity": [None, 1.0]})
        _, logger = _read(tmp_path, frame)
        logger.assert_clean()

    def test_scratch_text_in_a_helper_column_cannot_delete_a_real_row(self, tmp_path):
        # The ordering guarantee: ignored *columns* go first, so '##' written as
        # free text out in the working area is gone before rows are judged.
        # Judging rows first would let a note beside the table delete the row
        # next to it -- silently, and only for whichever rows the note lined up
        # with.
        frame = pd.DataFrame({
            "country": ["FI", "SE"],
            "capacity": [1000.0, 500.0],
            "## working": ["## remember to check this", ""],
        })
        df, _ = _read(tmp_path, frame)
        assert list(df["country"]) == ["FI", "SE"]


class TestASingleHashIsNotAMarker:
    """``#`` alone is how every Excel error value starts, so it cannot mean this.

    ``#REF!``, ``#N/A`` and ``#DIV/0!`` all begin with one hash, so under the old
    single-hash rule a broken formula read as a comment and deleted the row it
    sat in, with nothing logged. ``# of units`` and ``#1`` are ordinary things to
    write in a cell, too.
    """

    def test_a_single_hash_column_is_kept(self, tmp_path):
        frame = pd.DataFrame({"country": ["FI"], "# of units": [3.0]})
        df, _ = _read(tmp_path, frame)
        assert "# of units" in _model_columns(df)

    def test_a_single_hash_cell_does_not_drop_its_row(self, tmp_path):
        frame = pd.DataFrame({
            "country": ["FI", "SE"],
            "capacity": [1000.0, 500.0],
            "comment": ["#1 in the list", "second"],
        })
        df, _ = _read(tmp_path, frame)
        assert list(df["country"]) == ["FI", "SE"]

    @pytest.mark.parametrize("marker", ["##", "###", "## "])
    def test_two_or_more_hashes_do_mark(self, marker, tmp_path):
        frame = pd.DataFrame({"country": ["FI"], "capacity": [1.0], f"{marker}x": ["s"]})
        df, _ = _read(tmp_path, frame)
        assert _model_columns(df) == ["country", "capacity"]
