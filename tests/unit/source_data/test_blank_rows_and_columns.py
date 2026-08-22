"""Blank rows and blank columns: one rule, both axes.

    A blank row or blank-header column with any content after it on that axis is
    a warning. With nothing after it, it is silent.

That collapses what looked like four cases -- blank before the table, blank in
the middle, blank after it with more beyond, blank at the end -- into one
question: *is anything still to come?* If yes, something is being discarded and
nobody was told. If no, this is simply where the sheet stops, which is true of
every sheet ever written and must not produce a message.

The rule reads both axes the same way because the mistakes are the same. A
spacer row in the middle of a table truncates everything below it. A column
whose header someone deleted becomes ``Unnamed: 7`` and its values vanish. Both
were silent.

Order matters as much as the rule
---------------------------------
Columns marked ``##`` are dropped *before* this runs, so a spacer column between
the real table and a declared helper block has nothing named to its right by the
time it is judged, and stays silent. Measured against the shipped workbooks: 21
warnings if the rule ran on the raw sheets, none once ``##`` is honoured first.

The one case the rule cannot reach
----------------------------------
A blank row *above* the header cannot be classified, because the reader takes
row 1 as the header -- so a leading blank row makes every column unnamed rather
than leaving a blank row to find. All the columns are then dropped, and a frame
with no columns has every row read as empty, which truncates the sheet to
nothing. That needs its own guard, and it is an error rather than a warning:
the alternative is a sheet that silently contributes nothing while looking
exactly like one that was legitimately empty.
"""

import pandas as pd
import pytest
from openpyxl import Workbook

import src.source_data.source_data_loader as loader
from tests._common.fixtures import FakeLogger


def _read(tmp_path, rows, *, sheet="unitdata"):
    """Write rows straight to a sheet, so blank headers stay blank."""
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    for row in rows:
        ws.append(row)
    wb.save(tmp_path / "book.xlsx")

    logger = FakeLogger()
    frames = loader.read_input_excels(tmp_path, ["book.xlsx"], sheet, logger)
    return (frames[0] if frames else pd.DataFrame()), logger


class TestBlankColumns:
    def test_one_past_the_end_is_silent(self, tmp_path):
        # The scratch area. Every second sheet has one.
        _, logger = _read(tmp_path, [
            ["country", "capacity", None, None],
            ["FI", 1000, "scratch", 7],
            ["SE", 500, None, None],
        ])
        logger.assert_clean()

    def test_one_inside_the_table_is_reported(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", None, "capacity"],
            ["FI", "orphaned", 1000],
            ["SE", "values", 500],
        ])
        logger.assert_logged("without a header", level="warn")

    def test_the_message_says_how_much_is_being_discarded(self, tmp_path):
        # A count is what tells the reader whether this matters.
        _, logger = _read(tmp_path, [
            ["country", None, "capacity"],
            ["FI", "orphaned", 1000],
            ["SE", "values", 500],
        ])
        logger.assert_logged("2 values", level="warn")

    def test_it_is_dropped_either_way(self, tmp_path):
        # The header cannot be recovered, so there is nothing else to do with it.
        df, _ = _read(tmp_path, [
            ["country", None, "capacity"],
            ["FI", "orphaned", 1000],
        ])
        assert [c for c in df.columns if not str(c).startswith("_")] == ["country", "capacity"]

    def test_a_marked_helper_block_keeps_its_spacer_silent(self, tmp_path):
        # The ordering guarantee, as the layout that motivated it: real table,
        # blank spacer, then a declared helper block. Judged before '##' was
        # honoured, the spacer would have a named column to its right and warn.
        _, logger = _read(tmp_path, [
            ["country", "capacity", None, "##", None],
            ["FI", 1000, None, "=A1*2", 3],
            ["SE", 500, None, "scratch", 4],
        ])
        logger.assert_clean()

    def test_a_sheet_with_no_headers_at_all_is_an_error(self, tmp_path):
        # A blank row above the header: every column comes back unnamed, and the
        # sheet used to disappear without a word.
        df, logger = _read(tmp_path, [
            [None, None],
            ["country", "capacity"],
            ["FI", 1000],
        ])
        logger.assert_logged("No column has a header", level="error")
        assert df.empty


class TestBlankRows:
    def test_trailing_blank_rows_are_silent(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", "capacity"],
            ["FI", 1000],
            ["SE", 500],
            [None, None],
        ])
        logger.assert_clean()

    def test_a_spacer_row_mid_table_is_reported(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", "capacity"],
            ["FI", 1000],
            [None, None],
            ["SE", 500],
            ["NO", 250],
        ])
        logger.assert_logged("are not read", level="warn")

    def test_the_message_counts_the_rows_being_lost(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", "capacity"],
            ["FI", 1000],
            [None, None],
            ["SE", 500],
            ["NO", 250],
        ])
        logger.assert_logged("2 row(s)", level="warn")

    def test_the_rows_really_are_dropped(self, tmp_path):
        # The warning describes existing behaviour; it does not change it.
        df, _ = _read(tmp_path, [
            ["country", "capacity"],
            ["FI", 1000],
            [None, None],
            ["SE", 500],
        ])
        assert list(df["country"]) == ["FI"]

    def test_a_whitespace_only_row_counts_as_blank(self, tmp_path):
        # It looks empty in Excel, so it has to behave as empty here.
        _, logger = _read(tmp_path, [
            ["country", "capacity"],
            ["FI", 1000],
            ["   ", "  "],
            ["SE", 500],
        ])
        logger.assert_logged("are not read", level="warn")


class TestTheRealWorkbooksStaySilent:
    """The rule must fire on nothing that ships today, or it is noise."""

    def test_no_warnings_across_every_shipped_sheet(self, src_files_dir):
        import glob
        import os

        # '~$name.xlsx' is the lock file Excel writes while a workbook is open.
        # It matches the glob, is not readable, and is not a shipped workbook --
        # without this the suite fails for anyone who left one open.
        files = [
            os.path.basename(f)
            for f in sorted(glob.glob(str(src_files_dir / "data_files" / "*.xlsx")))
            if not os.path.basename(f).startswith("~$")
        ]
        prefixes = [
            "unitdata", "unittypedata", "nodedata", "transferdata",
            "demanddata", "emissiondata", "userconstraintdata",
        ]

        logger = FakeLogger()
        for prefix in prefixes:
            loader.read_input_excels(src_files_dir / "data_files", files, prefix, logger)

        logger.assert_clean()
