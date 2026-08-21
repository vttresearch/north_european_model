"""A header used twice: reported, never merged.

Excel lets two columns carry the same header. pandas does not, so it renames the
second ``capacity`` to ``capacity.1`` -- a name nothing in the pipeline reads.
The column is therefore never used, and until now nothing said so.

Why report rather than join
--------------------------
Joining ``capacity.1`` back into ``capacity`` would have to decide which value
wins, or how to combine them, and the sheet offers no basis for either. That is
a new seam of bugs in exchange for guessing. The file and sheet are still known
at this point, which is what makes the question answerable by the one person who
can answer it.

The two false positives this must not have
------------------------------------------
``##``, ``##.1``, ``##.2`` -- every helper block looks exactly like a repeated
header, because every one of its columns is headed ``##``. The check runs after
those columns are dropped.

``vomCosts`` next to ``vomCosts_output1`` -- a bare parameter name means
``_output1``, so the two collapse to one name when the suffix is stripped. That
pair is real and is already reported by ``normalize_dataframe``'s rename
collision; it never becomes a ``.N`` name and must not be reported twice.
"""

import pandas as pd
import pytest
from openpyxl import Workbook

import src.source_data.source_data_loader as loader
from tests._common.fixtures import FakeLogger


def _read(tmp_path, rows, *, sheet="unitdata"):
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    for row in rows:
        ws.append(row)
    wb.save(tmp_path / "book.xlsx")

    logger = FakeLogger()
    frames = loader.read_input_excels(tmp_path, ["book.xlsx"], sheet, logger)
    return (frames[0] if frames else pd.DataFrame()), logger


class TestARepeatedHeaderIsReported:
    def test_it_warns(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", "capacity", "capacity"],
            ["FI", 1000, 999],
        ])
        logger.assert_logged("Duplicate column header", level="warn")

    def test_the_message_names_the_header(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", "capacity", "capacity"],
            ["FI", 1000, 999],
        ])
        logger.assert_logged("'capacity'", level="warn")

    def test_the_message_names_the_file_and_sheet(self, tmp_path):
        # The point of reporting instead of merging is that someone has to go and
        # look; without the location they cannot.
        _, logger = _read(tmp_path, [
            ["country", "capacity", "capacity"],
            ["FI", 1000, 999],
        ])
        logger.assert_logged("book.xlsx:unitdata", level="warn")

    def test_the_message_says_how_much_is_being_ignored(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", "capacity", "capacity"],
            ["FI", 1000, 999],
            ["SE", 500, 888],
        ])
        logger.assert_logged("2 values", level="warn")

    def test_three_of_the_same_header_are_all_reported(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", "capacity", "capacity", "capacity"],
            ["FI", 1000, 999, 888],
        ])
        logger.assert_logged("repeated 2 time(s)", level="warn")

    def test_the_first_column_is_the_one_that_is_used(self, tmp_path):
        # The warning describes existing behaviour rather than changing it.
        df, _ = _read(tmp_path, [
            ["country", "capacity", "capacity"],
            ["FI", 1000, 999],
        ])
        assert list(df["capacity"]) == [1000]


class TestWhatMustNotBeReported:
    def test_a_helper_block_headed_with_the_marker(self, tmp_path):
        # '##', '##.1', '##.2' -- pandas' renaming makes every helper block look
        # like a repeated header. This is the common case in real workbooks.
        _, logger = _read(tmp_path, [
            ["country", "capacity", "##", "##", "##"],
            ["FI", 1000, "wip", 1, 2],
        ])
        logger.assert_clean()

    def test_a_bare_parameter_beside_its_output1_spelling(self, tmp_path):
        # A real pair that collapses when '_output1' is stripped. It is reported
        # by the rename-collision guard, not as a duplicate header, and the two
        # reports must not both fire.
        _, logger = _read(tmp_path, [
            ["country", "generator_id", "method", "vomCosts", "vomCosts_output1"],
            ["FI", "chp", "replace", 1.5, 2.5],
        ])
        logger.assert_not_logged("Duplicate column header")

    def test_distinct_headers(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", "capacity", "vomCosts"],
            ["FI", 1000, 1.5],
        ])
        logger.assert_clean()

    def test_a_header_that_genuinely_ends_in_a_number(self, tmp_path):
        # 'grid_input1' and friends are ordinary names; only a '.N' suffix whose
        # base is also present in the same sheet means pandas renamed something.
        _, logger = _read(tmp_path, [
            ["country", "grid_input1", "grid_input2"],
            ["FI", "elec", "heat"],
        ])
        logger.assert_clean()

    def test_a_dotted_header_whose_base_is_absent(self, tmp_path):
        # Someone may legitimately name a column 'eff.1'. With no 'eff' beside
        # it, nothing was renamed and there is nothing to report.
        _, logger = _read(tmp_path, [
            ["country", "eff.1"],
            ["FI", 0.4],
        ])
        logger.assert_clean()


class TestTheRealWorkbooksStaySilent:
    def test_no_duplicate_reports_across_every_shipped_sheet(self, src_files_dir):
        import glob
        import os

        files = [
            os.path.basename(f)
            for f in sorted(glob.glob(str(src_files_dir / "data_files" / "*.xlsx")))
        ]
        logger = FakeLogger()
        for prefix in ("unitdata", "unittypedata", "nodedata", "transferdata",
                       "demanddata", "emissiondata", "userconstraintdata"):
            loader.read_input_excels(src_files_dir / "data_files", files, prefix, logger)

        logger.assert_not_logged("Duplicate column header")
