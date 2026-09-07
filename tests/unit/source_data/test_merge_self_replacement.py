"""A sheet that overwrites its own earlier row.

Overriding a value is what the file order is for: a later workbook states a
different number and the ``method`` column says what to do with it. Two rows for
one key *in one sheet* is a different thing. The earlier row is read, applied,
and then overwritten, so it is dead text -- and nothing said so.

Why this is scoped to the replacing methods
-------------------------------------------
The obvious version of this check -- warn on any key seen twice in a sheet --
fires on correct data. ``dheat_unitdata_PL00_DE00_AT00.xlsx`` writes
``DE00 / gas ccgt present 2`` twice, both rows ``add-non-negative``, two
deliberate reductions that stack. A warning there would be noise, and a check
that fires every run on something right is not strict, it is broken.

So the question is not "was this key seen twice" but "did a row *replace* what
its own sheet had already established". ``add`` and ``multiply`` accumulate by
definition; ``replace`` and ``replace-partial`` discard.
"""

import pandas as pd

from src.source_data.source_data_loader import merge_row_by_row
from tests._common.fixtures import FakeLogger

MESSAGE = "written twice in the same sheet"


def _rows(*, methods, sheet="unitdata", file="book.xlsx"):
    """One frame holding several rows for the same key, as one sheet does."""
    count = len(methods)
    return pd.DataFrame({
        "country": pd.Series(["FI00"] * count, dtype="object"),
        "generator_id": pd.Series(["chp"] * count, dtype="object"),
        "capacity": pd.Series([100.0] * count, dtype="Float64"),
        "method": pd.Series(list(methods), dtype="object"),
        "_source_file": pd.Series([file] * count, dtype="object"),
        "_source_sheet": pd.Series([sheet] * count, dtype="object"),
    })


KEYS = ["country", "generator_id"]


class TestASheetReplacingItsOwnRow:
    def test_it_warns(self):
        logger = FakeLogger()
        merge_row_by_row([_rows(methods=["replace", "replace"])], logger,
                         key_columns=KEYS)

        logger.assert_logged(MESSAGE, level="warn")

    def test_replace_partial_counts_too(self):
        """It discards whatever it covers, which is the same failure."""
        logger = FakeLogger()
        merge_row_by_row([_rows(methods=["replace", "replace-partial"])], logger,
                         key_columns=KEYS)

        logger.assert_logged(MESSAGE, level="warn")

    def test_the_message_names_the_sheet(self):
        """A key alone is not findable; the sheet is what someone opens."""
        logger = FakeLogger()
        merge_row_by_row([_rows(methods=["replace", "replace"], sheet="unitdata_VRE")],
                         logger, key_columns=KEYS)

        assert any("unitdata_VRE" in m for m in logger.matching(MESSAGE))

    def test_the_later_row_still_wins(self):
        """The warning describes existing behaviour; it does not change it."""
        logger = FakeLogger()
        frame = _rows(methods=["replace", "replace"])
        frame.loc[1, "capacity"] = 250.0

        merged = merge_row_by_row([frame], logger, key_columns=KEYS)

        assert merged["capacity"].tolist() == [250.0]


class TestWhatMustStayQuiet:
    def test_rows_that_stack_are_deliberate(self):
        """The shipped case: two add-non-negative reductions on one key."""
        logger = FakeLogger()
        merge_row_by_row([_rows(methods=["add-non-negative", "add-non-negative"])],
                         logger, key_columns=KEYS)

        logger.assert_not_logged(MESSAGE)

    def test_multiply_accumulates_by_definition(self):
        logger = FakeLogger()
        merge_row_by_row([_rows(methods=["replace", "multiply"])], logger,
                         key_columns=KEYS)

        logger.assert_not_logged(MESSAGE)

    def test_a_later_sheet_overriding_an_earlier_one_is_the_point_of_the_merge(self):
        logger = FakeLogger()
        merge_row_by_row(
            [_rows(methods=["replace"], sheet="unitdata"),
             _rows(methods=["replace"], sheet="unitdata_VRE")],
            logger, key_columns=KEYS,
        )

        logger.assert_not_logged(MESSAGE)

    def test_a_later_file_overriding_an_earlier_one_is_too(self):
        logger = FakeLogger()
        merge_row_by_row(
            [_rows(methods=["replace"], file="base.xlsx"),
             _rows(methods=["replace"], file="overlay.xlsx")],
            logger, key_columns=KEYS,
        )

        logger.assert_not_logged(MESSAGE)

    def test_a_removed_key_can_be_written_again(self):
        """'remove' then 'replace' re-establishes the record; nothing was lost."""
        logger = FakeLogger()
        merge_row_by_row([_rows(methods=["replace", "remove", "replace"])], logger,
                         key_columns=KEYS)

        logger.assert_not_logged(MESSAGE)

    def test_frames_without_provenance_are_not_guessed_about(self):
        """A caller building frames by hand has no sheets to name."""
        logger = FakeLogger()
        frame = _rows(methods=["replace", "replace"]).drop(
            columns=["_source_file", "_source_sheet"]
        )

        merge_row_by_row([frame], logger, key_columns=KEYS)

        logger.assert_not_logged(MESSAGE)
