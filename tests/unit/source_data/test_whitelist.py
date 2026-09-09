"""A row that never says which run it belongs to.

``apply_whitelist`` keeps rows whose ``scenario`` matches this run or the
catch-all ``all``, and whose ``year`` matches this run or the catch-all ``1``. A
blank in either cell matches neither: ``astype(str)`` renders ``pd.NA`` as the
literal ``'<NA>'``, which is in no allowed set, so the row is dropped.

Dropping it is right. Reading a blank as "every run" would silently promote a
half-finished row into every scenario the workbook holds, and the catch-alls
exist precisely so that an author can say "every run" on purpose.

Saying nothing about it was the problem. An empty cell produced exactly the same
silence as a row for a scenario this build does not cover, so a workbook could
lose rows to an unfinished cell with nothing anywhere to suggest it.
"""

import pandas as pd

from src.source_data.source_data_loader import apply_whitelist
from tests._common.fixtures import FakeLogger

MESSAGE = "so those row(s) are not read"


def _frame(scenarios, years, *, source=True):
    frame = pd.DataFrame({
        "scenario": pd.Series(scenarios, dtype="object"),
        "year": pd.Series(years, dtype="Float64"),
        "capacity": pd.Series([100.0] * len(scenarios), dtype="Float64"),
    })
    if source:
        frame["_source_file"] = "book.xlsx"
        frame["_source_sheet"] = "unitdata"
    return frame


FILTERS = {"scenario": ["observed trends"], "year": [2030]}


def _filter(frame):
    logger = FakeLogger()
    return apply_whitelist(frame, FILTERS, logger, "unitdata"), logger


class TestABlankScenarioOrYear:
    def test_a_blank_scenario_is_reported(self):
        _, logger = _filter(_frame(["observed trends", None], [2030, 2030]))
        logger.assert_logged(MESSAGE, level="warn")

    def test_a_blank_year_is_reported(self):
        _, logger = _filter(_frame(["observed trends"] * 2, [2030, None]))
        logger.assert_logged(MESSAGE, level="warn")

    def test_the_row_is_still_dropped(self):
        """The warning describes the behaviour; it does not change it.

        Reading a blank as 'all' would put an unfinished row into every run.
        """
        kept, _ = _filter(_frame(["observed trends", None], [2030, 2030]))
        assert len(kept) == 1

    def test_the_message_counts_them(self):
        _, logger = _filter(
            _frame(["observed trends", None, None], [2030, 2030, 2030])
        )
        assert any("2 with no scenario" in m for m in logger.matching(MESSAGE))

    def test_the_message_names_the_sheet(self):
        """A count with no sheet is not something anyone can act on."""
        _, logger = _filter(_frame(["observed trends", None], [2030, 2030]))
        assert any("book.xlsx" in m for m in logger.matching(MESSAGE))

    def test_it_survives_a_frame_with_no_provenance(self):
        """A caller building frames by hand still gets the count."""
        _, logger = _filter(
            _frame(["observed trends", None], [2030, 2030], source=False)
        )
        logger.assert_logged(MESSAGE, level="warn")


class TestWhatIsNotABlank:
    def test_the_catch_alls_are_quiet(self):
        _, logger = _filter(_frame(["all", "all"], [1, 1]))
        logger.assert_not_logged(MESSAGE)

    def test_a_row_for_another_scenario_is_quiet(self):
        """Not this run's row, and nothing is wrong with it."""
        _, logger = _filter(
            _frame(["observed trends", "national trends"], [2030, 2030])
        )
        logger.assert_not_logged(MESSAGE)

    def test_a_row_for_another_year_is_quiet(self):
        _, logger = _filter(_frame(["observed trends"] * 2, [2030, 2040]))
        logger.assert_not_logged(MESSAGE)

    def test_a_frame_with_no_scenario_column_is_the_other_warning(self):
        """A missing column is a different statement from a blank cell."""
        frame = _frame(["observed trends"], [2030]).drop(columns=["scenario"])
        _, logger = _filter(frame)

        logger.assert_logged("missing column", level="warn")
        logger.assert_not_logged(MESSAGE)


class TestTheCountIsTakenBeforeFiltering:
    """Counted against the incoming frame, not against what is left of it.

    The counts used to be taken inside the filter loop, so ``year`` was counted
    against a frame the ``scenario`` filter had already narrowed -- and not at
    all once that filter emptied the frame and the loop short-circuited. A blank
    could then be hidden by an unrelated cell in the row beside it, which is the
    one case where silence is least affordable.
    """

    def test_a_blank_year_survives_the_scenario_filter(self):
        """The scenario filter removes this row before 'year' is ever reached."""
        _, logger = _filter(_frame(["national trends"], [None]))
        assert any("1 with no year" in m for m in logger.matching(MESSAGE))

    def test_both_are_named_when_the_frame_empties(self):
        _, logger = _filter(_frame([None], [None]))
        said = logger.matching(MESSAGE)
        assert any("1 with no scenario" in m and "1 with no year" in m for m in said), said
