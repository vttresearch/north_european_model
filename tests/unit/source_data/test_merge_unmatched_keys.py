"""An arithmetic row whose key matches nothing.

``add`` and ``multiply`` are instructions to change a value that already exists.
A row whose key matches no earlier row is therefore almost always a misspelled
key -- and nothing could detect it, because the merge simply treated the row as
the first statement about a new record.

The two methods then diverge, which is why both are reported rather than only
the one that does nothing:

``add`` establishes the record. That is right and stays: a missing value counts
as ``0.0`` for addition, so ``0 + cur`` is ``cur``. The damage is that a
misspelled key quietly creates a second unit carrying only the increment.

``multiply`` establishes nothing. There is no value to scale, and the
multiplicative equivalent of "the missing operand contributes nothing" is
leaving the value unset -- not writing the multiplier as though it were a
quantity. A ``0.5`` meant to halve a capacity used to become a capacity of
``0.5``.

Measured on the shipped configs: no row reaches either case.
"""

import pandas as pd

from src.source_data.source_data_loader import merge_row_by_row, normalize_dataframe
from tests._common.fixtures import FakeLogger

MESSAGE = "add to or multiply a key that no earlier row established"
KEY = ["country", "grid"]


def _frame(*rows):
    return normalize_dataframe(pd.DataFrame(list(rows)), "test", FakeLogger())


def _row(method="replace", country="FI", grid="elec", **values):
    return {"country": country, "grid": grid, "method": method, **values}


def _merge(*rows):
    logger = FakeLogger()
    merged = merge_row_by_row([_frame(*rows)], logger, key_columns=KEY)
    return merged, logger


class TestAMultiplyRowWithNothingToScale:
    def test_it_establishes_no_record(self):
        merged, _ = _merge(_row("multiply", capacity=0.5))
        assert merged.empty

    def test_it_is_reported(self):
        _, logger = _merge(_row("multiply", capacity=0.5))
        logger.assert_logged(MESSAGE, level="warn")

    def test_a_misspelled_key_leaves_the_real_row_alone(self):
        """The failure this prevents, as the shape it takes in a workbook.

        A row meant to halve FI's capacity, with the grid misspelled. Before,
        the model gained a second node carrying a capacity of 0.5.
        """
        merged, logger = _merge(
            _row("replace", capacity=100.0),
            _row("multiply", grid="elecc", capacity=0.5),
        )

        assert list(merged["grid"]) == ["elec"]
        assert merged.iloc[0]["capacity"] == 100.0
        logger.assert_logged("elecc", level="warn")

    def test_a_matching_multiply_still_scales(self):
        """Negative control: refusing the unmatched case must not disarm it."""
        merged, logger = _merge(
            _row("replace", capacity=100.0),
            _row("multiply", capacity=0.5),
        )

        assert merged.iloc[0]["capacity"] == 50.0
        logger.assert_not_logged(MESSAGE)


class TestAnAddRowWithNothingToAddTo:
    def test_it_still_establishes_the_record(self):
        """Unchanged, and correct: a missing value is 0.0 for addition."""
        merged, _ = _merge(_row("add", capacity=25.0))
        assert merged.iloc[0]["capacity"] == 25.0

    def test_it_is_reported_anyway(self):
        """Creating a unit is not what an 'add' row is usually written to do."""
        _, logger = _merge(_row("add", capacity=25.0))
        logger.assert_logged(MESSAGE, level="warn")

    def test_add_non_negative_counts_too(self):
        _, logger = _merge(_row("add-non-negative", capacity=25.0))
        logger.assert_logged(MESSAGE, level="warn")

    def test_a_matching_add_is_quiet(self):
        merged, logger = _merge(
            _row("replace", capacity=100.0),
            _row("add", capacity=25.0),
        )

        assert merged.iloc[0]["capacity"] == 125.0
        logger.assert_not_logged(MESSAGE)


class TestWhatIsNotArithmetic:
    def test_replace_establishing_a_record_is_the_normal_case(self):
        _, logger = _merge(_row("replace", capacity=100.0))
        logger.assert_not_logged(MESSAGE)

    def test_replace_partial_establishing_a_record_is_too(self):
        """It is a replace: it states values rather than changing them."""
        _, logger = _merge(_row("replace-partial", capacity=100.0))
        logger.assert_not_logged(MESSAGE)

    def test_a_removed_key_is_gone_for_arithmetic_too(self):
        """'remove' deletes the record, so a later 'add' has nothing to add to."""
        _, logger = _merge(
            _row("replace", capacity=100.0),
            _row("remove"),
            _row("add", capacity=25.0),
        )

        logger.assert_logged(MESSAGE, level="warn")
