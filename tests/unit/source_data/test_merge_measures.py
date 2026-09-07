"""What a row's arithmetic is allowed to reach.

``merge_row_by_row`` infers its measure columns: anything numeric in any frame
that is not in ``not_measure_cols``. Three handlers then act on that set, and
they did not agree about it. ``_handle_replace`` in partial mode skips key and
meta columns explicitly; ``_handle_add`` and ``_handle_multiply`` iterated every
inferred measure.

So a key column that happens to hold numbers was a measure. An ``add`` row would
sum it while the record kept the key it was found by -- the key that identifies
a record silently becoming the sum of every row that touched it. ``method`` is
in the same position: it is an instruction, not data.

Not live on the shipped workbooks, where every key column holds text. It is a
property of the merge rather than of today's data, which is why it is asserted
as one.
"""

import pandas as pd

from src.source_data.source_data_loader import MERGE_META_COLUMNS, merge_row_by_row
from tests._common.fixtures import FakeLogger


def _frame(prefix, capacity, method):
    """A unitdata row whose ``unit_name_prefix`` is numeric.

    Numeric key columns are legal: nothing says a suffix has to be a word, and
    normalize_dataframe types a column of digits as Float64 like any other.
    """
    return pd.DataFrame({
        "country": pd.Series(["FI00"], dtype="object"),
        "generator_id": pd.Series(["chp"], dtype="object"),
        "unit_name_prefix": pd.Series([prefix], dtype="Float64"),
        "capacity": pd.Series([capacity], dtype="Float64"),
        "method": pd.Series([method], dtype="object"),
    })


KEYS = ["country", "generator_id", "unit_name_prefix"]


class TestAKeyIsNotAMeasure:
    def test_add_does_not_sum_the_key_it_matched_on(self):
        merged = merge_row_by_row(
            [_frame(2, 100.0, "replace"), _frame(2, 50.0, "add")],
            FakeLogger(),
            key_columns=KEYS,
        )

        assert merged["unit_name_prefix"].tolist() == [2.0]

    def test_multiply_does_not_scale_the_key_either(self):
        merged = merge_row_by_row(
            [_frame(2, 100.0, "replace"), _frame(2, 0.5, "multiply")],
            FakeLogger(),
            key_columns=KEYS,
        )

        assert merged["unit_name_prefix"].tolist() == [2.0]

    def test_the_measure_still_moves(self):
        """The negative control: excluding the key must not disarm the handler."""
        merged = merge_row_by_row(
            [_frame(2, 100.0, "replace"), _frame(2, 50.0, "add")],
            FakeLogger(),
            key_columns=KEYS,
        )

        assert merged["capacity"].tolist() == [150.0]

    def test_one_record_not_two(self):
        """Both rows carry the same key, so they describe the same unit."""
        merged = merge_row_by_row(
            [_frame(2, 100.0, "replace"), _frame(2, 50.0, "add")],
            FakeLogger(),
            key_columns=KEYS,
        )

        assert len(merged) == 1


class TestMetaColumnsAreShared:
    def test_the_default_cannot_be_mutated_by_a_caller(self):
        """It was a set literal in the signature: one object, every call."""
        assert isinstance(MERGE_META_COLUMNS, frozenset)

    def test_they_do_not_reach_the_result(self):
        merged = merge_row_by_row(
            [_frame(2, 100.0, "replace")], FakeLogger(), key_columns=KEYS
        )

        assert not set(merged.columns) & MERGE_META_COLUMNS
