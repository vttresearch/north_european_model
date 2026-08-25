"""Tests for ``tests/_common/contracts.py``.

Heavy on negative controls.  A contract assertion that accepts anything would
make every sweep in the suite green and meaningless, so each rule gets a case
that must FAIL as well as one that must pass.
"""

import math
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from tests._common.contracts import (
    NASTY_CELLS,
    assert_no_na_became_zero,
    assert_normalized,
    frame_with_blank_column,
    frame_with_cell,
    nasty_id,
)


class TestAssertNormalizedAccepts:
    def test_a_conventionally_typed_frame(self):
        df = pd.DataFrame(
            {
                "country": pd.Series(["FI", "SE"], dtype="object"),
                "capacity": pd.Series([1.0, 2.0], dtype="Float64"),
            }
        )
        assert_normalized(df)

    def test_pd_na_inside_an_object_column(self):
        df = pd.DataFrame({"country": pd.Series(["FI", pd.NA], dtype="object")})
        assert_normalized(df)

    def test_pd_na_inside_a_float64_column(self):
        # Float64 is a nullable dtype -- NA there is the intended representation.
        df = pd.DataFrame({"capacity": pd.Series([1.0, pd.NA], dtype="Float64")})
        assert_normalized(df)

    def test_an_all_na_object_column(self):
        # The "assume nothing" state, and therefore explicitly legal.
        df = pd.DataFrame({"unused": pd.Series([pd.NA, pd.NA], dtype="object")})
        assert_normalized(df)

    def test_strings_that_the_pipeline_never_promised_to_convert(self):
        # Only 'nan' is converted (utils.py:75). Asserting 'NA'/'None'/'null'
        # away would fail on behaviour that was never promised -- and 'NA' is a
        # plausible real value.
        df = pd.DataFrame({"code": pd.Series(["NA", "None", "null"], dtype="object")})
        assert_normalized(df)

    def test_an_empty_frame(self):
        assert_normalized(pd.DataFrame())


class TestAssertNormalizedRejects:
    def test_a_disallowed_dtype(self):
        df = pd.DataFrame({"capacity": pd.Series([1.0, 2.0], dtype="float64")})
        with pytest.raises(AssertionError, match="dtype float64"):
            assert_normalized(df)

    def test_an_all_na_column_typed_float64(self):
        # THE cascade bug, as a negative control. If this assertion ever stops
        # firing, the sweep that protects against the bug is dead.
        df = pd.DataFrame({"capacity": pd.Series([pd.NA, pd.NA], dtype="Float64")})
        with pytest.raises(AssertionError, match="entirely NA"):
            assert_normalized(df)

    def test_none_inside_an_object_column(self):
        df = pd.DataFrame({"country": pd.Series(["FI", None], dtype="object")})
        with pytest.raises(AssertionError, match="None"):
            assert_normalized(df)

    def test_a_float_nan_inside_an_object_column(self):
        df = pd.DataFrame({"country": pd.Series(["FI", np.nan], dtype="object")})
        with pytest.raises(AssertionError, match="float NaN"):
            assert_normalized(df)

    def test_a_nat_inside_an_object_column(self):
        df = pd.DataFrame({"when": pd.Series(["x", pd.NaT], dtype="object")})
        with pytest.raises(AssertionError, match="NaT"):
            assert_normalized(df)

    @pytest.mark.parametrize("text", ["nan", "NaN", " NAN "])
    def test_a_nan_string_that_survived(self, text):
        df = pd.DataFrame({"country": pd.Series(["FI", text], dtype="object")})
        with pytest.raises(AssertionError, match="should have become pd.NA"):
            assert_normalized(df)

    def test_duplicate_column_names(self):
        df = pd.DataFrame([[1, 2]], columns=["grid", "grid"])
        with pytest.raises(AssertionError, match="duplicate column"):
            assert_normalized(df)

    def test_something_that_is_not_a_dataframe(self):
        with pytest.raises(AssertionError, match="expected a DataFrame"):
            assert_normalized("not a frame")

    def test_a_dirty_index_only_when_asked(self):
        # Build first, then relabel: passing index= to the constructor would
        # *reindex* the Series and silently fill NaN.
        df = pd.DataFrame({"a": pd.Series(["x", "y"], dtype="object")})
        df.index = [5, 9]
        assert_normalized(df)  # off by default
        with pytest.raises(AssertionError, match="RangeIndex"):
            assert_normalized(df, require_clean_index=True)

    def test_the_where_label_reaches_the_message(self):
        # Sweeps run one assertion across dozens of parametrized cases; without
        # the label a failure does not say which case produced it.
        df = pd.DataFrame({"a": pd.Series([1.0], dtype="float64")})
        with pytest.raises(AssertionError, match="build_node_column given 0"):
            assert_normalized(df, where="build_node_column given 0")


class TestAssertNoNaBecameZero:
    def _before(self):
        return pd.DataFrame({"capacity": pd.Series([pd.NA, 5.0], dtype="Float64")})

    def test_accepts_na_that_stayed_na(self):
        after = self._before().copy()
        assert_no_na_became_zero(self._before(), after)

    def test_accepts_na_that_became_a_real_value(self):
        # Filling NA with a genuine number is a legitimate transform; only the
        # collapse to zero destroys the NA/0 distinction.
        after = pd.DataFrame({"capacity": pd.Series([7.0, 5.0], dtype="Float64")})
        assert_no_na_became_zero(self._before(), after)

    def test_rejects_na_that_became_zero(self):
        after = pd.DataFrame({"capacity": pd.Series([0.0, 5.0], dtype="Float64")})
        with pytest.raises(AssertionError, match="was pd.NA and is now 0"):
            assert_no_na_became_zero(self._before(), after)

    def test_ignores_rows_the_transform_dropped(self):
        after = pd.DataFrame({"capacity": pd.Series([5.0], dtype="Float64")}, index=[1])
        assert_no_na_became_zero(self._before(), after)

    def test_ignores_columns_the_transform_added(self):
        after = self._before().copy()
        after["node"] = pd.Series(["FI_elec", "SE_elec"], dtype="object")
        assert_no_na_became_zero(self._before(), after)


class TestCatalogue:
    def test_ids_are_unique_so_failures_are_identifiable(self):
        # Duplicate parametrize ids make a sweep failure ambiguous -- you cannot
        # tell which of two entries produced it.
        ids = [nasty_id(v) for v in NASTY_CELLS]
        duplicates = {i for i in ids if ids.count(i) > 1}
        assert not duplicates, f"colliding nasty-value ids: {sorted(duplicates)}"

    def test_covers_the_failure_modes_that_motivated_it(self):
        def has(pred) -> bool:
            # Predicates must never compare a catalogue entry with ==: pd.NA
            # raises TypeError on truthiness. Type-check first, always.
            return any(pred(v) for v in NASTY_CELLS)

        assert has(lambda v: v is None)
        assert has(lambda v: v is pd.NA)
        assert has(lambda v: isinstance(v, float) and math.isnan(v))
        assert has(lambda v: isinstance(v, float) and math.isinf(v))
        assert has(lambda v: isinstance(v, str) and v == "")
        assert has(lambda v: isinstance(v, str) and v.strip().lower() == "nan")
        assert has(lambda v: isinstance(v, bool))
        assert has(lambda v: isinstance(v, datetime))
        # Both halves of the hash story: the ignore marker, and a single '#'
        # that is *not* a marker -- the two used to be the same character, which
        # is how a broken formula came to delete its own row.
        assert has(lambda v: isinstance(v, str) and v.startswith("##"))
        assert has(lambda v: isinstance(v, str) and v.startswith("#") and not v.startswith("##"))
        assert has(lambda v: isinstance(v, str) and "_" in v)
        assert has(lambda v: isinstance(v, str) and v.lower() == "all")
        assert has(lambda v: isinstance(v, (int, float)) and not isinstance(v, bool) and v == 0)

    def test_frame_with_cell_puts_the_value_in_every_column(self):
        df = frame_with_cell(["country", "grid"], "#comment")
        assert list(df.iloc[0]) == ["#comment", "#comment"]
        # The filler row matters: an all-NA frame would make every column
        # legitimately object and the sweep would prove nothing.
        assert len(df) == 2
        assert not df.iloc[1].isna().any()

    def test_frame_with_blank_column_blanks_exactly_one(self):
        df = frame_with_blank_column(["country", "grid"], "grid")
        assert df["grid"].isna().all()
        assert not df["country"].isna().any()
