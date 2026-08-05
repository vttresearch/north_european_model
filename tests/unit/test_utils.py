"""Tests for ``src/utils.py`` -- where the dtype and zero/NA rules actually live.

``standardize_df_dtypes`` is the gatekeeper: it decides what every downstream
function is entitled to assume.  ``is_col_empty`` is where ``0`` and ``NA``
become interchangeable, and ``fill_all_na`` / ``fill_numeric_na`` are the
functions that carry a frame across the boundary from "NA and 0 are distinct"
to the GAMS convention where they are not.

Boundary 1 and boundary 4 of the NA/zero map in tests/README.md.
"""

import sys

import numpy as np
import pandas as pd
import pytest

from src.utils import (
    fill_all_na,
    fill_numeric_na,
    is_col_empty,
    parse_sys_args,
    standardize_df_dtypes,
)
from tests._common.contracts import assert_normalized


def _col(values, dtype=None) -> pd.DataFrame:
    return pd.DataFrame({"c": pd.Series(values, dtype=dtype)})


class TestStandardizeDfDtypes:
    """The gatekeeper. Its output shape is the contract everything else assumes."""

    def test_numeric_columns_become_float64(self):
        out = standardize_df_dtypes(_col([1, 2]))
        assert str(out["c"].dtype) == "Float64"

    def test_text_columns_become_object(self):
        out = standardize_df_dtypes(_col(["FI", "SE"]))
        assert str(out["c"].dtype) == "object"

    def test_numeric_strings_become_float64(self):
        # Excel hands back text surprisingly often; a column of "1"/"2" is a
        # numeric column that happens to have arrived as text.
        out = standardize_df_dtypes(_col(["1", "2"]))
        assert str(out["c"].dtype) == "Float64"
        assert out["c"].tolist() == [1.0, 2.0]

    def test_mixed_text_and_numbers_stays_object(self):
        # Converting would introduce NAs that were not in the input, silently
        # deleting data. utils.py:85 refuses on exactly that basis.
        out = standardize_df_dtypes(_col(["1", "not a number"]))
        assert str(out["c"].dtype) == "object"

    @pytest.mark.parametrize("text", ["nan", "NaN", " NAN "])
    def test_the_string_nan_becomes_pd_na(self, text):
        out = standardize_df_dtypes(_col([text, "2"]))
        assert out["c"].isna().iloc[0]

    @pytest.mark.parametrize("text", ["NA", "None", "null", "-"])
    def test_other_missing_looking_strings_are_left_alone(self, text):
        # Deliberate: only 'nan' is converted (utils.py:75). 'NA' is a plausible
        # real value, and guessing here would destroy data.
        out = standardize_df_dtypes(_col([text, "x"]))
        assert not out["c"].isna().any()

    def test_an_all_na_column_becomes_object(self):
        """The cascade-bug fix, pinned.

        An empty string column and an empty float column are indistinguishable
        once both are all-NA.  Typing them ``object`` means "no assumption has
        been made"; typing either one ``Float64`` invents an assumption that
        the next consumer will act on and be wrong about.
        """
        out = standardize_df_dtypes(_col([None, None]))
        assert str(out["c"].dtype) == "object"

    def test_an_all_na_column_is_object_regardless_of_its_input_dtype(self):
        # The property that matters: the *source* dtype must not leak through.
        from_float = standardize_df_dtypes(_col([np.nan, np.nan], dtype="float64"))
        from_text = standardize_df_dtypes(_col([None, None], dtype="object"))
        assert str(from_float["c"].dtype) == str(from_text["c"].dtype) == "object"

    def test_object_columns_use_pd_na_never_float_nan(self):
        out = standardize_df_dtypes(_col(["FI", np.nan]))
        assert out["c"].iloc[1] is pd.NA

    def test_an_all_zero_numeric_column_still_lands_on_float64(self):
        """is_col_empty calls an all-zero numeric column "empty" (utils.py:168).

        Pass 1 therefore retypes it to object -- and pass 2 converts it straight
        back to numeric because nothing is actually missing.  The round trip is
        easy to misread as "all-zero columns become object", so it is pinned:
        zero is a value, and only genuine absence gets the object treatment.
        """
        out = standardize_df_dtypes(_col([0, 0]))
        assert str(out["c"].dtype) == "Float64"
        assert out["c"].tolist() == [0.0, 0.0]

    def test_zero_and_na_stay_distinguishable(self):
        # Boundary 1-2: the difference method=replace depends on.
        out = standardize_df_dtypes(_col([0, None]))
        assert out["c"].iloc[0] == 0
        assert pd.isna(out["c"].iloc[1])

    def test_does_not_mutate_its_input(self):
        original = _col(["1", "2"])
        before = original["c"].tolist()
        standardize_df_dtypes(original)
        assert original["c"].tolist() == before

    def test_output_satisfies_the_contract(self):
        messy = pd.DataFrame(
            {
                "text": ["FI", np.nan],
                "number": ["1", "2"],
                "empty": [None, None],
                "mixed": ["1", "x"],
            }
        )
        assert_normalized(standardize_df_dtypes(messy), where="standardize_df_dtypes")


class TestIsColEmpty:
    """Three different rules for three kinds of column (utils.py:142-147)."""

    @pytest.mark.parametrize(
        "values, expected",
        [
            pytest.param([0, 0], True, id="all-zero-is-empty"),
            pytest.param([0, None], True, id="zero-and-na-is-empty"),
            pytest.param([None, None], True, id="all-na-is-empty"),
            pytest.param([0, 1], False, id="a-nonzero-value-is-not-empty"),
            pytest.param([-1, 0], False, id="negative-counts-as-a-value"),
        ],
    )
    def test_numeric_columns_treat_zero_as_empty(self, values, expected):
        """This is where 0 and NA become interchangeable.

        Efficient for GAMS and correct at that boundary, but it means an
        explicitly-zero numeric column reads as "nothing set" -- the single
        most confusing rule in the codebase, so it is pinned exhaustively.
        """
        assert is_col_empty(pd.Series(values, dtype="Float64")) is expected

    @pytest.mark.parametrize(
        "values, expected",
        [
            pytest.param([False, False], False, id="all-false-is-NOT-empty"),
            pytest.param([None, None], True, id="all-na-is-empty"),
            pytest.param([True, False], False, id="mixed-is-not-empty"),
        ],
    )
    def test_boolean_columns_do_not_treat_false_as_empty(self, values, expected):
        # Deliberately unlike the numeric rule: False is a decision, 0 is not.
        assert is_col_empty(pd.Series(values, dtype="boolean")) is expected

    @pytest.mark.parametrize(
        "values, expected",
        [
            pytest.param(["", ""], True, id="empty-strings"),
            pytest.param(["   ", ""], True, id="whitespace-only"),
            pytest.param([None, ""], True, id="na-and-empty-string"),
            pytest.param(["FI", ""], False, id="a-real-value"),
            pytest.param(["0", ""], False, id="the-string-zero-is-a-value"),
        ],
    )
    def test_text_columns_count_whitespace_as_empty(self, values, expected):
        assert is_col_empty(pd.Series(values, dtype="object")) is expected

    def test_a_zero_length_column_is_empty(self):
        assert is_col_empty(pd.Series([], dtype="object")) is True


class TestFillHelpers:
    """The functions that cross into the GAMS convention (boundary 3)."""

    def test_fill_numeric_na_touches_only_float64_columns(self):
        df = pd.DataFrame(
            {
                "capacity": pd.Series([pd.NA, 5.0], dtype="Float64"),
                "country": pd.Series([pd.NA, "SE"], dtype="object"),
            }
        )
        out = fill_numeric_na(df)
        assert out["capacity"].tolist() == [0.0, 5.0]
        assert pd.isna(out["country"].iloc[0])  # text left alone

    def test_fill_all_na_zeroes_numbers_and_blanks_text(self):
        df = pd.DataFrame(
            {
                "capacity": pd.Series([pd.NA, 5.0], dtype="Float64"),
                "country": pd.Series([pd.NA, "SE"], dtype="object"),
            }
        )
        out = fill_all_na(df)
        assert out["capacity"].tolist() == [0.0, 5.0]
        assert out["country"].tolist() == ["", "SE"]

    def test_fill_all_na_leaves_no_na_behind(self):
        # The postcondition its callers rely on: past this point, no NA guards.
        df = pd.DataFrame(
            {
                "a": pd.Series([pd.NA], dtype="Float64"),
                "b": pd.Series([pd.NA], dtype="object"),
            }
        )
        assert not fill_all_na(df).isna().any().any()

    def test_filling_makes_explicit_zero_and_missing_indistinguishable(self):
        """The convention change, stated as one assertion.

        Before the fill, ``0`` and ``pd.NA`` are different values; after it they
        are the same value.  That is correct for GAMS -- and it is why the fill
        must not happen any earlier than the boundary that needs it, since
        everything upstream loses the ability to tell "explicitly zero" from
        "not given".
        """
        explicit_zero = fill_all_na(_col([0.0], dtype="Float64"))
        missing = fill_all_na(_col([pd.NA], dtype="Float64"))
        assert explicit_zero["c"].iloc[0] == missing["c"].iloc[0] == 0.0

    @pytest.mark.parametrize("filler", [fill_all_na, fill_numeric_na])
    def test_does_not_mutate_its_input(self, filler):
        df = _col([pd.NA, 1.0], dtype="Float64")
        filler(df)
        assert pd.isna(df["c"].iloc[0])


class TestParseSysArgs:
    def test_returns_input_folder_and_config_path_resolved_against_it(self):
        argv = ["build_input_data.py", "src_files", "config_test.ini"]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", argv)
            input_folder, config_file = parse_sys_args()

        assert str(input_folder) == "src_files"
        # The config path is relative to the input folder, not the CWD.
        assert config_file.name == "config_test.ini"
        assert config_file.parent.name == "src_files"

    def test_rejects_the_legacy_key_equals_value_syntax(self):
        argv = ["build_input_data.py", "input_folder=src_files"]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", argv)
            with pytest.raises(SystemExit) as excinfo:
                parse_sys_args()
        assert excinfo.value.code == 1

    def test_requires_both_positional_arguments(self):
        argv = ["build_input_data.py", "src_files"]
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(sys, "argv", argv)
            with pytest.raises(SystemExit):
                parse_sys_args()
