"""Boundary 7: ``GDX_exchange.prepare_values_for_gdx``, the last gate before GDX.

This is the single place where NaN becomes 0, because it is the single place
where the GAMS convention actually starts.  Everything upstream keeps NaN
meaning "no data".

The gate follows the project error policy: it never raises.  Unusable rows are
dropped and logged at ``error``, which makes the run report failure, while a
recoverable gap is filled and logged at ``warn``.
"""

import numpy as np
import pandas as pd
import pytest

from src.GDX_exchange import prepare_values_for_gdx
from tests._common.contracts import assert_gams_ready
from tests._common.fixtures import FakeLogger

DIMS = ("grid", "node")


def _frame(values, grids=None, nodes=None) -> pd.DataFrame:
    n = len(values)
    return pd.DataFrame(
        {
            "grid": grids if grids is not None else ["elec"] * n,
            "node": nodes if nodes is not None else ["FI00_elec"] * n,
            "value": pd.Series(values, dtype="Float64"),
        }
    )


def _gate(df, logger):
    return prepare_values_for_gdx(df, logger, dimensions=DIMS, where="test")


class TestMissingValues:
    def test_na_becomes_zero(self):
        out = _gate(_frame([1.0, pd.NA, 3.0]), FakeLogger())
        assert out["value"].tolist() == [1.0, 0.0, 3.0]

    def test_the_conversion_is_reported_with_a_count(self):
        """The fill is correct here; doing it *silently* is the bug.

        A missing wind capacity factor becoming zero generation is a real
        modelling outcome, and the user has to be able to see that it happened.
        """
        logger = FakeLogger()
        _gate(_frame([pd.NA, pd.NA, 3.0]), logger)
        logger.assert_logged("2 of 3", level="warn")

    def test_clean_data_logs_nothing(self):
        # The gate must be quiet when there is nothing to say, or the warning
        # above becomes noise people learn to ignore.
        logger = FakeLogger()
        _gate(_frame([1.0, 2.0]), logger)
        logger.assert_clean()

    def test_output_satisfies_the_gams_ready_contract(self):
        out = _gate(_frame([1.0, pd.NA]), FakeLogger())
        assert_gams_ready(out, dimensions=DIMS, where="after gate")


class TestNonFiniteValues:
    @pytest.mark.parametrize("bad", [np.inf, -np.inf], ids=["inf", "-inf"])
    def test_non_finite_rows_are_dropped_and_logged_as_errors(self, bad):
        # GAMS *accepts* INF, so these would be written happily and then make
        # the model unbounded somewhere unrelated. Better to fail here.
        logger = FakeLogger()
        out = _gate(_frame([1.0, bad, 3.0]), logger)

        assert out["value"].tolist() == [1.0, 3.0]
        logger.assert_logged("non-finite", level="error")


class TestDimensionValues:
    @pytest.mark.parametrize(
        "bad_node", [pd.NA, "", "   "], ids=["NA", "empty", "whitespace"]
    )
    def test_blank_dimension_rows_are_dropped_and_logged_as_errors(self, bad_node):
        """A blank dimension cell would become the GAMS set element ``''``.

        That is the failure mode the old code had: ``fill_all_na`` turned a
        missing country code into an empty string and the pipeline carried on,
        so the model silently gained a node named "".
        """
        logger = FakeLogger()
        out = _gate(_frame([1.0, 2.0], nodes=["FI00_elec", bad_node]), logger)

        assert out["node"].tolist() == ["FI00_elec"]
        logger.assert_logged("GAMS set", level="error")

    def test_a_dimension_is_checked_even_when_categorical(self):
        # ProcessorRunner casts dimension columns to `category` for speed, so
        # the check has to survive that dtype.
        df = _frame([1.0, 2.0], nodes=["FI00_elec", ""])
        df["node"] = df["node"].astype("category")
        logger = FakeLogger()

        out = _gate(df, logger)

        assert len(out) == 1
        logger.assert_logged("GAMS set", level="error")

    def test_zero_is_a_value_not_a_blank(self):
        # Guards against over-eager emptiness rules: node "0" is a legal, if
        # unusual, set element and must survive.
        logger = FakeLogger()
        out = _gate(_frame([1.0], nodes=["0"]), logger)
        assert out["node"].tolist() == ["0"]
        logger.assert_clean()


class TestPassThrough:
    def test_an_empty_frame_is_returned_untouched(self):
        empty = _frame([])
        assert len(_gate(empty, FakeLogger())) == 0

    def test_none_is_returned_untouched(self):
        assert _gate(None, FakeLogger()) is None

    def test_the_input_frame_is_not_mutated(self):
        # The gate is called per climate year inside a dict comprehension; if it
        # mutated in place, the caller's data would change under it.
        original = _frame([1.0, pd.NA])
        _gate(original, FakeLogger())
        assert original["value"].isna().any()

    def test_rows_and_dimension_columns_survive(self):
        out = _gate(_frame([1.0, 2.0]), FakeLogger())
        assert list(out.columns) == ["grid", "node", "value"]
        assert len(out) == 2
