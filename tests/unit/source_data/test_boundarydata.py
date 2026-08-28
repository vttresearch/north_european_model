"""``nodedata``'s boundary columns, and the long table they become.

p_gnBoundaryPropertiesForStates is indexed by boundary *type* -- upwardLimit,
maxSpill, balancePenalty -- while a spreadsheet is good at one row per node. So
the workbook writes a column per type and the source stage melts it into
``df_boundarydata``, one row per (grid, node, type). Same status as the
``emission_XX`` columns create_p_nEmission reads: a documented input format.

What the melt must not do is decide anything. It states ``useconstant`` because
a workbook column *is* a constant, and it keeps a ``0`` rather than dropping it,
because on this side of the pipeline ``0`` and ``pd.NA`` are still different
things. Whether an all-zero boundary says anything is the builder's call.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.backbone_params import PARAM_GN_BOUNDARY_TYPES
from src.source_data.source_data_contributions import BOUNDARY_COLUMNS, build_boundarydata
from tests._common.contracts import assert_normalized
from tests._common.fixtures import FakeLogger


@pytest.fixture
def logger():
    return FakeLogger()


def nodedata(*rows: dict) -> pd.DataFrame:
    defaults = {"grid": "reservoir", "node": "FI_reservoir"}
    return pd.DataFrame([{**defaults, **row} for row in rows])


def rows_for(table: pd.DataFrame, node: str, boundary_type: str) -> pd.DataFrame:
    return table[(table["node"] == node) & (table["param_gnboundarytypes"] == boundary_type)]


class TestWhatBecomesARow:
    def test_a_boundary_column_becomes_a_row_stating_a_constant(self, logger):
        table = build_boundarydata(nodedata({"upwardlimit": 5530000.0}), logger)

        row = rows_for(table, "FI_reservoir", "upwardLimit")
        assert len(row) == 1
        assert row["constant"].iloc[0] == 5530000.0
        assert row["useconstant"].iloc[0] == 1

    def test_the_boundary_type_keeps_its_backbone_spelling(self, logger):
        """Workbook column names are lower-cased; GAMS set elements are not."""
        table = build_boundarydata(nodedata({"balancepenalty": 5000.0}), logger)

        assert set(table["param_gnboundarytypes"]) == {"balancePenalty"}

    def test_several_boundary_columns_become_several_rows(self, logger):
        table = build_boundarydata(
            nodedata({"upwardlimit": 100.0, "maxspill": 20.0, "balancepenalty": 5000.0}),
            logger,
        )

        assert len(table) == 3
        assert len(table[["grid", "node"]].drop_duplicates()) == 1

    def test_a_column_that_is_not_a_boundary_type_is_left_alone(self, logger):
        table = build_boundarydata(nodedata({"price": 30.0, "upwardlimit": 100.0}), logger)

        assert set(table["param_gnboundarytypes"]) == {"upwardLimit"}

    def test_a_blank_cell_produces_no_row(self, logger):
        table = build_boundarydata(
            nodedata({"node": "FI_reservoir", "upwardlimit": 100.0},
                     {"node": "SE_reservoir", "upwardlimit": pd.NA}),
            logger,
        )

        assert list(table["node"]) == ["FI_reservoir"]

    def test_an_explicit_zero_does_produce_a_row(self, logger):
        """``0`` is data here, and stays data until the builder judges it.

        Dropping it at the melt would put the GAMS convention two stages too
        early, and a workbook could no longer say "this limit is deliberately
        zero" at all.
        """
        table = build_boundarydata(nodedata({"maxspill": 0.0}), logger)

        assert rows_for(table, "FI_reservoir", "maxSpill")["constant"].iloc[0] == 0.0


class TestTheTableItself:
    def test_the_column_set_is_the_same_whatever_the_workbook_holds(self, logger):
        """Consumers read `usetimeseries` whether or not anything set it yet."""
        table = build_boundarydata(nodedata({"upwardlimit": 100.0}), logger)

        assert list(table.columns) == BOUNDARY_COLUMNS

    def test_a_property_nothing_set_is_object_rather_than_float(self, logger):
        # The all-NA rule: no assumption has been made about usetimeseries here.
        table = build_boundarydata(nodedata({"upwardlimit": 100.0}), logger)

        assert table["usetimeseries"].dtype == "object"

    def test_the_result_satisfies_the_dtype_contract(self, logger):
        table = build_boundarydata(
            nodedata({"node": "FI_reservoir", "upwardlimit": 100.0, "maxspill": 20.0},
                     {"node": "SE_reservoir", "upwardlimit": 200.0}),
            logger,
        )

        assert_normalized(table, where="df_boundarydata")

    def test_every_declared_boundary_type_can_be_read(self, logger):
        """The melt is driven by the shared list, not by a private one.

        Adding minSpill to PARAM_GN_BOUNDARY_TYPES has to be enough to make
        `minspill` a readable nodedata column, with no edit here.
        """
        row = {t.lower(): 1.0 for t in PARAM_GN_BOUNDARY_TYPES}
        table = build_boundarydata(nodedata(row), logger)

        assert set(table["param_gnboundarytypes"]) == set(PARAM_GN_BOUNDARY_TYPES)


class TestNothingToRead:
    def test_no_node_data_gives_an_empty_table_of_the_right_shape(self, logger):
        table = build_boundarydata(pd.DataFrame(), logger)

        assert table.empty
        assert list(table.columns) == BOUNDARY_COLUMNS

    def test_node_data_with_no_boundary_columns_gives_an_empty_table(self, logger):
        table = build_boundarydata(nodedata({"price": 30.0}), logger)

        assert table.empty
        logger.assert_no_errors()

    def test_node_data_without_a_node_column_is_reported(self, logger):
        table = build_boundarydata(pd.DataFrame({"upwardlimit": [100.0]}), logger)

        assert table.empty
        logger.assert_logged("no 'grid' or 'node'", level="warn")
