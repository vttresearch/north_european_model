"""Which source table declares a dimension -- and why that is four of them.

A value that nothing declares is not a failure, it is a name that looks real: a
mistyped node becomes a node nothing balances, and a series built for one is
written and never read. Answering "is this declared?" needs the declaration set
to be right, and the interesting thing about it is how wide it is.

``nodedata`` and ``demanddata`` are the obvious half. The other half is not:
a `unitdata` row declares a grid and a node **per connection**, which is how
every battery, heat store and fuel grid enters the model without appearing in
nodedata at all. Miss that and the check reports about 110 perfectly correct
rows per build -- which is what the first version of this map did.

The set therefore has to be the same union ``_collect_gn_pairs`` builds. These
tests pin that correspondence rather than the membership of the map.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.source_workbook_shape import (
    DIMENSION_SOURCES,
    base_column_name,
    known_dimension_values,
    tables_of,
    unknown_dimension_values,
)


def tables(**frames: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {name: frame for name, frame in frames.items()}


class TestWhereAGridOrNodeComesFrom:
    def test_nodedata_declares_them(self):
        known = known_dimension_values(
            "node", tables(nodedata=pd.DataFrame({"grid": ["elec"], "node": ["FI_elec"]}))
        )
        assert known == {"FI_elec"}

    def test_demanddata_declares_them(self):
        known = known_dimension_values(
            "node", tables(demanddata=pd.DataFrame({"grid": ["elec"], "node": ["FI_elec"]}))
        )
        assert known == {"FI_elec"}

    def test_a_unit_connection_declares_them(self):
        """The half that is easy to miss.

        A battery unit brings `batterystor` and `FI_batterystor` into the model
        through its own connection columns; nodedata never mentions either.
        """
        known = known_dimension_values(
            "node",
            tables(unitdata=pd.DataFrame({
                "unit": ["FI_battery"],
                "node_output1": ["FI_batterystor"],
            })),
        )
        assert known == {"FI_batterystor"}

    def test_both_ends_of_a_transfer_declare_them(self):
        known = known_dimension_values(
            "node",
            tables(transferdata=pd.DataFrame({
                "grid": ["elec"], "from_node": ["FI_elec"], "to_node": ["SE_elec"],
            })),
        )
        assert known == {"FI_elec", "SE_elec"}

    @pytest.mark.parametrize(
        "column", ["grid_input1", "grid_output1", "grid_output5", "grid"]
    )
    def test_every_connection_suffix_counts(self, column):
        known = known_dimension_values(
            "grid", tables(unitdata=pd.DataFrame({"unit": ["u1"], column: ["batterystor"]}))
        )
        assert known == {"batterystor"}

    def test_a_column_that_is_not_a_declaration_is_ignored(self):
        # capacity_output1 carries the same suffix and declares nothing.
        known = known_dimension_values(
            "grid",
            tables(unitdata=pd.DataFrame({"unit": ["u1"], "capacity_output1": ["100"]})),
        )
        assert known is None


class TestTheMapMirrorsCollectGnPairs:
    """The correspondence that keeps the check honest.

    ``BBExcelPipeline._collect_gn_pairs`` unions nodedata, demanddata,
    p_gnu_io (from unitdata) and both ends of p_gnn (from transferdata). If this
    map were narrower, the check would report values the workbook then writes
    anyway; if it were wider, a real typo would pass.
    """

    COLLECTED_FROM = {"nodedata", "demanddata", "unitdata", "transferdata"}

    @pytest.mark.parametrize("dimension", ["grid", "node"])
    def test_every_table_collect_gn_pairs_reads_can_declare(self, dimension):
        assert set(DIMENSION_SOURCES[dimension]) == self.COLLECTED_FROM


class TestCannotTell:
    def test_no_loaded_table_answers_none_rather_than_empty(self):
        """An empty frame means the source excels were skipped this run.

        Treating it as "the model has no nodes" would report every value in the
        model, on the one run where the user can act on none of them.
        """
        assert known_dimension_values("node", tables(nodedata=pd.DataFrame())) is None

    def test_and_nothing_is_reported_unknown(self):
        assert unknown_dimension_values(["anything"], "node", tables()) == []


class TestWhatIsUnknown:
    SOURCES = {"nodedata": pd.DataFrame({"grid": ["elec"], "node": ["FI_elec"]})}

    def test_a_value_nothing_declares_is_named(self):
        assert unknown_dimension_values(
            ["FI_elec", "TYPO_elec"], "node", self.SOURCES
        ) == ["TYPO_elec"]

    def test_a_declared_value_is_not(self):
        assert unknown_dimension_values(["FI_elec"], "node", self.SOURCES) == []

    def test_also_known_covers_a_producer_declaring_its_own(self):
        # Contributing the node and using it are two halves of one sentence.
        assert unknown_dimension_values(
            ["NEW_elec"], "node", self.SOURCES, also_known=["NEW_elec"]
        ) == []

    def test_missing_values_are_not_reported(self):
        assert unknown_dimension_values(
            ["FI_elec", pd.NA], "node", self.SOURCES
        ) == []

    def test_a_categorical_column_is_accepted(self):
        # main_result's dimension columns are categorical by the time the runner
        # asks, and a plain set is what a contribution offers.
        column = pd.Series(["FI_elec", "TYPO_elec"], dtype="category")
        assert unknown_dimension_values(column, "node", self.SOURCES) == ["TYPO_elec"]


class TestBaseColumnName:
    @pytest.mark.parametrize(
        "column,expected",
        [
            ("grid", "grid"),
            ("grid_output1", "grid"),
            ("node_input5", "node"),
            ("Grid_Output1", "grid"),
            ("capacity_output1", "capacity"),
            ("from_node", "from_node"),
        ],
    )
    def test_the_suffix_is_stripped_and_nothing_else_is(self, column, expected):
        assert base_column_name(column) == expected


class TestTablesOf:
    def test_it_reads_the_df_attributes_a_question_needs(self):
        class Pipeline:
            df_nodedata = pd.DataFrame({"grid": ["elec"], "node": ["FI_elec"]})

        collected = tables_of(Pipeline())
        assert known_dimension_values("node", collected) == {"FI_elec"}

    def test_a_table_the_pipeline_lacks_reads_as_empty(self):
        # Not None: every caller iterates these, and a None would need a guard
        # at each one.
        class Pipeline:
            pass

        assert all(frame.empty for frame in tables_of(Pipeline()).values())
