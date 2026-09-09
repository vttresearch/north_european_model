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

import src.source_workbook_shape as sws
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


class TestWhatAColumnMayBeCalled:
    """The vocabulary behind ``unrecognised_columns``.

    A column nothing recognises is carried through the source stage and then
    ignored, so a mistyped header is indistinguishable from a deliberate one.
    These tests pin the shape of the answer, not the membership of the lists --
    which names Backbone has is `backbone_params`' business, and asserting it
    twice would only mean editing two files per parameter.
    """

    def test_a_parameter_is_recognised_on_the_table_that_carries_it(self):
        assert not sws.unrecognised_columns(["transferCap"], "transferdata")
        assert not sws.unrecognised_columns(["nodeBalance"], "nodedata")
        assert not sws.unrecognised_columns(["eff00"], "unitdata")

    def test_a_parameter_on_the_wrong_table_is_not(self):
        """The per-table split is the point: transferCap says nothing on a node."""
        assert sws.unrecognised_columns(["transferCap"], "nodedata")
        assert sws.unrecognised_columns(["nodeBalance"], "transferdata")

    def test_a_connection_suffix_does_not_hide_a_parameter(self):
        assert not sws.unrecognised_columns(
            ["capacity_output1", "capacity_input3", "grid_output2"], "unitdata"
        )

    def test_a_boundary_type_is_a_nodedata_column(self):
        """build_boundarydata melts these; the properties it produces are not input."""
        assert not sws.unrecognised_columns(["upwardLimit", "maxSpill"], "nodedata")

    def test_an_emission_factor_is_open_ended_on_a_node(self):
        assert not sws.unrecognised_columns(
            ["emission_CO2", "emission_somethingNobodyHasNamedYet"], "nodedata"
        )

    def test_an_emission_factor_on_a_unit_sheet_is_reported(self):
        """What splitting the two emission families per table buys.

        A single global ``emission_`` rule would wave this through, and the
        factor would reach nothing.
        """
        assert sws.unrecognised_columns(["emission_CO2"], "unitdata")
        assert not sws.unrecognised_columns(["emission_group1"], "unitdata")

    def test_country_is_only_a_column_where_it_filters_anything(self):
        """unittypedata is global, so a country column on it changes nothing."""
        assert not sws.unrecognised_columns(["country"], "unitdata")
        assert sws.unrecognised_columns(["country"], "unittypedata")
        assert sws.unrecognised_columns(["country"], "emissiondata")

    def test_the_spelling_comes_back_as_it_was_written(self):
        """The reader searches a workbook for what they typed, not for a slug."""
        assert sws.unrecognised_columns(["MaxRampUpp"], "nodedata") == ["MaxRampUpp"]

    def test_a_table_this_module_does_not_know_yields_nothing(self):
        """Same rule as known_dimension_values: cannot tell is not the same as all.

        A new data category must not have every one of its columns reported on
        the day someone adds it.
        """
        assert sws.unrecognised_columns(["anything", "at", "all"], "futuredata") == []

    def test_method_is_recognised_everywhere_because_it_is_created_everywhere(self):
        """Several shipped sheets omit it and normalize_dataframe adds it."""
        for table in sws.STRUCTURAL_COLUMNS:
            assert not sws.unrecognised_columns(["method"], table)


class TestTheHandDeclaredVocabularyHasNotRotted:
    """The two entries that nothing can derive, guarded from both directions.

    Everything else in the vocabulary comes from `backbone_params` or from this
    module, so it cannot drift. `DERIVATION_INPUTS` names columns read by name
    at a call site, and a consumer could stop reading one with nothing to
    notice -- which would leave the build silently accepting a column that no
    longer reaches anything.
    """

    def test_every_declared_derivation_input_is_still_read_somewhere(self, repo_root):
        """A declaration outliving its reader is the way this list rots."""
        sources = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in (repo_root / "src").rglob("*.py")
        ).lower()

        for table, columns in sws.DERIVATION_INPUTS.items():
            for column in columns:
                assert column.lower() in sources, (
                    f"{table!r} declares {column!r} as read by a later stage, "
                    "but no file under src/ mentions it any more."
                )

    def test_the_processors_and_the_static_table_agree(self):
        """Neither side imports the other, so only a test can hold them equal.

        The source stage runs first and must not depend on which processors a
        config enables, which is why the authority is static; the processors
        declare the same thing so the fact lives beside the code that uses it.
        """
        from src.timeseries.processors.DH_demand_fromTemperature import (
            DH_demand_fromTemperature,
        )
        from src.timeseries.processors.elec_demand_TYNDP2024 import (
            elec_demand_TYNDP2024,
        )

        declared = set(sws.DERIVATION_INPUTS["demanddata"])
        for processor in (DH_demand_fromTemperature, elec_demand_TYNDP2024):
            assert set(processor.reads_source_columns) <= declared, (
                f"{processor.__name__} reads a demanddata column that "
                "source_workbook_shape.DERIVATION_INPUTS does not list, so the "
                "source stage would report it as read by nothing."
            )
