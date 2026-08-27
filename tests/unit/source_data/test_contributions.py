"""Boundary 8: what an outside stage may add to a ``df_*`` table.

A timeseries processor returns contributions -- frames named after the
source-data tables -- and they are folded in once the timeseries phase is over.
Two things have to hold for ``BBExcelPipeline`` to be able to read the result
without knowing where any row came from:

- **the workbook wins**, so a contribution can add but never overwrite;
- **nothing else moves**, so a column no contribution mentions keeps the exact
  dtype the source stage gave it, and existing rows keep their order.

The source-side conventions apply on both sides of the merge: ``pd.NA`` and
``0`` are distinct, and an all-NA column is ``object``.
"""

from __future__ import annotations

import pandas as pd
import pytest

import src.utils as utils
from src.source_data.source_data_contributions import (
    CONTRIBUTION_KEYS,
    apply_contributions,
    combine_contributions,
    merge_contribution,
    validate_contribution,
)
from tests._common.contracts import assert_normalized
from tests._common.fixtures import FakeLogger

NODE_KEY = CONTRIBUTION_KEYS["nodedata"]


@pytest.fixture
def logger():
    return FakeLogger()


def nodedata(*rows: dict) -> pd.DataFrame:
    """A source-side nodedata frame; every row needs a grid and a node.

    Standardised on the way out, because that is what the source stage hands
    over: a bare ``pd.DataFrame`` types a numeric column ``float64`` rather than
    ``Float64``, and the merge is supposed to leave dtypes exactly as it found
    them -- which a fixture in the wrong dtype would hide.
    """
    defaults = {"grid": "elec", "node": "FI_elec"}
    return utils.standardize_df_dtypes(pd.DataFrame([{**defaults, **row} for row in rows]))


def merge(source, contribution, logger, name="nodedata"):
    return merge_contribution(
        source, contribution, CONTRIBUTION_KEYS[name], name=name, logger=logger
    )


class TestValidation:
    def test_an_unknown_frame_name_is_refused(self, logger):
        out = validate_contribution(
            "nonsense", nodedata({}), processor="P", logger=logger
        )
        assert out is None
        logger.assert_logged("not one of the source data tables", level="warn")

    @pytest.mark.parametrize(
        "value", ["a string", 42, None, [1, 2, 3]], ids=["str", "int", "None", "list"]
    )
    def test_anything_that_is_not_a_dataframe_is_refused(self, logger, value):
        assert validate_contribution("nodedata", value, processor="P", logger=logger) is None
        logger.assert_logged("expected pd.DataFrame", level="error")

    def test_an_empty_frame_is_refused_and_said_out_loud(self, logger):
        # A processor put the frame there on purpose, so producing nothing is
        # worth a word rather than silence that reads like success.
        out = validate_contribution(
            "nodedata", pd.DataFrame(columns=["grid", "node"]), processor="P", logger=logger
        )
        assert out is None
        logger.assert_logged("empty", level="warn")

    def test_a_missing_key_column_is_refused(self, logger):
        out = validate_contribution(
            "nodedata", pd.DataFrame({"node": ["FI_elec"]}), processor="P", logger=logger
        )
        assert out is None
        logger.assert_logged("key column", level="error")

    def test_a_blank_key_value_is_refused(self, logger):
        """A blank key becomes the GAMS set element '', which is a broken key."""
        out = validate_contribution(
            "nodedata",
            pd.DataFrame({"grid": ["elec", None], "node": ["FI_elec", "SE_elec"]}),
            processor="P",
            logger=logger,
        )
        assert out is None
        logger.assert_logged("blank", level="error")

    def test_column_names_are_lowercased(self, logger):
        out = validate_contribution(
            "nodedata",
            pd.DataFrame({"Grid": ["elec"], "Node": ["FI_elec"], "Influx": [-5.0]}),
            processor="P",
            logger=logger,
        )
        assert set(out.columns) == {"grid", "node", "influx"}

    def test_a_repeated_key_keeps_the_first_and_reports(self, logger):
        out = validate_contribution(
            "nodedata",
            pd.DataFrame({
                "grid": ["elec", "elec"],
                "node": ["FI_elec", "FI_elec"],
                "influx": [-5.0, -9.0],
            }),
            processor="P",
            logger=logger,
        )
        assert len(out) == 1
        assert out["influx"].iloc[0] == -5.0
        logger.assert_logged("repeating", level="warn")

    def test_a_well_formed_contribution_passes_the_dtype_contract(self, logger):
        out = validate_contribution(
            "nodedata",
            pd.DataFrame({"grid": ["elec"], "node": ["FI_elec"], "influx": [-5.0]}),
            processor="P",
            logger=logger,
        )
        assert_normalized(out, where="validated contribution")
        logger.assert_no_errors()


class TestTheWorkbookWins:
    def test_a_contribution_fills_where_the_workbook_said_nothing(self, logger):
        source = nodedata({"influx": pd.NA})
        out = merge(source, nodedata({"influx": -5.0}), logger)

        assert out["influx"].iloc[0] == -5.0

    def test_a_contribution_cannot_overwrite_a_workbook_value(self, logger):
        source = nodedata({"influx": -1.0})
        out = merge(source, nodedata({"influx": -5.0}), logger)

        assert out["influx"].iloc[0] == -1.0

    def test_an_explicit_zero_in_the_workbook_still_wins(self, logger):
        """The sharpest case: on the source side ``0`` is data, not absence.

        If it were treated as absence here the contribution would win, and a
        workbook could no longer say "this is deliberately zero" about anything
        a processor also has an opinion on.
        """
        source = nodedata({"usetimeseries": 0})
        out = merge(source, nodedata({"usetimeseries": 1}), logger)

        assert out["usetimeseries"].iloc[0] == 0


class TestRowsAndColumns:
    def test_an_unmatched_key_is_appended(self, logger):
        source = nodedata({"node": "FI_elec"})
        out = merge(source, nodedata({"node": "SE_elec"}), logger)

        assert list(out["node"]) == ["FI_elec", "SE_elec"]

    def test_a_matched_key_does_not_add_a_row(self, logger):
        source = nodedata({"node": "FI_elec", "influx": pd.NA})
        out = merge(source, nodedata({"node": "FI_elec", "influx": -5.0}), logger)

        assert len(out) == 1

    def test_existing_rows_keep_their_order(self, logger):
        source = nodedata({"node": "SE_elec"}, {"node": "AT_elec"}, {"node": "FI_elec"})
        out = merge(source, nodedata({"node": "DE_elec"}), logger)

        assert list(out["node"])[:3] == ["SE_elec", "AT_elec", "FI_elec"]

    def test_a_column_the_workbook_never_had_is_created(self, logger):
        out = merge(nodedata({}), nodedata({"influx": -5.0}), logger)

        assert out["influx"].iloc[0] == -5.0

    def test_a_column_nothing_touched_keeps_its_dtype(self, logger):
        """The reason the frame is not re-standardised as a whole.

        ``price`` was settled by the source stage; re-deciding it here would let
        a contribution to some other column silently change it.
        """
        source = nodedata({"price": 30.0, "influx": pd.NA})
        before = source["price"].dtype

        out = merge(source, nodedata({"influx": -5.0}), logger)

        assert out["price"].dtype == before

    def test_an_appended_row_is_blank_rather_than_zero(self, logger):
        """A node a processor knows and the workbook does not has no price.

        ``0`` would be a claim -- and on the source side a claim that survives
        into the workbook as a real value.
        """
        source = nodedata({"node": "FI_elec", "price": 30.0})
        out = merge(source, nodedata({"node": "SE_elec"}), logger)

        assert pd.isna(out.loc[out["node"] == "SE_elec", "price"].iloc[0])

    def test_the_result_satisfies_the_dtype_contract(self, logger):
        source = nodedata({"node": "FI_elec", "price": 30.0, "influx": pd.NA})
        out = merge(source, nodedata({"node": "SE_elec", "influx": -5.0}), logger)

        assert_normalized(out, where="merged nodedata")


class TestDegenerateInputs:
    def test_an_empty_source_takes_the_contribution_whole(self, logger):
        out = merge(pd.DataFrame(), nodedata({"influx": -5.0}), logger)

        assert list(out["node"]) == ["FI_elec"]

    def test_an_empty_contribution_leaves_the_source_alone(self, logger):
        source = nodedata({"price": 30.0})
        out = merge(source, pd.DataFrame(), logger)

        pd.testing.assert_frame_equal(out, source)

    def test_a_source_with_repeated_keys_is_reported_and_left_alone(self, logger):
        """Nothing can be matched unambiguously, so nothing is."""
        source = nodedata({"node": "FI_elec"}, {"node": "FI_elec"})
        out = merge(source, nodedata({"node": "FI_elec", "influx": -5.0}), logger)

        assert "influx" not in out.columns
        logger.assert_logged("more than one row", level="warn")


class TestCombining:
    def test_two_producers_of_the_same_table_are_stacked(self):
        combined = combine_contributions([
            {"nodedata": nodedata({"node": "FI_elec"})},
            {"nodedata": nodedata({"node": "SE_elec"})},
        ])

        assert list(combined["nodedata"]["node"]) == ["FI_elec", "SE_elec"]

    def test_different_tables_stay_apart(self):
        combined = combine_contributions([
            {"nodedata": nodedata({})},
            {"boundarydata": pd.DataFrame([{
                "grid": "elec", "node": "FI_elec",
                "param_gnboundarytypes": "upwardLimit", "usetimeseries": 1,
            }])},
        ])

        assert set(combined) == {"nodedata", "boundarydata"}

    def test_a_producer_with_nothing_to_say_contributes_no_key(self):
        combined = combine_contributions([{}, {"nodedata": pd.DataFrame()}])

        assert combined == {}


class TestApplying:
    def test_the_pipeline_frame_is_replaced_in_place(self, logger):
        class Pipeline:
            df_nodedata = nodedata({"node": "FI_elec", "influx": pd.NA})

        pipeline = Pipeline()
        apply_contributions(
            pipeline, {"nodedata": nodedata({"node": "FI_elec", "influx": -5.0})}, logger
        )

        assert pipeline.df_nodedata["influx"].iloc[0] == -5.0

    def test_an_unknown_table_is_reported_rather_than_created(self, logger):
        class Pipeline:
            df_nodedata = nodedata({})

        pipeline = Pipeline()
        apply_contributions(pipeline, {"nonsense": nodedata({})}, logger)

        assert not hasattr(pipeline, "df_nonsense")
        logger.assert_logged("not one of the source data tables", level="warn")
