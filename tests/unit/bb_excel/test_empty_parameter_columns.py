"""Empty parameter columns, and the one that must never be dropped.

``p_gn``, ``p_gnn`` and ``p_gnu_io`` are declared in ``src_files/indexSheet.xlsx``
with ``Cdim=1``: the parameter block *is* the column dimension. A sheet left with
no parameter column at all is not an empty sheet, it is a GDXXRW dimension error
-- so the drop that removes all-empty parameter columns has to stop at one.

Two things put that within reach:

- ``utils.is_col_empty`` is True for a zero-length column
  (``tests/unit/test_utils.py::TestIsColEmpty``), so a build that produced no
  rows makes *every* parameter column look empty at the same moment;
- every sheet is written unconditionally and ``add_index_sheet`` filters the
  INDEX by sheet name alone, so the degenerate sheet still gets an INDEX row
  pointing GAMS at it.

The zero-row cases below are all reachable from real config rather than
constructed: no (grid, node) pair survived the collection step, transfer rows
missing a domain, unit rows carrying no ``grid_*`` column.

With at least one row the guarantee holds by itself, because ``PARAM_GN_DEFAULTS``
and ``PARAM_GNN_DEFAULTS`` set ``isActive`` before the drop runs. That is exactly
why the empty build is the case worth pinning: it is the only one where the
defaults have nothing to write to.
"""

import pandas as pd
import pytest

from tests._common.bb_excel import make_pipeline

#: Dimension columns per sheet -- everything else in the frame is a parameter.
DIMENSIONS = {
    "p_gn": ["grid", "node"],
    "p_gnn": ["grid", "from_node", "to_node"],
    "p_gnu_io": ["grid", "node", "unit", "input_output"],
    "p_unit": ["unit"],
}


def _parameters(frame: pd.DataFrame, sheet: str) -> list[str]:
    return [c for c in frame.columns if c not in DIMENSIONS[sheet]]


def _p_gn_with_no_pairs() -> pd.DataFrame:
    """No (grid, node) pair survived collection, so the loop never runs."""
    return make_pipeline().create_p_gn(
        pd.DataFrame(columns=["grid", "node"]),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
    )


def _p_gnn_with_every_row_skipped() -> pd.DataFrame:
    """Transfer data exists but no row names all three domains, so all are skipped.

    Deliberately not an empty frame: create_p_gnn returns early on that, and the
    early return is not the case under test.
    """
    return make_pipeline().create_p_gnn(pd.DataFrame({"grid": ["elec"]}))


def _p_gnu_io_with_every_row_skipped() -> pd.DataFrame:
    """Unit rows with no grid_* column, so there is no put to write."""
    return make_pipeline().create_p_gnu_io(pd.DataFrame({"unit": ["u1"]}))


def _p_unit_with_no_units() -> pd.DataFrame:
    """No unit reached p_unit, so the parameter loop never runs."""
    return make_pipeline().create_p_unit(
        pd.DataFrame(columns=["unit", "unittype"]), pd.DataFrame()
    )


BUILDERS = {
    "p_gn": _p_gn_with_no_pairs,
    "p_gnn": _p_gnn_with_every_row_skipped,
    "p_gnu_io": _p_gnu_io_with_every_row_skipped,
    "p_unit": _p_unit_with_no_units,
}


class TestAZeroRowSheetKeepsItsColumnDimension:
    @pytest.mark.parametrize("sheet", sorted(BUILDERS))
    def test_at_least_one_parameter_column_survives(self, sheet):
        # The load-bearing assertion. Cdim=1 means GDXXRW reads the parameter
        # names off the header; with none left there is no column dimension to
        # read and the symbol fails to load.
        frame = BUILDERS[sheet]()
        assert _parameters(frame, sheet), (
            f"{sheet} kept no parameter column, so its Cdim=1 header is empty"
        )

    @pytest.mark.parametrize("sheet", sorted(BUILDERS))
    def test_the_dimension_columns_survive_too(self, sheet):
        # A drop that reaches a dimension column would be the same class of
        # error from the other side.
        frame = BUILDERS[sheet]()
        for dimension in DIMENSIONS[sheet]:
            assert dimension in frame.columns, f"{sheet} dropped dimension {dimension}"

    @pytest.mark.parametrize("sheet", sorted(BUILDERS))
    def test_there_is_no_data_row_to_speak_of(self, sheet):
        # Negative control: if a fixture started producing rows, the tests above
        # would pass for a reason that has nothing to do with the empty build.
        # create_fake_MultiIndex adds one marker row, so one row means no data.
        frame = BUILDERS[sheet]()
        assert len(frame) <= 1, f"{sheet} produced data rows; the fixture no longer isolates the empty build"


class TestAPopulatedSheetKeepsOnlyWhatWasSet:
    def test_p_unit_writes_only_the_parameters_in_use(self):
        """p_unit lists all 26 PARAM_UNIT entries; a build sets a handful.

        The rest used to be written as columns of zeros -- 19 of 29 columns in
        an OT2030 build. The survivors here are exactly the PARAM_UNIT_DEFAULTS
        entries, which is the point: a default is a value someone gets, so its
        column is never empty.
        """
        p_unit = make_pipeline().create_p_unit(
            pd.DataFrame([{"unit": "u1", "unittype": "WindOn"}]),
            pd.DataFrame([{"unit": "u1"}]),
        )

        assert _parameters(p_unit, "p_unit") == ["isActive", "availability", "eff00", "op00"]
