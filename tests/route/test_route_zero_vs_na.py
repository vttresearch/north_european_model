"""Boundary 3: where ``pd.NA`` and ``0`` stop being different things.

The source stage keeps them distinct -- NA is an empty cell, 0 is an explicitly
chosen zero, and ``method=replace`` depends on being able to tell them apart.
``BBExcelPipeline`` then crosses into the GAMS convention, where ``0``, NA and
"not set" are the same thing.

The convention is easy to state and easy to get wrong, so it is asserted where
it actually lives: the *same* source edit must produce a visible difference on
one side of the boundary and none at all on the other. Stated as a delta, so it
survives new parameter columns and changed defaults.
"""

import pandas as pd
import pytest

from tests._common.asserts import assert_workbook_consistent, cell, rows_for
from tests._common.delta import assert_delta, workbook_delta
from tests._common.routes import run_route, run_source
from tests._common.workbook_text import workbook_text_with

pytestmark = pytest.mark.route

# vomCosts is a plain optional cost: absent for most units, and legitimately
# zero for some. Exactly the shape where confusing NA with 0 goes unnoticed.
BASE = """\
// Two units so that one row can be edited while the other pins everything else.
[unittypedata]
Generator_ID | unittype | grid_output1 | eff00 | isSource
windturbine  | WindOn   | elec         | 1     | 1
gasturbine   | GasOCGT  | elec         | 0.4   | 1

[unitdata]
Country | Generator_ID | Scenario | Year | capacity_output1 | vomCosts
FI      | windturbine  | all      | 1    | 100              | 3
FI      | gasturbine   | all      | 1    | 200              | 7

[nodedata]
Country | Grid | Scenario | Year | nodeBalance
FI      | elec | all      | 1    | 1
"""

WHERE = {"Country": "FI", "Generator_ID": "gasturbine"}


def _variant(value):
    return workbook_text_with(BASE, sheet="unitdata", header="vomCosts",
                              value=value, where=WHERE)


ZERO = _variant(0)
EMPTY = _variant(None)


class TestSourceStageKeepsThemDistinct:
    """Boundaries 1-2: NA is an empty cell, 0 is a decision."""

    def test_an_explicit_zero_arrives_as_zero(self, tmp_path):
        source, _ = run_source(tmp_path / "zero", workbooks={"data.xlsx": ZERO})
        assert cell(source.df_unitdata, "vomcosts", generator_id="gasturbine") == 0

    def test_an_empty_cell_arrives_as_na(self, tmp_path):
        source, _ = run_source(tmp_path / "empty", workbooks={"data.xlsx": EMPTY})
        assert pd.isna(cell(source.df_unitdata, "vomcosts", generator_id="gasturbine"))

    def test_the_other_unit_is_untouched_either_way(self, tmp_path):
        # Guards the fixture edit itself: if workbook_text_with had hit the wrong
        # row, the tests above would still pass while testing the wrong thing.
        for name, text in (("zero", ZERO), ("empty", EMPTY)):
            source, _ = run_source(tmp_path / name, workbooks={"data.xlsx": text})
            assert cell(source.df_unitdata, "vomcosts", generator_id="windturbine") == 3


class TestExcelStageTreatsThemAlike:
    """Boundary 3: past here, ``0 = NA = None = not set``."""

    def test_zero_and_empty_produce_an_identical_workbook(self, tmp_path):
        """The convention, as one assertion.

        Both edits mean "no vomCosts for this unit" by the time GAMS reads the
        workbook, so the two builds must be indistinguishable -- including the
        column not materialising in one and not the other.
        """
        zero = run_route(tmp_path / "zero", workbooks={"data.xlsx": ZERO})
        empty = run_route(tmp_path / "empty", workbooks={"data.xlsx": EMPTY})

        zero.logger.assert_no_errors()
        empty.logger.assert_no_errors()
        assert_workbook_consistent(zero.sheets)

        assert_delta(workbook_delta(zero.sheets, empty.sheets), expect_no_change=True)

    def test_a_real_value_is_not_treated_as_absent(self, tmp_path):
        """The other direction, so the test above cannot pass by doing nothing.

        If the pipeline dropped vomCosts entirely, "zero == empty" would hold
        trivially. A genuine value must still reach the workbook.
        """
        zero = run_route(tmp_path / "zero", workbooks={"data.xlsx": ZERO})
        priced = run_route(tmp_path / "priced", workbooks={"data.xlsx": _variant(42)})

        delta = workbook_delta(zero.sheets, priced.sheets)
        assert not delta.is_empty(), (
            "changing vomCosts from 0 to 42 produced no difference at all; "
            "the parameter is not reaching inputData.xlsx"
        )

    def test_the_unedited_unit_keeps_its_cost(self, tmp_path):
        # Provenance rather than pinned values: both the unit's generated name
        # and its cost are read from the source stage, so this test says "the
        # workbook carries what the source produced" without naming either.
        # (The name is built from the *unittype*, not the generator_id --
        # build_unittype_unit_column -- which is exactly the kind of detail a
        # test should not hardcode.)
        route = run_route(tmp_path / "zero", workbooks={"data.xlsx": ZERO})

        unit_name = cell(route.source.df_unitdata, "unit", generator_id="windturbine")
        expected = cell(route.source.df_unitdata, "vomcosts", generator_id="windturbine")

        wind = rows_for(route.sheets["p_gnu_io"], unit=unit_name)
        assert len(wind) == 1
        assert float(wind.iloc[0]["vomCosts"]) == float(expected)
