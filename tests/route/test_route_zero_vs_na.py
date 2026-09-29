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
unittype | grid_output1 | eff00 | isSource
WindOn   | elec         | 1     | 1
GasOCGT  | elec         | 0.4   | 1

[unitdata]
Country | unittype | Scenario | Year | capacity_output1 | vomCosts
FI      | windon   | all      | 1    | 100              | 3
FI      | gasocgt  | all      | 1    | 200              | 7

[nodedata]
Country | Grid | Scenario | Year | nodeBalance
FI      | elec | all      | 1    | 1
"""

WHERE = {"Country": "FI", "unittype": "gasocgt"}


def _variant(value):
    return workbook_text_with(BASE, sheet="unitdata", header="vomCosts",
                              value=value, where=WHERE)


ZERO = _variant(0)
EMPTY = _variant(None)


class TestSourceStageKeepsThemDistinct:
    """Boundaries 1-2: NA is an empty cell, 0 is a decision."""

    def test_an_explicit_zero_arrives_as_zero(self, tmp_path):
        source, _ = run_source(tmp_path / "zero", workbooks={"data.xlsx": ZERO})
        assert cell(source.df_unitdata, "vomcosts", unittype="GasOCGT") == 0

    def test_an_empty_cell_arrives_as_na(self, tmp_path):
        source, _ = run_source(tmp_path / "empty", workbooks={"data.xlsx": EMPTY})
        assert pd.isna(cell(source.df_unitdata, "vomcosts", unittype="GasOCGT"))

    def test_the_other_unit_is_untouched_either_way(self, tmp_path):
        # Guards the fixture edit itself: if workbook_text_with had hit the wrong
        # row, the tests above would still pass while testing the wrong thing.
        for name, text in (("zero", ZERO), ("empty", EMPTY)):
            source, _ = run_source(tmp_path / name, workbooks={"data.xlsx": text})
            assert cell(source.df_unitdata, "vomcosts", unittype="WindOn") == 3


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
        # (The name is built from the unittype as unittypedata spells it, not as
        # this sheet wrote it -- canonicalize_unittype_and_build_unit -- which is
        # exactly the kind of detail a test should not hardcode.)
        route = run_route(tmp_path / "zero", workbooks={"data.xlsx": ZERO})

        unit_name = cell(route.source.df_unitdata, "unit", unittype="WindOn")
        expected = cell(route.source.df_unitdata, "vomcosts", unittype="WindOn")

        wind = rows_for(route.sheets["p_gnu_io"], unit=unit_name)
        assert len(wind) == 1
        assert float(wind.iloc[0]["vomCosts"]) == float(expected)


# A storage node with a start share. The share is the one number where a 0 is a
# value on the GAMS side too: Backbone reads it by its useConstant flag, and 0 is
# the floor of the band.
SHARE_BASE = """\
[unittypedata]
unittype | grid_output1 | eff00 | isSource
WindOnFI | elec         | 1     | 1

[unitdata]
Country | unittype | Scenario | Year | capacity_output1
FI      | WindOnFI | all      | 1    | 100

[nodedata]
Country | Grid | Scenario | Year | nodeBalance | upwardLimit | relative
FI      | elec | all      | 1    | 1           | 500         | 0.5

[demanddata]
Country | Grid | Scenario | Year | TWh/year
FI      | elec | all      | 1    | 5
"""


def _share(value):
    return workbook_text_with(SHARE_BASE, sheet="nodedata", header="relative",
                              value=value, where={"Country": "FI"})


class TestAStartShareOfZeroIsNotAbsent:
    """The documented exception to ``0 = NA`` past the boundary.

    An empty ``relative`` means "no share", and the node starts at a level. A 0
    means "start at the floor", and has to reach the workbook with its flag, or
    Backbone's ``boundStartRelative`` would bind nothing.
    """

    def test_zero_and_empty_produce_different_workbooks(self, tmp_path):
        zero = run_route(tmp_path / "zero", workbooks={"data.xlsx": _share(0)})
        empty = run_route(tmp_path / "empty", workbooks={"data.xlsx": _share(None)})

        zero.logger.assert_no_errors()
        empty.logger.assert_no_errors()
        assert_workbook_consistent(zero.sheets)

        assert not workbook_delta(zero.sheets, empty.sheets).is_empty()

    def test_zero_starts_the_node_at_the_floor_of_its_band(self, tmp_path):
        zero = run_route(tmp_path / "zero", workbooks={"data.xlsx": _share(0)})

        assert cell(zero.sheets["p_gn"], "boundStartRelative", node="FI_elec") == 1
        relative = rows_for(zero.sheets["p_gnBoundaryPropertiesForStates"],
                            node="FI_elec", param_gnBoundaryTypes="relative")
        assert len(relative) == 1
        assert relative["useConstant"].iloc[0] == 1

    def test_empty_starts_the_node_at_a_level(self, tmp_path):
        empty = run_route(tmp_path / "empty", workbooks={"data.xlsx": _share(None)})

        assert cell(empty.sheets["p_gn"], "boundStart", node="FI_elec") == 1
        assert rows_for(empty.sheets["p_gnBoundaryPropertiesForStates"],
                        node="FI_elec", param_gnBoundaryTypes="relative").empty
