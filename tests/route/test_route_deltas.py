"""One source cell changes -- what moves, and what must not.

The assertion that carries the weight here is the one nobody writes: everything
*not* listed is asserted unchanged. Enumerating the hundreds of cells that
stayed put is the golden file being reinvented, and it is exactly the part that
stops being maintainable.

Both workbooks are built by the code under test in the same run, so adding a
parameter column or changing a default moves both sides equally and needs no
edit here.
"""

import pytest

from tests._common.delta import assert_delta, workbook_delta
from tests._common.routes import run_route
from tests._common.workbook_text import load_workbook_fixture, workbook_text_with

pytestmark = pytest.mark.route

CHP = load_workbook_fixture("chp")
READER_RULES = load_workbook_fixture("reader_rules")

UNIT = {"Country": "FI", "Generator_ID": "chpbio"}
#: The generated unit name; a coordinate, so pinning it is correct (rule R5).
GNU_ELEC_OUT = ("elec", "FI_elec", "FI_CHPbio", "output")


def _sheets(tmp_path, name, text):
    result = run_route(tmp_path / name, workbooks={"data.xlsx": text})
    result.logger.assert_no_errors()
    return result.sheets


# The baselines are rebuilt identically by every test in this module, and a
# build is about a second. Module-scoped so they are paid for once; the variants
# differ per test and cannot be shared.
@pytest.fixture(scope="module")
def chp_base(tmp_path_factory):
    return _sheets(tmp_path_factory.mktemp("chp_base"), "base", CHP)


@pytest.fixture(scope="module")
def reader_rules_base(tmp_path_factory):
    return _sheets(tmp_path_factory.mktemp("rr_base"), "base", READER_RULES)


class TestASingleValueEdit:
    def test_raising_the_output_capacity_moves_only_that_cell(self, tmp_path, chp_base):
        """The declared capacity is carried, not derived.

        This unit sets cv, so fill_capacities does not infer the input or heat
        capacities from it -- they stay 0. If that ever changes, this test fails
        with the two extra cells named, which is the correct outcome: a
        derivation was added and the test should be told about it.
        """
        base = chp_base
        variant = _sheets(
            tmp_path,
            "variant",
            workbook_text_with(CHP, sheet="unitdata", header="capacity_output1",
                               value=750, where=UNIT),
        )

        assert_delta(
            workbook_delta(base, variant),
            changed=[("p_gnu_io", GNU_ELEC_OUT, "capacity", 750)],
        )

    def test_changing_a_cost_moves_only_that_cost(self, tmp_path, chp_base):
        base = chp_base
        variant = _sheets(
            tmp_path,
            "variant",
            workbook_text_with(CHP, sheet="unitdata", header="vomCosts",
                               value=9, where=UNIT),
        )

        assert_delta(
            workbook_delta(base, variant),
            changed=[("p_gnu_io", GNU_ELEC_OUT, "vomCosts", 9)],
        )

    def test_changing_an_efficiency_moves_only_p_unit(self, tmp_path, chp_base):
        # eff00 lives on the unittype, so this edits a different sheet and must
        # still touch exactly one output cell.
        base = chp_base
        variant = _sheets(
            tmp_path,
            "variant",
            workbook_text_with(CHP, sheet="unittypedata", header="eff00",
                               value=0.5, where={"Generator_ID": "chpbio"}),
        )

        assert_delta(
            workbook_delta(base, variant),
            changed=[("p_unit", ("FI_CHPbio",), "eff00", 0.5)],
        )


class TestEditsThatMustChangeNothing:
    def test_editing_a_note_is_a_no_op(self, tmp_path, reader_rules_base):
        """'Note' columns are dropped by the reader (read_input_excels:96-98).

        Free-text notes are used throughout the real workbooks, so a note edit
        reaching the model would be both surprising and invisible.
        """
        base = reader_rules_base
        variant = _sheets(
            tmp_path,
            "variant",
            workbook_text_with(READER_RULES, sheet="unitdata", header="Note",
                               value="an entirely different remark",
                               where={"Generator_ID": "keeper", "Scenario": "all"}),
        )

        assert_delta(workbook_delta(base, variant), expect_no_change=True)

    def test_editing_a_row_below_the_blank_row_is_a_no_op(self, tmp_path, reader_rules_base):
        # It was never read in the first place. Asserting this end to end is how
        # the truncation rule stays visible: silently losing those rows is what
        # made the previous fixture useless.
        base = reader_rules_base
        variant = _sheets(
            tmp_path,
            "variant",
            workbook_text_with(READER_RULES, sheet="unitdata",
                               header="capacity_output1", value=999,
                               where={"Generator_ID": "truncated"}),
        )

        assert_delta(workbook_delta(base, variant), expect_no_change=True)

    def test_editing_a_commented_row_is_a_no_op(self, tmp_path, reader_rules_base):
        base = reader_rules_base
        variant = _sheets(
            tmp_path,
            "variant",
            workbook_text_with(READER_RULES, sheet="unitdata",
                               header="capacity_output1", value=999,
                               where={"Country": "#FI"}),
        )

        assert_delta(workbook_delta(base, variant), expect_no_change=True)


class TestTheDeltaCanActuallyFail:
    def test_an_unlisted_change_is_caught(self, tmp_path, chp_base):
        """Guards the guard.

        Every no-op test above would pass trivially if the delta could not see a
        difference, so one test asserts that it can.
        """
        base = chp_base
        variant = _sheets(
            tmp_path,
            "variant",
            workbook_text_with(CHP, sheet="unitdata", header="capacity_output1",
                               value=750, where=UNIT),
        )

        with pytest.raises(AssertionError, match="unexpected cell change"):
            assert_delta(workbook_delta(base, variant), expect_no_change=True)
