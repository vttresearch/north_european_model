"""The route runs, and what it produces is a structurally valid workbook.

Deliberately says almost nothing about values. Everything here survives a new
parameter column, a changed default, or an edited fixture -- and it fails on
exactly the kind of breakage that makes GAMS reject the workbook.
"""

import pytest

from tests._common.asserts import (
    assert_fake_multiindex,
    assert_has_columns,
    assert_workbook_consistent,
    rows_for,
)
from tests._common.excel_read import EXPECTED_SHEETS
from tests._common.routes import run_route
from tests._common.workbook_text import load_workbook_fixture

pytestmark = pytest.mark.route

MINIMAL = load_workbook_fixture("minimal")


@pytest.fixture(scope="module")
def minimal(tmp_path_factory):
    # Module-scoped: building the workbook takes about a second, and every test
    # here only reads the result.
    return run_route(tmp_path_factory.mktemp("minimal"), workbooks={"data.xlsx": MINIMAL})


class TestItRuns:
    def test_without_errors(self, minimal):
        minimal.logger.assert_no_errors()

    def test_and_writes_the_workbook(self, minimal):
        assert minimal.output_file.is_file()

    def test_into_the_folder_it_was_given(self, minimal):
        # No CWD dependence on this tier: the output folder is absolute, so
        # nothing lands beside the repo.
        assert minimal.output_file.parent == minimal.output_folder


class TestStructure:
    def test_every_expected_sheet_is_present(self, minimal):
        assert EXPECTED_SHEETS <= set(minimal.sheets)

    def test_every_cross_sheet_invariant_holds(self, minimal):
        # The one assertion every route test starts with. A route test that
        # called only this would still be worth having.
        assert_workbook_consistent(minimal.sheets)

    def test_the_index_sheet_comes_first(self, minimal):
        # GDXXRW reads the index to find every other sheet, so its position is
        # part of the contract rather than cosmetic.
        assert list(minimal.sheets)[0] == "index"

    @pytest.mark.parametrize(
        "sheet, dimensions",
        [
            ("p_gn", ["grid", "node"]),
            ("p_gnu_io", ["grid", "node", "unit", "input_output"]),
            ("p_unit", ["unit"]),
            ("p_gnBoundaryPropertiesForStates", ["grid", "node", "param_gnBoundaryTypes"]),
        ],
    )
    def test_the_fake_multiindex_marker_row(self, minimal, sheet, dimensions):
        # Pinned exactly: this IS the format contract with GDXXRW, one of the
        # five cases where pinning is correct.
        # p_gnn is absent from this fixture (no transfers) and is covered by the
        # transfer fixture instead -- see test_route_features.py.
        assert_fake_multiindex(minimal.raw_sheets[sheet], dimensions)

    def test_a_sheet_with_no_data_is_written_empty_not_malformed(self, minimal):
        """A category the fixture does not use still yields a usable sheet.

        p_gnn has no rows here because there are no transfers. GDXXRW needs the
        sheet to exist regardless; what it must not be is half-written.
        """
        assert "p_gnn" in minimal.sheets
        assert minimal.sheets["p_gnn"].empty

    def test_scenario_tags_are_carried_through(self, minimal):
        tags = minimal.sheets["add_scen_tags"]
        assert_has_columns(tags, ["scenario", "year"])
        assert len(tags) == 1


class TestTheDataArrives:
    def test_the_declared_unit_reaches_the_unit_domain(self, minimal):
        # Relational, not counted: the fixture declares one unit, so exactly one
        # unit must exist -- but the test does not name its generated id.
        assert len(minimal.sheets["unit"]) == 1

    def test_the_unit_has_one_output_connection(self, minimal):
        # The unittype declares grid_output1 and nothing else, so one row.
        unit = minimal.sheets["unit"]["unit"].iloc[0]
        gnu = minimal.sheets["p_gnu_io"]
        assert len(rows_for(gnu, unit=unit)) == 1
        assert len(rows_for(gnu, unit=unit, input_output="output")) == 1

    def test_the_electricity_node_exists(self, minimal):
        assert len(rows_for(minimal.sheets["node"], node="FI_elec")) == 1

    def test_the_grid_domain_comes_from_the_data(self, minimal):
        assert minimal.sheets["grid"]["grid"].tolist() == ["elec"]

    def test_unused_domains_are_empty_rather_than_absent(self, minimal):
        # GDXXRW needs the sheet to exist even with no rows; dropping it would
        # break the import rather than merely omit a set.
        assert minimal.sheets["restype"].empty
