"""What each kind of source row turns into.

Organised by fixture rather than by output sheet, because a fixture is the unit
a contributor actually reasons about: "a CHP unit produces this".

Assertions follow the R6 order. Counting is used where the *fixture itself*
declares the number (a unittype with three grid_<put> columns must yield three
connections), provenance where a value is carried, and pinned values only for
the documented format contracts.
"""

import pytest

from tests._common.asserts import (
    assert_fake_multiindex,
    assert_passthrough,
    assert_workbook_consistent,
    cell,
    rows_for,
)
from tests._common.routes import run_route
from tests._common.workbook_text import load_workbook_fixture

pytestmark = pytest.mark.route


@pytest.fixture(scope="module")
def chp(tmp_path_factory):
    return run_route(
        tmp_path_factory.mktemp("chp"),
        workbooks={"data.xlsx": load_workbook_fixture("chp")},
    )


@pytest.fixture(scope="module")
def chp_unit(chp):
    units = chp.sheets["unit"]["unit"].tolist()
    assert len(units) == 1, f"fixture should declare exactly one unit, got {units}"
    return units[0]


class TestChpIsWellFormed:
    def test_the_route_runs_cleanly(self, chp):
        chp.logger.assert_no_errors()

    def test_and_the_workbook_is_consistent(self, chp):
        assert_workbook_consistent(chp.sheets)


class TestConnections:
    def test_one_gnu_row_per_declared_connection(self, chp, chp_unit):
        """The unittype declares grid_input1, grid_output1 and grid_output2.

        Counted against the fixture's own declaration rather than a magic 3, so
        adding a parameter column cannot break it.
        """
        assert len(rows_for(chp.sheets["p_gnu_io"], unit=chp_unit)) == 3

    def test_split_into_one_input_and_two_outputs(self, chp, chp_unit):
        gnu = chp.sheets["p_gnu_io"]
        assert len(rows_for(gnu, unit=chp_unit, input_output="input")) == 1
        assert len(rows_for(gnu, unit=chp_unit, input_output="output")) == 2

    def test_each_connection_lands_on_the_grid_its_unittype_named(self, chp, chp_unit):
        gnu = chp.sheets["p_gnu_io"]
        assert len(rows_for(gnu, unit=chp_unit, grid="biomass", input_output="input")) == 1
        assert len(rows_for(gnu, unit=chp_unit, grid="elec", input_output="output")) == 1
        assert len(rows_for(gnu, unit=chp_unit, grid="heat", input_output="output")) == 1

    def test_every_connection_gets_a_node_on_its_own_grid(self, chp, chp_unit):
        # Relational: node names are generated, so the test checks the pairing
        # rather than the spelling.
        for row in rows_for(chp.sheets["p_gnu_io"], unit=chp_unit).to_dict("records"):
            assert len(rows_for(chp.sheets["p_gn"], grid=row["grid"], node=row["node"])) == 1


class TestValuesAreCarried:
    def test_the_declared_output_capacity_reaches_p_gnu_io(self, chp, chp_unit):
        # Provenance: neither the test nor the fixture names the number, so
        # editing chp.wb.txt never edits this test.
        assert_passthrough(
            chp.sheets["p_gnu_io"], "capacity",
            chp.source.df_unitdata, "capacity",
            out_key={"unit": chp_unit, "grid": "elec", "input_output": "output"},
            src_key={"generator_id": "chpbio"},
        )

    def test_vom_costs_reach_the_priced_connection(self, chp, chp_unit):
        assert_passthrough(
            chp.sheets["p_gnu_io"], "vomCosts",
            chp.source.df_unitdata, "vomcosts",
            out_key={"unit": chp_unit, "grid": "elec", "input_output": "output"},
            src_key={"generator_id": "chpbio"},
        )

    def test_efficiency_reaches_p_unit(self, chp, chp_unit):
        assert_passthrough(
            chp.sheets["p_unit"], "eff00",
            chp.source.df_unitdata, "eff00",
            out_key={"unit": chp_unit},
            src_key={"generator_id": "chpbio"},
        )

    @pytest.mark.parametrize("parameter", ["cb", "cv"])
    def test_the_backpressure_parameters_reach_the_electricity_output(
        self, chp, chp_unit, parameter
    ):
        """cb and cv describe the heat-to-power relationship.

        They land on the *electricity output* row rather than on p_unit, which
        is worth pinning as a coordinate: putting them on the wrong connection
        would change what the optimiser is allowed to do while every count and
        reference still checked out.
        """
        value = cell(
            chp.sheets["p_gnu_io"], parameter,
            unit=chp_unit, grid="elec", input_output="output",
        )
        assert float(value) > 0

    def test_the_fuel_node_carries_its_price(self, chp):
        assert_passthrough(
            chp.sheets["p_gn"], "price",
            chp.source.df_nodedata, "price",
            out_key={"grid": "biomass", "node": "FI_biomass"},
            src_key={"grid": "biomass", "country": "FI"},
        )


class TestDomains:
    def test_each_declared_grid_appears_once(self, chp):
        assert sorted(chp.sheets["grid"]["grid"]) == ["biomass", "elec", "heat"]

    def test_one_node_per_grid_for_a_single_country(self, chp):
        assert len(chp.sheets["node"]) == 3

    def test_the_unit_is_linked_to_its_unittype(self, chp, chp_unit):
        # unitUnittype is what lets Backbone group units by technology.
        assert len(rows_for(chp.sheets["unitUnittype"], unit=chp_unit)) == 1

    def test_a_fuel_node_is_priced_rather_than_balanced(self, chp):
        """Node classification, the part of create_p_gn most easily got wrong.

        A node with a price and no nodeBalance is a source of fuel, not a
        balance point. Swapping the two makes the model quietly unbounded.
        """
        assert float(cell(chp.sheets["p_gn"], "usePrice", grid="biomass", node="FI_biomass")) == 1
        assert float(cell(chp.sheets["p_gn"], "nodeBalance", grid="biomass", node="FI_biomass")) == 0

    def test_a_demand_node_is_balanced_rather_than_priced(self, chp):
        assert float(cell(chp.sheets["p_gn"], "nodeBalance", grid="elec", node="FI_elec")) == 1
        assert float(cell(chp.sheets["p_gn"], "usePrice", grid="elec", node="FI_elec")) == 0


@pytest.fixture(scope="module")
def transfer(tmp_path_factory):
    return run_route(
        tmp_path_factory.mktemp("transfer"),
        workbooks={"data.xlsx": load_workbook_fixture("transfer")},
    )


class TestTransferLinks:
    def test_the_route_runs_cleanly(self, transfer):
        transfer.logger.assert_no_errors()
        assert_workbook_consistent(transfer.sheets)

    def test_one_directional_row_per_source_row(self, transfer):
        # The format is directional: a link present one way carries power one
        # way. Two source rows must not be collapsed into a single link.
        assert len(transfer.sheets["p_gnn"]) == 2

    def test_each_direction_is_its_own_row(self, transfer):
        gnn = transfer.sheets["p_gnn"]
        assert len(rows_for(gnn, from_node="FI_elec", to_node="SE_elec")) == 1
        assert len(rows_for(gnn, from_node="SE_elec", to_node="FI_elec")) == 1

    def test_the_directions_keep_their_own_capacities(self, transfer):
        # Asymmetric on purpose in the fixture: copying one direction's capacity
        # onto the other would be invisible in a symmetric test.
        forward = float(cell(transfer.sheets["p_gnn"], "transferCap",
                             from_node="FI_elec", to_node="SE_elec"))
        backward = float(cell(transfer.sheets["p_gnn"], "transferCap",
                              from_node="SE_elec", to_node="FI_elec"))
        assert forward != backward

    def test_the_capacity_is_carried_from_the_source_row(self, transfer):
        assert_passthrough(
            transfer.sheets["p_gnn"], "transferCap",
            transfer.source.df_transferdata, "transfercap",
            out_key={"from_node": "FI_elec", "to_node": "SE_elec"},
            src_key={"from_country": "FI", "to_country": "SE"},
        )

    def test_both_ends_contribute_to_the_node_domain(self, transfer):
        # A link to a country with no units of its own must still declare its
        # node, or the transfer references something that does not exist.
        nodes = set(transfer.sheets["node"]["node"])
        assert {"FI_elec", "SE_elec"} <= nodes

    def test_p_gnn_carries_the_fake_multiindex(self, transfer):
        # The sheet the minimal fixture cannot reach, since it has no transfers.
        assert_fake_multiindex(
            transfer.raw_sheets["p_gnn"], ["grid", "from_node", "to_node"]
        )


@pytest.fixture(scope="module")
def userconstraint(tmp_path_factory):
    return run_route(
        tmp_path_factory.mktemp("uc"),
        workbooks={"data.xlsx": load_workbook_fixture("userconstraint")},
    )


class TestUserConstraints:
    def test_the_route_runs_cleanly(self, userconstraint):
        userconstraint.logger.assert_no_errors()
        assert_workbook_consistent(userconstraint.sheets)

    def test_underscores_survive_in_user_constraint_dimensions(self, userconstraint):
        """The one category exempt from drop_underscore_values.

        Every other category loses a row whose string value contains '_',
        because underscore separates the parts of a node name. User constraint
        dimensions *refer* to nodes and units, whose names contain underscores
        by construction, so filtering them would silently delete constraints.

        Untestable with a hand-edited binary workbook, and the clearest single
        argument for fixtures you can write an underscore into on purpose.
        """
        assert len(rows_for(userconstraint.sheets["p_userconstraint"],
                            **{"1st dimension": "FI_elec"})) == 1

    def test_the_referenced_node_actually_exists(self, userconstraint):
        # The underscore is only worth keeping if it names something real.
        assert len(rows_for(userconstraint.sheets["node"], node="FI_elec")) == 1

    def test_a_sheet_using_only_some_dimensions_still_builds(self, userconstraint):
        """Regression, in two stages -- and the first fix was not enough.

        create_p_userconstraint detected the absent 3rd/4th dimension columns,
        logged a warning, and then selected them anyway, raising a bare
        KeyError. A constraint using one or two dimensions is ordinary, so that
        was fixed by creating the columns as NA.

        Which produced a workbook GAMS would not load. p_userconstraint is
        Rdim=6, so all four uc slots are labels, and Backbone checks that a slot
        a parameter does not use holds exactly '-' -- inc/1e_inputs.gms aborts
        otherwise. The build stopped crashing and started emitting a file that
        failed later, further away, with a message about the wrong thing.
        """
        uc = userconstraint.sheets["p_userconstraint"]
        assert len(uc) == 2
        assert "3rd dimension" in uc.columns
        assert set(uc["3rd dimension"]) == {"-"}

    def test_the_group_reaches_the_group_domain(self, userconstraint):
        assert "elecLimit".casefold() in {
            str(g).casefold() for g in userconstraint.sheets["group"]["group"]
        }
