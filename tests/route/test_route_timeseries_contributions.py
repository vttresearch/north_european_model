"""What the timeseries phase contributes, all the way to ``inputData.xlsx``.

The phase itself does not run here -- it needs the real PECD and TYNDP inputs --
but it reaches the builder through exactly one door: contributions merged into
the source data tables. So handing ``run_route`` the frames a processor would
have returned exercises the whole of that route, through the real
``apply_contributions``.

Three things travel it in the shipped configs:

- a node the workbooks do not have, which becomes a row of the model;
- a boundary that comes from a series rather than a constant, which nothing
  downstream could work out on its own;
- a constant influx for a demand grid with no profile of its own.
"""

import pandas as pd
import pytest

from tests._common.asserts import assert_workbook_consistent, cell, rows_for
from tests._common.routes import run_route
from tests._common.workbook_text import load_workbook_fixture

pytestmark = pytest.mark.route


def nodedata(**row) -> pd.DataFrame:
    return pd.DataFrame([{"grid": "elec", "node": "FI_elec", **row}])


def boundarydata(**row) -> pd.DataFrame:
    return pd.DataFrame([{
        "grid": "elec",
        "node": "FI_elec",
        "param_gnboundarytypes": "upwardLimit",
        **row,
    }])


@pytest.fixture(scope="module")
def minimal_text():
    return load_workbook_fixture("minimal")


class TestANodeOnlyTheProcessorKnows:
    """A processor may introduce a node, and that is how it does it.

    The domain sheets used to absorb any dimension value a processor produced,
    whatever the rest of the model said. Now the processor states it as data, and
    the node arrives as a node: in the domain sheet, in p_gn, with its grid.
    """

    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory, minimal_text):
        return run_route(
            tmp_path_factory.mktemp("ts_node"),
            workbooks={"data.xlsx": minimal_text},
            contributions={"nodedata": nodedata(grid="dheat", node="FI_dheat")},
        )

    def test_the_route_runs_cleanly(self, result):
        result.logger.assert_no_errors()

    def test_and_the_workbook_is_consistent(self, result):
        assert_workbook_consistent(result.sheets)

    def test_the_node_reaches_the_node_sheet(self, result):
        assert "FI_dheat" in result.sheets["node"]["node"].tolist()

    def test_and_its_grid_reaches_the_grid_sheet(self, result):
        assert "dheat" in result.sheets["grid"]["grid"].tolist()

    def test_and_the_pair_becomes_a_p_gn_row(self, result):
        assert len(rows_for(result.sheets["p_gn"], grid="dheat", node="FI_dheat")) == 1

    def test_a_node_the_workbook_already_has_is_not_duplicated(self, tmp_path, minimal_text):
        # The merge matches on (grid, node), so saying what the workbook already
        # says adds nothing.
        result = run_route(
            tmp_path,
            workbooks={"data.xlsx": minimal_text},
            contributions={"nodedata": nodedata(grid="elec", node="FI_elec")},
        )

        assert len(rows_for(result.sheets["p_gn"], grid="elec", node="FI_elec")) == 1


class TestABoundaryThatComesFromASeries:
    """The one claim a processor has to make about its own output.

    ``changes.inc`` turns ``useTimeseries`` *off* again for a series that proves
    to be flat, but nothing ever turns it on -- so if the workbook does not say
    it, Backbone reads the node's constant and never looks at the GDX.
    """

    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory, minimal_text):
        return run_route(
            tmp_path_factory.mktemp("ts_boundary"),
            workbooks={"data.xlsx": minimal_text},
            contributions={"boundarydata": boundarydata(usetimeseries=1)},
        )

    def test_the_route_runs_cleanly(self, result):
        result.logger.assert_no_errors()

    def test_the_boundary_says_it_uses_a_series(self, result):
        assert cell(
            result.sheets["p_gnBoundaryPropertiesForStates"],
            "useTimeseries",
            grid="elec", node="FI_elec", param_gnBoundaryTypes="upwardLimit",
        ) == 1

    def test_and_no_constant_beside_it(self, result):
        """Backbone treats the two as one either/or property.

        A constant next to the flag is the shape ``changes.inc`` writes when it
        decides a series is flat, so writing it here would be the build claiming
        a decision it has not made.
        """
        assert not cell(
            result.sheets["p_gnBoundaryPropertiesForStates"],
            "useConstant",
            grid="elec", node="FI_elec", param_gnBoundaryTypes="upwardLimit",
        )

    def test_the_node_gains_a_state_variable(self, result):
        # A bounded state is a state: the node has something to store.
        assert cell(
            result.sheets["p_gn"], "energyStoredPerUnitOfState",
            grid="elec", node="FI_elec",
        ) == 1


class TestAWorkbookConstantAndAProcessorSeries:
    """Both describe the same boundary. The series wins, and it is a stated rule.

    The workbook writes boundary constants as columns of nodedata; the source
    stage melts them into the same long table the processor contributes to. So
    this is where the two producers meet.
    """

    WORKBOOK = """
[unittypedata]
Generator_ID | unittype | grid_output1 | eff00 | isSource
windturbine  | WindOnFI | elec         | 1     | 1

[unitdata]
Country | Generator_ID | Scenario | Year | capacity_output1
FI      | windturbine  | all      | 1    | 100

[nodedata]
Country | Grid | Scenario | Year | nodeBalance | upwardLimit
FI      | elec | all      | 1    | 1           | 500

[demanddata]
Country | Grid | Scenario | Year | TWh/year
FI      | elec | all      | 1    | 5
"""

    def _boundary(self, result):
        return rows_for(
            result.sheets["p_gnBoundaryPropertiesForStates"],
            grid="elec", node="FI_elec", param_gnBoundaryTypes="upwardLimit",
        )

    def test_the_workbook_constant_stands_on_its_own(self, tmp_path):
        # The control: without a contribution the melted constant is what the
        # sheet says, which is what it said before this table existed.
        result = run_route(tmp_path, workbooks={"data.xlsx": self.WORKBOOK})

        boundary = self._boundary(result)
        assert boundary["useConstant"].iloc[0] == 1
        assert boundary["constant"].iloc[0] == 500

    def test_a_series_takes_precedence_over_it(self, tmp_path):
        result = run_route(
            tmp_path,
            workbooks={"data.xlsx": self.WORKBOOK},
            contributions={"boundarydata": boundarydata(usetimeseries=1)},
        )

        boundary = self._boundary(result)
        assert boundary["useTimeseries"].iloc[0] == 1
        assert not boundary["useConstant"].iloc[0]

    def test_the_displaced_constant_still_bounds_the_start(self, tmp_path):
        """It leaves the sheet but not the table, which is the point of the table.

        ``add_storage_starts`` reads the upwardLimit from ``df_boundarydata``
        rather than from the sheet it just wrote, so a node whose limit comes
        from a series still has a level to start at -- 70% of 500 here, and for
        hydro a number ``changes.inc`` will replace.
        """
        result = run_route(
            tmp_path,
            workbooks={"data.xlsx": self.WORKBOOK},
            contributions={"boundarydata": boundarydata(usetimeseries=1)},
        )

        assert cell(result.sheets["p_gn"], "boundStart", grid="elec", node="FI_elec") == 1
        assert cell(
            result.sheets["p_gnBoundaryPropertiesForStates"], "constant",
            grid="elec", node="FI_elec", param_gnBoundaryTypes="reference",
        ) == 350


class TestAConstantInfluxForAGridWithNoProfile:
    """What the "other demands" step contributes now that it writes no GDX.

    It used to inflate one number into every hour of the window and ship it as a
    timeseries, which ``changes.inc`` collapsed straight back into
    ``p_gn('influx')``.
    """

    @pytest.fixture(scope="class")
    def result(self, tmp_path_factory, minimal_text):
        return run_route(
            tmp_path_factory.mktemp("ts_influx"),
            workbooks={"data.xlsx": minimal_text},
            contributions={"nodedata": nodedata(influx=-570.78)},
        )

    def test_the_route_runs_cleanly(self, result):
        result.logger.assert_no_errors()

    def test_the_influx_reaches_p_gn(self, result):
        assert cell(
            result.sheets["p_gn"], "influx", grid="elec", node="FI_elec"
        ) == -570.78

    WITH_ITS_OWN_INFLUX = """
[unittypedata]
Generator_ID | unittype | grid_output1 | eff00 | isSource
windturbine  | WindOnFI | elec         | 1     | 1

[unitdata]
Country | Generator_ID | Scenario | Year | capacity_output1
FI      | windturbine  | all      | 1    | 100

[nodedata]
Country | Grid | Scenario | Year | nodeBalance | influx
FI      | elec | all      | 1    | 1           | -100
"""

    def test_a_workbook_influx_is_not_overwritten(self, tmp_path):
        # The workbook wins wherever it said anything, so a node whose demand
        # someone wrote by hand keeps the number they wrote.
        result = run_route(
            tmp_path,
            workbooks={"data.xlsx": self.WITH_ITS_OWN_INFLUX},
            contributions={"nodedata": nodedata(influx=-570.78)},
        )

        assert cell(
            result.sheets["p_gn"], "influx", grid="elec", node="FI_elec"
        ) == -100
