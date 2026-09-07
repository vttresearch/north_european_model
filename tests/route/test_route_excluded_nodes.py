"""Excluding a node takes the units connected to it, and the build says how many.

``exclude_nodes`` removes a node. For nodedata and demanddata that is one row.
A unitdata row is a whole unit, and it goes whole: a unit declares up to ten
connections, and if any of their nodes is excluded the entire row is dropped,
not just that connection.

That is intended -- a unit whose heat output has no node cannot be represented
as it stands, and turning it into a different unit silently would be worse. It
is also not what "exclude a node" sounds like, and both blacklists pass
``log_warning=False``, so it used to happen without a word.

A count, not names, and not a warning. Nothing is wrong: the reader configured
the exclusion. What they cannot otherwise know is how much of the model left
with it, and a CHP plant is the case where that includes electricity capacity
the exclusion was not aimed at.

The count is units, not rows, and it is exact: it is what the merged table
would have held minus what it holds. Rows would overstate it three ways --
several sheets can describe one unit, a row may belong to another scenario, and
a ``remove`` row may have been going to delete it anyway. Excluding all eighteen
Spanish nodes from ``config_OT2030.ini`` -- a real way to shorten a run --
counts 30 rows, 19 keys, and 18 units. 18 is what the model loses.
"""

from tests._common.routes import config_for_workbooks, run_source

MESSAGE = "removed 1 unit(s)"

# One unit with two outputs. Excluding the heat node takes the electricity
# capacity with it, which is the case worth counting.
CHP = """
[unittypedata]
scenario | year | Generator_ID | unittype | grid_input1 | grid_output1 | grid_output2 | eff00
all      | 1    | chp          | CHP      | biomass     | elec         | dheat        | 0.9

[nodedata]
Country | Grid  | Scenario | Year | nodeBalance
FI      | elec  | all      | 1    | 1
FI      | dheat | all      | 1    | 1

[unitdata]
Country | Generator_ID | unit_name_prefix | Scenario | Year | capacity_output1
FI      | chp          |                  | all      | 1    | 100
"""

WORKBOOKS = {"data.xlsx": CHP}


def _run(tmp_path, exclude_nodes=()):
    config = config_for_workbooks(WORKBOOKS)
    config["exclude_nodes"] = list(exclude_nodes)
    return run_source(tmp_path, workbooks=WORKBOOKS, config=config)


class TestAnExcludedNodeTakesItsUnits:
    def test_the_unit_goes_whole(self, tmp_path):
        """Its electricity output is not the excluded one, and it goes anyway."""
        pipeline, _ = _run(tmp_path, ["FI_dheat"])
        assert pipeline.df_unitdata.empty

    def test_the_count_is_reported(self, tmp_path):
        _, logger = _run(tmp_path, ["FI_dheat"])
        assert any(MESSAGE in message for message in logger.messages)

    def test_it_is_not_a_warning(self, tmp_path):
        """The reader configured the exclusion; there is nothing to act on."""
        _, logger = _run(tmp_path, ["FI_dheat"])
        logger.assert_not_logged(MESSAGE, level="warn")

    def test_excluding_nothing_says_nothing(self, tmp_path):
        pipeline, logger = _run(tmp_path)

        assert not pipeline.df_unitdata.empty
        assert not any(MESSAGE in message for message in logger.messages)

    def test_excluding_an_unrelated_node_says_nothing(self, tmp_path):
        """Absence is not a defect: a node no unit touches costs no units."""
        pipeline, logger = _run(tmp_path, ["SE_dheat"])

        assert not pipeline.df_unitdata.empty
        assert not any(MESSAGE in message for message in logger.messages)
