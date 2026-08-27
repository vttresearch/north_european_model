"""A node that only one of nodedata and demanddata has heard of.

A node name is built from three cells -- ``country``, ``grid``, ``node_suffix``
-- so a single wrong cell does not produce an error. It produces a *different
node*, spelled plausibly, that quietly takes the demand meant for another one.
The case this was written for was a Helsinki heat row whose country cell read
``NOS0``: it invented ``NOS0_dheat_HKI`` and left ``FI00_dheat_HKI`` a year
short, and ``exclude_nodes = ['NOS0_dheat']`` could not catch it because the
node strings differ.

Nothing downstream can see that. ``bb_excel_pipeline._collect_gn_pairs`` unions
the node names from every source, so the invented node simply becomes a Backbone
node with a balance penalty and no supply.

The obvious detectors do not work. A hardcoded list of node names cannot cover
one-off industrial nodes. A duplicate check on ``(grid, node)`` is wrong because
a suffix is legitimately reused across grids, and one on ``(country, node)``
because industrial demand is added to every country. What is left, and what this
tests, is referential integrity between the two tables -- restricted to grids the
other table actually describes, which is what keeps plain balance nodes quiet.
"""

import pandas as pd

from src.source_data.source_data_loader import report_node_disagreements
from tests._common.fixtures import FakeLogger


def frame(*pairs) -> pd.DataFrame:
    return pd.DataFrame(list(pairs), columns=["grid", "node"])


def dheat(*suffixes) -> list[tuple[str, str]]:
    return [("dheat", f"{code}_dheat") for code in suffixes]


class TestBothDirections:
    def test_a_demand_node_nodedata_has_never_heard_of_is_named(self):
        nodedata = frame(*dheat("FI00", "SE03", "PL00", "AT00"))
        demanddata = frame(*dheat("FI00", "SE03", "PL00", "AT00"),
                           ("dheat", "NOS0_dheat_HKI"))
        logger = FakeLogger()
        report_node_disagreements(nodedata, demanddata, logger)

        logger.assert_logged("NOS0_dheat_HKI", level="warn")
        logger.assert_logged("appear in demanddata but not in nodedata", level="warn")

    def test_the_reverse_direction_is_reported_too(self):
        # A node declared in nodedata that no demand row feeds: it reaches the
        # model with a balance penalty and nothing to serve.
        nodedata = frame(*dheat("FI00", "SE03", "PL00", "AT00"))
        demanddata = frame(*dheat("FI00", "SE03", "PL00"))
        logger = FakeLogger()
        report_node_disagreements(nodedata, demanddata, logger)

        logger.assert_logged("AT00_dheat", level="warn")
        logger.assert_logged("appear in nodedata but not in demanddata", level="warn")

    def test_the_message_does_not_assert_which_cause_it_is(self):
        """A demand row written as 0 and a mistyped country cell look identical here.

        ``filter_nonzero_numeric_rows`` drops an all-zero row, so a node whose
        demand is deliberately zero arrives looking exactly like one nobody
        wrote. Naming either cause sends the reader hunting for the wrong thing,
        so the message names the node and the two tables and stops there.
        """
        nodedata = frame(*dheat("FI00", "SE03", "PL00", "AT00"))
        demanddata = frame(*dheat("FI00", "SE03", "PL00"))
        logger = FakeLogger()
        report_node_disagreements(nodedata, demanddata, logger)

        message = logger.warnings[0]
        assert "AT00_dheat" in message
        assert "nodedata" in message and "demanddata" in message
        for cause in ("mistyped", "typo", "on purpose", "written as 0"):
            assert cause not in message

    def test_agreeing_tables_say_nothing(self):
        nodes = frame(*dheat("FI00", "SE03", "PL00"))
        logger = FakeLogger()
        report_node_disagreements(nodes, nodes.copy(), logger)
        logger.assert_clean()

    def test_it_reports_rather_than_dropping_the_row(self):
        # Which of the two workbooks is wrong is not something this can know.
        nodedata = frame(*dheat("FI00", "SE03", "PL00"))
        demanddata = frame(*dheat("FI00", "SE03", "PL00"), ("dheat", "XX00_dheat"))
        before = demanddata.copy()
        report_node_disagreements(nodedata, demanddata, FakeLogger())
        pd.testing.assert_frame_equal(demanddata, before)


class TestGridsNodedataDoesNotDescribe:
    """The coverage gate, which is what makes the check usable at all."""

    def test_electricity_nodes_with_no_nodedata_rows_are_not_a_finding(self):
        # Plain balance nodes carry no nodedata row. Every elec demand node would
        # otherwise be reported on every build.
        nodedata = frame(*dheat("FI00", "SE03"))
        demanddata = frame(
            *dheat("FI00", "SE03"),
            ("elec", "FI00_elec"), ("elec", "SE03_elec"), ("elec", "PL00_elec"),
        )
        logger = FakeLogger()
        report_node_disagreements(nodedata, demanddata, logger)
        logger.assert_clean()

    def test_one_new_row_of_a_grid_does_not_drag_in_all_the_others(self):
        """The fragility the threshold exists for.

        Adding a single elec node to nodedata must not turn every other elec
        demand node into a finding overnight.
        """
        nodedata = frame(*dheat("FI00", "SE03"), ("elec", "FI00_elec"))
        demanddata = frame(
            *dheat("FI00", "SE03"),
            ("elec", "FI00_elec"), ("elec", "SE03_elec"), ("elec", "PL00_elec"),
            ("elec", "AT00_elec"), ("elec", "DE00_elec"),
        )
        logger = FakeLogger()
        report_node_disagreements(nodedata, demanddata, logger)
        logger.assert_not_logged("elec")

    def test_a_grid_that_is_mostly_described_is_checked(self):
        nodedata = frame(("steam", "FI00_steam_industry"), ("steam", "SE03_steam_industry"),
                         ("steam", "PL00_steam_industry"))
        demanddata = frame(("steam", "FI00_steam_industry"), ("steam", "SE03_steam_industry"),
                           ("steam", "PL00_steam_industry"), ("steam", "XX00_steam_industry"))
        logger = FakeLogger()
        report_node_disagreements(nodedata, demanddata, logger)
        logger.assert_logged("XX00_steam_industry", level="warn")

    def test_a_one_off_industrial_node_present_in_both_is_fine(self):
        # The case that rules out every duplicate-based detector: one suffix,
        # many countries, and legitimately so.
        pairs = [("steam", f"{c}_steam_industry") for c in ("FI00", "SE03", "PL00")]
        nodes = frame(*dheat("FI00"), *pairs)
        logger = FakeLogger()
        report_node_disagreements(nodes, nodes.copy(), logger)
        logger.assert_clean()

    def test_a_grid_only_one_table_uses_is_left_alone(self):
        # Hydro nodes have nodedata rows and no demand rows at all.
        nodedata = frame(*dheat("FI00"), ("reservoir", "SE03_reservoir"),
                         ("reservoir", "NOS0_reservoir"))
        demanddata = frame(*dheat("FI00"))
        logger = FakeLogger()
        report_node_disagreements(nodedata, demanddata, logger)
        logger.assert_clean()


class TestItNeverGetsInTheWay:
    def test_an_empty_frame_is_not_a_disagreement(self):
        logger = FakeLogger()
        report_node_disagreements(pd.DataFrame(), frame(*dheat("FI00")), logger)
        report_node_disagreements(frame(*dheat("FI00")), pd.DataFrame(), logger)
        logger.assert_clean()

    def test_a_frame_without_the_columns_is_skipped(self):
        logger = FakeLogger()
        report_node_disagreements(
            pd.DataFrame({"country": ["FI00"]}), frame(*dheat("FI00")), logger
        )
        logger.assert_clean()

    def test_the_grid_match_folds_case(self):
        nodedata = frame(("dheat", "FI00_dheat"), ("dheat", "SE03_dheat"))
        demanddata = frame(("DHeat", "FI00_dheat"), ("DHeat", "SE03_dheat"))
        logger = FakeLogger()
        report_node_disagreements(nodedata, demanddata, logger)
        logger.assert_clean()
