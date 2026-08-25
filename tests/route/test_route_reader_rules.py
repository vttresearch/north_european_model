"""The quiet rules the source reader applies before any pipeline logic runs.

Each one is invisible in the output unless you know to look for it, and each one
has silently eaten real data at some point. The old binary fixture could not
express most of them; this module is the argument for the text format.

Read through ``run_source`` rather than ``run_route``: these are properties of
the *reader*, and asserting them on the source DataFrames says what is meant
without depending on how a unit later reaches ``p_gnu_io``.
"""

import pandas as pd
import pytest

from tests._common.asserts import cell, rows_for
from tests._common.routes import run_route, run_source
from tests._common.workbook_text import load_workbook_fixture

pytestmark = pytest.mark.route

FIXTURE = load_workbook_fixture("reader_rules")


@pytest.fixture(scope="module")
def source(tmp_path_factory):
    pipeline, logger = run_source(
        tmp_path_factory.mktemp("reader_rules"), workbooks={"data.xlsx": FIXTURE}
    )
    return pipeline, logger


@pytest.fixture(scope="module")
def units(source):
    return source[0].df_unitdata


class TestRowsThatSurvive:
    def test_an_ordinary_row_is_read(self, units):
        assert cell(units, "capacity", generator_id="keeper", unit_name_prefix=None) == 100

    def test_scenario_all_and_year_1_pass_the_whitelist(self, units):
        # The magic values (apply_whitelist:762-771). Nearly every real row uses
        # them, so a regression here would empty most of the input data.
        assert not rows_for(units, generator_id="keeper", unit_name_prefix=None).empty

    def test_an_explicit_scenario_and_year_also_pass(self, units):
        # The run is scenario='test', year=2030; both the catch-all and the
        # explicit spelling must be admitted, or scenario overrides stop working.
        assert cell(units, "capacity", generator_id="keeper", unit_name_prefix="scen") == 144


class TestRowsThatAreDropped:
    def test_a_row_marked_with_the_ignore_marker_is_dropped(self, units):
        # Used in real workbooks to park a row without deleting it, so it must
        # not reach the model. Every other field of that row is valid, so it can
        # only be missing because the marker was honoured -- see the fixture.
        assert rows_for(units, generator_id="commented").empty

    def test_a_value_containing_an_underscore_drops_its_row(self, units):
        # drop_underscore_values:255. Underscore is the node-name separator, so
        # a value containing one would produce an ambiguous GAMS set element.
        assert rows_for(units, generator_id="underscored").empty

    def test_the_dropped_underscore_row_is_reported(self, source):
        # Dropping data silently is what this rule must never do.
        _, logger = source
        logger.assert_logged("Underscores detected", level="warn")

    def test_a_blank_row_truncates_everything_below_it(self, units):
        """This is how the previous fixture died: a blank row directly under
        every header truncated all seven sheets to zero data rows, and nothing
        said so."""
        assert rows_for(units, generator_id="truncated").empty

    def test_and_the_truncation_is_now_reported(self, source):
        # "nothing said so" was the whole problem. A blank row with real rows
        # still below it discards them, and the count is what tells a reader
        # whether it matters.
        _, logger = source
        logger.assert_logged("are not read", level="warn")


class TestColumnsThatAreDropped:
    def test_a_note_column_never_reaches_the_data(self, units):
        # read_input_excels:96-98. 'Note' is used for free text in every real
        # workbook and must not become a parameter.
        assert "note" not in {str(c).lower() for c in units.columns}

    def test_a_blank_header_column_never_reaches_the_data(self, units):
        # Excel leaves these behind constantly.
        assert not [c for c in units.columns if str(c).startswith("Unnamed")]
        assert not [c for c in units.columns if not str(c).strip()]

    def test_and_one_inside_the_table_is_reported(self, source):
        # The fixture's blank header sits between Year and capacity_output1, so
        # it is a hole in the table rather than the scratch area past its end.
        # Its values are discarded and the header cannot be recovered, which is
        # worth a word; the same column past the last named one would be silent.
        _, logger = source
        logger.assert_logged("without a header", level="warn")

    def test_the_output1_suffix_is_stripped_from_numeric_columns(self, units):
        # normalize_dataframe:234-250. bb_excel_pipeline restores it later
        # (_normalize_unitdata_columns), so the round trip has to line up.
        assert "capacity" in units.columns
        assert "capacity_output1" not in units.columns


class TestMethodHandling:
    def test_an_unknown_method_is_coerced_rather_than_dropping_the_row(self, units):
        # Losing a row because of a typo in an optional column would be a harsh
        # response; the row survives with the default method.
        assert cell(units, "capacity", generator_id="methodless") == 133

    def test_and_it_is_reported(self, source):
        _, logger = source
        logger.assert_logged("Unknown method", level="warn")


class TestOnlyTheIntendedWarnings:
    def test_the_fixture_raises_no_errors(self, source):
        # Every rule here is a routine data condition, not a failure.
        _, logger = source
        logger.assert_no_errors()

    def test_and_no_warnings_beyond_the_ones_under_test(self, source):
        """Guards the fixture as much as the code.

        A new incidental warning usually means the fixture drifted out of the
        shape it is meant to demonstrate, and it would otherwise go unnoticed
        behind the warnings this module deliberately provokes.
        """
        _, logger = source
        expected = (
            "Underscores", "Unknown method", "Dropped",
            "without a header", "are not read",
        )
        unexpected = [w for w in logger.warnings if not any(e in w for e in expected)]
        assert not unexpected, unexpected


class TestFormalityIsJudgedBeforeRelevance:
    """A row that was never going to be used still has its faults reported.

    The whitelist drops rows for a scenario, year or country this run does not
    cover. If it ran before the checks on a row's *form*, a genuinely broken cell
    could disappear as merely irrelevant and never be mentioned -- and the author
    would find out only when they ran the scenario that does use it.

    This is not hypothetical. A test in this very file asserted that a
    ``#``-prefixed row was treated as a comment, and went on passing after the
    marker moved to ``##`` -- because the row's country was ``#FI``, which the
    whitelist rejected anyway. Nothing in that test could tell the two reasons
    apart, so it would have stayed green indefinitely; it only came to light
    because a neighbouring assertion about warnings failed.

    The pipeline reads in the right order today: read and gate, normalize, drop
    underscores, and only then whitelist. This pins that, so a reordering has to
    fail a test rather than quietly start swallowing reports.
    """

    FIXTURE = """
[unittypedata]
scenario | year | Generator_ID | unittype | grid_output1 | eff00 | isSource
all      | 1    | wanted       | Wanted   | elec         | 1     | 1

[nodedata]
Country | Grid | Scenario | Year | nodeBalance
FI      | elec | all      | 1    | 1

// ZZ is not in country_codes, so the whitelist removes this row. Its capacity
// is malformed all the same, and that is what has to be reported.
[unitdata]
Country | Generator_ID | Scenario | Year | capacity_output1 | method
FI      | wanted       | all      | 1    | 100              | replace
ZZ      | wanted       | all      | 1    | 1,000.0          | replace
"""

    def test_a_malformed_cell_is_reported_even_when_the_row_is_filtered_out(self, tmp_path):
        _, logger = run_source(tmp_path, workbooks={"data.xlsx": self.FIXTURE})
        logger.assert_logged("1,000.0", level="error")

    def test_and_the_row_is_still_gone(self, tmp_path):
        # Reporting it does not resurrect it: ZZ is not in this run.
        pipeline, _ = run_source(tmp_path, workbooks={"data.xlsx": self.FIXTURE})
        assert "ZZ" not in set(pipeline.df_unitdata["country"].dropna())


class TestTheRulesSurviveTheWholeRoute:
    def test_dropped_rows_do_not_reappear_in_the_workbook(self, tmp_path):
        """The rules are applied once, early -- but the assertion that matters
        is that nothing downstream resurrects what they removed."""
        route = run_route(tmp_path, workbooks={"data.xlsx": FIXTURE})
        route.logger.assert_no_errors()

        units = set(route.sheets["unit"]["unit"])
        for dropped in ("Commented", "Undersc", "Truncated"):
            assert not [u for u in units if dropped.lower() in str(u).lower()], (
                f"{dropped} was dropped by the reader but appears in the workbook"
            )

        # ...and the rows that should have survived did.
        assert [u for u in units if "keeper" in str(u).lower()]
