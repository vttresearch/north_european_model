"""A userconstraintdata sheet that is not shaped like the others.

``SourceDataPipeline.run()`` drops ``scenario``, ``year`` and ``country`` from
every userconstraint frame before merging, because its key makes them
redundant.  It is the only category that does so, and the drop used to be
unconditional -- so a sheet missing any one of the three raised ``KeyError``
straight out of the pipeline phase, aborting a build that every other category
would have survived with a warning.

Two shapes reach that line with fewer columns than it expects, and both are
things a workbook author does rather than things that require a bug:

1. a sheet that simply has no ``country`` column, because the constraint is not
   about one country;
2. a sheet whose every data row is ``##``, which ``read_input_excels`` empties
   and ``normalize_dataframe`` then returns as a bare ``DataFrame`` -- no rows
   and, importantly, **no columns at all**.

The error policy in ``CLAUDE.md`` is that nothing raises once the logger
exists.  These tests are the behavioural statement of that for this category.
"""

import pytest

from tests._common.routes import run_source


# No `country` column anywhere. The constraint is a system-wide limit, so there
# is nothing sensible to put in one.
NO_COUNTRY = """
[nodedata]
Country | Grid | Scenario | Year | nodeBalance
FI      | elec | all      | 1    | 1

[userconstraintdata]
Scenario | year | group     | 1st dimension | 2nd dimension | parameter   | value
all      | 1    | elecLimit | FI_elec       | v_state       | coefficient | 1
all      | 1    | elecLimit |               |               | constant    | 250
"""

# Every data row marked '##'. The sheet is legal input that says "nothing here
# yet"; the frame that reaches the drop has no columns.
ALL_ROWS_IGNORED = """
[nodedata]
Country | Grid | Scenario | Year | nodeBalance
FI      | elec | all      | 1    | 1

[userconstraintdata]
Scenario | year | country | group     | 1st dimension | parameter   | value
## all   | 1    | FI      | elecLimit | FI_elec       | coefficient | 1
"""


class TestAUserConstraintSheetNeverStopsTheBuild:
    def test_a_sheet_without_a_country_column_is_read(self, tmp_path):
        pipeline, logger = run_source(
            tmp_path, workbooks={"data.xlsx": NO_COUNTRY}
        )

        logger.assert_no_errors()
        # The rows survived: dropping the column must not drop the constraint.
        assert not pipeline.df_userconstraintdata.empty

    def test_a_sheet_whose_rows_are_all_ignored_is_read(self, tmp_path):
        pipeline, logger = run_source(
            tmp_path, workbooks={"data.xlsx": ALL_ROWS_IGNORED}
        )

        logger.assert_no_errors()
        assert pipeline.df_userconstraintdata.empty

    @pytest.mark.parametrize(
        "workbook", [NO_COUNTRY, ALL_ROWS_IGNORED], ids=["no_country", "all_ignored"]
    )
    def test_the_rest_of_the_phase_still_runs(self, workbook, tmp_path):
        """The failure mode was an abort, so what matters is what came after.

        ``nodedata`` is built before userconstraintdata and ``df_boundarydata``
        after it, so a frame on each side proves the phase ran to the end
        rather than merely surviving one statement.
        """
        pipeline, logger = run_source(tmp_path, workbooks={"data.xlsx": workbook})

        logger.assert_no_errors()
        assert not pipeline.df_nodedata.empty
        assert pipeline.df_boundarydata is not None
