"""A key column that cannot identify anything.

Two failures that produce no exception and no visible gap. A **blank** key means
the row is about nothing: it merges under a key of ``None`` and the table is
quietly one row short. A **year that is not a year** -- ``0`` is the one the
shipped workbooks actually contained -- matches no run and is filtered out beside
every row that legitimately belongs to a different year, which is why the
filtering itself can never report it.

Both are errors rather than warnings: the build writes its output either way, and
the difference is whether it is marked as having produced complete data.

Row numbers are asserted because they are the point. Three rows identical in
every other column are distinguishable only by where they sit, and 'check the
workbook' is not an instruction for a sheet of six hundred rows.
"""

import pandas as pd
import pytest

from src.source_data.source_data_loader import report_unusable_keys
from tests._common.fixtures import FakeLogger


def frame(**columns):
    return pd.DataFrame({name: pd.Series(values, dtype="object")
                         for name, values in columns.items()})


class TestBlankKeys:
    def test_a_blank_required_column_is_an_error(self):
        logger = FakeLogger()
        report_unusable_keys(
            frame(Country=["FI00", None], unittype=["CHPbio", "CHPbio"]),
            "unitdata", "d.xlsx:unitdata", logger,
        )
        logger.assert_logged("no 'Country'", level="error")

    def test_it_names_the_spreadsheet_row(self):
        """Frame row 1 is sheet row 3: the header is row 1 and pandas is 0-based."""
        logger = FakeLogger()
        report_unusable_keys(
            frame(Country=["FI00", None], unittype=["CHPbio", "CHPbio"]),
            "unitdata", "d.xlsx:unitdata", logger,
        )
        logger.assert_logged("row 3", level="error")

    def test_an_optional_column_may_be_blank(self):
        # unit_name_prefix is blank on most rows by design; reporting it would
        # fire on every correct sheet, which is the one thing a check must not do.
        logger = FakeLogger()
        report_unusable_keys(
            frame(Country=["FI00"], unittype=["CHPbio"], unit_name_prefix=[None]),
            "unitdata", "d.xlsx:unitdata", logger,
        )
        assert not logger.errors

    def test_a_column_the_sheet_does_not_have_is_not_missing(self):
        """A sheet without the column at all is a different failure.

        Whichever builder needs it reports that; here there is nothing to check.
        """
        logger = FakeLogger()
        report_unusable_keys(frame(unittype=["CHPbio"]), "unitdata",
                             "d.xlsx:unitdata", logger)
        assert not logger.errors

    def test_headers_are_matched_however_they_are_spelled(self):
        # This runs before normalize_dataframe lower-cases the headers, so the
        # column is whatever the author typed.
        logger = FakeLogger()
        report_unusable_keys(frame(COUNTRY=[None], UnitType=["CHPbio"]),
                             "unitdata", "d.xlsx:unitdata", logger)
        logger.assert_logged("no 'COUNTRY'", level="error")

    def test_an_empty_frame_says_nothing(self):
        logger = FakeLogger()
        report_unusable_keys(pd.DataFrame(), "unitdata", "d.xlsx:unitdata", logger)
        assert not logger.errors


class TestImpossibleYears:
    @pytest.mark.parametrize("year", [0, -1, 12, 3000])
    def test_a_year_that_is_not_a_year_is_an_error(self, year):
        logger = FakeLogger()
        report_unusable_keys(
            frame(Country=["FI00"], unittype=["CHPbio"], Year=[year]),
            "unitdata", "d.xlsx:unitdata", logger,
        )
        logger.assert_logged("neither a year nor", level="error")

    @pytest.mark.parametrize("year", [1, 2015, 2030, 2040])
    def test_a_real_year_and_the_wildcard_pass(self, year):
        """``1`` means every year -- see docs/source-workbook-conventions.md."""
        logger = FakeLogger()
        report_unusable_keys(
            frame(Country=["FI00"], unittype=["CHPbio"], Year=[year]),
            "unitdata", "d.xlsx:unitdata", logger,
        )
        assert not logger.errors

    def test_a_blank_year_is_left_to_the_whitelist(self):
        # apply_whitelist already counts a missing scenario or year. Saying it
        # twice would make the reader look for two problems.
        logger = FakeLogger()
        report_unusable_keys(
            frame(Country=["FI00"], unittype=["CHPbio"], Year=[None]),
            "unitdata", "d.xlsx:unitdata", logger,
        )
        assert not logger.errors

    def test_the_offending_row_is_named(self):
        logger = FakeLogger()
        report_unusable_keys(
            frame(Country=["FI00", "AT00"], unittype=["CHPbio", "coalCHP"],
                  Year=[2030, 0]),
            "unitdata", "d.xlsx:unitdata", logger,
        )
        logger.assert_logged("row 3", level="error")
        logger.assert_logged("Country=AT00", level="error")
