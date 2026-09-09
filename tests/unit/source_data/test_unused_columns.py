"""A column nothing reads, and the build saying so.

A source workbook may hold any column at all. The ones no stage recognises are
carried through the source stage and then dropped, so a mistyped header reaches
the model as silence: ``capacty`` sits in a sheet looking exactly like
``capacity``, and the unit built from that row simply has no capacity. Nothing
about a wrong name looks wrong, because a header is free text.

This is the check that ends that, and ``docs/identified-gaps.md`` called
measuring it the largest open question about the source workbooks. The answer is
a warning on every run rather than a number taken once, so the question cannot
go stale.

What it must not do
-------------------
Fire on correct data. A check that warns every run about something right is not
strict, it is broken, and the reader stops reading the ones that matter. Three
things exist to prevent that, and each has a test below: the author's ``##``
columns are dropped before the question is asked; a repeated header is skipped
because the duplicate-header check has already spoken about it; and
``TestTheRealWorkbooksStaySilent`` runs every sheet the four shipped configs
name and requires silence.
"""

import ast
import configparser

import pandas as pd
from openpyxl import Workbook

import src.source_data.source_data_loader as loader
import src.utils as utils
from tests._common.fixtures import FakeLogger

MESSAGE = "read by nothing"

CATEGORIES = ("unitdata", "unittypedata", "nodedata", "transferdata",
              "demanddata", "emissiondata", "userconstraintdata")


def _read(tmp_path, rows, *, sheet="unitdata"):
    wb = Workbook()
    ws = wb.active
    ws.title = sheet
    for row in rows:
        ws.append(row)
    wb.save(tmp_path / "book.xlsx")

    logger = FakeLogger()
    frames = loader.read_input_excels(tmp_path, ["book.xlsx"], sheet, logger)
    return (frames[0] if frames else pd.DataFrame()), logger


class TestAColumnNothingReadsIsReported:
    def test_it_warns(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", "unittype", "capacty"],
            ["FI", "windturbine", 1000],
        ])
        logger.assert_logged(MESSAGE, level="warn")

    def test_the_message_spells_the_header_the_way_it_was_typed(self, tmp_path):
        """The reader searches the workbook for what they wrote, not for a slug."""
        _, logger = _read(tmp_path, [
            ["country", "unittype", "MaxRampUpp"],
            ["FI", "windturbine", 5],
        ])
        assert any("MaxRampUpp" in message for message in logger.matching(MESSAGE))

    def test_the_message_offers_all_three_remedies(self, tmp_path):
        """Misspelled, working material, or a parameter this build cannot write.

        The check cannot tell the three apart, so it must not imply it can.
        """
        _, logger = _read(tmp_path, [
            ["country", "unittype", "capacty"],
            ["FI", "windturbine", 1000],
        ])
        message = logger.matching(MESSAGE)[0]
        assert "spelling" in message
        assert utils.IGNORE_MARKER in message
        assert "identified-gaps" in message

    def test_the_column_is_still_carried(self, tmp_path):
        """Reporting is the whole change; nothing is dropped on account of it."""
        df, _ = _read(tmp_path, [
            ["country", "unittype", "capacty"],
            ["FI", "windturbine", 1000],
        ])
        assert "capacty" in df.columns


class TestWhatMustNotBeReported:
    def test_a_recognised_parameter_is_quiet(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", "unittype", "capacity", "vomCosts"],
            ["FI", "windturbine", 1000, 2.5],
        ])
        logger.assert_not_logged(MESSAGE)

    def test_a_connection_suffix_is_quiet(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", "unittype", "capacity_output1", "grid_input2"],
            ["FI", "chp", 1000, "gas"],
        ])
        logger.assert_not_logged(MESSAGE)

    def test_an_emission_factor_on_a_node_is_quiet(self, tmp_path):
        """The family is open-ended by design: the suffix names the emission."""
        _, logger = _read(tmp_path, [
            ["country", "grid", "emission_CO2", "emission_somethingNew"],
            ["FI", "gas", 200, 1],
        ], sheet="nodedata")
        logger.assert_not_logged(MESSAGE)

    def test_a_column_the_author_marked_is_quiet(self, tmp_path):
        """The marker says "mine, not the model's", and is dropped beforehand."""
        _, logger = _read(tmp_path, [
            ["country", "unittype", "##", "##"],
            ["FI", "windturbine", "scratch", 12],
        ])
        logger.assert_not_logged(MESSAGE)

    def test_a_note_column_is_quiet(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", "unittype", "note"],
            ["FI", "windturbine", "from the 2019 study"],
        ])
        logger.assert_not_logged(MESSAGE)

    def test_a_repeated_header_is_left_to_the_duplicate_check(self, tmp_path):
        """One mistake must not earn two warnings, or people skim both."""
        _, logger = _read(tmp_path, [
            ["country", "unittype", "capacity", "capacity"],
            ["FI", "windturbine", 1000, 999],
        ])
        logger.assert_logged("Duplicate column header", level="warn")
        logger.assert_not_logged(MESSAGE)


class TestTheTableDecidesWhatAColumnMeans:
    def test_an_emission_factor_on_a_unit_sheet_is_reported(self, tmp_path):
        """What splitting the emission families per table buys.

        ``emission_CO2`` belongs on a node. On a unit sheet it reaches nothing,
        and a single global ``emission_`` rule would wave it through.
        """
        _, logger = _read(tmp_path, [
            ["country", "unittype", "emission_CO2"],
            ["FI", "gasturbine", 200],
        ])
        logger.assert_logged(MESSAGE, level="warn")

    def test_a_unit_group_on_a_unit_sheet_is_quiet(self, tmp_path):
        _, logger = _read(tmp_path, [
            ["country", "unittype", "emission_group1"],
            ["FI", "gasturbine", "ETS-CO2"],
        ])
        logger.assert_not_logged(MESSAGE)

    def test_a_country_column_where_nothing_filters_by_it_is_reported(self, tmp_path):
        """unittypedata is global, so a country column on it changes nothing.

        The author believes they made a country-specific unit type. They did
        not, and this is the only thing that would ever tell them.
        """
        _, logger = _read(tmp_path, [
            ["unittype", "country"],
            ["WindOn", "FI"],
        ], sheet="unittypedata")
        logger.assert_logged(MESSAGE, level="warn")


class TestTheRealWorkbooksStaySilent:
    def test_no_unused_column_reports_across_the_shipped_configs(self, src_files_dir):
        """Every sheet the four shipped configs name, and not one warning.

        Deliberately coupled to shipped data: this is what stops the hygiene
        decaying, and it is why the check can be trusted when it does speak.

        Scoped to the workbooks a config actually lists. Several files in the
        folder are named by no config, and their columns are genuinely unread --
        sweeping those too would assert that dormant files are clean, which is
        neither true nor anything this pass promises.
        """
        data_files = src_files_dir / "data_files"

        listed: dict[str, set[str]] = {category: set() for category in CATEGORIES}
        for config in sorted(src_files_dir.glob("config_*.ini")):
            parser = configparser.ConfigParser(inline_comment_prefixes=None)
            parser.read(config, encoding="utf-8")
            for category in CATEGORIES:
                raw = parser["inputdata"].get(f"{category}_files", "[]")
                listed[category].update(ast.literal_eval(raw))

        logger = FakeLogger()
        for category, files in listed.items():
            if files:
                loader.read_input_excels(data_files, sorted(files), category, logger)

        logger.assert_not_logged(MESSAGE)
