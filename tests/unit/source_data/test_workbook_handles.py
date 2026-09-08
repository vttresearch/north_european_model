"""A workbook is closed as soon as it has been read.

Windows keeps a file locked while any handle to it is open, and
``read_input_excels`` opens one per workbook. Left to garbage collection, a build
held all of them until the process exited, so every source workbook was read-only
in Excel for the length of the run -- which is exactly when someone is most likely
to have one open, editing the thing the build is complaining about.

Asserted by counting ``close()`` rather than by trying to write the file, so the
test means the same on a machine where an open handle would not have stopped it.
"""

import pandas as pd
import pytest

from src.source_data.source_data_loader import read_input_excels
from tests._common.fixtures import FakeLogger
from tests._common.workbook_text import write_workbook_text

SHEET = "[unitdata]\nCountry | unittype | Scenario | Year | capacity_output1\nFI00 | WindOn | all | 1 | 100\n"


@pytest.fixture
def counting_excel_file(monkeypatch):
    """``pd.ExcelFile`` that records how often it is opened and closed."""
    tally = {"opened": 0, "closed": 0}
    real = pd.ExcelFile

    class Counting(real):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            tally["opened"] += 1

        def close(self):
            tally["closed"] += 1
            super().close()

    monkeypatch.setattr(pd, "ExcelFile", Counting)
    return tally


def test_every_workbook_opened_is_closed(tmp_path, counting_excel_file):
    folder = tmp_path / "data_files"
    for name in ("a.xlsx", "b.xlsx"):
        write_workbook_text(SHEET, folder / name)

    read_input_excels(folder, ["a.xlsx", "b.xlsx"], "unitdata", FakeLogger())

    assert counting_excel_file["opened"] == 2
    assert counting_excel_file["closed"] == 2


def test_a_workbook_with_no_matching_sheet_is_closed_too(tmp_path, counting_excel_file):
    """The early `continue` is its own way out of the loop, and it leaked."""
    folder = tmp_path / "data_files"
    write_workbook_text("[nodedata]\nCountry | Grid\nFI00 | elec\n", folder / "a.xlsx")

    read_input_excels(folder, ["a.xlsx"], "unitdata", FakeLogger())

    assert counting_excel_file["opened"] == 1
    assert counting_excel_file["closed"] == 1
