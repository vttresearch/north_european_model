"""A unit dropped for having nothing to do is in no sheet at all.

``drop_redundant_units`` removes a unit with no capacity and no investment
parameters from ``p_gnu_io`` and ``p_unit``. The ``unit``, ``unittype`` and
``unitUnittype`` sheets have to follow: a unit still listed there reaches
Backbone with no data, which defaults its efficiency and conversionCoeff to 1,
deactivates it, and writes three notes about it to warnings.log.
"""

import pytest

from tests._common.asserts import assert_workbook_consistent
from tests._common.routes import run_route

pytestmark = pytest.mark.route

WORKBOOK = """
[unittypedata]
unittype  | grid_output1 | eff00 | isSource
WindOnFI  | elec         | 1     | 1
RetiredFI | elec         | 1     | 1

[unitdata]
Country | unittype  | Scenario | Year | capacity_output1
FI      | WindOnFI  | all      | 1    | 100
FI      | RetiredFI | all      | 1    | 0

[nodedata]
Country | Grid | Scenario | Year | nodeBalance
FI      | elec | all      | 1    | 1

[demanddata]
Country | Grid | Scenario | Year | TWh/year
FI      | elec | all      | 1    | 5
"""


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    return run_route(tmp_path_factory.mktemp("dropped_units"), workbooks={"data.xlsx": WORKBOOK})


def test_the_unit_is_dropped(result):
    result.logger.assert_logged("Dropped 1 unit(s)", level="skip")
    assert result.sheets["p_unit"]["unit"].tolist() == ["FI_WindOnFI"]


def test_and_is_in_no_unit_sheet(result):
    assert result.sheets["unit"]["unit"].tolist() == ["FI_WindOnFI"]
    assert result.sheets["unitUnittype"]["unit"].tolist() == ["FI_WindOnFI"]


def test_nor_is_its_unittype(result):
    assert result.sheets["unittype"]["unittype"].tolist() == ["WindOnFI"]


def test_and_the_workbook_is_consistent(result):
    assert_workbook_consistent(result.sheets)
