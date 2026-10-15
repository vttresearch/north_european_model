"""A named build reaches ``inputData.xlsx``: the add_scen_tags row.

Expectations: ``tests/expectations/variants.md``, G4 and G5.

Spine Toolbox workflows read this sheet by cell position: the importer in
``.spinetoolbox/specifications/Importer/import_from_gdx_ne.json`` takes cell (0, 0)
as its alternative name, and users' own workflows outside this repo read it too.
The column order is therefore the contract and is pinned (tests/README.md, pinning
case 4); scenario and year keep the first two cells.
"""

import pandas as pd
import pytest

from tests._common.routes import config_for_workbooks, run_route
from tests._common.workbook_text import load_workbook_fixture

VARIANTS = {"": {}, "sens1": {}}


@pytest.fixture(scope="module")
def minimal():
    return load_workbook_fixture("minimal")


def _tags(tmp_path, minimal, build):
    workbooks = {"minimal.xlsx": minimal}
    config = config_for_workbooks(workbooks, variants=VARIANTS)
    result = run_route(tmp_path, workbooks=workbooks, config=config, variant=build)
    result.logger.assert_no_errors()
    return result.sheets["add_scen_tags"]


def _cell(value) -> str:
    """A cell as text; a year read back as 2030.0 is still '2030'."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def test_the_four_cells_in_order(tmp_path, minimal):
    tags = _tags(tmp_path, minimal, "sens1")
    assert list(tags.columns)[:4] == ["scenario", "year", "variant", "name"]
    assert len(tags) == 1


def test_a_named_build(tmp_path, minimal):
    # G4, G5
    row = _tags(tmp_path, minimal, "sens1").iloc[0]
    assert _cell(row["scenario"]) == "test"
    assert _cell(row["year"]) == "2030"
    assert _cell(row["variant"]) == "sens1"
    assert _cell(row["name"]) == "test_2030_sens1"


def test_the_unsuffixed_build(tmp_path, minimal):
    # G4: the variant cell is empty, and name is today's folder name minus its prefix.
    row = _tags(tmp_path, minimal, "").iloc[0]
    assert _cell(row["variant"]) == ""
    assert _cell(row["name"]) == "test_2030"


def test_no_alternative_columns_remain(tmp_path, minimal):
    tags = _tags(tmp_path, minimal, "sens1")
    assert not [c for c in tags.columns if str(c).startswith("alternative")]
