"""The bb_excel stage should not emit NaN.

Past this boundary the convention is ``0 = NA = None = "not set"`` (CLAUDE.md),
and ``utils.fill_all_na`` exists precisely to cross it: numeric columns to 0,
everything else to the empty string. NaN inside a function is fine; NaN handed
*out* of one is what makes the next consumer guess.

Several constructing functions still emit it. That is recorded here as a strict
xfail rather than repaired, because the repair is visible in the written
workbook -- an empty cell becomes a literal 0 -- and whether that is an
improvement is a judgment call:

- GAMS reads an empty cell and a 0 identically, so nothing changes for the model;
- empty cells keep inputData.xlsx smaller and easier to read by eye;
- but a frame carrying NaN between functions is exactly the dtype hazard the
  source stage was cleaned up to avoid, and *some* of these columns are text
  (user constraint dimensions), where 0 would be wrong and '' is right.

``fill_all_na`` already draws that distinction correctly, so applying it is a
small change -- the decision is whether the workbook should look different.
"""

import pathlib
import tempfile

import pandas as pd
import pytest

from tests._common.routes import run_route
from tests._common.workbook_text import load_workbook_fixture

FIXTURES = ("minimal", "chp", "transfer", "userconstraint", "reader_rules")

#: Metadata rather than model data.
IGNORED_SHEETS = {"index"}


def _na_columns(sheets) -> dict[str, list[str]]:
    offenders = {}
    for name, frame in sheets.items():
        if name in IGNORED_SHEETS or frame.empty:
            continue
        columns = [c for c in frame.columns if frame[c].isna().any()]
        if columns:
            offenders[name] = columns
    return offenders


@pytest.fixture(scope="module")
def built():
    return {
        name: run_route(
            pathlib.Path(tempfile.mkdtemp()),
            workbooks={"data.xlsx": load_workbook_fixture(name)},
        ).sheets
        for name in FIXTURES
    }


class TestSheetsThatAreAlreadyClean:
    def test_a_chp_workbook_emits_no_na_at_all(self, built):
        # Proof the convention is reachable: this fixture exercises units,
        # nodes, demands and multiple grids without leaving a single NaN.
        assert _na_columns(built["chp"]) == {}

    @pytest.mark.parametrize("fixture", FIXTURES)
    def test_the_unit_and_domain_sheets_are_clean(self, built, fixture):
        sheets = built[fixture]
        for name in ("p_gnu_io", "p_unit", "unit", "node", "grid", "unittype"):
            frame = sheets.get(name)
            if frame is None or frame.empty:
                continue
            offenders = [c for c in frame.columns if frame[c].isna().any()]
            assert not offenders, f"{fixture}/{name} emits NaN in {offenders}"


class TestSheetsThatStillEmitNa:
    @pytest.mark.xfail(
        strict=True,
        reason="create_p_gn emits NaN in usePrice instead of 0",
    )
    def test_p_gn_is_clean(self, built):
        """usePrice is left NaN for nodes where it was never decided.

        nodeBalance and usePrice are mutually exclusive, so "not a price node"
        is a real answer and 0 states it. NaN leaves the next reader to infer it.
        """
        offenders = {f: _na_columns(built[f]).get("p_gn") for f in FIXTURES}
        assert not any(offenders.values()), offenders

    @pytest.mark.xfail(
        strict=True,
        reason="create_p_gnn emits NaN in every optional transfer parameter",
    )
    def test_p_gnn_is_clean(self, built):
        """Seven columns, all optional transfer parameters.

        rampLimit, diffCoeff, diffLosses, transferCapInvLimit, investMIP,
        invCost and annuityFactor are NaN whenever the source row omits them --
        which is the normal case for a plain transfer link.
        """
        assert not _na_columns(built["transfer"]).get("p_gnn")

    @pytest.mark.xfail(
        strict=True,
        reason="create_p_userconstraint emits NaN in unused dimension columns",
    )
    def test_p_userconstraint_is_clean(self, built):
        """The one case where 0 would be wrong.

        These are set-element slots, not measures, so an unused dimension should
        be the empty string. fill_all_na already makes that distinction --
        numeric to 0, everything else to '' -- which is why it is the right tool
        here rather than a blanket fillna(0).
        """
        assert not _na_columns(built["userconstraint"]).get("p_userconstraint")


class TestTheScanItself:
    def test_the_detector_notices_a_planted_na(self):
        # Negative control: every xfail above would pass vacuously if _na_columns
        # could not see a NaN.
        planted = {"p_gn": pd.DataFrame({"grid": ["elec"], "usePrice": [pd.NA]})}
        assert _na_columns(planted) == {"p_gn": ["usePrice"]}

    def test_and_passes_a_clean_frame(self):
        clean = {"p_gn": pd.DataFrame({"grid": ["elec"], "usePrice": [0]})}
        assert _na_columns(clean) == {}
