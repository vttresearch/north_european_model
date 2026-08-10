"""The bb_excel stage does not emit NaN.

Past this boundary the convention is ``0 = NA = None = "not set"`` (CLAUDE.md).
NaN inside a function is fine; NaN handed *out* of one is what makes the next
consumer guess.

Three sheets used to breach it, and all three for the same mechanical reason
rather than a missing call: they already ran
``fill_numeric_na(standardize_df_dtypes(df))``, but that pairing is **blind to an
entirely empty column**. ``standardize_df_dtypes`` types an all-NA column
``object`` (the all-NA-is-object rule) and ``fill_numeric_na`` fills only
``Float64``, so the one column that most needs filling is the one neither of them
can see. ``fill_all_na`` is no answer either: by then the dtype signal is gone,
so it would write ``''`` into every offender, never 0.

The repairs therefore had to come from what each column *is*, not what it is
typed as:

- ``p_gn`` / ``p_gnn`` -- an all-empty **parameter** column is dropped
  (``utils.drop_empty_parameter_columns``), keeping one so the ``Cdim=1`` column
  dimension survives. See ``test_empty_parameter_columns.py``.
- ``p_userconstraint`` -- an unused **dimension** slot is written as ``'-'``.
  Not a formatting preference: ``inc/1e_inputs.gms`` aborts the run on anything
  else, so this was a latent build failure rather than an untidy workbook.
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


class TestTheSheetsThatUsedToEmitNa:
    def test_p_gn_is_clean(self, built):
        """usePrice was left NaN for nodes where it was never decided.

        nodeBalance and usePrice are mutually exclusive, so "not a price node"
        is a real answer. With no price node anywhere the column carries no
        answer at all, and is dropped rather than written as a blank.
        """
        offenders = {f: _na_columns(built[f]).get("p_gn") for f in FIXTURES}
        assert not any(offenders.values()), offenders

    def test_p_gnn_is_clean(self, built):
        """Seven columns, all optional transfer parameters.

        rampLimit, diffCoeff, diffLosses, transferCapInvLimit, investMIP,
        invCost and annuityFactor are absent whenever the source row omits them
        -- the normal case for a plain transfer link -- so the column goes too.
        """
        assert not _na_columns(built["transfer"]).get("p_gnn")

    def test_p_userconstraint_is_clean(self, built):
        assert not _na_columns(built["userconstraint"]).get("p_userconstraint")

    def test_an_unused_userconstraint_dimension_is_a_dash(self, built):
        """The one case where neither 0 nor '' would do.

        ``p_userconstraint`` is Rdim=6 in indexSheet.xlsx, so all four uc slots
        are label columns and none can be dropped. Backbone then checks, per
        parameter type, that the slots a parameter does not use hold exactly
        ``'-'`` -- ``inc/1e_inputs.gms`` carries 21 aborts saying so, e.g.
        "should be '-' for <param> multiplier: (grid, node, '-', '-')".

        The fixture's sheet declares only the 1st and 2nd dimension, which is an
        ordinary thing to write; the 3rd and 4th are created for it and used to
        come out blank. Production data types the dashes by hand today.
        """
        frame = built["userconstraint"]["p_userconstraint"]
        for column in ("1st dimension", "2nd dimension", "3rd dimension", "4th dimension"):
            values = set(frame[column])
            assert "" not in values and 0 not in values, f"{column} has a blank slot: {values}"
        # The columns the sheet never declared are the ones that were broken.
        assert set(frame["3rd dimension"]) == {"-"}
        assert set(frame["4th dimension"]) == {"-"}


class TestTheScanItself:
    def test_the_detector_notices_a_planted_na(self):
        # Negative control: every assertion above would pass vacuously if
        # _na_columns could not see a NaN.
        planted = {"p_gn": pd.DataFrame({"grid": ["elec"], "usePrice": [pd.NA]})}
        assert _na_columns(planted) == {"p_gn": ["usePrice"]}

    def test_and_passes_a_clean_frame(self):
        clean = {"p_gn": pd.DataFrame({"grid": ["elec"], "usePrice": [0]})}
        assert _na_columns(clean) == {}
