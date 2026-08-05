"""Tests for the workbook delta helpers.

Heavy on negative controls. A delta that never reports a difference, or an
``assert_delta`` that accepts anything, would make every route delta test green
and meaningless -- and unlike a broken assertion in one test, it would do so
silently across all of them.
"""

import pandas as pd
import pytest

from tests._common.delta import (
    CellChange,
    Delta,
    RowChange,
    assert_delta,
    normalise,
    workbook_delta,
)

KEYS = {"p_gnu_io": ("grid", "node", "unit")}


def _workbook(capacity=100, extra_rows=(), **columns):
    rows = [{"grid": "elec", "node": "FI_elec", "unit": "u1", "capacity": capacity, **columns}]
    rows.extend(extra_rows)
    return {"p_gnu_io": pd.DataFrame(rows)}


class TestNormalise:
    @pytest.mark.parametrize(
        "a, b",
        [
            (500, 500.0),
            ("500", 500),
            (500.0, "500.0"),
            ("FI", "fi"),
            (" FI ", "FI"),
            (None, float("nan")),
            (None, pd.NA),
        ],
    )
    def test_values_that_must_compare_equal(self, a, b):
        # Excel round-trips numbers inconsistently; without this the delta would
        # fire on formatting and be as brittle as the goldens it replaces.
        assert normalise(a) == normalise(b)

    @pytest.mark.parametrize("a, b", [(500, 501), ("FI", "SE"), (0, None), (0.1, 0.2)])
    def test_values_that_must_compare_different(self, a, b):
        # In particular 0 vs blank: at the Excel boundary they mean the same to
        # GAMS, but the delta must still be able to see the edit.
        assert normalise(a) != normalise(b)


class TestWorkbookDelta:
    def test_identical_workbooks_produce_no_delta(self):
        assert workbook_delta(_workbook(), _workbook(), keys=KEYS).is_empty()

    def test_a_changed_cell_is_located_by_key_and_column(self):
        delta = workbook_delta(_workbook(100), _workbook(750), keys=KEYS)
        assert delta.changed == frozenset(
            {CellChange("p_gnu_io", ("elec", "fi_elec", "u1"), "capacity", "100", "750")}
        )

    def test_a_new_row_is_an_addition_not_a_change(self):
        after = _workbook(
            extra_rows=[{"grid": "heat", "node": "FI_heat", "unit": "u1", "capacity": 5}]
        )
        delta = workbook_delta(_workbook(), after, keys=KEYS)
        assert delta.added_rows == frozenset({RowChange("p_gnu_io", ("heat", "fi_heat", "u1"))})
        assert not delta.changed

    def test_a_missing_row_is_a_removal(self):
        before = _workbook(
            extra_rows=[{"grid": "heat", "node": "FI_heat", "unit": "u1", "capacity": 5}]
        )
        delta = workbook_delta(before, _workbook(), keys=KEYS)
        assert delta.removed_rows == frozenset({RowChange("p_gnu_io", ("heat", "fi_heat", "u1"))})

    def test_a_new_column_is_reported(self):
        # is_col_empty drops all-zero columns, so changing 0 to 500 can make a
        # whole column materialise. That is real behaviour, not noise.
        delta = workbook_delta(_workbook(), _workbook(vomCosts=3), keys=KEYS)
        assert delta.added_columns == frozenset({("p_gnu_io", "vomCosts")})

    def test_metadata_sheets_are_ignored_by_default(self):
        # add_scen_tags and index carry run metadata, not model data.
        before = {"index": pd.DataFrame([{"Symbol": "p_gn"}])}
        after = {"index": pd.DataFrame([{"Symbol": "p_gnu_io"}])}
        assert workbook_delta(before, after, keys={}).is_empty()

    def test_a_keyless_sheet_diffs_as_whole_rows(self):
        before = {"restype": pd.DataFrame([{"restype": "a"}])}
        after = {"restype": pd.DataFrame([{"restype": "b"}])}
        delta = workbook_delta(before, after, keys={})
        assert delta.added_rows and delta.removed_rows
        assert not delta.changed

    def test_describe_names_the_cell_that_moved(self):
        delta = workbook_delta(_workbook(100), _workbook(750), keys=KEYS)
        described = delta.describe()
        assert "capacity" in described and "100" in described and "750" in described


class TestAssertDelta:
    def _delta(self):
        return workbook_delta(_workbook(100), _workbook(750), keys=KEYS)

    def test_accepts_a_listed_change_without_a_value(self):
        assert_delta(self._delta(), changed=[("p_gnu_io", ("elec", "FI_elec", "u1"), "capacity")])

    def test_accepts_a_listed_change_with_its_value(self):
        assert_delta(
            self._delta(), changed=[("p_gnu_io", ("elec", "FI_elec", "u1"), "capacity", 750)]
        )

    def test_rejects_a_wrong_expected_value(self):
        with pytest.raises(AssertionError, match="expected '999', got '750'"):
            assert_delta(
                self._delta(),
                changed=[("p_gnu_io", ("elec", "FI_elec", "u1"), "capacity", 999)],
            )

    def test_rejects_an_unlisted_change(self):
        """THE negative control: omission is the assertion.

        If an unlisted cell could move without failing, every delta test would
        be asserting only what it happens to mention.
        """
        with pytest.raises(AssertionError, match="unexpected cell change"):
            assert_delta(self._delta(), expect_no_change=True)

    def test_rejects_a_listed_change_that_did_not_happen(self):
        no_change = workbook_delta(_workbook(), _workbook(), keys=KEYS)
        with pytest.raises(AssertionError, match="to change, but it did not"):
            assert_delta(
                no_change, changed=[("p_gnu_io", ("elec", "FI_elec", "u1"), "capacity")]
            )

    def test_accepts_a_genuinely_unchanged_workbook(self):
        assert_delta(workbook_delta(_workbook(), _workbook(), keys=KEYS), expect_no_change=True)

    def test_refuses_to_assert_nothing(self):
        # Without this, a variant that failed to differ would pass silently --
        # the single way a delta test can be green for the wrong reason.
        with pytest.raises(ValueError, match="no expected changes"):
            assert_delta(self._delta())

    def test_refuses_contradictory_arguments(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            assert_delta(
                self._delta(),
                changed=[("p_gnu_io", ("elec", "FI_elec", "u1"), "capacity")],
                expect_no_change=True,
            )

    def test_rejects_an_unlisted_new_column(self):
        delta = workbook_delta(_workbook(), _workbook(vomCosts=3), keys=KEYS)
        with pytest.raises(AssertionError, match="unexpected added column"):
            assert_delta(delta, expect_no_change=True)

    def test_accepts_a_listed_new_column(self):
        delta = workbook_delta(_workbook(), _workbook(vomCosts=3), keys=KEYS)
        assert_delta(delta, added_columns=[("p_gnu_io", "vomCosts")])

    def test_the_failure_message_shows_the_full_delta(self):
        with pytest.raises(AssertionError, match="full delta"):
            assert_delta(self._delta(), expect_no_change=True)

    def test_a_malformed_spec_is_rejected(self):
        with pytest.raises(ValueError, match="changed-cell spec"):
            assert_delta(self._delta(), changed=[("p_gnu_io", ("elec",))])
