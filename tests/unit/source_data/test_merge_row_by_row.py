"""``merge_row_by_row`` -- the overlay engine, and the densest logic in the repo.

Historically the second largest source of bugs here, and the current
implementation is the Nth generation. Two things make it hard, and both get
their own section below.

**Frame boundaries must not matter.** A second row overriding a first must
behave identically whether the two rows came from the same Excel sheet or from
two different workbooks. Users reason about "the later row wins", not about file
boundaries. The core loop flattens rows across frames to achieve this, and
:class:`TestFrameBoundariesDoNotMatter` pins it as a property rather than as
examples -- it is the invariant most likely to be broken by a well-meant
refactor that starts grouping work per frame.

**Overwriting versus accumulating.** The point of the method column is to offer
both a blunt full-row operation (``replace`` overwrites everything, blanks
included) and a surgical one (``replace-partial`` touches only the cells the
overriding row actually filled in). The tension between those two is where the
subtle bugs live.

``add-non-negative`` exists because of real compounding in practice: one
scenario reduces coal capacity because a CHP overlay added the same plant back,
another reduces the starting electricity capacity for a climate policy, and the
third overlay tips the total below zero. Clamping keeps a stack of independent,
individually-reasonable edits from producing a negative capacity.

The truth table in the docstring at ``source_data_loader.py:884-909`` is the
specification, so exact values are pinned here -- pinning case 3.
"""

import itertools

import pandas as pd
import pytest

from src.source_data.source_data_loader import merge_row_by_row, normalize_dataframe
from tests._common.contracts import assert_normalized
from tests._common.fixtures import FakeLogger

KEY = ["country", "grid"]


def _frame(*rows: dict) -> pd.DataFrame:
    """Build a frame the way the pipeline would hand it over.

    ``merge_row_by_row`` documents a precondition of normalized input
    (:875), so the real normalizer is used rather than a hand-built frame that
    might not match what actually arrives.
    """
    return normalize_dataframe(pd.DataFrame(list(rows)), "test", FakeLogger())


def _row(method="replace", country="FI", grid="elec", **values) -> dict:
    return {"country": country, "grid": grid, "method": method, **values}


def _merge(*frames, key=None, **kwargs):
    return merge_row_by_row(list(frames), FakeLogger(), key_columns=key or KEY, **kwargs)


def _value(df, column="capacity", country="FI", grid="elec"):
    match = df[(df["country"] == country) & (df["grid"] == grid)]
    assert len(match) == 1, f"expected one row for {country}/{grid}, got {len(match)}"
    return match.iloc[0][column]


# ---------------------------------------------------------------------------
# The documented truth table
# ---------------------------------------------------------------------------


class TestAddMissingValueRules:
    """``source_data_loader.py:890-897``, quoted verbatim in the docstring."""

    @pytest.mark.parametrize(
        "previous, incoming, expected",
        [
            pytest.param(None, None, None, id="missing+missing=NA"),
            pytest.param(None, 0.0, 0.0, id="missing+0=0"),
            pytest.param(0.0, None, 0.0, id="0+missing=0"),
            pytest.param(2.0, 3.0, 5.0, id="2+3=5"),
            pytest.param(2.0, None, 2.0, id="2+missing=2"),
            pytest.param(None, 3.0, 3.0, id="missing+3=3"),
        ],
    )
    def test_add(self, previous, incoming, expected):
        merged = _merge(
            _frame(_row("replace", capacity=previous), _row("add", capacity=incoming))
        )
        got = _value(merged)
        if expected is None:
            assert pd.isna(got)
        else:
            assert got == pytest.approx(expected)


class TestMultiplyMissingValueRules:
    """``source_data_loader.py:901-907``.

    The asymmetry is deliberate and easy to get backwards: a missing *previous*
    value zeroes the product, a missing *current* one leaves it alone.
    """

    @pytest.mark.parametrize(
        "previous, incoming, expected",
        [
            pytest.param(None, None, None, id="missing*missing=NA"),
            pytest.param(None, 3.0, 0.0, id="missing*3=0"),
            pytest.param(2.0, None, 2.0, id="2*missing=2"),
            pytest.param(2.0, 3.0, 6.0, id="2*3=6"),
            pytest.param(2.0, 0.0, 0.0, id="2*0=0"),
        ],
    )
    def test_multiply(self, previous, incoming, expected):
        merged = _merge(
            _frame(_row("replace", capacity=previous), _row("multiply", capacity=incoming))
        )
        got = _value(merged)
        if expected is None:
            assert pd.isna(got)
        else:
            assert got == pytest.approx(expected)


class TestRemove:
    def test_deletes_a_previously_merged_row(self):
        merged = _merge(_frame(_row("replace", capacity=100), _row("remove")))
        assert merged.empty or merged[merged["country"] == "FI"].empty

    def test_leaves_other_keys_alone(self):
        merged = _merge(
            _frame(
                _row("replace", country="FI", capacity=100),
                _row("replace", country="SE", capacity=200),
                _row("remove", country="FI"),
            )
        )
        assert merged[merged["country"] == "FI"].empty
        assert _value(merged, country="SE") == 200

    def test_a_later_row_can_reintroduce_a_removed_key(self):
        # remove deletes the accumulated record, it does not blacklist the key.
        merged = _merge(
            _frame(
                _row("replace", capacity=100),
                _row("remove"),
                _row("replace", capacity=50),
            )
        )
        assert _value(merged) == 50

    def test_removing_something_that_was_never_there_is_harmless(self):
        merged = _merge(_frame(_row("remove")))
        assert merged.empty


# ---------------------------------------------------------------------------
# Full versus partial overwriting -- the second hard part
# ---------------------------------------------------------------------------


class TestReplaceIsBlunt:
    def test_replace_overwrites_every_column_including_blanks(self):
        """The whole point of ``replace``: the later row wins outright.

        A blank in the overriding row is a decision to clear the value, not an
        omission. Users rely on this to wipe a parameter set by an earlier file.
        """
        merged = _merge(
            _frame(
                _row("replace", capacity=100, vomcosts=5),
                _row("replace", capacity=200),
            )
        )
        assert _value(merged) == 200
        assert pd.isna(_value(merged, "vomcosts"))

    def test_replace_can_set_a_value_to_zero(self):
        # NA and 0 are distinct in the source stage precisely so this works.
        merged = _merge(
            _frame(_row("replace", capacity=100), _row("replace", capacity=0))
        )
        assert _value(merged) == 0


class TestReplacePartialIsSurgical:
    def test_only_the_columns_the_row_filled_in_are_overwritten(self):
        merged = _merge(
            _frame(
                _row("replace", capacity=100, vomcosts=5),
                _row("replace-partial", capacity=200),
            )
        )
        assert _value(merged) == 200
        assert _value(merged, "vomcosts") == 5   # survives

    def test_a_blank_leaves_the_earlier_value_standing(self):
        """The difference from ``replace``, stated directly.

        This is the pair of behaviours the method column exists to offer, and
        confusing them is the classic failure: an overlay meant to adjust one
        parameter silently blanks every other one.
        """
        merged = _merge(
            _frame(
                _row("replace", capacity=100, vomcosts=5),
                _row("replace-partial", capacity=None, vomcosts=9),
            )
        )
        assert _value(merged) == 100
        assert _value(merged, "vomcosts") == 9

    def test_but_an_explicit_zero_does_overwrite(self):
        # Zero is a provided value, not an omission -- documented at :888.
        merged = _merge(
            _frame(_row("replace", capacity=100), _row("replace-partial", capacity=0))
        )
        assert _value(merged) == 0


# ---------------------------------------------------------------------------
# The non-negative clamp
# ---------------------------------------------------------------------------


class TestAddNonNegative:
    def test_a_stack_of_independent_reductions_cannot_go_negative(self):
        """The scenario the clamp was added for.

        A starting capacity, a scenario that removes it because a CHP overlay
        adds the same plant back, and a climate-policy overlay that reduces the
        starting electricity capacity again. Each edit is reasonable on its own;
        together they undershoot. Without the clamp the model gets a negative
        capacity, which is not a modelling statement -- it is nonsense that
        propagates.
        """
        merged = _merge(
            _frame(
                _row("replace", capacity=1000),
                _row("add-non-negative", capacity=-1000),
                _row("add-non-negative", capacity=-500),
            )
        )
        assert _value(merged) == 0

    def test_it_clamps_the_running_total_not_each_addend(self):
        # -600 then +100 must land on 100, not on 0: clamping each step would
        # discard the recovery and quietly lose capacity.
        merged = _merge(
            _frame(
                _row("replace", capacity=500),
                _row("add-non-negative", capacity=-600),
                _row("add-non-negative", capacity=100),
            )
        )
        assert _value(merged) == 100

    def test_plain_add_is_left_to_go_negative(self):
        # The unclamped method still exists; a negative result is sometimes
        # meaningful, which is why the clamp is opt-in per row.
        merged = _merge(
            _frame(_row("replace", capacity=100), _row("add", capacity=-500))
        )
        assert _value(merged) == -400

    def test_a_negative_first_occurrence_is_clamped_too(self):
        # Initialisation goes through a different branch (:1029-1034), so the
        # first row needs its own check.
        merged = _merge(_frame(_row("add-non-negative", capacity=-50)))
        assert _value(merged) == 0

    def test_the_clamp_applies_to_every_measure_on_the_row(self):
        """The clamp is per row, not per column -- by design.

        ``add-non-negative`` floors *all* inferred measure columns, so a row
        that adjusts a capacity downward also floors any cost on the same row.
        Pinned so the scope is visible rather than discovered: a user who wants
        a negative cost puts it in its own ``add`` row, which is what the
        per-row method is for.
        """
        merged = _merge(
            _frame(
                _row("replace", capacity=1000, vomcosts=10),
                _row("add-non-negative", capacity=-1000, vomcosts=-30),
            )
        )
        assert _value(merged) == 0
        assert _value(merged, "vomcosts") == 0     # clamped along with capacity

    def test_a_cost_row_can_still_go_negative_using_plain_add(self):
        # The escape hatch that makes the current design workable: negative
        # costs are ordinary (subsidies, revenues, negative market prices), and
        # a row using plain 'add' is never clamped.
        merged = _merge(
            _frame(_row("replace", vomcosts=10), _row("add", vomcosts=-30))
        )
        assert _value(merged, "vomcosts") == -20


class TestNegativeCostsAreLegitimate:
    """Negative costs are ordinary; negative capacities are not.

    Subsidies, revenue streams and negative market prices are all real, so a
    cost below zero is a modelling statement. A negative capacity never is.

    The method is chosen **per row**, as the function's name says, and that is
    the design rather than a limitation: a row asking for a floor gets a floor
    on everything it touches. Expressing a capacity guard and a subsidy at once
    is done with two rows -- ``add-non-negative`` for the capacity and plain
    ``add`` for the cost. Deciding per column would need a ``{column: rule}``
    table, which this project deliberately avoids, or guessing intent from
    column spelling.

    So the tests here pin the split as the supported way to do it, rather than
    treating it as something to route around.
    """

    def test_capacity_is_guarded_and_cost_is_free_when_split_across_rows(self):
        merged = _merge(
            _frame(
                _row("replace", capacity=1000, vomcosts=10),
                _row("add-non-negative", capacity=-5000),
                _row("add", vomcosts=-30),
            )
        )
        assert _value(merged) == 0                  # capacity floored
        assert _value(merged, "vomcosts") == -20    # subsidy survives

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "add-non-negative clamps every measure column in the sheet, not the "
            "ones the row supplies, so it floors a negative cost an earlier row "
            "established -- and the result depends on row order"
        ),
    )
    def test_an_add_non_negative_row_does_not_touch_columns_it_never_mentions(self):
        """Open: the clamp reads *sheet* columns, not the row's own columns.

        ``present_measures`` is computed once, from the union of columns across
        all frames (:942-975), and ``_handle_add`` then loops over all of it
        regardless of what the current row actually filled in (:1039-1051). A
        row supplying only ``capacity`` still reaches every other measure in the
        accumulated record -- and clamps it.

        Concretely:

            replace           capacity=1000  vomcosts=10
            add               vomcosts=-30      -> vomcosts = -20
            add-non-negative  capacity=-5000    -> capacity = 0, vomcosts = 0

        The subsidy is gone, silently, in a column that row never named. Reverse
        the last two rows and it survives.

        This is why "put costs in an 'add' row and capacities in an
        'add-non-negative' row" is necessary but not sufficient: the cost row
        must also come *after* every add-non-negative row touching the same key.
        Two people writing separate overlay workbooks cannot see or control
        that ordering, which is the failure mode overlays exist to avoid.

        The likely repair is to apply a handler only to the columns the row
        supplied, which would also make the two rows commute. It is not made
        here: this is the most failure-prone function in the repo, every source
        category flows through it, and the change alters merge semantics rather
        than repairing a slip.
        """
        subsidy_then_guard = _merge(
            _frame(
                _row("replace", capacity=1000, vomcosts=10),
                _row("add", vomcosts=-30),
                _row("add-non-negative", capacity=-5000),
            )
        )
        guard_then_subsidy = _merge(
            _frame(
                _row("replace", capacity=1000, vomcosts=10),
                _row("add-non-negative", capacity=-5000),
                _row("add", vomcosts=-30),
            )
        )

        # The capacity guard works either way; only the untouched column differs.
        assert _value(subsidy_then_guard) == _value(guard_then_subsidy) == 0
        assert _value(guard_then_subsidy, "vomcosts") == -20
        assert _value(subsidy_then_guard, "vomcosts") == -20

    def test_and_it_still_works_across_a_frame_boundary(self):
        # The two rows commonly come from different overlay workbooks -- one
        # scenario adjusting capacity, another applying a subsidy.
        rows = (
            _row("replace", capacity=1000, vomcosts=10),
            _row("add-non-negative", capacity=-5000),
            _row("add", vomcosts=-30),
        )
        together = _merge(_frame(*rows))
        apart = _merge(_frame(rows[0]), _frame(rows[1]), _frame(rows[2]))
        pd.testing.assert_frame_equal(together, apart)


# ---------------------------------------------------------------------------
# Frame boundaries -- the invariant most at risk from refactoring
# ---------------------------------------------------------------------------


class TestFrameBoundariesDoNotMatter:
    """Two rows must merge identically however they are split across frames.

    Users override values by adding a row, and whether that row lives in the
    same sheet or in a different workbook is an accident of how they organise
    their files. The moment this stops holding, an overlay behaves differently
    depending on where it was written down, which is close to impossible to
    debug from a symptom.
    """

    METHODS = ("replace", "replace-partial", "add", "add-non-negative", "multiply")

    @pytest.mark.parametrize("second", METHODS)
    @pytest.mark.parametrize("first", METHODS)
    def test_one_frame_equals_two_frames(self, first, second):
        rows = (_row(first, capacity=100, vomcosts=5), _row(second, capacity=7, vomcosts=None))

        together = _merge(_frame(*rows))
        apart = _merge(_frame(rows[0]), _frame(rows[1]))

        pd.testing.assert_frame_equal(together, apart)

    @pytest.mark.parametrize(
        "methods", list(itertools.product(("replace", "add", "multiply"), repeat=3))
    )
    def test_three_rows_split_at_every_boundary(self, methods):
        rows = [_row(m, capacity=v) for m, v in zip(methods, (100, 3, 2))]

        one = _merge(_frame(*rows))
        split_after_first = _merge(_frame(rows[0]), _frame(rows[1], rows[2]))
        split_after_second = _merge(_frame(rows[0], rows[1]), _frame(rows[2]))
        all_separate = _merge(*[_frame(r) for r in rows])

        for other in (split_after_first, split_after_second, all_separate):
            pd.testing.assert_frame_equal(one, other)

    def test_remove_works_across_a_frame_boundary(self):
        # A workbook loaded later must be able to delete a row an earlier one
        # created; this is how a scenario drops a plant it does not model.
        together = _merge(_frame(_row("replace", capacity=100), _row("remove")))
        apart = _merge(_frame(_row("replace", capacity=100)), _frame(_row("remove")))
        pd.testing.assert_frame_equal(together, apart)

    def test_an_empty_frame_between_two_others_changes_nothing(self):
        rows = (_row("replace", capacity=100), _row("add", capacity=50))
        without = _merge(_frame(*rows))
        with_empty = _merge(_frame(rows[0]), pd.DataFrame(), _frame(rows[1]))
        pd.testing.assert_frame_equal(without, with_empty)


class TestOrderWithinTheMerge:
    def test_the_last_row_wins_for_replace(self):
        merged = _merge(
            _frame(
                _row("replace", capacity=1),
                _row("replace", capacity=2),
                _row("replace", capacity=3),
            )
        )
        assert _value(merged) == 3

    def test_accumulating_methods_apply_in_order(self):
        # (100 + 50) * 2 = 300, not 100 + (50 * 2).
        merged = _merge(
            _frame(
                _row("replace", capacity=100),
                _row("add", capacity=50),
                _row("multiply", capacity=2),
            )
        )
        assert _value(merged) == 300


# ---------------------------------------------------------------------------
# Keys, measures and output shape
# ---------------------------------------------------------------------------


class TestKeys:
    def test_rows_with_different_keys_do_not_interact(self):
        merged = _merge(
            _frame(
                _row("replace", country="FI", capacity=100),
                _row("add", country="SE", capacity=50),
            )
        )
        assert _value(merged, country="FI") == 100
        assert _value(merged, country="SE") == 50

    def test_every_key_column_participates(self):
        # Same country, different grid: a key that ignored 'grid' would merge
        # these two into one row and lose a whole energy carrier.
        merged = _merge(
            _frame(
                _row("replace", grid="elec", capacity=100),
                _row("replace", grid="heat", capacity=200),
            )
        )
        assert _value(merged, grid="elec") == 100
        assert _value(merged, grid="heat") == 200

    def test_a_missing_key_value_is_its_own_key(self):
        merged = _merge(
            _frame(_row("replace", grid=None, capacity=1), _row("replace", grid="elec", capacity=2))
        )
        assert len(merged) == 2


class TestMeasureInference:
    def test_year_is_not_a_measure_by_default(self):
        # not_measure_cols=("year",): adding two rows must not sum their years.
        merged = _merge(
            _frame(
                _row("replace", capacity=100, year=2030),
                _row("add", capacity=50, year=2030),
            )
        )
        assert _value(merged, "year") == 2030
        assert _value(merged) == 150

    def test_text_columns_are_not_accumulated(self):
        merged = _merge(
            _frame(
                _row("replace", capacity=100, unittype="coal"),
                _row("add", capacity=50, unittype="coal"),
            )
        )
        assert _value(merged, "unittype") == "coal"

    def test_explicit_measure_cols_limit_what_accumulates(self):
        merged = _merge(
            _frame(
                _row("replace", capacity=100, vomcosts=5),
                _row("add", capacity=50, vomcosts=5),
            ),
            measure_cols=["capacity"],
        )
        assert _value(merged) == 150
        assert _value(merged, "vomcosts") == 5   # not a measure, so replaced

    def test_unknown_measure_cols_are_reported(self):
        logger = FakeLogger()
        merge_row_by_row(
            [_frame(_row("replace", capacity=1))],
            logger,
            key_columns=KEY,
            measure_cols=["capacity", "nonexistent"],
        )
        logger.assert_logged("Some measure_cols not found", level="warn")

    def test_a_method_does_not_change_because_another_column_is_numeric(self):
        """Regression: a sheet with no numeric column made every method blunt.

        ``present_measures`` is inferred from the columns, and the merge loop
        used to fall back to a full replace whenever it came out empty. So the
        very same 'add' row overwrote a text column when the sheet had no
        numbers in it, and left that column alone when it did -- behaviour
        depending on whether some *unrelated* column happened to be numeric.

        'add' concerns measures. With none present it has nothing to do.
        """
        without_numbers = _merge(
            _frame(_row("replace", unittype="coal"), _row("add", unittype="gas"))
        )
        with_numbers = _merge(
            _frame(
                _row("replace", unittype="coal", capacity=100),
                _row("add", unittype="gas", capacity=1),
            )
        )
        assert _value(without_numbers, "unittype") == "coal"
        assert _value(with_numbers, "unittype") == "coal"


class TestEmptyRows:
    """What a row carrying no values does, per method.

    A blank overlay row is easy to produce by accident -- a spreadsheet row left
    half-filled, a template copied but not completed -- so every method needs a
    defined answer. Only ``replace`` may destroy anything: clearing values is
    what it is for.
    """

    ACCUMULATING = ("replace-partial", "add", "add-non-negative", "multiply")

    @pytest.mark.parametrize("method", ACCUMULATING)
    def test_an_empty_row_leaves_the_record_untouched(self, method):
        merged = _merge(
            _frame(
                _row("replace", capacity=100, vomcosts=5, unittype="coal"),
                _row(method),
            )
        )
        assert _value(merged) == 100
        assert _value(merged, "vomcosts") == 5
        assert _value(merged, "unittype") == "coal"

    @pytest.mark.parametrize("method", ACCUMULATING)
    def test_and_still_does_so_when_the_sheet_has_no_numeric_column(self, method):
        # The regression: with no measures to iterate, these used to fall
        # through to a full replace and blank the record.
        merged = _merge(
            _frame(_row("replace", unittype="coal"), _row(method))
        )
        assert _value(merged, "unittype") == "coal"

    @pytest.mark.parametrize("method", ACCUMULATING)
    def test_and_across_a_frame_boundary(self, method):
        # The accidental blank row usually arrives in a separate overlay file.
        rows = (_row("replace", capacity=100, unittype="coal"), _row(method))
        together = _merge(_frame(*rows))
        apart = _merge(_frame(rows[0]), _frame(rows[1]))
        pd.testing.assert_frame_equal(together, apart)

    def test_an_empty_replace_row_does_clear_the_record(self):
        # The one method allowed to destroy: 'replace' means the later row wins
        # outright, blanks included, and users rely on it to clear a value.
        merged = _merge(
            _frame(_row("replace", capacity=100, unittype="coal"), _row("replace"))
        )
        assert pd.isna(_value(merged))
        assert pd.isna(_value(merged, "unittype"))

    def test_an_empty_row_as_the_first_occurrence_creates_a_blank_record(self):
        # Nothing to accumulate onto; the key exists but carries no values.
        merged = _merge(_frame(_row("add", capacity=None)))
        assert len(merged) == 1
        assert pd.isna(_value(merged))

    def test_remove_ignores_any_values_it_carries(self):
        """Accepted quirk, pinned deliberately.

        A 'remove' row with values still only removes. Reading it as
        "remove, then re-add these values" would be defensible, but deleting is
        the more intuitive reading of the word and it is what users expect.
        """
        merged = _merge(
            _frame(_row("replace", capacity=100), _row("remove", capacity=999))
        )
        assert merged.empty


class TestOutputShape:
    def test_meta_columns_do_not_survive(self):
        merged = _merge(_frame(_row("replace", capacity=100)))
        assert "method" not in merged.columns

    def test_the_output_satisfies_the_dtype_contract(self):
        merged = _merge(
            _frame(
                _row("replace", capacity=100, vomcosts=None),
                _row("add", capacity=50),
            )
        )
        assert_normalized(merged, where="merge_row_by_row")

    def test_no_frames_gives_an_empty_frame(self):
        assert merge_row_by_row([], FakeLogger(), key_columns=KEY).empty

    def test_only_empty_frames_gives_an_empty_frame(self):
        assert merge_row_by_row(
            [pd.DataFrame(), None], FakeLogger(), key_columns=KEY
        ).empty

    def test_columns_from_every_frame_are_kept(self):
        merged = _merge(
            _frame(_row("replace", capacity=100)),
            _frame(_row("replace-partial", country="SE", vomcosts=5)),
        )
        assert {"capacity", "vomcosts"} <= set(merged.columns)
