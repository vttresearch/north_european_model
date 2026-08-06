"""``order_timeseries_for_labelling`` and ``find_time_axis_defects``.

Backbone indexes time by label, and ``split_timeseries_to_climate_windows``
derives the label from a row's **position** within its group. That is fast and
exactly correct as long as every group holds one row per hour of the window --
and silently, undetectably wrong otherwise, because a hole does not leave a hole
in the labels. It pulls every later hour one label earlier, and the values that
land there are all perfectly plausible.

These two functions are the proof of that precondition. They are tested directly
rather than only through ``ProcessorRunner`` because the runner delegates to them
and ``assert_even_hourly_coverage`` wraps them: if all three shared one
implementation *and* one test, a wrong answer would agree with itself everywhere.

The checker is deliberately step-generic. The pipeline's hourly assumption lives
in ``split_timeseries_to_climate_windows`` (its window is ``bb_ts_length * 24``
labels), not here.
"""

import numpy as np
import pandas as pd
import pytest

from src.timeseries.timeseries_helpers import (
    find_incomplete_climate_windows,
    find_time_axis_defects,
    order_timeseries_for_labelling,
)

DIMS = ["grid", "node"]


def _frame(times, node="FI_elec", grid="elec"):
    return pd.DataFrame(
        {
            "grid": grid,
            "node": node,
            "time": pd.Index(times),
            "value": np.arange(len(times), dtype=float),
        }
    )


def _hours(n, start="2014-01-01"):
    return pd.date_range(start, periods=n, freq="h")


def _check(df, *, group_dims=DIMS, **kwargs):
    """Order then check, which is the only supported way to call the checker."""
    ordered, gids = order_timeseries_for_labelling(df, group_dims=group_dims)
    return find_time_axis_defects(ordered, gids, **kwargs)


class TestOrdering:
    def test_rows_come_out_grouped_and_then_chronological(self):
        times = _hours(4)
        shuffled = pd.concat(
            [_frame(times[::-1], "B"), _frame(times[::-1], "A")], ignore_index=True
        )
        ordered, _ = order_timeseries_for_labelling(shuffled, group_dims=DIMS)

        assert ordered["node"].tolist() == ["A"] * 4 + ["B"] * 4
        assert ordered["time"].tolist() == list(times) * 2

    def test_group_ids_align_with_the_rows_positionally(self):
        times = _hours(3)
        both = pd.concat([_frame(times, "A"), _frame(times, "B")], ignore_index=True)
        ordered, gids = order_timeseries_for_labelling(both, group_dims=DIMS)

        assert len(gids) == len(ordered)
        assert gids.tolist() == [0, 0, 0, 1, 1, 1]

    def test_ids_change_exactly_where_the_group_changes(self):
        """The property the checker relies on -- not the specific numbers."""
        times = _hours(2)
        many = pd.concat(
            [_frame(times, n) for n in ("C", "A", "B")], ignore_index=True
        )
        ordered, gids = order_timeseries_for_labelling(many, group_dims=DIMS)

        changed = [ordered["node"].iloc[i] != ordered["node"].iloc[i - 1]
                   for i in range(1, len(ordered))]
        id_changed = (np.diff(gids) != 0).tolist()
        assert id_changed == changed

    def test_a_frame_with_no_grouping_dimensions_is_still_sorted_by_time(self):
        """Used to be left in whatever order the processor returned.

        Handing out a label that means "hour n of the window" on the strength of
        an arbitrary row order is not defensible, whether or not any current
        spec reaches this branch.
        """
        times = _hours(5)
        df = pd.DataFrame({"time": times[::-1], "value": np.arange(5.0)})
        ordered, gids = order_timeseries_for_labelling(df, group_dims=[])

        assert ordered["time"].is_monotonic_increasing
        assert gids.tolist() == [0] * 5

    def test_value_is_float64_afterwards(self):
        df = _frame(_hours(3))
        df["value"] = df["value"].astype("Float64")
        ordered, _ = order_timeseries_for_labelling(df, group_dims=DIMS)
        assert ordered["value"].dtype == np.float64

    def test_ordering_an_already_ordered_frame_changes_nothing(self):
        df = _frame(_hours(6))
        once, ids_once = order_timeseries_for_labelling(df, group_dims=DIMS)
        twice, ids_twice = order_timeseries_for_labelling(once, group_dims=DIMS)

        pd.testing.assert_frame_equal(once, twice)
        assert ids_once.tolist() == ids_twice.tolist()


class TestACleanAxisPasses:
    def test_one_complete_group(self):
        assert _check(_frame(_hours(72))).ok

    def test_several_groups_on_the_same_span(self):
        times = _hours(48)
        both = pd.concat([_frame(times, "A"), _frame(times, "B")], ignore_index=True)
        report = _check(both)

        assert report.ok
        assert report.n_groups == 2
        assert report.n_rows == 96

    def test_a_single_row_is_a_complete_axis(self):
        # One row cannot have a gap, a repeat or a ragged edge.
        assert _check(_frame(_hours(1))).ok

    def test_an_empty_frame_is_not_a_defect(self):
        # Emptiness is rejected earlier, with a message that can say so.
        report = _check(_frame(_hours(0)))
        assert report.ok
        assert report.n_rows == 0

    def test_the_span_is_reported_even_when_clean(self):
        report = _check(_frame(_hours(24)))
        assert report.first_time == pd.Timestamp("2014-01-01 00:00")
        assert report.last_time == pd.Timestamp("2014-01-01 23:00")


class TestGaps:
    def test_a_missing_hour_is_found(self):
        report = _check(_frame(_hours(72).delete(10)))

        assert not report.ok
        assert report.n_gaps == 1
        assert report.n_duplicate_or_finer_than_step == 0

    def test_the_gap_is_located(self):
        """The message has to name a timestamp, so the report has to carry one."""
        report = _check(_frame(_hours(72).delete(10)))

        # Hour 10 is gone, so the step lands on the row that follows it.
        assert report.first_defect_time == pd.Timestamp("2014-01-01 11:00")
        assert report.first_defect_index == 10

    def test_gaps_are_counted_per_group_not_across_the_boundary(self):
        """Two groups on the same span are not a gap where one ends.

        Consecutive rows spanning a group boundary jump backwards in time by the
        whole window. If boundaries were not excluded, every multi-group frame
        would report a defect.
        """
        times = _hours(24)
        both = pd.concat([_frame(times, "A"), _frame(times, "B")], ignore_index=True)
        assert _check(both).ok

    def test_each_group_is_checked(self):
        times = _hours(48)
        both = pd.concat(
            [_frame(times, "A"), _frame(times.delete(5), "B")], ignore_index=True
        )
        report = _check(both)

        assert not report.ok
        assert report.n_gaps == 1


class TestRepeatsAndSubStepData:
    def test_a_duplicated_timestamp_is_found(self):
        times = _hours(24)
        df = pd.concat([_frame(times), _frame(times[3:4])], ignore_index=True)
        report = _check(df)

        assert not report.ok
        assert report.n_duplicate_or_finer_than_step == 1
        assert report.n_gaps == 0

    def test_sub_hourly_rows_are_found(self):
        """What ``duplicated()`` could not see.

        00:00 and 00:15 are distinct timestamps, so a duplicate check passes
        them, and then row-position labelling silently treats the quarter-hour
        as the next model hour. Bucketing by the step is what catches it.
        """
        times = pd.to_datetime(
            ["2014-01-01 00:00", "2014-01-01 00:15", "2014-01-01 01:00"]
        )
        report = _check(_frame(times))

        assert not report.ok
        assert report.n_duplicate_or_finer_than_step == 1

    def test_a_finer_step_is_accepted_when_it_is_the_declared_step(self):
        """The rule is regularity on the given step, not hourliness."""
        times = pd.date_range("2014-01-01", periods=8, freq="15min")
        assert _check(_frame(times), step=pd.Timedelta(15, unit="min")).ok
        assert not _check(_frame(times)).ok


class TestRaggedExtents:
    def test_groups_covering_different_spans_are_rejected(self):
        """Every step is 1 and the frame is still fatal.

        Each group is internally flawless, so the step check alone passes them.
        They still disagree about which real hour ``t000001`` names, which is
        the same corruption a gap causes, arriving by a different route.
        """
        both = pd.concat(
            [
                _frame(_hours(24, "2014-01-01"), "A"),
                _frame(_hours(24, "2014-01-02"), "B"),
            ],
            ignore_index=True,
        )
        report = _check(both)

        assert report.n_gaps == 0
        assert report.n_duplicate_or_finer_than_step == 0
        assert report.ragged_extent
        assert not report.ok

    def test_a_group_that_stops_early_is_rejected(self):
        times = _hours(48)
        both = pd.concat(
            [_frame(times, "A"), _frame(times[:36], "B")], ignore_index=True
        )
        report = _check(both)

        assert report.ragged_extent
        assert report.group_last_range == (
            pd.Timestamp("2014-01-02 11:00"),
            pd.Timestamp("2014-01-02 23:00"),
        )

    def test_the_disagreement_is_reported_at_both_ends(self):
        both = pd.concat(
            [
                _frame(_hours(24, "2014-01-01"), "A"),
                _frame(_hours(24, "2014-01-02"), "B"),
            ],
            ignore_index=True,
        )
        report = _check(both)

        assert report.group_first_range[0] != report.group_first_range[1]
        assert report.group_last_range[0] != report.group_last_range[1]

    def test_matching_spans_are_not_ragged(self):
        times = _hours(24)
        both = pd.concat([_frame(times, "A"), _frame(times, "B")], ignore_index=True)
        report = _check(both)

        assert not report.ragged_extent
        assert report.group_first_range[0] == report.group_first_range[1]


class TestMissingTimestamps:
    def test_a_nat_is_found(self):
        times = list(_hours(5))
        times[2] = pd.NaT
        report = _check(_frame(times))

        assert not report.ok
        assert report.n_missing_timestamps == 1

    def test_a_nat_short_circuits_the_rest(self):
        """Reporting a 292-year gap next to a NaT would be noise, not a finding.

        NaT is int64 minimum underneath, so every difference involving it is
        nonsense. The count of missing timestamps is the actionable fact and the
        only one reported.
        """
        times = list(_hours(5))
        times[2] = pd.NaT
        report = _check(_frame(times))

        assert report.n_gaps == 0
        assert report.n_duplicate_or_finer_than_step == 0
        assert not report.ragged_extent

    def test_an_unconvertible_time_becomes_a_missing_timestamp(self):
        df = _frame(_hours(4))
        df["time"] = ["2014-01-01 00:00", "not a date", "2014-01-01 02:00", "2014-01-01 03:00"]
        report = _check(df)

        assert report.n_missing_timestamps == 1


class TestIncompleteClimateWindows:
    def test_a_full_window_is_not_reported(self):
        frames = {2014: pd.DataFrame(index=range(72)), 2015: pd.DataFrame(index=range(72))}
        assert find_incomplete_climate_windows(frames, expected_rows=72) == {}

    def test_a_short_window_is_reported_with_its_size(self):
        frames = {2014: pd.DataFrame(index=range(72)), 2015: pd.DataFrame(index=range(60))}
        assert find_incomplete_climate_windows(frames, expected_rows=72) == {2015: 60}

    def test_nothing_is_claimed_when_the_expectation_is_unknown(self):
        # expected_rows <= 0 means the caller could not compute it, which is not
        # the same as every window being wrong.
        frames = {2014: pd.DataFrame(index=range(60))}
        assert find_incomplete_climate_windows(frames, expected_rows=0) == {}


class TestTheCheckerRefusesToGuess:
    def test_it_reads_group_ids_positionally_and_does_not_re_sort(self):
        """Passing ids that do not match the frame is a caller error, not a
        thing the checker quietly repairs -- it has no way to tell a wrong id
        array from a legitimately odd grouping."""
        df = _frame(_hours(4))
        ordered, gids = order_timeseries_for_labelling(df, group_dims=DIMS)

        # Claim every row is its own group: no two rows are ever compared.
        lied_to = find_time_axis_defects(ordered, np.arange(len(ordered)))
        assert lied_to.n_groups == 4
        assert lied_to.n_gaps == 0

    @pytest.mark.parametrize("step_hours", [1, 2, 24])
    def test_the_step_is_honoured(self, step_hours):
        times = pd.date_range("2014-01-01", periods=10, freq=f"{step_hours}h")
        assert _check(_frame(times), step=pd.Timedelta(step_hours, unit="h")).ok
