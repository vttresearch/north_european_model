"""``split_timeseries_to_climate_windows`` -- turning timestamps into t-labels.

Backbone indexes time by label (``t000001``, ``t000002``, ...), not by
timestamp, so this function is where real hours become model hours. It assigns
the labels by **row position within each group** (:247-258), which is fast and
correct as long as every group holds exactly one row per hour of the window.

Nothing enforces that. The processor contract documents that processors must
cover the full date range, and ``ProcessorRunner`` checks columns, emptiness and
duplicates -- but not completeness. So a gap does not leave a hole in the
labels: it pulls everything after it one step earlier.

That matters most across groups. Two nodes whose source data differ by a single
missing hour end up disagreeing about what ``t000024`` means, for the remainder
of the window -- and for a model whose value is largely in the correlation
between countries, a silent one-hour offset between them is not a small error.
"""

import pandas as pd
import pytest

from src.timeseries.timeseries_helpers import split_timeseries_to_climate_windows

DIMS = ["grid", "node", "f", "t"]
FULL_DAY = 24


def _frame(times, node="FI_elec"):
    """Value equals the hour-of-day, so a shift is readable straight off the value."""
    return pd.DataFrame(
        {
            "grid": "elec",
            "node": node,
            "time": times,
            "value": [float(t.hour) for t in times],
        }
    )


def _split(df, *, days=3, year=2014, start="01-01"):
    return split_timeseries_to_climate_windows(
        df,
        bb_parameter_dimensions=DIMS,
        bb_ts_start=start,
        bb_ts_length=days,
        valid_climate_years=[year],
    )


def _at(frame, label, node=None):
    sub = frame if node is None else frame[frame["node"] == node]
    match = sub[sub["t"] == label]
    return None if match.empty else match["value"].iloc[0]


class TestCompleteData:
    def test_every_hour_of_the_window_gets_a_label(self):
        out = _split(_frame(pd.date_range("2014-01-01", periods=72, freq="h")))[2014]
        assert len(out) == 72
        assert out["t"].iloc[0] == "t000001"
        assert out["t"].iloc[-1] == "t000072"

    def test_labels_line_up_with_the_hours_they_came_from(self):
        out = _split(_frame(pd.date_range("2014-01-01", periods=72, freq="h")))[2014]
        assert _at(out, "t000001") == 0.0     # midnight
        assert _at(out, "t000011") == 10.0
        assert _at(out, "t000025") == 0.0     # midnight, day two

    def test_the_realized_weather_branch_is_inserted(self):
        out = _split(_frame(pd.date_range("2014-01-01", periods=48, freq="h")), days=2)[2014]
        assert set(out["f"]) == {"f00"}

    def test_groups_are_labelled_independently(self):
        times = pd.date_range("2014-01-01", periods=48, freq="h")
        both = pd.concat([_frame(times, "A"), _frame(times, "B")], ignore_index=True)
        out = _split(both, days=2)[2014]
        assert _at(out, "t000024", "A") == _at(out, "t000024", "B") == 23.0

    def test_only_the_requested_window_is_taken(self):
        # A whole year in, three days out.
        out = _split(_frame(pd.date_range("2014-01-01", periods=8760, freq="h")))[2014]
        assert len(out) == 72

    def test_a_year_without_data_is_skipped_rather_than_emitted_empty(self):
        out = split_timeseries_to_climate_windows(
            _frame(pd.date_range("2014-01-01", periods=72, freq="h")),
            bb_parameter_dimensions=DIMS,
            bb_ts_start="01-01",
            bb_ts_length=3,
            valid_climate_years=[2014, 2015],
        )
        assert set(out) == {2014}


class TestGapsShiftTheRemainderOfTheWindow:
    """Characterisation. None of this is announced anywhere."""

    def _gapped(self, drop=10, periods=72, node="FI_elec"):
        times = pd.date_range("2014-01-01", periods=periods, freq="h").delete(drop)
        return _frame(times, node)

    def test_a_missing_hour_shortens_the_window(self):
        out = _split(self._gapped())[2014]
        assert len(out) == 71
        assert out["t"].iloc[-1] == "t000071"

    def test_and_pulls_every_later_hour_one_step_earlier(self):
        """The label no longer means the hour it names.

        Hour 10 is missing, so hour 11's value lands on t000011 -- and hour 71's
        value lands on t000071, an hour early, for the rest of the window.
        """
        out = _split(self._gapped())[2014]
        assert _at(out, "t000011") == 11.0      # would be 10.0 with no gap
        assert _at(out, "t000010") == 9.0       # everything before the gap is fine

    def test_two_groups_with_different_gaps_desynchronise(self):
        """The consequence that matters for a multi-country model.

        Node B is missing one hour, so from that point on the two nodes'
        t-labels refer to different real hours. Every cross-country
        relationship the model is built to capture is computed across that
        offset.
        """
        times = pd.date_range("2014-01-01", periods=72, freq="h")
        both = pd.concat(
            [_frame(times, "A"), self._gapped(node="B")], ignore_index=True
        )
        out = _split(both)[2014]

        assert _at(out, "t000024", "A") == 23.0
        assert _at(out, "t000024", "B") == 0.0     # already an hour ahead
        assert _at(out, "t000024", "A") != _at(out, "t000024", "B")

    def test_a_short_group_simply_stops_early(self):
        # Not padded, not reported -- the window is just shorter for that node.
        times = pd.date_range("2014-01-01", periods=72, freq="h")
        both = pd.concat(
            [_frame(times, "A"), _frame(times[:60], "B")], ignore_index=True
        )
        out = _split(both)[2014]
        assert len(out[out["node"] == "A"]) == 72
        assert len(out[out["node"] == "B"]) == 60

    def test_nothing_reports_any_of_this(self):
        # The function takes no logger and returns no diagnostics, so a gap is
        # indistinguishable from complete data to every caller.
        out = _split(self._gapped())[2014]
        assert len(out) < 72


class TestWhatShouldHappenInstead:
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "t-labels are assigned by row position, so a gap shifts every later "
            "hour instead of leaving a hole; nothing detects or reports it"
        ),
    )
    def test_a_gap_keeps_later_hours_on_their_own_labels(self):
        """Open: labels should follow the timestamp, not the row number.

        Deriving the label from the offset between the row's timestamp and the
        window start would make a gap a *missing row* -- which the GDX gate
        already handles, converting it to 0 and able to report it -- instead of
        a silent one-hour shift of everything after it.

        Row-position labelling is the faster path and is correct whenever the
        data is complete, which it is for the well-behaved processors. The
        exposure is that nothing checks completeness: the processor contract
        asks for full coverage, and ProcessorRunner validates columns,
        emptiness and duplicates but not this.

        Detecting it is cheap per run but not per *build*: checking that every
        group has the same row count costs about 2 s per parameter, ~16 s a
        build, and users generate 20-50 input folders for a scenario sweep. The
        answer never changes between those builds -- it is a property of the
        processor and its source data -- so it belongs in the timeseries data
        verifier, run once when either changes, not in the pipeline.

        Decided during phase 5. The pipeline checks form; coverage is content.
        ``tests/_common/processor_contract.assert_even_hourly_coverage`` already
        implements the check and is opt-in for exactly this reason.
        """
        times = pd.date_range("2014-01-01", periods=72, freq="h")
        out = _split(_frame(times.delete(10)))[2014]

        assert _at(out, "t000011") == 10.0   # hour 10 absent, hour 11 stays put
        assert _at(out, "t000012") == 11.0
