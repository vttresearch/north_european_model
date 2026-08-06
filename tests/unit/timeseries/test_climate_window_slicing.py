"""``split_timeseries_to_climate_windows`` -- turning timestamps into t-labels.

Backbone indexes time by label (``t000001``, ``t000002``, ...), not by
timestamp, so this function is where real hours become model hours. It assigns
the labels by **row position within each group**, which is fast and correct as
long as every group holds exactly one row per hour of the window.

It still does, and always will: deriving the label from the timestamp instead
was considered and rejected. What changed is that the precondition is now
*proved* rather than assumed. ``ProcessorRunner`` calls
``find_time_axis_defects`` before this function ever sees the frame, and rejects
a processor whose axis has holes, repeats, sub-hourly rows, or groups covering
different spans.

So the behaviour characterised below is no longer reachable through the
pipeline. The tests that pin it are kept deliberately: they are the executable
statement of *why* the gate exists, and they are the only place the consequence
is written down in full. A gap does not leave a hole in the labels -- it pulls
everything after it one step earlier. Two nodes whose source data differ by a
single missing hour end up disagreeing about what ``t000024`` means for the
remainder of the window, and for a model whose value is largely in the
correlation between countries, a silent one-hour offset between them is not a
small error.
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
    """Characterisation of the helper called directly.

    Reaching any of this through the pipeline is now impossible -- these pin
    why. The function itself is unchanged and undefended: it is fast because it
    trusts its input, and ``ProcessorRunner`` is what makes that trust sound.
    """

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

    def test_this_function_still_reports_none_of_it(self):
        # It takes no logger and returns no diagnostics, by design: the check
        # lives one level up, where it can name the processor. See
        # test_processor_contract.py::TestTimeAxis for the rejection this now
        # gets instead.
        out = _split(self._gapped())[2014]
        assert len(out) < 72

# Retired here: an xfail proposing that labels follow the timestamp rather than
# the row number. That design was not adopted -- row-position labelling stays,
# and the gap is now rejected upstream instead of being tolerated downstream --
# so the test could never have passed and was a permanent fixture in the xfail
# register rather than an open question. What it gave up: any hope that this
# function is safe called in isolation. What replaced it: proof that it never is
# called that way. Its cost argument (~2 s per parameter, so the check belongs
# in a separate tool) was measured and found wrong by a factor of about 25.
