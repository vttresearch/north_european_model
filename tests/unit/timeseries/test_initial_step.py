"""``add_initial_step`` -- the t000000 copy Backbone would otherwise make itself.

Backbone reads ``ts_node`` at t000000, the step before the first hour of data,
where a storage's starting state sits. When a series has no value there it
copies t000001 in and writes a note to warnings.log. The build writes that copy
instead, for ``ts_node`` alone: realized and forecast branches, every series.
"""

import pandas as pd
import pytest

from src.GDX_exchange import read_gdx_parameter
from src.timeseries.timeseries_helpers import (
    add_initial_step,
    split_timeseries_to_climate_windows,
)
from tests._common.processor_contract import run_fake_processor

DIMS = ["grid", "node", "param_gnBoundaryTypes", "f", "t"]


def _window(days=2):
    """Two series whose values read off the hour: 10 + hour and 100 + hour."""
    times = pd.date_range("2014-01-01", periods=days * 24, freq="h")
    frames = [
        pd.DataFrame({
            "grid": "reservoir", "node": node, "param_gnBoundaryTypes": "upwardLimit",
            "time": times, "value": offset + times.hour,
        })
        for node, offset in (("FI00_reservoir", 10.0), ("SE01_reservoir", 100.0))
    ]
    out = split_timeseries_to_climate_windows(
        pd.concat(frames, ignore_index=True),
        bb_parameter_dimensions=DIMS,
        bb_ts_start="01-01",
        bb_ts_length=days,
        valid_climate_years=[2014],
    )
    return out[2014]


def _value(frame, node, label):
    rows = frame[(frame["node"] == node) & (frame["t"] == label)]
    assert len(rows) == 1, f"expected one {node} row at {label}, got {len(rows)}"
    return rows["value"].iloc[0]


class TestTheCopy:
    def test_each_series_gets_its_own_first_hour_at_t000000(self):
        out = add_initial_step(_window())

        assert _value(out, "FI00_reservoir", "t000000") == 10.0
        assert _value(out, "SE01_reservoir", "t000000") == 100.0

    def test_one_row_per_series_is_added_and_nothing_else_changes(self):
        window = _window()
        out = add_initial_step(window)

        assert len(out) == len(window) + 2
        rest = out[out["t"] != "t000000"].reset_index(drop=True)
        pd.testing.assert_frame_equal(rest, window.astype({"t": rest["t"].dtype}))

    def test_t_stays_categorical_with_t000000_first(self):
        out = add_initial_step(_window())

        assert isinstance(out["t"].dtype, pd.CategoricalDtype)
        assert out["t"].cat.categories[0] == "t000000"
        assert out["t"].iloc[0] == "t000000"

    def test_every_forecast_branch_gets_one(self):
        frame = pd.DataFrame({
            "node": ["A", "A", "A", "A"],
            "f": ["f01", "f01", "f02", "f02"],
            "t": ["t000001", "t000002", "t000001", "t000002"],
            "value": [1.0, 2.0, 3.0, 4.0],
        })
        out = add_initial_step(frame)

        initial = out[out["t"] == "t000000"].set_index("f")["value"]
        assert initial.to_dict() == {"f01": 1.0, "f02": 3.0}

    def test_a_frame_without_t000001_is_returned_as_it_is(self):
        frame = pd.DataFrame({"t": ["t000002"], "value": [1.0]})
        assert add_initial_step(frame) is frame


class TestThroughTheRunner:
    """Only ``ts_node`` gets the step, in the per-year files and the forecasts."""

    BODY = '_t = pd.date_range("2014-01-01", "2015-12-31 23:00", freq="h")'
    MAIN_RESULT = (
        "pd.concat([pd.DataFrame({"
        '"grid": "reservoir", "node": _n, "param_gnBoundaryTypes": "upwardLimit", '
        '"time": _t, "value": _k + _t.hour})'
        ' for _n, _k in [("FI00_reservoir", 10.0), ("SE01_reservoir", 100.0)]], '
        "ignore_index=True)"
    )

    def _run(self, tmp_path, bb_parameter):
        return run_fake_processor(
            tmp_path, self.MAIN_RESULT,
            body=self.BODY,
            dimensions=DIMS,
            spec_overrides={"bb_parameter": bb_parameter, "gdx_name_suffix": "limits"},
        )

    @pytest.mark.gams
    def test_ts_node_files_carry_t000000(self, tmp_path):
        run = self._run(tmp_path, "ts_node")
        run.logger.assert_no_errors()

        names = [f.name for f in run.gdx_files]
        assert names == [
            "ts_node_limits_2014.gdx", "ts_node_limits_2015.gdx", "ts_node_limits_forecasts.gdx",
        ]
        for path in run.gdx_files:
            records = read_gdx_parameter(str(path), "ts_node")
            records = records.astype({"node": str, "f": str, "t": str})
            for f in records["f"].unique():
                branch = records[records["f"] == f]
                assert _value(branch, "FI00_reservoir", "t000000") == 10.0, (path.name, f)
                assert _value(branch, "SE01_reservoir", "t000000") == 100.0, (path.name, f)

    @pytest.mark.gams
    def test_other_parameters_start_at_t000001(self, tmp_path):
        run = self._run(tmp_path, "ts_test")
        run.logger.assert_no_errors()

        assert run.gdx_files
        for path in run.gdx_files:
            records = read_gdx_parameter(str(path), "ts_test")
            assert "t000000" not in set(records["t"].astype(str)), path.name
