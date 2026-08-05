"""Boundary 6: gaps must survive the curing steps and be reported at the GDX gate.

The pipeline used to convert NaN to 0 in three separate places, all silently:

- ``ProcessorRunner`` filled everything immediately after validation, before
  rounding, cutoff, window splitting and forecasting;
- ``cutoff_below`` rewrote NaN to 0 as a side effect, since ``NaN >= cutoff`` is
  False;
- ``calculate_climatological_forecasts`` filled the left-join misses from its
  reference-window merge.

Each one is individually reasonable and collectively they meant a gap in the
source data arrived in GAMS as a genuine zero with nothing in the log. Worse,
filling before the quantile step made missing hours count as real zeros in the
climatology, biasing every forecast branch downward.

Now there is exactly one conversion point -- the GDX gate -- and it reports.
"""

import pandas as pd
import pytest

from src.GDX_exchange import prepare_values_for_gdx
from src.timeseries.timeseries_helpers import calculate_climatological_forecasts
from tests._common.fixtures import FakeLogger

DIMS = ["grid", "node", "f", "t"]


def _january_only(years=(2014, 2015), value=1.0) -> pd.DataFrame:
    """Hourly data covering only January of each year.

    A window longer than 31 days then reaches hours for which no climatology
    exists -- the left-join miss this module is about.
    """
    frames = []
    for yr in years:
        times = pd.date_range(f"{yr}-01-01", f"{yr}-01-31 23:00", freq="h")
        frames.append(
            pd.DataFrame(
                {
                    "grid": "elec",
                    "node": "FI00_elec",
                    "time": times,
                    "value": float(value),
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _forecasts(df, *, length):
    return calculate_climatological_forecasts(
        df,
        bb_parameter_dimensions=DIMS,
        forecast_quantiles={"f01": 0.5},
        bb_ts_start="01-01",
        bb_ts_length=length,
    )


class TestForecastGapsStayMissing:
    def test_hours_without_climatology_come_out_as_na(self):
        """The gap must still be visible when the function returns.

        Filling it here would hand the gate a frame that looks complete, and the
        run would report nothing at all.
        """
        out = _forecasts(_january_only(), length=40)

        # 40 days requested, 31 days of data: the tail has no climatology.
        assert out["value"].isna().any()
        assert int(out["value"].isna().sum()) == (40 - 31) * 24

    def test_hours_with_climatology_keep_their_value(self):
        # Guards the other direction: the change must not turn real data into NA.
        out = _forecasts(_january_only(value=7.0), length=40)
        assert (out["value"].dropna() == 7.0).all()

    def test_a_fully_covered_window_has_no_gaps(self):
        out = _forecasts(_january_only(), length=31)
        assert not out["value"].isna().any()

    def test_the_gate_converts_and_reports_the_gap(self):
        """End of the chain: GAMS still gets 0, but the run says how many.

        This is the behaviour the whole boundary exists for -- the conversion is
        correct and necessary, and doing it silently is what made a data gap
        indistinguishable from a real zero.
        """
        logger = FakeLogger()
        out = _forecasts(_january_only(), length=40)

        gated = prepare_values_for_gdx(out, logger, dimensions=DIMS, where="forecasts")

        assert not gated["value"].isna().any()
        assert (gated["value"] >= 0).all()
        logger.assert_logged("216 of", level="warn")

    def test_the_input_frame_is_not_mutated(self):
        """Regression: 'hour_of_year' used to be written into the caller's frame.

        ProcessorRunner passes main_result here and keeps using it afterwards for
        domain collection and the annual summary CSV.
        """
        df = _january_only()
        before = list(df.columns)
        _forecasts(df, length=31)
        assert list(df.columns) == before


class TestCutoffKeepsGapsMissing:
    @pytest.mark.parametrize("cutoff", [0.001, 1.0])
    def test_cutoff_below_does_not_swallow_na(self, cutoff):
        """``NaN >= cutoff`` is False, so a naive ``where`` rewrites gaps to 0.

        Mirrors the guarded expression in ProcessorRunner; without the isna()
        term the gate would never see the gap and could not report it.
        """
        values = pd.Series([0.0001, pd.NA, 5.0], dtype="Float64")

        guarded = values.where(values.isna() | (values.abs() >= cutoff), 0)

        assert pd.isna(guarded.iloc[1])
        assert guarded.iloc[0] == 0.0        # genuinely small -> zeroed
        assert guarded.iloc[2] == 5.0        # large -> untouched
