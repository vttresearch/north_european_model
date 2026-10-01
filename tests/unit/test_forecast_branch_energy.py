"""tools/forecast_branch_energy.py -- the arithmetic a wrong answer would hide in.

The tool exists because a per-hour quantile is not a quantile of the energy a
branch carries, and its output is what quantiles get chosen from. A mistake here
does not crash: it prints a plausible ratio, and a branch is then built on it.
So the pieces pinned are the ones that decide the number -- how a window is
summed, where a branch sits among the years, which quantile reproduces a given
energy -- each on data whose answer is known without running the tool.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import forecast_branch_energy as energy  # noqa: E402


class TestWindowSums:
    def test_one_window_per_day_start(self):
        x = np.ones(24 * 10)
        assert energy.window_sums(x, 48).shape == (10,)

    def test_a_window_is_its_hours_summed(self):
        x = np.arange(24 * 4, dtype=float)
        out = energy.window_sums(x, 24)
        assert out[0] == x[:24].sum()
        assert out[2] == x[48:72].sum()

    def test_a_window_past_the_end_wraps_to_the_start(self):
        # Backbone circulates the data, so the last day's two-day window is the
        # last day plus the first.
        x = np.arange(24 * 4, dtype=float)
        assert energy.window_sums(x, 48)[3] == x[72:].sum() + x[:24].sum()

    def test_leading_axes_are_kept(self):
        x = np.ones((3, 5, 24 * 7))
        assert energy.window_sums(x, 24).shape == (3, 5, 7)

    def test_a_window_longer_than_the_data_is_the_whole_of_it(self):
        x = np.ones(24 * 3)
        assert energy.window_sums(x, 24 * 30).tolist() == [72.0, 72.0, 72.0]


class TestDescribeBranch:
    HOURS = 24 * 20

    def _years(self, levels):
        return np.array([np.full(self.HOURS, level) for level in levels], dtype=float)

    def test_a_branch_at_the_mean_is_one(self):
        realized = self._years([1.0, 2.0, 3.0])
        row = energy.describe_branch(np.full(self.HOURS, 2.0), realized, 48)
        assert row["ratio"] == pytest.approx(1.0)
        assert (row["low"], row["high"]) == pytest.approx((0.5, 1.5))

    def test_years_below_counts_the_years_with_less_than_the_branch(self):
        realized = self._years([1.0, 2.0, 3.0, 4.0])
        assert energy.describe_branch(np.full(self.HOURS, 2.5), realized, 48)["below"] == 0.5
        assert energy.describe_branch(np.full(self.HOURS, 0.5), realized, 48)["below"] == 0.0
        assert energy.describe_branch(np.full(self.HOURS, 9.0), realized, 48)["below"] == 1.0

    def test_demand_is_negative_and_more_of_it_is_above_one(self):
        realized = self._years([-1.0, -2.0, -3.0])
        row = energy.describe_branch(np.full(self.HOURS, -3.0), realized, 48)
        assert row["ratio"] == pytest.approx(1.5)
        # No year has more demand than the branch, so none is below it.
        assert row["below"] == 0.0


class TestEquivalentQuantiles:
    def test_a_series_that_is_a_level_per_year_gives_its_own_quantiles(self):
        # Every hour of a year is that year's level, so a per-hour quantile *is*
        # the quantile of the window energy and the two must agree.
        hours = 24 * 10
        levels = np.linspace(1.0, 2.0, 21)
        realized = np.array([np.full((1, hours), level) for level in levels])
        found = energy.equivalent_quantiles(realized, np.ones(1), 48)
        assert found["p10"][1] == pytest.approx(0.10, abs=0.011)
        assert found["p90"][1] == pytest.approx(0.90, abs=0.011)
        assert found["mean"][1] == pytest.approx(0.50, abs=0.011)

    def test_years_that_trade_hours_need_a_quantile_near_the_mean(self):
        # The case the tool is for. Each year is windy in a different half of the
        # window, so every year's energy is the same while the per-hour spread is
        # wide: the per-hour p10 carries far less than any year, and the quantile
        # that gives the realized p10 energy is the one that gives the mean.
        rng = np.random.default_rng(0)
        hours = 24 * 40
        realized = rng.permuted(
            np.tile(np.linspace(0.0, 1.0, hours), (30, 1)), axis=1
        )[:, None, :]
        found = energy.equivalent_quantiles(realized, np.ones(1), hours)
        ratio_p10, q_p10 = found["p10"]
        assert ratio_p10 == pytest.approx(1.0, abs=1e-9)
        assert q_p10 == pytest.approx(found["mean"][1], abs=0.011)
        assert 0.4 < q_p10 < 0.6

    def test_weights_decide_which_series_counts(self):
        hours = 24 * 5
        levels = np.linspace(1.0, 2.0, 11)
        flat = np.array([np.full(hours, 1.0) for _ in levels])
        varying = np.array([np.full(hours, level) for level in levels])
        realized = np.stack([flat, varying], axis=1)
        only_flat = energy.equivalent_quantiles(realized, np.array([1.0, 0.0]), 24)
        assert only_flat["p10"][0] == pytest.approx(1.0)
        only_varying = energy.equivalent_quantiles(realized, np.array([0.0, 1.0]), 24)
        assert only_varying["p10"][0] < 1.0


class TestReadSchedule:
    def _folder(self, tmp_path, text):
        (tmp_path / "scheduleInit.gms").write_text(text, encoding="utf-8")
        return tmp_path

    def test_reads_the_lengths_the_build_wrote(self, tmp_path):
        out = energy.read_schedule(self._folder(tmp_path, (
            "    mSettings('schedule', 'dataLength') =  8760;\n"
            "    mSettings('schedule', 'forecastLength') = 3576;\n"
            "    p_forecast('f02', 'forecastLength') = 6048;\n"
            "    p_forecast('f02', 'boundForecastEnds') = 2;\n"
            "    p_forecast('f04', 'forecastLength') = 120;\n"
        )))
        assert out == {"data_length": 8760, "default_length": 3576,
                       "lengths": {"f02": 6048, "f04": 120}}

    def test_reads_a_folder_built_before_forecast_branches(self, tmp_path):
        out = energy.read_schedule(self._folder(tmp_path, (
            "    mSettings('schedule', 'dataLength') =  8760;\n"
            "    mSettings('schedule', 't_forecastLengthUnchanging') = 3576;\n"
        )))
        assert out["default_length"] == 3576
        assert out["lengths"] == {}

    def test_the_real_template_is_readable(self, src_files_dir):
        out = energy.read_schedule(src_files_dir / "GAMS_files")
        assert out["data_length"] == 8760
        assert out["default_length"] > 0
