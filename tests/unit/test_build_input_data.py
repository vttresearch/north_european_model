"""Tests for the top-level entry module ``build_input_data.py``.

``_patch_gams_file_content`` is a pure string transform with a fixed contract,
so exact values are pinned here -- one of the five cases where pinning is
correct (see tests/README.md).  The arithmetic it performs *is* the contract:
``dataLength = bb_timeseries_length * 24``, ``t_max`` rounded up to the next
thousand, ``forecastNumber = len(quantiles)``.

The three GAMS templates carry in-file warnings reading "do not edit ... unless
updating also _patch_gams_file_content() in build_input_data.py".  The
``TestRealTemplatesStillMatch`` class is the automated half of that warning: it
fails when a template drifts away from the anchors the patcher searches for,
which would otherwise turn every patch into a silent no-op.
"""

import math

import pytest

import build_input_data
from build_input_data import _patch_gams_file_content
from tests._common.fixtures import FakeLogger, make_config

#: 24 * 7 * 65, mirroring mSettings('schedule', 't_horizon') in scheduleInit.gms.
DEF_T_HORIZON = 10920


class TestScheduleInit:
    ANCHOR = "    mSettings('schedule', 'dataLength') =  8760;\n"

    @pytest.mark.parametrize(
        "length, expected",
        [(1, 24), (2, 48), (365, 8760), (1825, 43800)],
        ids=["1d", "2d", "1y", "5y"],
    )
    def test_data_length_is_days_times_24(self, length, expected):
        out = _patch_gams_file_content(
            "scheduleInit.gms", self.ANCHOR, make_config(bb_timeseries_length=length)
        )
        assert f"'dataLength') =  {expected};" in out
        # The template value must be gone, not merely accompanied.
        assert "=  8760;" not in out or expected == 8760

    def test_forecast_weights_replace_the_probability_block(self):
        content = (
            "    // NOTE: do not edit the lines below in the git version\n"
            "    // unless updating also _patch_gams_file_content() in build_input_data.py\n"
            "    // No restrictions for local versions.\n"
            "    p_mfProbability('schedule', 'f01') = 0.6;\n"
            "    p_mfProbability('schedule', 'f02') = 0.2;\n"
            "    p_mfProbability('schedule', 'f03') = 0.2;\n"
        )
        config = make_config(
            forecast_quantiles={"f01": 0.5, "f02": 0.9},
            forecast_weights={"f01": 0.75, "f02": 0.25},
        )

        out = _patch_gams_file_content("scheduleInit.gms", content, config)

        assert "p_mfProbability('schedule', 'f01') = 0.75;" in out
        assert "p_mfProbability('schedule', 'f02') = 0.25;" in out
        # The old three-branch block must be replaced wholesale, not appended to;
        # a leftover f03 would be weighted into the model as a real branch.
        assert "f03" not in out

    def test_the_explanatory_comment_block_survives(self):
        # It is the anchor for the next run's regex. Consuming it would make the
        # patch work exactly once.
        content = (
            "    // NOTE: do not edit the lines below in the git version\n"
            "    // unless updating also _patch_gams_file_content() in build_input_data.py\n"
            "    // No restrictions for local versions.\n"
            "    p_mfProbability('schedule', 'f01') = 0.6;\n"
        )
        out = _patch_gams_file_content("scheduleInit.gms", content, make_config())
        assert "// NOTE: do not edit the lines below" in out

        twice = _patch_gams_file_content("scheduleInit.gms", out, make_config())
        assert twice == out  # idempotent


class TestTimeAndSamples:
    ANCHOR = "    t \"Model time steps\" / t000000 * t020000 /\n"

    @pytest.mark.parametrize(
        "length", [1, 2, 30, 365, 1825], ids=lambda v: f"{v}d"
    )
    def test_t_max_is_rounded_up_to_the_next_thousand(self, length):
        expected = math.ceil((length * 24 + DEF_T_HORIZON) / 1000) * 1000

        out = _patch_gams_file_content(
            "timeAndSamples.inc", self.ANCHOR, make_config(bb_timeseries_length=length)
        )

        assert f"t000000 * t{expected:06d}" in out
        # Six digits, zero padded: GAMS set elements are compared as text, so
        # t11000 and t011000 are different symbols.
        assert len(f"{expected:06d}") == 6

    def test_t_max_leaves_room_for_the_horizon_beyond_the_data(self):
        # Property rather than a pinned number: the t index must extend past the
        # modelled data by at least one horizon, or the last solve runs off the end.
        length = 365
        out = _patch_gams_file_content(
            "timeAndSamples.inc", self.ANCHOR, make_config(bb_timeseries_length=length)
        )
        t_max = int(out.split("t000000 * t")[1][:6])
        assert t_max >= length * 24 + DEF_T_HORIZON

    @pytest.mark.parametrize(
        "quantiles, expected",
        [
            ({"f01": 0.5, "f02": 0.1, "f03": 0.9}, "f00 * f03"),
            ({"f01": 0.5}, "f00 * f01"),
            ({}, "f00 * f01"),
        ],
        ids=["three", "one", "none"],
    )
    def test_forecast_range_is_floored_at_f01(self, quantiles, expected):
        # "f00 * f00" is not a valid GAMS range, so an empty quantile set must
        # still declare f01. Backbone filters active forecasts at runtime.
        content = '    f "Forecasts for the short term" / f00 * f03 /\n'
        out = _patch_gams_file_content(
            "timeAndSamples.inc", content, make_config(forecast_quantiles=quantiles)
        )
        assert expected in out


class TestChangesInc:
    @pytest.mark.parametrize(
        "quantiles, expected",
        [({"f01": 0.5, "f02": 0.1, "f03": 0.9}, 3), ({"f01": 0.5}, 1), ({}, 0)],
        ids=["three", "one", "none"],
    )
    def test_forecast_number_counts_branches_beside_f00(self, quantiles, expected):
        # mSettings(mType, 'forecasts') counts the branches beside the realization
        # f00, so it equals the quantile count -- not one more. 0 is a deterministic
        # run using f00 alone. See mSettings in ../docs/dictionary.md.
        # 9 is not any of the expected values, so a patch that silently failed to
        # match cannot leave the anchor's own number behind and still pass.
        content = "$if not set forecasts $evalglobal forecastNumber 9\n"
        out = _patch_gams_file_content(
            "changes.inc", content, make_config(forecast_quantiles=quantiles)
        )
        assert f"forecastNumber {expected}" in out

    def test_the_user_override_line_is_left_alone(self):
        content = (
            "$if not set forecasts $evalglobal forecastNumber 4\n"
            "$if set forecasts $evalglobal forecastNumber %forecasts%\n"
        )
        out = _patch_gams_file_content("changes.inc", content, make_config())
        assert "$if set forecasts $evalglobal forecastNumber %forecasts%" in out


class TestUnknownFiles:
    @pytest.mark.parametrize(
        "filename",
        ["1_options.gms", "modelsInit.gms", "changes_loop.inc", "remove_constraints.inc"],
    )
    def test_other_gams_files_are_copied_verbatim(self, filename):
        content = "mSettings('schedule', 'dataLength') =  8760;\nt000000 * t020000\n"
        assert _patch_gams_file_content(filename, content, make_config()) == content


class TestRealTemplatesStillMatch:
    """The automated half of the "do not edit" warnings in the templates.

    If a template is edited so that an anchor no longer matches, the patch
    becomes a silent no-op: the build still succeeds and the model runs with
    8760 hours of data regardless of what the config asked for. These tests turn
    that into a test failure.
    """

    @pytest.fixture
    def templates(self, src_files_dir):
        return src_files_dir / "GAMS_files"

    def test_schedule_init_data_length_anchor_is_present(self, templates):
        content = (templates / "scheduleInit.gms").read_text(encoding="utf-8")
        out = _patch_gams_file_content(
            "scheduleInit.gms", content, make_config(bb_timeseries_length=2)
        )
        assert out != content, "dataLength anchor no longer matches scheduleInit.gms"
        assert "'dataLength') =  48;" in out

    def test_schedule_init_probability_block_anchor_is_present(self, templates):
        content = (templates / "scheduleInit.gms").read_text(encoding="utf-8")
        config = make_config(
            forecast_quantiles={"f01": 0.5},
            forecast_weights={"f01": 1.0},
        )
        out = _patch_gams_file_content("scheduleInit.gms", content, config)
        assert "p_mfProbability('schedule', 'f01') = 1;" in out
        assert "p_mfProbability('schedule', 'f02')" not in out

    def test_time_and_samples_anchors_are_present(self, templates):
        content = (templates / "timeAndSamples.inc").read_text(encoding="utf-8")
        out = _patch_gams_file_content(
            "timeAndSamples.inc", content, make_config(bb_timeseries_length=2)
        )
        expected = math.ceil((2 * 24 + DEF_T_HORIZON) / 1000) * 1000
        assert f"t000000 * t{expected:06d}" in out
        assert "f00 * f03" in out  # three default quantiles

    def test_changes_inc_anchor_is_present(self, templates):
        # One quantile, so the expected number differs from the template's own
        # default: a patch that no longer matches cannot pass by leaving the
        # template untouched.
        content = (templates / "changes.inc").read_text(encoding="utf-8")
        out = _patch_gams_file_content(
            "changes.inc", content, make_config(forecast_quantiles={"f01": 0.5})
        )
        assert out != content, "forecastNumber anchor no longer matches changes.inc"
        assert "$if not set forecasts $evalglobal forecastNumber 1" in out


class TestGamsSettingsSummary:
    """The Finalizing block reports what the patcher wrote into the GAMS files.

    The whole value of that block is that it agrees with the files.  A summary
    free to drift away from them is worse than no summary, so these tests read
    the reported text and the patched templates side by side.
    """

    @pytest.fixture
    def templates(self, src_files_dir):
        return src_files_dir / "GAMS_files"

    @staticmethod
    def _report(config):
        logger = FakeLogger()
        settings = build_input_data._derive_gams_settings(config)
        build_input_data._log_gams_settings(logger, settings)
        return settings, logger

    def test_reported_values_are_the_ones_written_to_the_templates(self, templates):
        # Every value deliberately differs from the template's own default, so a
        # substitution that stopped happening cannot pass by accident.
        config = make_config(
            bb_timeseries_length=40,
            forecast_quantiles={"f01": 0.5, "f02": 0.9},
            forecast_weights={"f01": 0.7, "f02": 0.3},
        )
        settings, logger = self._report(config)
        reported = "\n".join(logger.messages)

        def patched(name):
            return _patch_gams_file_content(
                name, (templates / name).read_text(encoding="utf-8"), config
            )

        schedule = patched("scheduleInit.gms")
        times = patched("timeAndSamples.inc")
        changes = patched("changes.inc")

        assert f"'dataLength') =  {settings['data_length']};" in schedule
        assert str(settings["data_length"]) in reported

        t_range = f"t000000 * t{settings['t_max']:06d}"
        assert t_range in times
        assert t_range in reported

        f_range = f"f00 * {settings['last_f']}"
        assert f_range in times
        assert f_range in reported

        forecasts = f"forecastNumber {settings['forecast_number']}"
        assert forecasts in changes
        assert forecasts in reported

        for label, weight in settings["weights"].items():
            assert f"p_mfProbability('schedule', '{label}') = {weight:g};" in schedule
            assert f"{label} {weight:g}" in reported

    def test_a_deterministic_run_reports_no_probabilities(self):
        _, logger = self._report(make_config(forecast_quantiles={}, forecast_weights={}))
        reported = "\n".join(logger.messages)
        assert "forecastNumber 0" in reported
        assert "deterministic" in reported

    def test_the_block_stays_short_and_carries_no_warnings(self):
        # A lead-in and one row per patched symbol.  The block is a reassurance
        # on a normal build, so growing it is a decision, not an accident.
        _, logger = self._report(make_config())
        assert len(logger.records) == 5
        assert all(level == "none" for level, _ in logger.records)


class TestCheckDependencies:
    def test_reports_a_missing_gams_executable(self, monkeypatch):
        monkeypatch.setattr(build_input_data.shutil, "which", lambda name: None)
        with pytest.raises(RuntimeError) as excinfo:
            build_input_data._check_dependencies()
        # Substring, not the whole message: the wording is free to change.
        assert "gams" in str(excinfo.value).lower()
