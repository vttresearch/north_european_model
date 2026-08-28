"""Tests for ``tests/_common/fixtures.py`` -- the config factory and logger double."""

import pytest

from tests._common.fixtures import DEFAULT_CONFIG, FakeLogger, make_config


class TestMakeConfig:
    def test_returns_every_key_load_config_produces(self):
        # Guards against a config key being added to config_reader.load_config
        # without DEFAULT_CONFIG following: route tests would then exercise a
        # config shape that never occurs in production.
        from src.infrastructure import config_reader  # noqa: F401  (import must work)

        config = make_config()
        assert set(config) == set(DEFAULT_CONFIG)

    def test_overrides_are_applied(self):
        assert make_config(country_codes=["DK"])["country_codes"] == ["DK"]

    def test_unknown_key_raises_rather_than_being_ignored(self):
        # A typo'd override that is silently accepted produces a test that passes
        # for the wrong reason -- the exact failure mode this suite exists to avoid.
        with pytest.raises(KeyError, match="contry_codes"):
            make_config(contry_codes=["FI"])

    def test_each_call_gets_an_independent_deep_copy(self):
        # config_reader._validate_timeseries_specs mutates the specs dict it is
        # handed (entry.setdefault at :155), so shared state would leak defaults
        # from one test into the next.
        first = make_config()
        first["timeseries_specs"]["leaked"] = {"processor_name": "x"}
        first["country_codes"].append("NO")

        second = make_config()
        assert second["timeseries_specs"] == {}
        assert "NO" not in second["country_codes"]


class TestFakeLogger:
    def test_records_level_and_message(self):
        logger = FakeLogger()
        logger.log_status("all good", level="info")
        logger.log_status("careful", level="warn")
        logger.log_status("broken", level="error")

        assert logger.messages == ["all good", "careful", "broken"]
        assert logger.warnings == ["careful", "broken"]
        assert logger.errors == ["broken"]
        assert logger.error_count == 1
        assert logger.has_errors is True

    def test_accepts_the_real_loggers_presentation_kwargs(self):
        # The point of **_ignored: IterationLogger.log_status carries formatting
        # kwargs that mean nothing to a test double. Adding another one upstream
        # must not require an edit here.
        logger = FakeLogger()
        logger.log_status(
            "section",
            level="info",
            section_start_length=80,
            add_empty_line_before=True,
            add_empty_line_after=True,
        )
        assert logger.messages == ["section"]

    def test_matches_the_real_loggers_warn_and_error_semantics(self):
        # logger.py:75-78 -- 'warn' and 'error' both count as warnings, only
        # 'error' counts as an error. Pinned because it IS the contract the
        # pipelines' success checks rely on (build_input_data.py:146,153).
        from src.infrastructure.logger import IterationLogger

        real = IterationLogger(print_all_elapsed_times=False)
        fake = FakeLogger()
        for level in ("info", "warn", "error", "run", "done", "skip", "none"):
            real.log_status(f"msg {level}", level=level)
            fake.log_status(f"msg {level}", level=level)

        assert len(fake.warnings) == len(real.warnings)
        assert fake.error_count == real.error_count
        assert fake.has_errors == real.has_errors

    @pytest.mark.parametrize("level", [None, "error"])
    def test_assert_logged_passes_on_a_match(self, level):
        logger = FakeLogger()
        logger.log_status("Unknown method 'frobnicate'", level="error")
        logger.assert_logged("Unknown method", level=level)

    def test_assert_logged_fails_when_absent_and_shows_the_log(self):
        # Negative control: the assertion must be capable of failing, and its
        # message must be diagnosable without a debugger.
        logger = FakeLogger()
        logger.log_status("something else entirely", level="info")
        with pytest.raises(AssertionError) as excinfo:
            logger.assert_logged("Unknown method")
        assert "something else entirely" in str(excinfo.value)

    def test_assert_logged_respects_the_level_filter(self):
        logger = FakeLogger()
        logger.log_status("capacity missing", level="warn")
        logger.assert_logged("capacity missing", level="warn")
        with pytest.raises(AssertionError):
            logger.assert_logged("capacity missing", level="error")

    def test_assert_no_errors_fails_only_on_errors(self):
        logger = FakeLogger()
        logger.log_status("just a warning", level="warn")
        logger.assert_no_errors()          # warnings are not errors
        with pytest.raises(AssertionError):
            logger.assert_clean()          # ...but they are not clean either

        logger.log_status("now broken", level="error")
        with pytest.raises(AssertionError, match="expected no errors"):
            logger.assert_no_errors()

    def test_assert_not_logged_reports_the_offending_lines(self):
        logger = FakeLogger()
        logger.log_status("deprecated key fueldata_files", level="error")
        with pytest.raises(AssertionError, match="fueldata_files"):
            logger.assert_not_logged("deprecated key")

    def test_empty_log_dump_is_still_informative(self):
        logger = FakeLogger()
        with pytest.raises(AssertionError, match="nothing was logged"):
            logger.assert_logged("anything")
