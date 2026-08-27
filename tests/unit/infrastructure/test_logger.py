"""Tests for ``src/infrastructure/logger.py``.

The logger is not merely reporting: ``build_input_data.py`` decides whether a
phase succeeded by comparing ``logger.error_count`` before and after it
(:146, :153, :192).  The counting semantics are therefore load-bearing, and are
pinned here.  Message *formatting* is not pinned -- it is free to change.
"""

import pytest

from src.infrastructure.logger import IterationLogger


def _logger(**kwargs) -> IterationLogger:
    kwargs.setdefault("print_all_elapsed_times", False)
    return IterationLogger(**kwargs)


def _log(logger, message, level="none"):
    """Log a line. pytest captures the print; the assertions read `messages`."""
    logger.log_status(message, level=level)


class TestErrorAndWarningCounting:
    def test_a_fresh_logger_is_clean(self):
        logger = _logger()
        assert logger.error_count == 0
        assert logger.has_errors is False
        assert logger.warnings == []

    @pytest.mark.parametrize("level", ["info", "run", "done", "skip", "none"])
    def test_ordinary_levels_count_as_neither(self, level):
        logger = _logger()
        _log(logger, "msg", level)
        assert logger.error_count == 0
        assert logger.warnings == []

    def test_warn_counts_as_a_warning_but_not_an_error(self):
        logger = _logger()
        _log(logger, "careful", "warn")
        assert len(logger.warnings) == 1
        assert logger.error_count == 0
        assert logger.has_errors is False

    def test_error_counts_as_both(self):
        # An error is also a warning (logger.py:75-78), so the end-of-run
        # warning summary shows errors too.
        logger = _logger()
        _log(logger, "broken", "error")
        assert logger.error_count == 1
        assert logger.has_errors is True
        assert len(logger.warnings) == 1

    def test_errors_accumulate(self):
        # build_input_data compares the count before and after a phase, so the
        # counter must be monotonic rather than a boolean.
        logger = _logger()
        for i in range(3):
            _log(logger, f"broken {i}", "error")
        assert logger.error_count == 3

    def test_an_unknown_level_is_treated_as_ordinary(self):
        # Robustness: a typo'd level must not silently become an error and fail
        # a phase that actually succeeded.
        logger = _logger()
        _log(logger, "msg", "wrn")
        assert logger.error_count == 0
        assert logger.warnings == []


class TestMessageAccumulation:
    def test_every_message_is_recorded_regardless_of_level(self):
        logger = _logger()
        for level in ("info", "warn", "error", "none"):
            _log(logger, f"msg {level}", level)
        assert len(logger.messages) == 4

    def test_the_message_text_survives_formatting(self):
        # Formatting itself is not pinned; only that the text is still findable,
        # which is what the summary.log and the test assertions rely on.
        logger = _logger()
        _log(logger, "capacity missing for FI_elec", "warn")
        assert any("capacity missing for FI_elec" in m for m in logger.messages)

    def test_warnings_returns_a_copy(self):
        # Callers iterate and re-print this list; handing out the internal list
        # would let a caller corrupt the log.
        logger = _logger()
        _log(logger, "careful", "warn")
        logger.warnings.append("injected")
        assert len(logger.warnings) == 1


class TestElapsedTime:
    def test_returns_minutes_and_seconds(self, monkeypatch):
        import src.infrastructure.logger as logger_module

        clock = iter([1000.0, 1125.5])
        monkeypatch.setattr(logger_module.time, "time", lambda: next(clock))

        logger = _logger()                      # consumes 1000.0 as start_time
        minutes, seconds = logger.elapsed_time()  # consumes 1125.5

        assert minutes == 2
        assert seconds == pytest.approx(5.5)

    def test_accepts_an_explicit_start_time(self, monkeypatch):
        import src.infrastructure.logger as logger_module

        monkeypatch.setattr(logger_module.time, "time", lambda: 500.0)
        logger = _logger()
        assert logger.elapsed_time(start_time=440.0) == (1, 0.0)

    def test_elapsed_prefix_is_opt_in(self, monkeypatch):
        import src.infrastructure.logger as logger_module

        monkeypatch.setattr(logger_module.time, "time", lambda: 0.0)

        quiet = _logger(print_all_elapsed_times=False)
        _log(quiet, "msg")
        assert "min" not in quiet.messages[0]

        loud = IterationLogger(print_all_elapsed_times=True)
        loud.log_status("msg")
        assert "min" in loud.messages[0]


class TestIsolationBetweenIterations:
    def test_two_loggers_do_not_share_state(self):
        """Each scenario iteration constructs its own logger.

        If state leaked, one scenario's errors would fail the next scenario's
        phase checks -- and the build would report failures against the wrong
        output folder.
        """
        first = _logger()
        _log(first, "broken", "error")

        second = _logger()
        assert second.error_count == 0
        assert second.messages == []
        assert first.error_count == 1
