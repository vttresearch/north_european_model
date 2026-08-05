"""Log output must survive being redirected to a file or a pipe.

On Windows, Python writes to a *console* as UTF-8 but falls back to the locale
encoding (cp1252 here) the moment stdout is redirected. ``IterationLogger``
prefixes every line with a non-ASCII status marker, so

    python build_input_data.py src_files config_OT2030.ini > build.log

raised ``UnicodeEncodeError`` on the first log line and killed the run before
any work started. Interactively it was fine, which is why it went unnoticed.

The markers are deliberate -- warnings get missed without them, to the point of
being printed twice -- so the encoding is what gives way, not the message.

The subprocess test is the one that matters: it reproduces the original crash
end to end, which no amount of in-process mocking can.
"""

import subprocess
import sys
import textwrap

import pytest

from src.utils import force_utf8_output

# The exact markers IterationLogger emits (logger.py:48-56).
STATUS_MARKERS = ["✓", "⚠️", "❌", "⚡", "\U0001f3af", "⏩"]


class TestForceUtf8Output:
    def test_sets_stdout_and_stderr_to_utf8(self, capsys):
        force_utf8_output()
        # capsys replaces the streams, so assert the call is harmless rather
        # than asserting an encoding capsys does not really have.
        print("".join(STATUS_MARKERS))
        assert "✓" in capsys.readouterr().out

    def test_is_idempotent(self):
        force_utf8_output()
        force_utf8_output()

    def test_survives_a_stream_without_reconfigure(self, monkeypatch):
        # Some IDE and CI wrappers replace sys.stdout with a plain object.
        class Bare:
            def write(self, s):
                return len(s)

        monkeypatch.setattr(sys, "stdout", Bare())
        force_utf8_output()  # must not raise

    def test_survives_a_stream_that_refuses_reconfiguration(self, monkeypatch):
        class Detached:
            def reconfigure(self, **kwargs):
                raise ValueError("underlying buffer has been detached")

        monkeypatch.setattr(sys, "stdout", Detached())
        force_utf8_output()  # must not raise


class TestRedirectedOutputDoesNotCrash:
    """End-to-end: run a child process with stdout redirected to a pipe."""

    def _run(self, body: str, repo_root, env_extra=None) -> subprocess.CompletedProcess:
        import os

        env = dict(os.environ)
        # Reproduce the original conditions: no PYTHONIOENCODING escape hatch,
        # and a legacy Windows codepage as the locale encoding.
        env.pop("PYTHONIOENCODING", None)
        env.update(env_extra or {})
        return subprocess.run(
            [sys.executable, "-c", textwrap.dedent(body)],
            capture_output=True,       # redirected, so not a console
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=repo_root,
            env=env,
        )

    def test_the_status_markers_reproduce_the_original_crash_without_the_fix(
        self, repo_root
    ):
        """Negative control: prove the failure this guards against is real."""
        result = self._run(
            """
            import sys
            sys.stdout.reconfigure(encoding="cp1252")
            print("\\u2713 info line")
            """,
            repo_root,
        )
        assert result.returncode != 0
        assert "UnicodeEncodeError" in result.stderr

    def test_force_utf8_output_prevents_it(self, repo_root):
        result = self._run(
            """
            import sys
            sys.path.insert(0, ".")
            sys.stdout.reconfigure(encoding="cp1252")
            from src.utils import force_utf8_output
            force_utf8_output()
            print("\\u2713 info line")
            """,
            repo_root,
        )
        assert result.returncode == 0, result.stderr
        assert "✓ info line" in result.stdout

    @pytest.mark.parametrize("marker", STATUS_MARKERS, ids=lambda m: f"U+{ord(m[0]):04X}")
    def test_every_logger_status_marker_survives_redirection(self, repo_root, marker):
        result = self._run(
            f"""
            import sys
            sys.path.insert(0, ".")
            sys.stdout.reconfigure(encoding="cp1252")
            from src.utils import force_utf8_output
            force_utf8_output()
            print({marker!r})
            """,
            repo_root,
        )
        assert result.returncode == 0, result.stderr

    def test_the_real_logger_writes_every_level_through_a_redirected_stream(
        self, repo_root
    ):
        """The actual failure path: IterationLogger, not a bare print."""
        result = self._run(
            """
            import sys
            sys.path.insert(0, ".")
            sys.stdout.reconfigure(encoding="cp1252")
            from src.utils import force_utf8_output
            from src.infrastructure.logger import IterationLogger
            force_utf8_output()
            logger = IterationLogger(print_all_elapsed_times=False)
            for level in ("info", "warn", "error", "run", "done", "skip", "none"):
                logger.log_status(f"message at {level}", level=level)
            print("REACHED THE END")
            """,
            repo_root,
        )
        assert result.returncode == 0, result.stderr
        assert "REACHED THE END" in result.stdout


class TestEntryPointCallsIt:
    def test_main_forces_utf8_before_anything_logs(self, repo_root, monkeypatch):
        """Placement matters: after the first log_status it would be too late.

        Asserted on the source rather than by running main(), which needs GAMS
        and a full input folder.
        """
        source = (repo_root / "build_input_data.py").read_text(encoding="utf-8")
        body = source.split("def main(", 1)[1]

        assert "force_utf8_output()" in body, "main() must call force_utf8_output()"
        assert body.index("force_utf8_output()") < body.index("log_status"), (
            "force_utf8_output() must run before the first log_status call"
        )
