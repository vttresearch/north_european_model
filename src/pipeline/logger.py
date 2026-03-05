# src/pipeline/logger.py

import time


class IterationLogger:
    """Encapsulates per-iteration log state.

    Creating an instance starts a fresh elapsed-time clock and clears all
    accumulated messages and warnings. Pass the instance through all sub-pipelines
    so that every log_status() call writes directly into this object.
    """

    def __init__(self, *, print_all_elapsed_times: bool) -> None:
        self.print_all_elapsed_times = bool(print_all_elapsed_times)
        self.messages: list[str] = []
        self._warning_log: list[str] = []
        self.start_time: float = time.time()

    def elapsed_time(self, start_time: float = None) -> tuple[int, float]:
        """Return (minutes, seconds) elapsed since start_time (or iteration start if None)."""
        if start_time is None:
            start_time = self.start_time
        elapsed_seconds = time.time() - start_time
        minutes = int(elapsed_seconds // 60)
        seconds = round(float(elapsed_seconds % 60), 2)
        return minutes, seconds

    def log_status(self,
                   message: str,
                   level: str = "none",
                   section_start_length: int = 0,
                   add_empty_line_before: bool = False,
                   add_empty_line_after: bool = False,
                   print_to_screen: bool = True) -> None:
        """
        Log a formatted status message with emoji prefix and optional console print.

        If print_all_elapsed_times is True, prepends elapsed time since iteration start.

        Parameters:
            message (str): The message to log.
            level (str): Status level — info, warn, error, run, done, skip, none.
            section_start_length (int): If > 0, format as a section header padded to this length.
            add_empty_line_before (bool): Print a blank line before the message.
            add_empty_line_after (bool): Print a blank line after the message.
            print_to_screen (bool): Whether to print immediately to stdout.
        """
        prefix = {
            "info": "✓",
            "warn": "⚠️",
            "error": "❌",
            "run": "⚡",
            "done": "🎯",
            "skip": "⏩",
            "none": " "
        }.get(level, " ")

        elapsed_prefix = ""
        if self.print_all_elapsed_times:
            m, s = self.elapsed_time()
            elapsed_prefix = f"{m} min {s} sec: "

        start = "\n" if add_empty_line_before else ""
        end = "\n" if add_empty_line_after else ""

        if section_start_length > 0:
            base = f"{start}---  {prefix} {elapsed_prefix} {message}{end} "
            padding_length = max(section_start_length, len(base))
            formatted = base + "-" * (padding_length - len(base))
        else:
            formatted = f"{start}{prefix} {elapsed_prefix}{message}{end}"

        self.messages.append(formatted)

        if level in ("warn", "error", "skip"):
            self._warning_log.append(formatted)

        if print_to_screen:
            print(formatted)

    @property
    def warnings(self) -> list[str]:
        """Return a copy of all warn/error/skip messages accumulated this iteration."""
        return list(self._warning_log)
