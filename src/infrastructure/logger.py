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
        self._error_log: list[str] = []
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
                   add_empty_line_after: bool = False) -> None:
        """
        Log a formatted status message with emoji prefix, and print it.

        If print_all_elapsed_times is True, prepends elapsed time since iteration start.

        Every message goes to the screen. There used to be a `print_to_screen`
        flag that could send one to `messages` alone: only one caller ever used
        it, for a separator, and a flag that can hide a warning from the person
        watching the build is worth more as a removed option than as an unused
        one. A caller that wants a line in the log file and not on the screen
        appends to `messages` itself, which says so plainly.

        Parameters:
            message (str): The message to log.
            level (str): Status level — info, warn, error, run, done, skip, none.
            section_start_length (int): If > 0, format as a section header padded to this length.
            add_empty_line_before (bool): Print a blank line before the message.
            add_empty_line_after (bool): Print a blank line after the message.
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

        if level in ("warn", "error"):
            self._warning_log.append(formatted)
        if level == "error":
            self._error_log.append(formatted)

        print(formatted)

    @property
    def warnings(self) -> list[str]:
        """Return a copy of all warn/error messages accumulated this iteration."""
        return list(self._warning_log)

    @property
    def has_errors(self) -> bool:
        """True if any error-level message has been logged this iteration."""
        return bool(self._error_log)

    @property
    def error_count(self) -> int:
        """Number of error-level messages logged so far this iteration."""
        return len(self._error_log)
