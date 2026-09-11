"""
build_input_summary.py -- read one built input-data folder as a report.

A thin wrapper around tools/input_data_summary.py, which holds the whole
implementation and documents itself in its own module docstring -- what it
shows, what it cannot see, and what the exit codes mean. Read that file, not
this one.

This file exists so that the two things anyone does with this repository are one
command each, side by side: build a folder, then read what is in it. Reaching
into tools/ for the second one made it look like a diagnostic for whoever
maintains the pipeline, which is the opposite of who it is for.

Usage:
    python build_input_summary.py <built_folder> [--zones]
        [--no-neighbours] [--out-subdir NAME]

Examples:
    python build_input_summary.py input_ObservedTrends_2030
    python build_input_summary.py input_tyndp2024_NationalTrends_2040 --zones

The built folder is resolved against the working directory, the way
build_input_data.py resolves its input folder. Exit code is the tool's: 0 when
report.md was written -- including when the timeseries sections were skipped for
want of a GAMS install or a GDX file -- 1 when inputData.xlsx is there but
unreadable, and 2 when the folder or the workbook is missing.
"""
import sys
from pathlib import Path

from src.utils import force_utf8_output

#: tools/ is not a package and is not on the path when this file is the entry
#: point. Added here rather than turned into one: the scripts in there are run
#: directly far more often than they are imported, and a package would make
#: `python tools/input_data_summary.py ...` the odd spelling instead of this.
_TOOLS = Path(__file__).resolve().parent / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

import input_data_summary  # noqa: E402 -- needs the sys.path line above


def main(argv=None) -> int:
    """Write the report for one built folder. Arguments are the tool's own."""
    # Matches build_input_data.py: on Windows a redirected stream falls back to
    # the locale encoding, so `> summary.txt` can kill a run on a path this
    # cannot control. Nothing the tool itself prints is non-ASCII.
    force_utf8_output()
    return input_data_summary.main(argv)


if __name__ == "__main__":
    sys.exit(main())
