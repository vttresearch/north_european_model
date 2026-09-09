"""
profile_build.py -- cProfile one whole run of build_input_data.main.

Usage:
    python tools/profile_build.py <input_folder> <config_file> [--output-root DIR]
                                  [--top N] [--save FILE] [--warm]

Example:
    python tools/profile_build.py src_files config_OT2030.ini

Exit code is 0 when the build reported no errors and 1 otherwise, so it can gate
a loop.

By default the build writes into a fresh temporary folder, which is what makes
the measurement a *cold* one: the cache lives inside the output folder, so an
empty folder forces every phase to run without editing `force_full_rerun` in a
shipped config and without touching an output folder you already built. Pass
--output-root to profile into a folder of your choosing, and --warm to profile a
second run into the same folder, which is how you measure the cache path rather
than the build path.

What it reports
---------------
Three views of the same profile, in order:
    1. Phases      -- cumulative time inside each pipeline's run(), which is the
                      only breakdown the build itself does not print.
    2. Watchlist   -- named symbols known to be expensive here, so that a
                      regression in one of them is visible without reading the
                      whole table.
    3. Top N       -- by own time, then by cumulative. Own time is the one to
                      trust: cumulative only means anything while calls nest,
                      and the summing described below breaks that nesting.

What it sees, and what it does not
----------------------------------
cProfile counts function calls, and its overhead falls on call-heavy code far
more than on code that spends its time inside one numpy or pandas call. Reading
a CSV is one call and a lot of time; a groupby-apply is many calls and a lot of
time, and only the second is inflated. **So this ranks hotspots, it does not
size them.** A change that looks like it removes 30% here is confirmed by wall
clock, not by comparing two profiles.

It sees the main thread of one process. Anything a C extension does internally
is one entry, and time spent waiting on disk is charged to whatever asked.

**Do not read these numbers out of `pstats` directly.** `ProcessorRunner` loads
every processor through `importlib.util.spec_from_file_location`, so a class
serving several specs is executed from a fresh module object each time --
`VRE_PECD` three times, once per PECD folder. Those copies produce distinct code
objects carrying the same `(file, line, name)` label, and `pstats` keys on that
label and **overwrites**, keeping one of the three. A run profiled through
`pstats` reports `VRE_PECD.process` as one call rather than three, so every VRE
number comes out at a third of the truth. This tool therefore aggregates
`Profile.getstats()` itself, summing by label. The `.prof` written by --save is
the raw dump and still has the flaw; it is there for snakeviz, not for reading
totals off.

`build_input_data._check_dependencies` requires GAMS on PATH and an importable
`gams.transfer`, so a profiled run needs the full environment even though it
never launches GAMS itself.

The wall-clock line printed at the end is the profiled wall clock, i.e. it
includes the overhead described above. It is there to be compared against other
runs of this tool, not against a normal build.
"""

import argparse
import cProfile
import pstats
import sys
import tempfile
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import build_input_data  # noqa: E402  (needs the path above)


#: Entry points of the phases build_input_data.main runs, in the order it runs
#: them. Matched on (file stem, function name) because several classes have a
#: method called run() and the qualified name is not in the profile keys.
PHASES: tuple[tuple[str, str, str], ...] = (
    ("cache_manager",        "run", "CacheManager.run"),
    ("source_data_pipeline", "run", "SourceDataPipeline.run"),
    ("timeseries_pipeline",  "run", "TimeseriesPipeline.run"),
    ("bb_excel_pipeline",    "run", "BBExcelPipeline.run"),
)

#: Symbols worth watching by name. These are the ones a reader of the profile
#: goes looking for anyway: the per-processor entry point, the helpers every
#: processor pays for, and the GDX writer.
WATCHLIST: tuple[str, ...] = (
    "run_processor",
    "_read_and_compile_input_CSVs",
    "_describe_file",
    "order_timeseries_for_labelling",
    "find_time_axis_defects",
    "split_timeseries_to_climate_windows",
    "calculate_climatological_forecasts",
    "write_climate_window_GDX_files",
    "write_df_to_gdx",
    "prepare_values_for_gdx",
    "read_excel_to_parquet",
)


def _rows_for(profiler: cProfile.Profile):
    """(filename, lineno, funcname) -> (calls, tottime, cumtime), summed.

    Summed rather than read off `pstats`, which keeps only one of several code
    objects sharing a label -- see "What it sees" above. A built-in has no code
    object; it is labelled by its repr so it still appears in the tables.
    """
    rows: dict[tuple, list] = {}
    for entry in profiler.getstats():
        code = entry.code
        if isinstance(code, str):
            key = ("~", 0, code)
        else:
            key = (code.co_filename, code.co_firstlineno, code.co_name)
        row = rows.setdefault(key, [0, 0.0, 0.0])
        row[0] += entry.callcount
        row[1] += entry.inlinetime
        row[2] += entry.totaltime
    return {key: tuple(value) for key, value in rows.items()}


def _report_phases(stats: pstats.Stats) -> None:
    """Cumulative time per phase, read from pstats rather than the summed rows.

    Cumulative time only means anything while calls nest, and summing raw
    `getstats` entries by label breaks that -- a caller can come out smaller than
    its callee. The pipeline `run()` methods are each imported once, so their
    labels do not collide and pstats is right about them; it is only the
    dynamically re-imported processor modules it gets wrong.
    """
    print("\nPhases (cumulative seconds inside each run())")
    print("-" * 60)
    for stem, funcname, label in PHASES:
        total = sum(
            value[3]
            for (filename, _lineno, name), value in stats.stats.items()
            if name == funcname and Path(filename).stem == stem
        )
        if total:
            print(f"  {label:<28} {total:9.2f}")
        else:
            print(f"  {label:<28} {'not called':>9}")


def _report_watchlist(rows) -> None:
    print("\nWatchlist (calls / total / cumulative seconds)")
    print("-" * 60)
    found = False
    for wanted in WATCHLIST:
        matches = [
            (calls, tot, cum)
            for (_filename, _lineno, name), (calls, tot, cum) in rows.items()
            if name == wanted
        ]
        if not matches:
            continue
        found = True
        calls = sum(m[0] for m in matches)
        tot = sum(m[1] for m in matches)
        cum = sum(m[2] for m in matches)
        print(f"  {wanted:<38} {calls:>7} {tot:8.2f} {cum:9.2f}")
    if not found:
        print("  nothing on the watchlist was called")


def _report_top(rows, top: int) -> None:
    """Ranked twice. Own time is the number to trust -- see _report_phases for
    why the cumulative column is only indicative here."""
    for index, title in ((1, "own"), (2, "cumulative (indicative)")):
        print(f"\nTop {top} by {title} time (calls / own / cumulative)")
        print("-" * 92)
        ranked = sorted(rows.items(), key=lambda item: item[1][index], reverse=True)
        for (filename, lineno, name), (calls, tot, cum) in ranked[:top]:
            where = f"{Path(filename).name}:{lineno}({name})" if filename != "~" else name
            print(f"  {calls:>10} {tot:8.2f} {cum:9.2f}  {where}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="cProfile one whole run of build_input_data.main.",
    )
    parser.add_argument("input_folder", help="e.g. src_files")
    parser.add_argument("config_file", help="e.g. config_OT2030.ini, relative to input_folder")
    parser.add_argument(
        "--output-root",
        default=None,
        help="where the per-scenario output folders go. Default: a fresh temporary "
             "folder, which forces a cold build.",
    )
    parser.add_argument("--top", type=int, default=30, help="how many rows per table (default 30)")
    parser.add_argument("--save", default=None, help="write the raw .prof here")
    parser.add_argument(
        "--warm",
        action="store_true",
        help="run once to fill the cache, then profile a second run into the same "
             "folder. Measures the cache path instead of the build path.",
    )
    args = parser.parse_args(argv)

    input_folder = Path(args.input_folder)
    config_file = Path(input_folder, args.config_file)
    if not config_file.is_file():
        print(f"No such config file: {config_file}")
        return 1

    temporary = None
    if args.output_root:
        output_root = Path(args.output_root)
        output_root.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.mkdtemp(prefix="profile_build_")
        output_root = Path(temporary)

    print(f"Profiling: {config_file}")
    print(f"Output root: {output_root}")
    if args.warm:
        print("Warm run: building once unprofiled to fill the cache...")
        build_input_data.main(input_folder, config_file, output_root=output_root)
        print("Cache filled. Profiling the second run.")

    profiler = cProfile.Profile()
    started = time.time()
    profiler.enable()
    result = build_input_data.main(input_folder, config_file, output_root=output_root)
    profiler.disable()
    elapsed = time.time() - started

    rows = _rows_for(profiler)

    _report_phases(pstats.Stats(profiler))
    _report_watchlist(rows)
    _report_top(rows, args.top)

    print(f"\nProfiled wall clock: {elapsed / 60:.0f} min {elapsed % 60:.1f} sec "
          f"(includes profiler overhead)")
    if temporary:
        print(f"Output left in: {temporary}")

    if args.save:
        pstats.Stats(profiler).dump_stats(args.save)
        print(f"Raw profile written to: {args.save} "
              f"(for snakeviz; under-counts re-imported modules, see the docstring)")

    return 1 if result else 0


if __name__ == "__main__":
    sys.exit(main())
