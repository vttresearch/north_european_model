"""What a timeseries processor was given, written down so the next run can compare.

A processor is rerun when its input changed or its own code changed, and this
module is the first half of that. The record is not a description of what the
processor needs -- a description is a second thing that drifts from the code.
It is the input itself: the source-data frames ProcessorRunner handed over,
narrowed to the columns the processor declared, the spec and config values it
was called with, and a manifest of the files it said it would read.

Because the recorded frame *is* the delivered frame, the two cannot disagree.
That is the whole point of narrowing the delivery in ProcessorRunner: a
`unitdata` edit that does not touch the columns a processor was given cannot
move its record, so a cost tweak stops rebuilding the weather.

Canonical, not merely serialised
--------------------------------
An unstable record is worse than none: it either reruns everything every build
or hides a real edit. So rows are sorted, columns are sorted by name, and every
cell becomes a string token by one rule.

Three rules earn their place:

- **`pd.NA` and `0` stay distinct.** These are SourceDataPipeline frames, where
  the distinction is live -- `method=replace` depends on it -- unlike the
  BBExcel and GDX boundaries where `0 = NA = "not set"`. `MISSING` is a token no
  cell can produce, rather than an empty string, which a cell can.
- **Floats round-trip.** `repr` rather than any formatting, so 0.1 does not
  become 0.09999999 on one side of a comparison and not the other.
- **`_`-prefixed provenance columns are dropped.** The loader adds them to carry
  the file and sheet a row came from, so keeping them would rerun every
  processor when a workbook is renamed and nothing else changed.
"""

from pathlib import Path

import pandas as pd

#: The token standing for a missing cell. A NUL rather than "" or "-",
#: because a workbook cell can hold either of those and the whole point is
#: that "not set" stays distinguishable from an empty string and from a zero.
MISSING = "\u0000"

#: How many differing values a message names before it starts counting. Matches
#: utils.summarise, which is what renders them.
DIFFERENCE_LIMIT = 3


def _cell(value) -> str:
    """One cell as a comparable token.

    A string for everything, so that rows sort without comparing an int against
    a NA against a str -- and so that the record reads as what it is when
    someone opens it to find out why a processor reran.
    """
    if pd.isna(value):
        return MISSING
    if isinstance(value, float):
        return repr(value)
    return str(value)


def frame_record(frame: pd.DataFrame) -> dict:
    """One delivered frame, canonicalised."""
    columns = sorted(
        (c for c in frame.columns if not str(c).startswith("_")), key=str
    )
    rows = [
        [_cell(value) for value in row]
        for row in frame[columns].itertuples(index=False, name=None)
    ]
    rows.sort()
    return {"columns": [str(c) for c in columns], "rows": rows}


def file_manifest(folder, patterns) -> dict | None:
    """Size and mtime of the files a processor said it reads, or None.

    ``None`` means the processor declared nothing, and the caller must treat
    that as "cannot tell" and rerun -- there is no comparing files nobody named,
    and a cache that quietly assumed "no files" would serve a stale GDX after a
    replaced download, which is the failure docs/vre-timeseries.md warns about
    hardest and the one nothing else can see.

    Size and mtime rather than a hash of the contents: the PECD folders alone
    are 742 MB, and re-reading them every build to find out whether they changed
    would cost more than the rebuild it saves. A re-copy of identical files
    moves mtime and so reruns, which for a download is the right answer.
    """
    if not patterns:
        return None

    return {
        "folder": str(folder),
        "patterns": sorted(str(pattern) for pattern in patterns),
        "entries": _entries(folder, patterns),
    }


def _entries(folder, patterns) -> dict:
    """``{relative path: [size, mtime_ns]}`` for everything the patterns match."""
    root = Path(folder)
    seen: dict[str, list] = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if not path.is_file():
                continue
            stat = path.stat()
            seen[str(path.relative_to(root))] = [stat.st_size, stat.st_mtime_ns]
    return dict(sorted(seen.items()))


def files_still_match(files: dict | None) -> bool:
    """Whether the recorded input files are still what is on disk.

    The folder and the patterns are kept in the record precisely so this can be
    asked without the frames, and without importing a processor module to find
    out what it reads. That is what lets CacheManager notice a replaced download
    on a build where nothing else changed -- the case no other check can see,
    because a new PECD folder touches no workbook and no config.

    A record with no file declaration is not "unchanged": the caller reruns.
    """
    if not files:
        return False
    return _entries(files.get("folder", ""), files.get("patterns", ())) == files.get("entries")


def build_record(frames: dict, scalars: dict, files: dict | None) -> dict:
    """The whole input to one spec's run, ready to store and compare."""
    return {
        "scalars": {key: scalars[key] for key in sorted(scalars)},
        "frames": {name: frame_record(frames[name]) for name in sorted(frames)},
        "files": files,
    }


def _readable(row, columns) -> str:
    """A row as `col=value`, skipping what it does not say."""
    return " ".join(
        f"{column}={'not set' if value == MISSING else value}"
        for column, value in zip(columns, row)
    )


def _describe_frame_difference(name: str, old: dict, new: dict) -> list[str]:
    """What changed in one frame, phrased for a log line.

    Compared as sets rather than by position. Rows are sorted, so a change to a
    value early in the sort order moves the row, and a positional walk then
    reports every row after it as changed -- three rows shuffling along reads as
    three edits when there was one, which is worse than saying nothing useful.

    A row that left and a row that arrived differing in exactly one column is
    the same row edited, and gets the message this whole record exists for:
    which value was what, and is now what.
    """
    if old.get("columns") != new.get("columns"):
        return [f"{name} columns"]

    columns = new.get("columns", [])
    old_rows = [tuple(r) for r in old.get("rows", [])]
    new_rows = [tuple(r) for r in new.get("rows", [])]

    gone = [r for r in old_rows if r not in set(new_rows)]
    came = [r for r in new_rows if r not in set(old_rows)]
    if not gone and not came:
        return []

    differences = []
    unmatched_new = list(came)
    for old_row in gone:
        match = next(
            (candidate for candidate in unmatched_new
             if sum(a != b for a, b in zip(old_row, candidate)) == 1),
            None,
        )
        if match is None:
            continue
        unmatched_new.remove(match)
        for column, was, now in zip(columns, old_row, match):
            if was != now:
                # Only the columns that say something: a row identified as
                # "node_output2=not set" is identified by nothing.
                identity = _readable(
                    [v for v, w in zip(old_row, match) if v == w and v != MISSING],
                    [c for c, v, w in zip(columns, old_row, match)
                     if v == w and v != MISSING],
                )
                differences.append(
                    f"{name} {identity} {column} "
                    f"{'not set' if was == MISSING else was} -> "
                    f"{'not set' if now == MISSING else now}"
                )
        if len(differences) >= DIFFERENCE_LIMIT:
            return differences

    for row in unmatched_new[:DIFFERENCE_LIMIT]:
        differences.append(f"{name} gained {_readable(row, columns)}")
    still_gone = [r for r in gone if r not in set(new_rows)]
    if not came and still_gone:
        for row in still_gone[:DIFFERENCE_LIMIT]:
            differences.append(f"{name} lost {_readable(row, columns)}")
    return differences


def describe_difference(old: dict | None, new: dict) -> str | None:
    """Why this input differs from the recorded one, or None if it does not.

    The reason is the message the build prints, which is the point of keeping a
    readable record rather than a hash: "elec demand FI00_elec twh/year 100 ->
    101" tells the reader what they changed, where "the demand data changed"
    only tells them a file was touched.
    """
    if old is None:
        return "nothing was recorded for it yet"

    if old.get("files") != new.get("files"):
        if new.get("files") is None:
            return "it does not say which input files it reads, so it always reruns"
        return "its input files changed"

    old_scalars, new_scalars = old.get("scalars", {}), new.get("scalars", {})
    if old_scalars != new_scalars:
        changed = sorted(
            key for key in set(old_scalars) | set(new_scalars)
            if old_scalars.get(key) != new_scalars.get(key)
        )
        return f"its settings changed: {', '.join(changed[:DIFFERENCE_LIMIT])}"

    old_frames, new_frames = old.get("frames", {}), new.get("frames", {})
    if set(old_frames) != set(new_frames):
        return "it reads different source data than it did"

    for name in sorted(new_frames):
        differences = _describe_frame_difference(name, old_frames[name], new_frames[name])
        if differences:
            return "; ".join(differences[:DIFFERENCE_LIMIT])

    return None
