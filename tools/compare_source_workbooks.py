"""
compare_source_workbooks.py -- what changed, numerically, between two versions of
the source workbooks.

Usage:
    python tools/compare_source_workbooks.py <old_folder> [new_folder]
    python tools/compare_source_workbooks.py --git-ref HEAD

Examples:
    python tools/compare_source_workbooks.py --git-ref HEAD
    python tools/compare_source_workbooks.py --git-ref v3.2 --map-generator-ids
    python tools/compare_source_workbooks.py /tmp/before src_files/data_files

`new_folder` defaults to ``src_files/data_files``. Exit code is 0 when nothing
differs and 1 otherwise, so it can gate a loop.

Why this exists, and why the other two tools do not cover it
------------------------------------------------------------
A source workbook is a spreadsheet, so a name typed in a cell is often also a
*lookup key*: `SUMIF(EE00!C:C, $B2, EE00!F:F)`, `VLOOKUP($B2,$L$5:$M$29,2,FALSE)`,
`COUNTIF($B2,"*heat pump*")`. Rename the data and leave the criterion, or the
helper table it points at, and the formula does not fail -- it returns 0, or the
value from a neighbouring row. Nothing is logged, and the build is happy.

`check_unittype_columns.py` finds a leftover *name*. It cannot find a leftover name
that is being *used as a key*, because that cell looks exactly like one that was
never meant to change. Only the numbers show it. During the generator_ID removal
this found six such breakages that every other check passed over -- among them
58 hydro units silently left with zero vomCosts and ramp costs, and every Finnish
heat pump and electric boiler deleted from the model.

It is equally the tool for reviewing a deliberate data edit: run it after changing
a workbook and read the report as "here is everything I changed", which is the
question a diff of a binary .xlsx cannot answer.

What it compares
----------------
Only the sheets a build reads -- those whose name starts with a data-type prefix.
Rows are matched on their dimension columns, not on position, so reordering a
sheet is not a difference. Every numeric cell of a matched row is compared;
non-numeric cells are not, because a rename is exactly what they are for.

What it cannot see
------------------
Cached formula values, which is what openpyxl reads. A workbook edited by a script
and never reopened in Excel has no cached values, and its formula cells read as
empty -- that shows up here as a sheet full of differences rather than as a
diagnosis. Blank versus zero is reported; at the Excel-builder boundary the two
mean the same thing, so such a difference is usually noise.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import openpyxl

#: Sheets a build reads. A sheet whose name starts with one of these is input.
PREFIXES = ("unitdata", "unittypedata", "nodedata", "demanddata",
            "transferdata", "emissiondata", "userconstraintdata")

IGNORE_MARKER = "##"

#: Columns that say *which thing* a row is about, across all the data types. A
#: row is matched on whichever of these its sheet carries, so the comparison
#: survives a sheet being reordered.
KEY_COLUMNS = ("country", "unittype", "unit_name_prefix", "scenario", "year",
               "grid", "node", "node_suffix", "from_country", "to_country",
               "from_suffix", "to_suffix", "emission", "group", "parameter",
               "1st dimension", "2nd dimension", "3rd dimension", "4th dimension")

#: Never compared: dropped by the reader, or the key itself under its old name.
SKIP_COLUMNS = ("note", "generator_id", "method")

DETAIL_LIMIT = 12


def read_sheet(path: Path, sheet_name: str):
    """``(lower-cased header, rows)`` as the reader would see them.

    Truncated at the first fully-empty row and with ``##`` rows dropped, so the
    working area below and beside a table is not compared.
    """
    book = openpyxl.load_workbook(path, data_only=True, read_only=True)
    if sheet_name not in book.sheetnames:
        book.close()
        return None, None
    rows = book[sheet_name].iter_rows(values_only=True)
    try:
        header = next(rows)
    except StopIteration:
        book.close()
        return None, None
    if header is None:
        book.close()
        return None, None
    names = [str(c).strip().lower() if c is not None else "" for c in header]

    body = []
    for row in rows:
        if all(c is None or not str(c).strip() for c in row):
            break
        if any(c is not None and str(c).strip().startswith(IGNORE_MARKER) for c in row):
            continue
        body.append(row)
    book.close()
    return names, body


def used_unittypes(folder: Path) -> set:
    """Every ``unittype`` a folder's *unitdata* sheets write, lower-cased.

    Used rather than declared, because a unittypedata catalogue declares names
    nothing uses -- including both halves of a disagreement. Only a migrated
    unitdata sheet carries a ``unittype`` column at all, so this is exactly the
    set of names the new side has actually adopted.
    """
    used = set()
    for path in sorted(folder.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        book = openpyxl.load_workbook(path, read_only=True)
        sheets = [s for s in book.sheetnames
                  if s.lower().startswith("unitdata")]
        book.close()
        for sheet_name in sheets:
            names, body = read_sheet(path, sheet_name)
            if not names or "unittype" not in names:
                continue
            position = names.index("unittype")
            for row in body:
                if position < len(row) and row[position] is not None:
                    used.add(str(row[position]).strip().lower())
    return used


def generator_id_map(folder: Path, prefer: set) -> dict:
    """``old Generator_ID (lower-cased) -> unittype``, from a folder's unittypedata.

    Only needed with ``--map-generator-ids``, to compare across the migration that
    removed the column. Afterwards both sides spell the key the same way.

    One id can have two answers: the workbooks disagree about ``Electrolyser``,
    which `H2 heavy.xlsx` calls ``P2H`` and the compilation calls ``Electrolyser``.
    `prefer` breaks that tie -- a target the *new* side's unitdata sheets actually
    write is the one this comparison is about. A tie nothing settles is reported rather than
    guessed at, because guessing would report every row of that unit as moved.
    """
    candidates = {}
    for path in sorted(folder.glob("*.xlsx")):
        if path.name.startswith("~$"):
            continue
        book = openpyxl.load_workbook(path, read_only=True)
        sheets = [s for s in book.sheetnames if s.lower().startswith("unittypedata")]
        book.close()
        for sheet_name in sheets:
            names, body = read_sheet(path, sheet_name)
            if not names or "generator_id" not in names or "unittype" not in names:
                continue
            old, new = names.index("generator_id"), names.index("unittype")
            for row in body:
                if old < len(row) and new < len(row) and row[old] is not None and row[new] is not None:
                    candidates.setdefault(str(row[old]).strip().lower(), []).append(
                        str(row[new]).strip())

    mapping = {}
    for key, targets in candidates.items():
        unique = list(dict.fromkeys(targets))
        if len(unique) > 1:
            settled = [t for t in unique if t.lower() in prefer]
            if len(settled) == 1:
                mapping[key] = settled[0]
                continue
            print(f"  '{key}' maps to {unique} and nothing settles it; "
                  f"using {unique[0]!r}")
        mapping[key] = unique[0]
    return mapping


def index_rows(names, body, mapping=None):
    """``{key tuple: {column: number}}`` for one sheet.

    `mapping` translates an old ``generator_id`` into the ``unittype`` a migrated
    sheet writes. It is applied to whichever side still spells the key the old
    way -- a folder holds migrated and unmigrated workbooks at the same time, and
    mapping only one side would report every row of the unmigrated ones as moved.
    """
    use_legacy_key = (mapping is not None
                      and "unittype" not in names
                      and "generator_id" in names)
    indexed = {}
    for row in body:
        key = []
        for column in KEY_COLUMNS:
            if column == "unittype" and use_legacy_key:
                position = names.index("generator_id")
                value = row[position] if position < len(row) else None
                value = mapping.get(str(value).strip().lower(), value) if value is not None else None
            elif column in names:
                position = names.index(column)
                value = row[position] if position < len(row) else None
            else:
                continue
            key.append("" if value is None else str(value).strip().lower())

        numbers = {}
        for position, column in enumerate(names):
            if not column or column.startswith(IGNORE_MARKER) or column in SKIP_COLUMNS:
                continue
            value = row[position] if position < len(row) else None
            if isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numbers[column] = float(value)
        # First row wins, matching merge_row_by_row: a repeated key is a question
        # for the build, not for this comparison.
        indexed.setdefault(tuple(key), numbers)
    return indexed


def compare_sheet(old_path, new_path, sheet_name, mapping, tolerance):
    old_names, old_body = read_sheet(old_path, sheet_name)
    new_names, new_body = read_sheet(new_path, sheet_name)
    if old_names is None or new_names is None:
        side = "old" if old_names is None else "new"
        return [f"sheet is missing or empty on the {side} side"]

    old_rows = index_rows(old_names, old_body, mapping)
    new_rows = index_rows(new_names, new_body, mapping)

    findings = []
    for key in sorted(set(old_rows) | set(new_rows)):
        label = "|".join(part for part in key if part) or "(no key columns)"
        if key not in old_rows:
            findings.append(f"{label}: row only in the new workbook")
            continue
        if key not in new_rows:
            findings.append(f"{label}: row only in the old workbook")
            continue
        before, after = old_rows[key], new_rows[key]
        for column in sorted(set(before) | set(after)):
            a, b = before.get(column), after.get(column)
            if a is None or b is None:
                if a != b:
                    findings.append(f"{label}: {column} {a} -> {b}")
            elif abs(a - b) > tolerance:
                findings.append(f"{label}: {column} {a:g} -> {b:g}")
    return findings


def sheets_of(path: Path):
    book = openpyxl.load_workbook(path, read_only=True)
    names = [s for s in book.sheetnames if s.lower().startswith(PREFIXES)]
    book.close()
    return names


def extract_git_ref(ref: str, folder: Path, destination: Path) -> None:
    """Every tracked workbook of `folder` at `ref`, written into `destination`."""
    listing = subprocess.run(
        ["git", "ls-files", str(folder)],
        capture_output=True, text=True, check=True,
    ).stdout.split("\n")
    destination.mkdir(parents=True, exist_ok=True)
    for tracked in listing:
        tracked = tracked.strip()
        if not tracked.endswith(".xlsx"):
            continue
        blob = subprocess.run(
            ["git", "show", f"{ref}:{tracked}"],
            capture_output=True, check=True,
        ).stdout
        (destination / Path(tracked).name).write_bytes(blob)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("old_folder", nargs="?", type=Path,
                        help="folder holding the earlier version of the workbooks")
    parser.add_argument("new_folder", nargs="?", type=Path,
                        default=Path("src_files/data_files"),
                        help="the later version (default: src_files/data_files)")
    parser.add_argument("--git-ref",
                        help="take the earlier version from this git ref instead "
                             "of a folder, e.g. HEAD")
    parser.add_argument("--map-generator-ids", action="store_true",
                        help="the earlier version predates the unittype migration: "
                             "translate its Generator_ID values through its own "
                             "unittypedata before matching rows")
    parser.add_argument("--tolerance", type=float, default=1e-6,
                        help="numbers closer than this count as equal (default 1e-6)")
    args = parser.parse_args()

    if bool(args.old_folder) == bool(args.git_ref):
        parser.error("give either an old_folder or --git-ref, not both or neither")
    if not args.new_folder.is_dir():
        print(f"Not a folder: {args.new_folder}")
        return 1

    temporary = None
    try:
        if args.git_ref:
            temporary = Path(tempfile.mkdtemp(prefix="workbooks-"))
            extract_git_ref(args.git_ref, args.new_folder, temporary)
            old_folder = temporary
            print(f"Old: {args.new_folder} at {args.git_ref}")
        else:
            old_folder = args.old_folder
            print(f"Old: {old_folder}")
        print(f"New: {args.new_folder}\n")

        mapping = (generator_id_map(old_folder, used_unittypes(args.new_folder))
                   if args.map_generator_ids else None)
        if mapping is not None:
            print(f"Translating {len(mapping)} Generator_ID(s) through the old "
                  "unittypedata.\n")

        total = 0
        compared = 0
        for new_path in sorted(args.new_folder.glob("*.xlsx")):
            # An open workbook leaves a '~$' stub beside it; it is a lock file,
            # not a workbook, and openpyxl cannot read it.
            if new_path.name.startswith("~$"):
                continue
            old_path = old_folder / new_path.name
            if not old_path.exists():
                print(f"{new_path.name}: no earlier version, skipped")
                continue
            for sheet_name in sheets_of(new_path):
                compared += 1
                findings = compare_sheet(old_path, new_path, sheet_name,
                                         mapping, args.tolerance)
                if not findings:
                    continue
                total += len(findings)
                print(f"{new_path.name}:{sheet_name} -- {len(findings)} difference(s)")
                for line in findings[:DETAIL_LIMIT]:
                    print(f"   {line}")
                if len(findings) > DETAIL_LIMIT:
                    print(f"   ... and {len(findings) - DETAIL_LIMIT} more")
                print()

        print(f"{compared} sheet(s) compared, {total} difference(s).")
        return 1 if total else 0
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
