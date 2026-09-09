"""Is the unittype migration done, and is what it produced consistent?

A source workbook used to name a technology twice: a human-readable
``Generator_ID`` on every row, and a ``unittype`` that ``unittypedata`` mapped it
to. Only the second reaches the model -- a unit is named
``{country}[_{unit_name_prefix}]_{unittype}`` -- so the first was removed and a
``unitdata`` sheet now writes its ``unittype`` directly.

This checks a folder of workbooks against that rule. Three questions, in the order
they matter:

1. does any sheet still carry a ``Generator_ID`` header;
2. does any cell anywhere in a workbook still hold a legacy human-readable name --
   which is how a VLOOKUP or SUMIF helper table gets left behind when the Excel
   replace was run at sheet scope instead of workbook scope;
3. does every ``unitdata`` ``unittype`` resolve in some ``unittypedata`` sheet.

What it cannot see
------------------
It reads the workbooks, not a config, so it does not know which of them a build
lists -- it checks every ``.xlsx`` in the folder and says what it finds. Question 3
is therefore asked across the whole folder at once, because one workbook's
``unitdata`` is routinely declared by another's ``unittypedata``; a config listing
only some of them can still leave a unittype undeclared, and this cannot tell.

It reads cached cell values, so a workbook edited by a script and never reopened in
Excel looks empty wherever its formulas are. That is a real defect and shows up here
as a unitdata sheet with no unittype values rather than as a diagnosis.

Usage:
    python tools/check_unittype_columns.py <folder> [--legacy-names FILE]

Exit 0 when nothing is found, 1 otherwise.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import openpyxl

IGNORE_MARKER = "##"
LEGACY_HEADER = "generator_id"


def sheet_header_and_rows(worksheet):
    """``(lower-cased header, rows)``, truncated at the first fully-empty row.

    Mirrors ``read_input_excels``: a blank row ends the sheet. Without the
    truncation, the scratch area workbooks keep below their tables would be read as
    if it were data.
    """
    rows = worksheet.iter_rows(values_only=True)
    try:
        header = next(rows)
    except StopIteration:
        return [], []
    if header is None:
        return [], []
    names = [str(c).strip().lower() if c is not None else "" for c in header]

    body = []
    for row in rows:
        if all(c is None or not str(c).strip() for c in row):
            break
        body.append(row)
    return names, body


def cell_texts(worksheet, header):
    """Every non-blank cell the reader would look at, as stripped text.

    Columns the reader drops are skipped, and that is the whole difficulty of
    this check: ``## Description`` exists precisely to keep the old
    human-readable name, so scanning it would report every migrated row as a
    leftover. ``note`` is dropped by the reader for the same legacy reason.
    Rows carrying the marker go too, so a parked helper row is not evidence.

    The dimension columns are skipped for a different reason: a name can be both.
    ``Light oil`` was a Generator_ID and is also the name of a fuel grid, so the
    ``grid_input1`` cell that says the LFO unit burns it is correct and must stay.
    """
    skipped = {
        index for index, name in enumerate(header)
        if name.startswith(IGNORE_MARKER)
        or name == "note"
        or name.startswith(("grid", "node", "country", "from_", "to_"))
    }
    for row in worksheet.iter_rows(values_only=True):
        if any(c is not None and str(c).strip().startswith(IGNORE_MARKER) for c in row):
            continue
        for index, cell in enumerate(row):
            if cell is None or index in skipped:
                continue
            text = str(cell).strip()
            if text:
                yield text


def scan(folder: Path, *, collect_texts: bool = False):
    """Read every workbook once.

    Returns the legacy headers found, the unittypes ``unittypedata`` declares, the
    unittypes ``unitdata`` uses and where, and -- only when `collect_texts` --
    every cell text per workbook.

    `collect_texts` is off by default because it is the whole cost of this tool:
    reading every cell of every sheet means reading all 22 MB of
    ``demanddata_elec_own_projection.xlsx``, which the other two checks never open
    past its header rows.
    """
    legacy_headers = []                # (file, sheet)
    declared = {}                      # lower-cased unittype -> its spelling
    used = defaultdict(Counter)        # lower-cased unittype -> {file:sheet: rows}
    texts_by_file = defaultdict(Counter)

    for path in sorted(folder.glob("*.xlsx")):
        if path.name.startswith("~$"):   # an Excel lock file, not a workbook
            continue
        book = openpyxl.load_workbook(path, read_only=True, data_only=True)
        try:
            for sheet_name in book.sheetnames:
                sheet = book[sheet_name]
                names, body = sheet_header_and_rows(sheet)
                if not names:
                    continue

                if LEGACY_HEADER in names:
                    legacy_headers.append((path.name, sheet_name))

                if collect_texts:
                    for text in cell_texts(sheet, names):
                        texts_by_file[path.name][text] += 1

                # Only the sheets a build reads. A workbook may keep a working
                # sheet with a unittype column of its own -- industrialCHP's
                # 'TYNDP2024' does -- and the pipeline never opens it, so
                # reporting its values as undeclared would be noise.
                is_unittypedata = sheet_name.lower().startswith("unittypedata")
                if not is_unittypedata and not sheet_name.lower().startswith("unitdata"):
                    continue
                if "unittype" not in names:
                    continue
                column = names.index("unittype")
                where = f"{path.name}:{sheet_name}"
                for row in body:
                    if any(c is not None and str(c).strip().startswith(IGNORE_MARKER)
                           for c in row):
                        continue
                    if column >= len(row) or row[column] is None:
                        continue
                    value = str(row[column]).strip()
                    if not value:
                        continue
                    if is_unittypedata:
                        declared.setdefault(value.lower(), value)
                    else:
                        used[value.lower()][where] += 1
        finally:
            book.close()

    return legacy_headers, declared, used, texts_by_file


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="folder of source workbooks")
    parser.add_argument(
        "--legacy-names",
        type=Path,
        help="a file of one legacy Generator_ID name per line; without it, the "
             "leftover-name check is skipped and says so",
    )
    args = parser.parse_args(argv)

    if not args.folder.is_dir():
        print(f"Not a folder: {args.folder}")
        return 1

    legacy_headers, declared, used, texts_by_file = scan(
        args.folder, collect_texts=args.legacy_names is not None
    )
    problems = 0

    workbooks = len(list(args.folder.glob("*.xlsx")))
    print(f"{len(declared)} unittype(s) declared, {len(used)} used, "
          f"across {workbooks} workbook(s).\n")

    if legacy_headers:
        problems += len(legacy_headers)
        print(f"Still carrying a 'Generator_ID' header ({len(legacy_headers)}):")
        for file_name, sheet_name in legacy_headers:
            print(f"  {file_name}:{sheet_name}")
        print()

    if args.legacy_names:
        wanted = [line.strip()
                  for line in args.legacy_names.read_text(encoding="utf-8").splitlines()]
        # A legacy name that is also a real unittype -- 'Electrolyser', 'Nuclear' --
        # is not evidence of anything: it is what the migration was to produce.
        wanted = [n for n in wanted
                  if n and not n.startswith("#") and n.lower() not in declared]

        leftovers = []
        for file_name, texts in texts_by_file.items():
            for name in wanted:
                count = texts.get(name, 0)
                if count:
                    leftovers.append((file_name, name, count))
        if leftovers:
            problems += len(leftovers)
            print(f"Legacy names still in cells ({len(leftovers)}) -- a helper table "
                  "the workbook-scope replace missed:")
            for file_name, name, count in sorted(leftovers):
                print(f"  {file_name}: {name!r} x{count}")
            print()
    else:
        print("Leftover-name check skipped: pass --legacy-names to run it.\n")

    undeclared = sorted(set(used) - set(declared))
    if undeclared:
        problems += len(undeclared)
        print(f"unitdata unittype(s) no unittypedata declares ({len(undeclared)}):")
        for key in undeclared:
            where = ", ".join(f"{w} x{n}" for w, n in used[key].most_common(3))
            print(f"  {key!r} -- {where}")
        print()

    unused = sorted(set(declared) - set(used))
    if unused:
        # Not a problem: a unittypedata workbook is a catalogue and a build lists
        # the entries it wants. Said out loud because a typo looks exactly like this.
        print(f"Declared but used by no unitdata row ({len(unused)}), for information:")
        print("  " + ", ".join(declared[k] for k in unused))
        print()

    print("OK" if not problems else f"{problems} problem(s)")
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
