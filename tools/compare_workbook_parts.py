"""
compare_workbook_parts.py -- prove two inputData.xlsx files are the same workbook.

Usage:
    python tools/compare_workbook_parts.py <new_file> <reference_file> [-v]

Example:
    python tools/compare_workbook_parts.py input_tyndp2024_NationalTrends_2030/inputData.xlsx dev/inputData-NT2030-ref.xlsx

An .xlsx is a zip of XML parts, so this unzips both and compares part by part.
That covers everything a build can change -- cell values and their types, column
order, column widths, alignment and rotation, freeze panes, table styles, sheet
order -- which is what makes it the right check for a refactor that must not
change its output. compare_input_excels.py answers a different question: it reads
sheets through pandas as strings, so it sees no formatting at all and calls 500
and 500.0 a difference. Reach for that one when you want to know *what* changed
in the data; reach for this one when you want to know *whether* anything did.

Literal file bytes are not comparable and never were. openpyxl stamps
docProps/core.xml with datetime.now() (see openpyxl/packaging/core.py), so two
builds of unchanged code already differ there. Those two timestamps are the only
thing this ignores; everything else must match exactly.

Exit code is 0 when the workbooks are identical, 1 when they are not, so it can
gate a loop.
"""

import argparse
import difflib
import re
import sys
import zipfile
from pathlib import Path

#: The build-time stamps openpyxl writes into docProps/core.xml. Nothing else is
#: allowed to differ, so these are matched narrowly rather than by element name.
TIMESTAMP_ELEMENTS = re.compile(
    rb"<dcterms:(created|modified)[^>]*>[^<]*</dcterms:(created|modified)>"
)

#: Lines of unified diff to print for the first differing part. Enough to see
#: what changed; a whole sheet's XML is not readable and not the point.
DIFF_LINES = 40


def read_parts(path: Path) -> dict:
    """{member name: bytes} for every part of the workbook, timestamps blanked."""
    parts = {}
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            data = archive.read(name)
            if name == "docProps/core.xml":
                data = TIMESTAMP_ELEMENTS.sub(b"<dcterms:STAMP/>", data)
            parts[name] = data
    return parts


def part_diff(name: str, new: bytes, ref: bytes) -> list:
    """A readable unified diff of one part, or a byte summary if it is not text."""
    try:
        new_text = new.decode("utf-8")
        ref_text = ref.decode("utf-8")
    except UnicodeDecodeError:
        return [f"  binary part, {len(ref)} bytes in reference, {len(new)} in new"]

    # xlsx XML is written as one long line, so split on tags to get diffable units.
    new_lines = [line + ">" for line in new_text.split(">") if line]
    ref_lines = [line + ">" for line in ref_text.split(">") if line]

    diff = difflib.unified_diff(ref_lines, new_lines, fromfile="reference", tofile="new", n=1)
    lines = [f"  {line.rstrip()}" for line in diff]
    if len(lines) > DIFF_LINES:
        remaining = len(lines) - DIFF_LINES
        lines = lines[:DIFF_LINES] + [f"  ... and {remaining} more diff line(s)"]
    return lines


def compare(new_path: Path, ref_path: Path, verbose: bool = False) -> bool:
    """True when the two workbooks are the same. Prints what differs."""
    new_parts = read_parts(new_path)
    ref_parts = read_parts(ref_path)

    print(f"new       : {new_path}")
    print(f"reference : {ref_path}")
    print()

    identical = True

    missing = sorted(set(ref_parts) - set(new_parts))
    extra = sorted(set(new_parts) - set(ref_parts))
    if missing:
        identical = False
        print(f"MISSING parts (were in reference): {missing}")
    if extra:
        identical = False
        print(f"EXTRA parts (not in reference):    {extra}")

    common = sorted(set(new_parts) & set(ref_parts))
    differing = [name for name in common if new_parts[name] != ref_parts[name]]

    if differing:
        identical = False
        print(f"{len(differing)} of {len(common)} common part(s) differ:")
        for name in differing:
            print(f"  {name}")
        print()
        shown = differing if verbose else differing[:1]
        for name in shown:
            print(f"--- {name} ---")
            for line in part_diff(name, new_parts[name], ref_parts[name]):
                print(line)
            print()
        if not verbose and len(differing) > 1:
            print(f"({len(differing) - 1} further differing part(s) not shown; pass -v for all)")
            print()

    if identical:
        print(f"IDENTICAL -- {len(common)} parts match (docProps timestamps ignored)")
    return identical


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two .xlsx files part by part, ignoring build timestamps."
    )
    parser.add_argument("new_file", type=Path)
    parser.add_argument("reference_file", type=Path)
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="show a diff for every differing part, not just the first")
    args = parser.parse_args()

    for path in (args.new_file, args.reference_file):
        if not path.is_file():
            print(f"No such file: {path}")
            return 2

    return 0 if compare(args.new_file, args.reference_file, args.verbose) else 1


if __name__ == "__main__":
    sys.exit(main())
