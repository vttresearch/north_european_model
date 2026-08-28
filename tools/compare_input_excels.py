"""
compare_input_excels.py -- compare two Backbone inputData Excel files sheet by sheet.

Usage:
    python tools/compare_input_excels.py <new_file> <reference_file>

Example:
    python tools/compare_input_excels.py input_tyndp2024_NationalTrends_2030/inputData.xlsx dev/inputData-NT2030-ref.xlsx

Exit code is 0 when every common sheet matched and 1 when anything differed, so
it can gate a loop.

Checks performed (in order):
    1. Sheet existence  -- missing sheets are reported, comparison continues on common sheets
    2. Column names     -- case-sensitive; missing columns are flagged as severe errors;
                          case-only changes are flagged as standardization failures;
                          column order changes are reported
    3. Row content      -- set-based diff (row order does not matter);
                          detailed output for <= DETAIL_THRESHOLD differences per
                          category, summary only for larger diffs

What it sees, and what it does not
----------------------------------
Sheets are read through pandas with ``dtype=str``, so this compares the *values* a
reader would see and nothing else. It cannot see cell formatting, column widths,
sheet order, or the number formats Excel applied -- use compare_workbook_parts.py
when the question is whether the file is the same file.

Reading everything as text also means ``500`` and ``500.0`` compare as different.
That is deliberate here: a change in how a number is written is usually what you
are looking for when comparing two builds by hand.

The fake-MultiIndex marker row (the second header row, blank in the dimension
columns) is read as an ordinary data row, so it takes part in the row diff. A
sheet that gained or lost a parameter column shows that twice: once as a column
difference and once as a changed marker row.
"""

import sys
import argparse
from pathlib import Path

import pandas as pd

DETAIL_THRESHOLD = 20   # show individual rows below this many total differences


def load_sheets(path: Path) -> dict:
    """Return {sheet_name: DataFrame} for all sheets, values as stripped strings."""
    xl = pd.ExcelFile(path, engine="openpyxl")
    sheets = {}
    for name in xl.sheet_names:
        df = xl.parse(name, header=0, dtype=str)
        # Strip whitespace from all string cells; leave NaN as empty string for diffing
        for col in df.columns:
            df[col] = df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
        sheets[name] = df
    return sheets


def check_sheet_existence(new_sheets: dict, ref_sheets: dict) -> list:
    new_names = set(new_sheets)
    ref_names = set(ref_sheets)
    lines = []
    missing = sorted(ref_names - new_names)
    extra = sorted(new_names - ref_names)
    if missing:
        lines.append(f"  MISSING in new (were in ref): {missing}")
    if extra:
        lines.append(f"  Extra in new (not in ref):    {extra}")
    if not lines:
        lines.append("  All sheets present -- OK")
    return lines


def check_columns(new_df: pd.DataFrame, ref_df: pd.DataFrame) -> list:
    new_cols = list(new_df.columns)
    ref_cols = list(ref_df.columns)
    new_set = set(new_cols)
    ref_set = set(ref_cols)
    lines = []

    missing = sorted(ref_set - new_set)
    extra = sorted(new_set - ref_set)

    if missing:
        lines.append(f"  [SEVERE] Missing columns: {missing}")

    # Case-change detection: columns that exist in both but with different casing
    ref_lower = {c.lower(): c for c in ref_cols}
    new_lower = {c.lower(): c for c in new_cols}
    for lname, ref_col in ref_lower.items():
        if lname in new_lower and new_lower[lname] != ref_col:
            lines.append(f"  [CASE CHANGE] '{ref_col}' -> '{new_lower[lname]}'  (standardization failure)")

    if extra:
        lines.append(f"  Extra columns in new: {extra}")

    # Reported even when something else is also wrong: it used to be suppressed
    # whenever there was a missing column, which is exactly when a reordering is
    # most likely and most confusing.
    if new_cols != ref_cols and not missing and not extra:
        lines.append(f"  Column order changed: {ref_cols} -> {new_cols}")

    return lines


def check_rows(new_df: pd.DataFrame, ref_df: pd.DataFrame, common_cols: list) -> list:
    """Set-based row diff over common columns only."""
    def to_row_set(df):
        return set(tuple(row) for row in df[common_cols].fillna("").values.tolist())

    new_rows = to_row_set(new_df)
    ref_rows = to_row_set(ref_df)

    added = sorted(new_rows - ref_rows)
    removed = sorted(ref_rows - new_rows)
    total = len(added) + len(removed)

    if total == 0:
        return []

    lines = []
    if removed:
        lines.append(f"  Removed rows ({len(removed)}):")
        for row in removed[:DETAIL_THRESHOLD]:
            lines.append(f"    - {list(row)}")
    if added:
        lines.append(f"  Added rows ({len(added)}):")
        for row in added[:DETAIL_THRESHOLD]:
            lines.append(f"    + {list(row)}")
    # Only when something was actually cut. The condition used to be on the
    # total, so 15 added and 15 removed announced a truncation that had not
    # happened.
    if len(added) > DETAIL_THRESHOLD or len(removed) > DETAIL_THRESHOLD:
        lines.append(
            f"  ... showing up to {DETAIL_THRESHOLD} differences per category"
            f" (+{len(added)} added, -{len(removed)} removed)"
        )
    return lines


def compare_sheet(new_df: pd.DataFrame, ref_df: pd.DataFrame) -> list:
    lines = []

    new_empty = new_df.empty or (len(new_df) == 1 and new_df.iloc[0].isna().all())
    ref_empty = ref_df.empty or (len(ref_df) == 1 and ref_df.iloc[0].isna().all())

    if new_empty and ref_empty:
        return []
    if new_empty and not ref_empty:
        lines.append(f"  Sheet is empty in new, has {len(ref_df)} rows in ref")
        return lines
    if not new_empty and ref_empty:
        lines.append(f"  Sheet is empty in ref, has {len(new_df)} rows in new -- new content")
        return lines

    col_lines = check_columns(new_df, ref_df)
    lines.extend(col_lines)

    common_cols = [c for c in ref_df.columns if c in set(new_df.columns)]
    if common_cols:
        row_lines = check_rows(new_df, ref_df, common_cols)
        lines.extend(row_lines)
    else:
        lines.append("  No common columns -- skipping row comparison")

    return lines


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare two Backbone inputData Excel files sheet by sheet."
    )
    parser.add_argument("new_file", help="Path to the new file (relative or absolute)")
    parser.add_argument("reference_file", help="Path to the reference file")
    args = parser.parse_args()

    new_path = Path(args.new_file)
    ref_path = Path(args.reference_file)

    for p in (new_path, ref_path):
        if not p.exists():
            print(f"ERROR: file not found: {p}")
            return 2

    print(f"New file : {new_path}")
    print(f"Ref file : {ref_path}")
    print()
    print("Loading files...")
    new_sheets = load_sheets(new_path)
    ref_sheets = load_sheets(ref_path)
    print(f"  New: {len(new_sheets)} sheets    Ref: {len(ref_sheets)} sheets")
    print()

    # --- 1. Sheet existence ---
    print("=" * 64)
    print("1. SHEET EXISTENCE")
    print("=" * 64)
    for line in check_sheet_existence(new_sheets, ref_sheets):
        print(line)
    print()

    # --- 2 & 3. Per-sheet column and row comparison ---
    print("=" * 64)
    print("2. SHEET CONTENT (columns + rows)")
    print("=" * 64)

    common = [s for s in ref_sheets if s in new_sheets]
    sheets_ok = 0
    sheets_with_issues = 0

    for sheet_name in common:
        new_df = new_sheets[sheet_name]
        ref_df = ref_sheets[sheet_name]
        issues = compare_sheet(new_df, ref_df)
        if issues:
            sheets_with_issues += 1
            print(f"\nSheet '{sheet_name}'  (new rows: {len(new_df)}, ref rows: {len(ref_df)})")
            for line in issues:
                print(line)
        else:
            sheets_ok += 1

    print()
    print("=" * 64)
    print("SUMMARY")
    print("=" * 64)
    print(f"  Sheets identical  : {sheets_ok}")
    print(f"  Sheets with issues: {sheets_with_issues}")
    missing = sorted(set(ref_sheets) - set(new_sheets))
    extra = sorted(set(new_sheets) - set(ref_sheets))
    if missing:
        print(f"  Missing sheets    : {missing}")
    if extra:
        print(f"  Extra sheets      : {extra}")

    return 0 if not (sheets_with_issues or missing or extra) else 1


if __name__ == "__main__":
    sys.exit(main())
