"""Cell-level diffing of two built workbooks.

Why this instead of golden files
---------------------------------
Both sides of a delta are produced by the code under test, in the same process,
in the same run. Add a column to ``PARAM_GNU`` and it appears in both workbooks
with the same values -> zero delta -> no test edit. Change a default -> both
sides move -> still zero. A golden file cannot do that: one side is frozen at
record time and every schema change moves the other.

That is why there is no ``--regenerate`` flag in this suite, and why the parent
Backbone repo's goldens are not an inconsistency -- it compares against a
solver, an external oracle that cannot be re-run per assertion. Here a baseline
is recomputable in about a second.

Omission is the assertion
-------------------------
``assert_delta`` lists what *did* change; everything else is asserted unchanged
by not being mentioned. That is the part which would be unmaintainable any other
way -- enumerating the hundreds of cells that stayed put is exactly the golden
file being reinvented.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from tests._common.excel_read import sheet_keys

#: Sheets whose content is metadata rather than model data.
DEFAULT_IGNORED_SHEETS = ("index", "add_scen_tags")


def normalise(value: Any) -> str:
    """Render a cell for comparison.

    Excel round-trips numbers inconsistently -- 500 may come back as 500.0 --
    so numbers are compared by value and everything else casefolded. Without
    this a delta would fire on formatting and the whole approach would be as
    brittle as the golden files it replaces.
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if value is pd.NA or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    if isinstance(value, (int, float)):
        as_float = float(value)
        if as_float == int(as_float):
            return str(int(as_float))
        return repr(round(as_float, 10))
    text = str(value).strip()
    try:
        as_float = float(text)
    except ValueError:
        return text.casefold()
    if as_float == int(as_float):
        return str(int(as_float))
    return repr(round(as_float, 10))


@dataclass(frozen=True)
class CellChange:
    sheet: str
    key: tuple
    column: str
    before: str
    after: str


@dataclass(frozen=True)
class RowChange:
    sheet: str
    key: tuple


@dataclass
class Delta:
    changed: frozenset[CellChange] = field(default_factory=frozenset)
    added_rows: frozenset[RowChange] = field(default_factory=frozenset)
    removed_rows: frozenset[RowChange] = field(default_factory=frozenset)
    added_columns: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    removed_columns: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    added_sheets: frozenset[str] = field(default_factory=frozenset)
    removed_sheets: frozenset[str] = field(default_factory=frozenset)

    def is_empty(self) -> bool:
        return not (
            self.changed
            or self.added_rows
            or self.removed_rows
            or self.added_columns
            or self.removed_columns
            or self.added_sheets
            or self.removed_sheets
        )

    def describe(self) -> str:
        if self.is_empty():
            return "  (no differences)"
        lines: list[str] = []
        for label, items in (
            ("added sheets", sorted(self.added_sheets)),
            ("removed sheets", sorted(self.removed_sheets)),
            ("added columns", sorted(self.added_columns)),
            ("removed columns", sorted(self.removed_columns)),
        ):
            if items:
                lines.append(f"  {label}: {items}")
        for label, rows in (
            ("added rows", sorted(self.added_rows, key=lambda r: (r.sheet, r.key))),
            ("removed rows", sorted(self.removed_rows, key=lambda r: (r.sheet, r.key))),
        ):
            if rows:
                lines.append(f"  {label}:")
                lines += [f"    {r.sheet} {r.key}" for r in rows[:10]]
                if len(rows) > 10:
                    lines.append(f"    ... and {len(rows) - 10} more")
        if self.changed:
            ordered = sorted(self.changed, key=lambda c: (c.sheet, c.key, c.column))
            lines.append("  changed cells:")
            lines += [
                f"    {c.sheet} {c.key} {c.column}: {c.before!r} -> {c.after!r}"
                for c in ordered[:20]
            ]
            if len(ordered) > 20:
                lines.append(f"    ... and {len(ordered) - 20} more")
        return "\n".join(lines)


def _index_rows(frame: pd.DataFrame, key: Sequence[str]) -> dict[tuple, dict[str, Any]]:
    if frame.empty:
        return {}
    rows: dict[tuple, dict[str, Any]] = {}
    for record in frame.to_dict("records"):
        identity = tuple(normalise(record.get(k)) for k in key)
        rows[identity] = record
    return rows


def _row_multiset(frame: pd.DataFrame) -> dict[tuple, int]:
    """Whole-row counts, for sheets with no declared key."""
    counts: dict[tuple, int] = {}
    for record in frame.to_dict("records"):
        identity = tuple(sorted((str(k), normalise(v)) for k, v in record.items()))
        counts[identity] = counts.get(identity, 0) + 1
    return counts


def workbook_delta(
    before: Mapping[str, pd.DataFrame],
    after: Mapping[str, pd.DataFrame],
    *,
    keys: Mapping[str, Sequence[str]] | None = None,
    ignore: Iterable[str] = DEFAULT_IGNORED_SHEETS,
) -> Delta:
    """Diff two read-back workbooks cell by cell.

    Keyed sheets are matched row-for-row on their key columns; sheets with no
    declared key fall back to a whole-row set comparison, so a change there
    shows up as one removal plus one addition rather than a cell edit.
    """
    ignored = set(ignore)
    keys = dict(keys or sheet_keys(before))

    before_names = set(before) - ignored
    after_names = set(after) - ignored

    changed: set[CellChange] = set()
    added_rows: set[RowChange] = set()
    removed_rows: set[RowChange] = set()
    added_columns: set[tuple[str, str]] = set()
    removed_columns: set[tuple[str, str]] = set()

    for name in sorted(before_names & after_names):
        old, new = before[name], after[name]

        old_cols, new_cols = list(old.columns), list(new.columns)
        added_columns |= {(name, c) for c in new_cols if c not in old_cols}
        removed_columns |= {(name, c) for c in old_cols if c not in new_cols}
        shared_cols = [c for c in old_cols if c in new_cols]

        key = [k for k in keys.get(name, ()) if k in shared_cols]
        if not key:
            old_counts, new_counts = _row_multiset(old), _row_multiset(new)
            for identity, count in old_counts.items():
                if new_counts.get(identity, 0) < count:
                    removed_rows.add(RowChange(name, identity))
            for identity, count in new_counts.items():
                if old_counts.get(identity, 0) < count:
                    added_rows.add(RowChange(name, identity))
            continue

        old_rows, new_rows = _index_rows(old, key), _index_rows(new, key)
        removed_rows |= {RowChange(name, k) for k in old_rows.keys() - new_rows.keys()}
        added_rows |= {RowChange(name, k) for k in new_rows.keys() - old_rows.keys()}

        for identity in old_rows.keys() & new_rows.keys():
            old_row, new_row = old_rows[identity], new_rows[identity]
            for column in shared_cols:
                if column in key:
                    continue
                old_value = normalise(old_row.get(column))
                new_value = normalise(new_row.get(column))
                if old_value != new_value:
                    changed.add(
                        CellChange(name, identity, column, old_value, new_value)
                    )

    return Delta(
        changed=frozenset(changed),
        added_rows=frozenset(added_rows),
        removed_rows=frozenset(removed_rows),
        added_columns=frozenset(added_columns),
        removed_columns=frozenset(removed_columns),
        added_sheets=frozenset(after_names - before_names),
        removed_sheets=frozenset(before_names - after_names),
    )


def _expected_cell(spec: Sequence) -> tuple[str, tuple, str, str | None]:
    if len(spec) == 3:
        sheet, key, column = spec
        return sheet, tuple(normalise(k) for k in key), column, None
    if len(spec) == 4:
        sheet, key, column, after = spec
        return sheet, tuple(normalise(k) for k in key), column, normalise(after)
    raise ValueError(
        f"a changed-cell spec is (sheet, key, column) or "
        f"(sheet, key, column, after); got {spec!r}"
    )


def assert_delta(
    delta: Delta,
    *,
    changed: Iterable[Sequence] = (),
    added_rows: Iterable[tuple[str, tuple]] = (),
    removed_rows: Iterable[tuple[str, tuple]] = (),
    added_columns: Iterable[tuple[str, str]] = (),
    removed_columns: Iterable[tuple[str, str]] = (),
    expect_no_change: bool = False,
) -> None:
    """The workbook changed in exactly these ways and no others.

    ``changed`` items are ``(sheet, key, column)`` when only the fact of the
    change matters, or ``(sheet, key, column, after)`` when the value is the
    point. Prefer the 3-tuple: it says "this had to move, by how much is not my
    business", and it survives an edit to the fixture.

    An unlisted column appearing or vanishing is a failure. ``is_col_empty``
    drops all-zero columns (bb_excel_pipeline.py:332), so changing a value from
    0 to 500 can make a whole column materialise -- real behaviour worth
    catching rather than noise.

    Passing nothing requires ``expect_no_change=True``, so a variant that
    silently failed to differ cannot pass vacuously.
    """
    expectations = [_expected_cell(spec) for spec in changed]
    nothing_expected = not (
        expectations or added_rows or removed_rows or added_columns or removed_columns
    )
    if nothing_expected and not expect_no_change:
        raise ValueError(
            "assert_delta was given no expected changes. Pass "
            "expect_no_change=True to assert the workbook is identical -- "
            "otherwise a variant that failed to differ would pass silently."
        )
    if expect_no_change and not nothing_expected:
        raise ValueError("expect_no_change=True cannot be combined with expected changes")

    problems: list[str] = []

    matched: set[CellChange] = set()
    for sheet, key, column, expected_after in expectations:
        candidates = [
            c for c in delta.changed
            if c.sheet == sheet and c.key == key and c.column == column
        ]
        if not candidates:
            problems.append(f"expected {sheet} {key} {column} to change, but it did not")
            continue
        found = candidates[0]
        matched.add(found)
        if expected_after is not None and found.after != expected_after:
            problems.append(
                f"{sheet} {key} {column}: expected {expected_after!r}, got {found.after!r}"
            )

    unexpected = delta.changed - matched
    if unexpected:
        ordered = sorted(unexpected, key=lambda c: (c.sheet, c.key, c.column))
        problems.append(f"{len(ordered)} unexpected cell change(s):")
        problems += [
            f"    {c.sheet} {c.key} {c.column}: {c.before!r} -> {c.after!r}"
            for c in ordered[:15]
        ]
        if len(ordered) > 15:
            problems.append(f"    ... and {len(ordered) - 15} more")

    for label, actual, expected in (
        ("added row", delta.added_rows, {RowChange(s, tuple(normalise(k) for k in key)) for s, key in added_rows}),
        ("removed row", delta.removed_rows, {RowChange(s, tuple(normalise(k) for k in key)) for s, key in removed_rows}),
        ("added column", delta.added_columns, set(added_columns)),
        ("removed column", delta.removed_columns, set(removed_columns)),
    ):
        for item in sorted(actual - expected, key=str)[:10]:
            problems.append(f"unexpected {label}: {item}")
        for item in sorted(expected - actual, key=str)[:10]:
            problems.append(f"expected {label} that did not happen: {item}")

    for label, items in (
        ("added sheet", delta.added_sheets),
        ("removed sheet", delta.removed_sheets),
    ):
        for item in sorted(items):
            problems.append(f"unexpected {label}: {item}")

    if problems:
        raise AssertionError(
            "workbook delta did not match:\n"
            + "\n".join(f"  {p}" for p in problems)
            + "\n\nfull delta:\n"
            + delta.describe()
        )
