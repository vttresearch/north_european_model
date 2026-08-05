"""The assertion vocabulary for route tests.

Ordered by rule R6 -- contract > relational > provenance > derivational > delta
> counting > pinned. Reach for the earliest family that can express what the
test is about, because the earlier ones survive schema growth and the later ones
do not.

The centrepiece is :func:`assert_workbook_consistent`. It states every
cross-sheet invariant of a valid ``inputData.xlsx`` in one call, so adding a
parameter column costs zero edits here while adding a *relationship* costs one
edit and covers every route test at once. Every route test calls it first, and a
route test that calls only it is still a useful test.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from tests._common.excel_read import EXPECTED_SHEETS, sheet, sheet_keys

# ---------------------------------------------------------------------------
# Structural
# ---------------------------------------------------------------------------


def assert_has_columns(df: pd.DataFrame, columns: Iterable[str], *, where: str = "") -> None:
    """Assert `columns` are present. A superset is fine -- see rule R3."""
    prefix = f"{where}: " if where else ""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise AssertionError(
            f"{prefix}missing column(s) {missing}; present: {list(df.columns)}"
        )


def assert_lacks_columns(df: pd.DataFrame, columns: Iterable[str], *, where: str = "") -> None:
    prefix = f"{where}: " if where else ""
    present = [c for c in columns if c in df.columns]
    if present:
        raise AssertionError(f"{prefix}unexpected column(s) {present}")


def assert_unique_key(df: pd.DataFrame, key_columns: Sequence[str], *, where: str = "") -> None:
    prefix = f"{where}: " if where else ""
    key_columns = [c for c in key_columns if c in df.columns]
    if not key_columns or df.empty:
        return
    duplicated = df.duplicated(subset=key_columns, keep=False)
    if duplicated.any():
        examples = df.loc[duplicated, key_columns].head(3).to_dict("records")
        raise AssertionError(
            f"{prefix}{int(duplicated.sum())} row(s) share a key on {key_columns}; "
            f"first: {examples}"
        )


def assert_fake_multiindex(raw_df: pd.DataFrame, dimensions: Sequence[str]) -> None:
    """Assert the GDXXRW marker row: blank in dimensions, name repeated elsewhere.

    Pinned exactly -- this IS the format contract with GDXXRW
    (create_fake_MultiIndex, bb_excel_pipeline.py:1627-1632), not an incidental
    layout choice.
    """
    if raw_df.empty:
        raise AssertionError("expected a marker row, got an empty sheet")
    first = raw_df.iloc[0]
    for column in raw_df.columns:
        if str(column).startswith("Unnamed") or str(column).startswith("The "):
            continue
        if column in dimensions:
            if not pd.isna(first[column]):
                raise AssertionError(
                    f"dimension column {column!r} must be blank in the marker row, "
                    f"got {first[column]!r}"
                )
        elif str(first[column]).strip() != str(column).strip():
            raise AssertionError(
                f"parameter column {column!r} must repeat its own name in the "
                f"marker row, got {first[column]!r}"
            )


# ---------------------------------------------------------------------------
# Relational -- the workhorse
# ---------------------------------------------------------------------------


def _norm(value: Any) -> Any:
    """Normalise a cell for comparison: text compared case-insensitively.

    The pipeline lowercases in several places and a test should not encode
    *where* that happens.
    """
    if isinstance(value, str):
        return value.strip().casefold()
    return value


def rows_for(df: pd.DataFrame, **key: Any) -> pd.DataFrame:
    """Rows matching every ``column=value`` pair, compared leniently.

    Pass ``None`` to select rows where the cell is missing. In the source stage
    an empty cell is ``pd.NA`` and *not* the empty string, so matching on ``""``
    would quietly find nothing -- the same NA/blank distinction the pipeline
    depends on, applied to the test helper.
    """
    if df.empty:
        return df
    mask = pd.Series(True, index=df.index)
    for column, wanted in key.items():
        if column not in df.columns:
            raise AssertionError(
                f"no column {column!r}; present: {list(df.columns)}"
            )
        if wanted is None:
            mask &= df[column].isna()
        else:
            mask &= df[column].map(_norm) == _norm(wanted)
    return df[mask]


def cell(df: pd.DataFrame, column: str, **key: Any) -> Any:
    """The single value of `column` on the unique row matching `key`.

    Fails loudly when the key selects zero or several rows: a silent ``.iloc[0]``
    on an ambiguous match is how a test starts asserting something other than
    what it says.
    """
    matched = rows_for(df, **key)
    if len(matched) != 1:
        raise AssertionError(
            f"expected exactly one row for {key}, got {len(matched)}"
        )
    if column not in df.columns:
        raise AssertionError(f"no column {column!r}; present: {list(df.columns)}")
    return matched.iloc[0][column]


def assert_referential(
    child: pd.DataFrame,
    child_col: str,
    parent: pd.DataFrame,
    parent_col: str,
    *,
    label: str = "",
) -> None:
    """Every non-blank value of ``child[child_col]`` appears in ``parent[parent_col]``."""
    if child.empty or child_col not in child.columns:
        return
    if parent_col not in parent.columns:
        raise AssertionError(f"{label}: parent has no column {parent_col!r}")

    known = {_norm(v) for v in parent[parent_col].dropna()}
    seen = {_norm(v) for v in child[child_col].dropna()}
    seen.discard("")
    orphans = sorted(str(v) for v in seen - known)
    if orphans:
        raise AssertionError(
            f"{label or f'{child_col} -> {parent_col}'}: "
            f"{len(orphans)} value(s) not declared in the parent sheet: "
            f"{orphans[:5]}{' ...' if len(orphans) > 5 else ''}"
        )


#: (child sheet, child column, parent sheet, parent column) for every
#: cross-sheet reference a valid inputData.xlsx must satisfy. Adding a
#: relationship to the pipeline is one entry here and covers every route test.
_REFERENCES: tuple[tuple[str, str, str, str], ...] = (
    ("p_gnu_io", "grid", "grid", "grid"),
    ("p_gnu_io", "node", "node", "node"),
    ("p_gnu_io", "unit", "unit", "unit"),
    ("p_unit", "unit", "unit", "unit"),
    ("p_gn", "grid", "grid", "grid"),
    ("p_gn", "node", "node", "node"),
    ("p_gnn", "grid", "grid", "grid"),
    ("p_gnn", "from_node", "node", "node"),
    ("p_gnn", "to_node", "node", "node"),
    ("p_gnBoundaryPropertiesForStates", "grid", "grid", "grid"),
    ("p_gnBoundaryPropertiesForStates", "node", "node", "node"),
    ("unitUnittype", "unit", "unit", "unit"),
    ("unitUnittype", "unittype", "unittype", "unittype"),
    ("flowUnit", "unit", "unit", "unit"),
    ("flowUnit", "flow", "flow", "flow"),
    ("effLevelGroupUnit", "unit", "unit", "unit"),
    ("p_nEmission", "node", "node", "node"),
    ("p_nEmission", "emission", "emission", "emission"),
    ("gnGroup", "grid", "grid", "grid"),
    ("gnGroup", "node", "node", "node"),
    ("gnGroup", "group", "group", "group"),
    ("ts_emissionPriceChange", "emission", "emission", "emission"),
    ("p_userconstraint", "group", "group", "group"),
)


def assert_workbook_consistent(sheets: Mapping[str, pd.DataFrame]) -> None:
    """Every cross-sheet invariant of a valid ``inputData.xlsx``, in one call.

    Checks that all expected sheets exist, that every keyed sheet has unique
    keys, that every cross-sheet reference resolves, and that the unit domain
    agrees between ``unit``, ``p_unit`` and ``p_gnu_io``.

    Deliberately says nothing about *values*: it is immune to a new parameter
    column, a changed default, or an edited fixture, and it fails on exactly the
    kind of breakage that makes GAMS reject the workbook.
    """
    missing = sorted(EXPECTED_SHEETS - set(sheets))
    if missing:
        raise AssertionError(f"workbook is missing sheet(s): {missing}")

    for name, key in sheet_keys(sheets).items():
        assert_unique_key(sheets[name], key, where=f"sheet {name!r}")

    for child_name, child_col, parent_name, parent_col in _REFERENCES:
        child, parent = sheets.get(child_name), sheets.get(parent_name)
        if child is None or parent is None:
            continue
        assert_referential(
            child,
            child_col,
            parent,
            parent_col,
            label=f"{child_name}.{child_col} -> {parent_name}.{parent_col}",
        )

    # nodeBalance and usePrice are mutually exclusive: one enforces an energy
    # balance at the node, the other enables price calculation and disables that
    # balance. docs/dictionary.md:241-263 -- "activating both is invalid".
    # A node with both is accepted by the workbook writer and rejected by the
    # model, far from the row that caused it.
    p_gn = sheets.get("p_gn")
    if p_gn is not None and not p_gn.empty:
        if {"nodeBalance", "usePrice"} <= set(p_gn.columns):
            def _set(series):
                return pd.to_numeric(series, errors="coerce").fillna(0) != 0

            both = _set(p_gn["nodeBalance"]) & _set(p_gn["usePrice"])
            if both.any():
                offenders = p_gn.loc[both, ["grid", "node"]].head(5).to_dict("records")
                raise AssertionError(
                    f"{int(both.sum())} node(s) set both nodeBalance and usePrice, "
                    f"which the model rejects: {offenders}"
                )

    # The unit domain must agree in both directions: a unit declared but never
    # connected, or connected but never declared, breaks the model rather than
    # merely looking untidy.
    units = {_norm(v) for v in sheets["unit"].get("unit", pd.Series(dtype=object)).dropna()}
    for name in ("p_unit", "p_gnu_io"):
        frame = sheets.get(name)
        if frame is None or frame.empty or "unit" not in frame.columns:
            continue
        used = {_norm(v) for v in frame["unit"].dropna()}
        undeclared = sorted(str(v) for v in used - units)
        if undeclared:
            raise AssertionError(
                f"{name} references unit(s) absent from the 'unit' sheet: {undeclared[:5]}"
            )


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def assert_passthrough(
    out_df: pd.DataFrame,
    out_col: str,
    src_df: pd.DataFrame,
    src_col: str,
    *,
    out_key: Mapping[str, Any],
    src_key: Mapping[str, Any],
    rel: float = 1e-9,
) -> None:
    """Assert an output cell equals the source cell it came from.

    Tests that a parameter is *carried* without either the test or the fixture
    naming its value, so editing the ``.wb.txt`` never edits the test -- and it
    checks more than a pinned number would, namely that the value came from
    where you think it did.
    """
    produced = cell(out_df, out_col, **out_key)
    expected = cell(src_df, src_col, **src_key)

    if pd.isna(expected) and pd.isna(produced):
        return
    try:
        assert float(produced) == pytest_approx(float(expected), rel)
    except (TypeError, ValueError):
        if _norm(produced) != _norm(expected):
            raise AssertionError(
                f"{out_col}{dict(out_key)} = {produced!r}, but source "
                f"{src_col}{dict(src_key)} = {expected!r}"
            )


def pytest_approx(value: float, rel: float):
    import pytest

    return pytest.approx(value, rel=rel)
