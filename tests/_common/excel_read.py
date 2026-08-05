"""Reading ``inputData.xlsx`` back for assertions.

The workbook is written for GDXXRW, not for pandas, so reading it back needs two
undos.

**The fake MultiIndex.** Five sheets carry a second header row: blank in the
dimension columns, repeating the parameter name in the others, with data
starting on row 3 (``create_fake_MultiIndex``, bb_excel_pipeline.py:1601). It is
*detected* here rather than listed, by the same rule ``adjust_excel`` uses --
the first cell of that row is blank. Detecting means a dimension added to a
sheet needs no edit here, and the dimension names are read off the workbook
instead of being pinned in a table that can drift from the writer.

**Trailing explanation columns.** ``adjust_excel`` writes two explanatory
strings two columns to the right of every fake-MultiIndex table (:1806-1807),
which read back as ``Unnamed: N`` columns. They are dropped, the same rule
``read_input_excels`` applies to source workbooks.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping

import pandas as pd

_UNNAMED = re.compile(r"^Unnamed:\s*\d+$")

#: ``adjust_excel`` writes two explanatory strings two columns right of every
#: fake-MultiIndex table (bb_excel_pipeline.py:1806-1807). The first lands in the
#: header row, so pandas reads it as a *named* column and the Unnamed filter
#: misses it; the second becomes that column's only value. Matched on a stable
#: prefix rather than the full sentence, which is free to be reworded.
_EXPLANATION_PREFIXES = (
    "The first row labels are for excel Table headers",
    "The Second row labels are for GDXXRW",
)


def _is_noise_column(name: object) -> bool:
    text = str(name).strip()
    if _UNNAMED.match(text):
        return True
    return any(text.startswith(prefix) for prefix in _EXPLANATION_PREFIXES)

#: Key columns for sheets that do NOT carry a fake MultiIndex; the others have
#: their dimensions read off the workbook. Used to diff row-by-row rather than
#: as opaque sets.
PLAIN_SHEET_KEYS: dict[str, tuple[str, ...]] = {
    "grid": ("grid",),
    "node": ("node",),
    "unit": ("unit",),
    "unittype": ("unittype",),
    "flow": ("flow",),
    "emission": ("emission",),
    "group": ("group",),
    "restype": ("restype",),
    "unitUnittype": ("unit", "unittype"),
    "flowUnit": ("flow", "unit"),
    "effLevelGroupUnit": ("effLevel", "effSelector", "unit"),
    "gnGroup": ("grid", "node", "group"),
    "p_nEmission": ("node", "emission"),
    "ts_emissionPriceChange": ("emission", "group", "t"),
    "p_userconstraint": (
        "group",
        "1st dimension",
        "2nd dimension",
        "3rd dimension",
        "4th dimension",
        "parameter",
    ),
}

#: Sheets every valid inputData.xlsx must contain.
EXPECTED_SHEETS = frozenset(
    {
        "index",
        "add_scen_tags",
        "grid",
        "node",
        "p_gn",
        "p_gnBoundaryPropertiesForStates",
        "p_gnn",
        "unit",
        "unittype",
        "unitUnittype",
        "flowUnit",
        "effLevelGroupUnit",
        "p_gnu_io",
        "p_unit",
        "p_userconstraint",
        "p_nEmission",
        "ts_emissionPriceChange",
        "gnGroup",
        "group",
        "flow",
        "emission",
        "restype",
    }
)


def _has_fake_multiindex(frame: pd.DataFrame) -> bool:
    """True when row 0 is the dimension/parameter marker row rather than data.

    ``adjust_excel`` identifies these sheets by ``A2 is None`` (:1757); read back
    with ``header=0`` that is the first cell of row 0. The extra check that some
    other cell repeats its own column name keeps a genuine data row whose first
    cell happens to be blank from being eaten.
    """
    if frame.empty:
        return False
    first = frame.iloc[0]
    if not pd.isna(first.iloc[0]):
        return False
    return any(
        str(first[col]).strip() == str(col).strip()
        for col in frame.columns
        if not pd.isna(first[col])
    )


def _drop_noise_columns(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, [c for c in frame.columns if not _is_noise_column(c)]]


def dimensions_of(frame: pd.DataFrame) -> tuple[str, ...]:
    """Dimension columns of a fake-MultiIndex sheet, read off the marker row.

    Drops the writer's explanation columns first, so they are never mistaken for
    dimensions regardless of whether the caller cleaned the frame already.
    """
    frame = _drop_noise_columns(frame)
    if not _has_fake_multiindex(frame):
        return ()
    first = frame.iloc[0]
    return tuple(str(col) for col in frame.columns if pd.isna(first[col]))


def read_output_workbook(path: Path) -> dict[str, pd.DataFrame]:
    """Read ``inputData.xlsx`` into ``{sheet_name: DataFrame}``, ready to assert on.

    Fake-MultiIndex marker rows are dropped so data starts at index 0, and the
    trailing ``Unnamed: N`` explanation columns are removed.
    """
    raw = pd.read_excel(Path(path), sheet_name=None, header=0)
    out: dict[str, pd.DataFrame] = {}
    for name, frame in raw.items():
        frame = _drop_noise_columns(frame)
        dims = dimensions_of(frame)
        if dims:
            frame = frame.iloc[1:].reset_index(drop=True)
        # Recorded now: once the marker row is dropped the sheet is
        # indistinguishable from a plain one, so detection cannot run twice.
        frame.attrs["dimensions"] = dims
        out[name] = frame
    return out


def read_output_workbook_raw(path: Path) -> dict[str, pd.DataFrame]:
    """Read the workbook verbatim, marker rows and all.

    For tests about the *format* rather than the data -- the fake MultiIndex is
    a contract with GDXXRW and is asserted directly.
    """
    return pd.read_excel(Path(path), sheet_name=None, header=0)


def sheet_keys(sheets: Mapping[str, pd.DataFrame]) -> dict[str, tuple[str, ...]]:
    """Key columns per sheet: detected for fake-MultiIndex sheets, listed for the rest."""
    keys: dict[str, tuple[str, ...]] = {}
    for name, frame in sheets.items():
        # attrs is set by read_output_workbook; dimensions_of covers a raw frame.
        dims = tuple(frame.attrs.get("dimensions") or ()) or dimensions_of(frame)
        if dims:
            keys[name] = dims
        elif name in PLAIN_SHEET_KEYS:
            present = tuple(c for c in PLAIN_SHEET_KEYS[name] if c in frame.columns)
            if present:
                keys[name] = present
    return keys


def sheet(sheets: Mapping[str, pd.DataFrame], name: str) -> pd.DataFrame:
    """Fetch a sheet, failing with the list of what is actually there."""
    if name not in sheets:
        raise AssertionError(
            f"no sheet {name!r} in the workbook; available: {sorted(sheets)}"
        )
    return sheets[name]
