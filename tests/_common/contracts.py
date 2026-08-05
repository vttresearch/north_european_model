"""Executable postconditions for the pipeline's dtype / NA conventions.

Why this module exists
----------------------
GAMS has no NaN, and plain ``0`` *is* empty.  Python is precise about types;
GAMS is not.  Every bug in this class lives at the seam between those two
worlds, and a function tested in isolation can be perfectly correct while still
handing the next function something it cannot digest.

So the shape of a DataFrame is asserted by a *contract* applied uniformly, not
by assertions written one test at a time.  ``utils.standardize_df_dtypes``
already defines the canonical post-normalization shape; this module turns its
docstring into something that fails a test.

The all-NA rule
---------------
An all-``pd.NA`` column is ``object``, never ``Float64``.  This is deliberate
and it is the fix for a real cascade bug: empty string columns and empty float
columns both became all-NA, both were then inferred to float, and code
expecting a specific dtype crashed.  ``object`` on an all-NA column means
**no assumption has been made**.

The corollary is that the burden sits on *consumers*: every function must
tolerate an all-NA object column where it would normally see ``Float64``.  That
property is swept in ``tests/unit/source_data/test_contract_sweep.py``.

What is deliberately NOT asserted
---------------------------------
Only guarantees the code actually makes.  ``standardize_df_dtypes`` converts the
string ``'nan'`` to NA (utils.py:75) but leaves ``'NA'``, ``'None'`` and
``'null'`` alone -- and rightly so, since ``'NA'`` could be a legitimate value.
Asserting those away would fail on behaviour that was never promised, which is
how a contract turns into noise and then gets deleted.

There is no ``{column: dtype}`` map here, and there must never be one: it would
need editing on every schema change and would re-create the very assumption the
all-NA rule exists to avoid.  Contracts are stated as *properties*.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

#: The only dtypes ``standardize_df_dtypes`` is allowed to leave behind
#: (utils.py:88-95).  ``string`` appears as ``string[python]``, hence the prefix
#: match in :func:`_dtype_is_allowed`.
ALLOWED_DTYPES = ("Float64", "object", "string")


# ---------------------------------------------------------------------------
# The nasty-value catalogue
# ---------------------------------------------------------------------------

#: Cell values that have caused, or plausibly could cause, a live bug.
#:
#: Deliberately a fixed, curated list rather than a property-based generator:
#: a failure here is reproducible from the test id alone, which matters more in
#: a suite whose whole purpose is to stay cheap to maintain.  It doubles as
#: documentation of what users actually type into the source workbooks.
#:
#: **When a live bug is found, add the value that caused it.**  That is how each
#: incident becomes permanently regression-tested.
NASTY_CELLS: list[Any] = [
    # -- emptiness, in all its spellings
    "",
    "   ",
    None,
    np.nan,
    pd.NA,
    # -- the string/real confusion. Only 'nan' is converted (utils.py:75);
    #    the rest ride through as text and must not break anything.
    "nan",
    "NaN",
    "NA",
    "None",
    "null",
    # -- zero vs empty: the GAMS convention's sharpest edge
    0,
    0.0,
    -0.0,
    "0",
    "0.0",
    # -- numbers that arrive as text, including the European decimal comma
    1,
    1.0,
    "1",
    " 1 ",
    "1e3",
    "5,5",
    # -- booleans, which pandas counts as numeric
    True,
    False,
    "TRUE",
    "yes",
    # -- values with meaning to the pipeline itself
    "#comment",   # normalize_dataframe drops rows whose cells start with '#'
    "a_b",        # drop_underscore_values deletes rows containing '_'
    "all",        # apply_whitelist / expand_all_country magic value
    "ALL",
    "All",
    # -- the arithmetic edge cases
    float("inf"),
    float("-inf"),
    # -- shapes that break naive string handling
    "åäö",
    "x" * 300,
    datetime(2030, 1, 1),
]


def nasty_id(value: Any) -> str:
    """A short, stable pytest id for a catalogue entry.

    ``repr`` alone produces unreadable ids for the long string and collides
    ``nan``/``NaN``, so the type is folded in where it disambiguates.
    """
    if isinstance(value, str) and len(value) > 20:
        return f"str_len{len(value)}"
    if value is None:
        return "None"
    if value is pd.NA:
        return "pd.NA"
    if isinstance(value, float) and math.isnan(value):
        return "np.nan"
    if isinstance(value, bool):
        return f"bool_{value}"
    if isinstance(value, datetime):
        return "datetime"
    return f"{type(value).__name__}_{value!r}"


# ---------------------------------------------------------------------------
# Frame builders for the sweeps
# ---------------------------------------------------------------------------


def frame_with_cell(
    columns: Sequence[str],
    cell: Any,
    *,
    rows: int = 2,
    filler: Any = "x",
) -> pd.DataFrame:
    """A frame whose **first row** is `cell` in every column.

    The second row carries `filler` so that the frame is not accidentally
    all-NA -- otherwise every column would legitimately be ``object`` and the
    sweep would prove nothing.
    """
    data = {col: [cell] + [filler] * (rows - 1) for col in columns}
    return pd.DataFrame(data)


def frame_with_blank_column(
    columns: Sequence[str],
    blank_column: str,
    *,
    rows: int = 2,
    filler: Any = "x",
) -> pd.DataFrame:
    """A frame where one column is entirely ``pd.NA`` -- the 'assume nothing' state.

    This is the shape that caused the cascade bug, and the input for the
    consumer-tolerance sweep.
    """
    data = {col: [filler] * rows for col in columns}
    data[blank_column] = [pd.NA] * rows
    return pd.DataFrame(data)


# ---------------------------------------------------------------------------
# The contract itself
# ---------------------------------------------------------------------------


def _dtype_is_allowed(dtype: Any) -> bool:
    name = str(dtype)
    return name in ALLOWED_DTYPES or name.startswith("string")


def _is_real_nan(value: Any) -> bool:
    """True for ``float('nan')`` / ``np.nan``, false for ``pd.NA``.

    The distinction is the whole point: ``pd.NA`` is the intended missing
    marker, a bare float NaN inside an object column is the bug.
    """
    return isinstance(value, float) and math.isnan(value)


def describe_dtypes(df: pd.DataFrame) -> str:
    """Compact ``column: dtype`` listing for failure messages."""
    return ", ".join(f"{col}: {df[col].dtype}" for col in df.columns) or "(no columns)"


def assert_normalized(df: Any, *, where: str = "", require_clean_index: bool = False) -> None:
    """Assert `df` satisfies the post-normalization contract.

    Checks, each of which corresponds to a guarantee ``standardize_df_dtypes``
    actually makes:

    1. every column dtype is one of :data:`ALLOWED_DTYPES`;
    2. an all-NA column is ``object`` -- never ``Float64`` (the cascade-bug fix);
    3. object columns hold ``pd.NA`` for missing, never ``None``, ``np.nan`` or
       ``pd.NaT`` (utils.py:100-102);
    4. no ``'nan'`` string survived as text (utils.py:74-76);
    5. no duplicate column names.

    Parameters
    ----------
    require_clean_index:
        Additionally require a 0..n-1 ``RangeIndex``.  Off by default because
        several loader functions legitimately return sliced frames; switch it on
        where a clean index is genuinely part of the contract.

    The failure message names the column, its dtype and the first offending
    row, because a contract nobody can diagnose is a contract that gets deleted.
    """
    prefix = f"{where}: " if where else ""

    if not isinstance(df, pd.DataFrame):
        raise AssertionError(f"{prefix}expected a DataFrame, got {type(df).__name__}")

    duplicates = df.columns[df.columns.duplicated()].tolist()
    if duplicates:
        raise AssertionError(f"{prefix}duplicate column name(s): {duplicates}")

    for col in df.columns:
        series = df[col]
        dtype = series.dtype

        if not _dtype_is_allowed(dtype):
            raise AssertionError(
                f"{prefix}column {col!r} has dtype {dtype}, expected one of "
                f"{list(ALLOWED_DTYPES)}.\n  all dtypes -> {describe_dtypes(df)}"
            )

        if len(series) and series.isna().all() and str(dtype) != "object":
            raise AssertionError(
                f"{prefix}column {col!r} is entirely NA but has dtype {dtype}; "
                f"an all-NA column must be 'object'.\n"
                f"  'object' on an all-NA column means 'no assumption made'. "
                f"Typing it Float64 is the cascade bug: an empty string column "
                f"and an empty float column become indistinguishable, and the "
                f"next consumer crashes on the dtype it did not expect."
            )

        if str(dtype) == "object":
            for position, value in enumerate(series):
                if value is None:
                    raise AssertionError(
                        f"{prefix}column {col!r} row {series.index[position]} holds "
                        f"None; object columns must use pd.NA for missing."
                    )
                if _is_real_nan(value):
                    raise AssertionError(
                        f"{prefix}column {col!r} row {series.index[position]} holds "
                        f"a float NaN; object columns must use pd.NA for missing."
                    )
                if value is pd.NaT:
                    raise AssertionError(
                        f"{prefix}column {col!r} row {series.index[position]} holds "
                        f"pd.NaT; object columns must use pd.NA for missing."
                    )
                if isinstance(value, str) and value.strip().lower() == "nan":
                    raise AssertionError(
                        f"{prefix}column {col!r} row {series.index[position]} holds "
                        f"the string {value!r}; it should have become pd.NA."
                    )

    if require_clean_index:
        expected = pd.RangeIndex(len(df))
        if not df.index.equals(expected):
            raise AssertionError(
                f"{prefix}expected a clean RangeIndex(0..{len(df) - 1}), got "
                f"{df.index!r}"
            )


def assert_gams_ready(
    df: pd.DataFrame,
    *,
    dimensions: Sequence[str] = (),
    value_col: str = "value",
    where: str = "",
) -> None:
    """Assert `df` is safe to hand to gams.transfer -- boundary 7 of the NA/zero map.

    GAMS has no NaN and a plain ``0`` *is* empty, so by this point:

    1. `value_col` is numeric;
    2. it holds no NA -- every gap must already have been converted to 0
       *and reported*, not silently carried in;
    3. it holds no ``inf``/``-inf``. GAMS would accept these as INF and the
       model would then fail somewhere far away from the cause;
    4. every dimension value is a non-blank string. ``''`` is not a usable GAMS
       set element, and a blank key is silently wrong rather than loudly wrong.

    Everything *upstream* of this gate may hold NA, and there it means
    "no data". That is deliberate: pandas' ``quantile`` skips NA, so leaving
    gaps alone until here keeps the climatological forecasts honest.
    """
    prefix = f"{where}: " if where else ""

    for dim in dimensions:
        if dim not in df.columns:
            raise AssertionError(f"{prefix}dimension column {dim!r} is missing")
        col = df[dim]
        if col.isna().any():
            first = col.index[col.isna()][0]
            raise AssertionError(
                f"{prefix}dimension {dim!r} row {first} is missing; GAMS set "
                f"elements cannot be blank"
            )
        blank = col.astype("string").str.strip().eq("")
        if blank.any():
            first = col.index[blank][0]
            raise AssertionError(
                f"{prefix}dimension {dim!r} row {first} is blank; '' is not a "
                f"usable GAMS set element"
            )

    if value_col not in df.columns:
        raise AssertionError(f"{prefix}value column {value_col!r} is missing")

    values = df[value_col]
    if not pd.api.types.is_numeric_dtype(values):
        raise AssertionError(
            f"{prefix}{value_col!r} has dtype {values.dtype}, expected numeric"
        )
    if values.isna().any():
        first = values.index[values.isna()][0]
        raise AssertionError(
            f"{prefix}{value_col!r} row {first} is NA. GAMS has no NaN, so every "
            f"gap must be converted to 0 -- and counted in the log -- before this "
            f"point, never carried in silently."
        )
    as_float = values.to_numpy(dtype="float64", na_value=0.0)
    if np.isinf(as_float).any():
        first = values.index[np.isinf(as_float)][0]
        raise AssertionError(
            f"{prefix}{value_col!r} row {first} is non-finite ({values.loc[first]}); "
            f"GAMS accepts INF and the model would fail far from the cause"
        )


def assert_no_na_became_zero(
    before: pd.DataFrame,
    after: pd.DataFrame,
    *,
    where: str = "",
    columns: Iterable[str] | None = None,
) -> None:
    """Assert the source stage kept ``pd.NA`` and ``0`` distinct (boundaries 1-2).

    Compares only rows and columns present in *both* frames, so a transform that
    filters rows or adds columns does not trip it.  Where a cell was NA before
    and survives after, it must not have turned into ``0``.

    This is the invariant that lets ``method=replace`` overwrite a value with a
    genuine zero: if NA and 0 are conflated upstream, the distinction the merge
    logic depends on is already gone.
    """
    prefix = f"{where}: " if where else ""

    shared_cols = [c for c in before.columns if c in after.columns]
    if columns is not None:
        wanted = set(columns)
        shared_cols = [c for c in shared_cols if c in wanted]
    shared_rows = before.index.intersection(after.index)
    if not shared_cols or len(shared_rows) == 0:
        return

    for col in shared_cols:
        was_na = before.loc[shared_rows, col].isna()
        if not was_na.any():
            continue
        now = after.loc[shared_rows, col]
        for idx in shared_rows[was_na]:
            value = now.loc[idx]
            if pd.isna(value):
                continue
            if isinstance(value, (int, float, np.integer, np.floating)) and value == 0:
                raise AssertionError(
                    f"{prefix}column {col!r} row {idx} was pd.NA and is now 0. "
                    f"The source stage must keep NA and 0 distinct -- collapsing "
                    f"them here destroys the difference method=replace relies on. "
                    f"(0 = NA is a GAMS convention and belongs at the "
                    f"inputData.xlsx / GDX boundary, not before it.)"
                )
