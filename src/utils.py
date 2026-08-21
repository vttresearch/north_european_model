import sys
import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype


#: Marks something in a source workbook as the author's, not the model's.
#:
#: One marker, two placements, same meaning: in a **column header** it ignores
#: that column; in a **data row** it ignores that row. Both require the author to
#: type it deliberately, so nothing is skipped by accident.
#:
#: Two hashes rather than one. A single '#' is how every Excel error value starts
#: -- ``#REF!``, ``#N/A``, ``#DIV/0!`` -- so a broken formula used to read as a
#: comment and delete the row it sat in, silently. It also collides with ordinary
#: text like ``# of units``. No Excel error value begins with ``##``, and '#' is
#: not a character Excel treats as the start of a formula, unlike ``=``, ``+``,
#: ``-`` and ``@``.
IGNORE_MARKER = "##"


def ignored_row_mask(df: pd.DataFrame) -> pd.Series:
    """Rows the author marked with ``IGNORE_MARKER`` in any of their cells.

    Columns whose name starts with ``_`` are provenance the loader added, not
    something anyone typed, so they are not consulted.

    Shared by ``read_input_excels`` and ``normalize_dataframe`` rather than
    written out twice: the first drops ignored rows before the numeric gate runs,
    the second is the guarantee for anything that calls it directly, and the two
    silently disagreeing about what a comment row is would be a bad way to lose
    data.
    """
    if df.empty:
        return pd.Series(False, index=df.index)

    data_cols = [c for c in df.columns if not str(c).startswith("_")]
    if not data_cols:
        return pd.Series(False, index=df.index)

    return df[data_cols].apply(
        lambda row: any(
            isinstance(v, str) and v.strip().startswith(IGNORE_MARKER) for v in row
        ),
        axis=1,
    )


#: What Excel writes into a cell whose formula failed. None of these is ever a
#: legitimate value, so they are reported wherever they appear -- in a text
#: column as much as a numeric one.
EXCEL_ERROR_VALUES = frozenset({
    "#REF!", "#N/A", "#DIV/0!", "#VALUE!", "#NAME?", "#NUM!",
    "#NULL!", "#SPILL!", "#CALC!", "#GETTING_DATA", "#FIELD!", "#UNKNOWN!",
})


def force_utf8_output() -> None:
    """Make stdout/stderr UTF-8 so log output survives being redirected.

    On Windows, Python writes to a *console* as UTF-8 but falls back to the
    locale encoding (cp1252 here) as soon as the stream is redirected to a file
    or a pipe. The status prefixes in IterationLogger are non-ASCII, so
    redirecting output raised UnicodeEncodeError on the very first log line and
    killed the run before any work started:

        python build_input_data.py src_files config_OT2030.ini > build.log

    The prefixes are deliberate -- warnings get missed without them -- so the
    encoding is what gives way, not the message.

    Safe to call more than once, and a no-op on streams that cannot be
    reconfigured (some IDE and CI wrappers replace sys.stdout with objects that
    have no reconfigure()).
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):
            # Detached or already-closed stream: printing is the caller's
            # problem, and failing here would be worse than the mojibake.
            pass


def parse_sys_args():
    # Instructions in case of mispelled input cmd
    USAGE_MSG = (
        "Usage: python build_input_data.py <input_folder> <config_file>,\n"
        "       e.g. python build_input_data.py src_files config_test.ini"
    )

    # detect legacy key=val syntax
    if any("=" in arg for arg in sys.argv[1:]):
        print(USAGE_MSG)
        sys.exit(1)
    else:
        # strict positional: both required
        parser = argparse.ArgumentParser(
            usage=USAGE_MSG,
            description="NorthEuropeanBackbone Input Builder"
        )
        parser.add_argument(
            "input_folder",
            type=str,
            help="Input folder (e.g. src_files)"
        )
        parser.add_argument(
            "config_file",
            type=str,
            help="Config file name (relative to input_folder)"
        )
        # argparse will print our USAGE_MSG if args are missing
        args = parser.parse_args()
        input_folder = Path(args.input_folder)
        config_file  = Path(input_folder, args.config_file)

        return (input_folder, config_file)


@dataclass(frozen=True)
class MalformedCellReport:
    """What :func:`find_malformed_numeric_cells` found, without phrasing it.

    Holds the finding; the caller decides severity and wording. Same split as
    ``TimeAxisReport`` in ``timeseries_helpers``, for the same reason: the source
    workbook route and the processor route want the same detection and different
    consequences.
    """

    #: Same shape as the frame examined; True where a cell is a failed number.
    mask: pd.DataFrame
    #: Column name -> how many of its cells are malformed.
    counts: Dict[str, int] = field(default_factory=dict)
    #: Column name -> up to three of the offending values, for the message.
    examples: Dict[str, List[str]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True when nothing was found."""
        return not self.counts

    @property
    def total(self) -> int:
        """Total malformed cells across all columns."""
        return sum(self.counts.values())


#: What ``pandas.api.types.infer_dtype`` reports for a column that may contain a
#: string. ``empty`` is an all-NA column, where no assumption has been made yet.
#: Everything absent from this set -- ``floating``, ``integer``, ``boolean``,
#: ``datetime``, ``mixed-integer-float`` -- holds no strings at all.
_MAY_HOLD_TEXT = frozenset({"string", "bytes", "mixed", "mixed-integer", "empty"})


def _holds_raw_text(s: pd.Series) -> bool:
    """Whether a column could still be hiding an unparsed string.

    A typed column -- float, int, bool, datetime -- cannot: it was typed by
    whoever read the file, which means every cell in it already parsed, and a
    stray ``'1,000.0'`` or ``'#REF!'`` would have prevented that.

    The dtype alone is not enough, though. An ``object`` column may hold nothing
    but floats (or bools, or datetimes), and the ``.str`` accessor in
    :func:`find_excel_error_values` raises ``AttributeError: Can only use .str
    accessor with string values!`` on exactly that. ``infer_dtype`` looks at the
    values rather than the label and answers the question actually being asked.

    It is also what keeps the cost down: on a multi-million-row frame the typed
    columns are the bulk of it, and this retires them without a per-cell pass.
    """
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return False
    return pd.api.types.infer_dtype(s, skipna=True) in _MAY_HOLD_TEXT


#: Characters a number may legitimately open with before its first digit: sign,
#: currency, accounting parenthesis. Anything else in first position means the
#: value was never meant to be a number.
_NUMERIC_LEAD_CHARS = " \t+-−(¤$€£¥"


def _is_failed_number(value) -> bool:
    """Whether a cell looks like a number that did not parse.

    A value counts when, after any sign or currency symbol, it *starts with a
    digit*. That is what separates the two ways a column ends up mixed:

    - ``['1,000.0', 500, 250]`` -- a numeric column with a typo in it. The
      offending cell opens with a digit, so it is a *number* pandas could not
      read: a thousands separator, a unit suffix, a stray percent sign.
    - ``['1', 'chp1', 'dh2']`` -- an ordinary label column that happens to
      contain a numeric-looking value. These open with a letter.

    "Contains a digit" is not enough, and picking the weaker test was a real
    mistake caught in testing: it blanked ``chp1`` and ``dh2`` out of an
    identifier column, which is exactly the damage this function exists to
    prevent. Leading position is the discriminator that survives ``100 MW``
    and ``(500)`` while leaving labels alone.

    Deliberately not claimed: a purely textual value dropped into a numeric
    column -- ``'unknown'`` in a capacity column -- is not caught here. That is a
    different mistake, and no content-only rule can tell it from a text column
    without risking the labels above.
    """
    if not isinstance(value, str):
        return False
    stripped = value.lstrip(_NUMERIC_LEAD_CHARS)
    return bool(stripped) and stripped[0].isdigit()


def find_malformed_numeric_cells(df: pd.DataFrame) -> MalformedCellReport:
    """Find cells that should hold a number and do not.

    A column is reported when it holds **both** at least one value that parses
    as a number and at least one that looks like a number but does not parse
    (see :func:`_is_failed_number`). Only the failing cells are listed, never the
    whole column.

    Why this matters more than the crash that prompted it
    ----------------------------------------------------
    ``standardize_df_dtypes`` converts a column to numeric only when
    ``pd.to_numeric`` introduces no new NA, so one unparseable cell leaves the
    *entire* column ``object`` -- and a dozen places downstream branch on dtype.
    The column then silently changes behaviour rather than failing: a sheet whose
    only numeric column is poisoned is dropped whole by
    ``filter_nonzero_numeric_rows``, and ``normalize_dataframe``'s ``_output1``
    rename stops firing, which loses the capacity column outright. A visible
    ``TypeError`` is the lucky outcome.

    Structural on purpose: no column list and no ``{column: dtype}`` map, which
    would need editing on every schema change (tests/README.md, R7). The rule is
    stated as a property of the values.

    Parameters
    ----------
    df : pandas.DataFrame
        Any frame. Columns whose name starts with ``_`` are provenance columns
        and are skipped; already-numeric columns have nothing to parse.

    Returns
    -------
    MalformedCellReport
        ``.ok`` is True when the frame is clean.
    """
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    counts: Dict[str, int] = {}
    examples: Dict[str, List[str]] = {}

    for col in df.columns:
        if str(col).startswith("_"):
            continue
        s = df[col]
        if not _holds_raw_text(s):
            continue

        # Count, convert, count again -- and only look at individual cells when
        # the two counts disagree. Everything down to `failed` is vectorised;
        # `_is_failed_number` runs per cell in Python and would dominate on a
        # multi-million-row frame if it ran over whole columns. Two counts are
        # enough to retire the two cases that make up virtually every column:
        # nothing parsed at all (a label column) and everything parsed (a clean
        # numeric column). Only a column that is genuinely part numeric and part
        # not reaches the per-cell test, and then only on its failing rows.
        present = s.notna()
        n_present = int(present.sum())
        if n_present == 0:
            continue

        parsed = pd.to_numeric(s, errors="coerce")
        n_parsed = int(parsed.notna().sum())
        if n_parsed == 0 or n_parsed == n_present:
            continue

        failed = present & parsed.isna()
        offenders = s[failed]
        # Whitespace-only cells land here too and are filtered out by
        # _is_failed_number, which is why no separate emptiness pass is needed.
        looks_numeric = offenders.map(_is_failed_number).astype(bool)
        if not looks_numeric.any():
            continue

        suspect = pd.Series(False, index=df.index)
        suspect.loc[offenders.index[looks_numeric]] = True

        mask[col] = suspect
        counts[col] = int(suspect.sum())
        examples[col] = [str(v) for v in s[suspect].unique()[:3]]

    return MalformedCellReport(mask=mask, counts=counts, examples=examples)


def find_excel_error_values(df: pd.DataFrame) -> MalformedCellReport:
    """Find cells holding an Excel error value such as ``#REF!`` or ``#DIV/0!``.

    Separate from :func:`find_malformed_numeric_cells` because these are not
    failed numbers -- they are what Excel leaves behind when a formula breaks,
    and they are equally wrong in a text column. ``#REF!`` in particular is what
    appears when someone deletes a column another sheet referred to, so it marks
    a workbook that has quietly lost a reference.
    """
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    counts: Dict[str, int] = {}
    examples: Dict[str, List[str]] = {}

    for col in df.columns:
        if str(col).startswith("_"):
            continue
        s = df[col]
        if not _holds_raw_text(s):
            continue

        # Every Excel error value starts with '#', so one vectorised prefix test
        # retires the whole column before any per-cell work happens. Same reason
        # as in find_malformed_numeric_cells: these frames run to millions of rows.
        starts_with_hash = s.str.match(r"\s*#", na=False)
        if not starts_with_hash.any():
            continue

        candidates = s[starts_with_hash]
        is_error = candidates.map(
            lambda v: isinstance(v, str) and v.strip().upper() in EXCEL_ERROR_VALUES
        ).astype(bool)
        if not is_error.any():
            continue

        hit = pd.Series(False, index=df.index)
        hit.loc[candidates.index[is_error]] = True
        mask[col] = hit
        counts[col] = int(hit.sum())
        examples[col] = [str(v) for v in s[hit].unique()[:3]]

    return MalformedCellReport(mask=mask, counts=counts, examples=examples)


def gate_xlsx_frame(df: pd.DataFrame, source: str, logger) -> pd.DataFrame:
    """Report malformed cells in a freshly read sheet, and blank them.

    Applied at the one place every source workbook passes through, so it is
    unavoidable rather than remembered.

    Blanking rather than interpreting is deliberate. ``1.000`` is a thousand to a
    German author and one to an English one, and the cell carries nothing that
    says which; a parser that guesses would put a confidently wrong number into
    the model. ``pd.NA`` means "not set", which the rest of the pipeline already
    knows how to carry, and the logged message names the cell so the author can
    fix it at the source.

    The level is ``error`` rather than ``warn`` because ``logger.has_errors``
    feeds ``workflow_run_successfully`` in the cache flags: the build still
    finishes and still writes its output, but it is marked failed, forces a full
    rerun next time, and is repeated in the run summary. A warning would let a
    wrong capacity through green.

    Blanking the cells here also makes ``standardize_df_dtypes`` do the right
    thing by itself further down: with the offending values already NA, its
    "convert only if no new NAs appear" rule no longer has a reason to refuse,
    and the column becomes ``Float64`` as it was always meant to.

    Parameters
    ----------
    df : pandas.DataFrame
        A sheet, after ``read_input_excels`` has done its column cleaning.
    source : str
        Where the sheet came from, as ``file.xlsx:sheetname``. Goes into the
        message; it is the finest locator the source stage has.
    logger : IterationLogger

    Returns
    -------
    pandas.DataFrame
        `df` with every reported cell set to ``pd.NA``.
    """
    df = df.copy()

    for report, what, consequence in (
        (find_malformed_numeric_cells(df), "malformed number", "treated as not set"),
        (find_excel_error_values(df), "Excel error value", "treated as not set"),
    ):
        if report.ok:
            continue
        for col, count in report.counts.items():
            shown = ", ".join(repr(v) for v in report.examples[col])
            logger.log_status(
                f"[{source}] Column '{col}': {count} {what}(s) -- {shown}. "
                f"Each is {consequence}; fix the cell in the source workbook.",
                level="error",
            )
        df = df.mask(report.mask, other=pd.NA)

    return df


def standardize_df_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column dtypes to a consistent set:
    - Replace 'NaN' strings with pd.NA
    - Attempt to convert object columns containing numeric strings to numeric
    - Empty columns (all NA) → object
    - Numeric columns → Float64
    - Everything else → object

    NA values are preserved

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y'], 'c': [None, None]})
    >>> df = standardize_df_dtypes(df)
    >>> df.dtypes
    a    Float64
    b     object
    c     object
    dtype object

    # 'NaN' strings are treated as NA:
    #   ['NaN', '2'] becomes Float64 with [NA, 2.0]
    #   ['x', 'NaN'] becomes object with ['x', NA]
    """
    df = df.copy()

    # First pass: replace 'NaN' strings with pd.NA and identify empty columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(
                lambda x: pd.NA if isinstance(x, str) and x.strip().lower() == 'nan' else x
            )
        if is_col_empty(df[col]):
            df[col] = df[col].astype("object")

    # Second pass: try to convert object columns to numeric
    for col in df.columns:
        if df[col].dtype == 'object' and not df[col].isna().all():
            converted = pd.to_numeric(df[col], errors="coerce")
            # Only convert if no new NAs were introduced
            if converted.isna().sum() == df[col].isna().sum():
                df[col] = converted

    # Third pass: standardize dtypes
    for col in df.columns:
        if df[col].isna().all():
            df[col] = df[col].astype("object")
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("Float64")
        else:
            df[col] = df[col].astype("object")

    # Fourth pass: replace numpy NaN with pd.NA in object columns
    # Float64 columns already use pd.NA natively, but object columns
    # can still contain float('nan').
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].where(df[col].notna(), other=pd.NA)

    return df


def fill_numeric_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NA values with 0 in numeric (Float64) columns only.

    This avoids the FutureWarning from filling NA on mixed-dtype DataFrames.
    """
    df = df.copy()
    float_cols = df.select_dtypes(include=['Float64']).columns
    df[float_cols] = df[float_cols].fillna(0)
    return df


def fill_all_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill all NA values in a DataFrame:
      - Numeric columns (Float64, int, etc.): fill with 0
      - All other columns (object, string, …): fill with ''

    Use this at pipeline boundaries to eliminate pd.NA from output DataFrames
    so that downstream code can use simple value comparisons without NA guards.
    """
    df = df.copy()
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("")
    return df


def is_col_empty(s: pd.Series) -> bool:
    """
    Determine whether a pandas Series should be considered "empty."
    NaN values are always treated as empty.

    Rules:
    ------
    - Boolean columns: considered empty only if all values are NaN.
      (All-False is NOT empty.)
    - Numeric columns (excluding bool): NaNs treated as 0, so "all zero or NaN" means empty.
    - Non-numeric columns: NaN or "" (empty/whitespace-only string) counts as empty.

    Parameters
    ----------
    s : pd.Series
        The column (Series) to test.

    Returns
    -------
    bool
        True if the column is "empty" according to the above rules, False otherwise.
    """
    if len(s) == 0:
        return True

    # Booleans: usually don't drop just because all False; only NaNs count as empty
    # bool(...) throughout: pandas reductions return numpy.bool_, and the
    # annotation promises a plain bool. Callers that use `is True` or serialise
    # the result get the wrong answer otherwise.
    if is_bool_dtype(s):
        return bool(s.isna().all())

    # Numeric (excluding bool): empty if all zeros or NaN
    if is_numeric_dtype(s) and not is_bool_dtype(s):
        return bool((s.fillna(0) == 0).all())

    # Non-numeric: empty if all are NaN or whitespace-only strings
    na_mask = s.isna()
    # Safe elementwise test; no vectorized == on arbitrary objects
    empty_str_mask = s.map(lambda v: isinstance(v, str) and v.strip() == "")
    return bool((na_mask | empty_str_mask).all())


def drop_empty_parameter_columns(
    df: pd.DataFrame,
    parameters: list,
    must_keep: str,
    ) -> pd.DataFrame:
    """
    Drop all-empty parameter columns from a fake-MultiIndex sheet, but never the last one.

    For these sheets the parameter block *is* the column dimension -- indexSheet.xlsx
    declares p_gn as Rdim=2/Cdim=1, p_gnn as Rdim=3/Cdim=1, p_gnu_io as Rdim=4/Cdim=1.
    A sheet left with no parameter column is therefore a GDXXRW dimension error rather
    than an empty sheet, so `must_keep` is retained even when it is empty.

    Only names listed in `parameters` are considered, which is what keeps a dimension
    column out of reach: is_col_empty() is True for a zero-length column, so a build
    that produced no rows would otherwise drop every column in the frame -- dimensions
    included, leaving the later sort_values() to fail with a bare KeyError.

    Parameters
    ----------
    df : pd.DataFrame
        Sheet frame, before the fake MultiIndex is added.
    parameters : list
        The sheet's parameter names (PARAM_GN, PARAM_GNN, PARAM_GNU). Anything not
        in this list is left alone whether it is empty or not.
    must_keep : str
        Parameter kept even when empty, so the column dimension always has a member.

    Returns
    -------
    pd.DataFrame
        `df` without its all-empty parameter columns.
    """
    droppable = [
        col for col in parameters
        if col in df.columns and col != must_keep and is_col_empty(df[col])
    ]
    return df.drop(columns=droppable)