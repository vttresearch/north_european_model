"""Frame helpers the Backbone input builders share.

Three questions that came up in more than one builder, answered once:

- how a pile of collected rows becomes a finished parameter sheet
  (:func:`build_parameter_sheet`, which closed all five parameter builders in the
  same five acts),
- how a list of collected names becomes a domain sheet
  (:func:`compile_domain_df`, where the project's identity convention lives),
- how a column, or a single cell, is read as a number
  (:func:`coerce_numeric_columns`, :func:`as_float64`, :func:`is_positive`).

They are frame arithmetic, not model knowledge: nothing here needs to know what a
grid or a unit is. The 0 = NA = "not set" convention of the builder still governs
them, and each says where it applies that.
"""

import pandas as pd

import src.utils as utils


def build_parameter_sheet(
    rows: list,
    dimensions: list[str],
    parameters: list[str],
    *,
    sort_by: list[str],
    defaults: dict | None = None,
    must_keep: str | None = None,
    ) -> pd.DataFrame:
    """Turn collected rows into a finished parameter sheet.

    The same five acts closed all five parameter builders: frame the rows over
    ``dimensions + parameters``, apply the non-zero defaults, standardize and
    fill, drop the parameter columns nothing set, sort case-insensitively.
    They are here once so that a change to the order is a change in one place;
    the builders differ in what they collect, which is the part worth reading.

    `defaults` are applied before the NA fill, so a parameter whose Backbone
    default is not 0 gets that value on every row that did not state one --
    including rows from a source carrying no such column at all. Cast to
    Float64 first, or fillna on an object column raises a downcasting
    FutureWarning.

    `must_keep` is the parameter kept even when empty, so the Cdim=1 column
    dimension always has a member. ``None`` means the sheet has no column
    dimension to protect here and no drop is applied -- either because the sheet
    has none, or because the caller has more to add before it can be dropped
    safely (see create_p_unit's clamp).
    """
    sheet = pd.DataFrame(rows, columns=dimensions + parameters)

    for parameter, default in (defaults or {}).items():
        if parameter in sheet.columns:
            sheet[parameter] = sheet[parameter].astype('Float64').fillna(default)

    sheet = utils.fill_numeric_na(utils.standardize_df_dtypes(sheet))

    if must_keep is not None:
        sheet = utils.drop_empty_parameter_columns(sheet, parameters, must_keep)

    return sheet.sort_values(
        by=sort_by,
        key=lambda col: col.str.lower() if col.dtype == 'object' else col,
    )


def compile_domain_df(
    values: list,
    domain: str
    ) -> pd.DataFrame:
    """
    Produce the final single-column domain DataFrame ready to write to the Backbone input Excel.

    Deduplicates case-insensitively (first occurrence wins) and sorts alphabetically.
    This is the last step before output -- call this once all sources have been gathered
    into a flat list.

    Parameters:
    - values: list of domain values (strings) collected from all sources
    - domain: name for the output column

    Returns:
    - pd.DataFrame with one column named `domain`, or empty DataFrame if no values
    """
    if not values:
        return pd.DataFrame()

    domain_mapping = {}
    for d in values:
        if isinstance(d, str):
            lower_d = d.lower()
            if lower_d not in domain_mapping:
                domain_mapping[lower_d] = d

    if not domain_mapping:
        return pd.DataFrame()

    result = pd.DataFrame({domain: list(domain_mapping.values())})
    result = result.sort_values(by=domain, key=lambda x: x.str.lower()).reset_index(drop=True)
    return result


def as_float64(column: pd.Series) -> pd.Series:
    """The column as Float64, anything unparseable becoming NA."""
    return pd.to_numeric(column, errors='coerce').astype('Float64')


def coerce_numeric_columns(
    df: pd.DataFrame,
    parameters: list,
    *,
    prefix: str | None = None,
    skip_all_na: bool = False,
    ) -> pd.DataFrame:
    """`df` with its named parameter columns cast to Float64.

    Four of the five blocks in BBExcelPipeline._coerce_numeric_dtypes were this
    loop with a different set of names; the fifth is df_unitdata, which has a rule
    of its own and stays written out there.

    `prefix` additionally takes every column whose name starts with it, which is
    how the open-ended ``emission_XX`` family is caught -- by the same test
    create_p_nEmission uses to find it.

    `skip_all_na` leaves a column nothing set alone rather than typing it Float64,
    because an all-NA column is ``object`` and that is what says no assumption has
    been made about it.

    An empty frame is returned unchanged: there is nothing to make an assumption
    about, and typing its columns would make one.
    """
    if df is None or df.empty:
        return df

    wanted = {p.lower() for p in parameters}
    df = df.copy()
    for col in df.columns:
        col_l = col.lower()
        if not (col_l in wanted or (prefix is not None and col_l.startswith(prefix))):
            continue
        if skip_all_na and df[col].isna().all():
            continue
        df[col] = as_float64(df[col])
    return df


def is_positive(value) -> bool:
    """Is this value a number above zero?

    The one spelling of a question three call sites used to ask three ways
    (``isinstance(v, (int, float)) and v > 0``, a ``pd.notna`` variant of the
    same, and ``pd.to_numeric(...) > 0``). ``0`` is "not set" by the time a
    boundary reaches the builder, so zero is silence, and so is anything that is
    not a number -- including ``pd.NA``, which raises rather than answers if it
    reaches an ``if`` on its own.
    """
    number = pd.to_numeric(value, errors='coerce')
    return bool(pd.notna(number) and number > 0)
