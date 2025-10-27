import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional, Tuple, Iterable, Sequence, Any
from src.utils import log_status
import math




def build_node_column(
        df: pd.DataFrame
        ) -> pd.DataFrame:
    """
    Add a 'node' column to the DataFrame by concatenating country and grid identifiers.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing at minimum 'country' and 'grid' columns.
        Optional 'node_suffix' column can be included for more specific node naming.
    
    Returns:
    --------
    pandas.DataFrame
        Input DataFrame with new 'node' column added
    
    Raises:
    -------
    ValueError
        If required columns ('country', 'grid') are missing from the DataFrame
    
    Notes:
    ------
    The node format is: "{country}_{grid}" or "{country}_{grid}_{node_suffix}" 
    when node_suffix is present and not empty.
    """
    if df.empty: return df

    # Check that required columns exist in the DataFrame
    required_columns = ["country", "grid"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing_columns)}")

    # Create node column with optional suffix if available
    if "node_suffix" in df.columns:
        # Apply function to each row: combine country and grid, add suffix if it exists and is not empty
        df['node'] = df.apply(
            lambda row: f"{row['country']}_{row['grid']}" +
                        (f"_{row['node_suffix']}" if pd.notnull(row['node_suffix']) and row['node_suffix'] != "" else ""),
            axis=1
        )
    else:
        # Simple case: just combine country and grid
        df['node'] = df.apply(lambda row: f"{row['country']}_{row['grid']}", axis=1)

    return df


def build_unit_grid_and_node_columns(
    df_unitdata: pd.DataFrame,
    df_unittypedata: pd.DataFrame,
    log_messages: list[str] = None,
    *,
    country_col: str = "country",
    generator_id_col: str = "generator_id",
    blank_markers: Iterable[str] = ("", "-"),
    empty_value: Any = np.nan,
) -> pd.DataFrame:
    """
    Add node_<put> columns (e.g., node_output1, node_input1) without merging full tech tables.

    Rules:
      - grid_<put> is fetched by generator_id from df_unittypedata (first row per id).
      - node = f"{country}_{grid}" and ONLY if 'node_suffix_<put>' exists & is non-blank, append "_<suffix>".
      - No generic suffix fallback.
      - Does not drop or modify existing columns.

    Parameters
    ----------
    df_unitdata : pd.DataFrame
        Must include [country_col, generator_id_col].
        May include per-connection 'node_suffix_<put>'.
    df_unittypedata : pd.DataFrame
        Must include [generator_id_col] and any subset of 'grid_<put>' columns.
        If multiple rows per generator_id exist, the first is used.
    log_messages: list[str]
        A list of strings where new log events are added.
    country_col : str
        Column name in df_unitdata used for country.
    generator_id_col : str
        Column name in df_unitdata used for join key.
    blank_markers : iterable of str
        Suffix values treated as blank (ignored).

    Returns
    -------
    pd.DataFrame
        Copy of df_unitdata with added node_<put> columns.
    """
    out = df_unitdata.copy()

    # Determine which grid_input1...6, grid_output1...6 are used in the unittype data
    candidate_puts = [f"input{i}" for i in range(1, 6)] + [f"output{i}" for i in range(1, 6)]
    puts = [p for p in candidate_puts if f"grid_{p}" in df_unittypedata.columns]
    if not puts:
        log_status(
            (f"unittypedata table does not have a any column named grid_input1...6 or grid_output1...6 "
             "Check the files and names in the config file."),
            log_messages,
            level="warn",
        )
        return out

    # Build a compact lookup: first tech row per generator_id
    key = generator_id_col
    techs = (
        df_unittypedata
        .sort_values(by=[key])
        .drop_duplicates(subset=[key], keep="first")
        .set_index(key)
    )

    # Note: Not warning about missing generator_id values, because build_unittype_unit_column already does that

    # Pre-build Series mappers for each <put>
    grid_maps = {}
    for p in puts:
        grid_col = f"grid_{p}"
        grid_maps[p] = techs[grid_col] if grid_col in techs.columns else pd.Series(dtype="object")

    # Construct node_<put> per connection using map
    genID_series = out[generator_id_col]
    country_series = out[country_col].astype(object)

    for p in puts:
        grid_col = f"grid_{p}"
        # map -> may yield NaN if this put is not defined for the generator_id
        grids = genID_series.map(techs[grid_col]) if grid_col in techs.columns else pd.Series(np.nan, index=out.index)

        # mask: valid grid (not NaN, not in blank markers)
        valid = grids.notna()
        for bm in blank_markers:
            valid &= (grids != bm)

        # Start with empty_value for all stored grid names
        grid = pd.Series(empty_value, index=out.index, dtype=object)
        grid.loc[valid] = grids[valid].astype(str)
        out[f"grid_{p}"] = grid

        # Start with empty_value for stored node names
        node = pd.Series(empty_value, index=out.index, dtype=object)

        # Build base where valid
        base = country_series[valid].astype(str) + "_" + grids[valid].astype(str)

        # Append per-connection suffix only if present & non-blank
        suffix_col = f"node_suffix_{p}"
        if suffix_col in out.columns:
            suffix = out.loc[valid, suffix_col].astype(object)
            use_suffix = suffix.notna() & (suffix.astype(str).str.len() > 0) & ~suffix.isin(blank_markers)
            node.loc[valid & ~use_suffix] = base
            node.loc[valid & use_suffix] = base[use_suffix] + "_" + suffix[use_suffix].astype(str)
        else:
            node.loc[valid] = base

        out[f"node_{p}"] = node

    return out



def build_from_to_columns(
        df: pd.DataFrame
        ) -> pd.DataFrame:
    """
    Constructs 'from_node' and 'to_node' columns using 'from', 'to', 'grid', and optional suffixes.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'from', 'to', 'grid', and optionally 'from_suffix', 'to_suffix'.

    Returns:
    --------
    pandas.DataFrame
        DataFrame with new 'from_node', 'to_node' columns added.

    Raises:
    -------
    ValueError
        If required columns ('from', 'to', 'grid') are missing.
    """
    if df.empty: return df

    required_columns = ['from', 'to', 'grid']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing_columns)}")


    df = df.copy()

    # Fill suffix columns with empty strings if missing or NaN
    for suffix_col in ['from_suffix', 'to_suffix']:
        if suffix_col not in df.columns:
            df[suffix_col] = ''
        else:
            df[suffix_col] = df[suffix_col].fillna('').astype(str)

    # Build from_node and to_node with optional suffixes
    df['from_node'] = df['from'] + '_' + df['grid'] + df['from_suffix'].apply(lambda x: f'_{x}' if x else '')
    df['to_node']   = df['to']   + '_' + df['grid'] + df['to_suffix'].apply(lambda x: f'_{x}' if x else '')

    return df


def build_unittype_unit_column(
    df: pd.DataFrame,
    df_unittypedata: pd.DataFrame,
    source_data_logs: list[str] = None
    ) -> pd.DataFrame:
    """
    Add 'unittype' and 'unit' columns to DataFrame based on generator mappings.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing at minimum 'country' and 'generator_id' columns.
        Optional 'unit_name_prefix' column can be included for more specific unit naming.
        
    df_unittypedata : pandas.DataFrame
        Reference DataFrame mapping 'generator_id' to 'unittype' values

    source_data_logs: list[str]
        An input list of strings where new log events are added.
    
    Returns:
    --------
    pandas.DataFrame
        Input DataFrame with new 'unittype' and 'unit' columns added
    
    Raises:
    -------
    ValueError
        If required columns ('country', 'generator_id') are missing from the DataFrame
    
    Notes:
    ------
    - The 'unittype' is determined by case-insensitive lookup in df_unittypedata
    - The 'unit' format is: "{country}_{unittype}" or "{country}_{unit_name_prefix}_{unittype}"
      when unit_name_prefix is present and not empty
    - If 'generator_id' does not have any matching 'unittype', the code uses 'generator_id' instead of 'unittype'
    """
    from src.utils import log_status

    # Return input data if empty unittypedata
    if df.empty or df_unittypedata.empty:
        return df

    # Check that required columns exist in the DataFrame
    required_columns = ["country", "generator_id"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"DataFrame is missing required columns: {', '.join(missing_columns)}. Check all unitdata_files and remove_files.")

    # Create a mapping from generator_id (lowercase) to unittype for efficient lookup
    unit_mapping = df_unittypedata.set_index(df_unittypedata['generator_id'].str.lower())['unittype']

    # Add unittype column by mapping lowercase generator_id to the corresponding unittype
    df['unittype'] = df['generator_id'].str.lower().map(unit_mapping)

    # Identify generator_ids without match
    # Note: build_unit_grid_and_node_columns assumes that this is warned here
    missing_mask = df['unittype'].isna()
    if missing_mask.any() and log_status is not None:
        for generator_id in df.loc[missing_mask, 'generator_id'].unique():
            log_status(
                f"unitdata generator_ID '{generator_id}' does not have a matching generator_ID "
                "in any of the unittypedata files, check spelling.",
                source_data_logs,
                level="warn"
            )

    # Fallback: Fill in missing unittype with original generator_id
    df['unittype'] = df['unittype'].fillna(df['generator_id'])

    # Create unit column with optional prefix if available
    if "unit_name_prefix" in df.columns:
        # Apply function to each row: combine country and unittype, add prefix if it exists and is not empty
        df['unit'] = df.apply(
            lambda row: f"{row['country']}" +
                        (f"_{row['unit_name_prefix']}" if pd.notnull(row['unit_name_prefix']) and row['unit_name_prefix'] != "" else "") +
                        f"_{row['unittype']}",
            axis=1
        )
    else:
        # Simple case: just combine country and unittype
        df['unit'] = df.apply(
            lambda row: f"{row['country']}_{row['unittype']}",
            axis=1
        )

    return df


def normalize_dataframe(
    df: pd.DataFrame,
    df_identifier: str,
    logs: List[str],
    *,
    allowed_methods: Sequence[str] = ("replace", "replace-partial", "add", "add-non-negative", "multiply", "remove"),
    lowercase_value_cols: Sequence[str] = ("scenario", "generator_id", "method"),
    check_underscores: bool = True,
    drop_invalid_strings: bool = True,
    standardize_column_case: bool = True,
    treat_empty_as_na: bool = True,
) -> pd.DataFrame:
    """
    Normalize a DataFrame with consistent column naming, 'method' handling,
    optional underscore checks across string columns, and automatic dtype normalization.

    What it does
    ------------
    1) Column names: optionally lower-cases all column names.
    2) 'method' column: ensures existence; trims/lower-cases values; unknown methods
       are warned and coerced to 'replace' against `allowed_methods`.
    3) Value case: lower-cases selected identifier-like columns (e.g. 'scenario').
    4) Missing/empties: optionally treat empty/whitespace as NA, then globally fill NA with 0.
    5) Dtypes:
       - Columns that are fully numeric after coercion -> numeric dtype (downcast to int when possible).
    6) Column rename: for **numeric** columns named `*_output1`, drop the suffix to become the base
       name; skip and warn if renaming would collide with an existing column.
    7) Underscore check (quality gate): for **all** string-typed columns whose **column name
       does not start with "_"**, detect underscores in values; warn with examples and either
       drop those rows (default) or keep them based on `drop_invalid_strings`.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame to be normalized. `None` or empty are skipped.
    df_identifier : str
        Fallback identifier for logging. If a DataFrame contains any of
        `['_source_file','_source_sheet','file','sheet','source_file','source_sheet']`,
        a per-DataFrame identifier is derived from their uniform values.
    logs : list[str]
        Log sink passed to `log_status(...)`. Messages (warn/info) are appended there.
    allowed_methods : list[str]
        Allowed values for the 'method' column (case-insensitive). Unknown values default to 'replace'.
    lowercase_value_cols : Sequence[str], default ("scenario","generator_id","method")
        Columns whose **values** should be lower-cased. (The 'method' column is canonicalized separately.)
    check_underscores : bool, default True
        If True, scan string columns (excluding columns whose name starts with "_") for underscores in values.
    drop_invalid_strings : bool, default True
        Backward-compatible flag name. If True, **drop** rows that contained underscores in any checked column.
        If False, keep rows and only log.
    standardize_column_case : bool, default True
        If True, convert column **names** to lower-case.
    treat_empty_as_na : bool, default True
        If True, empty/whitespace-only strings are treated as NA before the global fillna(0).

    Returns
    -------
    pandas.DataFrame
        A normalized DataFrame (order preserved, empty inputs removed). Returns empty DataFrame if all inputs are empty.
    """

    allowed_set = {str(m).strip().lower() for m in allowed_methods}

    def _infer_identifier_from_df(df: pd.DataFrame, fallback: str) -> str:
        # Build an identifier from common source columns if available
        parts = []
        for col in ("_source_file", "_source_sheet", "file", "sheet", "source_file", "source_sheet"):
            if col in df.columns:
                vals = df[col].dropna().astype(str).unique()
                if len(vals) > 0:
                    parts.append(vals[0])
        return ":".join(parts) if parts else fallback

    def _canonicalize_method_series(s: pd.Series, *, ident: str) -> pd.Series:
        # Normalize 'method' values to lower-case, strip, default unknowns to 'replace'
        s2 = s.astype("string").replace(r"^\s*$", pd.NA, regex=True).str.strip().str.lower()
        s2 = s2.fillna("replace")
        unknown_vals = sorted(set(s2.unique()) - allowed_set - {"replace"})
        if unknown_vals:
            log_status(
                f"[{ident}] Unknown method(s) {unknown_vals} encountered; defaulting to 'replace'.",
                logs, level="warn"
            )
            s2 = s2.where(~s2.isin(unknown_vals), "replace")
        return s2

    # Skip None / empty DataFrames
    if df is None or df.empty:
        return pd.DataFrame()

    df_out = df.copy()

    # 1) Standardize column name case
    if standardize_column_case:
        df_out.columns = df_out.columns.str.lower()

    ident = _infer_identifier_from_df(df_out, df_identifier)

    # 2) Ensure and normalize 'method'
    if "method" not in df_out.columns:
        df_out["method"] = "replace"
    df_out["method"] = _canonicalize_method_series(df_out["method"], ident=ident)

    # 3) Lower-case selected value columns (excluding 'method' which is already canonicalized)
    for col in lowercase_value_cols:
        if col in df_out.columns and col != "method":
            df_out[col] = df_out[col].astype("string").str.lower()

    # 4) Treat empty strings as NA (optional)
    if treat_empty_as_na:
        df_out = df_out.replace(r"^\s*$", pd.NA, regex=True)

    # 5) Auto-type columns: numeric vs string
    numeric_cols: List[str] = []
    string_cols: List[str] = []
    for c in df_out.columns:
        converted = pd.to_numeric(df_out[c], errors="coerce")
        if converted.isna().sum() == 0:
            numeric_cols.append(c)
        else:
            string_cols.append(c)

    # 5a) Apply numeric dtype (downcast ints where possible)
    for c in numeric_cols:
        df_out[c] = pd.to_numeric(df_out[c], errors="coerce")
        if pd.api.types.is_float_dtype(df_out[c]):
            as_int = pd.to_numeric(df_out[c], downcast="integer")
            if pd.api.types.is_integer_dtype(as_int):
                df_out[c] = as_int
        if df_out[c].isna().any():
            df_out[c] = df_out[c].fillna(0)

    # 6) Drop '_output1' suffix from **numeric** column names (avoid collisions)
    num_cols_now = set(df_out.select_dtypes(include="number").columns)  # work on the typed df_out
    to_rename: Dict[str, str] = {}
    collisions: Dict[str, str] = {}
    for c in num_cols_now:
        if isinstance(c, str) and c.endswith("_output1"):
            new = c[:-8]
            if new in df_out.columns:  # would collide -> skip + warn
                collisions[c] = new
            else:
                to_rename[c] = new

    if collisions:
        pairs = ", ".join(f"{old} -> {new}" for old, new in collisions.items())
        log_status(
            f"[{ident}] Skipped renaming due to existing column name(s): {pairs}.",
            logs, level="warn"
        )

    if to_rename:
        df_out = df_out.rename(columns=to_rename)

    # 7) Underscore check in string columns (exclude columns starting with "_")
    if check_underscores:
        cols_to_check = [
            c for c in df_out.columns
            if not str(c).startswith("_") and pd.api.types.is_string_dtype(df_out[c])
        ]

        bad_mask = pd.Series(False, index=df_out.index)
        examples_per_col: Dict[str, List[str]] = {}

        for c in cols_to_check:
            m = df_out[c].astype("string").str.contains("_", na=False)
            if m.any():
                bad_mask |= m
                examples_per_col[c] = list(df_out.loc[m, c].dropna().astype(str).unique()[:5])

        if bad_mask.any():
            total_bad = int(bad_mask.sum())
            example_str = "; ".join(f"{col}: {vals}" for col, vals in list(examples_per_col.items())[:4])
            log_status(
                f"[{ident}] Underscores detected in {total_bad} row(s) across columns "
                f"[{', '.join(examples_per_col.keys())}]. Examples -> {example_str}",
                logs, level="warn"
            )
            if drop_invalid_strings:
                df_out = df_out.loc[~bad_mask].copy()
                log_status(
                    f"[{ident}] Dropped {total_bad} row(s) containing underscores per configuration.",
                    logs, level="info"
                )
            else:
                log_status(
                    f"[{ident}] Kept rows with underscores per configuration.",
                    logs, level="info"
                )

    return df_out


def apply_whitelist(
    df: pd.DataFrame,
    filters: Optional[Dict[str, Union[str, int, List[Union[str, int]]]]],
    logs: List[str],
    df_identifier: str,
) -> pd.DataFrame:
    """
    Apply AND-combined whitelist filters to a DataFrame, tolerantly.
    Missing filter columns never raise; they are logged and skipped.

    Matching semantics
    ------------------
    - Applies each (column -> allowed values) filter sequentially (logical AND).
    - Case-insensitive matching for string-typed columns.
    - Special always-include values (if the corresponding filter key is present):
        * scenario: include 'all' (case-insensitive) in addition to provided values
        * year    : include 1 in addition to provided values

    Specificity collapse (priority)
    --------------------------------
    After filtering, if 'scenario' and/or 'year' were among the filters and present
    in the DataFrame, the result is collapsed to the most specific rows:
      - Rows with scenario != 'all' are more specific than scenario == 'all'.
      - Rows with year != 1 are more specific than year == 1.
      - Scenario specificity outranks year specificity.
    Only rows with the highest specificity are retained (ties kept).

    Parameters
    ----------
    df : pd.DataFrame
    filters : dict[str, str|int|list[str|int]] or None
    logs : list[str]
    df_identifier : str

    Returns
    -------
    pd.DataFrame
        Filtered copy. If `filters` is falsy or df is empty, returns original copy.

    Logging
    -------
    - WARN  "[ident] Whitelist skipped: missing column 'col'."
    """
    # Fast exits for None/empty/no-filters
    if df is None:
        return pd.DataFrame()
    if df.empty or not filters:
        return df

    df_out = df.copy()

    # Track which special filters are actually applied and available
    scenario_filter_applied = False
    year_filter_applied = False

    # Apply each filter with AND semantics
    for col, val in filters.items():
        if col not in df_out.columns:
            log_status(f"[{df_identifier}] Whitelist skipped: missing column '{col}'.", logs, level="warn")
            continue

        vals = val if isinstance(val, list) else [val]

        if col == "scenario":
            scenario_filter_applied = True
            # Include 'all' (universal)
            allowed = {str(v).lower() for v in (vals + ["all"])}
            s = df_out[col].astype(str).str.lower()
            df_out = df_out[s.isin(allowed)]

        elif col == "year":
            year_filter_applied = True
            # Include 1 (universal)
            allowed_set = set(vals + [1])
            df_out = df_out[df_out[col].isin(allowed_set)]

        else:
            # Case-insensitive for strings; exact for non-strings
            if pd.api.types.is_string_dtype(df_out[col]):
                allowed = {str(v).lower() for v in vals}
                s = df_out[col].astype(str).str.lower()
                df_out = df_out[s.isin(allowed)]
            else:
                df_out = df_out[df_out[col].isin(vals)]

        if df_out.empty:
            return df_out  # Short-circuit: nothing left

    # If nothing left or no special priority applicable, return
    if df_out.empty:
        return df_out

    # Scenario specificity: True if not 'all'
    if scenario_filter_applied and ("scenario" in df_out.columns):
        scen_is_specific = df_out["scenario"].astype(str).str.lower() != "all"
    else:
        scen_is_specific = pd.Series(False, index=df_out.index)

    # Year specificity: True if not 1 (handles string/int robustly)
    if year_filter_applied and ("year" in df_out.columns):
        col_y = df_out["year"]
        if pd.api.types.is_numeric_dtype(col_y):
            year_is_specific = col_y != 1
        else:
            year_is_specific = col_y.astype(str).str.strip() != "1"
    else:
        year_is_specific = pd.Series(False, index=df_out.index)

    # Weight: scenario more important than year
    score = (scen_is_specific.astype(int) * 2) + (year_is_specific.astype(int) * 1)

    if score.max() > 0 or (scenario_filter_applied or year_filter_applied):
        max_score = int(score.max())
        df_out = df_out[score == max_score]

    return df_out


def apply_blacklist(
    df_input: pd.DataFrame,
    df_name: str,
    filters: Dict[str, Union[str, int, List[Union[str, int]]  ]  ],
    source_data_logs: list[str] = None,
    *,
    log_warning: bool = True
    ) -> pd.DataFrame:
    """
    Filter DataFrame by excluding rows containing blacklisted values.
    
    Parameters:
    -----------
    df_input : pandas.DataFrame
        The DataFrame to be filtered
    df_name : str
        Name of the DataFrame (used for error reporting)
    filters : dict
        Dictionary of {column_name: blacklisted_values} pairs
    source_data_logs: list[str]
        A list of strings where new log events are added.
    
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame with rows containing blacklisted values removed
    
    Notes:
    ------
    - String comparisons are case-insensitive
    """
    # Skip processing if df_input is empty
    if df_input.empty:
        return df_input

    # Create a copy to avoid modifying the original DataFrame
    df_filtered = df_input.copy()

    # Apply each blacklist filter condition 
    for col, val in filters.items():
        if col not in df_filtered.columns:
            if log_warning:
                log_status(f"Missing column in {df_name}: {col!r}", source_data_logs, level="warn")
            continue  

        # Ensure val is a list for consistent processing
        if not isinstance(val, list):
            val = [val]

        # Handle string columns with case-insensitive comparison
        if pd.api.types.is_string_dtype(df_filtered[col]):
            # Convert filter values to lowercase if they're strings
            lowered_vals = [v.lower() if isinstance(v, str) else v for v in val]
            # Note the '~' operator to invert the filter (exclude matches)
            df_filtered = df_filtered[~df_filtered[col].str.lower().isin(lowered_vals)]
        else:
            # Direct comparison for non-string columns
            df_filtered = df_filtered[~df_filtered[col].isin(val)]

    return df_filtered


def apply_unit_grids_blacklist(
    df: pd.DataFrame,
    exclude_grids: List[str],
    df_name: str = "unitdata",
    logs: list[str] = None
) -> pd.DataFrame:
    filters = {**{f"grid_input{i}":  exclude_grids for i in range(1, 6)},
               **{f"grid_output{i}": exclude_grids for i in range(1, 6)}}
    return apply_blacklist(df, df_name, filters, source_data_logs=logs, log_warning=False)


def apply_unit_nodes_blacklist(
    df: pd.DataFrame,
    exclude_nodes: List[str],
    df_name: str = "unitdata",
    logs: list[str] = None
) -> pd.DataFrame:
    filters = {**{f"node_input{i}":  exclude_nodes for i in range(1, 6)},
               **{f"node_output{i}": exclude_nodes for i in range(1, 6)}}
    return apply_blacklist(df, df_name, filters, source_data_logs=logs, log_warning=False)


def merge_row_by_row(
    dfs: Iterable[pd.DataFrame],
    logs: List[str],
    *,
    key_columns: Optional[Sequence[str]] = None,  # e.g. ['generator_id']
    allowed_methods: Sequence[str] = ("replace", "replace-partial", "add", "add-non-negative", "multiply", "remove"),
    measure_cols: Sequence[str] = (),
    not_measure_cols: Sequence[str] = ("year",),
) -> pd.DataFrame:
    """
    Merge DataFrames row-by-row in order, applying a per-row 'method'.

    Methods
    -------
    - 'replace'          : Overwrite the entire row (last row wins), empties/zeros included.

    - 'replace-partial'  : Overwrite only columns provided in the second DataFrame
                           where the provided value is not empty.
                           Zero replaces numeric values.

    - 'add'              : Sum into measures with special missing rules:
                           * (missing + missing) → NaN
                           * single missing value is treated as 0.0 
                             e.g. (missing + 0.0) or (0.0 + missing) → 0.0
                           Only affects measure columns present in the second DataFrame
                           (absent → treated as missing → apply the above rules).
                           On first occurrence, use incoming values directly (initialize).

    - 'add-non-negative' : Same as 'add', but clamp results to ≥ 0. On first occurrence,
                           initialize measures with max(0, incoming).                       

    - 'multiply'         : Multiply measures with special missing rules:
                           * (missing x missing) → NaN
                           * previous missing → 0.0 (zeroes product)
                           * current (new) missing → 1.0 (no change)
                           Only affects measure columns present in the second DataFrame
                           (absent → treated as missing → apply the above rules).
                           On first occurrence, use incoming values directly (initialize).

    - 'remove'   : Delete any previously merged row for the same key.

    Column typing & safety
    ----------------------
    - If `measure_cols` is blank, we **infer measures conservatively**:
        * Exclude booleans and `not_measure_cols`
        * A column qualifies only if **all non-missing values across all frames are numeric or numeric-like**
          AND there is at least one non-missing numeric value overall.
        * Any evidence of non-numeric text ⇒ not a measure.
    - Only measure columns are coerced to numeric at the end (nullable Float64).
      Non-measure columns keep their original dtypes (no global casting).
    """

    # --- Logging helper --------------------------------------------------------
    # Lightweight wrapper that uses external log_status if available, otherwise
    # appends to the provided logs list. Keeps this function self-contained.
    def _log(msg: str, level: str = "info"):
        try:
            log_status(msg, logs, level=level)  # type: ignore[name-defined]
        except Exception:
            logs.append(f"[{level.upper()}] {msg}")

    # --- Input filtering & column union ---------------------------------------
    # Drop None/empty frames early; return empty DF if nothing usable.
    frames = [df for df in dfs if df is not None and not getattr(df, "empty", True)]
    if not frames:
        _log("merge_row_by_row: No data provided. Returning empty DataFrame.", level="warn")
        return pd.DataFrame()

    # Build a union of columns preserving first-seen order across frames.
    cols_union: List[str] = []
    for df in frames:
        for c in df.columns:
            if c not in cols_union:
                cols_union.append(c)

    # Meta columns that are dropped at the end (kept through merge to avoid loss).
    meta_cols = {"_source_file", "_source_sheet"}

    # --- Missing / numeric helpers --------------------------------------------
    # Centralized notions of "missing" and "numeric-like" so the rules are consistent.

    def _is_missing(x) -> bool:
        # True for None, NaN-like, empty/whitespace strings, and common text tokens.
        if x is None:
            return True
        try:
            if pd.isna(x):
                return True
        except Exception:
            pass
        if isinstance(x, str):
            xs = x.strip()
            if xs == "" or xs.lower() in {"nan", "none", "null", "n/a"}:
                return True
        try:
            if isinstance(x, float) and math.isnan(x):
                return True
        except Exception:
            pass
        return False

    def _is_numeric_like(x) -> bool:
        """True if value can be safely interpreted as a number (ignoring missing tokens)."""
        if _is_missing(x):
            return True  # missing does not disqualify numeric-ness for inference
        if isinstance(x, (int, float)) and not (isinstance(x, bool)):
            return True
        if isinstance(x, str):
            s = x.strip()
            try:
                float(s)
                return True
            except Exception:
                return False
        return False

    def _to_float_or_nan(x):
        # Convert value to float; missing or bad parses become NaN.
        if _is_missing(x):
            return math.nan
        try:
            if isinstance(x, str):
                return float(x.strip())
            return float(x)
        except Exception:
            return math.nan

    def _to_non_neg_float_or_nan(x):
        # Convert value to non-negative float; missing or bad parses become NaN.
        if _is_missing(x):
            return math.nan
        try:
            if isinstance(x, str):
                return float(x.strip())
            return max(0, float(x))
        except Exception:
            return math.nan


    # --- Measure column inference ---------------------------------------------
    # If user didn't specify measure columns, derive them defensively.
    def _is_effectively_empty(seq: Optional[Sequence[str]]) -> bool:
        if seq is None or len(seq) == 0:
            return True
        return all((s is None) or (str(s).strip() == "") for s in seq)

    def _infer_measure_columns(frames: List[pd.DataFrame]) -> List[str]:
        not_meas = set(not_measure_cols)
        candidates: List[str] = []
        for c in cols_union:
            if c in not_meas:
                continue

            # Skip booleans if any frame treats column as boolean.
            if any(c in df.columns and pd.api.types.is_bool_dtype(df[c]) for df in frames):
                continue

            any_numeric_value = False
            disqualify = False

            for df in frames:
                if c not in df.columns:
                    continue
                col = df[c]

                # Numeric dtype with any non-NA value is acceptable.
                if pd.api.types.is_numeric_dtype(col):
                    if col.notna().any():
                        any_numeric_value = True
                    continue

                # Object/string columns: ensure every non-missing is numeric-like.
                non_missing_vals = col[col.map(lambda v: not _is_missing(v))]
                if non_missing_vals.empty:
                    continue
                for v in non_missing_vals.iloc[:10000]:  # safety bound on inspection
                    if not _is_numeric_like(v):
                        disqualify = True
                        break
                if disqualify:
                    break
                any_numeric_value = True

            if (not disqualify) and any_numeric_value:
                candidates.append(c)
        return candidates

    # Decide actual measure set: user-provided or inferred.
    if _is_effectively_empty(measure_cols):
        present_measures = _infer_measure_columns(frames)
    else:
        present_measures = [c for c in (measure_cols or []) if c in set(cols_union)]
        missing = [c for c in (measure_cols or []) if c not in set(cols_union)]
        if missing:
            _log(
                f"[merge_row_by_row] Some measure_cols not present in inputs and will be ignored: {missing}",
                level="warn",
            )

    # --- Key derivation & normalization ---------------------------------------
    # If no explicit keys, derive keys as "everything that is not a measure/meta/method".
    if key_columns is None:
        key_columns = [c for c in cols_union if c not in set(present_measures) | meta_cols | {"method"}]

    # Without any key, we fall back to "last occurrence wins" over full rows.
    if not key_columns:
        _log("merge_row_by_row: No key columns available; using full-row 'last occurrence wins'.", level="warn")
        merged = pd.concat(frames, ignore_index=True, sort=False).drop_duplicates(subset=None, keep="last")
        merged = merged.drop(columns=list(meta_cols), errors="ignore")
        return merged

    # Ensure all frames have all key columns (filled with <NA>) so tuple keys are well-defined.
    if key_columns:
        new_frames = []
        for df in frames:
            if any(k not in df.columns for k in key_columns):
                df = df.copy()
                for k in key_columns:
                    if k not in df.columns:
                        df[k] = pd.NA
            new_frames.append(df)
        frames = new_frames

    def _norm_key_val(x):
        # Canonicalize key parts: trim strings, None for missing.
        if _is_missing(x):
            return None
        if isinstance(x, str):
            return x.strip()
        return x

    def _key_tuple(row_dict: Dict[str, object]) -> Tuple:
        # Deterministic key across all rows/frames.
        return tuple(_norm_key_val(row_dict.get(k)) for k in key_columns)

    # --- Core merge loop -------------------------------------------------------
    # We process frames in order. For each row, apply its 'method' against an accumulator.
    allowed_set = {m.lower().strip() for m in allowed_methods}
    acc: Dict[Tuple, Dict[str, object]] = {}
    unknown_seen: set = set()

    for df in frames:
        # Default method is 'replace' if not provided.
        if "method" not in df.columns:
            df = df.copy()
            df["method"] = "replace"

        for row_dict in df.to_dict(orient="records"):
            method = str(row_dict.get("method", "replace")).strip().lower()
            # Apply 'replace' for unknown methods 
            if method not in allowed_set:
                unknown_seen.add(method)
                method = "replace"

            k = _key_tuple(row_dict)
            existing = acc.get(k)

            # --- 'remove': delete any existing record for this key --------------
            if method == "remove":
                if existing is not None:
                    del acc[k]
                continue

            # --- First occurrence of key: initialize new record -----------------
            if existing is None:
                # Start from the incoming row; we'll normalize measures if needed.
                new_rec = {c: row_dict.get(c, None) for c in cols_union}

                if present_measures and method in {"replace-partial", 
                                                   "add", 
                                                   "add-non-negative", 
                                                   "multiply"}:
                    # On first occurrence, all but 'remove' as initialization:
                    # store numeric values as-is, keep missings as NaN.
                    for mc in present_measures:
                        if method == "add-non-negative":
                            new_rec[mc] = _to_non_neg_float_or_nan(new_rec.get(mc))
                        else:
                            new_rec[mc] = _to_float_or_nan(new_rec.get(mc))
                
                # 'replace' (or no measures) already behaves as initialization via new_rec
                new_rec["method"] = method
                acc[k] = new_rec
                continue

            # --- Subsequent occurrences of key: merge with existing -------------
            if method == "replace" or not present_measures:
                # Full overwrite (non-measures included).
                new_rec = {c: row_dict.get(c, None) for c in cols_union}
                new_rec["method"] = method
                acc[k] = new_rec

            elif method == "replace-partial":
                # Overwrite only with provided and non-empty, non-zero values.
                for c in cols_union:
                    if c in key_columns or c in meta_cols or c in {"method"}:
                        continue
                    if c not in row_dict:
                        continue
                    val = row_dict.get(c, None)
                    if _is_missing(val) or not _is_numeric_like(val):
                        continue
                    existing[c] = val
                existing["method"] = method

            elif method == "add":
                # Elementwise addition 
                for mc in present_measures:
                    prev = _to_float_or_nan(existing.get(mc))
                    cur  = _to_float_or_nan(row_dict.get(mc))
            
                    if math.isnan(prev) and math.isnan(cur):
                        existing[mc] = math.nan
                    else:
                        # Treat any single missing as 0.0
                        existing[mc] = (0.0 if math.isnan(prev) else prev) + \
                                       (0.0 if math.isnan(cur)  else cur)
                existing["method"] = method

            elif method == "add-non-negative":
                # Elementwise addition 
                for mc in present_measures:
                    prev = _to_float_or_nan(existing.get(mc))
                    cur  = _to_float_or_nan(row_dict.get(mc))
            
                    if math.isnan(prev) and math.isnan(cur):
                        existing[mc] = math.nan
                    else:
                        # Treat any single missing as 0.0
                        existing[mc] = max(0, (0.0 if math.isnan(prev) else prev) + \
                                              (0.0 if math.isnan(cur)  else cur) \
                                            )    
                existing["method"] = method                

            elif method == "multiply":
                # Elementwise multiplication
                for mc in present_measures:
                    prev = _to_float_or_nan(existing.get(mc))
                    cur  = _to_float_or_nan(row_dict.get(mc))

                    if math.isnan(prev) and math.isnan(cur):
                        existing[mc] = math.nan
                    else:
                        prev_eff = 0.0 if math.isnan(prev) else prev
                        cur_eff  = 1.0 if math.isnan(cur)  else cur
                        existing[mc] = prev_eff * cur_eff
                existing["method"] = method

            else:
                # Shouldn't happen (filtered earlier), but keep a safe fallback.
                new_rec = {c: row_dict.get(c, None) for c in cols_union}
                new_rec["method"] = "replace"
                acc[k] = new_rec

    # --- Assemble output frame -------------------------------------------------
    merged = pd.DataFrame.from_records(list(acc.values()), columns=cols_union)

    # Drop meta/helper columns that shouldn't survive the merge result.
    merged = merged.drop(columns=list(meta_cols), errors="ignore")
    merged = merged.drop(columns=["year", "scenario", "method"], errors="ignore")

    # --- Final dtype coercion for measure columns ------------------------------
    # Only cast measures → avoids surprising dtype changes elsewhere.
    for mc in present_measures:
        if mc in merged.columns:
            merged[mc] = pd.to_numeric(merged[mc], errors="coerce").astype("Float64")

    # --- Epilogue: warnings about unknown methods ------------------------------
    if unknown_seen:
        _log(
            f"[merge_row_by_row] Unknown method(s) {sorted(unknown_seen)} encountered; defaulting to 'replace'.",
            level="warn",
        )

    return merged


def filter_nonzero_numeric_rows(
        df: pd.DataFrame, exclude: list[str] = None
        ) -> pd.DataFrame:
    """
    Removes rows from the DataFrame where the sum of numeric columns is zero.
    Optionally excludes specific numeric columns from the summation.

    Parameters:
        df (pd.DataFrame): Input DataFrame.
        exclude (list[str], optional): List of column names to exclude from summing.

    Returns:
        pd.DataFrame: Filtered DataFrame with only rows having non-zero numeric data.
    """
    if exclude is None:
        exclude = []

    numeric_cols = df.select_dtypes(include='number').columns.difference(exclude)
    return df[df[numeric_cols].sum(axis=1) != 0]