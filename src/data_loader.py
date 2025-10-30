import pandas as pd
import numpy as np
from typing import Union, List, Dict, Optional, Tuple, Iterable, Sequence, Any, Set
import src.utils as utils



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
        utils.log_status(
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
    if missing_mask.any() and utils.log_status is not None:
        for generator_id in df.loc[missing_mask, 'generator_id'].unique():
            utils.log_status(
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
    lowercase_col_values: Sequence[str] = ("scenario", "generator_id", "method"),
    lowercase_column_names: bool = True,
    apply_numeric_dtype: bool = True,
    treat_empty_as_na: bool = True,
    check_underscore_values: bool = True,
    drop_invalid_strings: bool = True,
) -> pd.DataFrame:
    """
    Normalize a DataFrame with consistent column naming, 'method' handling,
    optional underscore checks across string columns, and automatic dtype normalization.

    What it does
    ------------
    1) Column names (lowercase_column_names): optionally lower-cases all column names.
    2) 'method' column: ensures existence; trims/lower-cases values; unknown methods
       are warned and coerced to 'replace' against `allowed_methods`.
    3) Lower-case column values (lowercase_col_values): lower-cases selected identifier-like columns (e.g. 'scenario').
    4) Missing/empties (treat_empty_as_na): optionally treat empty/whitespace as NA 
    5) DType conversions :
       - Convert empty/NaN columns to Object dtype
       - Convert float64 to Float64
       - For columns that are fully numeric after coercion 
          -> numeric dtype
          -> then fill NA with 0.
    6) Column rename: for **numeric** columns named `*_output1`, drop the suffix to become the base
       name; skip and warn if renaming would collide with an existing column.
    7) Underscore check (check_underscores): for **all** string-typed columns whose **column name
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
        Log sink passed to `utils.log_status(...)`. Messages (warn/info) are appended there.
    allowed_methods : list[str]
        Allowed values for the 'method' column (case-insensitive). Unknown values default to 'replace'.
    lowercase_col_values : Sequence[str], default ("scenario","generator_id","method")
        Columns whose **values** should be lower-cased. (The 'method' column is canonicalized separately.)
    lowercase_column_names : bool, default True
        If True, convert column **names** to lower-case.
    apply_numeric_dtype : bool, default True
        If True, applies numeric datatype to fully numeric columns
    treat_empty_as_na : bool, default True
        If True, empty/whitespace-only strings are treated as NA before the global fillna(0).
    check_underscore_values : bool, default True
        If True, scan string columns (excluding columns whose name starts with "_") for underscores in values.
    drop_invalid_strings : bool, default True
        Backward-compatible flag name. If True, **drop** rows that contained underscores in any checked column.
        If False, keep rows and only log.

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
            utils.log_status(
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
    if lowercase_column_names:
        df_out.columns = df_out.columns.str.lower()

    ident = _infer_identifier_from_df(df_out, df_identifier)

    # 2) Ensure and normalize 'method'
    if "method" not in df_out.columns:
        df_out["method"] = "replace"
    df_out["method"] = _canonicalize_method_series(df_out["method"], ident=ident)

    # 3) Lower-case selected columns' values
    # Note: cannot be applied to 'method' which is already handled above
    for col in lowercase_col_values:
        if col in df_out.columns and col != "method":
            df_out[col] = df_out[col].astype("string").str.lower()

    # 4) Treat empty strings as NA (optional)
    if treat_empty_as_na:
        df_out = df_out.replace(r"^\s*$", pd.NA, regex=True)

    # 5) dtype conversions
    # Convert empty columns (all NA/NaN/null values) to object dtype
    for col in df_out.columns:
        if df_out[col].isna().all():  # pd.isna() catches both NaN and pd.NA
            df_out[col] = df_out[col].astype('object')

    # Convert float64 columns to Float64
    for col in df_out.columns:
        if pd.api.types.is_float_dtype(df_out[col]) and df_out[col].dtype == 'float64':
            df_out[col] = df_out[col].astype('Float64')            

    # Classify numeric and object columns
    numeric_cols: List[str] = []
    object_cols: List[str] = []
    for c in df_out.columns:
        converted = pd.to_numeric(df_out[c], errors="coerce")
        # Count sum of coerced NA
        if converted.isna().sum() == 0:
            numeric_cols.append(c)
        else:
            object_cols.append(c)

    # Apply Float64 dtype when possible
    if apply_numeric_dtype:
        for c in numeric_cols:
            df_out[c] = pd.to_numeric(df_out[c], errors="coerce").astype("Float64")
            # Current code cannot reach this safeguard, but this is kept here if 
            # if converted.isna().sum() == 0: is later relax e.g. to if converted.isna().mean() < 0.1:
            if df_out[c].isna().any():
                df_out[c] = df_out[c].fillna(0)

    # 6) Drop '_output1' suffix from **numeric** column names (avoid collisions)
    to_rename: Dict[str, str] = {}
    collisions: Dict[str, str] = {}
    for c in numeric_cols:
        if isinstance(c, str) and c.endswith("_output1"):
            new = c[:-8]
            if new in df_out.columns:  # would collide -> skip + warn
                collisions[c] = new
            else:
                to_rename[c] = new

    if collisions:
        pairs = ", ".join(f"{old} -> {new}" for old, new in collisions.items())
        utils.log_status(
            f"[{ident}] Skipped renaming due to existing column name(s): {pairs}.",
            logs, level="warn"
        )

    if to_rename:
        df_out = df_out.rename(columns=to_rename)

    # 7) Underscore check in object_cols (exclude columns starting with "_")
    if check_underscore_values:
        cols_to_check = [c for c in object_cols if not str(c).startswith("_")]

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
            utils.log_status(
                f"[{ident}] Underscores detected in {total_bad} row(s) across columns "
                f"[{', '.join(examples_per_col.keys())}]. Examples -> {example_str}",
                logs, level="warn"
            )
            if drop_invalid_strings:
                df_out = df_out.loc[~bad_mask].copy()
                utils.log_status(
                    f"[{ident}] Dropped {total_bad} row(s) containing underscores per configuration.",
                    logs, level="info"
                )
            else:
                utils.log_status(
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
            utils.log_status(f"[{df_identifier}] Whitelist skipped: missing column '{col}'.", logs, level="warn")
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
                utils.log_status(f"Missing column in {df_name}: {col!r}", source_data_logs, level="warn")
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
    key_columns: Sequence[str],
    measure_cols: Sequence[str] = (),
    not_measure_cols: Sequence[str] = ("year",),
    meta_cols: Set[str] = {"_source_file", "_source_sheet", "method"},
) -> pd.DataFrame:
    """
    Merge DataFrames row-by-row in order, applying a per-row 'method'.

    **PRECONDITION**: All input DataFrames must be normalized via normalize_dataframe() first.
    This function assumes:
    - Column names are lowercase
    - 'method' column exists with valid, lowercase values
    - Empty strings converted to NA
    - Numeric columns properly typed
    - No data quality issues remain

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
        * A column qualifies if it's numeric-typed with at least one non-NA value
    - Only measure columns are coerced to nullable Float64 at the end.
    - Other columns specifically keep their dtype
    """

    # --- Input filtering & column union ---------------------------------------
    frames = [df for df in dfs if df is not None and not getattr(df, "empty", True)]
    if not frames:
        utils.log_status("merge_row_by_row: No data provided. Returning empty DataFrame.", logs, level="warn")
        return pd.DataFrame()

    # Build a union of columns preserving first-seen order across frames.
    cols_union: List[str] = []
    for df in frames:
        for c in df.columns:
            if c not in cols_union:
                cols_union.append(c)

    # --- Measure column inference -----------------------------------------
    def _infer_measure_columns(frames: List[pd.DataFrame]) -> List[str]:
        """
        Infer measure columns from normalized frames.
        Since frames are already normalized:
        - Column names are lowercase
        - Numeric columns are already typed correctly
        - Empty values are already NA
        """
        not_meas = set(not_measure_cols)
        candidates = []
        
        for c in cols_union:
            if c in not_meas:
                continue
            
            # Skip booleans
            if any(c in df.columns and pd.api.types.is_bool_dtype(df[c]) for df in frames):
                continue
            
            # Check if any frame has it as numeric with non-NA values
            for df in frames:
                if c in df.columns:
                    if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
                        candidates.append(c)
                        break
        
        return candidates

    # Decide actual measure set: user-provided or inferred
    if not measure_cols:
        present_measures = _infer_measure_columns(frames)
    else:
        # Keep only measures that exist in the data
        present_measures = [c for c in measure_cols if c in cols_union]
        missing = [c for c in measure_cols if c not in cols_union]
        if missing:
            utils.log_status(f"[merge_row_by_row] Some measure_cols not found: {missing}", logs, level="warn")

    # --- Key validation -------------------------------------------------------
    key_columns = list(key_columns)
    missing = [k for k in key_columns if k not in cols_union]
    if missing:
        utils.log_status(f"[merge_row_by_row] Some key_columns not found and will be created as <NA>: {missing}", 
                   logs, level="warn")

    # Ensure all frames have all key columns (filled with <NA>)
    new_frames = []
    for df in frames:
        missing_keys = [k for k in key_columns if k not in df.columns]
        if missing_keys:
            df = df.copy()
            for k in missing_keys:
                df[k] = pd.NA
        new_frames.append(df)
    frames = new_frames

    # --- Method handlers ------------------------------------------------------
    def _handle_replace(existing: Dict[str, object], row_dict: Dict[str, object], method: str, partial: bool = False) -> Dict[str, object]:
        """Full or partial row replacement."""
        if existing is None:
            # Initialize new record
            new_rec = {c: row_dict.get(c) for c in cols_union}
            new_rec["method"] = method
            return new_rec
        
        if partial:
            # Overwrite only provided non-missing values
            for c in cols_union:
                if c in key_columns or c in meta_cols:
                    continue
                if c not in row_dict:
                    continue
                val = row_dict.get(c)
                if val is not None and not pd.isna(val):
                    existing[c] = val
            existing["method"] = method
            return existing
        else:
            # Full replacement
            new_rec = {c: row_dict.get(c) for c in cols_union}
            new_rec["method"] = method
            return new_rec

    def _handle_add(existing: Dict[str, object], row_dict: Dict[str, object], method: str, clamp_non_negative: bool = False) -> Dict[str, object]:
        """Elementwise addition with special missing rules."""
        if existing is None:
            # Initialize new record
            new_rec = {c: row_dict.get(c) for c in cols_union}
            # Clamp negative values if needed
            if clamp_non_negative and present_measures:
                for mc in present_measures:
                    val = new_rec.get(mc)
                    if val is not None and not pd.isna(val) and val < 0:
                        new_rec[mc] = 0.0
            new_rec["method"] = method
            return new_rec
        
        # Subsequent occurrence: add to existing
        for mc in present_measures:
            prev_val = existing.get(mc)
            cur_val = row_dict.get(mc)
            
            prev_missing = prev_val is None or pd.isna(prev_val)
            cur_missing = cur_val is None or pd.isna(cur_val)
    
            if prev_missing and cur_missing:
                existing[mc] = pd.NA
            else:
                # Treat single missing as 0.0
                result = (0.0 if prev_missing else prev_val) + (0.0 if cur_missing else cur_val)
                existing[mc] = max(0.0, result) if clamp_non_negative else result
        existing["method"] = method
        return existing

    def _handle_multiply(existing: Dict[str, object], row_dict: Dict[str, object], method: str) -> Dict[str, object]:
        """Elementwise multiplication with special missing rules."""
        if existing is None:
            # Initialize new record - no normalization needed
            new_rec = {c: row_dict.get(c) for c in cols_union}
            new_rec["method"] = method
            return new_rec
        
        # Subsequent occurrence: multiply with existing
        for mc in present_measures:
            prev_val = existing.get(mc)
            cur_val = row_dict.get(mc)
            
            prev_missing = prev_val is None or pd.isna(prev_val)
            cur_missing = cur_val is None or pd.isna(cur_val)

            if prev_missing and cur_missing:
                existing[mc] = pd.NA
            else:
                # Previous missing → 0.0, current missing → 1.0
                prev_eff = 0.0 if prev_missing else prev_val
                cur_eff = 1.0 if cur_missing else cur_val
                existing[mc] = prev_eff * cur_eff
        existing["method"] = method
        return existing

    # --- Core merge loop -------------------------------------------------------
    # Process frames in order. For each row, apply its 'method' against an accumulator.
    acc: Dict[Tuple, Dict[str, object]] = {}

    for df in frames:
        # Default method is 'replace' if not provided (should not happen after normalization)
        if "method" not in df.columns:
            df = df.copy()
            df["method"] = "replace"

        for row_dict in df.to_dict(orient="records"):
            # Method is already validated and lowercased by normalize_dataframe
            method = str(row_dict.get("method", "replace"))
            # Build key tuple inline
            k = tuple(None if val is None or pd.isna(val) else val 
                     for val in (row_dict.get(kc) for kc in key_columns))
            existing = acc.get(k)

            # --- 'remove': delete any existing record for this key --------------
            if method == "remove":
                if existing is not None:
                    del acc[k]
                continue

            # --- Apply appropriate handler based on method ----------------------
            if method == "replace" or not present_measures:
                acc[k] = _handle_replace(existing, row_dict, method, partial=False)
            elif method == "replace-partial":
                acc[k] = _handle_replace(existing, row_dict, method, partial=True)
            elif method == "add":
                acc[k] = _handle_add(existing, row_dict, method, clamp_non_negative=False)
            elif method == "add-non-negative":
                acc[k] = _handle_add(existing, row_dict, method, clamp_non_negative=True)
            elif method == "multiply":
                acc[k] = _handle_multiply(existing, row_dict, method)

    # --- Assemble output frame -------------------------------------------------
    merged = pd.DataFrame.from_records(list(acc.values()), columns=cols_union)

    # Drop meta/helper columns that shouldn't survive
    merged = merged.drop(columns=list(meta_cols), errors="ignore")

    # --- Standardize dtypes: Float64 for numerics, object for rest ------------
    merged = utils.standardize_df_dtypes(merged)

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