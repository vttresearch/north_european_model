# src/utils.py

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_bool_dtype


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
            df[col] = df[col].fillna(pd.NA)

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


def collect_domains_for_cache(df, possible_domains: list[str]) -> dict[str, list]:
    """
    Collect domain values from a processor result for JSON caching and cross-processor accumulation.

    Produces a dict that serializes directly to JSON and can be merged across processors. 
    
    Final compilation and normalization of domain names happens downstream when the Excel output 
    is assembled.

    Parameters:
    - df: pandas.DataFrame containing possible domain columns
    - possible_domains: list of domain column names to check in df

    Returns:
    - dict[str, list]: dictionary of domain -> unique values (unsorted)
    """
    result = {}

    for domain in possible_domains:
        if domain in df.columns:
            unique_values = df[domain].dropna().unique()
            if len(unique_values) > 0:
                result[domain] = list(unique_values)

    return result


def collect_domain_pairs_for_cache(df, domain_pairs: list[list[str]]) -> dict[str, list[tuple]]:
    """
    Collect domain value pairs from a processor result for JSON caching and cross-processor accumulation.

    Produces a dict that serializes directly to JSON and can be merged across processors. 
    
    Possible domain pairs is additional information needed in addition to domains, to avoid
    generating input excel data for non-existent domain pairs.

    Parameters:
    - df: pandas.DataFrame containing the domain columns
    - domain_pairs: list of domain pair lists, e.g. [['grid', 'node'], ['flow', 'node']]

    Returns:
    - dict[str, list[tuple]]: mapping from pair key like 'grid_node' to unique domain tuples
    """
    result = {}

    for pair in domain_pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Each domain pair must be a list of exactly two domain names")

        domain1, domain2 = pair

        # Skip pair if any column is missing
        if domain1 not in df.columns or domain2 not in df.columns:
            continue

        # Extract and deduplicate
        pairs_df = df[[domain1, domain2]].drop_duplicates()
        pair_key = f"{domain1}_{domain2}"
        new_pairs = list(pairs_df.itertuples(index=False, name=None))

        if new_pairs:
            result[pair_key] = new_pairs

    return result


def is_val_empty(val) -> bool:
    """
    Return True if `val` is None or NaN-like (pd.NA, np.nan, etc.).

    Upstream normalization (normalize_dataframe + standardize_df_dtypes) guarantees
    that values reaching this function are only None, pd.NA/np.nan, float, or str
    — with whitespace already stripped and empty strings converted to pd.NA.
    """
    if val is None:
        return True
    try:
        return pd.isna(val)
    except (ValueError, TypeError):
        return False


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
    if is_bool_dtype(s):
        return s.isna().all()

    # Numeric (excluding bool): empty if all zeros or NaN
    if is_numeric_dtype(s) and not is_bool_dtype(s):
        return (s.fillna(0) == 0).all()

    # Non-numeric: empty if all are NaN or whitespace-only strings
    na_mask = s.isna()
    # Safe elementwise test; no vectorized == on arbitrary objects
    empty_str_mask = s.map(lambda v: isinstance(v, str) and v.strip() == "")
    return (na_mask | empty_str_mask).all()