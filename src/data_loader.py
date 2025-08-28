import pandas as pd
import sys
from pathlib import Path
from typing import Union, List, Dict, Optional, Tuple


def process_dataset(
    input_folder: Union[Path, str],
    files: List[str],
    prefix: str,
    isMandatory: bool = True
) -> pd.DataFrame:
    """
    Merge Excel files and perform a quality check for a given dataset.

    The function first merges Excel files and then performs quality checks.
    If any errors occur during processing, the program will exit with status code 1.

    Parameters:
        input_folder (Path or str): Location of files.
        files (list): List of file paths (filenames).
        prefix (str): A string prefix to be used for the dataset.
        isMandatory (bool): Whether missing data should abort the run.

    Returns:
        pd.DataFrame: The processed DataFrame after merging and quality check.
    """
    from src.excel_exchange import merge_excel_files

    try:
        # First step: merge all Excel files into one DataFrame
        df = merge_excel_files(input_folder, files, prefix, isMandatory)
    except (FileNotFoundError, ValueError) as e:
        # Exit if there are issues with finding or merging files
        print(e)
        sys.exit(1)

    try:
        # Second step: perform quality checks on the merged data
        df = quality_check(df, prefix)
    except (TypeError, ValueError) as e:
        # Exit if the data doesn't pass quality checks
        print(e)
        sys.exit(1)

    return df


def quality_check(
    df_input: pd.DataFrame,
    df_identifier: str
) -> pd.DataFrame:
    """
    Perform quality checks and standardization on an input DataFrame.
    - Converts column names to lowercase
    - Converts 'scenario' and 'generator_id' values to lowercase if present
    - Checks grid columns for invalid underscore characters

    Args:
        df_input (pd.DataFrame): DataFrame to validate.
        df_identifier (str): Name for error messages.

    Returns:
        pd.DataFrame: Cleaned DataFrame.
    """
    # Skip processing if DataFrame is empty
    if df_input.empty:
        return pd.DataFrame()

    # Standardize column names to lowercase
    df_input.columns = df_input.columns.str.lower()

    # Standardize specific columns to lowercase strings
    cols = ['scenario', 'generator_id']
    for col in cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).str.lower()

    # Check grid columns for invalid underscores
    grid_columns = [col for col in df_input.columns if 'grid' in col.lower()]
    for col in grid_columns:
        if df_input[col].astype(str).str.contains('_', na=False).any():
            bad_values = df_input.loc[df_input[col].astype(str).str.contains('_', na=False), col].unique()
            raise ValueError(f"Invalid values in '{df_identifier}' column '{col}' containing underscores: {bad_values}")

    return df_input


def build_node_column(df: pd.DataFrame) -> pd.DataFrame:
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


def build_from_to_columns(df: pd.DataFrame) -> pd.DataFrame:
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
    missing_mask = df['unittype'].isna()
    if missing_mask.any() and log_status is not None:
        for generator_id in df.loc[missing_mask, 'generator_id'].unique():
            log_status(
                f"unitdata generator_id '{generator_id}' does not have a matching generator_id "
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


def filter_df_whitelist(
    df_input: pd.DataFrame,
    df_name: str,
    filters: Dict[str, Union[str, int, List[Union[str, int]]  ]  ]
) -> pd.DataFrame:
    """
    Filter DataFrame based on whitelist of allowed values.
    
    Parameters:
    -----------
    df_input : pandas.DataFrame
        The DataFrame to be filtered
    df_name : str
        Name of the DataFrame (used for error reporting)
    filters : dict
        Dictionary of {column_name: allowed_values} pairs
    
    Returns:
    --------
    pandas.DataFrame
        Filtered DataFrame containing only rows where column values match the whitelist
    
    Notes:
    ------
    - Always accepts 'all' for scenario column and 1 for year column
    - String comparisons are case-insensitive
    """
    # Skip processing if df_input is empty
    if df_input.empty:
        return df_input

    # Check if all filter columns exist in the DataFrame
    missing_cols = [col for col in filters if col not in df_input.columns]
    if missing_cols:
        raise Exception(f"Missing columns in {df_name}: {missing_cols}")

    # Create a copy to avoid modifying the original DataFrame
    df_filtered = df_input.copy()
    
    # Apply each filter condition
    for col, val in filters.items():
        # Ensure val is a list for consistent processing
        if not isinstance(val, list):
            val = [val]

        # Add special always-included values
        if col == 'scenario': val.append('all')  # Always include 'all' scenario
        if col == 'year': val.append(1)          # Always include year 1

        # Handle string columns with case-insensitive comparison
        if pd.api.types.is_string_dtype(df_filtered[col]):
            # Convert filter values to lowercase if they're strings
            lowered_vals = [v.lower() if isinstance(v, str) else v for v in val]
            df_filtered = df_filtered[df_filtered[col].str.lower().isin(lowered_vals)]
        else:
            # Direct comparison for non-string columns
            df_filtered = df_filtered[df_filtered[col].isin(val)]

    return df_filtered


def filter_df_blacklist(
    df_input: pd.DataFrame,
    df_name: str,
    filters: Dict[str, Union[str, int, List[Union[str, int]]  ]  ]
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

    # Check if all filter columns exist in the DataFrame
    missing_cols = [col for col in filters if col not in df_input.columns]
    if missing_cols:
        raise Exception(f"Missing columns in {df_name}: {missing_cols}")

    # Create a copy to avoid modifying the original DataFrame
    df_filtered = df_input.copy()
    
    # Apply each filter condition (excluding blacklisted values)
    for col, val in filters.items():
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


def keep_last_occurance(
    df_input: pd.DataFrame,
    key_columns: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Keep only the last occurrence of rows with duplicate values in key columns.
    
    Parameters:
    -----------
    df_input : pandas.DataFrame
        The DataFrame to process
    key_columns : list or None
        Columns to consider when identifying duplicates. 
        If None, all columns except 'value' are used.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with duplicates removed, keeping only the last occurrence
    """
    # If key_columns is not provided, default to all columns except 'value'
    if key_columns is None:
        key_columns = [col for col in df_input.columns if col.lower() != 'value']
    else:
        # Keep only the columns that exist in the DataFrame
        key_columns = [col for col in key_columns if col in df_input.columns]
    
    # If the DataFrame is not empty and we have valid key columns, drop duplicates
    if not df_input.empty and key_columns:
        df_input = df_input.drop_duplicates(subset=key_columns, keep='last')
    
    return df_input


def filter_nonzero_numeric_rows(df: pd.DataFrame, exclude: list[str] = None) -> pd.DataFrame:
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