# src/GDX_exchange.py

from typing import Any, Dict, Optional, Sequence, List
from src.utils import log_status
import pandas as pd
import numpy as np
import os
import glob
import gams.transfer as gt
from tqdm import tqdm



# ==============================================================================
# WRITE FUNCTIONS
# ==============================================================================

def write_BB_gdx(
    df: Optional[pd.DataFrame],
    output_file: str,
    logs: List[str],
    **kwargs: Any
    ) -> None:
    """
    Write a DataFrame to a GDX file using gams.transfer.

    Parameters:
        df: DataFrame with columns matching bb_parameter_dimensions + 'value'
        output_file: Path to output GDX file
        logs: List for logging messages
        **kwargs: Must include:
            - bb_parameter: str -> GDX parameter name
            - bb_parameter_dimensions: Sequence[str] -> dimension columns (optional, inferred if missing)

    Returns:
        None: Writes content to output_file
    """
    if df is None or len(df) == 0:
        log_status(f"Skipping writing GDX '{output_file}': No data to write", logs, level="warn")
        return

    bb_parameter: Optional[str] = kwargs.get("bb_parameter")
    dims: Optional[Sequence[str]] = kwargs.get("bb_parameter_dimensions")

    if not bb_parameter:
        log_status(f"Missing required kwarg 'bb_parameter' from '{output_file}'", logs, level="warn")
        return

    # Infer dimensions if not provided
    if not dims or len(dims) == 0:
        dims = [c for c in df.columns if c != "value"]

    # Validate columns
    final_cols = list(dims) + ["value"]
    missing = [c for c in final_cols if c not in df.columns]
    if missing:
        log_status(f"DataFrame missing required columns: {missing} for '{output_file}'", logs, level="warn")
        return

    # Select only required columns
    work = df[final_cols]

    m = gt.Container()

    # Create Sets for each dimension
    dim_sets = {}
    for d in dims:
        unique_vals = work[d].unique()
        dim_sets[d] = gt.Set(m, d, records=unique_vals.tolist(), description=f"{d} domain")

    # Create Parameter
    domain = [dim_sets[d] for d in dims]
    p_desc = str(kwargs.get(bb_parameter, f"{bb_parameter} written via gams.transfer"))
    param = gt.Parameter(m, bb_parameter, domain, description=p_desc)
    param.setRecords(work)

    # Write
    m.write(output_file)

def write_BB_gdx_annual(
    df: Optional[pd.DataFrame],
    output_folder: str,
    logs: List[str],
    **kwargs: Any
    ) -> None:
    """
    Split a multi-year timeseries DataFrame by year and write each year to a separate GDX file.

    This function:
      1. Splits the DataFrame by year using split_timeseries_to_annual_gdx_frames()
      2. Remaps 't' indices to t000001..t008760 for each year
      3. Writes each year to a separate GDX file (or single file if only one year)
    
    Parameters:
        df: Multi-year DataFrame with columns matching bb_parameter_dimensions + helper columns
        output_folder: Directory where GDX files will be written
        logs: List for logging messages
        **kwargs: Must include:
            - bb_parameter: str -> GDX parameter name
            - bb_parameter_dimensions: Sequence[str] -> dimension columns
            - gdx_name_suffix: str (optional) -> suffix for output filename
    
    Returns: 
        None: Writes content to
        - Single year: {bb_parameter}_{gdx_name_suffix}.gdx
        - Multiple years: {bb_parameter}_{gdx_name_suffix}_{year}.gdx
    """
    # --- initialization and checks ---
    if df is None or len(df) == 0:
        log_status(f"Skipping annual GDX writing for '{kwargs.get('bb_parameter', '?')}_{kwargs.get('gdx_name_suffix', '?')}': no data to write.", logs, level="warn")
        return

    bb_parameter: Optional[str] = kwargs.get("bb_parameter")
    dims: Optional[Sequence[str]] = kwargs.get("bb_parameter_dimensions")
    gdx_name_suffix: Optional[str] = kwargs.get("gdx_name_suffix", "")

    if not bb_parameter:
        log_status(f"Missing required kwarg 'bb_parameter' for annual GDX writing (gdx_name_suffix='{gdx_name_suffix}').", logs, level="warn")
        return
    
    # If dims not provided, infer all columns except 'value' as dimensions, preserving order.
    if not dims or len(dims) == 0:       
        dims = [c for c in df.columns if c != "value"]

    # Validate columns
    missing = [c for c in list(dims) + ["value"] if c not in df.columns]
    if missing:
        log_status(f"DataFrame missing required columns: {missing} for '{output_file}'", logs, level="warn")
        return 

    # Final columns of the written dataframe
    final_cols = list(dims) + ["value"]

    # --- Split to annual dfs ---
    # Split into annual frames
    annual_dfs = split_timeseries_to_annual_gdx_frames(
        df, logs, bb_parameter_dimensions=dims
    )
    
    if not annual_dfs:
        log_status(f"No annual data available to write for '{bb_parameter}_{gdx_name_suffix}'.", logs, level="warn")
        return
    
    # pick key characteristics
    years = sorted(annual_dfs.keys())
    single_year = (len(years) == 1)

    # --- prepare and write annual gdx files ---
    fname_base = f"{bb_parameter}_{gdx_name_suffix}" if gdx_name_suffix else f"{bb_parameter}"

    # Build container and sets
    m = gt.Container()

    # Create a Set for each dimension with records = unique labels
    dim_sets = {}
    for d in dims:
        if d == 't':
            # For 't', use only one year's worth (always t000001..t008760)
            # Pick from first annual df since all years have identical 't' structure
            unique_vals = annual_dfs[years[0]][d].unique()
        else:
            # For other dimensions, collect unique values across ALL years
            unique_vals = pd.concat([annual_dfs[yr][d] for yr in years]).unique()

        dim_sets[d] = gt.Set(m, d, records=unique_vals.tolist(), description=f"{d} domain")

    # Prepare parameter
    domains = [dim_sets[d] for d in dims]
    p_desc = str(kwargs.get(bb_parameter, f"{bb_parameter}"))
    param = gt.Parameter(m, bb_parameter, domains, description=p_desc)

    for yr in tqdm(years, desc="  Writing"):
        # pick annual df
        df_y = annual_dfs[yr]

        # Filter to final_cols
        df_y = df_y[final_cols]

        # populate parameter
        param.setRecords(df_y)

        # Add year to filename if multiple years
        fname = f"{fname_base}_{yr}.gdx" if not single_year else f"{fname_base}.gdx"
        output_file = os.path.join(output_folder, fname)

        # Write
        m.write(output_file)


# ==============================================================================
# DF PROCESSING
# ==============================================================================

def prepare_BB_df(
    df: pd.DataFrame,
    start_date: str,
    country_codes: List[str],
    **kwargs: Any
    ) -> pd.DataFrame:
    """
    Assumes wide format hourly input DataFrame where value column titles are node names
        
    Prepares the input DataFrame for Backbone-compatible GDX conversion.
        * Creates and processes 't' column if 't' in bb_parameter_dimensions
        * converts tables from wide format to long format
        * Creates and processes 'grid' column if 'grid' in bb_parameter_dimensions
        * Creates and processes 'flow' column if 'flow' in bb_parameter_dimensions
        * Creates and processes 'group' column if 'group' in bb_parameter_dimensions
        * Creates and processes 'f' column if 'f' in bb_parameter_dimensions
        * returns DataFrame based on bb_parameter_dimensions
        
    Raises ValueError if not all bb_parameter_dimensions are in the final returned df
    """
    # picking mandatory kwargs
    processor_name = kwargs.get('processor_name')
    bb_parameter_dimensions = kwargs.get('bb_parameter_dimensions')
    custom_column_value = kwargs.get('custom_column_value')
    # optional kwargs
    is_demand = kwargs.get('is_demand', False)

    # Create 't' column and convert it to t-index
    if 't' in bb_parameter_dimensions:
        # reset index, rename to 'time' if not already named
        df = df.reset_index()
        if 'time' not in df.columns and 'index' in df.columns:
            df = df.rename(columns={'index': 'time'})
            
        # calculate 't' column values 't000001' etc based on default_timeorigin
        default_timeorigin = pd.to_datetime(start_date, dayfirst=True)
        times = pd.to_datetime(df['time'], dayfirst=True)
        first_time = times.iloc[0]
        if first_time != default_timeorigin:
            time_diff = first_time - default_timeorigin
            times = times - time_diff
        unique_times = pd.unique(times)
        time_mapping = {time: 't' + str(i + 1).zfill(6) for i, time in enumerate(unique_times)}
        df['t'] = times.map(time_mapping)

        # categorize
        df['t'] = df['t'].astype('category')

    # Identify value columns based on country codes, treat other as dimensions
    value_vars = [col for col in df.columns if any(code in col for code in country_codes)]
    id_vars = [col for col in df.columns if col not in value_vars]
    # Melt the DataFrame to long format.
    melted_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                        var_name='node', value_name='value')

    # categorize node
    melted_df['node'] = melted_df['node'].astype('category')  

    # Manage possible grid column.
    if 'grid' in bb_parameter_dimensions:
        if custom_column_value is not None and custom_column_value.get('grid') is not None:
            melted_df['grid'] = custom_column_value['grid']
        else:
            melted_df['grid'] = melted_df['node'].apply(lambda x: x.split('_')[1] if '_' in x else x)
        # categorize
        melted_df['grid'] = melted_df['grid'].astype('category')

    # Manage possible flow column.
    if 'flow' in bb_parameter_dimensions:
        if custom_column_value is not None and custom_column_value.get('flow') is not None:
            melted_df['flow'] = custom_column_value['flow']
        else:
            melted_df['flow'] = melted_df['node'].apply(lambda x: x.split('_')[1] if '_' in x else x)
        # categorize
        melted_df['flow'] = melted_df['flow'].astype('category')            

    # Manage possible group column.
    if 'group' in bb_parameter_dimensions:
        if custom_column_value is not None and custom_column_value.get('group') is not None:
            melted_df['group'] = custom_column_value['group']
        else:
            melted_df['group'] = "UC_" + melted_df['node'].astype(str)
        # categorize
        melted_df['group'] = melted_df['group'].astype('category') 

    # Manage possible f column.
    if 'f' in bb_parameter_dimensions:
        if custom_column_value is not None and custom_column_value.get('f') is not None:
            melted_df['f'] = custom_column_value['f']
        else:
            melted_df['f'] = 'f00'
        # categorize
        melted_df['f'] = melted_df['f'].astype('category')             

    # negative influx for demands
    if is_demand:
        melted_df['value'] = melted_df['value'] * -1

    # add year helper column
    melted_df['year'] = melted_df['time'].dt.year

    # add value, year, and time to final_cols
    final_cols = bb_parameter_dimensions.copy()
    final_cols.append('value')
    final_cols.append('year')
    final_cols.append('time')

    # Raise error if some of the final_cols not in melted_df
    missing_cols = set(final_cols) - set(melted_df.columns)
    if missing_cols:
        raise ValueError(f"The following columns are missing in the {processor_name} DataFrame: {missing_cols}")

    # pick only final_cols to returned df, polish, and return
    melted_df = melted_df[final_cols]
    melted_df['value'] = melted_df['value'].fillna(value=0)

    return melted_df


def calculate_average_year_df(
    input_df: pd.DataFrame,
    round_precision: int = 0,
    **kwargs: Any,
    ) -> pd.DataFrame:
    """
    Assumes long format input_df with columns bb_parameter_dimensions and 'time' and 'value'
        * 'time' is datetime format presentation of time, e.g 2000-01-01 00:00:00
        * 't' is t-index format presentation of time, e.g. t000001

    Calculates 'hour_of_year' as hours elapsed since Jan 1, starting at 1.
    Checks dim_cols as columns in bb_parameter_dimensions except 'f' and 't'
    for each unique dim_col:
        * Calculates quantiles for each hour_of_year of processed years using the quantile mapping.
        * Maps quantiles to 'f' based on quantile_map provided via kwargs

    Raises following errors
        * ValueError if input_df does not have 'time' columns
        * ValueError if input_df cover less than two years
        * ValueError if input_df columns do not have all bb_parameter_dimensions
        * ValueError if bb_parameter_dimensions cover only 'f' and 't'
    """

    # ---- Parameters and checks ----
    processor_name = kwargs.get("processor_name")  # for error messages
    bb_parameter_dimensions = kwargs.get("bb_parameter_dimensions")
    quantile_map = kwargs.get("quantile_map")

    if bb_parameter_dimensions is None or quantile_map is None:
        raise ValueError(
            f"processor '{processor_name}': 'bb_parameter_dimensions' and 'quantile_map' must be provided."
        )

    # Check for 'time' column.
    if "time" not in input_df.columns:
        raise ValueError(
            f"processor '{processor_name}' does not have 'time' column. "
            "Check previous functions."
        )

    # Ensure time column is datetime (avoid re-parsing if already correct dtype).
    if not np.issubdtype(input_df["time"].dtype, np.datetime64):
        input_df = input_df.copy()
        input_df["time"] = pd.to_datetime(input_df["time"])

    # Check that data covers more than one year (no need to sort for this).
    years = input_df["time"].dt.year.unique()
    if len(years) <= 1:
        raise ValueError(
            f"processor '{processor_name}' DataFrame covers only a year or less. "
            "Cannot calculate average year. Use 'calculate_average_year': False in config file?"
        )

    # Check that bb_parameter_dimensions includes both 't' and 'f'
    if not ("t" in bb_parameter_dimensions and "f" in bb_parameter_dimensions):
        raise ValueError(
            f"processor '{processor_name}' dimensions '{bb_parameter_dimensions}' "
            "do not include 'f' and 't'. Cannot calculate average year. "
            "Use 'calculate_average_year': False in config file?"
        )

    # ---- Dimension handling ----
    # Determine additional dimension columns from bb_parameter_dimensions, excluding 'f' and 't'
    dim_cols = [
        col
        for col in bb_parameter_dimensions
        if col not in ("f", "t") and col in input_df.columns
    ]

    # Check that we have all requested dimensions in the DataFrame
    if set(dim_cols + ["f", "t"]) != set(bb_parameter_dimensions):
        raise ValueError(
            f"Average year calculation did not find all bb_parameter_dimensions: "
            f"'{bb_parameter_dimensions}' from processor '{processor_name}' DataFrame columns."
        )

    if not dim_cols:
        raise ValueError(
            f"processor '{processor_name}' dimensions '{bb_parameter_dimensions}' "
            "do not include anything but 'f' and 't'."
        )

    # Restrict columns early to reduce memory use in heavy operations.
    # We no longer need original 'f', 't', or any extra columns at this point.
    cols_to_keep = dim_cols + ["time", "value"]
    input_df = input_df[cols_to_keep].copy()

    # ---- Create helper columns ----
    # Fast hour_of_year: avoid datetime arithmetic, use dayofyear + hour.
    time = input_df["time"]
    day_of_year = time.dt.dayofyear.to_numpy()
    hour = time.dt.hour.to_numpy()
    hour_of_year = (day_of_year - 1) * 24 + hour + 1

    input_df["hour_of_year"] = hour_of_year.astype(np.int32)

    # Only process hours up to 8760 (ignore extra hours from leap years)
    input_df = input_df[input_df["hour_of_year"] <= 8760]

    # ---- Quantile computation ----
    # Vectorized quantile computation:
    # Group by the additional dimensions and 'hour_of_year' then compute the quantiles.
    q_values = list(quantile_map.keys())

    df_quant = (
        input_df
        .groupby(dim_cols + ["hour_of_year"], observed=True)["value"]
        .quantile(q_values)
        # quantile with sequence -> MultiIndex with a 'quantile' level
        .rename_axis(index=dim_cols + ["hour_of_year", "quantile"])
        .reset_index()
    )
    # df_quant now has columns: dim_cols..., 'hour_of_year', 'quantile', 'value'

    # ---- Build full grid (Cartesian product) ----
    # Unique combinations of all dimension columns
    unique_dims = input_df[dim_cols].drop_duplicates()

    # Hours 1..8760
    hours_df = pd.DataFrame({"hour_of_year": np.arange(1, 8761, dtype=np.int32)})

    # Quantiles as in quantile_map (order preserved)
    quantiles_df = pd.DataFrame({"quantile": q_values})

    # Cross join using pandas 'cross' merge 
    full_grid = (
        unique_dims
        .merge(hours_df, how="cross")
        .merge(quantiles_df, how="cross")
    )

    # Merge the computed quantile results with the complete grid.
    df_full = full_grid.merge(
        df_quant,
        on=dim_cols + ["hour_of_year", "quantile"],
        how="left",
    )

    # ---- Prepare final DataFrame ----
    # Create t-index as 't000001' style, vectorized
    df_full["t"] = "t" + df_full["hour_of_year"].astype(str).str.zfill(6)
    df_full["t"] = df_full["t"].astype("category")

    # Map quantile -> f
    df_full["f"] = df_full["quantile"].map(quantile_map)
    df_full["f"] = df_full["f"].astype("category")

    # Fill missing quantile values with 0, then round
    df_full["value"] = df_full["value"].fillna(0)
    if round_precision is not None:
        df_full["value"] = df_full["value"].round(round_precision)

    # Reorder columns to match bb_parameter_dimensions plus 'value'
    df_final = df_full[bb_parameter_dimensions + ["value"]]

    return df_final


def split_timeseries_to_annual_gdx_frames(
    df: Optional[pd.DataFrame],
    logs: List[str],
    *,
    bb_parameter_dimensions: Sequence[str],
    time_col: str = "time",
    year_col: str = "year",
    max_hours: int = 8760,
    nan_to_zero: bool = True,
    inf_to_zero: bool = True,
    ) -> Dict[int, pd.DataFrame]:
    """
    Split a multi-year timeseries DF into annual DF chunks and remap 't' labels per year.

    - Drops 't' if present and rebuilds it as t000001.. per (year × dims{f,t} ) group
    - Truncates to `max_hours` to avoid leap-year overflow
    - Casts all dimension columns to str and 'value' to float64
    - Converts NaN/inf to zero if nan_to_zero=True
    - Ensures each annual frame has a RangeIndex (fast path for gams.transfer.setRecords)

    Returns:
        dict[year -> DataFrame with columns bb_parameter_dimensions + ['value']]
    """
    if df is None or len(df) == 0:
        log_status(f"[split_timeseries_to_annual_gdx_frames] No data to split (empty df), dims={list(bb_parameter_dimensions)}.", logs, level="error")
        return {}

    dims = list(bb_parameter_dimensions)
    dims_no_t = [d for d in dims if d != "t"]
    required_cols = set(dims_no_t) | {time_col, year_col, "value"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        log_status(f"[split_timeseries_to_annual_gdx_frames] Missing required columns {missing} (dims={list(bb_parameter_dimensions)}).", logs, level="error")
        return {}

    # Work copy without 't' (rebuild it below)
    work = df.drop(columns="t", errors="ignore").copy()

    if work["value"].dtype != np.float64:
        work["value"] = work["value"].astype(np.float64)


    # Handle NaN/inf before splitting
    if nan_to_zero or inf_to_zero:
        mask = pd.isna(work["value"])
        if inf_to_zero:
            mask |= ~np.isfinite(work["value"])
        
        n_bad = mask.sum()
        if n_bad:
            log_status(f"Converted {n_bad} NaN/inf values to 0.0 during annual split.", logs, level="info")
            work.loc[mask, "value"] = 0.0

    # Grouping excludes f and t
    group_dims = [c for c in dims if c not in {"f", "t"}]
    
    # Pre-compute 't' labels once (reusable array)
    t_labels = np.array(['t' + str(i).zfill(6) for i in range(1, max_hours + 1)])
    
    final_cols = dims + ["value"]
    out: Dict[int, pd.DataFrame] = {}

    # Pre-compute 't' labels once
    t_labels = np.array(['t' + str(i).zfill(6) for i in range(1, max_hours + 1)])

    final_cols = dims + ["value"]
    out: Dict[int, pd.DataFrame] = {}

    # Process by year first (smaller chunks)
    for yr, df_yr in work.groupby(year_col, observed=True, sort=False):
        df_yr = df_yr.copy()

        if group_dims:
            # Sort only within this year
            sort_cols = group_dims + [time_col]
            df_yr = df_yr.sort_values(sort_cols, kind="mergesort")

            # OPTIMIZATION: Replace cumcount() with numpy operations
            # Create group IDs and compute differences
            group_ids = df_yr.groupby(group_dims, observed=True, sort=False).ngroup()

            # Fast row numbering: count within groups using diff
            group_changes = np.diff(group_ids.values, prepend=-1) != 0
            row_nums = np.arange(len(df_yr))
            row_nums -= np.repeat(row_nums[group_changes], np.diff(np.append(np.where(group_changes)[0], len(df_yr))))

            df_yr['_row_num'] = row_nums
        else:
            # No grouping needed
            df_yr['_row_num'] = np.arange(len(df_yr))

        # Filter to max_hours
        df_yr = df_yr[df_yr['_row_num'] < max_hours]

        # OPTIMIZATION: Direct array assignment (already fast, but ensure no copy)
        row_nums_filtered = df_yr['_row_num'].values
        df_yr['t'] = t_labels[row_nums_filtered]

        # Drop temporary columns
        frame = df_yr[final_cols].reset_index(drop=True)
        out[int(yr)] = frame
    
    if not out:
        log_status(f"[split_timeseries_to_annual_gdx_frames] No annual frames produced after remap/filter (dims={list(bb_parameter_dimensions)}).", logs, level="warn")

    return out


# ==============================================================================
# TS IMPORT FILE GENERATION
# ==============================================================================

def update_import_timeseries_inc(
    output_folder: str,
    file_suffix: Optional[str] = None,
    **kwargs: Any
    ) -> None:
    """
    Updates the import_timeseries.inc file by generating a GAMS code block that imports
    parameter data from GDX files. The function looks for matching GDX files in the output folder
    based on specified parameter names and patterns, then creates the necessary GAMS code to load
    parameters from these files.
        
    Args:
        output_folder (str): Directory path where GDX files are located and where import_timeseries.inc will be created/updated
        file_suffix (str, optional): Specific suffix for the GDX file. If None, searches for files with standard patterns
        **kwargs: Additional parameters including:
            - bb_parameter (str): Name of the Backbone parameter to import
            - gdx_name_suffix (str): Suffix to be used in the GDX filename
        
    Returns:
        None: Writes content to import_timeseries.inc file in the output_folder
    """        
    # Prepare required parameters
    bb_parameter = kwargs.get('bb_parameter')
    gdx_name_suffix = kwargs.get('gdx_name_suffix')

    # If file_suffix flag is True, search for the specific file.
    if file_suffix is not None:
        filename = os.path.join(output_folder, f'{bb_parameter}_{gdx_name_suffix}_{file_suffix}.gdx')
        if os.path.exists(filename):
            matching_files = filename
        else:
            raise FileNotFoundError(f"{bb_parameter}_{gdx_name_suffix}_{file_suffix}.gdx not found in {output_folder}.")

    else:
        # Check for the two patterns in the output_folder
        # Pattern a: a single file: f'{bb_parameter}_{gdx_name_suffix}.gdx'
        file_a = os.path.join(output_folder, f'{bb_parameter}_{gdx_name_suffix}.gdx')
        if os.path.exists(file_a):
            matching_files = file_a
            file_suffix = None
        else:
            # Pattern b: multiple files, e.g., f'{bb_parameter}_{gdx_name_suffix}_{yr}.gdx' where yr is four digit integer, e.g. 2014
            pattern_b = os.path.join(output_folder, f'{bb_parameter}_{gdx_name_suffix}_[0-9][0-9][0-9][0-9].gdx')
            matching_files = glob.glob(pattern_b)
            if matching_files:
                file_suffix = "%climateYear%"

    if matching_files is None:
        raise FileNotFoundError(f"{bb_parameter}_{gdx_name_suffix}.gdx or {bb_parameter}_{gdx_name_suffix}_year.gdx not found in {output_folder}.")


    # --- build text_block ---
    # Creating a text block with a specific structure to read GDX to Backbone
    if file_suffix is None:
        gdx_name = f"{bb_parameter}_{gdx_name_suffix}.gdx"
    else:
        gdx_name = f"{bb_parameter}_{gdx_name_suffix}_{file_suffix}.gdx"

    # Constructing text block content: 
    text_block = "\n".join([
        f"$ifthen exist '%input_dir%/{gdx_name}'",
        f"    // If {gdx_name} exists, load input data",
        f"    $$gdxin '%input_dir%/{gdx_name}'",
        f"    $$loaddcm {bb_parameter}",
        "    $$gdxin",
        "$endIf",
        ""
    ]) + "\n"


    # --- write text_block only if not already present ---
    # Define the output file path
    output_file = os.path.join(output_folder, 'import_timeseries.inc')

    # Read existing content (or empty string if file doesn't exist)
    try:
        with open(output_file, 'r') as f:
            existing = f.read()
    except FileNotFoundError:
        existing = ''

    # Append only if the exact block isn't already in the file
    if text_block not in existing:
        with open(output_file, 'a') as f:
            f.write(text_block)
    else:
        pass

