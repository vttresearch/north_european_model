from typing import Any, Dict, Optional, Sequence, List
from src.utils import log_status
import pandas as pd
import numpy as np
import os
import glob
import gams.transfer as gt


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
    melted_df = melted_df.convert_dtypes()
    return melted_df


def write_BB_gdx(
    df: Optional[pd.DataFrame],
    output_file: str,
    logs: List[str],
    **kwargs: Any
) -> None:
    """
    Writes the DataFrame to gdx

    Variables: output_file (DataFrame) with 'value' column. Other columns are treated as dimensions.
               bb_parameter (string) used to create gdx parameter name
               bb_parameter_dimensions (list of strings) used to filter written columns.  E.g. ['grid', 'node', 'f', 't']
    """
    from gdxpds import to_gdx

    if df is None:
        log_status(f"Skipping writing GDX '{output_file}': No data to write", logs, level="warn")
        return
        
    # Prepare required parameters
    bb_parameter = kwargs.get('bb_parameter')        
    bb_parameter_dimensions = kwargs.get('bb_parameter_dimensions')

    # add 'value' to final_cols
    final_cols = bb_parameter_dimensions.copy()
    final_cols.append('value')

    # write gdx
    dataframes = {bb_parameter: df}
    to_gdx(dataframes, path=output_file)


def write_BB_gdx_gt(
    df: Optional[pd.DataFrame],
    output_file: str,
    logs: List[str],
    **kwargs: Any
) -> None:
    """
    Write a DataFrame to a GDX using gams.transfer.

    Expected kwargs:
      - bb_parameter: str   -> GDX parameter name
      - bb_parameter_dimensions: Sequence[str] -> domain order, e.g. ['grid','node','f','t']

    Notes:
      - Only the columns listed in bb_parameter_dimensions plus 'value' are written.
      - Each dimension becomes a Set with the same name and records from the DataFrame.
      - Non-finite values in 'value' are dropped (NaN/inf).
    """
    if df is None or len(df) == 0:
        log_status(f"Skipping writing GDX '{output_file}': No data to write", logs, level="warn")
        return

    bb_parameter: Optional[str] = kwargs.get("bb_parameter")
    dims: Optional[Sequence[str]] = kwargs.get("bb_parameter_dimensions")

    if not bb_parameter:
        log_status(f"Missing required kwarg 'bb_parameter' from '{output_file}'", logs, level="warn")
        return        

    if not dims or len(dims) == 0:
        # If not provided, infer all columns except 'value' as dimensions, preserving order.
        dims = [c for c in df.columns if c != "value"]

    # Validate columns
    missing = [c for c in list(dims) + ["value"] if c not in df.columns]
    if missing:
        log_status(f"DataFrame missing required columns: {missing} for '{output_file}'", logs, level="warn")
        return

    # Select & order columns
    final_cols = list(dims) + ["value"]
    work = df[final_cols].copy()

    # Enforce types: all dims -> string labels; value -> float
    for d in dims:
        work[d] = work[d].astype(str)

    # Drop non-finite values (NaN/inf) because GDX parameters require numeric values
    before = len(work)
    work = work[np.isfinite(work["value"].astype(float))]
    after = len(work)
    if after < before:
        log_status(f"write_gdx_with_gt: dropped {before - after} rows with non-finite 'value' for '{output_file}'", logs, level="info")
    work["value"] = work["value"].astype(float) 

    # Build container and sets
    m = gt.Container()

    # Create a Set for each dimension with records = unique labels (order of first appearance)
    dim_sets = {}
    for d in dims:
        # Preserve order of first appearance
        labels = pd.unique(work[d])
        dim_sets[d] = gt.Set(m, d, records=[str(x) for x in labels], description=f"{d} domain")

    # Create the parameter with the domain sets in the given order
    domain = [dim_sets[d] for d in dims]
    p_desc = str(kwargs.get(bb_parameter, f"{bb_parameter} written via gams.transfer"))
    param = gt.Parameter(m, bb_parameter, domain, description=p_desc)

    # GT expects a DataFrame with columns = dims + ['value']
    param.setRecords(work[final_cols])

    # Finally, write the GDX
    m.write(output_file)


def calculate_average_year_df(
    input_df: pd.DataFrame,
    round_precision: int = 0,
    **kwargs: Any
) -> pd.DataFrame:
    """
    Assumes long format input_df with columns bb_parameter_dimensions and 'time' and 'value'
        * 'time' is datatime format presentation of time, e.g 2000-01-01 00:00:00 
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
    # Retrieve mandatory kwargs
    processor_name = kwargs.get('processor_name') # for error messages
    bb_parameter_dimensions = kwargs.get('bb_parameter_dimensions')
    quantile_map = kwargs.get('quantile_map')

    # Check for 'time' column.
    if 'time' not in input_df.columns:
        raise ValueError(f"processor '{processor_name}' does not have 'time' column. Check previous functions.")

    # Ensure time column is datetime and sort by time.
    input_df['time'] = pd.to_datetime(input_df['time'])
    input_df = input_df.sort_values(by='time')

    # Check that data covers more than one year.
    years = input_df['time'].dt.year.unique()
    if len(years) <= 1:
        raise ValueError(f"processor '{processor_name}' DataFrame covers only a year or less. Cannot calculate average year. Use 'calculate_average_year': False in config file?")

    # Check that bb_parameter_dimensions includes both 't' and 'f'
    if not ('t' in bb_parameter_dimensions and 'f' in bb_parameter_dimensions):
        raise ValueError(f"processor '{processor_name}' dimensions '{bb_parameter_dimensions}' do not include 'f' and 't'. Cannot calculate average year. Use 'calculate_average_year': False in config file?")

    # ---- Create helper columns ----
    # Create 'hour_of_year' column (hours elapsed since Jan 1, starting at 1).
    start_of_year = input_df['time'].dt.to_period('Y').dt.start_time
    input_df['hour_of_year'] = ((input_df['time'] - start_of_year).dt.total_seconds() // 3600 + 1).astype(int)
    # Only process hours up to 8760 (ignore extra hours from leap years)
    input_df = input_df[input_df['hour_of_year'] <= 8760]

    # Determine additional dimension columns from bb_parameter_dimensions, excluding 'f' and 't'
    dim_cols = [col for col in bb_parameter_dimensions if col != 'f' and col != 't' and col in input_df.columns]
    if set(dim_cols + ['f'] + ['t']) != set(bb_parameter_dimensions):
        raise ValueError(f"Average year calculation did not find all bb_parameter_dimensions: '{bb_parameter_dimensions}' from processor '{processor_name}' DataFrame columns.")
    if not dim_cols:
        raise ValueError(f"processor '{processor_name}' dimensions '{bb_parameter_dimensions}' do not include anything but 'f' and 't'.")

    # ---- Quantile computation ----
    # Vectorized quantile computation:
    # Group by the additional dimensions and 'hour_of_year' then compute the three quantiles.
    df_quant = (
        input_df.groupby(dim_cols + ['hour_of_year'], observed=True)['value']
        .quantile(list(quantile_map.keys()))
        .reset_index(name='value')
    )
    # Rename the quantile column (it comes from the groupby quantile call)
    df_quant = df_quant.rename(columns={'level_{}'.format(len(dim_cols) + 1): 'quantile'})

    # Create a complete Cartesian product for each group, each quantile, and every hour from 1 to 8760.
    # First, get unique group combinations.
    unique_dims = input_df[dim_cols].drop_duplicates()
    # DataFrame for quantile values.
    quantiles_df = pd.DataFrame({'quantile': list(quantile_map.keys())})
    # DataFrame for hours.
    hours_df = pd.DataFrame({'hour_of_year': np.arange(1, 8761)})

    # Create the full_grid (Cartesian product using a cross join).
    unique_dims['_key'] = 1
    quantiles_df['_key'] = 1
    hours_df['_key'] = 1
    full_grid = unique_dims.merge(quantiles_df, on='_key').merge(hours_df, on='_key')
    full_grid = full_grid.drop('_key', axis=1)

    # Merge the computed quantile results with the complete grid.
    df_full = full_grid.merge(df_quant, on=dim_cols + ['hour_of_year', 'quantile'], how='left')

    # ---- Prepare df_final ----
    # Add the 't' and 'f' columns vectorized and categorize
    df_full['t'] = df_full['hour_of_year'].apply(lambda x: 't' + str(x).zfill(6))
    df_full['t'] = df_full['t'].astype('category')  
    df_full['f'] = df_full['quantile'].map(quantile_map)
    df_full['f'] = df_full['f'].astype('category') 

    # Fill missing quantile values with 0, round values
    df_full['value'] = df_full['value'].fillna(0)
    df_full['value'] = df_full['value'].round(round_precision)

    # Reorder columns to match bb_parameter_dimensions plus 'value'.
    # Here, bb_parameter_dimensions is assumed to include 't', 'f', and the remaining dimension columns.
    df_final = df_full[bb_parameter_dimensions + ['value']]
    return df_final


def write_BB_gdx_annual(
    df: Optional[pd.DataFrame],
    output_folder,
    logs: List[str],
    **kwargs: Any
) -> None:
    """
    Splits the processed DataFrame by year, remaps the time labels per year,
    and writes each year's data to a separate GDX file. 
    
    Variables: 
        df (DataFrame): DataFrame with 'value' column. Other columns are treated as dimensions.
        **kwargs: Additional parameters including:
            bb_parameter (string): Used to create GDX parameter name
            bb_parameter_dimensions (list of strings): Used to filter written columns. 
                                                       E.g. ['grid', 'node', 'f', 't']
            gdx_name_suffix (string): Used when generating the output file name
    
    Returns:
        None: Creates GDX files in the output folder
    
    Note: 
        - Drops hours after t8760 to handle leap years
    """
    from gdxpds import to_gdx

    if df is None:
        log_status("Abort: No data to write", logs, level="warn")
        return

    # Drop t for improved performance and reconstruct it later 
    df = df.drop(columns='t', errors='ignore')

    # Prepare required parameters
    bb_parameter = kwargs.get('bb_parameter')        
    bb_parameter_dimensions = kwargs.get('bb_parameter_dimensions')
    gdx_name_suffix = kwargs.get('gdx_name_suffix')

    # Assure consistent column order
    dim_cols = [col for col in bb_parameter_dimensions if col not in {'f', 't'}]
    final_cols = bb_parameter_dimensions + ['value']
    single_year = df['year'].nunique() == 1

    # Assign new hourly 't' values per dimension group and year
    grouped = df.groupby(['year'] + dim_cols, group_keys=False, observed=True)

    def map_t(group):
        group = group.sort_values('time')
        group = group.iloc[:8760]  # truncate leap years if needed
        group['t'] = ['t' + str(i + 1).zfill(6) for i in range(len(group))]
        return group
    
    df_remapped = grouped.apply(map_t)

    # process year-by-year
    for yr, group in df_remapped.groupby('year'):
        group_out = group[final_cols]

        # File naming
        file_name = (
            f"{bb_parameter}_{gdx_name_suffix}.gdx"
            if single_year
            else f"{bb_parameter}_{gdx_name_suffix}_{yr}.gdx"
        )
        file_path = os.path.join(output_folder, file_name)

        # Write to GDX
        to_gdx({bb_parameter: group_out}, path=file_path)


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
        log_status("Abort: No data to split (empty df).", logs, level="error")
        return {}

    dims = list(bb_parameter_dimensions)
    dims_no_t = [d for d in dims if d != "t"]
    required_cols = set(dims_no_t) | {time_col, year_col, "value"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        log_status(f"Skipping annual GDX writing due to missing required columns: {missing}", logs, level="error")
        return {}

    # Work copy without 't' (rebuild it below)
    work = df.drop(columns="t", errors="ignore").copy()

    # Convert types before grouping/sorting
    for d in dims_no_t:
        work[d] = work[d].astype(str)
    work["value"] = work["value"].astype(float)

    # Handle NaN/inf before splitting
    if nan_to_zero:
        n_nans = work["value"].isna().sum()
        if n_nans:
            log_status(f"Converted {n_nans} NaN values to 0.0 during annual split.", logs, level="info")
            work["value"] = work["value"].fillna(0.0)
    if inf_to_zero:
        mask_infs = ~np.isfinite(work["value"]) & work["value"].notna()
        n_infs = mask_infs.sum()
        if n_infs:
            log_status(f"Converted {n_infs} -inf/+inf values to 0.0 during annual split.", logs, level="info")
            work.loc[mask_infs, "value"] = 0.0

    # Grouping excludes f and t
    group_dims = [c for c in dims if c not in {"f", "t"}]
    
    # Sort once by year + group_dims + time
    sort_cols = [year_col] + group_dims + [time_col]
    work = work.sort_values(sort_cols, kind="mergesort")
    
    # Create row numbers within each group
    work['_row_num'] = work.groupby([year_col] + group_dims, observed=True, group_keys=False).cumcount()
    
    # Filter to max_hours
    work = work[work['_row_num'] < max_hours]
    
    # Create 't' labels vectorized
    work['t'] = 't' + (work['_row_num'] + 1).astype(str).str.zfill(6)
    
    # Drop temporary column and unnecessary columns
    work = work.drop(columns=['_row_num', time_col])

    final_cols = dims + ["value"]
    out: Dict[int, pd.DataFrame] = {}

    # Split by year
    for yr, df_y in work.groupby(year_col, observed=True, group_keys=False):
        # Select columns and reset index
        frame = df_y[final_cols].reset_index(drop=True)
        out[int(yr)] = frame

    if not out:
        log_status("No annual frames were produced after remap/filter.", logs, level="warn")

    return out


def write_BB_gdx_annual_gt(
    df: Optional[pd.DataFrame],
    output_folder: str,
    logs: List[str],
    **kwargs: Any
) -> None:
    """
    Split a multi-year timeseries DataFrame into annual frames and write each year
    to a separate GDX file using gams.transfer.

    Notes
    -----
    - This function assumes `split_timeseries_to_annual_gdx_frames(...)` has already
      normalized dtypes (dims -> str, value -> float64), rebuilt `t`, and ensured a
      RangeIndex for fast `setRecords`.
    - The domain sets for GDX symbols are constructed **from the first year's data**
      (except 't', which is fixed to t000001..t0008760). If later years contain
      additional labels, they will not be present in the sets and records may fail
      to match. Use with consistent domain coverage, or swap to a union-of-all-years
      approach if needed.

    Parameters
    ----------
    df : pandas.DataFrame or None
        Input timeseries with columns matching `bb_parameter_dimensions` + helper columns
        required by the splitter (e.g., 'year', 'time', etc.). If None or empty, a warning
        is logged and the function returns.
    output_folder : str
        Folder where per-year .gdx files will be written.
    logs : list[str]
        Accumulator for log messages; used by `log_status`.
    **kwargs : Any
        - bb_parameter : str
            Name of the GDX parameter to write.
        - bb_parameter_dimensions : Sequence[str]
            Domain order for the parameter (e.g., ['grid', 'node', 'f', 't']).
        - gdx_name_suffix : str, optional
            Suffix used in output file names (e.g., "..._<year>.gdx").

        Optionally (if you pass one) a key named equal to `bb_parameter` may hold a
        description string for the parameter; otherwise a default description is used.

    Returns
    -------
    None
        Writes one .gdx per year into `output_folder`.
    """
    bb_parameter = kwargs["bb_parameter"]
    dims = kwargs["bb_parameter_dimensions"]
    gdx_name_suffix = kwargs.get("gdx_name_suffix", "")

    # Split timeseries into annual frames
    annual = split_timeseries_to_annual_gdx_frames(
        df, logs, bb_parameter_dimensions=dims
    )
    if not annual:
        log_status("No annual data available to write.", logs, level="warn")
        return

    # Compute union labels from the first year only
    # NOTE: this assumes that all different values are present in the 
    # first year and can otherwise cause a crash.
    union_labels = {}
    first_year = min(annual.keys())
    first_year_data = annual[first_year]   
    for d in dims:
        if d == "t":
            # Fixed domain for time
            union_labels[d] = [f"t{str(i+1).zfill(6)}" for i in range(8760)]
        else:
            # Get unique values from first year only
            union_labels[d] = sorted(first_year_data[d].unique())

    # --- Build container & symbols ---
    m = gt.Container()
    dim_sets = {d: gt.Set(m, d, records=union_labels[d], description=f"{d} domain")
                for d in dims}
    domain = [dim_sets[d] for d in dims]
    desc = str(kwargs.get(bb_parameter, f"{bb_parameter} written via gams.transfer"))
    param = gt.Parameter(m, bb_parameter, domain, description=desc)

    final_cols = list(dims) + ["value"]

    # --- Write each year ---
    years_sorted = sorted(annual.keys())
    single_year = (len(years_sorted) == 1)

    for yr in years_sorted:
        df_y = annual[yr]

        # Frames from splitter already have correct dtypes and RangeIndex
        work = df_y[final_cols]
        param.setRecords(work)

        if single_year:
            fname = f"{bb_parameter}_{gdx_name_suffix}.gdx" if gdx_name_suffix else f"{bb_parameter}.gdx"
        else:
            suffix = f"_{gdx_name_suffix}" if gdx_name_suffix else ""
            fname = f"{bb_parameter}{suffix}_{yr}.gdx"

        m.write(os.path.join(output_folder, fname))


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

    # Read existing content (or empty string if file doesn’t exist)
    try:
        with open(output_file, 'r') as f:
            existing = f.read()
    except FileNotFoundError:
        existing = ''

    # Append only if the exact block isn’t already in the file
    if text_block not in existing:
        with open(output_file, 'a') as f:
            f.write(text_block)
    else:
        pass
