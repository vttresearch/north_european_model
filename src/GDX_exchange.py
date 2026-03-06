# src/GDX_exchange.py

from typing import Any, Dict, Optional, Sequence
import pandas as pd
import os
import glob
import gams.transfer as gt
from tqdm import tqdm



# ==============================================================================
# WRITE FUNCTIONS
# ==============================================================================

def write_df_to_gdx(
    df: Optional[pd.DataFrame],
    output_file: str,
    logger,
    parameter_name: str,
    parameter_dimensions: Sequence[str],
    ) -> None:
    """
    Write a DataFrame to a GDX file using gams.transfer.

    Parameters:
        df: DataFrame with columns matching parameter_dimensions + 'value'
        output_file: Path to output GDX file
        logger: IterationLogger instance for status messages
        parameter_name: GDX parameter name
        parameter_dimensions: dimension columns

    Returns:
        None: Writes content to output_file
    """
    if df is None or len(df) == 0:
        logger.log_status(f"Skipping writing GDX '{output_file}': No data to write", level="warn")
        return

    work = df[list(parameter_dimensions) + ["value"]]

    m = gt.Container()

    # Create Sets for each dimension
    dim_sets = {}
    for d in parameter_dimensions:
        dim_sets[d] = gt.Set(m, d, records=work[d].unique().tolist(), description=f"{d} domain")

    # Create Parameter
    domain = [dim_sets[d] for d in parameter_dimensions]
    param = gt.Parameter(m, parameter_name, domain, description=parameter_name)
    param.setRecords(work)

    # Write
    m.write(output_file)


def write_climate_window_GDX_files(
    annual_dfs: Dict[int, pd.DataFrame],
    output_folder: str,
    logger,
    bb_parameter: str,
    bb_parameter_dimensions: Sequence[str],
    gdx_name_suffix: str = "",
    ) -> None:
    """
    Write pre-split climate window DataFrames to per-year GDX files.

    Parameters:
        annual_dfs: Dict mapping year -> DataFrame with bb_parameter_dimensions + 'value',
                    as returned by _split_timeseries_to_climate_windows.
        output_folder: Directory where GDX files will be written
        logger: IterationLogger instance for status messages
        bb_parameter: GDX parameter name
        bb_parameter_dimensions: dimension columns
        gdx_name_suffix: suffix for output filename (optional)

    Returns:
        None: Writes content to
        - Single year: {bb_parameter}_{gdx_name_suffix}.gdx
        - Multiple years: {bb_parameter}_{gdx_name_suffix}_{year}.gdx
    """
    if not annual_dfs:
        logger.log_status(f"Skipping GDX writing for '{bb_parameter}_{gdx_name_suffix}': no data to write.", level="warn")
        return

    years = sorted(annual_dfs.keys())
    single_year = (len(years) == 1)
    fname_base = f"{bb_parameter}_{gdx_name_suffix}" if gdx_name_suffix else bb_parameter
    final_cols = list(bb_parameter_dimensions) + ["value"]

    # Build container and sets
    m = gt.Container()

    # Create a Set for each dimension.
    # For 't', use only one year's labels (all years share the same t-structure).
    # For other dimensions, collect unique values across all years.
    dim_sets = {}
    for d in bb_parameter_dimensions:
        if d == 't':
            unique_vals = annual_dfs[years[0]][d].unique()
        else:
            unique_vals = pd.concat([annual_dfs[yr][d] for yr in years]).unique()
        dim_sets[d] = gt.Set(m, d, records=unique_vals.tolist(), description=f"{d} domain")

    domains = [dim_sets[d] for d in bb_parameter_dimensions]
    param = gt.Parameter(m, bb_parameter, domains, description=bb_parameter)

    for yr in tqdm(years, desc="  Writing"):
        param.setRecords(annual_dfs[yr][final_cols])
        fname = f"{fname_base}_{yr}.gdx" if not single_year else f"{fname_base}.gdx"
        m.write(os.path.join(output_folder, fname))


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
