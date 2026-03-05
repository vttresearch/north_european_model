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

def write_BB_gdx(
    df: Optional[pd.DataFrame],
    output_file: str,
    logger,
    **kwargs: Any
    ) -> None:
    """
    Write a DataFrame to a GDX file using gams.transfer.

    Parameters:
        df: DataFrame with columns matching bb_parameter_dimensions + 'value'
        output_file: Path to output GDX file
        logger: IterationLogger instance for status messages
        **kwargs: Must include:
            - bb_parameter: str -> GDX parameter name
            - bb_parameter_dimensions: Sequence[str] -> dimension columns (optional, inferred if missing)

    Returns:
        None: Writes content to output_file
    """
    if df is None or len(df) == 0:
        logger.log_status(f"Skipping writing GDX '{output_file}': No data to write", level="warn")
        return

    bb_parameter: Optional[str] = kwargs.get("bb_parameter")
    dims: Optional[Sequence[str]] = kwargs.get("bb_parameter_dimensions")

    if not bb_parameter:
        logger.log_status(f"Missing required kwarg 'bb_parameter' from '{output_file}'", level="warn")
        return

    # Infer dimensions if not provided
    if not dims or len(dims) == 0:
        dims = [c for c in df.columns if c != "value"]

    # Validate columns
    final_cols = list(dims) + ["value"]
    missing = [c for c in final_cols if c not in df.columns]
    if missing:
        logger.log_status(f"DataFrame missing required columns: {missing} for '{output_file}'", level="warn")
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
    annual_dfs: Dict[int, pd.DataFrame],
    output_folder: str,
    logger,
    **kwargs: Any
    ) -> None:
    """
    Write pre-split annual timeseries DataFrames to per-year GDX files.

    Parameters:
        annual_dfs: Dict mapping year -> DataFrame with bb_parameter_dimensions + 'value',
                    as returned by split_timeseries_to_annual_gdx_frames.
        output_folder: Directory where GDX files will be written
        logger: IterationLogger instance for status messages
        **kwargs: Must include:
            - bb_parameter: str -> GDX parameter name
            - bb_parameter_dimensions: Sequence[str] -> dimension columns
            - gdx_name_suffix: str (optional) -> suffix for output filename

    Returns:
        None: Writes content to
        - Single year: {bb_parameter}_{gdx_name_suffix}.gdx
        - Multiple years: {bb_parameter}_{gdx_name_suffix}_{year}.gdx
    """
    if not annual_dfs:
        logger.log_status(f"Skipping annual GDX writing for '{kwargs.get('bb_parameter', '?')}_{kwargs.get('gdx_name_suffix', '?')}': no data to write.", level="warn")
        return

    bb_parameter: Optional[str] = kwargs.get("bb_parameter")
    dims: Optional[Sequence[str]] = kwargs.get("bb_parameter_dimensions")
    gdx_name_suffix: Optional[str] = kwargs.get("gdx_name_suffix", "")

    if not bb_parameter:
        logger.log_status(f"Missing required kwarg 'bb_parameter' for annual GDX writing (gdx_name_suffix='{gdx_name_suffix}').", level="warn")
        return

    # If dims not provided, infer from first frame
    first_df = next(iter(annual_dfs.values()))
    if not dims or len(dims) == 0:
        dims = [c for c in first_df.columns if c != "value"]

    # Final columns of the written dataframe
    final_cols = list(dims) + ["value"]

    # Validate columns against first frame
    missing = [c for c in final_cols if c not in first_df.columns]
    if missing:
        logger.log_status(f"DataFrame missing required columns: {missing} for annual GDX writing.", level="warn")
        return

    years = sorted(annual_dfs.keys())
    single_year = (len(years) == 1)
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
        df_y = annual_dfs[yr][final_cols]
        param.setRecords(df_y)
        fname = f"{fname_base}_{yr}.gdx" if not single_year else f"{fname_base}.gdx"
        output_file = os.path.join(output_folder, fname)
        m.write(output_file)


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
