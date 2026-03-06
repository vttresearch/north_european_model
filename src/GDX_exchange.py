# src/GDX_exchange.py

from typing import Dict, Optional, Sequence
import pandas as pd
import os
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


