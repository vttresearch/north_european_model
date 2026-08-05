"""
Utility functions used exclusively by the timeseries pipeline.
"""

import os
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Sequence


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


def update_import_timeseries_inc(
    output_folder: str | Path,
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


def split_timeseries_to_climate_windows(
    df: pd.DataFrame,
    *,
    bb_parameter_dimensions: Sequence[str],
    bb_ts_start: str,
    bb_ts_length: int,
    valid_climate_years: List[int],
    ) -> Dict[int, pd.DataFrame]:
    """
    Split a multi-year timeseries DataFrame into per-year climate window chunks
    and assign Backbone t-labels.

    A climate window for year Y starts at {Y}-{bb_ts_start} 00:00 and spans
    bb_ts_length * 24 consecutive hours.  One output DataFrame is produced for
    every year in valid_climate_years for which the data covers a complete window.
    valid_climate_years is computed in run() from the config start/end years and
    bb_ts_start/bb_ts_length, so only years that can start a full window with the
    available data are included.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format input with columns from bb_parameter_dimensions (excluding 't'
        and 'f', which are both absent from the processor output), 'time' (datetime),
        and 'value'.
    bb_parameter_dimensions : sequence of str
        Backbone dimension names for the output (must include 't').
    bb_ts_start : str
        Window start within each year in "MM-DD" format (e.g. "01-01").
    bb_ts_length : int
        Window length in days.
    valid_climate_years : list of int
        Years for which to extract windows.

    Returns
    -------
    dict[int, pd.DataFrame]
        Keys are climate years; values are DataFrames with columns
        bb_parameter_dimensions + ['value'] and t-labels t000001..t{bb_ts_length*24}.
        If 'f' is in bb_parameter_dimensions, every row is assigned 'f00' (realized
        weather branch).
    """
    dims = list(bb_parameter_dimensions)

    if df["value"].dtype != np.float64:
        df = df.copy()
        df["value"] = df["value"].astype(np.float64)

    # Grouping dimensions exclude f and t
    group_dims = [c for c in dims if c not in {"f", "t"}]

    max_hours = bb_ts_length * 24
    t_labels = np.array(['t' + str(i).zfill(6) for i in range(1, max_hours + 1)])
    final_cols = dims + ["value"]
    out: Dict[int, pd.DataFrame] = {}

    # Pre-sort and pre-compute group ids once before the per-year loop.
    # A mask applied to a pre-sorted DataFrame yields an already-sorted subset,
    # and group_ids[mask] correctly identifies group boundaries in that subset.
    if group_dims:
        df = df.sort_values(group_dims + ["time"], kind="mergesort")
        group_ids = df.groupby(group_dims, observed=True, sort=False).ngroup().values
    time_np = df["time"].to_numpy()  # numpy datetime64 for fast per-year masking

    for yr in valid_climate_years:
        window_start = pd.Timestamp(f"{yr}-{bb_ts_start}")
        window_end   = window_start + pd.Timedelta(hours=max_hours - 1)
        mask = (time_np >= window_start.to_datetime64()) & (time_np <= window_end.to_datetime64())
        df_yr = df[mask].copy()

        # Skipping start years for which there is not enough data for the whole climate window
        if len(df_yr) == 0:
            continue

        if group_dims:
            # group_ids[mask] reuses the pre-computed group structure; no re-sort or re-groupby needed.
            group_changes = np.diff(group_ids[mask], prepend=-1) != 0

            # Fast row numbering within groups
            row_nums = np.arange(len(df_yr))
            row_nums -= np.repeat(
                row_nums[group_changes],
                np.diff(np.append(np.where(group_changes)[0], len(df_yr))),
            )
            df_yr['_row_num'] = row_nums
        else:
            df_yr['_row_num'] = np.arange(len(df_yr))

        row_nums_filtered = df_yr['_row_num'].values
        t_cat = pd.Categorical(t_labels[row_nums_filtered], categories=t_labels)
        df_yr['t'] = t_cat

        # Insert f00 as the realized-weather branch when f is a spec dimension.
        if "f" in dims:
            df_yr['f'] = 'f00'

        frame = df_yr[final_cols].reset_index(drop=True)
        out[int(yr)] = frame

    return out


def calculate_climatological_forecasts(
    input_df: pd.DataFrame,
    *,
    bb_parameter_dimensions,
    forecast_quantiles,
    bb_ts_start: str,
    bb_ts_length: int,
    round_precision: int = 0,
    ) -> pd.DataFrame:
    """
    Build stochastic forecast timeseries for Backbone from long-term climatological statistics.

    Backbone can represent uncertainty via multiple forecast branches (f-index). This
    function creates forecast data by computing quantiles of the input timeseries
    across all available climate years, so each branch reflects a different statistical
    outcome drawn from the long-term climatological data provided in the input dataframe.

    The caller controls how many forecasts to create and which quantile each represents via
    ``forecast_quantiles`` from the config file. Keys are Backbone f-labels and values are
    quantile probabilities (0..1). For example ``{'f01': 0.5, 'f02': 0.1, 'f03': 0.9}``
    creates three forecast branches where f01 is the median, f02 the lowest 10%,
    and f03 the highest 90% of values.

    These values are calculated and stored only once because they are the same for every climate window.

    Algorithm
    ---------
    For every combination of the non-f/t dimension columns (e.g. grid, node):

    1. Compute the requested quantiles across all years at each hour-of-year position
       (1..8760). Leap-day hours are excluded so that the statistics are always aligned
       on a common 8760-hour calendar.
    2. Map the resulting quantile values onto the output climate window
       (``bb_ts_start`` + ``bb_ts_length`` * 24 hours), using hour-of-year as the key.
       Windows longer than one calendar year are tiled correctly.
    3. Assign Backbone t-labels (t000001..) starting from the first hour of the climate
       window and f-labels from ``quantile_map``.

    Input requirements
    ------------------
    - Long-format DataFrame with the dimension columns from ``bb_parameter_dimensions``
      excluding 't' and 'f' which are absent from the intermediate format, plus ``time``
      (datetime) and ``value``. The columns are guarded beforehand.
    - Data must cover more than one climate year (checked before calling this function).

    Returns
    -------
    pd.DataFrame
        Single-year long-format DataFrame with columns ``bb_parameter_dimensions + ['value']``.
        The dataframe must contain the same t labels than timeseries produced in split_timeseries_to_climate_windows.
        The dataframe contains f column with defined quantile headers.
    """

    dim_cols = [col for col in bb_parameter_dimensions if col not in ("f", "t")]

    # ---- Create helper columns ----
    # Fast hour_of_year: avoid datetime arithmetic, use dayofyear + hour.
    time = input_df["time"]
    day_of_year = time.dt.dayofyear.to_numpy()
    hour = time.dt.hour.to_numpy()
    hour_of_year = (day_of_year - 1) * 24 + hour + 1

    # copy() first: this used to write 'hour_of_year' into the caller's frame,
    # so main_result silently gained a column that ProcessorRunner then carried
    # on using for domain collection and the annual summary.
    input_df = input_df.copy()
    input_df["hour_of_year"] = hour_of_year.astype(np.int32)

    # Only process hours up to 8760 (ignore extra hours from leap years)
    input_df = input_df[input_df["hour_of_year"] <= 8760]

    # ---- Quantile computation ----
    # Vectorized quantile computation:
    # Group by the additional dimensions and 'hour_of_year' then compute the quantiles.
    # Always computed over the full 8760-hour calendar year regardless of bb_ts_length.
    q_values = list(forecast_quantiles.values())

    df_quant = (
        input_df
        .groupby(dim_cols + ["hour_of_year"], observed=True)["value"]
        .quantile(q_values)
        # quantile with sequence -> MultiIndex with a 'quantile' level
        .rename_axis(index=dim_cols + ["hour_of_year", "quantile"])
        .reset_index()
    )
    # df_quant now has columns: dim_cols..., 'hour_of_year', 'quantile', 'value'

    # ---- Build window reference sequence ----
    # Generate the sequence of hour_of_year positions (1..8760) that correspond
    # to each hour in the output window.  Use a fixed non-leap reference year (2001)
    # so that the sequence wraps correctly across calendar year boundaries.
    ref_start = pd.Timestamp(f"2001-{bb_ts_start}")
    ref_times = pd.date_range(ref_start, periods=bb_ts_length * 24, freq='h')
    ref_hoy = ((ref_times.dayofyear - 1) * 24 + ref_times.hour + 1).astype(np.int32)
    # Safety clip (should not be needed for non-leap 2001/2002, but guards edge cases)
    ref_hoy = np.clip(ref_hoy, 1, 8760)

    # t-labels for the full window length
    t_labels_arr = np.array(['t' + str(i + 1).zfill(6) for i in range(bb_ts_length * 24)])

    # Window dimension DataFrame: one row per window position
    window_df = pd.DataFrame({
        "hour_of_year": ref_hoy,
        "t": t_labels_arr,
    })

    # ---- Build full grid (Cartesian product) ----
    # Unique combinations of all dimension columns
    unique_dims = input_df[dim_cols].drop_duplicates()

    # Quantiles as in quantile_map (order preserved)
    quantiles_df = pd.DataFrame({"quantile": q_values})

    # Cross join: unique_dims x window_df x quantiles_df
    full_grid = (
        unique_dims
        .merge(window_df, how="cross")
        .merge(quantiles_df, how="cross")
    )

    # Merge the computed quantile results using hour_of_year as the lookup key.
    # Multiple window positions can share the same hour_of_year (e.g. when tiling
    # the average year for bb_ts_length > 365).
    df_full = full_grid.merge(
        df_quant,
        on=dim_cols + ["hour_of_year", "quantile"],
        how="left",
    )

    # ---- Prepare final DataFrame, categorize ----
    # 't' is already set from window_df
    df_full["t"] = df_full["t"].astype("category")

    # Map quantile probability -> f label
    df_full["f"] = df_full["quantile"].map({v: k for k, v in forecast_quantiles.items()})
    df_full["f"] = df_full["f"].astype("category")

    # Missing quantile values are deliberately left as NaN.
    #
    # The merge above is a LEFT join onto the full (dims x window x quantile)
    # grid, so any window hour with no climatological data lands here as NaN --
    # for example when the source data does not span a whole calendar year but
    # the requested window does. Filling silently with 0 turned "no climatology
    # for this hour" into "a forecast of exactly zero", which is a real value the
    # optimiser will act on, with nothing in the log to say so.
    #
    # GDX_exchange.prepare_values_for_gdx performs the conversion instead, and
    # reports how many entries it converted. GAMS still receives 0; the
    # difference is that the run now says so.
    if round_precision is not None:
        df_full["value"] = df_full["value"].round(round_precision)

    # Reorder columns to match bb_parameter_dimensions plus 'value'
    df_final = df_full[bb_parameter_dimensions + ["value"]]

    return df_final
