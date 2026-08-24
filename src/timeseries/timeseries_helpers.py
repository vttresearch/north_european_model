"""
Utility functions used exclusively by the timeseries pipeline.
"""

import os
import glob
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Sequence, Tuple


#: How many items a log line names before it starts counting instead.
LOG_LIST_LIMIT = 3


def summarise(items, limit: int = LOG_LIST_LIMIT) -> str:
    """Render a list for a log line: the first few, then how many are left.

    A build prints a handful of report lines per processor and a person has to
    be able to read them at a glance. The full lists -- every country code,
    every zone choice, every node that could not be built -- belong in the
    documentation pages, because a log line carrying one of them in full is a
    line nobody reads and the warning next to it gets skipped too.

    Callers order `items` most-interesting-first, since that is what survives
    the truncation. Where the whole list is what a reader has to act on -- the
    nodes that were *not* built, rather than the ones that were -- name it in
    full and do not call this.
    """
    items = [str(item) for item in items]
    if len(items) <= limit:
        return ", ".join(items)
    return f"{', '.join(items[:limit])} and {len(items) - limit} more"


def nodes_present_in_nodedata(df_nodedata, *, suffixes: Sequence[str]) -> set:
    """Which nodes with these name endings does the model actually contain?

    ``nodedata`` is the statement of what exists. By the time a processor sees the
    frame the source workflow has already applied its scenario, year and country
    filtering, so a node absent from it is absent from the model -- not missing,
    not broken, and not something to report. A country dropped from the run takes
    its hydro nodes with it, and the processor should fall silent about them
    rather than describe them as gaps.

    Presence is the whole test. Whether a node's parameters are *usable* is a
    separate question, and it belongs to whichever processor needs them: inflow
    describes a node and needs nothing else, while a fill limit is a fraction of a
    reservoir size and is meaningless without one. Keeping the two apart is what
    stops an ``upwardLimit`` that is blank or zero -- which may simply be an
    oversight in the workbook -- from cascading into "this node has no inflow".

    Returns an empty set when the frame or its ``node`` column is missing, which
    callers must read as "cannot tell" rather than "nothing exists".
    """
    if df_nodedata is None or getattr(df_nodedata, "empty", True):
        return set()
    columns = {str(c).lower(): c for c in df_nodedata.columns}
    node_col = columns.get('node')
    if node_col is None:
        return set()

    wanted = tuple(suffixes)
    return {
        str(node)
        for node in df_nodedata[node_col]
        if not pd.isna(node) and str(node).endswith(wanted)
    }


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


def order_timeseries_for_labelling(
    df: pd.DataFrame,
    *,
    group_dims: Sequence[str],
    time_col: str = "time",
    ) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Sort a long-format timeseries into t-label order and return its group ids.

    t-labels are assigned by row position within each group, so this ordering
    *is* the labelling: sort by group then time, and row n of a group becomes
    t{n+1}. Both the ordering and the group ids are returned because every
    consumer needs them together and recomputing either is expensive -- the sort
    and the ``ngroup`` cost about a second each on a nine-million-row parameter.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format input with the grouping dimensions and `time_col`.
    group_dims : sequence of str
        The dimensions that define a series -- the spec dimensions minus
        't' and 'f'. May be empty.
    time_col : str
        Name of the datetime column.

    Returns
    -------
    (pd.DataFrame, np.ndarray)
        The frame sorted by ``group_dims + [time_col]``, and group ids aligned
        to it **positionally**. The frame keeps its original index; everything
        downstream indexes by position, so resetting it would only cost a copy.

    Notes
    -----
    With no `group_dims` the frame is still sorted by time. That case used to
    skip sorting entirely and then hand out t-labels in whatever order the
    processor happened to return, which is not a defensible thing to do with a
    label that means "hour n of the window".

    ``ngroup`` returns -1 for rows whose grouping key is missing, and
    ``sort_values`` puts those rows last, so they arrive as one trailing
    pseudo-group. Callers should reject missing dimension values before getting
    here -- ``ProcessorRunner`` does -- because "blank is not a GAMS set
    element" is a better message than anything derivable from the time axis.
    """
    group_dims = list(group_dims)

    if "value" in df.columns and df["value"].dtype != np.float64:
        df = df.copy()
        df["value"] = df["value"].astype(np.float64)

    if group_dims:
        df = df.sort_values(group_dims + [time_col], kind="mergesort")
        group_ids = df.groupby(group_dims, observed=True, sort=False).ngroup().to_numpy()
    else:
        df = df.sort_values([time_col], kind="mergesort")
        group_ids = np.zeros(len(df), dtype=np.int64)

    return df, group_ids


@dataclass(frozen=True)
class TimeAxisReport:
    """What :func:`find_time_axis_defects` found. See ``ok`` for the verdict."""

    n_rows: int
    n_groups: int
    #: Rows whose timestamp is NaT, or was not convertible to one.
    n_missing_timestamps: int
    #: Steps of zero or less within a group: a repeated timestamp, or two
    #: timestamps that fall in the same step-sized bucket (sub-hourly data).
    n_duplicate_or_finer_than_step: int
    #: Steps of more than one within a group: a hole.
    n_gaps: int
    #: Groups disagree about which span they cover, even if each is internally
    #: complete.
    ragged_extent: bool
    first_defect_index: Optional[int] = None
    first_defect_time: Optional[pd.Timestamp] = None
    first_time: Optional[pd.Timestamp] = None
    last_time: Optional[pd.Timestamp] = None
    #: (earliest, latest) first timestamp across groups; equal unless ragged.
    group_first_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None
    #: (earliest, latest) last timestamp across groups; equal unless ragged.
    group_last_range: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None

    @property
    def ok(self) -> bool:
        return (
            self.n_missing_timestamps == 0
            and self.n_duplicate_or_finer_than_step == 0
            and self.n_gaps == 0
            and not self.ragged_extent
        )


def find_time_axis_defects(
    sorted_df: pd.DataFrame,
    group_ids: np.ndarray,
    *,
    time_col: str = "time",
    step: pd.Timedelta = pd.Timedelta(1, unit="h"),
    ) -> TimeAxisReport:
    """
    Check that every group is one complete grid on `step`, and the same grid.

    Requires `sorted_df` and `group_ids` from a single call to
    :func:`order_timeseries_for_labelling`; it reads them positionally and does
    not re-sort. Pure numpy over already-ordered data: no groupby, no per-group
    Python loop, so a nine-million-row parameter costs tens of milliseconds
    rather than the second and a half a ``duplicated()`` on the same frame does.

    Two independent things have to hold, and neither implies the other:

    - **within a group**, consecutive rows differ by exactly one `step`. That
      one comparison proves no repeats, no sub-`step` rows, no holes, and
      monotonic time all at once -- a repeat gives a difference of zero, a
      hole gives more than one.
    - **across groups**, every group starts and ends at the same timestamp.
      Groups can each be internally perfect and still cover different spans,
      and then they disagree about what a given t-label means.

    Why any of it matters: ``split_timeseries_to_climate_windows`` labels by row
    position. A hole does not leave a hole in the labels -- it pulls every later
    hour of that group one label earlier, for the rest of the window. Nothing
    downstream can detect that, because the numbers are all perfectly plausible
    and merely attached to the wrong hours. For a model whose value is largely
    the correlation between countries, a silent one-hour offset between two of
    them is not a small error.

    `step` is a parameter rather than a hard-coded hour because the checker has
    no reason to know the pipeline's business. The hourly assumption lives in
    ``split_timeseries_to_climate_windows``, whose window is ``bb_ts_length * 24``
    labels. At a one-hour step, 00:00 and 00:15 land in the same bucket and are
    reported as a duplicate -- which is the intent: the pipeline cannot label
    sub-hourly data, so it must not accept it silently.

    Returns
    -------
    TimeAxisReport
        Counts and locations. ``report.ok`` is the verdict; the rest exists so
        the caller can say *what* was wrong and *where*.
    """
    n_rows = len(sorted_df)
    if n_rows == 0:
        return TimeAxisReport(0, 0, 0, 0, 0, False)

    gid = np.asarray(group_ids)
    newg = np.empty(n_rows, dtype=bool)
    newg[0] = True
    np.not_equal(gid[1:], gid[:-1], out=newg[1:])
    starts = np.flatnonzero(newg)

    col = sorted_df[time_col]
    if not pd.api.types.is_datetime64_any_dtype(col):
        col = pd.to_datetime(col, errors="coerce")
    times = col.to_numpy(dtype="datetime64[ns]")

    # First, because a NaT makes every comparison below meaningless: it would
    # read as an integer near the bottom of the int64 range and manufacture a
    # gap of about 292 years next to it.
    nat = np.isnat(times)
    if nat.any():
        return TimeAxisReport(
            n_rows=n_rows,
            n_groups=starts.size,
            n_missing_timestamps=int(nat.sum()),
            n_duplicate_or_finer_than_step=0,
            n_gaps=0,
            ragged_extent=False,
            first_defect_index=int(np.flatnonzero(nat)[0]),
        )

    # Explicit floor-divide on int64 nanoseconds rather than
    # `.astype("datetime64[h]")`: numpy's unit-downcast rounding for pre-epoch
    # values is not something a t-label should depend on, and pinning [ns] in
    # to_numpy above stops a different pandas resolution changing the divisor.
    ticks = times.view("int64") // step.value

    diff = np.empty(n_rows, dtype=np.int64)
    diff[0] = 1
    np.subtract(ticks[1:], ticks[:-1], out=diff[1:])
    bad_idx = np.flatnonzero(~(newg | (diff == 1)))
    bad_steps = diff[bad_idx]

    ends = np.append(starts[1:], n_rows) - 1
    group_firsts, group_lasts = times[starts], times[ends]
    first_range = (pd.Timestamp(group_firsts.min()), pd.Timestamp(group_firsts.max()))
    last_range = (pd.Timestamp(group_lasts.min()), pd.Timestamp(group_lasts.max()))

    return TimeAxisReport(
        n_rows=n_rows,
        n_groups=starts.size,
        n_missing_timestamps=0,
        n_duplicate_or_finer_than_step=int((bad_steps <= 0).sum()),
        n_gaps=int((bad_steps > 1).sum()),
        ragged_extent=bool(first_range[0] != first_range[1] or last_range[0] != last_range[1]),
        first_defect_index=int(bad_idx[0]) if bad_idx.size else None,
        first_defect_time=pd.Timestamp(times[bad_idx[0]]) if bad_idx.size else None,
        first_time=first_range[0],
        last_time=last_range[1],
        group_first_range=first_range,
        group_last_range=last_range,
    )


def find_incomplete_climate_windows(
    annual_dfs: Dict[int, pd.DataFrame],
    *,
    expected_rows: int,
    ) -> Dict[int, int]:
    """
    Years whose window did not come out the expected size -> the size it did.

    The one hazard a whole-frame time-axis check cannot see: data can be a
    flawless grid and still not reach the end of the requested window, in which
    case the window is simply short and every label in it is still correct.
    Once :func:`find_time_axis_defects` has passed, ``expected_rows`` is exact
    (``bb_ts_length * 24 * n_groups``), so this is one ``len()`` per year.
    """
    if expected_rows <= 0:
        return {}
    return {
        year: len(frame)
        for year, frame in annual_dfs.items()
        if len(frame) != expected_rows
    }


def split_timeseries_to_climate_windows(
    df: pd.DataFrame,
    *,
    bb_parameter_dimensions: Sequence[str],
    bb_ts_start: str,
    bb_ts_length: int,
    valid_climate_years: List[int],
    group_ids: Optional[np.ndarray] = None,
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
    group_ids : np.ndarray, optional
        Group ids from :func:`order_timeseries_for_labelling`. Supplying them
        asserts that `df` is **already** ordered by ``group_dims + ['time']``
        and that the ids align with it positionally -- both must come from the
        same call. Omit it and this function does the ordering itself, which is
        what every caller did before the check was added; it exists so
        ``ProcessorRunner``, which has to order the frame anyway to verify the
        time axis, does not pay for a second sort.

    Returns
    -------
    dict[int, pd.DataFrame]
        Keys are climate years; values are DataFrames with columns
        bb_parameter_dimensions + ['value'] and t-labels t000001..t{bb_ts_length*24}.
        If 'f' is in bb_parameter_dimensions, every row is assigned 'f00' (realized
        weather branch).
    """
    dims = list(bb_parameter_dimensions)

    # Grouping dimensions exclude f and t
    group_dims = [c for c in dims if c not in {"f", "t"}]

    max_hours = bb_ts_length * 24
    t_labels = np.array(['t' + str(i).zfill(6) for i in range(1, max_hours + 1)])
    final_cols = dims + ["value"]
    out: Dict[int, pd.DataFrame] = {}

    # Sort and group ids come once, before the per-year loop. A mask applied to
    # a pre-sorted DataFrame yields an already-sorted subset, and group_ids[mask]
    # correctly identifies group boundaries in that subset.
    if group_ids is None:
        df, group_ids = order_timeseries_for_labelling(df, group_dims=group_dims)
    time_np = df["time"].to_numpy()  # numpy datetime64 for fast per-year masking

    for yr in valid_climate_years:
        window_start = pd.Timestamp(f"{yr}-{bb_ts_start}")
        window_end   = window_start + pd.Timedelta(max_hours - 1, unit="h")
        mask = (time_np >= window_start.to_datetime64()) & (time_np <= window_end.to_datetime64())
        df_yr = df[mask].copy()

        # Skipping start years for which there is not enough data for the whole climate window
        if len(df_yr) == 0:
            continue

        # group_ids[mask] reuses the pre-computed group structure; no re-sort or re-groupby needed.
        # With no grouping dimensions the ids are all zero, which marks a single
        # group and reduces the row numbering below to a plain arange -- so there
        # is no second code path to keep in agreement with this one.
        group_changes = np.diff(group_ids[mask], prepend=-1) != 0

        # Fast row numbering within groups
        row_nums = np.arange(len(df_yr))
        row_nums -= np.repeat(
            row_nums[group_changes],
            np.diff(np.append(np.where(group_changes)[0], len(df_yr))),
        )
        df_yr['_row_num'] = row_nums

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


@dataclass(frozen=True)
class GridGapReport:
    """What :func:`complete_native_grid` found and what it did about it.

    ``ok`` is the verdict a processor should act on: the grid is whole, and every
    value in it is one the model can use.
    """

    label: str
    #: Slots on the standard grid, from the first real value onwards.
    n_slots: int
    #: Slots that held no usable value before filling.
    n_missing: int
    #: Single-slot gaps, filled here without ceremony.
    n_autofilled: int
    #: Slots left empty because their run was longer than one and this function
    #: does not invent that much. Whoever owns the processor decides.
    n_left: int
    #: Length of the longest untouched run, in slots.
    longest_run_left: int
    first_left: Optional[pd.Timestamp] = None
    #: Mean of the completed series times 8760, for judging whether a gap is
    #: worth anyone's attention. A missing week means one thing in a 20 TWh
    #: catchment and another in a 0.2 TWh one.
    twh_per_year: float = 0.0

    @property
    def ok(self) -> bool:
        return self.n_left == 0


def complete_native_grid(
    series: pd.Series,
    standard_index: pd.DatetimeIndex,
    *,
    label: str,
    zero_is_missing: bool = True,
    isolated_zero_is_missing: bool = True,
) -> Tuple[pd.Series, GridGapReport]:
    """Make a weekly or daily series whole *before* it is cast to hourly.

    The order is the point. At native resolution a missing week is one step from
    its neighbours and interpolates cleanly; scattered onto an hourly index it is
    168 steps, and whether it gets bridged depends on an interpolation limit. Fill
    first and upsample second, and the hourly pass never has to reach across a gap
    it cannot close -- so it cannot leave the holes that reach GAMS as zeros.

    Only single-slot gaps are filled. Anything longer is left alone and counted:
    two consecutive missing weeks is no longer a repair, it is an invention, and
    the person adopting a new data source should decide what it ought to be rather
    than discover later that this function decided for them.

    Parameters
    ----------
    series : pd.Series
        Values on a DatetimeIndex at the native step. Need not be complete.
    standard_index : pd.DatetimeIndex
        Every slot the series is supposed to have.
    label : str
        Column or node name, used in the report.
    zero_is_missing : bool
        Whether ``0`` counts as absent. True for inflow and generation, where a
        real zero does not occur and a recorded one is a gap. False where zero is
        a legitimate value -- ``downwardLimit`` of zero means the reservoir is
        allowed to empty, which is an ordinary thing for a series to say.
    isolated_zero_is_missing : bool
        Applies only when ``zero_is_missing`` is False. A legitimate zero arrives
        as a *stretch*: a season during which the reservoir may empty. One zero
        wedged between two non-zero neighbours is not that, it is a dropped value
        wearing a plausible costume, and it is treated as a gap. SE04's weekly
        pattern is the case in point -- a two-week run at weeks 46-47 that is real,
        and a lone zero at week 15 between 0.001 and 0.015 that is not.

    Returns
    -------
    (pd.Series, GridGapReport)
        The completed series, and what had to be done to it.
    """
    empty_report = GridGapReport(label=label, n_slots=0, n_missing=0,
                                 n_autofilled=0, n_left=0, longest_run_left=0)
    if series is None or series.empty:
        return series, empty_report

    combined = series.reindex(series.index.union(standard_index))
    is_zero = combined.notna() & (combined == 0)
    usable = combined.notna()
    if zero_is_missing:
        usable &= ~is_zero
    elif isolated_zero_is_missing and is_zero.any():
        zero_run = (is_zero != is_zero.shift()).cumsum()
        zero_len = is_zero.groupby(zero_run).transform('size').where(is_zero, 0)
        usable &= ~(is_zero & (zero_len == 1))
    if not usable.any():
        return combined.iloc[0:0], empty_report

    # Slots before the first real value are not gaps -- there is nothing to
    # interpolate from, and the hourly pass reaches back far enough to cover the
    # few days before the first one.
    first_real = combined.index[usable][0]
    combined = combined.loc[first_real:]
    usable = usable.loc[first_real:]

    marked = combined.where(usable)
    missing = ~usable

    # Run lengths, so a lone gap can be told from a stretch of them.
    run_id = (missing != missing.shift()).cumsum()
    run_len = missing.groupby(run_id).transform('size').where(missing, 0)

    singles = missing & (run_len == 1)
    filled = marked.copy()
    if singles.any():
        interpolated = marked.interpolate(method='time', limit_area='inside')
        filled[singles] = interpolated[singles]

        # A single slot at the very end has nothing after it to interpolate
        # towards, and limit_area='inside' deliberately refuses to guess. It is
        # still a single-slot gap though, so it is still repaired -- by carrying
        # the previous value forward, which is all a one-step persistence
        # assumption amounts to. Escalating this would be noise.
        trailing = singles & filled.isna()
        if trailing.any():
            filled[trailing] = marked.ffill()[trailing]

    left = filled.isna()
    longest_left = int(run_len[left].max()) if left.any() else 0
    twh = float(filled.dropna().mean()) * 8760 / 1e6 if filled.notna().any() else 0.0

    report = GridGapReport(
        label=label,
        n_slots=int(len(combined)),
        n_missing=int(missing.sum()),
        n_autofilled=int(singles.sum()),
        n_left=int(left.sum()),
        longest_run_left=longest_left,
        first_left=(filled.index[left][0] if left.any() else None),
        twh_per_year=twh,
    )
    return filled, report
