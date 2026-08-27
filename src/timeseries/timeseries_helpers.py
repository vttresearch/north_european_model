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

    A log line carrying a full list is a line nobody reads, and the warning next
    to it gets skipped too. The full lists belong in the documentation pages.

    Callers order `items` most-interesting-first, since that is what survives the
    truncation. Where the whole list is what a reader has to act on -- the nodes
    that were *not* built, rather than the ones that were -- name it in full and
    do not call this.
    """
    items = [str(item) for item in items]
    if len(items) <= limit:
        return ", ".join(items)
    return f"{', '.join(items[:limit])} and {len(items) - limit} more"


def nodes_present_in_nodedata(df_nodedata, *, suffixes: Sequence[str]) -> set:
    """Which nodes with these name endings does the model actually contain?

    ``nodedata`` is the statement of what exists. The source workflow has already
    applied its scenario, year and country filtering by the time a processor sees
    the frame, so a node absent from it is absent from the model -- not missing,
    not broken, and not something to report.

    Presence is the whole test. Whether a node's parameters are *usable* belongs
    to whichever processor needs them, which is what stops a blank
    ``upwardLimit`` from cascading into "this node has no inflow". See "Which
    nodes get built" in docs/hydro.md.

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


def update_import_timeseries_inc(
    output_folder: str | Path,
    file_suffix: Optional[str] = None,
    **kwargs: Any
    ) -> None:
    """
    Append a GAMS block to import_timeseries.inc that loads one parameter's GDX.

    Args:
        output_folder (str): where the GDX files are and where the .inc is written
        file_suffix (str, optional): suffix of a specific GDX. If None, the two
            standard patterns are searched for instead -- one file, or one per
            climate year.
        **kwargs: bb_parameter (str), the Backbone parameter to import, and
            gdx_name_suffix (str), the rest of the GDX filename.
    """
    bb_parameter = kwargs.get('bb_parameter')
    gdx_name_suffix = kwargs.get('gdx_name_suffix')

    if file_suffix is not None:
        filename = os.path.join(output_folder, f'{bb_parameter}_{gdx_name_suffix}_{file_suffix}.gdx')
        if os.path.exists(filename):
            matching_files = filename
        else:
            raise FileNotFoundError(f"{bb_parameter}_{gdx_name_suffix}_{file_suffix}.gdx not found in {output_folder}.")

    else:
        # One file for the whole run...
        file_a = os.path.join(output_folder, f'{bb_parameter}_{gdx_name_suffix}.gdx')
        if os.path.exists(file_a):
            matching_files = file_a
            file_suffix = None
        else:
            # ...or one per climate year, which GAMS picks by %climateYear%.
            pattern_b = os.path.join(output_folder, f'{bb_parameter}_{gdx_name_suffix}_[0-9][0-9][0-9][0-9].gdx')
            matching_files = glob.glob(pattern_b)
            if matching_files:
                file_suffix = "%climateYear%"

    if matching_files is None:
        raise FileNotFoundError(f"{bb_parameter}_{gdx_name_suffix}.gdx or {bb_parameter}_{gdx_name_suffix}_year.gdx not found in {output_folder}.")


    if file_suffix is None:
        gdx_name = f"{bb_parameter}_{gdx_name_suffix}.gdx"
    else:
        gdx_name = f"{bb_parameter}_{gdx_name_suffix}_{file_suffix}.gdx"

    text_block = "\n".join([
        f"$ifthen exist '%input_dir%/{gdx_name}'",
        f"    // If {gdx_name} exists, load input data",
        f"    $$gdxin '%input_dir%/{gdx_name}'",
        f"    $$loaddcm {bb_parameter}",
        "    $$gdxin",
        "$endIf",
        ""
    ]) + "\n"


    output_file = os.path.join(output_folder, 'import_timeseries.inc')

    try:
        with open(output_file, 'r') as f:
            existing = f.read()
    except FileNotFoundError:
        existing = ''

    # Appended, so the file accumulates one block per parameter across calls --
    # hence the check that this exact block is not in it already.
    if text_block not in existing:
        with open(output_file, 'a') as f:
            f.write(text_block)


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
    t{n+1}. Both are returned because every consumer needs them together, and
    the sort and the ``ngroup`` cost about a second each on a nine-million-row
    parameter.

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
    With no `group_dims` the frame is still sorted by time. Skipping the sort
    would hand out t-labels in whatever order the processor happened to return,
    which is not defensible for a label meaning "hour n of the window".

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
    not re-sort. Pure numpy over already-ordered data, so a nine-million-row
    parameter costs tens of milliseconds rather than the second and a half a
    ``duplicated()`` on the same frame does.

    Two independent things have to hold, and neither implies the other:

    - **within a group**, consecutive rows differ by exactly one `step`. That one
      comparison proves no repeats, no sub-`step` rows, no holes and monotonic
      time all at once: a repeat gives a difference of zero, a hole more than one.
    - **across groups**, every group starts and ends at the same timestamp.
      Groups can each be internally perfect and still cover different spans, and
      then they disagree about what a given t-label means.

    Why it matters: ``split_timeseries_to_climate_windows`` labels by row
    position, so a hole does not leave a hole in the labels -- it pulls every
    later hour of that group one label earlier, for the rest of the window. The
    numbers stay perfectly plausible and are merely attached to the wrong hours,
    and for a model whose value is largely the correlation between countries, a
    silent one-hour offset between two of them is not a small error.

    `step` is a parameter rather than a hard-coded hour because the checker has
    no reason to know the pipeline's business; the hourly assumption lives in
    ``split_timeseries_to_climate_windows``. At a one-hour step, 00:00 and 00:15
    land in the same bucket and are reported as a duplicate -- which is the
    intent, since the pipeline cannot label sub-hourly data.

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

    # First, because a NaT makes every comparison below meaningless: it reads as
    # an integer near the bottom of the int64 range, manufacturing a gap of about
    # 292 years next to it.
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
    case the window is short and every label in it is still correct. Once
    :func:`find_time_axis_defects` has passed, ``expected_rows`` is exact, so
    this is one ``len()`` per year.
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
    bb_ts_length * 24 consecutive hours. One output DataFrame per year in
    valid_climate_years that the data covers a complete window for.

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
        Years for which to extract windows. Computed in run() from the config,
        so only years that can start a full window are in it.
    group_ids : np.ndarray, optional
        Group ids from :func:`order_timeseries_for_labelling`. Supplying them
        asserts that `df` is **already** ordered by ``group_dims + ['time']``
        and that the ids align with it positionally -- both must come from the
        same call. Omit it and this function orders the frame itself. It exists
        so that ``ProcessorRunner``, which has to order the frame anyway to
        verify the time axis, does not pay for a second sort.

    Returns
    -------
    dict[int, pd.DataFrame]
        Keys are climate years; values are DataFrames with columns
        bb_parameter_dimensions + ['value'] and t-labels t000001..t{bb_ts_length*24}.
        If 'f' is in bb_parameter_dimensions, every row is assigned 'f00' (realized
        weather branch).
    """
    dims = list(bb_parameter_dimensions)
    group_dims = [c for c in dims if c not in {"f", "t"}]

    max_hours = bb_ts_length * 24
    t_labels = np.array(['t' + str(i).zfill(6) for i in range(1, max_hours + 1)])
    final_cols = dims + ["value"]
    out: Dict[int, pd.DataFrame] = {}

    # Sort and group ids come once, before the per-year loop: a mask applied to a
    # pre-sorted frame yields an already-sorted subset, and group_ids[mask] still
    # identifies the group boundaries in it.
    if group_ids is None:
        df, group_ids = order_timeseries_for_labelling(df, group_dims=group_dims)
    time_np = df["time"].to_numpy()

    for yr in valid_climate_years:
        window_start = pd.Timestamp(f"{yr}-{bb_ts_start}")
        window_end   = window_start + pd.Timedelta(max_hours - 1, unit="h")
        mask = (time_np >= window_start.to_datetime64()) & (time_np <= window_end.to_datetime64())
        df_yr = df[mask].copy()

        # A year with no data at all cannot start a window.
        if len(df_yr) == 0:
            continue

        # With no grouping dimensions the ids are all zero, which marks a single
        # group and reduces the row numbering below to a plain arange -- so there
        # is no second code path to keep in agreement with this one.
        group_changes = np.diff(group_ids[mask], prepend=-1) != 0

        # Row number within each group, which is the t-label minus one.
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
    Build Backbone forecast branches from long-term climatological statistics.

    Each f-branch is a quantile of the input timeseries taken across all climate
    years, so it reflects a different statistical outcome rather than a different
    forecast run. ``forecast_quantiles`` decides how many and which: keys are
    f-labels, values are probabilities, so ``{'f01': 0.5, 'f02': 0.1}`` gives a
    median branch and a lowest-10% one.

    Computed once, because the result is the same for every climate window.

    Algorithm
    ---------
    Per combination of the non-f/t dimension columns:

    1. Quantiles across all years at each hour-of-year position (1..8760).
       Leap-day hours are excluded so the statistics align on one calendar.
    2. Map those onto the output window by hour-of-year, tiling correctly when
       the window is longer than a calendar year.
    3. Assign t-labels from the first hour of the window and f-labels from
       ``forecast_quantiles``.

    Input is the same long format the processors return, plus the guarantee --
    checked by the caller -- that it covers more than one climate year.

    Returns
    -------
    pd.DataFrame
        Single-year long format ``bb_parameter_dimensions + ['value']``, carrying
        the same t-labels as ``split_timeseries_to_climate_windows`` produces.
    """

    dim_cols = [col for col in bb_parameter_dimensions if col not in ("f", "t")]

    # hour_of_year from dayofyear + hour, avoiding datetime arithmetic.
    time = input_df["time"]
    day_of_year = time.dt.dayofyear.to_numpy()
    hour = time.dt.hour.to_numpy()
    hour_of_year = (day_of_year - 1) * 24 + hour + 1

    # copy() first: writing 'hour_of_year' into the caller's frame would leave
    # main_result with an extra column that ProcessorRunner goes on to use for
    # domain collection and the annual summary.
    input_df = input_df.copy()
    input_df["hour_of_year"] = hour_of_year.astype(np.int32)

    # Leap-day hours dropped, so every year contributes the same 8760 positions.
    input_df = input_df[input_df["hour_of_year"] <= 8760]

    # Over the full calendar year regardless of bb_ts_length: the window is cut
    # from these statistics afterwards, not built into them.
    q_values = list(forecast_quantiles.values())

    df_quant = (
        input_df
        .groupby(dim_cols + ["hour_of_year"], observed=True)["value"]
        .quantile(q_values)
        # A sequence of quantiles gives a MultiIndex with a 'quantile' level.
        .rename_axis(index=dim_cols + ["hour_of_year", "quantile"])
        .reset_index()
    )

    # Which hour_of_year each position in the output window corresponds to. The
    # reference year is a fixed non-leap one, so the sequence wraps correctly
    # across a calendar year boundary.
    ref_start = pd.Timestamp(f"2001-{bb_ts_start}")
    ref_times = pd.date_range(ref_start, periods=bb_ts_length * 24, freq='h')
    ref_hoy = ((ref_times.dayofyear - 1) * 24 + ref_times.hour + 1).astype(np.int32)
    ref_hoy = np.clip(ref_hoy, 1, 8760)

    t_labels_arr = np.array(['t' + str(i + 1).zfill(6) for i in range(bb_ts_length * 24)])

    # One row per window position.
    window_df = pd.DataFrame({
        "hour_of_year": ref_hoy,
        "t": t_labels_arr,
    })

    unique_dims = input_df[dim_cols].drop_duplicates()
    quantiles_df = pd.DataFrame({"quantile": q_values})

    # Every dimension combination x window position x quantile.
    full_grid = (
        unique_dims
        .merge(window_df, how="cross")
        .merge(quantiles_df, how="cross")
    )

    # hour_of_year is the lookup key, and several window positions can share one
    # when bb_ts_length > 365 tiles the average year.
    df_full = full_grid.merge(
        df_quant,
        on=dim_cols + ["hour_of_year", "quantile"],
        how="left",
    )

    df_full["t"] = df_full["t"].astype("category")
    df_full["f"] = df_full["quantile"].map({v: k for k, v in forecast_quantiles.items()})
    df_full["f"] = df_full["f"].astype("category")

    # Missing quantile values are deliberately left as NaN. The merge above is a
    # LEFT join onto the full grid, so a window hour with no climatology behind
    # it lands here empty -- and filling it with 0 would turn "no climatology"
    # into "a forecast of exactly zero", which the optimiser acts on.
    # GDX_exchange.prepare_values_for_gdx does the conversion instead, and counts
    # it: GAMS still receives 0, but the run says so.
    if round_precision is not None:
        df_full["value"] = df_full["value"].round(round_precision)

    return df_full[bb_parameter_dimensions + ["value"]]


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
    168 steps, and whether it gets bridged depends on an interpolation limit.
    Fill first and upsample second, and the hourly pass never has to reach across
    a gap it cannot close.

    Only single-slot gaps are filled. Anything longer is left alone and counted,
    because bridging it is invention rather than repair and the person adopting a
    data source should be the one deciding.

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
        as a *stretch* -- a season during which the reservoir may empty. One zero
        wedged between two non-zero neighbours is a dropped value wearing a
        plausible costume, and is treated as a gap.

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
        # towards, and limit_area='inside' refuses to guess. It is still a
        # single-slot gap, so it is still repaired -- by carrying the previous
        # value forward, which is what one-step persistence amounts to.
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
