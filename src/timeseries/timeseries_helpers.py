"""
Utility functions used exclusively by the timeseries pipeline.
"""

import os
import glob
import warnings
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple


def select_declared_columns(frame, patterns: Sequence[str]) -> list:
    """The columns of `frame` a processor's ``requires_source_data`` asks for.

    Patterns are matched case-insensitively against the column names, because
    every reader of these frames already looks them up that way. A pattern
    ending in ``*`` matches by prefix: ``node_output*`` has to, since which of
    ``node_output1..5`` exist is decided at source-load time by
    ``build_unit_grid_and_node_columns`` from unittypedata's ``grid_output{i}``
    columns, so a fixed list would be wrong on a different workbook.

    Returns the columns in the frame's own order, spelled as the frame spells
    them. A pattern matching nothing contributes nothing rather than raising:
    the processor asking for it will fail on its own terms, saying which column
    it wanted, and that is a better message than one from here.
    """
    if frame is None:
        return []

    selected = []
    for column in frame.columns:
        lowered = str(column).lower()
        for pattern in patterns:
            wanted = str(pattern).lower()
            hit = lowered.startswith(wanted[:-1]) if wanted.endswith("*") else lowered == wanted
            if hit:
                selected.append(column)
                break
    return selected


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


def nodes_needing_flow(df_unitdata, flow: str) -> set | None:
    """Which nodes does the model attach units of this ``flow`` to?

    A capacity factor series is read by Backbone only through a unit: no unit of
    that flow on a node means the series is inert, and its absence means nothing.
    ``unitdata`` is where the question is answered -- ``flow`` arrives on it from
    ``unittypedata``, and the merge has already applied this run's scenario, year
    and country filtering, including a row removed by ``method: remove``.

    Output connections only. A unit's flow describes what it takes from the
    weather to produce, so a fuel node on the input side is not a node that needs
    a capacity factor.

    ``None`` means the question could not be asked -- no frame, no ``flow``
    column, no node column, or no flow to look for -- and is a different answer
    from the empty set, which says the model has no unit of this flow at all.
    Callers fail open on ``None`` the way `nodes_present_in_nodedata` does, and
    build nothing on the empty set.
    """
    if df_unitdata is None or getattr(df_unitdata, "empty", True) or not flow:
        return None

    columns = {str(c).lower(): c for c in df_unitdata.columns}
    flow_col = columns.get('flow')
    node_cols = [
        original for lowered, original in columns.items()
        if lowered.startswith('node_output') or lowered == 'node'
    ]
    if flow_col is None or not node_cols:
        return None

    wanted = str(flow).strip().lower()
    rows = df_unitdata[
        df_unitdata[flow_col].astype('string').str.strip().str.lower() == wanted
    ]
    return {
        str(node)
        for column in node_cols
        for node in rows[column]
        if not pd.isna(node) and str(node).strip()
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

    for yr, mask, row_nums in climate_window_rows(
        df["time"].to_numpy(), group_ids,
        bb_ts_start=bb_ts_start,
        bb_ts_length=bb_ts_length,
        valid_climate_years=valid_climate_years,
    ):
        df_yr = df[mask].copy()
        df_yr['t'] = pd.Categorical(t_labels[row_nums], categories=t_labels)

        # Insert f00 as the realized-weather branch when f is a spec dimension.
        if "f" in dims:
            df_yr['f'] = 'f00'

        out[yr] = df_yr[final_cols].reset_index(drop=True)

    return out


def climate_window_rows(
    time_np: np.ndarray,
    group_ids: np.ndarray,
    *,
    bb_ts_start: str,
    bb_ts_length: int,
    valid_climate_years: Sequence[int],
    ):
    """Yield ``(year, mask, row_nums)`` for every climate window the data has rows in.

    The one definition of a climate window, shared by the realized branch and the
    forecast branches so that the two cannot disagree about which hour a t-label
    names: year Y's window starts at {Y}-{bb_ts_start} 00:00 and spans
    bb_ts_length * 24 consecutive hours, and t-label n is the n-th row of a series
    inside it.

    ``time_np`` and ``group_ids`` come from :func:`order_timeseries_for_labelling`
    -- ordered by group then time, aligned positionally. ``mask`` selects the
    window's rows, and ``row_nums`` is each selected row's position within its own
    series, which is the t-label minus one. A year with no rows at all yields
    nothing: it cannot start a window.
    """
    max_hours = bb_ts_length * 24
    for yr in valid_climate_years:
        window_start = pd.Timestamp(f"{yr}-{bb_ts_start}")
        window_end   = window_start + pd.Timedelta(max_hours - 1, unit="h")
        mask = (time_np >= window_start.to_datetime64()) & (time_np <= window_end.to_datetime64())
        n_rows = int(mask.sum())
        if n_rows == 0:
            continue

        # With no grouping dimensions the ids are all zero, which marks a single
        # group and reduces the row numbering below to a plain arange -- so there
        # is no second code path to keep in agreement with this one.
        group_changes = np.diff(group_ids[mask], prepend=-1) != 0

        # Row number within each group, which is the t-label minus one.
        row_nums = np.arange(n_rows)
        row_nums -= np.repeat(
            row_nums[group_changes],
            np.diff(np.append(np.where(group_changes)[0], n_rows)),
        )
        yield int(yr), mask, row_nums


class ClimatologicalForecasts(NamedTuple):
    """What :func:`calculate_climatological_forecasts` returns."""

    #: Long format ``bb_parameter_dimensions + ['value']``: the branches.
    frame: pd.DataFrame
    #: One row per series: its grouping dimensions and, for every branch, the
    #: per-hour quantile its energy target came to.
    per_hour_quantiles: pd.DataFrame
    #: Per branch, the series whose energy target lies beyond what the lowest or
    #: the highest per-hour quantile carries, and which got 0 or 1 instead.
    unreachable: Dict[str, List[str]]


def calculate_climatological_forecasts(
    df: pd.DataFrame,
    *,
    bb_parameter_dimensions,
    energy_quantiles,
    bb_ts_start: str,
    bb_ts_length: int,
    valid_climate_years: Sequence[int],
    branch_days: Optional[Dict[str, int]] = None,
    round_precision: Optional[int] = 0,
    group_ids: Optional[np.ndarray] = None,
    ) -> ClimatologicalForecasts:
    """
    Build Backbone forecast branches from the realized climate windows.

    Each branch has the shape of the climate years and the energy its energy
    quantile asks for, series by series:

    - **Shape.** At every t a branch is a quantile of the realized values at that
      same t across the climate windows -- the windows that
      ``split_timeseries_to_climate_windows`` writes as f00, cut by the same
      :func:`climate_window_rows`. So a forecast t and a realized t always name
      the same hour, whatever the start date, the length or the leap years
      inside the window, and the only join is the wrap from the window's last
      hour to its first, where f00 has one too.
    - **Energy.** Which quantile is not given; it is solved for each series
      separately. ``energy_quantiles`` maps a branch to an energy quantile p, and
      the series' branch must carry, over the branch's own length,

          Q_p + (mean - Q_0.5) * (1 - |1 - 2p|)

      of the energies the climate windows carry over that length, where Q_p is
      their p-quantile. At 0.5 that is the mean exactly. Towards 0 and 1 the
      lift from the median to the mean fades out, so a low value is the years'
      own low: 0.05 is close to their one-in-twenty, and 0 their lowest. Short
      wind windows are where this matters -- a few very windy spells pull the
      mean well above the median, and a lift that did not fade would keep even
      0 far from the calmest spell on record. A series whose years are skewed
      gets a different per-hour quantile than one whose years are not, which is
      the point: a per-hour p0.5 of onshore wind carries about 70 % of its mean
      energy, and how far off it is differs by country.

    How the energy is measured
    --------------------------
    A branch of L days is used from every day of the window in turn, as the
    rolling solves move on, so its energy is measured over every L-day window,
    wrapping at the end as Backbone's circulation does, and summed. Every hour
    lies in exactly L of those windows, so the condition comes down to the
    branch's energy over the whole window:

        sum over hours of branch = mean total + (sum over start days of the
        L-day target minus the L-day mean) / L

    ``branch_days`` gives L per label; a label it does not name -- the central
    branch -- and a length of the window or more are measured over the whole
    window. The per-hour quantile whose energy meets the target is found by
    bisection: a branch's energy never falls as its quantile rises.

    Computed once, because the result is the same for every climate window.

    Weak spot: long windows
    -----------------------
    The windows are the samples, and they overlap once a window is longer than a
    year: in 1982-2016 a window of N years leaves 36 - N of them. That is 31 for
    five years and 16 for twenty. At 365*35+9 it is one, and every branch is the
    realized year itself -- perfect foresight labelled as a forecast. The build
    warns about any window longer than five years
    (``build_input_data._warn_about_long_window``); a limit raised past that needs
    another source of statistics here first.

    Missing values
    --------------
    NaN is skipped: an hour a window has no value for is left out of that hour's
    sample, and an hour no window has a value for comes out NaN. Filling it with 0
    would turn "no climatology" into "a forecast of exactly zero", which the
    optimiser acts on; ``GDX_exchange.prepare_values_for_gdx`` converts it and
    counts it instead. For the energy targets alone, a window's missing hour
    counts as the mean of the windows that have it, so a gap does not read as a
    dry year.

    Parameters
    ----------
    df : pd.DataFrame
        Long-format processor output: the dimensions of bb_parameter_dimensions
        other than 'f' and 't', plus 'time' and 'value'. Not modified.
    energy_quantiles : dict
        f-label -> energy quantile between 0 and 1.
    valid_climate_years : sequence of int
        The same years ``split_timeseries_to_climate_windows`` is given.
    branch_days : dict, optional
        f-label -> the branch's length in days, over which its energy is measured.
    group_ids : np.ndarray, optional
        From :func:`order_timeseries_for_labelling`; supplying it asserts that
        `df` is already ordered by it, as in ``split_timeseries_to_climate_windows``.

    Returns
    -------
    ClimatologicalForecasts
        The branches in long format ``bb_parameter_dimensions + ['value']``, one
        row per series, t-label and branch, with the t-labels the realized
        windows carry; the per-hour quantile every series came to; and the
        series whose target was out of reach.
    """
    dims = list(bb_parameter_dimensions)
    group_dims = [c for c in dims if c not in ("f", "t")]
    if group_ids is None:
        df, group_ids = order_timeseries_for_labelling(df, group_dims=group_dims)

    max_hours = bb_ts_length * 24
    n_series = int(group_ids.max()) + 1 if len(group_ids) else 0
    values = df["value"].to_numpy(dtype=np.float64)

    # windows x series x hours. NaN wherever a window has no row, so a window
    # that stops short, or a gap, drops out of the sample instead of counting as
    # a zero.
    windows = list(climate_window_rows(
        df["time"].to_numpy(), group_ids,
        bb_ts_start=bb_ts_start,
        bb_ts_length=bb_ts_length,
        valid_climate_years=valid_climate_years,
    ))
    samples = np.full((len(windows), n_series, max_hours), np.nan)
    for k, (_, mask, row_nums) in enumerate(windows):
        samples[k, group_ids[mask], row_nums] = values[mask]

    first_rows = np.flatnonzero(np.diff(group_ids, prepend=-1) != 0)
    series = {col: df[col].iloc[first_rows].to_numpy() for col in group_dims}
    names = ["|".join(str(series[col][i]) for col in group_dims) for i in range(n_series)]

    labels = list(energy_quantiles.keys())
    n_branches = len(labels)
    branch_days = branch_days or {}
    stats = np.full((n_branches, n_series, max_hours), np.nan)
    per_hour_quantiles = pd.DataFrame(series)
    unreachable: Dict[str, List[str]] = {}

    if len(windows):
        ordered = np.sort(samples, axis=0)
        n_valid = np.sum(~np.isnan(samples), axis=0)
        filled = _fill_with_hour_means(samples)
        for i, label in enumerate(labels):
            p = float(energy_quantiles[label])
            targets = _energy_targets(filled, p, branch_days.get(label))
            q, beyond = _solve_per_hour_quantiles(ordered, n_valid, targets, fallback=p)
            stats[i] = _quantile_at(ordered, n_valid, q)
            per_hour_quantiles[label] = q
            unreachable[label] = [names[s] for s in np.flatnonzero(beyond)]
    else:
        for label in labels:
            per_hour_quantiles[label] = np.nan
            unreachable[label] = []

    # One row per series, t and branch, in that order.
    series_idx = np.repeat(np.arange(n_series), max_hours * n_branches)
    t_idx = np.tile(np.repeat(np.arange(max_hours), n_branches), n_series)
    f_idx = np.tile(np.arange(n_branches), n_series * max_hours)

    t_labels = np.array(['t' + str(i).zfill(6) for i in range(1, max_hours + 1)])
    out = {col: series[col][series_idx] for col in group_dims}
    out["t"] = pd.Categorical.from_codes(t_idx, categories=t_labels)
    out["f"] = pd.Categorical.from_codes(f_idx, categories=labels)

    value = stats.transpose(1, 2, 0).reshape(-1)
    if round_precision is not None:
        value = np.round(value, round_precision)
    out["value"] = value

    return ClimatologicalForecasts(
        frame=pd.DataFrame(out)[dims + ["value"]],
        per_hour_quantiles=per_hour_quantiles,
        unreachable=unreachable,
    )


def _fill_with_hour_means(samples: np.ndarray) -> np.ndarray:
    """samples with each missing value replaced by its hour's mean across windows.

    An hour no window has a value for becomes 0, so it adds nothing to any energy.
    Used for the energy targets only, never written.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)  # all-NaN hours
        hour_mean = np.nanmean(samples, axis=0)
    hour_mean = np.nan_to_num(hour_mean, nan=0.0)
    return np.where(np.isnan(samples), hour_mean[None], samples)


def _energy_targets(filled: np.ndarray, energy_quantile: float, days: Optional[int]) -> np.ndarray:
    """Per series, the energy a branch must carry over the whole window.

    filled: (windows, series, hours) without NaN. ``days`` is the branch's length,
    or None for the whole window. See calculate_climatological_forecasts for the
    rule, and for why the L-day condition comes down to the whole window.
    """
    n_windows, n_series, n_hours = filled.shape
    totals = filled.sum(-1)                                        # windows, series
    mean_total = totals.mean(0)
    n_days = n_hours // 24
    if days is None or days >= n_days:
        return _energy_target(totals, energy_quantile)
    daily = filled.reshape(n_windows, n_series, n_days, 24).sum(-1)
    wrapped = np.concatenate([daily, daily[..., :days]], axis=-1)
    cumulative = np.concatenate(
        [np.zeros((n_windows, n_series, 1)), np.cumsum(wrapped, axis=-1)], axis=-1
    )
    starts = np.arange(n_days)
    sums = cumulative[..., starts + days] - cumulative[..., starts]  # windows, series, start day
    offset = _energy_target(sums, energy_quantile) - sums.mean(0)
    return mean_total + offset.sum(-1) / days


def _energy_target(sums: np.ndarray, energy_quantile: float) -> np.ndarray:
    """Q_p + (mean - Q_0.5) * (1 - |1 - 2p|) along the first axis, the climate windows.

    The mean at 0.5; the windows' own quantile at 0 and 1, with the lift from the
    median to the mean fading linearly between.
    """
    lift = (sums.mean(0) - np.quantile(sums, 0.5, axis=0)) * (1 - abs(1 - 2 * energy_quantile))
    return np.quantile(sums, energy_quantile, axis=0) + lift


def _solve_per_hour_quantiles(
    ordered: np.ndarray,
    n_valid: np.ndarray,
    targets: np.ndarray,
    *,
    fallback: float,
    iterations: int = 60,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per series, the per-hour quantile whose branch carries targets[s] over the window.

    A branch's energy never falls as its quantile rises, so bisection finds it.
    A series whose windows are all alike carries the same energy at every
    quantile and gets ``fallback``. Returns (q, beyond): beyond marks the series
    whose target lies outside what quantiles 0 to 1 carry; they get 0 or 1.
    """
    n_series = ordered.shape[1]

    def energy(q):
        return np.nansum(_quantile_at(ordered, n_valid, q), axis=-1)

    lowest = energy(np.zeros(n_series))
    highest = energy(np.ones(n_series))
    lo, hi = np.zeros(n_series), np.ones(n_series)
    for _ in range(iterations):
        mid = (lo + hi) / 2
        short = energy(mid) < targets
        lo = np.where(short, mid, lo)
        hi = np.where(short, hi, mid)
    q = (lo + hi) / 2

    scale = np.maximum(np.maximum(np.abs(lowest), np.abs(highest)), 1.0)
    alike = (highest - lowest) <= 1e-12 * scale
    too_low = ~alike & (targets < lowest - 1e-9 * scale)
    too_high = ~alike & (targets > highest + 1e-9 * scale)
    q = np.where(alike, fallback, np.where(too_low, 0.0, np.where(too_high, 1.0, q)))
    return q, too_low | too_high


def _at_positions(ordered: np.ndarray, n_valid: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Values at fractional positions of the sorted samples, as numpy interpolates.

    ordered is sorted along axis 0 with NaN last, so each column's valid values
    are its first n_valid. As numpy's _lerp: interpolate from whichever end is
    nearer. A column with no valid value comes out NaN.
    """
    below = np.floor(position).astype(np.int64)
    above = np.ceil(position).astype(np.int64)
    low = np.take_along_axis(ordered, below[None], axis=0)[0]
    high = np.take_along_axis(ordered, above[None], axis=0)[0]
    fraction = position - below
    step = high - low
    value = np.where(fraction >= 0.5, high - step * (1 - fraction), low + step * fraction)
    return np.where(n_valid > 0, value, np.nan)


def _quantile_at(ordered: np.ndarray, n_valid: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Per-hour quantile q[s] of every series s. ordered: (windows, series, hours)."""
    return _at_positions(ordered, n_valid, q[:, None] * np.maximum(n_valid - 1, 0))


def _nan_quantiles(samples: np.ndarray, quantiles: Sequence[float]) -> np.ndarray:
    """Quantiles along the first axis, skipping NaN, by linear interpolation.

    The same numbers as ``np.nanquantile(samples, quantiles, axis=0)`` -- and as
    pandas' default -- to floating-point precision, without its Python loop over
    every other axis position, which takes minutes on a multi-year parameter. One
    sort puts each column's NaN last, so its valid values are its first n and each
    quantile is two lookups and one interpolation. A column with no valid value at
    all comes out NaN.

    Returns an array of shape ``(len(quantiles),) + samples.shape[1:]``.
    """
    out = np.full((len(quantiles),) + samples.shape[1:], np.nan)
    if samples.shape[0] == 0:
        return out
    ordered = np.sort(samples, axis=0)
    n_valid = np.sum(~np.isnan(samples), axis=0)
    for i, q in enumerate(quantiles):
        out[i] = _at_positions(ordered, n_valid, q * np.maximum(n_valid - 1, 0))
    return out


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
