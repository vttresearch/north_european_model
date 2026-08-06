from pathlib import Path
from typing import Dict, Optional, Sequence
import numpy as np
import pandas as pd
import os
import gams.transfer as gt
from tqdm import tqdm


# --- Which GAMS installation gams.transfer binds to -------------------------
#
# A bare ``gt.Container()`` binds to whatever a machine-global setting names --
# on Windows the registry key ``HKCU\Software\Classes\gams.location``, which is
# simply whichever GAMS installer ran last. On a machine with several GAMS
# versions that is usually not the one ``gamsapi`` is pinned to, and every
# container then prints:
#
#     UserWarning: The GAMS version (53.5.0) differs from the API version (47.4.1)
#
# Reads and writes still succeed, so it scrolls past -- but the binding is
# arbitrary. The parent Backbone repo resolves this explicitly rather than
# silencing the warning (see its scripts/gams_api.py and
# .claude/skills/backbone-quickstart/references/gams-transfer.md); the same
# ``BB_GAMS_API`` variable is honoured here so one setting covers both repos.

#: gams.transfer requires GAMS 45+ to read/write through the API.
_API_MIN_GAMS_VERSION = 45


def _installed_gamsapi_major() -> Optional[int]:
    """Major version of the installed ``gamsapi`` package, or None if unknown."""
    try:
        from importlib.metadata import version

        return int(version("gamsapi").split(".")[0])
    except Exception:
        return None


def resolve_gams_system_directory() -> Optional[str]:
    """Return the GAMS system directory gams.transfer should bind to.

    Resolution order, mirroring the parent repo:

    1. ``BB_GAMS_API`` -- an explicit pin, e.g. ``C:\\GAMS\\47``.
    2. The GAMS install whose major version matches the installed ``gamsapi``.
       The GAMS installer's own layout (``C:\\GAMS\\<major>``) makes this a
       direct lookup, so most machines need no configuration at all.
    3. ``None`` -- let gams.transfer auto-discover, the previous behaviour.

    Returning None rather than raising is deliberate: an unresolvable binding is
    a noisy warning, not a reason to fail a build.
    """
    explicit = os.environ.get("BB_GAMS_API")
    if explicit:
        return explicit

    major = _installed_gamsapi_major()
    if major is None or major < _API_MIN_GAMS_VERSION:
        return None

    candidate = Path(f"C:/GAMS/{major}")
    if candidate.is_dir():
        return str(candidate)
    return None


def new_container(load_from: Optional[str] = None) -> "gt.Container":
    """Create a gams.transfer Container bound to a known GAMS installation.

    Always use this rather than ``gt.Container(...)`` directly, so the binding is
    the one we asked for rather than whatever the machine-global setting names.
    """
    kwargs = {}
    system_directory = resolve_gams_system_directory()
    if system_directory:
        kwargs["system_directory"] = system_directory
    if load_from is not None:
        return gt.Container(load_from, **kwargs)
    return gt.Container(**kwargs)


def prepare_values_for_gdx(
    df: pd.DataFrame,
    logger,
    *,
    dimensions: Sequence[str],
    where: str,
    value_col: str = "value",
    report_missing: bool = False,
    ) -> pd.DataFrame:
    """
    Last gate before a DataFrame becomes GDX. Returns a writable copy.

    This is where the GAMS data convention begins and nowhere earlier: GAMS has
    no NaN, and a plain 0 *is* empty.  Upstream of this function ``NaN`` keeps
    its Python meaning -- "no data" -- which is what lets
    ``calculate_climatological_forecasts`` compute quantiles over the hours it
    actually has, rather than counting missing hours as genuine zeros.

    Three checks, in order of severity:

    - **Missing dimension values** are an error.  A NaN or blank dimension cell
      would otherwise be written as the empty-string GAMS set element ``''``,
      which is silently wrong rather than loudly wrong.  The offending rows are
      dropped so the GDX stays clean; the error makes the run report failure.
    - **Non-finite values** (inf/-inf) are an error.  GAMS accepts INF, so these
      would be written happily and then make the model unbounded or infeasible
      somewhere far away from the cause.  Dropped as well.
    - **Missing values** are filled with 0, silently unless `report_missing`.
      The fill is correct and necessary; the gap itself originates in the source
      timeseries and is not something the person running a build can act on.
      Reporting it here would bury the warnings their own data does cause, so
      the audience for it is the person writing or checking a timeseries
      processor -- see the timeseries data verifier.

    Parameters:
        df: DataFrame with the dimension columns plus `value_col`
        logger: IterationLogger (or any object with log_status)
        dimensions: dimension column names that become GAMS sets
        where: label used in log messages to identify the caller
        value_col: name of the numeric column
        report_missing: log a count of the NaN entries converted to 0. Off for
            normal builds; the data verifier turns it on.

    Returns:
        A copy of `df` safe to hand to gams.transfer.
    """
    if df is None or len(df) == 0:
        return df

    work = df.copy()

    # --- dimensions must be usable as GAMS set elements ---
    present_dims = [d for d in dimensions if d in work.columns]
    if present_dims:
        blank = pd.Series(False, index=work.index)
        for d in present_dims:
            col = work[d]
            if isinstance(col.dtype, pd.CategoricalDtype):
                # A dimension column is categorical by the time it gets here
                # (timeseries_processor casts it), and it has a handful of
                # distinct labels against hundreds of thousands of rows. Decide
                # blankness once per label and map back through the codes.
                # Materialising the whole column as strings asks the same
                # question of every row: measured at ~200 ms per climate window,
                # ~6 s per parameter, which made this gate cost four times more
                # than writing the GDX it guards.
                cats = col.cat.categories.to_series().astype("string")
                bad = (cats.isna() | cats.str.strip().eq("")).to_numpy()
                # -1 is the code for a missing value, and indexing with -1 wraps
                # to this appended True.
                blank |= np.append(bad, True)[col.cat.codes.to_numpy()]
            else:
                # Compare as text: a whitespace-only label is as unusable as an
                # empty one.
                as_text = col.astype("string")
                blank |= col.isna() | as_text.fillna("").str.strip().eq("")
        if blank.any():
            examples = ", ".join(
                str(v) for v in work.loc[blank, present_dims[0]].head(3).tolist()
            )
            logger.log_status(
                f"{where}: {int(blank.sum())} row(s) have a missing or blank value in "
                f"dimension column(s) {present_dims}; these cannot become GAMS set "
                f"elements and have been dropped. First affected keys: {examples}.",
                level="error",
            )
            work = work.loc[~blank]

    if value_col not in work.columns or len(work) == 0:
        return work

    values = pd.to_numeric(work[value_col], errors="coerce")

    # --- non-finite values ---
    finite_check = values.to_numpy(dtype="float64", na_value=0.0)
    non_finite = np.isinf(finite_check)
    if non_finite.any():
        logger.log_status(
            f"{where}: {int(non_finite.sum())} row(s) have a non-finite {value_col} "
            f"(inf/-inf) and have been dropped. GAMS would accept these as INF and "
            f"the model would fail far from the cause.",
            level="error",
        )
        work = work.loc[~non_finite]
        values = values.loc[~non_finite]

    # --- missing values: fill, quietly ---
    # GAMS has no NaN, so this conversion has to happen. It is reported only when
    # asked for: a gap in a source timeseries is not actionable by whoever is
    # running a build, and warning about it every time trains people to skim past
    # the warnings their own data does cause.
    na_count = int(values.isna().sum())
    if na_count:
        if report_missing:
            logger.log_status(
                f"{where}: {na_count} of {len(values)} {value_col} entries were "
                f"missing and are written to GDX as 0, because GAMS has no NaN. "
                f"Gaps in a source timeseries reach the model as zero "
                f"generation/demand.",
                level="warn",
            )
        values = values.fillna(0)

    work[value_col] = values.astype("float64")
    return work


def read_gdx_parameter(
    gdx_file: str,
    parameter_name: str,
    ) -> pd.DataFrame:
    """
    Read a single parameter from a GDX file using gams.transfer.

    Parameters:
        gdx_file: path to the .gdx file
        parameter_name: name of the parameter to read

    Returns:
        DataFrame with one column per domain dimension plus a 'value' column.
        Returns an empty DataFrame if the parameter is missing or has no records.
        Note: GAMS drops zero values; missing keys should be treated as 0 by callers.
    """
    m = new_container(gdx_file)
    if parameter_name not in m.data:
        return pd.DataFrame()
    param = m[parameter_name]
    df = param.records
    if df is None or len(df) == 0:
        return pd.DataFrame()
    return df.reset_index(drop=True)


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

    df = prepare_values_for_gdx(
        df, logger, dimensions=parameter_dimensions, where=f"GDX '{os.path.basename(output_file)}'"
    )
    if df is None or len(df) == 0:
        logger.log_status(
            f"Skipping writing GDX '{output_file}': no rows survived validation", level="warn"
        )
        return

    work = df[list(parameter_dimensions) + ["value"]]

    m = new_container()

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
    output_folder: Path,
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

    # Gate every window before any container is built, so that a bad year is
    # reported once here rather than as an opaque gams.transfer failure below.
    annual_dfs = {
        yr: prepare_values_for_gdx(
            frame,
            logger,
            dimensions=bb_parameter_dimensions,
            where=f"GDX '{bb_parameter}_{gdx_name_suffix or ''}' climate year {yr}",
        )
        for yr, frame in annual_dfs.items()
    }
    annual_dfs = {yr: frame for yr, frame in annual_dfs.items() if frame is not None and len(frame)}
    if not annual_dfs:
        logger.log_status(
            f"Skipping GDX writing for '{bb_parameter}_{gdx_name_suffix}': "
            f"no rows survived validation.",
            level="warn",
        )
        return

    years = sorted(annual_dfs.keys())
    single_year = (len(years) == 1)
    fname_base = f"{bb_parameter}_{gdx_name_suffix}" if gdx_name_suffix else bb_parameter
    final_cols = list(bb_parameter_dimensions) + ["value"]

    # Build container once
    m = new_container()

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


