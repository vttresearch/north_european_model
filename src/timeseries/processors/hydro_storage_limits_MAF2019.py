import os
import numpy as np
import pandas as pd
from src.timeseries.processors.base_processor import BaseProcessor, SourceDataError
from src.timeseries.timeseries_helpers import (
    complete_native_grid,
    nodes_present_in_nodedata,
)


class hydro_storage_limits_MAF2019(BaseProcessor):
    """
    Class to process reservoir state limits data for a parametrized date range
    (start_date to end_date).

    What is covered, and what is not
    --------------------------------
    Only two of the four hydro types get limits here, and that is a property of
    the source data rather than a decision:

    - ``_reservoir`` for every country with rows in the weekly levels CSV;
    - ``_psOpen`` for the three Norwegian zones, whose ratios come from PEMMDB
      workbooks instead;
    - ``_psOpen`` elsewhere and ``_psClosed`` anywhere: **not built**, because
      PECD carries no weekly levels for them. Those nodes fall back to the
      constant bounds in nodedata.

    ``process()`` reports that gap per run rather than leaving it to be noticed.
    See ``docs/hydro.md`` for why the four-type split is shaped the way it is.

    Units
    -----
    The CSV and workbook values are **dimensionless fill ratios** in [0, 1]. They
    are scaled by the node's reservoir size to give **MWh of stored energy**,
    which is what ts_node(upwardLimit/downwardLimit) expects. The size comes from
    ``nodedata.upwardLimit`` and is already MWh, so no unit conversion happens
    anywhere in this processor.

    Parameters:
        input_folder (str): relative location of input files.
        country_codes (list): List of country codes.
        start_year (int): First climate year to include (e.g., 1982).
        end_year (int): Last climate year to include (e.g., 2016).
        df_nodedata (pd.DataFrame): merged nodedata, for reservoir sizes.

    Returns:
        main_result (pd.DataFrame): A MultiIndex DataFrame with time series data for each node
                                     with datetime index from start_date to end_date.
                                     Columns named by country (e.g., CH00_reservoir, NOS0_psOpen)
        secondary_result (pd.DataFrame): A DataFrame listing valid (node, boundary type, average_value) combinations
    """

    #: Bounds are stored energy, which cannot be negative.
    value_sign = "non_negative"

    #: Reservoir sizes come from nodedata rather than from a CSV of this
    #: processor's own. The two used to be kept in step by hand -- the CSV in
    #: GWh, nodedata.upwardLimit in MWh, the same numbers written twice -- and
    #: nothing could tell if they drifted.
    #:
    #: No value_range to go with value_sign: the valid maximum is each node's own
    #: size in MWh, which no single class-level tuple can say. The equivalent
    #: check is on the input instead, where the ratios are read.
    requires_source_data = ('nodedata',)

    #: Stated exceptions to "a run longer than one week needs a decision". Each
    #: entry is a decision taken after looking at the numbers, and the list
    #: doubles as a register of where the source data is bad.
    ACCEPTED_LONG_RUNS = {
        'SE04_reservoir downwardLimit': (
            "weeks 46-47 are the only multi-week zero run in the whole level dataset; "
            "they sit between 0.0268 and 0.0386 in a column whose smallest non-zero is "
            "0.00044, so they are dropped values rather than a season of permitted "
            "emptying, and interpolate to about 0.03"
        ),
    }

    def __init__(self, **kwargs):
        # Initialize base class
        super().__init__(**kwargs)

        # List of required parameters
        required_params = [
            'input_folder',
            'country_codes',
            'start_year',
            'end_year',
            'df_nodedata',
        ]

        # Check if all required parameters are present
        missing_params = [param for param in required_params if param not in kwargs]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Unpack parameters
        self.input_folder = kwargs['input_folder']
        self.country_codes = kwargs['country_codes']
        self.start_year = kwargs['start_year']
        self.end_year = kwargs['end_year']
        self.df_nodedata = kwargs['df_nodedata']

        # Derive full-year date boundaries from integer year values
        self.start_date = pd.Timestamp(f"{self.start_year}-01-01")
        self.end_date   = pd.Timestamp(f"{self.end_year}-12-31 23:00")

        # Parameters for processing "reservoir" data.
        self.minvariable = 'downwardLimit'
        self.maxvariable = 'upwardLimit'
        self.minvariable_header = 'Minimum Reservoir levels at beginning of each week (ratio) 0<=x<=1.0'
        self.maxvariable_header = 'Maximum Reservoir level at beginning of each week (ratio) 0<=x<=1.0'
        self.suffix_reservoir = '_reservoir'
        self.suffix_open = '_psOpen'
        self.suffix_closed = '_psClosed'

        # Parameters for processing Norway-specific area data.
        self.minvariable_header_norway = 'Minimum Reservoir levels at beginning of each week'
        self.maxvariable_header_norway = 'Maximum Reservoir level at beginning of each week'
        self.file_first = 'PEMMDB_'
        self.file_last = '_Hydro Inflow_SOR 20.xlsx'
        self.norway_codes = ['NOS0', 'NOM1', 'NON1']

        # (node, boundary type, blend length in weeks) for each pattern whose tail
        # was rewritten. Populated by _bound_year_change_step, summarised once at
        # the end of process().
        self.blended_patterns: set = set()

        # Nodes this run could not build, as (node, reason). Reported together at
        # the end of process() rather than one line at a time -- a country set that
        # only partly overlaps the hydro data is normal, and a stream of separate
        # warnings about it reads as breakage.
        self.unbuilt_nodes: list = []

        # Which storage-bearing nodes the model actually has. Empty means nodedata
        # could not be read, which callers must treat as "cannot tell" rather than
        # "nothing exists" -- see _is_in_model.
        self.model_nodes = nodes_present_in_nodedata(
            self.df_nodedata,
            suffixes=(self.suffix_reservoir, self.suffix_open, self.suffix_closed),
        )

        # Input file paths.
        self.levels_file = os.path.join(self.input_folder, 'PECD-hydro-weekly-reservoir-levels.csv')

    def _is_in_model(self, node) -> bool:
        """Whether nodedata carries this node at all, ignoring its parameters.

        Separate from having a usable size on purpose. A node absent here is not
        in the model and there is nothing to report about it; a node present but
        without an ``upwardLimit`` is a real node whose workbook row is
        inconsistent, and that is worth a warning.
        """
        return not self.model_nodes or node in self.model_nodes

    def _node_sizes(self) -> dict:
        """
        Map node name to its reservoir size in MWh, read from nodedata 'upwardLimit'.

        Deliberately not a {column: dtype} assumption: nodedata arrives under the
        source-data conventions, where an all-NA column is `object` rather than
        `Float64`, so the value is converted per cell and anything unconvertible is
        left out for the caller to report as a missing size.

        `0` is treated as absent here. On the source-data side 0 and pd.NA are
        distinct and 0 means "explicitly zero", but a reservoir of zero size has no
        bounds to scale either way, so both end up as "no usable size".
        """
        sizes: dict = {}
        df = self.df_nodedata
        if df is None or df.empty:
            return sizes

        columns = {str(c).lower(): c for c in df.columns}
        node_col = columns.get('node')
        limit_col = columns.get(self.maxvariable.lower())
        if node_col is None or limit_col is None:
            return sizes

        for node, raw in zip(df[node_col], df[limit_col]):
            if pd.isna(node) or pd.isna(raw):
                continue
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if value > 0:
                sizes[str(node)] = value
        return sizes

    def _check_ratio_range(self, weekly_df, columns, source):
        """
        Warn when a fill ratio is outside [0, 1].

        The whole `ratio * size` arithmetic rests on these being fractions of a
        full reservoir. The levels CSV says so in its column headers; the PEMMDB
        workbooks are read by column position with names assigned here, so they
        say nothing at all. Neither is checked anywhere else.
        """
        for column in columns:
            if column not in weekly_df.columns:
                continue
            values = pd.to_numeric(weekly_df[column], errors='coerce').dropna()
            outside = values[(values < 0) | (values > 1)]
            if not outside.empty:
                self.logger.log_status(
                    f"[{source}] {len(outside)} fill ratio(s) outside 0..1 in '{column}' "
                    f"(min {outside.min():.4g}, max {outside.max():.4g}). They are scaled by "
                    f"reservoir size as if they were fractions, so the resulting bounds are wrong.",
                    level="warn"
                )

    def _fill_weekly_data_for_year(self, lowerBound, upperBound, weekly_df,
                                   year, country, suffix, size_mwh, lower_col, upper_col):
        """
        Fill weekly anchor timestamps in lowerBound/upperBound for a single calendar year.

        Weeks are placed at Jan 4 noon + 7*i days for i=0..51 (matching the PECD convention).
        Anchor points outside [start_date, end_date] are skipped. weekly_df must be indexed
        0..N with one row per week; only the first 52 rows are used.

        `size_mwh` is the node's reservoir size in MWh and `weekly_df` holds
        dimensionless ratios, so the product is MWh of stored energy.
        """
        fourthday = pd.Timestamp(year, 1, 4, 12)
        col = country + suffix

        for i in range(min(52, len(weekly_df))):
            t = fourthday + pd.DateOffset(days=7 * i)
            if t > self.end_date:
                break
            if t < self.start_date:
                continue
            lowerBound.at[t, col] = weekly_df.at[i, lower_col] * size_mwh
            upperBound.at[t, col] = weekly_df.at[i, upper_col] * size_mwh

    def _bound_pattern_year_change(self, df_week_data, node, lower, upper):
        """Apply the year-change bound to both bounds of one node's weekly pattern.

        `lower` and `upper` are (column header, boundary type name) pairs -- the
        header differs between the PECD levels CSV and the Norwegian workbooks,
        while the boundary type is what the report and the output index call it.

        Truncated to 52 rows first, because that is the pattern that actually gets
        placed: the source carries a 53rd row and ``_fill_weekly_data_for_year``
        has always stopped at 52. Blending the full frame would measure the year
        change against week 53 and then rewrite rows nothing reads.
        """
        out = df_week_data.head(52).reset_index(drop=True).copy()
        for header, boundary_type in (lower, upper):
            if header not in out.columns:
                continue
            out[header] = self._bound_year_change_step(
                out[header].to_numpy(dtype=float), node, boundary_type
            )
        return out

    def _bound_year_change_step(self, weekly_values, node, boundary_type):
        """Blend the tail of the weekly pattern so the year change is an ordinary step.

        The pattern is climatological and the same every year, so replicating it
        per calendar year wraps week 52 back onto week 1 -- and the profile is not
        cyclic. AT00's minimum fill limit steps 0.504 -> 0.244 there, twenty-six
        percentage points of reservoir, against a largest ordinary weekly move of
        0.133. With ``bb_timeseries_start`` at 01-01 that lands on the window edge
        where nobody meets it; a summer start or a length past 365 days puts it
        mid-sample, where the solver has to absorb it.

        Week 1 is trusted and never moved: the model's year begins there. The
        blend walks backwards from week 52 only as far as it has to, so that the
        step into week 1 is no larger than one the profile already makes inside
        the year. "Already makes" is the 95th percentile rather than the maximum,
        so a single outlying week cannot license a seam as large as itself.

        This runs on the 52-row pattern before replication rather than on the
        hourly frame per year: one computation instead of thirty-five identical
        ones, and it covers the Norwegian path without a second call site.

        Returns the pattern with its tail replaced, and records what it took.
        """
        values = np.asarray(weekly_values, dtype=float)
        if len(values) < 4 or not np.isfinite(values).all():
            return values

        interior = np.abs(np.diff(values))
        normal = float(np.percentile(interior, 95))
        if normal <= 0:
            return values

        # Smallest N whose ramp from week 52-N to week 1 stays within `normal`.
        # N == 0 means the year change is already ordinary, which is 15 of the 18
        # series the shipped data carries.
        first = values[0]
        blend = None
        for n in range(0, len(values) - 2):
            anchor = values[len(values) - 1 - n]
            if abs(first - anchor) / (n + 1) <= normal:
                blend = n
                break

        if blend is None:
            self.logger.log_status(
                f"{node} {boundary_type}: the year change steps {abs(first - values[-1]):.3f} and "
                f"no blend inside the year brings it within {normal:.3f}. Left as the source has "
                f"it -- this pattern is not seasonal in any shape this can repair.",
                level="warn"
            )
            return values
        if blend == 0:
            return values

        blended = values.copy()
        anchor = values[len(values) - 1 - blend]
        for k in range(1, blend + 1):
            blended[len(values) - 1 - blend + k] = anchor + k * (first - anchor) / (blend + 1)

        self.blended_patterns.add((node, boundary_type, blend))
        return blended

    def merge_bounds(self, lowerBound, upperBound, minvariable, maxvariable):
        """
        After filling the lower and upper bound DataFrames, set the
        'param_gnBoundaryTypes' column, concatenate them, and tidy up the index.
        """
        lowerBound['param_gnBoundaryTypes'] = minvariable
        upperBound['param_gnBoundaryTypes'] = maxvariable

        result_df = pd.concat([lowerBound, upperBound])
        result_df = result_df.reset_index()
        result_df = result_df.sort_values(['index', 'param_gnBoundaryTypes'])
        result_df = result_df.set_index(['index', 'param_gnBoundaryTypes'])
        result_df.index = result_df.index.set_names(['time', 'param_gnBoundaryTypes'])
        return result_df

    def _complete_weekly_ratios(self, df_week_data, node, min_header, max_header):
        """Fill single-week holes in the weekly ratio pattern, refuse longer runs.

        A recorded zero is treated as a hole in **both** bounds. For
        ``upwardLimit`` that needs no argument: no zone asks for a reservoir kept
        empty. ``downwardLimit`` of zero is a meaningful statement in principle --
        the reservoir may run dry -- but this source never actually makes it.

        The whole shipped level dataset contains two zero runs, both in SE04's
        minimum, of one week and two weeks. SE04 expresses "essentially empty"
        constantly and in a different way: its smallest non-zero is 0.00044 and
        eleven weeks sit below 0.01. The lone zero at week 15 falls between 0.0015
        and 0.0145, and the pair at weeks 46-47 between 0.0268 and 0.0386. A
        column with that vocabulary would have written 0.0004, not 0. They are
        dropped values.

        The pattern is replicated across every year in the range, so a hole here
        is not one bad week -- it is the same bad week in all 35 of them. That is
        why anything longer than a single week is refused rather than invented,
        unless it is named in ACCEPTED_LONG_RUNS.

        Returns the repaired frame indexed 0..N by week, or None if a run survived.
        """
        weeks = df_week_data['week'].astype(int)
        # A nominal axis, only so interpolation knows the weeks are evenly spaced.
        axis = pd.DatetimeIndex(
            [pd.Timestamp(2001, 1, 4, 12) + pd.Timedelta(7 * (w - 1), unit='D') for w in weeks]
        )
        standard = pd.DatetimeIndex(
            [pd.Timestamp(2001, 1, 4, 12) + pd.Timedelta(7 * (w - 1), unit='D')
             for w in range(1, int(weeks.max()) + 1)]
        )

        repaired = {}
        for header, bound in ((min_header, self.minvariable), (max_header, self.maxvariable)):
            label = f"{node} {bound}"
            series = pd.Series(
                pd.to_numeric(df_week_data[header], errors='coerce').to_numpy(), index=axis
            ).sort_index()
            filled, report = complete_native_grid(
                series, standard, label=label, zero_is_missing=True
            )
            if not report.ok:
                if label not in self.ACCEPTED_LONG_RUNS:
                    self.unbuilt_nodes.append((
                        node,
                        f"weekly {bound} data has a run of {report.longest_run_left} week(s) with "
                        f"no usable value, which would apply to every year in the range",
                        "warn"
                    ))
                    return None
                filled = filled.interpolate(method='time', limit_area='inside').ffill().bfill()
                self.logger.log_status(
                    f"{label}: run of {report.longest_run_left} week(s) with no usable value "
                    f"interpolated over. Accepted rather than escalated: "
                    f"{self.ACCEPTED_LONG_RUNS[label]}.",
                    level="info"
                )
            elif report.n_autofilled:
                self.logger.log_status(
                    f"{label}: {report.n_autofilled} single-week gap(s) in the level pattern "
                    f"interpolated.",
                    level="info"
                )
            repaired[header] = filled.reindex(standard)

        out = pd.DataFrame(
            {min_header: repaired[min_header].to_numpy(),
             max_header: repaired[max_header].to_numpy()}
        )
        return out.reset_index(drop=True)

    def process_country(self, country, df_country, size_mwh,
                        minvariable_header, maxvariable_header,
                        minvariable, maxvariable,
                        suffix_reservoir):
        """
        Process data for a single country (non-Norway) across the full date range.

        The weekly pattern from the CSV is identical for all years; it is replicated
        for each calendar year in [start_date.year, end_date.year].

        Returns None when the country's rows carry a gap, so the caller can report
        it. The gap test looks only at rows the run actually uses, which is why
        process() filters to the climate range before calling: AT00's 2017 block is
        entirely NA, and testing it disqualified AT00 from runs that end in 2016.
        """
        date_index = pd.date_range(self.start_date, self.end_date, freq='60 min')
        df_lowerBound = pd.DataFrame(index=date_index)
        df_upperBound = pd.DataFrame(index=date_index)

        if df_country.empty:
            return None

        df_country = df_country.sort_values(by=['year', 'week']).reset_index(drop=True)
        available_years = df_country['year'].unique()
        if len(available_years) == 0:
            return None

        # Weekly data is identical across years; use the first year's rows
        df_week_data = df_country[df_country['year'] == available_years[0]].reset_index(drop=True)
        self._check_ratio_range(
            df_week_data, (minvariable_header, maxvariable_header),
            f"{self.levels_file}:{country}"
        )

        # Complete the weekly pattern before it is replicated and cast to hourly.
        # A single missing week is repaired; a longer run is refused, because the
        # bound it would invent applies to every year of the run.
        node = country + suffix_reservoir
        df_week_data = self._complete_weekly_ratios(
            df_week_data, node, minvariable_header, maxvariable_header
        )
        if df_week_data is None:
            return None
        col = country + suffix_reservoir
        df_week_data = self._bound_pattern_year_change(
            df_week_data, col, (minvariable_header, minvariable), (maxvariable_header, maxvariable)
        )
        for year in range(self.start_date.year, self.end_date.year + 1):
            self._fill_weekly_data_for_year(
                df_lowerBound, df_upperBound, df_week_data,
                year, country, suffix_reservoir, size_mwh,
                minvariable_header, maxvariable_header
            )

        df_lowerBound.interpolate(inplace=True, limit_direction='both')
        df_upperBound.interpolate(inplace=True, limit_direction='both')

        result_df = self.merge_bounds(df_lowerBound, df_upperBound, minvariable, maxvariable)
        return result_df

    def process_norway_area(self, country, filename,
                            size_mwh,
                            minvariable_header_norway, maxvariable_header_norway,
                            minvariable, maxvariable,
                            suffix_open):
        """
        Process Norway-specific area data using an Excel input file across the full date range.

        The weekly pattern from the Excel is replicated for each calendar year.

        The columns are taken by position (L and M) and named here, so unlike the
        levels CSV the file itself never states that these are ratios. That is what
        _check_ratio_range is for.
        """
        date_index = pd.date_range(self.start_date, self.end_date, freq='60 min')
        df_lowerBound = pd.DataFrame(index=date_index)
        df_upperBound = pd.DataFrame(index=date_index)

        try:
            df = self.read_input_excel(
                os.path.normpath(filename),
                sheet_name='Pump storage - Open Loop',
                usecols="L,M",
                names=[minvariable_header_norway, maxvariable_header_norway],
                skiprows=12
            )
        except SourceDataError:
            raise
        except Exception as e:
            self.logger.log_status(f"Error reading Norway input Excel: {e}", level="warn")
            self.unbuilt_nodes.append(
                (country + suffix_open, "PEMMDB workbook could not be read", "warn")
            )
            return None

        self._check_ratio_range(
            df.head(52), (minvariable_header_norway, maxvariable_header_norway),
            f"{os.path.basename(filename)}:{country}"
        )
        col = country + suffix_open
        df = self._bound_pattern_year_change(
            df, col,
            (minvariable_header_norway, minvariable),
            (maxvariable_header_norway, maxvariable),
        )
        for year in range(self.start_date.year, self.end_date.year + 1):
            self._fill_weekly_data_for_year(
                df_lowerBound, df_upperBound, df,
                year, country, suffix_open, size_mwh,
                minvariable_header_norway, maxvariable_header_norway
            )

        df_lowerBound.interpolate(inplace=True, limit_direction='both')
        df_upperBound.interpolate(inplace=True, limit_direction='both')

        result_df = self.merge_bounds(df_lowerBound, df_upperBound, minvariable, maxvariable)
        return result_df

    def _report_coverage(self, node_sizes, built_nodes):
        """
        Say once what was built and what was not, instead of leaving it to be noticed.

        Split by whether the user can do anything about it. A zone the source data
        does not cover is information -- most countries in a run have no reservoir,
        and splitting a country into regions produces codes PECD has never seen. A
        node that has ratios but no size, or that came out all zero, is a warning:
        something in the input data is inconsistent with itself.
        """
        built = sorted(built_nodes)
        self.logger.log_status(
            f"Storage limits built for {len(built)} node(s): {', '.join(built) if built else 'none'}.",
            level="info"
        )

        for level in ("warn", "info"):
            entries = sorted((n, r) for n, r, lvl in self.unbuilt_nodes if lvl == level)
            if not entries:
                continue
            detail = ', '.join(f"{node} ({reason})" for node, reason in entries)
            self.logger.log_status(
                f"No storage limit time series for {len(entries)} node(s): {detail}. "
                f"These keep whatever constant bounds nodedata gives them.",
                level=level
            )

        # Nodes this processor never even attempts. PECD carries no weekly levels
        # for open-loop pumped storage outside Norway, nor for closed loop
        # anywhere, so those nodes are constant-bounded by construction rather
        # than by failure. Worth stating, because the node exists and its absence
        # from the time series otherwise looks like a bug.
        never_attempted = sorted(
            node for node in node_sizes
            if (node.endswith(self.suffix_open)
                and node[:-len(self.suffix_open)] not in self.norway_codes)
            or node.endswith(self.suffix_closed)
        )
        if never_attempted:
            self.logger.log_status(
                f"{len(never_attempted)} pumped storage node(s) have no weekly level data in "
                f"PECD and are not built here: {', '.join(never_attempted)}. See docs/hydro.md.",
                level="info"
            )

    def process(self) -> pd.DataFrame:
        """
        Main processing logic that executes the hydro storage data processing pipeline.

        This function performs the following steps:
        1. Reads reservoir levels data and filters it to the climate range
        2. Reads reservoir sizes from nodedata
        3. Creates a time-series framework for the full date range
        4. For each calendar year in the range, replicates the weekly pattern and
           detects/smooths large jumps at year boundaries
        5. Interpolates weekly anchor points to hourly resolution
        6. Creates ts_hydro_storage_limits_df of valid (node, boundary type) combinations
        7. Reports which requested nodes were not built, and why

        A country code that the hydro data does not cover is normal -- most of a
        run's countries have no reservoir at all, and a country can be split into
        regions the source data has never heard of. None of that is an error: the
        node keeps the constant bounds from nodedata and the run continues.

        Returns:
            pd.DataFrame: A MultiIndex DataFrame with time series data for each node
                         with datetime index from start_date to end_date
        """
        # --- Read and prepare the input data
        self.logger.log_status("Reading input files...")
        # SourceDataError is re-raised rather than swallowed: a missing or
        # unreadable file is a warning and an empty result, but a file whose
        # numbers are malformed has already been reported at error level and
        # must stop the processor instead of quietly yielding no rows.
        try:
            df_levels = self.read_input_csv(self.levels_file)
        except SourceDataError:
            raise
        except Exception as e:
            self.logger.log_status(f"Error reading hydro storage levels CSV '{self.levels_file}': {e}", level="warn")
            return pd.DataFrame()

        # Filter to the climate range before anything looks at completeness.
        # hydro_inflow_MAF2019 has always done this; this processor did not, so a
        # year outside the range could disqualify a country from a run that never
        # touches it.
        df_levels = df_levels[
            (df_levels["year"] >= self.start_year) & (df_levels["year"] <= self.end_year)
        ]

        node_sizes = self._node_sizes()
        if not node_sizes:
            self.logger.log_status(
                f"No usable '{self.maxvariable}' values in nodedata, so no reservoir sizes are "
                f"available to scale the fill ratios. No hydro storage limits will be built.",
                level="warn"
            )
            return pd.DataFrame()

        # Create a summary DataFrame with a MultiIndex (time, param_gnBoundaryTypes) for the full date range
        idx = pd.MultiIndex.from_product(
            [pd.date_range(self.start_date, self.end_date, freq='60 min'),
             [self.minvariable, self.maxvariable]],
            names=['time', 'param_gnBoundaryTypes']
        )
        summary_df = pd.DataFrame(index=idx)

        # --- Build country level timeseries
        self.logger.log_status(
            f"Building country level timeseries ({self.start_date.year}-{self.end_date.year})..."
        )

        for country in self.country_codes:
            is_norway = country in self.norway_codes
            node = country + (self.suffix_open if is_norway else self.suffix_reservoir)

            # A size is required before anything is built. Missing one used to be a
            # KeyError out of the capacity lookup, which ProcessorRunner caught at
            # whole-processor level and turned into no GDX for *any* country.
            # A node the model does not have is not an absence to describe.
            # nodedata answers that directly; this used to guess it from whether
            # PECD happened to carry ratios for the zone, which is a proxy for the
            # wrong question.
            if not self._is_in_model(node):
                continue

            size_mwh = node_sizes.get(node)
            if size_mwh is None:
                self.unbuilt_nodes.append((
                    node, f"no usable '{self.maxvariable}' in nodedata to scale the ratios",
                    "warn"
                ))
                continue

            if is_norway:
                # Special handling for Norway using Excel data
                filename = os.path.join(self.input_folder, f"{self.file_first}{country}{self.file_last}")
                result_df = self.process_norway_area(country, filename, size_mwh,
                                                     self.minvariable_header_norway, self.maxvariable_header_norway,
                                                     self.minvariable, self.maxvariable,
                                                     self.suffix_open)
            else:
                # Standard processing for other countries
                df_country = df_levels[df_levels["zone"] == country]
                if df_country.empty:
                    self.unbuilt_nodes.append((node, "no weekly level data for this zone", "info"))
                    continue
                result_df = self.process_country(country, df_country, size_mwh,
                                                 self.minvariable_header, self.maxvariable_header,
                                                 self.minvariable, self.maxvariable,
                                                 self.suffix_reservoir)

            # Merge country results into summary dataframe
            if result_df is not None:
                result_df.index = result_df.index.set_names(['time', 'param_gnBoundaryTypes'])
                summary_df = summary_df.join(result_df, how='left')

        # Remove columns with no data (sum equals zero)
        dropped = [c for c in summary_df.columns if not summary_df[c].sum() > 0]
        for node in dropped:
            self.unbuilt_nodes.append((node, "all bounds came out as zero", "warn"))
        summary_df = summary_df.loc[:, summary_df.sum() > 0]

        self._report_coverage(node_sizes, list(summary_df.columns))

        # Say which patterns needed their tail blended, and by how much. The
        # blend length is the point: it is the smallest that brings the year
        # change within an ordinary weekly step, so a long one says the source
        # profile is far from cyclic rather than that the code tried hard.
        if self.blended_patterns:
            series_list = ', '.join(
                f"{col} ({btype}, {weeks} week{'s' if weeks > 1 else ''})"
                for col, btype, weeks in sorted(self.blended_patterns)
            )
            self.logger.log_status(
                f"Blended the last weeks of {len(self.blended_patterns)} weekly pattern(s) so the "
                f"year change is no larger a step than the profile makes inside the year: "
                f"{series_list}. Week 1 is unchanged in every case.",
                level="info"
            )

        # --- Create secondary results for nodes that have timeseries format limits
        # This is used when building the input excel to create correct info
        # e.g. in p_gnBoundaryProperties

        # Create a mask where values are > 0
        mask = summary_df > 0

        # Check which (node, boundary type) combinations have data
        has_data = mask.groupby(level='param_gnBoundaryTypes').sum() > 0

        # Get all combinations with data and their average values
        combinations_with_data = []
        for boundary_type in summary_df.index.get_level_values('param_gnBoundaryTypes').unique():
            boundary_data = summary_df.xs(boundary_type, level='param_gnBoundaryTypes')

            for node in summary_df.columns:
                if has_data.loc[boundary_type, node]:
                    positive_values = boundary_data[node][boundary_data[node] > 0]
                    avg_value = positive_values.mean() if len(positive_values) > 0 else 0
                    combinations_with_data.append((node, boundary_type, avg_value))

        ts_hydro_storage_limits = pd.DataFrame(
            combinations_with_data,
            columns=['node', 'param_gnBoundaryTypes', 'average_value']
        )

        self.secondary_result = ts_hydro_storage_limits

        self.logger.log_status("Hydro storage limit time series built.", level="info")

        # Convert to long format: [grid, node, param_gnBoundaryTypes, time, value]
        result = summary_df.reset_index()  # MultiIndex (time, param_gnBoundaryTypes) -> regular columns
        node_cols = [c for c in result.columns if c not in ('time', 'param_gnBoundaryTypes')]
        result = result.melt(id_vars=['time', 'param_gnBoundaryTypes'], value_vars=node_cols,
                             var_name='node', value_name='value')
        result['grid'] = result['node'].str.split('_').str[1]
        return result[['grid', 'node', 'param_gnBoundaryTypes', 'time', 'value']]
