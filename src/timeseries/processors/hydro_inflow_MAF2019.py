import os
import calendar
import pandas as pd
import numpy as np
from src.timeseries.processors.base_processor import BaseProcessor
from src.timeseries.timeseries_helpers import (
    complete_native_grid,
    nodes_present_in_nodedata,
)


class hydro_inflow_MAF2019(BaseProcessor):
    """
    Class to process hydro inflows (reservoir, pump open cycle, and run-of-river) data.

    Units
    -----
    The source files give energy per period -- GWh per week for reservoir and
    pump-storage inflow, GWh per day for run-of-river. Both are converted to
    average power in MW by multiplying by 1000 and dividing by the number of hours
    in the period, which is what ts_influx expects. No installed capacity is
    involved anywhere: these are inflows, not capacity factors.

    Parameters:
        input_folder (str): relative location of input files.
        country_codes (list): List of country codes.
        start_year (int): First climate year to include (e.g., 1982).
        end_year (int): Last climate year to include (e.g., 2016).
    """

    #: Inflow is water arriving, which cannot be negative. No value_range: the
    #: plausible maximum is a property of the catchment, not of the parameter.
    value_sign = "non_negative"

    #: nodedata answers which hydro nodes the model actually has. Without it this
    #: processor built the cross product of every country code and every hydro
    #: type and then reported on all of it -- 37 lines about 35 nodes that do not
    #: exist, burying the two that do. See _model_hydro_nodes.
    requires_source_data = ('nodedata',)

    #: How far a week 52 -> week 1 change may exceed the node's own 95th-percentile
    #: within-year weekly change before the build says so. The seam is one sample
    #: of the same weekly distribution, so a plain "above the 95th percentile" test
    #: fires about as often as chance predicts -- 6 to 11 times in 34 years for
    #: CH00, DE00 and ES00 -- and reports nothing. At three times it names the
    #: three genuine outliers in the shipped data and stays quiet otherwise.
    YEAR_CHANGE_OUTLIER_MULTIPLIER = 3.0

    #: Stated exceptions to "a run longer than one slot needs a decision". Each
    #: entry is a decision already taken, with the reason, rather than a silence.
    #:
    #: A new one is meant to be added deliberately, after looking at the numbers.
    #: If a data refresh introduces a gap that is not listed here, the build warns
    #: and someone decides -- which is the point of keeping the list explicit
    #: rather than inferring which gaps are safe to close.
    ACCEPTED_LONG_RUNS = {
        'PL00_psOpen': (
            "Polish pump storage averages ~1.2 GWh/week against roughly 3400 GWh of "
            "weekly demand, so the longest run costs a fraction of a per mille"
        ),
        'CH00_reservoir': (
            "1990, 2000, 2005 and 2013 stop at week 51, leaving both year-end slots "
            "empty; interpolating from ~18 GWh in week 51 to ~90 GWh in the new year "
            "adds ~110 GWh, about 0.55% of a 19.7 TWh year"
        ),
        'SE02_reservoir': (
            "1985 has no weeks 6-7; they sit between 176.6 and 112.2 GWh in "
            "mid-February and interpolate to ~289 GWh, about 0.75% of a 38.5 TWh year"
        ),
        'SE01_reservoir': (
            "PECD omits winter weeks in ten years, longest five weeks in 1985. This "
            "repo used to carry a hand-patched PECD-hydro-weekly-inflows-corrected.csv "
            "that filled them by holding the previous week; interpolating instead costs "
            "~744 GWh over 35 years, about 0.1% of a 20.2 TWh year"
        ),
        # The two Norwegian entries below rest on a check made once, recorded
        # here, and deliberately not repeated at runtime.
        #
        # NOS0 reads exactly zero for five weeks of 1995 and NON1 for four across
        # 1984 and 1987 -- February and March weeks of unusually dry years, which
        # invites the explanation that the catchment was frozen and the melt had
        # not started. That explanation fails its own test: if cold drove inflow
        # to nothing it would do so region-wide, and NOM1 does not record a single
        # zero in 35 years. In those same weeks NOM1 sits at 5-25% of its median
        # and NON1 at 17-77%. Cold takes a Norwegian catchment to a fraction of
        # normal, not to nothing. Nor is it a rounding artefact: NON1's smallest
        # non-zero is 0.0286 GWh/week, so the source can say "almost nothing" when
        # it means it.
        #
        # So the zeros are dropped values in genuinely dry weeks. Interpolating
        # across this zone's own neighbouring weeks keeps the drought -- ~13 and
        # ~17 GWh between real weeks of 9.3 and 20.3 -- and adds roughly 100 GWh
        # to each affected year, far too little to move any conclusion about them.
        # What it removes is a discontinuity the solver has no reason to be handed.
        'NOS0_psOpen': (
            "five zero weeks in 1995 that the neighbouring Norwegian zones "
            "contradict; interpolating adds ~100 GWh to a 62 TWh year"
        ),
        'NON1_psOpen': (
            "four zero weeks across 1984 and 1987 that the neighbouring Norwegian "
            "zones contradict; interpolating adds ~100 GWh to a 23 TWh year"
        ),
    }

    def __init__(self, **kwargs_processor):
        # Initialize base class
        super().__init__(**kwargs_processor)
        
        # List of required parameters
        required_params = [
            'input_folder',
            'country_codes',
            'start_year',
            'end_year',
            'df_nodedata',
        ]

        # Check if all required parameters are present
        missing_params = [param for param in required_params if param not in kwargs_processor]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Unpack parameters
        self.input_folder = kwargs_processor['input_folder']
        self.country_codes = kwargs_processor['country_codes']
        self.start_year = kwargs_processor['start_year']
        self.end_year = kwargs_processor['end_year']
        self.df_nodedata = kwargs_processor['df_nodedata']

        # Derive full-year date boundaries from integer year values
        self.start_date = pd.Timestamp(f"{self.start_year}-01-01")
        self.end_date   = pd.Timestamp(f"{self.end_year}-12-31 23:00")

        # Define folders and file paths.
        self.file_weekly = os.path.join(self.input_folder, 'PECD-hydro-weekly-inflows.csv')
        self.file_daily = os.path.join(self.input_folder, 'PECD-hydro-daily-ror-generation.csv')

        # Define column headers and suffixes.
        self.inflow1_header = 'Cumulated inflow into reservoirs per week in GWh'
        self.inflow2_header = 'Cumulated NATURAL inflow into the pump-storage reservoirs per week in GWh'
        self.inflow3_header = 'Run of River Hydro Generation in GWh per day'
        self.inflow1_suffix = '_reservoir'
        self.inflow2_suffix = '_psOpen'
        self.inflow3_suffix = '_ror'

        # Nodes the run asked for and this processor did not build, as
        # (node, reason, level) triples for _report_coverage to say once.
        self.unbuilt_nodes = []

        self.model_nodes = self._model_hydro_nodes()

    def _model_hydro_nodes(self):
        """The inflow-bearing hydro nodes this run's model actually contains.

        An empty answer means nodedata could not be read rather than that the
        model has no hydro, so it falls back to building every country and type,
        which is what this processor did before the gate existed. Failing open
        keeps a malformed workbook from silently deleting the hydro fleet; the
        warning says which of the two happened.
        """
        suffixes = (self.inflow1_suffix, self.inflow2_suffix, self.inflow3_suffix)
        nodes = nodes_present_in_nodedata(self.df_nodedata, suffixes=suffixes)
        if not nodes:
            self.logger.log_status(
                "nodedata carries no usable 'node' column, so which hydro nodes the model "
                "has cannot be determined. Building inflow for every configured country and "
                "hydro type instead, which may include nodes the model does not have.",
                level="warn"
            )
        return nodes

    def _is_in_model(self, node):
        """Gate on presence in nodedata, never on a node parameter.

        Deliberately not conditioned on upwardLimit: inflow describes water
        arriving, which happens whether or not the workbook records a reservoir
        size, and a size left blank or zero by oversight must not cascade into a
        node that reaches GAMS with no inflow at all. The storage limits are the
        opposite case and do require a size, because a fill limit is a fraction
        of one.
        """
        return not self.model_nodes or node in self.model_nodes

    def _report_grid(self, report, unit):
        """Say what the grid needed, and warn only when it is still not whole."""
        if report.n_missing == 0:
            return
        if report.ok:
            self.logger.log_status(
                f"{report.label}: {report.n_autofilled} single-{unit} gap(s) interpolated "
                f"({report.twh_per_year:.2f} TWh/year series).",
                level="info"
            )
            return
        self.logger.log_status(
            f"{report.label}: {report.n_left} {unit}(s) have no usable value and were left as "
            f"they are, the longest run being {report.longest_run_left} {unit}(s) from "
            f"{report.first_left:%Y-%m-%d}. Runs longer than one are not interpolated, because "
            f"that is invention rather than repair -- this one needs a decision. The series "
            f"averages {report.twh_per_year:.2f} TWh/year.",
            level="warn"
        )

    def _process_reservoir_inflows(self, weekly_df):
        """
        Processes the weekly (reservoir and pump‐storage) data for each country.
        Returns one combined DataFrame with appropriately suffixed columns.
        """
        # Create a full hourly index for the given date range.
        full_index = pd.date_range(self.start_date, self.end_date, freq='60min')
        end_ts = pd.Timestamp(self.end_date)
        result_df = pd.DataFrame(index=full_index)

        # The weekly grid every year in range: weeks 1..52, a week apart from
        # Jan 4. Anything the source does not supply is interpolated onto this
        # before the hourly cast, so the hourly pass never meets a gap wider than
        # the year change itself.
        standard_weeks = pd.DatetimeIndex(sorted(
            pd.Timestamp(year, 1, 4, 12) + pd.Timedelta(7 * (week - 1), unit='D')
            for year in range(self.start_date.year, self.end_date.year + 1)
            for week in range(1, 53)
        ))
        standard_weeks = standard_weeks[standard_weeks <= end_ts]

        for country in self.country_codes:
            col_name1 = country + self.inflow1_suffix
            col_name2 = country + self.inflow2_suffix

            # Nothing below runs for a node the model does not have -- not the
            # column, not the reporting. A country in country_codes without a
            # reservoir is the ordinary case, not an absence worth describing.
            wanted = [c for c in (col_name1, col_name2) if self._is_in_model(c)]
            if not wanted:
                continue

            # Filter for the country's data.
            #
            # Week 53 is dropped rather than read. It is not a week: the year has
            # 52 whole weeks and a remainder of one or two days, and the source
            # says different things about that remainder in different zones. Ten
            # of the twenty-eight reservoir zones -- SE01-04, CH00, HR00 and the
            # three Norwegian ones -- repeat week 52 verbatim, so the cell carries
            # no information. AT00 reports the remainder day itself: its week 53 is
            # 0.136-0.145 of its week 52 in all 36 years, and dividing that by 168
            # like a whole week collapsed AT00 inflow from 51.7 to 14.2 MWh/h in a
            # single day, every New Year. See docs/hydro.md.
            df_country = weekly_df[
                (weekly_df["zone"] == country) & (weekly_df["week"] <= 52)
            ].copy()
            if df_country.empty:
                for node in wanted:
                    result_df[node] = np.nan
                    self.unbuilt_nodes.append(
                        (node, "no weekly inflow data for this zone", "info")
                    )
                continue

            # Sort by year and week, and fill missing values.
            df_country.sort_values(by=['year', 'week'], inplace=True)

            # Anchor each row by its week *number*, not by its position in the
            # frame. Weeks 1..52 sit at Jan 4 12:00 + (w-1) weeks.
            #
            # Position was the wrong key. SE01 1991 has no week 7 and SE02 1985 no
            # weeks 6 or 7, so every later week was placed one or two weeks early
            # for the whole rest of the year -- values attached to the wrong hours,
            # which no value check can see. That is the same failure the t-label
            # checks exist to prevent, one stage earlier.
            years = df_country['year'].astype(int)
            weeks = df_country['week'].astype(int)
            anchors = (
                pd.to_datetime(years.astype(str) + '-01-04')
                + pd.Timedelta(12, unit='h')
                + pd.to_timedelta((weeks - 1) * 7, unit='D')
            )
            df_country = df_country.assign(_anchor=anchors)
            df_country = df_country[df_country['_anchor'] <= end_ts]
            if df_country.empty:
                for node in wanted:
                    result_df[node] = np.nan
                    self.unbuilt_nodes.append(
                        (node, "no weekly inflow data inside the climate range", "info")
                    )
                continue

            # No fillna(0) here. A blank cell is missing data and a zero is a
            # recorded zero, and the two must stay distinguishable all the way to
            # complete_native_grid, which decides what to do about each. Filling
            # early is the exact anti-pattern the project's data conventions warn
            # about: it makes a source gap indistinguishable from a real value.
            def anchor_series(header):
                """Weekly anchors for one inflow column, in MWh/h."""
                values = 1000 * df_country[header].to_numpy() / 168
                series = pd.Series(values, index=pd.DatetimeIndex(df_country['_anchor']))
                return series[~series.index.duplicated(keep='first')].sort_index()

            for header, col_name, kind in (
                (self.inflow1_header, col_name1, "reservoir"),
                (self.inflow2_header, col_name2, "pump-storage"),
            ):
                if col_name not in wanted:
                    continue
                series = anchor_series(header)
                if self._has_no_inflow(series):
                    result_df[col_name] = np.nan
                    self.unbuilt_nodes.append(
                        (col_name, f"the source carries no {kind} inflow for this zone", "info")
                    )
                    continue
                series = self._complete_and_report(series, col_name, standard_weeks, "week")
                self._report_year_change_outliers(series, col_name)
                result_df[col_name] = self._cast_to_hourly(series, full_index, standard_weeks)

        return self._drop_empty_columns(result_df, "all inflow came out as zero")

    @staticmethod
    def _has_no_inflow(series):
        """True when the source says nothing about this zone and series.

        PECD ships a row for every zone whether or not the zone has a plant of
        that type, so "no reservoir in Belgium" arrives either as an empty column
        or as thirty-five years of zeros. Both mean the same thing, and neither is
        a gap: sending them through complete_native_grid would report the whole
        record as missing and ask someone to decide about a technology that does
        not exist. Answered here instead, once, at info level.
        """
        values = series.to_numpy(dtype=float)
        return not np.any(np.isfinite(values) & (values != 0))

    def _cast_to_hourly(self, weekly_series, full_index, standard_weeks):
        """Spread a completed weekly series over the hourly index.

        The rule is "fill between two adjacent weekly anchors that both have a
        value, and nowhere else". Stated that way it needs no interpolation limit,
        which is what the old ``limit=84`` was standing in for from before the
        weekly grid was completed first. A limit cannot express this anyway: the
        spans are not all the same length, since 52 weeks from Jan 4 reach only day
        361, so a year change is 192 h in a common year and **216 h after a leap
        one** -- and no single number both bridges 216 h and refuses the 336 h that
        a two-week gap leaves.

        Refusing that longer span is the point. ``complete_native_grid`` fills a
        single missing week and deliberately leaves a longer run alone, because
        bridging it is invention rather than repair. An unlimited interpolation
        would put the invention back at the hourly stage, where nothing reports it.

        The two ends are a separate question, and the answer is not "leave them".
        Linear interpolation does not extrapolate, so the 84 h from Jan 1 00:00 to
        the first anchor and the 107 h (131 h after a leap year) from the last
        anchor to Dec 31 23:00 would stay empty -- and nothing downstream would
        say so, because ``find_time_axis_defects`` checks that the rows are there
        rather than what is in them, and ``prepare_values_for_gdx`` turns the
        blanks into zeros without a word. So the nearest week is carried outward,
        which is what ``complete_native_grid`` already does for a terminal slot.
        """
        if not len(standard_weeks):
            return weekly_series.reindex(full_index)

        on_grid = weekly_series.reindex(standard_weeks)
        hourly = on_grid.reindex(full_index).interpolate(limit_area='inside')

        # An hour is legitimate only if the anchors on both sides of it carry a
        # value. ffill answers "was the previous anchor filled", bfill "is the next
        # one" -- and both are false in the stubs, which have no anchor on one side.
        known = on_grid.notna().astype(float).reindex(full_index)
        hourly = hourly.where((known.ffill() > 0) & (known.bfill() > 0))

        head = hourly.loc[:standard_weeks[0]]
        hourly.loc[:standard_weeks[0]] = head.bfill()
        tail = hourly.loc[standard_weeks[-1]:]
        hourly.loc[standard_weeks[-1]:] = tail.ffill()
        return hourly

    def _drop_empty_columns(self, result_df, reason):
        """Drop nodes that carry nothing, and record why before they vanish.

        An all-NaN column -- a country the source has never heard of -- sums to
        zero just like a genuinely all-zero one, so both used to disappear here
        without a word. They are different situations: the first is recorded by
        the caller as information, and this one is a warning, because a zone with
        rows that yields no inflow is inconsistent with itself.
        """
        empty = [c for c in result_df.columns if not result_df[c].sum() > 0]
        for node in empty:
            if not any(node == n for n, _, _ in self.unbuilt_nodes):
                self.unbuilt_nodes.append((node, reason, "warn"))
        return result_df.loc[:, result_df.sum() != 0]

    def _report_year_change_outliers(self, weekly_series, label):
        """Name a year change that is a gross outlier in this node's own weeks.

        Weeks 52 and 1 are seven days apart like any other pair, so the change
        between them is one sample of the same weekly distribution and is usually
        unremarkable -- across the shipped zones it exceeds the node's own 95th
        percentile in none of the 34 year changes for AT00, FI00, SE01-04 and the
        Norwegian zones. Nothing is smoothed here, because a large change at New
        Year is weather and inventing a gentler one would be a lie. It is reported
        so that a data refresh which makes the seam anomalous is visible rather
        than absorbed, which is the job ACCEPTED_LONG_RUNS does for gaps.
        """
        series = weekly_series.dropna()
        if len(series) < 3:
            return

        by_year = {year: group for year, group in series.groupby(series.index.year)}
        interior = np.concatenate(
            [np.abs(np.diff(g.to_numpy())) for g in by_year.values() if len(g) > 1]
            or [np.array([])]
        )
        if interior.size == 0:
            return
        threshold = self.YEAR_CHANGE_OUTLIER_MULTIPLIER * np.percentile(interior, 95)
        if threshold <= 0:
            return

        years = sorted(by_year)
        outliers = []
        for earlier, later in zip(years, years[1:]):
            if later != earlier + 1:
                continue
            jump = abs(by_year[later].iloc[0] - by_year[earlier].iloc[-1])
            if jump > threshold:
                outliers.append((later, jump))
        if not outliers:
            return

        detail = ', '.join(f"{year} ({jump:.0f} MWh/h)" for year, jump in outliers)
        self.logger.log_status(
            f"{label}: the week 52 to week 1 change is more than "
            f"{self.YEAR_CHANGE_OUTLIER_MULTIPLIER:g} times this node's own 95th-percentile "
            f"weekly change at {len(outliers)} year change(s): {detail}. Left as the source "
            f"has it -- a large change at New Year is weather, not a seam.",
            level="info"
        )

    def _report_coverage(self, built_nodes):
        """
        Say once what was built and what was not, instead of leaving it to be noticed.

        Split by whether the user can do anything about it. A zone the source data
        does not cover is information -- most countries in a run have no reservoir,
        and splitting a country into regions produces codes PECD has never seen. A
        zone that has rows but yields nothing is a warning: something in the input
        data is inconsistent with itself.
        """
        built = sorted(built_nodes)
        self.logger.log_status(
            f"Inflow built for {len(built)} node(s): {', '.join(built) if built else 'none'}.",
            level="info"
        )

        for level in ("warn", "info"):
            entries = sorted((n, r) for n, r, lvl in self.unbuilt_nodes if lvl == level)
            if not entries:
                continue
            detail = ', '.join(f"{node} ({reason})" for node, reason in entries)
            self.logger.log_status(
                f"No inflow time series for {len(entries)} node(s): {detail}. These reach "
                f"GAMS with no inflow at all.",
                level=level
            )

    def _complete_and_report(self, series, label, standard_index, unit):
        """Complete one native-resolution series and say what that took.

        Single-slot gaps are repaired here. A longer run is left alone and
        warned about, unless it appears in ACCEPTED_LONG_RUNS -- a run of two or
        more is no longer a repair, and whoever adopts a data source should be
        the one deciding what it ought to be.
        """
        filled, report = complete_native_grid(
            series, standard_index, label=label, zero_is_missing=True
        )
        if not report.ok and label in self.ACCEPTED_LONG_RUNS:
            filled = filled.interpolate(method='time', limit_area='inside').ffill()
            self.logger.log_status(
                f"{label}: {report.n_left} {unit}(s) with no usable value interpolated over, "
                f"longest run {report.longest_run_left}. Accepted rather than escalated: "
                f"{self.ACCEPTED_LONG_RUNS[label]}.",
                level="info"
            )
            return filled.dropna()
        self._report_grid(report, unit)
        return filled.dropna()

    def _process_ror_inflows(self, daily_df):
        """
        Processes daily run‐of‐river data for each country.
        Returns one combined DataFrame with appropriately suffixed columns.
        """
        full_index = pd.date_range(self.start_date, self.end_date, freq='60min')
        end_ts = pd.to_datetime(self.end_date)
        result_df = pd.DataFrame(index=full_index)

        # Every calendar day at noon, which is where the anchors land. Leap years
        # get their 366th: the source ships 365 rows regardless, and the block
        # above re-emits Dec 31 to make up the difference.
        standard_days = pd.date_range(
            pd.Timestamp(self.start_date.year, 1, 1, 12), end_ts, freq='D'
        )

        for country in self.country_codes:
            col_name = country + self.inflow3_suffix
            if not self._is_in_model(col_name):
                continue

            # Filter for the country's data.
            df_country = daily_df[daily_df["zone"] == country].copy()
            if df_country.empty:
                result_df[col_name] = np.nan
                self.unbuilt_nodes.append(
                    (col_name, "no daily run-of-river data for this zone", "info")
                )
                continue

            # No fillna(0): a blank day and a recorded zero are different things,
            # and complete_native_grid below is what decides between them.
            df_country.sort_values(by=['year', 'Day'], inplace=True)

            # Anchor each row by its Day *number*, not by its position in the
            # frame, for the same reason the weekly path does: a missing row would
            # otherwise pull every later day of that year one day early, putting
            # right values on wrong hours where no value check can see them. The
            # shipped file is a clean 1..365 block in all 1548 zone-years, so this
            # changes nothing today and is here so a data refresh cannot.
            df_country['timestamp'] = (
                pd.to_datetime(df_country['year'].astype(str), format='%Y') +
                pd.to_timedelta(df_country['Day'].astype(int) - 1, unit='D') +
                pd.Timedelta(12, unit="h")
            )

            # Compute the flow values.
            df_country['value'] = 1000 * df_country[self.inflow3_header] / 24

            # Handle leap years: the source ships 365 days whatever the calendar,
            # so day 365 lands on Dec 30 and Dec 31 has no anchor of its own. Give
            # it day 365's value again, which is the convention
            # elec_demand_TYNDP2024 also follows. Selected by Day number rather
            # than by row position, for the reason given above.
            leap_rows = []
            for yr, grp in df_country.groupby('year'):
                last_day = grp[grp['Day'].astype(int) == 365]
                if calendar.isleap(yr) and not last_day.empty:
                    leap_ts = pd.Timestamp(yr, 12, 31, 12)
                    if leap_ts <= end_ts:
                        leap_val = 1000 * last_day.iloc[0][self.inflow3_header] / 24
                        leap_rows.append({'timestamp': leap_ts, 'value': leap_val})

            ts_series = pd.Series(df_country['value'].values, index=df_country['timestamp'])
            ts_series = ts_series[ts_series.index <= end_ts]

            if leap_rows:
                leap_df = pd.DataFrame(leap_rows)
                leap_series = pd.Series(leap_df['value'].values, index=leap_df['timestamp'])
                ts_series = pd.concat([ts_series, leap_series])

            ts_series = ts_series[~ts_series.index.duplicated(keep='first')].sort_index()

            # PECD ships a full 365-row block for every zone whether or not the
            # zone has any run-of-river, so "no run-of-river here" arrives as an
            # empty column rather than as absent rows -- the Nordic zones are the
            # case in point. That is a fact about the technology, not a gap.
            if self._has_no_inflow(ts_series):
                result_df[col_name] = np.nan
                self.unbuilt_nodes.append(
                    (col_name, "the source carries no run-of-river generation for this zone", "info")
                )
                continue

            # Complete the daily grid, then cast to hourly -- same order as the
            # weekly path. A run-of-river river does not stop for exactly one day,
            # so a recorded zero is a gap here just as a blank is.
            ts_series = self._complete_and_report(ts_series, col_name, standard_days, "day")

            result_df[col_name] = self._cast_to_hourly(ts_series, full_index, standard_days)

        return self._drop_empty_columns(
            result_df, "all run-of-river generation came out as zero"
        )

    def process(self) -> pd.DataFrame:
        """
        Executes the full processing pipeline: reading input files, processing the inflow data,
        and combining results.
        
        Returns
        -------
        pd.DataFrame
            Combined DataFrame with reservoir and run-of-river inflow timeseries.
        """
        self.logger.log_status("Reading input files...")
        
        # read_input_csv rejects a file whose numbers did not survive whatever
        # produced it, so the year comparison below cannot meet a string. The
        # bare pd.to_numeric() calls that used to follow each read had no
        # errors= argument and would have raised an unhandled ValueError here.
        weekly_df = self.read_input_csv(self.file_weekly)
        weekly_df = weekly_df[(weekly_df["year"] >= self.start_year) & (weekly_df["year"] <= self.end_year)]

        daily_df = self.read_input_csv(self.file_daily)
        daily_df = daily_df[(daily_df["year"] >= self.start_year) & (daily_df["year"] <= self.end_year)]

        self.logger.log_status("Processing reservoir inflows for all countries...")
        reservoir_all = self._process_reservoir_inflows(weekly_df)

        self.logger.log_status("Processing run-of-river inflows for all countries...")
        ror_all = self._process_ror_inflows(daily_df)

        # Combine the two DataFrames (they share the same hourly index).
        summary_df = pd.concat([reservoir_all, ror_all], axis=1)

        # Set secondary result if needed (None in this case)
        self.secondary_result = None

        self._report_coverage(list(summary_df.columns))

        # Convert to long format: [grid, node, time, value]
        result = summary_df.reset_index(names='time')
        result = result.melt(id_vars=['time'], var_name='node', value_name='value')
        result['grid'] = result['node'].str.split('_').str[1]
        return result[['grid', 'node', 'time', 'value']]