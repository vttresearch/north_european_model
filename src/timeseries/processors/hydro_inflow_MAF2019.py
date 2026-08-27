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
    Hydro inflow for the three inflow-bearing types: reservoir, open-loop pump
    storage, and run-of-river.

    Energy per period becomes average power in MWh/h, which is what ts_influx
    expects: GWh per week for reservoir and pump storage, GWh per day for
    run-of-river. No installed capacity is involved anywhere -- these are
    inflows, not capacity factors.

    docs/hydro.md is the documentation and carries the reasoning: why week 53 is
    not read, why a recorded zero counts as a gap, and which nodes deliberately
    get no series.

    Parameters:
        input_folder (str): relative location of input files.
        country_codes (list): List of country codes.
        start_year (int): First climate year to include (e.g., 1982).
        end_year (int): Last climate year to include (e.g., 2016).
        df_nodedata (pd.DataFrame): merged nodedata, via requires_source_data.
    """

    #: Inflow is water arriving, which cannot be negative. No value_range: the
    #: plausible maximum is a property of the catchment, not of the parameter.
    value_sign = "non_negative"

    #: nodedata answers which hydro nodes the model actually has. Without it this
    #: processor reported on the cross product of every country code and every
    #: hydro type, burying the few nodes that exist. See _model_hydro_nodes.
    requires_source_data = ('nodedata',)

    #: How far a week 52 -> week 1 change may exceed the node's own 95th-percentile
    #: within-year weekly change before the build says so. The seam is one sample
    #: of the same weekly distribution, so a plain "above the 95th percentile" test
    #: fires about as often as chance predicts and reports nothing. At three times
    #: it names the genuine outliers in the shipped data and stays quiet otherwise.
    YEAR_CHANGE_OUTLIER_MULTIPLIER = 3.0

    #: Stated exceptions to "a run longer than one slot needs a decision", each a
    #: decision already taken rather than a silence. A gap not listed here warns
    #: and someone decides; see the rule section of docs/hydro.md.
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
        # The two Norwegian entries rest on a check made once, recorded in the
        # Norwegian pump-storage section of docs/hydro.md and deliberately not
        # repeated at runtime: the zeros are dropped values in genuinely dry
        # weeks rather than a frozen catchment, because a frozen catchment would
        # have shown in NOM1, which records no zero in 35 years.
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
        super().__init__(**kwargs_processor)

        required_params = [
            'input_folder',
            'country_codes',
            'start_year',
            'end_year',
            'df_nodedata',
        ]

        missing_params = [param for param in required_params if param not in kwargs_processor]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        self.input_folder = kwargs_processor['input_folder']
        self.country_codes = kwargs_processor['country_codes']
        self.start_year = kwargs_processor['start_year']
        self.end_year = kwargs_processor['end_year']
        self.df_nodedata = kwargs_processor['df_nodedata']

        self.start_date = pd.Timestamp(f"{self.start_year}-01-01")
        self.end_date   = pd.Timestamp(f"{self.end_year}-12-31 23:00")

        self.file_weekly = os.path.join(self.input_folder, 'PECD-hydro-weekly-inflows.csv')
        self.file_daily = os.path.join(self.input_folder, 'PECD-hydro-daily-ror-generation.csv')

        self.inflow1_header = 'Cumulated inflow into reservoirs per week in GWh'
        self.inflow2_header = 'Cumulated NATURAL inflow into the pump-storage reservoirs per week in GWh'
        self.inflow3_header = 'Run of River Hydro Generation in GWh per day'
        self.inflow1_suffix = '_reservoir'
        self.inflow2_suffix = '_psOpen'
        self.inflow3_suffix = '_ror'

        # Nodes the run asked for and this processor did not build, as
        # (node, reason, level) triples for _report_coverage to say once.
        self.unbuilt_nodes = []

        # Node names only: what the run says about these is how many there were,
        # and nothing about any one of them. See _report_repairs.
        self.repaired_nodes = set()
        self.year_change_outliers = set()

        self.model_nodes = self._model_hydro_nodes()

    def _model_hydro_nodes(self):
        """The inflow-bearing hydro nodes this run's model actually contains.

        An empty answer means nodedata could not be read rather than that the
        model has no hydro, so it falls back to building every country and type.
        Failing open keeps a malformed workbook from silently deleting the hydro
        fleet; the warning says which of the two happened.
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
        size. The storage limits are the opposite case and do require a size --
        see "Which nodes get built" in docs/hydro.md.
        """
        return not self.model_nodes or node in self.model_nodes

    def _report_grid(self, report, unit):
        """Note what the grid needed, and warn only when it is still not whole.

        A repaired single-slot gap is counted for _report_repairs rather than
        described here: it is the rule working. A run that is still open is the
        opposite -- it asks someone about one node, so it stays where it happens
        and keeps the series size, which is what that decision turns on.
        """
        if report.n_missing == 0:
            return
        if report.ok:
            self.repaired_nodes.add(report.label)
            return
        self.logger.log_status(
            f"{report.label}: {report.n_left} {unit}(s) have no usable value and were left "
            f"empty, the longest run being {report.longest_run_left} {unit}(s) from "
            f"{report.first_left:%Y-%m-%d} in a {report.twh_per_year:.2f} TWh/year series. "
            f"Only single-{unit} gaps are filled automatically; this one needs a decision, "
            f"recorded in ACCEPTED_LONG_RUNS.",
            level="warn"
        )

    def _process_reservoir_inflows(self, weekly_df):
        """
        Processes the weekly (reservoir and pump-storage) data for each country.
        Returns one combined DataFrame with appropriately suffixed columns.
        """
        full_index = pd.date_range(self.start_date, self.end_date, freq='60min')
        end_ts = pd.Timestamp(self.end_date)
        result_df = pd.DataFrame(index=full_index)

        # The weekly grid for every year in range: weeks 1..52, a week apart from
        # Jan 4. The source is completed onto this before the hourly cast, so the
        # hourly pass never meets a gap wider than the year change itself.
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

            # Week 53 is dropped rather than read: it is not a week but the one
            # or two days left over, and the source says different things about
            # that remainder in different zones. Reading it put a one-day cliff
            # at every New Year -- see the year change section of docs/hydro.md.
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

            df_country.sort_values(by=['year', 'week'], inplace=True)

            # Anchor each row by its week *number*, not by its position in the
            # frame: weeks 1..52 sit at Jan 4 12:00 + (w-1) weeks. A zone-year
            # missing a week would otherwise place every later week one week
            # early -- right values on wrong hours, which no value check can see,
            # and the same failure the t-label checks exist to prevent.
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

            # No fillna(0) here. A blank cell and a recorded zero must stay
            # distinguishable all the way to complete_native_grid, which is what
            # decides between them.
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
                self._note_year_change_outlier(series, col_name)
                result_df[col_name] = self._cast_to_hourly(series, full_index, standard_weeks)

        return self._drop_empty_columns(result_df, "all inflow came out as zero")

    @staticmethod
    def _has_no_inflow(series):
        """True when the source says nothing about this zone and series.

        PECD ships a row for every zone whether or not the zone has a plant of
        that type, so "no reservoir in Belgium" arrives as an empty column or as
        thirty-five years of zeros. Neither is a gap, and sending them through
        complete_native_grid would ask someone to decide about a technology that
        does not exist. Answered here instead, once, at info level.
        """
        values = series.to_numpy(dtype=float)
        return not np.any(np.isfinite(values) & (values != 0))

    def _cast_to_hourly(self, weekly_series, full_index, standard_weeks):
        """Spread a completed weekly series over the hourly index.

        The rule is "fill between two adjacent weekly anchors that both have a
        value, and nowhere else", which needs no interpolation limit -- and no
        limit could express it. The spans differ: 52 weeks from Jan 4 reach only
        day 361, so a year change is 192 h in a common year and **216 h after a
        leap one**, and no single number both bridges 216 h and refuses the 336 h
        that a two-week gap leaves. Refusing that longer span is the point, since
        ``complete_native_grid`` deliberately leaves a long run alone and an
        unlimited interpolation would put the invention back at the hourly stage.

        The two ends are a separate question, and the answer is not "leave them".
        Linear interpolation does not extrapolate, so the hours before the first
        anchor and after the last would stay empty with nothing downstream saying
        so: ``find_time_axis_defects`` checks that the rows are there rather than
        what is in them, and ``prepare_values_for_gdx`` turns the blanks into
        zeros without a word. So the nearest week is carried outward.
        """
        if not len(standard_weeks):
            return weekly_series.reindex(full_index)

        on_grid = weekly_series.reindex(standard_weeks)
        hourly = on_grid.reindex(full_index).interpolate(limit_area='inside')

        # An hour is legitimate only if the anchors on both sides of it carry a
        # value: ffill answers "was the previous one filled", bfill "is the next".
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
        zero just like a genuinely all-zero one, but they are different
        situations. The caller records the first as information; this one is a
        warning, because a zone with rows that yields no inflow is inconsistent
        with itself.

        "Carries nothing" is `sum() > 0` rather than `sum() != 0`, and the same
        test decides both what is reported and what is dropped. Inflow is
        declared non-negative, so a negative total is not a column with content:
        it is a column whose content is wrong, and reporting it as empty while
        writing it anyway would be the worst of both.
        """
        keep = result_df.sum() > 0
        for node in result_df.columns[~keep]:
            if not any(node == n for n, _, _ in self.unbuilt_nodes):
                self.unbuilt_nodes.append((node, reason, "warn"))
        return result_df.loc[:, keep]

    def _note_year_change_outlier(self, weekly_series, label):
        """Mark a node whose year change is a gross outlier in its own weeks.

        Weeks 52 and 1 are seven days apart like any other pair, so the change
        between them is one sample of the same weekly distribution and is usually
        unremarkable. Nothing is smoothed here: a large change at New Year is
        weather, and inventing a gentler one would be a lie. The node is marked so
        that a data refresh which makes the seam anomalous shows up in the count
        rather than being absorbed, which is the job ACCEPTED_LONG_RUNS does for
        gaps.
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
        for earlier, later in zip(years, years[1:]):
            if later != earlier + 1:
                continue
            jump = abs(by_year[later].iloc[0] - by_year[earlier].iloc[-1])
            if jump > threshold:
                self.year_change_outliers.add(label)
                return

    def _report_repairs(self):
        """One short line of counts for everything the rules handled on their own.

        A single-slot gap filled, a run interpolated because ACCEPTED_LONG_RUNS
        says so, and a large year change left alone are all outcomes of decisions
        already taken, and on the shipped data they are the same ten nodes every
        run. Named and reasoned about per node they were ten paragraphs -- longer
        than everything else the processor says, and the surest way to stop a
        reader noticing the line that is new.

        So counts, and nothing else. Which nodes and why is in ACCEPTED_LONG_RUNS
        and in docs/hydro.md, where it stays put between runs. A gap that is
        *not* covered by a rule is the opposite case and keeps its own warning,
        with the node and the series size in it.
        """
        parts = []
        if self.repaired_nodes:
            parts.append(f"Gaps interpolated at {len(self.repaired_nodes)} node(s)")
        if self.year_change_outliers:
            parts.append(
                f"{len(self.year_change_outliers)} large year change(s) left as they are"
            )
        if parts:
            self.logger.log_status(f"{', '.join(parts)}.", level="info")

    def _report_coverage(self, built_nodes):
        """
        Say once what was built and what was not, instead of leaving it to be noticed.

        Split by whether the user can do anything about it. A zone the source data
        does not cover is information -- most countries in a run have no reservoir.
        A zone that has rows but yields nothing is a warning: something in the
        input data is inconsistent with itself.
        """
        self.logger.log_status(
            f"Inflow built for {len(built_nodes)} node(s).",
            level="info"
        )

        for level in ("warn", "info"):
            entries = sorted((n, r) for n, r, lvl in self.unbuilt_nodes if lvl == level)
            if not entries:
                continue
            # Grouped by reason, which is what a reader acts on and what the
            # nodes usually share. Names stay in full: a node reaching GAMS with
            # no inflow is exactly what someone has to go and look up.
            by_reason = {}
            for node, reason in entries:
                by_reason.setdefault(reason, []).append(node)
            detail = '; '.join(
                f"{', '.join(nodes)}: {reason}"
                for reason, nodes in sorted(by_reason.items())
            )
            self.logger.log_status(
                f"No inflow for {len(entries)} node(s) -- {detail}. These reach "
                f"GAMS with no inflow at all.",
                level=level
            )

    def _complete_and_report(self, series, label, standard_index, unit):
        """Complete one native-resolution series and note what that took.

        Single-slot gaps are repaired here. A longer run is left alone and warned
        about unless it appears in ACCEPTED_LONG_RUNS: bridging two or more slots
        is invention rather than repair, so a person decides. Either repair only
        adds the node to the count _report_repairs states; the warning is the one
        that describes itself.
        """
        filled, report = complete_native_grid(
            series, standard_index, label=label, zero_is_missing=True
        )
        if not report.ok and label in self.ACCEPTED_LONG_RUNS:
            filled = filled.interpolate(method='time', limit_area='inside').ffill()
            self.repaired_nodes.add(label)
            return filled.dropna()
        self._report_grid(report, unit)
        return filled.dropna()

    def _process_ror_inflows(self, daily_df):
        """
        Processes daily run-of-river data for each country.
        Returns one combined DataFrame with appropriately suffixed columns.
        """
        full_index = pd.date_range(self.start_date, self.end_date, freq='60min')
        end_ts = pd.to_datetime(self.end_date)
        result_df = pd.DataFrame(index=full_index)

        # Every calendar day at noon, where the anchors land. Leap years get
        # their 366th: the source ships 365 rows regardless, and the leap block
        # below re-emits Dec 31 to make up the difference.
        standard_days = pd.date_range(
            pd.Timestamp(self.start_date.year, 1, 1, 12), end_ts, freq='D'
        )

        for country in self.country_codes:
            col_name = country + self.inflow3_suffix
            if not self._is_in_model(col_name):
                continue

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

            # Anchor by Day *number*, not row position, for the same reason the
            # weekly path does. The shipped file is a clean 1..365 block in every
            # zone-year, so this changes nothing today and is here so that a data
            # refresh cannot.
            df_country['timestamp'] = (
                pd.to_datetime(df_country['year'].astype(str), format='%Y') +
                pd.to_timedelta(df_country['Day'].astype(int) - 1, unit='D') +
                pd.Timedelta(12, unit="h")
            )

            df_country['value'] = 1000 * df_country[self.inflow3_header] / 24

            # The source ships 365 days whatever the calendar, so in a leap year
            # day 365 lands on Dec 30 and Dec 31 has no anchor of its own. Give it
            # day 365's value again, the convention elec_demand_TYNDP2024 follows.
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

            # PECD ships a full 365-row block for every zone whether or not it has
            # any run-of-river, so "none here" arrives as an empty column rather
            # than as absent rows. That is a fact about the technology, not a gap.
            if self._has_no_inflow(ts_series):
                result_df[col_name] = np.nan
                self.unbuilt_nodes.append(
                    (col_name, "the source carries no run-of-river generation for this zone", "info")
                )
                continue

            # Complete the daily grid, then cast to hourly -- same order as the
            # weekly path. A river does not stop for exactly one day, so a
            # recorded zero is a gap here just as a blank is.
            ts_series = self._complete_and_report(ts_series, col_name, standard_days, "day")

            result_df[col_name] = self._cast_to_hourly(ts_series, full_index, standard_days)

        return self._drop_empty_columns(
            result_df, "all run-of-river generation came out as zero"
        )

    def process(self) -> pd.DataFrame:
        """
        Read the two PECD files, build all three inflow types, and combine them.

        Returns
        -------
        pd.DataFrame
            Long format ['grid', 'node', 'time', 'value'], hourly, in MWh/h.
        """
        self.logger.log_status("Reading input files...")

        # read_input_csv rejects a file whose numbers did not survive whatever
        # produced it, so the year comparisons below cannot meet a string.
        weekly_df = self.read_input_csv(self.file_weekly)
        weekly_df = weekly_df[(weekly_df["year"] >= self.start_year) & (weekly_df["year"] <= self.end_year)]

        daily_df = self.read_input_csv(self.file_daily)
        daily_df = daily_df[(daily_df["year"] >= self.start_year) & (daily_df["year"] <= self.end_year)]

        self.logger.log_status("Processing reservoir inflows for all countries...")
        reservoir_all = self._process_reservoir_inflows(weekly_df)

        self.logger.log_status("Processing run-of-river inflows for all countries...")
        ror_all = self._process_ror_inflows(daily_df)

        # Both were built on the same hourly index.
        summary_df = pd.concat([reservoir_all, ror_all], axis=1)

        self._report_repairs()
        self._report_coverage(list(summary_df.columns))

        # Long format. The grid is the node name's own suffix, which is what the
        # three inflow*_suffix constants put there.
        result = summary_df.reset_index(names='time')
        result = result.melt(id_vars=['time'], var_name='node', value_name='value')
        result['grid'] = result['node'].str.split('_').str[1]
        return result[['grid', 'node', 'time', 'value']]