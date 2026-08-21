import os
import calendar
import pandas as pd
import numpy as np
from src.timeseries.processors.base_processor import BaseProcessor
from src.timeseries.timeseries_helpers import complete_native_grid


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

        # The weekly grid every year in range should have: weeks 1..52 a week
        # apart from Jan 4, plus week 53's slot at Dec 28. Anything the source
        # does not supply is interpolated onto this before the hourly cast, so the
        # hourly pass never meets a gap wider than one week.
        standard_weeks = pd.DatetimeIndex(sorted({
            *(pd.Timestamp(year, 1, 4, 12) + pd.Timedelta(7 * (week - 1), unit='D')
              for year in range(self.start_date.year, self.end_date.year + 1)
              for week in range(1, 53)),
            *(pd.Timestamp(year, 12, 28, 12)
              for year in range(self.start_date.year, self.end_date.year + 1)),
        }))
        standard_weeks = standard_weeks[standard_weeks <= end_ts]

        for country in self.country_codes:
            col_name1 = country + self.inflow1_suffix
            col_name2 = country + self.inflow2_suffix

            # Filter for the country's data.
            df_country = weekly_df[weekly_df["zone"] == country].copy()
            if df_country.empty:
                result_df[col_name1] = np.nan
                result_df[col_name2] = np.nan
                continue

            # Sort by year and week, and fill missing values.
            df_country.sort_values(by=['year', 'week'], inplace=True)

            # Anchor each row by its week *number*, not by its position in the
            # frame. Weeks 1..52 sit at Jan 4 12:00 + (w-1) weeks; week 53 is
            # special-cased to Dec 28, one day after week 52 rather than a week
            # after it, because a full week later would be Jan 3 and would collide
            # with the next year's week 1 on Jan 4. That special case is the
            # existing convention and is kept.
            #
            # Position was the wrong key. SE01 1991 has no week 7 and SE02 1985 no
            # weeks 6 or 7, so every later week was placed one or two weeks early
            # for the whole rest of the year -- values attached to the wrong hours,
            # which no value check can see. That is the same failure the t-label
            # checks exist to prevent, one stage earlier.
            years = df_country['year'].astype(int)
            weeks = df_country['week'].astype(int)
            regular = (
                pd.to_datetime(years.astype(str) + '-01-04')
                + pd.Timedelta(12, unit='h')
                + pd.to_timedelta((weeks - 1) * 7, unit='D')
            )
            week53 = pd.to_datetime(years.astype(str) + '-12-28') + pd.Timedelta(12, unit='h')
            anchors = regular.where(weeks <= 52, week53)
            df_country = df_country.assign(_anchor=anchors)
            df_country = df_country[df_country['_anchor'] <= end_ts]
            if df_country.empty:
                result_df[col_name1] = np.nan
                result_df[col_name2] = np.nan
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

            s_inflow1 = anchor_series(self.inflow1_header)
            s_inflow2 = anchor_series(self.inflow2_header)

            s_inflow1 = self._complete_and_report(s_inflow1, col_name1, standard_weeks, "week")
            s_inflow2 = self._complete_and_report(s_inflow2, col_name2, standard_weeks, "week")

            # Reindex to the full hourly index and interpolate missing values.
            # Mechanical now: the weekly grid above has no gap wider than a week.
            s_inflow1 = s_inflow1.reindex(full_index).interpolate(limit=84, limit_direction='both')
            s_inflow2 = s_inflow2.reindex(full_index).interpolate(limit=84, limit_direction='both')

            result_df[col_name1] = s_inflow1
            result_df[col_name2] = s_inflow2

        # Drop empty columns, return result_df
        result_df = result_df.loc[:, result_df.sum() != 0]
        return result_df

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

            # Filter for the country's data.
            df_country = daily_df[daily_df["zone"] == country].copy()
            if df_country.empty:
                result_df[col_name] = np.nan
                continue

            # No fillna(0): a blank day and a recorded zero are different things,
            # and complete_native_grid below is what decides between them.
            df_country.sort_values(by=['year', 'Day'], inplace=True)

            # Compute the day offset within each year.
            df_country['day_offset'] = df_country.groupby('year').cumcount()

            # Compute the timestamp for each row.
            df_country['timestamp'] = (
                pd.to_datetime(df_country['year'].astype(str), format='%Y') +
                pd.to_timedelta(df_country['day_offset'], unit='D') +
                pd.Timedelta(12, unit="h")
            )

            # Compute the flow values.
            df_country['value'] = 1000 * df_country[self.inflow3_header] / 24

            # Handle leap years: add an extra record for December 31 at 12:00 if needed.
            leap_rows = []
            for yr, grp in df_country.groupby('year'):
                if calendar.isleap(yr) and len(grp) > 364:
                    leap_ts = pd.Timestamp(yr, 12, 31, 12)
                    if leap_ts <= end_ts:
                        leap_val = 1000 * grp.iloc[364][self.inflow3_header] / 24
                        leap_rows.append({'timestamp': leap_ts, 'value': leap_val})

            ts_series = pd.Series(df_country['value'].values, index=df_country['timestamp'])
            ts_series = ts_series[ts_series.index <= end_ts]

            if leap_rows:
                leap_df = pd.DataFrame(leap_rows)
                leap_series = pd.Series(leap_df['value'].values, index=leap_df['timestamp'])
                ts_series = pd.concat([ts_series, leap_series])

            ts_series = ts_series[~ts_series.index.duplicated(keep='first')].sort_index()

            # Complete the daily grid, then cast to hourly -- same order as the
            # weekly path. A run-of-river river does not stop for exactly one day,
            # so a recorded zero is a gap here just as a blank is.
            ts_series = self._complete_and_report(ts_series, col_name, standard_days, "day")

            ts_series = ts_series.reindex(full_index)
            ts_series = ts_series.interpolate(limit=12, limit_direction='both')

            result_df[col_name] = ts_series

        # Drop empty columns, return result_df
        result_df = result_df.loc[:, result_df.sum() != 0]
        return result_df

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

        self.logger.log_status("Inflow time series built.", level="info")

        # Convert to long format: [grid, node, time, value]
        result = summary_df.reset_index(names='time')
        result = result.melt(id_vars=['time'], var_name='node', value_name='value')
        result['grid'] = result['node'].str.split('_').str[1]
        return result[['grid', 'node', 'time', 'value']]