# src/processors/hydro_storage_limits_MAF2019.py

import os
import pandas as pd
from src.processors.base_processor import BaseProcessor


class hydro_storage_limits_MAF2019(BaseProcessor):
    """
    Class to process reservoir state limits data (reservoir, pump open cycle, pumped closed cycle)
    for a parametrized date range (start_date to end_date).

    The weekly reservoir level CSV contains identical min/max values for every year.
    A full multi-year timeseries is built by replicating the weekly pattern for each
    calendar year in the range and interpolating to hourly resolution.
    Large jumps at year boundaries are detected, logged, and smoothed.

    Parameters:
        input_folder (str): relative location of input files.
        country_codes (list): List of country codes.
        start_year (int): First climate year to include (e.g., 1982).
        end_year (int): Last climate year to include (e.g., 2016).

    Returns:
        main_result (pd.DataFrame): A MultiIndex DataFrame with time series data for each node
                                     with datetime index from start_date to end_date.
                                     Columns named by country (e.g., AT00_reservoir, AT00_psOpen, CH00_reservoir)
        secondary_result (pd.DataFrame): A DataFrame listing valid (node, boundary type, average_value) combinations
    """

    def __init__(self, **kwargs):
        # Initialize base class
        super().__init__(**kwargs)

        # List of required parameters
        required_params = [
            'input_folder',
            'country_codes',
            'start_year',
            'end_year',
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

        # Tracks which (col, boundary_type) series had year-boundary jumps smoothed.
        # Populated by _smooth_year_boundaries; summarised at the end of process().
        self.smoothed_series: set = set()

        # Input file paths.
        self.levels_file = os.path.join(self.input_folder, 'PECD-hydro-weekly-reservoir-levels.csv')
        self.capacities_file = os.path.join(self.input_folder, 'PECD-hydro-capacities.csv')

    def _fill_weekly_data_for_year(self, lowerBound, upperBound, weekly_df,
                                   year, country, suffix, cap_key, lower_col, upper_col, df_capacities):
        """
        Fill weekly anchor timestamps in lowerBound/upperBound for a single calendar year.

        Weeks are placed at Jan 4 noon + 7*i days for i=0..51 (matching the PECD convention).
        Anchor points outside [start_date, end_date] are skipped. weekly_df must be indexed
        0..N with one row per week; only the first 52 rows are used.
        """
        fourthday = pd.Timestamp(year, 1, 4, 12)
        cap_value = df_capacities.at[cap_key, 'value']
        col = country + suffix

        for i in range(min(52, len(weekly_df))):
            t = fourthday + pd.DateOffset(days=7 * i)
            if t > self.end_date:
                break
            if t < self.start_date:
                continue
            lowerBound.at[t, col] = 1000 * weekly_df.at[i, lower_col] * cap_value
            upperBound.at[t, col] = 1000 * weekly_df.at[i, upper_col] * cap_value

    def _smooth_year_boundaries(self, bound_df, col, boundary_type, typical_multiplier=3.0):
        """
        Detect and smooth large value jumps at year boundaries in a sparse weekly series.

        For each year boundary in the date range, computes the jump between the last weekly
        anchor of year Y and the first of year Y+1. If the jump exceeds typical_multiplier
        times the typical mid-year week-to-week variation, logs an info message and linearly
        blends the two boundary-week values toward each other.
        """
        if col not in bound_df.columns:
            return
        series = bound_df[col].dropna()
        if len(series) < 4:
            return

        for year in range(self.start_date.year, self.end_date.year):
            year_vals = series[series.index.year == year]
            next_year_vals = series[series.index.year == year + 1]

            if len(year_vals) < 3 or len(next_year_vals) < 2:
                continue

            val_prev = year_vals.iloc[-2]    # week 51 of year Y
            val_last = year_vals.iloc[-1]    # week 52 of year Y
            val_first = next_year_vals.iloc[0]   # week 1 of year Y+1
            val_second = next_year_vals.iloc[1]  # week 2 of year Y+1

            boundary_jump = abs(val_last - val_first)

            # Typical week-to-week change from inner weeks (skip first and last 2)
            inner = year_vals.iloc[2:-2]
            if len(inner) < 2:
                continue
            typical_change = inner.diff().abs().mean()

            if typical_change > 0 and boundary_jump > typical_multiplier * typical_change:
                self.smoothed_series.add((col, boundary_type))
                # Blend boundary weeks linearly from val_prev toward val_second
                t_last = year_vals.index[-1]
                t_first = next_year_vals.index[0]
                bound_df.at[t_last, col] = val_prev + (val_second - val_prev) / 3
                bound_df.at[t_first, col] = val_prev + 2 * (val_second - val_prev) / 3

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

    def process_country(self, country, df_country, df_capacities,
                        minvariable_header, maxvariable_header,
                        minvariable, maxvariable,
                        suffix_reservoir):
        """
        Process data for a single country (non-Norway) across the full date range.

        The weekly pattern from the CSV is identical for all years; it is replicated
        for each calendar year in [start_date.year, end_date.year].
        """
        date_index = pd.date_range(self.start_date, self.end_date, freq='60 min')
        df_lowerBound = pd.DataFrame(index=date_index)
        df_upperBound = pd.DataFrame(index=date_index)

        cap_key_reservoir = (country, 'Reservoir', 'Reservoir capacity (GWh)')
        if not df_country.empty and not df_country.iloc[:, 3].isna().any():
            df_country = df_country.sort_values(by=['year', 'week']).reset_index(drop=True)
            available_years = df_country['year'].unique()
            if len(available_years) > 0:
                # Weekly data is identical across years; use the first year's rows
                df_week_data = df_country[df_country['year'] == available_years[0]].reset_index(drop=True)
                col = country + suffix_reservoir
                for year in range(self.start_date.year, self.end_date.year + 1):
                    self._fill_weekly_data_for_year(
                        df_lowerBound, df_upperBound, df_week_data,
                        year, country, suffix_reservoir, cap_key_reservoir,
                        minvariable_header, maxvariable_header, df_capacities
                    )
                self._smooth_year_boundaries(df_lowerBound, col, minvariable)
                self._smooth_year_boundaries(df_upperBound, col, maxvariable)

        df_lowerBound.interpolate(inplace=True, limit_direction='both')
        df_upperBound.interpolate(inplace=True, limit_direction='both')

        result_df = self.merge_bounds(df_lowerBound, df_upperBound, minvariable, maxvariable)
        return result_df

    def process_norway_area(self, country, filename,
                            df_capacities,
                            minvariable_header_norway, maxvariable_header_norway,
                            minvariable, maxvariable,
                            suffix_open):
        """
        Process Norway-specific area data using an Excel input file across the full date range.

        The weekly pattern from the Excel is replicated for each calendar year.
        """
        date_index = pd.date_range(self.start_date, self.end_date, freq='60 min')
        df_lowerBound = pd.DataFrame(index=date_index)
        df_upperBound = pd.DataFrame(index=date_index)

        try:
            df = pd.read_excel(
                os.path.normpath(filename),
                sheet_name='Pump storage - Open Loop',
                usecols="L,M",
                names=[minvariable_header_norway, maxvariable_header_norway],
                skiprows=12
            )
        except Exception as e:
            self.logger.log_status(f"Error reading Norway input Excel: {e}", level="warn")
            return None

        cap_key_psOpen = (country, 'Pump Storage - Open Loop', 'Cumulated (upper or head) reservoir capacity (GWh)')
        col = country + suffix_open
        for year in range(self.start_date.year, self.end_date.year + 1):
            self._fill_weekly_data_for_year(
                df_lowerBound, df_upperBound, df,
                year, country, suffix_open, cap_key_psOpen,
                minvariable_header_norway, maxvariable_header_norway, df_capacities
            )
        self._smooth_year_boundaries(df_lowerBound, col, minvariable)
        self._smooth_year_boundaries(df_upperBound, col, maxvariable)

        df_lowerBound.interpolate(inplace=True, limit_direction='both')
        df_upperBound.interpolate(inplace=True, limit_direction='both')

        result_df = self.merge_bounds(df_lowerBound, df_upperBound, minvariable, maxvariable)
        return result_df

    def process(self) -> pd.DataFrame:
        """
        Main processing logic that executes the hydro storage data processing pipeline.

        This function performs the following steps:
        1. Reads and validates reservoir levels data from CSV files
        2. Reads capacity data for different zones and types
        3. Creates a time-series framework for the full date range
        4. For each calendar year in the range, replicates the weekly pattern and
           detects/smooths large jumps at year boundaries
        5. Interpolates weekly anchor points to hourly resolution
        6. Creates ts_hydro_storage_limits_df of valid (node, boundary type) combinations

        Returns:
            pd.DataFrame: A MultiIndex DataFrame with time series data for each node
                         with datetime index from start_date to end_date
        """
        # --- Read and prepare the input data
        self.logger.log_status("Reading input files...")
        try:
            df_levels = pd.read_csv(self.levels_file)
        except Exception as e:
            self.logger.log_status(f"Error reading hydro storage levels CSV '{self.levels_file}': {e}", level="warn")
            return pd.DataFrame()

        df_levels["year"] = pd.to_numeric(df_levels["year"])
        df_levels["week"] = pd.to_numeric(df_levels["week"])

        try:
            df_capacities = pd.read_csv(self.capacities_file)
        except Exception as e:
            self.logger.log_status(f"Error reading hydro capacities CSV '{self.capacities_file}': {e}", level="warn")
            return pd.DataFrame()
        df_capacities = df_capacities.set_index(["zone", "type", "variable"])

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
            if country in self.norway_codes:
                # Special handling for Norway using Excel data
                filename = os.path.join(self.input_folder, f"{self.file_first}{country}{self.file_last}")
                result_df = self.process_norway_area(country, filename, df_capacities,
                                                     self.minvariable_header_norway, self.maxvariable_header_norway,
                                                     self.minvariable, self.maxvariable,
                                                     self.suffix_open)
            else:
                # Standard processing for other countries
                df_country = df_levels[df_levels["zone"] == country]
                if df_country.empty:
                    continue
                result_df = self.process_country(country, df_country, df_capacities,
                                                 self.minvariable_header, self.maxvariable_header,
                                                 self.minvariable, self.maxvariable,
                                                 self.suffix_reservoir)

            # Merge country results into summary dataframe
            if result_df is not None:
                result_df.index = result_df.index.set_names(['time', 'param_gnBoundaryTypes'])
                summary_df = summary_df.join(result_df, how='left')

        # Remove columns with no data (sum equals zero)
        summary_df = summary_df.loc[:, summary_df.sum() > 0]

        # Log a summary of any year-boundary smoothing that was applied
        if self.smoothed_series:
            series_list = ', '.join(f"{col} ({btype})" for col, btype in sorted(self.smoothed_series))
            self.logger.log_status(
                f"Smoothed year-to-year jumps in {len(self.smoothed_series)} series: {series_list}.",
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
