# src/processors/hydro_storage_limits_MAF2019.py

import os
import pandas as pd
from src.processors.base_processor import BaseProcessor


class hydro_storage_limits_MAF2019(BaseProcessor):
    """
    Class to process reservoir state limits data (reservoir, pump open cycle, pumped closed cycle)
    for a single non-leap year (2015).

    Parameters:
        input_folder (str): relative location of input files.
        country_codes (list): List of country codes.

    Returns:
        main_result (pd.DataFrame): A MultiIndex DataFrame with time series data for each node 
                                     with datetime index from 2015-01-01 00:00:00 to 2015-12-31 23:00:00
                                     Columns named by country (e.g., AT00_reservoir, AT00_psOpen, CH00_reservoir)
        secondary_result (pd.DataFrame): A DataFrame listing valid (node, boundary type, average_value) combinations
    """

    def __init__(self, **kwargs):
        # Initialize base class
        super().__init__(**kwargs)
        
        # List of required parameters
        required_params = [
            'input_folder', 
            'country_codes'
        ]

        # Check if all required parameters are present
        missing_params = [param for param in required_params if param not in kwargs]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Unpack parameters
        self.input_folder = kwargs['input_folder']
        self.country_codes = kwargs['country_codes']

        # Single non-leap year (2015)
        self.reference_year = 2015
        self.start_date = '2015-01-01 00:00:00'
        self.end_date = '2015-12-31 23:00:00'

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

        # Input file paths.
        self.levels_file = os.path.join(self.input_folder, 'PECD-hydro-weekly-reservoir-levels.csv')
        self.capacities_file = os.path.join(self.input_folder, 'PECD-hydro-capacities.csv')

    def fill_weekly_data(self, lowerBound, upperBound, weekly_df,
                         country, suffix, cap_key, lower_col, upper_col, df_capacities):
        """
        Loop through the weekly data (weekly_df) and fill in the
        lower and upper bound DataFrames for the reference year (2015).
        The capacity value is looked up from df_capacities using cap_key.
        """
        # Use reference year 2015 for all calculations
        fourthday = pd.Timestamp(self.reference_year, 1, 4) + pd.DateOffset(hours=12)
        
        for i in weekly_df.index:
            t = fourthday + pd.DateOffset(days=7 * i)
            
            # For weeks 0 to 51 (if within year bounds)
            if t <= pd.Timestamp(self.end_date) and i < 52:
                cap_value = df_capacities.at[cap_key, 'value']
                lowerBound.at[t, country + suffix] = 1000 * weekly_df.at[i, lower_col] * cap_value
                upperBound.at[t, country + suffix] = 1000 * weekly_df.at[i, upper_col] * cap_value
            # For week 52 (if present), use the value at index 51
            elif i == 52 and 51 in weekly_df.index:
                t = pd.Timestamp(self.reference_year, 12, 28) + pd.DateOffset(hours=12)
                cap_value = df_capacities.at[cap_key, 'value']
                lowerBound.at[t, country + suffix] = 1000 * weekly_df.at[51, lower_col] * cap_value
                upperBound.at[t, country + suffix] = 1000 * weekly_df.at[51, upper_col] * cap_value

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
                        suffix_reservoir, suffix_open, suffix_closed):
        """
        Process the data for a single country (non-Norway) using the provided levels
        and capacities DataFrames for the reference year (2015).
        """
        # Create an hourly datetime index for the reference year (2015)
        date_index = pd.date_range(self.start_date, self.end_date, freq='60 min')
        df_lowerBound = pd.DataFrame(index=date_index)
        df_upperBound = pd.DataFrame(index=date_index)
        
        cap_key_reservoir = (country, 'Reservoir', 'Reservoir capacity (GWh)')
        # If detailed level data is available and has no NaNs in the fourth column
        if not df_country.empty and not df_country.iloc[:, 3].isna().any():
            df_country = df_country.sort_values(by=['year', 'week']).reset_index(drop=True)
            # Take data from any available year (they're all identical)
            # Just use the first year's data
            available_years = df_country['year'].unique()
            if len(available_years) > 0:
                df_year = df_country[df_country["year"] == available_years[0]].copy()
                df_year = df_year.reset_index(drop=True)
                self.fill_weekly_data(df_lowerBound, df_upperBound, df_year,
                                      country, suffix_reservoir, cap_key_reservoir, 
                                      minvariable_header, maxvariable_header, df_capacities)

        # Interpolate to fill in missing hourly data
        df_lowerBound.interpolate(inplace=True, limit=84, limit_direction='both')
        df_upperBound.interpolate(inplace=True, limit=84, limit_direction='both')
       
        result_df = self.merge_bounds(df_lowerBound, df_upperBound, minvariable, maxvariable)
        return result_df

    def process_norway_area(self, country, filename, 
                            df_capacities,
                            minvariable_header_norway, maxvariable_header_norway, 
                            minvariable, maxvariable, 
                            suffix_open, suffix_closed):
        """
        Process Norway-specific area data using an Excel input file for the reference year (2015).
        """
        # Create an hourly datetime index for the reference year (2015)
        date_index = pd.date_range(self.start_date, self.end_date, freq='60 min')
        df_lowerBound = pd.DataFrame(index=date_index)
        df_upperBound = pd.DataFrame(index=date_index)
        
        # Read the Excel data
        try:
            df = pd.read_excel(
                os.path.normpath(filename),
                sheet_name='Pump storage - Open Loop',
                usecols="L,M",
                names=[minvariable_header_norway, maxvariable_header_norway],
                skiprows=12
            )
        except Exception as e:
            self.log(f"Error reading Norway input Excel: {e}", level="warn")
            return None
        
        # Fill weekly data for the reference year
        cap_key_psOpen = (country, 'Pump Storage - Open Loop', 'Cumulated (upper or head) reservoir capacity (GWh)')
        self.fill_weekly_data(df_lowerBound, df_upperBound, df,
                              country, suffix_open, cap_key_psOpen,
                              minvariable_header_norway, maxvariable_header_norway, df_capacities)
        
        df_lowerBound.interpolate(inplace=True, limit=84, limit_direction='both')
        df_upperBound.interpolate(inplace=True, limit=84, limit_direction='both')
               
        result_df = self.merge_bounds(df_lowerBound, df_upperBound, minvariable, maxvariable)
        return result_df

    def process(self) -> pd.DataFrame:
        """
        Main processing logic that executes the hydro storage data processing pipeline.

        This function performs the following steps:
        1. Reads and validates reservoir levels data from CSV files
        2. Reads capacity data for different zones and types
        3. Creates a time-series framework for the reference year (2015)
        4. Processes each country's data (with special handling for Norway)
        5. Creates ts_hydro_storage_limits_df of valid (node, boundary type) combinations

        Returns:
            pd.DataFrame: A MultiIndex DataFrame with time series data for each node
                         with datetime index for 2015
        """
        # --- Read and prepare the input data
        # Read the levels CSV file
        self.log("Reading input files...")
        try:
            df_levels = pd.read_csv(self.levels_file)
        except Exception as e:
            self.log(f"Error reading hydro storage levels CSV '{self.levels_file}': {e}", level="warn")
            return pd.DataFrame()

        # Convert columns to numeric (data from any year will work since they're identical)
        df_levels["year"] = pd.to_numeric(df_levels["year"])
        df_levels["week"] = pd.to_numeric(df_levels["week"])

        # Read the capacities CSV file
        try:
            df_capacities = pd.read_csv(self.capacities_file)
        except Exception as e:
            self.log(f"Error reading hydro capacities CSV '{self.capacities_file}': {e}", level="warn")
            return pd.DataFrame()
        df_capacities = df_capacities.set_index(["zone", "type", "variable"])

        # Create a summary DataFrame with a MultiIndex (time, param_gnBoundaryTypes)
        # for the reference year (2015)
        idx = pd.MultiIndex.from_product(
            [pd.date_range(self.start_date, self.end_date, freq='60 min'),
             [self.minvariable, self.maxvariable]],
            names=['time', 'param_gnBoundaryTypes']
        )
        summary_df = pd.DataFrame(index=idx)

        # --- build country level df
        self.log("Building country level timeseries...")
        
        # Process each country
        for country in self.country_codes:
            if country in self.norway_codes:
                # Special handling for Norway using Excel data
                filename = os.path.join(self.input_folder, f"{self.file_first}{country}{self.file_last}")
                result_df = self.process_norway_area(country, filename, df_capacities,
                                                     self.minvariable_header_norway, self.maxvariable_header_norway,
                                                     self.minvariable, self.maxvariable,
                                                     self.suffix_open, self.suffix_closed)
            else:
                # Standard processing for other countries
                df_country = df_levels[df_levels["zone"] == country]
                if df_country.empty:
                    continue
                result_df = self.process_country(country, df_country, df_capacities,
                                                 self.minvariable_header, self.maxvariable_header,
                                                 self.minvariable, self.maxvariable,
                                                 self.suffix_reservoir, self.suffix_open, self.suffix_closed)

            # Merge country results into summary dataframe
            if result_df is not None:
                result_df.index = result_df.index.set_names(['time', 'param_gnBoundaryTypes'])
                summary_df = summary_df.join(result_df, how='left')

        # Remove columns with no data (sum equals zero)
        summary_df = summary_df.loc[:, summary_df.sum() > 0]


        # --- Create secondary results for nodes that have timeseries format limits
        # This is used when building the input excel to create correct info
        # e.g. in p_gnBoundaryProperties

        # Create a mask where values are > 0
        mask = summary_df > 0

        # Check which (node, boundary type) combinations have data
        # Group by boundary type and check if sum > 0
        has_data = mask.groupby(level='param_gnBoundaryTypes').sum() > 0

        # Get all combinations with data and their average values
        combinations_with_data = []
        for boundary_type in summary_df.index.get_level_values('param_gnBoundaryTypes').unique():
            # Filter to get only rows with this boundary type
            boundary_data = summary_df.xs(boundary_type, level='param_gnBoundaryTypes')

            for node in summary_df.columns:
                if has_data.loc[boundary_type, node]:
                    # Calculate average across time for this boundary type and node
                    # Get only positive values for this node
                    positive_values = boundary_data[node][boundary_data[node] > 0]
                    avg_value = positive_values.mean() if len(positive_values) > 0 else 0

                    combinations_with_data.append((node, boundary_type, avg_value))

        # Create a dataframe of all valid combinations with their averages
        ts_hydro_storage_limits = pd.DataFrame(
            combinations_with_data, 
            columns=['node', 'param_gnBoundaryTypes', 'average_value']
        )

        # Store secondary result
        self.secondary_result = ts_hydro_storage_limits

        self.log("Hydro storage limit time series built.")


        # Return the main result DataFrame
        return summary_df