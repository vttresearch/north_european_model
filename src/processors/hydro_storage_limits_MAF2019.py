import os
import pandas as pd
from src.utils import log_status


class hydro_storage_limits_MAF2019:
    """
    Class to process reservoir state limits data (reservoir, pump open cycle, pumped closed cycle).

    Parameters:
        input_folder (str): relative location of input files.
        country_codes (list): List of country codes.
        start_date (str): Start datetime (e.g., '1982-01-01 00:00:00').
        end_date (str): End datetime (e.g., '2021-01-01 00:00:00').

        Returns:
            tuple or DataFrame: If valid combinations are found, returns a tuple containing:
                - summary_df: A MultiIndex DataFrame with time series data for each node
                - ts_hydro_storage_limits_df: A DataFrame listing valid (node, boundary type, average_value) combinations
              If no valid combinations are found, returns only the summary_df
    """

    def __init__(self, **kwargs_processor):
        # List of required parameters
        required_params = [
            'input_folder', 
            'country_codes', 
            'start_date', 
            'end_date'
        ]

        # Check if all required parameters are present
        missing_params = [param for param in required_params if param not in kwargs_processor]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Unpack parameters
        self.input_folder = kwargs_processor['input_folder']
        self.country_codes = kwargs_processor['country_codes']
        self.start_date = kwargs_processor['start_date']
        self.end_date = kwargs_processor['end_date']

        # Date range parameters
        self.startyear = pd.to_datetime(self.start_date).year
        self.endyear = pd.to_datetime(self.end_date).year

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

        # Initialize log message list
        self.processor_log = []


    def fill_weekly_data(self, lowerBound, upperBound, weekly_df, year, end_date,
                         country, suffix, cap_key, lower_col, upper_col, df_capacities):
        """
        For a given year, loop through the weekly data (weekly_df) and fill in the
        lower and upper bound DataFrames. The capacity value is looked up from df_capacities
        using cap_key.
        """
        fourthday = pd.Timestamp(year, 1, 4) + pd.DateOffset(hours=12)
        for i in weekly_df.index:
            t = fourthday + pd.DateOffset(days=7 * i)
            # For weeks 0 to 51 (if within end_date)
            if (t <= pd.to_datetime(end_date)) and (i < 52):
                cap_value = df_capacities.at[cap_key, 'value']
                lowerBound.at[t, country + suffix] = 1000 * weekly_df.at[i, lower_col] * cap_value
                upperBound.at[t, country + suffix] = 1000 * weekly_df.at[i, upper_col] * cap_value
            # For week 52 (if present), use the value at index 51.
            elif i == 52 and 51 in weekly_df.index:
                t = pd.Timestamp(year, 12, 28) + pd.DateOffset(hours=12)
                cap_value = df_capacities.at[cap_key, 'value']
                lowerBound.at[t, country + suffix] = 1000 * weekly_df.at[51, lower_col] * cap_value
                upperBound.at[t, country + suffix] = 1000 * weekly_df.at[51, upper_col] * cap_value

    def fill_constant_data(self, lowerBound, upperBound, country, suffix, cap_key, df_capacities):
        """
        Adds a constant column (set to zero for lowerBound and a capacity-based constant for upperBound)
        for the given pump storage type.
        """
        lowerBound[country + suffix] = 0
        try:
            upperBound[country + suffix] = 1000 * df_capacities.at[cap_key, 'value']
        except Exception as e:
            log_status(f"{country} has no capacity data for {suffix}", self.processor_log, level="warn")

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
        result_df = result_df.rename_axis(["", "param_gnBoundaryTypes"], axis="rows")
        result_df.index = result_df.index.set_names(['time', 'param_gnBoundaryTypes'])
        return result_df

    def process_country(self, country, df_country, df_capacities, start_date, end_date, 
                        minvariable_header, maxvariable_header, 
                        minvariable, maxvariable, 
                        suffix_reservoir, suffix_open, suffix_closed):
        """
        Process the data for a single country (non-Norway) using the provided levels
        and capacities DataFrames.
        """
        startyear = pd.to_datetime(start_date).year
        endyear = pd.to_datetime(end_date).year

        # Create an hourly time index.
        date_index = pd.date_range(start_date, end_date, freq='60 min')
        df_lowerBound = pd.DataFrame(index=date_index)
        df_upperBound = pd.DataFrame(index=date_index)
        
        cap_key_reservoir = (country, 'Reservoir', 'Reservoir capacity (GWh)')
        # If detailed level data is available and has no NaNs in the fourth column.
        if not df_country.empty and not df_country.iloc[:, 3].isna().any():
            df_country = df_country.sort_values(by=['year', 'week']).reset_index(drop=True)
            # Process data year by year.
            for year in range(startyear, endyear + 1):
                df_year = df_country[df_country["year"] == year].copy()
                if df_year.empty:
                    continue
                df_year = df_year.reset_index(drop=True)
                self.fill_weekly_data(df_lowerBound, df_upperBound, df_year, year, end_date,
                                      country, suffix_reservoir, cap_key_reservoir, 
                                      minvariable_header, maxvariable_header, df_capacities)
        else:
            # No detailed data available – set constant values.
            self.fill_constant_data(df_lowerBound, df_upperBound, country, suffix_reservoir, cap_key_reservoir, df_capacities)

        # Interpolate to fill in missing hourly data.
        df_lowerBound.interpolate(inplace=True, limit=84, limit_direction='both')
        df_upperBound.interpolate(inplace=True, limit=84, limit_direction='both')

        # Add pump storage open loop constant data.
        cap_key_psOpen = (country, 'Pump Storage - Open Loop', 'Cumulated (upper or head) reservoir capacity (GWh)')
        self.fill_constant_data(df_lowerBound, df_upperBound, country, suffix_open, cap_key_psOpen, df_capacities)
        
        # Add pump storage closed loop constant data.
        cap_key_psClosed = (country, 'Pump Storage - Closed Loop', 'Cumulated (upper or head) reservoir capacity (GWh)')
        self.fill_constant_data(df_lowerBound, df_upperBound, country, suffix_closed, cap_key_psClosed, df_capacities)
        
        result_df = self.merge_bounds(df_lowerBound, df_upperBound, minvariable, maxvariable)
        return result_df

    def process_norway_area(self, country, filename, 
                            df_capacities,
                            start_date, end_date, 
                            minvariable_header_norway, maxvariable_header_norway, 
                            minvariable, maxvariable, 
                            suffix_open, suffix_closed):
        """
        Process Norway-specific area data using an Excel input file.
        """
        startyear = pd.to_datetime(start_date).year
        endyear = pd.to_datetime(end_date).year

        # Create an hourly time index.
        date_index = pd.date_range(start_date, end_date, freq='60 min')
        df_lowerBound = pd.DataFrame(index=date_index)
        df_upperBound = pd.DataFrame(index=date_index)
        
        # Read the Excel data.
        try:
            df = pd.read_excel(
                os.path.normpath(filename),
                sheet_name='Pump storage - Open Loop',
                usecols="L,M",
                names=[minvariable_header_norway, maxvariable_header_norway],
                skiprows=12
            )
        except Exception as e:
            log_status(f"Error reading Norway input Excel: {e}", self.processor_log, level="warn")
            return    
        
        # Process each year.
        cap_key_psOpen = (country, 'Pump Storage - Open Loop', 'Cumulated (upper or head) reservoir capacity (GWh)')
        for year in range(startyear, endyear + 1):
            self.fill_weekly_data(df_lowerBound, df_upperBound, df, year, end_date,
                                  country, suffix_open, cap_key_psOpen,
                                  minvariable_header_norway, maxvariable_header_norway, df_capacities)
        df_lowerBound.interpolate(inplace=True, limit=84, limit_direction='both')
        df_upperBound.interpolate(inplace=True, limit=84, limit_direction='both')
        
        # Add pump storage closed loop constant data.
        cap_key_psClosed = (country, 'Pump Storage - Closed Loop', 'Cumulated (upper or head) reservoir capacity (GWh)')
        self.fill_constant_data(df_lowerBound, df_upperBound, country, suffix_closed, cap_key_psClosed, df_capacities)
        
        result_df = self.merge_bounds(df_lowerBound, df_upperBound, minvariable, maxvariable)
        return result_df

    def run_processor(self):
        """
        Executes the hydro storage data processing pipeline.

        This function performs the following steps:
        1. Reads and validates reservoir levels data from CSV files
        2. Reads capacity data for different zones and types
        3. Creates a time-series framework with minimum and maximum boundary types
        4. Processes each country's data (with special handling for Norway)
        5. Creates ts_hydro_storage_limits_df of valid (node, boundary type) combinations

        Returns:
            tuple or DataFrame: If valid combinations are found, returns a tuple containing:
                - summary_df: A MultiIndex DataFrame with time series data for each node
                - ts_hydro_storage_limits_df: A DataFrame listing valid (node, boundary type, average_value) combinations
              If no valid combinations are found, returns only the summary_df
        """
        # Read the levels CSV file.
        try:
            df_levels = pd.read_csv(self.levels_file)
        except Exception as e:
            log_status(f"Error reading input CSV file: {e}", self.processor_log, level="warn")
            return

        log_status(f"Processing input files...", self.processor_log, level="info")

        # Filter data by year range
        df_levels = df_levels[(df_levels["year"] >= self.startyear) & (df_levels["year"] <= self.endyear)]
        df_levels["year"] = pd.to_numeric(df_levels["year"])
        df_levels["week"] = pd.to_numeric(df_levels["week"])

        # Read the capacities CSV file.
        try:
            df_capacities = pd.read_csv(self.capacities_file)
        except Exception as e:
            log_status(f"Error reading capacity CSV file: {e}", self.processor_log, level="warn")
            return
        df_capacities = df_capacities.set_index(["zone", "type", "variable"])

        # Create a summary DataFrame with a MultiIndex (time, param_gnBoundaryTypes).
        idx = pd.MultiIndex.from_product(
            [pd.date_range(self.start_date, self.end_date, freq='60 min'),
             [self.minvariable, self.maxvariable]],
            names=['time', 'param_gnBoundaryTypes']
        )
        summary_df = pd.DataFrame(index=idx)


        log_status("Building country level timeseries...", self.processor_log, level="info")
        # Process each country.
        for country in self.country_codes:
            if country in self.norway_codes:
                # Special handling for Norway using Excel data
                filename = os.path.join(self.input_folder, f"{self.file_first}{country}{self.file_last}")
                result_df = self.process_norway_area(country, filename, df_capacities,
                                                     self.start_date, self.end_date,
                                                     self.minvariable_header_norway, self.maxvariable_header_norway,
                                                     self.minvariable, self.maxvariable,
                                                     self.suffix_open, self.suffix_closed)
            else:
                # Standard processing for other countries
                df_country = df_levels[df_levels["zone"] == country]
                if df_country.empty:
                    continue
                result_df = self.process_country(country, df_country, df_capacities,
                                                 self.start_date, self.end_date,
                                                 self.minvariable_header, self.maxvariable_header,
                                                 self.minvariable, self.maxvariable,
                                                 self.suffix_reservoir, self.suffix_open, self.suffix_closed)

            # Merge country results into summary dataframe
            if result_df is not None:
                result_df.index = result_df.index.set_names(['time', 'param_gnBoundaryTypes'])
                summary_df = summary_df.join(result_df, how='left')

        # Remove columns with no data (sum equals zero)
        summary_df = summary_df.loc[:, summary_df.sum() > 0]

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

        log_status("Hydro storage limit time series built.", self.processor_log, level="info")

        return summary_df, ts_hydro_storage_limits, "\n".join(self.processor_log)




# Example usage:
if __name__ == '__main__':
    input_folder = os.path.join("..\\inputFiles\\timeseries")
    output_folder = os.path.join("..\\inputData-test")
    output_file = os.path.join(output_folder, f'test_hydro_reservoir_limits.csv')
    country_codes = [
        'AT00', 'BE00', 'CH00', 'DE00', 'DKW1', 'DKE1', 'EE00', 'ES00',
        'FI00', 'FR00', 'UK00', 'LT00', 'LV00', 'NL00', 'NOS0', 'NOM1',
        'NON1', 'PL00', 'SE01', 'SE02', 'SE03', 'SE04'
    ]
    start_date = '2015-01-01 00:00:00'
    end_date = '2015-12-31 23:00:00'

    kwargs_processor = {'input_folder': input_folder,
                        'country_codes': country_codes,
                        'start_date': start_date,
                        'end_date': end_date,
                        'scenario_year': scenario_year, 
                        'df_annual_demands': df_annual_demands
    }

    processor = hydro_storage_limits_MAF2019(**kwargs_processor)
    result = processor.run_processor()

    # Handle an optional second DataFrame and possible no result
    if isinstance(result, tuple) and len(result) == 2:
        summary_df, df_optional = result
    elif result is None:
        print(f"processor did not return any DataFrame.")
    else:
        summary_df = result 

    # Ensure the output directory exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Write to a csv.
    print(f"writing {output_file}")
    summary_df.to_csv(output_file)