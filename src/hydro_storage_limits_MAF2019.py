import os
import pandas as pd
import numpy as np


class hydro_storage_limits_MAF2019:
    """
    Class to process reservoir state limits data (reservoir, pump open cycle, pumped closed cycle).

    Parameters:
        input_folder (str): relative location of input files.
        country_codes (list): List of country codes.
        start_date (str): Start datetime (e.g., '1982-01-01 00:00:00').
        end_date (str): End datetime (e.g., '2021-01-01 00:00:00').
    """

    def __init__(self, input_folder, country_codes, start_date, end_date):
        self.input_folder = input_folder
        self.country_codes = country_codes

        # Fixed date range parameters
        self.start_date = start_date
        self.end_date = end_date
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
            print(f"{country} has no capacity data for {suffix}")

    def merge_bounds(self, lowerBound, upperBound, minvariable, maxvariable):
        """
        After filling the lower and upper bound DataFrames, set the
        'boundarytype' column, concatenate them, and tidy up the index.
        """
        lowerBound['boundarytype'] = minvariable
        upperBound['boundarytype'] = maxvariable

        result_df = pd.concat([lowerBound, upperBound])
        result_df = result_df.reset_index()
        result_df = result_df.sort_values(['index', 'boundarytype'])
        result_df = result_df.set_index(['index', 'boundarytype'])
        result_df = result_df.rename_axis(["", "boundarytype"], axis="rows")
        result_df.index = result_df.index.set_names(['time', 'boundarytype'])
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
            print(f"Error reading Norway input Excel: {e}")
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

    def run(self):
        """
        Executes the processing: reading input files, processing each area,
        merging results, and return the summary_df
        """
        # Read the levels CSV file.
        try:
            df_levels = pd.read_csv(self.levels_file)
        except Exception as e:
            print(f"Error reading input CSV file: {e}")
            return

        df_levels = df_levels[(df_levels["year"] >= self.startyear) & (df_levels["year"] <= self.endyear)]
        df_levels["year"] = pd.to_numeric(df_levels["year"])
        df_levels["week"] = pd.to_numeric(df_levels["week"])

        # Read the capacities CSV file.
        try:
            df_capacities = pd.read_csv(self.capacities_file)
        except Exception as e:
            print(f"Error reading capacity CSV file: {e}")
            return
        df_capacities = df_capacities.set_index(["zone", "type", "variable"])

        # Create a summary DataFrame with a MultiIndex (time, boundarytype).
        idx = pd.MultiIndex.from_product(
            [pd.date_range(self.start_date, self.end_date, freq='60 min'),
             [self.minvariable, self.maxvariable]],
            names=['time', 'boundarytype']
        )
        summary_df = pd.DataFrame(index=idx)

        # Process each country.
        for country in self.country_codes:
            if country in self.norway_codes:
                # Build the filename for Norway-specific Excel data.
                filename = os.path.join(self.input_folder, f"{self.file_first}{country}{self.file_last}")
                result_df = self.process_norway_area(country, filename, df_capacities,
                                                     self.start_date, self.end_date,
                                                     self.minvariable_header_norway, self.maxvariable_header_norway,
                                                     self.minvariable, self.maxvariable,
                                                     self.suffix_open, self.suffix_closed)
            else:
                df_country = df_levels[df_levels["zone"] == country]
                if df_country.empty:
                    print(f"   No reservoir limit data for {country}")
                    continue
                result_df = self.process_country(country, df_country, df_capacities,
                                                 self.start_date, self.end_date,
                                                 self.minvariable_header, self.maxvariable_header,
                                                 self.minvariable, self.maxvariable,
                                                 self.suffix_reservoir, self.suffix_open, self.suffix_closed)

            if result_df is not None:
                result_df.index = result_df.index.set_names(['time', 'boundarytype'])
                summary_df = summary_df.join(result_df, how='left')

        return summary_df


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

    processor = hydro_reservoir_limits_MAF2019(input_folder, country_codes, start_date, end_date)
    summary_df = processor.run()

    # Ensure the output directory exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Write to a csv.
    print(f"writing {output_file}")
    summary_df.to_csv(output_file)