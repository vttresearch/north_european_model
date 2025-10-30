import os
import calendar
import pandas as pd
import numpy as np
from src.utils import log_status


class elec_demand_TYNDP2024:
    """
    Class to process electricity demand timeseries for countries.

    Parameters:
        input_folder (str): Relative location of input files.
        country_codes (list): List of country codes.
        start_date (str): Start datetime (e.g., '1982-01-01 00:00:00').
        end_date (str): End datetime (e.g., '2021-01-01 00:00:00').
        scenario_year (int): Target year for scaling the profiles.
    """

    def __init__(self, **kwargs_processor):
        # List of required parameters
        required_params = [
            'input_folder', 
            'country_codes', 
            'start_date', 
            'end_date', 
            'df_annual_demands', 
            'scenario_year'
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
        self.df_annual_demands = kwargs_processor['df_annual_demands']
        self.scenario_year = kwargs_processor['scenario_year']

        # Extract start and end years from the provided dates.
        self.startyear = pd.to_datetime(self.start_date).year
        self.endyear = pd.to_datetime(self.end_date).year

        # Choose the appropriate electricity profile file based on scenario_year.
        if self.scenario_year <= 2035:
            self.input_file = os.path.join(self.input_folder, 'elec_2030_National_Trends.xlsx')
        else:
            self.input_file = os.path.join(self.input_folder, 'elec_2040_National_Trends.xlsx')


        # Initialize log message list
        self.processor_log = []


    def get_values_from_excel(self):
        """
        Reads the Excel file for each country, melts the data, and pivots the DataFrame so that each row 
        is identified by a unique combination of year, month, day, and hour, with each country's values 
        in its own column.

        Returns:
            pivot_df (pd.DataFrame): DataFrame with columns ['year', 'month', 'day', 'hour', ...country columns...]
        """
        # Load Excel file and available sheet names.
        try:
            xl = pd.ExcelFile(self.input_file)
            available_sheets = xl.sheet_names
        except Exception as e:
            raise FileNotFoundError(f"Unable to open {self.input_file}: {e}")

        melted_list = []
        for country in self.country_codes:
            if country not in available_sheets:
                log_status(f"Warning! Sheet for country '{country}' not found in {self.input_file}!!! Skipping.", self.processor_log, level="warn")
                continue

            # Read the sheet (headers are on row 8; zero-indexed header=7).
            df = pd.read_excel(self.input_file, sheet_name=country, header=7)
            df = df.drop(labels='Date', axis=1)

            # Rename the first three columns to: Month, Day, Hour.
            df.rename(columns={df.columns[0]: 'month', 
                               df.columns[1]: 'day', 
                               df.columns[2]: 'hour'}, inplace=True)
            # The remaining columns are the values for different years.
            year_cols = df.columns[3:]
          
            # Melt the DataFrame so that each row corresponds to a specific year and time.
            df_melted = df.melt(id_vars=['month', 'day', 'hour'], 
                                value_vars=year_cols,
                                var_name='year', 
                                value_name='value')
         
            # Convert the "year" column to integer.
            df_melted['year'] = df_melted['year'].astype(int)
            # Add a column to denote which country this row is from.
            df_melted['country'] = country

            melted_list.append(df_melted)


        if not melted_list:
            return pd.DataFrame()

        # Combine the melted DataFrames and pivot.
        combined_df = pd.concat(melted_list, ignore_index=True)
        pivot_df = combined_df.pivot_table(index=['year', 'month', 'day', 'hour'], 
                                             columns='country', 
                                             values='value',
                                             aggfunc='first').reset_index()
        return pivot_df

    
    def process_datetime_index(self, input_df):
        """
        Processes the pivoted DataFrame to compute the datetime index. This includes:
          - Applying the date adjustment logic (incrementing the month and handling leap-year shifts).
          - Exploding rows that produce multiple timestamps (e.g. December 31 in a leap year).
          - Setting and sorting the datetime index.
          - Reindexing to a full hourly datetime index defined by self.start_date and self.end_date.

        Args:
            pivot_df (pd.DataFrame): DataFrame with columns ['year', 'month', 'day', 'hour', ...country columns...]

        Returns:
            result_df (pd.DataFrame): Final DataFrame indexed by a full hourly datetime index.
        """
        
        def adjust_date(row):
            # Adjust month (Excel months are zero-indexed).
            y = int(row['year'])
            m = int(row['month']) + 1
            d = int(row['day'])
            h = int(row['hour'])
            # For leap years, if month is after February, shift the day by -1.
            if calendar.isleap(y) and m > 2:
                d = d - 1
                if d < 1:
                    m = m - 1
                    d = calendar.monthrange(y, m)[1]
            ts = pd.Timestamp(year=y, month=m, day=d, hour=h)

            # If the row represents December 31 in a leap year, duplicate the timestamp.
            if calendar.isleap(y) and int(row['month']) == 11 and int(row['day']) == calendar.monthrange(y, m)[1]:
                extra_ts = ts + pd.Timedelta(days=1)
                return [ts, extra_ts]

            return [ts]

        # Apply the date adjustment once per row.
        input_df['datetime'] = input_df.apply(adjust_date, axis=1)
        # Explode rows with multiple timestamps.
        input_df = input_df.explode('datetime')
        input_df = input_df.dropna(subset=['datetime'])
        # Set the datetime as the index and sort.
        input_df = input_df.set_index('datetime').sort_index()

        # Create a full hourly index for the defined date range.
        full_index = pd.date_range(self.start_date, self.end_date, freq='60min')
        result_df = input_df.reindex(full_index)

        return result_df


    def normalize_profiles(self, df_profiles):
        """
        For each country, the profile is normalized so that the sum of the hourly values 
        (over the years that have valid data) equals the number of valid years.
        """
        # --- Normalize country values ---
        for country in self.country_codes:
            if country not in df_profiles.columns:
                continue
            s = df_profiles[country]

            # Group the series by year and check if at least one value is positive.
            # This returns a Boolean for each year.
            yearly_valid = s.groupby(s.index.year).apply(lambda x: (x.dropna() > 0).any())
            num_years = int(yearly_valid.sum())  # Count of years with valid (positive) data

            # If no valid year is found, mark the entire series as NaN.
            if num_years == 0:
                df_profiles[country] = np.nan
                continue

            # Sum over the whole series (skipping NaNs).
            total = s.sum(skipna=True)

            if total and total != 0:
                # Scale so that the total equals the number of valid years.
                df_profiles[country] = s * (num_years / total)
            else:
                df_profiles[country] = np.nan

        return df_profiles
    
    

    def build_demands(self, df_profiles_norm, df_annual_demands):

        # Create a new DataFrame to store the computed profiles.
        df_profiles_result = pd.DataFrame(index=df_profiles_norm.index)

        for country in self.country_codes:
            # Check that the base hourly profile exists for this country.
            if country not in df_profiles_norm.columns:
                continue

            # Filter demand rows for the country using case-insensitive matching.
            country_demand = df_annual_demands[df_annual_demands['country'].str.lower() == country.lower()]
            if country_demand.empty:
                log_status(f"Warning! No demand data for {country}, will skip the processing!!!", self.processor_log, level="warn")
                continue

            # Process each demand row for the country.
            for _, row in country_demand.iterrows():

                # Pick node name to column name
                col_name = str(row['node'])

                # Retrieve annual demand from 'mwh/year'.
                annual_demand = row['twh/year']*10**6

                # Retrieve constant_share if available; default to 0.0 if missing or NaN.
                constant_share = row['constant_share'] if 'constant_share' in row else 0.0
                if pd.isna(constant_share):
                    constant_share = 0.0

                # Validate constant_share.
                if not (0 <= constant_share <= 1):
                    raise ValueError(
                        f"Constant_share for {col_name} must be between 0 and 1. Got {constant_share}."
                    )

                # Calculate hourly demands = Ax + B, where:
                # A = annual_demand * (1 - constant_share)
                # x = hourly profile
                # B = annual_demand * constant_share / 8760
                A = annual_demand * (1 - constant_share)
                B = annual_demand * constant_share / 8760

                #print(f"      {col_name}.. {round(annual_demand/10**6, 1)} twh/year")
                
                # Compute the new profile using the base profile from df_profiles_norm.
                df_profiles_result[col_name] = A * df_profiles_norm[country] + B

        return df_profiles_result


    def run_processor(self):

        # Get the raw hourly profiles.
        log_status(f"Reading electricity demand profiles from '{self.input_file}'...", self.processor_log)
        df_demands = self.get_values_from_excel()

        log_status("Processing datetime index, handling leap days, etc...", self.processor_log)
        df_demands = self.process_datetime_index(df_demands)

        log_status("Normalizing demand profiles...", self.processor_log)
        summary_df = self.normalize_profiles(df_demands)

        log_status("Building demand time series...", self.processor_log)
        summary_df = self.build_demands(summary_df, self.df_annual_demands)

        # Mandatory secondary results
        secondary_result = None

        log_status("Demand time series built.", self.processor_log, level="info")

        # Note: returning processor log as a string, because then we can distinct it from secondary results which might be a list of strings
        return summary_df, secondary_result, "\n".join(self.processor_log)


# __main__ allows testing by calling this .py file directly.
if __name__ == '__main__':
    # Define the input parameters.
    input_folder = os.path.join("..\\inputFiles\\timeseries")
    output_folder = os.path.join("..\\inputData-test")
    output_file = os.path.join(output_folder, 'test_elec_demand.csv')
    country_codes = [
        'FI00', 'FR00', 'NOM1'
    ]
    start_date = '1982-01-01 00:00:00'
    end_date = '2021-01-01 00:00:00'
    scenario_year = 2025 

    # Constructed demand data for testing
    annual_demands = {
        'country': ['FI00', 'FR00'],
        'grid': ['elec', 'Elec'],
        'year': [2025, 2025],
        'twh/year': [10**5, 10**6],
        'constant_share': [0.9, 0]
        }

    df_annual_demands = pd.DataFrame(annual_demands)

    kwargs_processor = {'input_folder': input_folder,
                        'country_codes': country_codes,
                        'start_date': start_date,
                        'end_date': end_date,
                        'scenario_year': scenario_year, 
                        'df_annual_demands': df_annual_demands
    }


    # Create an instance of elec_demand_ts and run the processing.
    processor = elec_demand_TYNDP2024(**kwargs_processor)
    summary_df = processor.run_processor()

    # Ensure the output directory exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Write the scaled profiles to a CSV file.
    print(f"writing {output_file}")
    summary_df.to_csv(output_file)