import os
import pandas as pd
import numpy as np
from src.utils import log_status


class DH_demand_fromTemperature:
    """
    Class to process district heating demand timeseries for countries.

    Parameters:
        input_folder (str): Relative location of input files.
        country_codes (list): List of country codes.
        start_date (str): Start datetime (e.g., '1982-01-01 00:00:00').
        end_date (str): End datetime (e.g., '2021-01-01 00:00:00').
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
        optional_params = {
            'exclude_grids': [],
            'exclude_nodes': []
        }

        # Check if all required parameters are present
        missing_params = [param for param in required_params if param not in kwargs_processor]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Unpack required parameters
        for param in required_params:
            setattr(self, param, kwargs_processor.get(param))        

        # Unpack optional parameters with defaults if not provided
        for key, default in optional_params.items():
            setattr(self, key, kwargs_processor.get(key, default))

        # Extract start and end years from the provided dates.
        self.startyear = pd.to_datetime(self.start_date).year
        self.endyear = pd.to_datetime(self.end_date).year

        # build input file path
        self.input_file = os.path.join(self.input_folder, 'Temperature.csv')

        # Initialize log message list
        self.processor_log = []


    def get_temperature_profile(self, processed_countries):
        """
        Reads temperature data from csv and converts that to heating profile for each country
            - Calculates sliding 24h average from temperature
            - heating profile = max(0, 17 - 24h_temp) for each hour
            - requires mapping from countries in temperatures to country_codes
        """

        # Read temperatures from csv to dataframe
        try:
            df_temperature = pd.read_csv(self.input_file)
        except Exception as e:
            raise FileNotFoundError(f"Unable to open {self.input_file}: {e}")

        df_temperature['Time'] = pd.to_datetime(df_temperature['Time'])
        df_temperature = df_temperature.set_index('Time')

        # Define mapping: internal country_code -> CSV country column
        mapping = {
            'AT00': 'AT',
            'BE00': 'BE',
            'CH00': 'CH',
            'DE00': 'DE',
            'DKW1': 'DK',
            'DKE1': 'DK',
            'EE00': 'EE',
            'ES00': 'ES',
            'FI00': 'FI',
            'FR00': 'FR',
            'UK00': 'UK',
            'LT00': 'LT',
            'LV00': 'LV',
            'NL00': 'NL',
            'NOS0': 'NO',
            'NOM1': 'NO',
            'NON1': 'NO',
            'PL00': 'PL',
            'SE01': 'SE',
            'SE02': 'SE',
            'SE03': 'SE',
            'SE04': 'SE',
            }

        # Create a full datetime index to ensure no timesteps are missing.
        full_index = pd.date_range(self.start_date, self.end_date, freq='60min')
        df_heating_profile = pd.DataFrame(index=full_index)

        # Calculate heating profile for each country.
        for country in processed_countries:
            # Get the corresponding CSV column name using the mapping.
            csv_country = mapping.get(country, None)
            if csv_country and csv_country in df_temperature.columns:
                # Calculate the sliding 24h average.
                rolling_avg = df_temperature[csv_country].rolling(window=24, min_periods=24).mean()
                # Reindex the rolling average to the full date range.
                rolling_avg_full = rolling_avg.reindex(full_index)
                # Compute the heating profile: heating demand = max(0, 17 - rolling_avg)
                heating_profile = (17 - rolling_avg_full).clip(lower=0)
                df_heating_profile[country] = heating_profile
            else:
                # If the country is not available in the source data, assign NaN.
                df_heating_profile[country] = np.nan

        return df_heating_profile
    

    def normalize_profiles(self, df_profiles, processed_countries):
        """
        For each country, the profile is normalized so that the sum of the hourly values 
        (over the years that have valid data) equals the number of valid years.
        """
        # --- Normalize country values ---
        for country in processed_countries:
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
    

    def build_demands(self, df_profiles_norm, df_annual_demands, processed_countries):

        # Create a new DataFrame to store the computed profiles.
        df_profiles_result = pd.DataFrame(index=df_profiles_norm.index)

        for country in processed_countries:
            # Check that the base hourly profile exists for this country.
            if country not in df_profiles_norm.columns:
                continue

            # Filter demand rows for the country using case-insensitive matching.
            country_demand = df_annual_demands[df_annual_demands['country'].str.lower() == country.lower()]

            # Process each demand row for the country.
            for _, row in country_demand.iterrows():
                # Determine the node suffix in a case-insensitive way.
                suffix = ''
                if 'node_suffix' in row and pd.notna(row['node_suffix']):
                    suffix = str(row['node_suffix'])

                # Build the new column name: if no suffix, use the country code; otherwise, use country+suffix.
                col_name = country if suffix == '' else country + '_' + suffix

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

                #print(f"      {col_name}.. {round(annual_demand/10**6, 1)} TWh/year")
                
                # Compute the new profile using the base profile from df_profiles_norm.
                df_profiles_result[col_name] = A * df_profiles_norm[country] + B

        return df_profiles_result


    def run_processor(self):

        # Filter countries in country codes that actually have annual demand data
        countries_with_data = self.df_annual_demands['country'].unique()
        processed_countries = [code for code in self.country_codes if code in countries_with_data]

        # Get the raw hourly profiles.
        log_status(f"Constructing heat demand profiles from '{self.input_file}'...", self.processor_log)
        temp_profiles = self.get_temperature_profile(processed_countries)

        log_status("Normalizing demand profiles...", self.processor_log)
        norm_profiles = self.normalize_profiles(temp_profiles, processed_countries)

        log_status("Building demand time series...", self.processor_log)
        summary_df = self.build_demands(norm_profiles, self.df_annual_demands, processed_countries)
        
        # Renaming column titles from country to <country>_<grid> or <country>_<grid>_<suffix> if suffix exists,
        # e.g. DE00 -> DE00_dheat and FI00 -> FI00_dheat_HKI
        grid = self.df_annual_demands['grid'].iloc[0]
        new_columns = {}
        for col in summary_df.columns:
            for country in processed_countries:
                if col.startswith(country):
                    # Capture any suffix after the country code.
                    suffix = col[len(country):]  # e.g., '' or 'HKI'
                    # Build the new column name
                    new_columns[col] = f"{country}_{grid}" + suffix
                    break  # Found matching country code, move to next column.
        summary_df = summary_df.rename(columns=new_columns)

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
    output_file = os.path.join(output_folder, 'test_heat_demand.csv')
    country_codes = [
        'FI00', 'FR00', 'NOM1'
    ]
    start_date = '1982-01-01 00:00:00'
    end_date = '2021-01-01 00:00:00'
    scenario_year = 2025 

    # Constructed demand data for testing
    annual_demands = {
        'country': ['FI00', 'FR00'],
        'grid': ['dheat', 'dheat'],
        'node_suffix': ['HKI', ''],
        'year': [2025, 2025],
        'twh/year': [1000, 100],
        'constant_share': [0.3, 0.4]
        }

    df_annual_demands = pd.DataFrame(annual_demands)

    kwargs_processor = {'input_folder': input_folder,
                        'country_codes': country_codes,
                        'start_date': start_date,
                        'end_date': end_date,
                        'scenario_year': scenario_year, 
                        'df_annual_demands': df_annual_demands
    }

    # Create an instance of DH_demand_ts and run the processing.
    processor = DH_demand_fromTemperature(**kwargs_processor)
    summary_df = processor.run_processor()

    # Ensure the output directory exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Write the scaled profiles to a CSV file.
    print(f"writing {output_file}")
    summary_df.to_csv(output_file)