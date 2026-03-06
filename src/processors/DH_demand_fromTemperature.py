# src/processors/DH_demand_fromTemperature.py

import os
import pandas as pd
import numpy as np
from src.processors.base_processor import BaseProcessor


class DH_demand_fromTemperature(BaseProcessor):
    """
    Class to process district heating demand timeseries for countries.
    """

    def __init__(self, **kwargs_processor):
        """
        Initialize the processor.

        Required kwargs: input_folder, country_codes, start_year, end_year,
        df_annual_demands, scenario_year.
        """
        # Initialize base class
        super().__init__(**kwargs_processor)

        # List of required parameters
        required_params = [
            'input_folder',
            'country_codes',
            'start_year',
            'end_year',
            'df_annual_demands',
            'scenario_year'
        ]

        # Check if all required parameters are present
        missing_params = [param for param in required_params if param not in kwargs_processor]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Unpack required parameters
        for param in required_params:
            setattr(self, param, kwargs_processor.get(param))
        self.demand_grid = kwargs_processor.get('demand_grid', '')

        # Derive full-year date boundaries from integer year values
        self.start_date = pd.Timestamp(f"{self.start_year}-01-01")
        self.end_date   = pd.Timestamp(f"{self.end_year}-12-31 23:00")

        # build input file path
        # The current source file Temperature.csv has hourly data from 1980-01-01 to 2019-12-31
        # data is for countries AT,BE,CH,DE,DK,EE,FI,FR,GB,LT,LV,NL,NO,PL,SE,ES
        self.input_file = os.path.join(self.input_folder, 'Temperature.csv')


    def get_temperature_profile(self, processed_countries):
        """
        Read temperature data from CSV and compute an hourly heating profile for each country.

        The CSV column for a country is derived from the first two letters of the country code
        (e.g. 'FI00' -> 'FI', 'SE01' -> 'SE', 'NOS0' -> 'NO').

        The 24-hour sliding average is computed over a window that starts 2 days before
        self.start_date so that the first timesteps of the returned profile are not affected
        by incomplete rolling windows. The returned DataFrame covers exactly
        [self.start_date, self.end_date].

        Heating profile formula: max(0, 17 - 24h_rolling_average_temperature).

        Countries not present in the CSV receive NaN for all timesteps.

        Parameters
        ----------
        processed_countries : list[str]
            Country codes to process.

        Returns
        -------
        pd.DataFrame
            Hourly heating profiles indexed by timestamp, one column per country.
        """

        # Read temperatures from csv to dataframe
        try:
            df_temperature = pd.read_csv(self.input_file)
        except Exception as e:
            raise FileNotFoundError(f"Unable to open {self.input_file}: {e}")

        df_temperature['Time'] = pd.to_datetime(df_temperature['Time'])
        df_temperature = df_temperature.set_index('Time')

        # Start 2 days before the requested period so the rolling window is fully
        # populated at self.start_date. The climate year range (1982-2016) guarantees
        # that two extra days are always available.
        pre_start = pd.Timestamp(self.start_date) - pd.Timedelta(days=2)
        df_temperature = df_temperature.loc[pre_start:self.end_date]

        # Create a full datetime index for the requested output range.
        full_index = pd.date_range(self.start_date, self.end_date, freq='60min')
        df_heating_profile = pd.DataFrame(index=full_index)

        # Calculate heating profile for each country.
        for country in processed_countries:
            # Map country code to CSV column using the first two letters (e.g. 'FI00' -> 'FI').
            csv_country = country[:2]
            if csv_country in df_temperature.columns:
                # Calculate the sliding 24h average over the extended window.
                rolling_avg = df_temperature[csv_country].rolling(window=24, min_periods=24).mean()
                # Reindex to the requested output range (drops the pre-start warm-up rows).
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
        Normalize heating profiles so that each country's annual average sums to 1.

        For each country the profile is scaled so that the total sum of hourly values
        equals the number of years that contain at least one positive value.  This makes
        the profile unit-neutral: multiplying by an annual demand figure in any energy
        unit gives the correct hourly breakdown.

        Countries whose column is absent or has no positive data are set to NaN.

        Parameters
        ----------
        df_profiles : pd.DataFrame
            Hourly heating profiles as returned by get_temperature_profile.
        processed_countries : list[str]
            Country codes to normalize.

        Returns
        -------
        pd.DataFrame
            Normalized profiles (in-place modification of df_profiles).
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
        """
        Scale normalized heating profiles to absolute hourly demands.

        For each demand row in df_annual_demands the hourly series is computed as:

            demand(t) = A * profile(t) + B

        where:
            A = annual_demand * (1 - constant_share)   (temperature-driven part)
            B = annual_demand * constant_share / 8760   (constant base load per hour)

        Parameters
        ----------
        df_profiles_norm : pd.DataFrame
            Normalized heating profiles from normalize_profiles, indexed by timestamp.
        df_annual_demands : pd.DataFrame
            Demand table with columns: country, node, twh/year, constant_share.
        processed_countries : list[str]
            Country codes to process.

        Returns
        -------
        pd.DataFrame
            Hourly demand time series in MWh, one column per node.
        """

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
                
                # Compute the new profile using the base profile from df_profiles_norm.
                df_profiles_result[col_name] = A * df_profiles_norm[country] + B

        return df_profiles_result


    def process(self) -> pd.DataFrame:
        """
        Run the full district heating demand pipeline.

        Steps:
        1. Filter country_codes to those present in df_annual_demands.
        2. get_temperature_profile  -- raw hourly heating profiles from temperature data.
        3. normalize_profiles       -- scale profiles to unit annual sum.
        4. build_demands            -- multiply by annual demand figures.

        Returns
        -------
        pd.DataFrame
            Hourly district heating demand in MWh, one column per node, indexed by timestamp.
        """
        # Filter countries
        countries_with_data = self.df_annual_demands['country'].unique()
        processed_countries = [
            code for code in self.country_codes if code in countries_with_data
        ]

        # Get the raw hourly profiles
        self.logger.log_status(f"Constructing heat demand profiles from '{self.input_file}'...")
        out_df = self.get_temperature_profile(processed_countries)

        self.logger.log_status("Normalizing demand profiles...")
        out_df = self.normalize_profiles(out_df, processed_countries)

        self.logger.log_status("Building demand time series...")
        out_df = self.build_demands(out_df, self.df_annual_demands, processed_countries)

        # Set secondary result if needed
        self.secondary_result = None

        self.logger.log_status("Demand time series built.", level="info")

        # Convert to long format: [grid, node, time, value]
        result = out_df.reset_index(names='time')
        result = result.melt(id_vars=['time'], var_name='node', value_name='value')
        result['value'] = -result['value']
        result['grid'] = self.demand_grid
        return result[['grid', 'node', 'time', 'value']]