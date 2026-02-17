# src/processors/hydro_mingen_limits_MAF2019.py

import os
import pandas as pd
from datetime import date
from src.processors.base_processor import BaseProcessor


class hydro_mingen_limits_MAF2019(BaseProcessor):
    """
    This class processes min/max generation data in several steps:

      1. For each country (zone) in a merged list, it:
         - Uses the global CSV input if the country is not one of the Norway zones.
         - Uses the Norway Excel files if the country is one of the Norway zones.
      2. It collects each country's processed DataFrame in memory.
      3. It then creates a summary DataFrame by merging all individual DataFrames and writes the result to a CSV file.

    Parameters:
        input_folder (str): Folder containing the input files.
        country_codes (list): List of country codes to process.
        start_date (str): Start date in string format (e.g., "1982-01-01 00:00:00").
        end_date (str): End date in string format (e.g., "2021-01-01 00:00:00").
        process_maxGen (bool): If True, process and print maximum generation limits as well. Default is False.

    Returns:
        main_result (pd.DataFrame): DateTime index from start_date to end_date
                                    Processed countries as column names
                                    hydro power minimum generation in MW as column values
        secondary_result (list): List of nodes that have minimum generation limits
    """

    def __init__(self, **kwargs):
        # Initialize base class
        super().__init__(**kwargs)
        
        # List of required parameters
        required_params = [
            'input_folder', 
            'country_codes', 
            'start_date', 
            'end_date'
        ]

        # Check if all required parameters are present
        missing_params = [param for param in required_params if param not in kwargs]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Unpack parameters
        self.input_folder = kwargs['input_folder']
        self.country_codes = kwargs['country_codes']
        self.start_date = kwargs['start_date']
        self.end_date = kwargs['end_date']

        # Global input file (for non-Norway countries)
        self.input_csv = "PECD-hydro-weekly-reservoir-min-max-generation.csv"

        # Parameters for processing the generation limits.
        self.minvariable_header = "Minimum Generation in MW"
        self.suffix_reservoir = {
            'FI00': '_reservoir',
            'SE01': '_reservoir',
            'SE02': '_reservoir',
            'SE03': '_reservoir',
            'SE04': '_reservoir',
            'FR00': '_reservoir',
            'NON1': '_psOpen',
            'NOM1': '_psOpen',
            'NOS0': '_psOpen',
        }

        # Norway-specific file parameters.
        self.norway_file_first = "PEMMDB_"
        self.norway_file_last = "_Hydro Inflow_SOR 20.xlsx"
        self.norway_countries = ['NOM1', 'NON1', 'NOS0']

        # Determine the start and end years.
        self.start_year = pd.to_datetime(self.start_date).year
        self.end_year = pd.to_datetime(self.end_date).year

    def process_global_country(self, country, country_df):
        """
        Processes global minimum generation data for one country from the CSV input.
        
        Parameters:
            country (str): The country code (e.g., 'AT00')
            country_df (pd.DataFrame): The country-level CSV DataFrame (already filtered by year)

        Returns:
            pd.DataFrame: A DataFrame with a time index, 
                         column 'group' = 'UC_'<country>_<suffix_reservoir>, 
                         column 'param_policy' = 'userconstraintRHS'
        """
        date_index = pd.date_range(self.start_date, self.end_date, freq='60 min')
        df_result = pd.DataFrame(index=date_index)
        country_suffix = self.suffix_reservoir.get(country, '')

        for year in range(self.start_year, self.end_year + 1):
            df_year = country_df[country_df["year"] == year].copy().reset_index(drop=True)
            if df_year.empty:
                continue

            # The starting timestamp is based on the fourth day of January plus a 12-hour offset.
            fourthday = date(year, 1, 4) + pd.DateOffset(hours=12)
            for i in df_year.index:
                t = fourthday + i * pd.DateOffset(days=7)
                # For weeks 0 to 51 (if within end_date)
                if (t <= pd.to_datetime(self.end_date)) and (i < 52):
                    df_result.at[t, country + country_suffix] = df_year.at[i, self.minvariable_header]
                else:
                    if i == 52:
                        # For the final week of the year, use the value at index 51.
                        t = pd.Timestamp(year, 12, 28) + pd.DateOffset(hours=12)
                        df_result.at[t, country + country_suffix] = df_year.at[51, self.minvariable_header]

        # Interpolate missing values for min generation.
        df_result.interpolate(inplace=True, limit=84, limit_direction='both')

        # fill group param_policy
        df_result['param_policy'] = 'userconstraintRHS'

        # create multi-index and return
        df_result = df_result.reset_index(names='time')
        df_result = df_result.set_index(['time', 'param_policy'])
        return df_result

    def process_norway_country(self, country, filename):
        """
        Processes min/max generation data for one Norway country from an Excel file.

        Parameters:
            country (str): The Norway country code (e.g., 'NOM1')
            filename (str): The path to the Norway Excel input file

        Returns:
            pd.DataFrame: A DataFrame with a time index and a "genLimit" level containing the processed data.
                         If process_maxGen is False, only the min generation data is returned.
        """
        date_index = pd.date_range(self.start_date, self.end_date, freq='60 min')
        df_result = pd.DataFrame(index=date_index)
        country_suffix = self.suffix_reservoir.get(country, '')

        try:
            df_input = pd.read_excel(
                filename,
                sheet_name='Pump storage - Open Loop',
                usecols="GN:HV",
                skiprows=12
            )
        except Exception as e:
            self.log(f"Error reading file {filename}: {e}", level="warn")
            return None

        # Convert Excel column headers (which represent years) from floats to ints.
        years_float = [float(x) for x in df_input.columns.tolist()]
        years = [int(x) for x in years_float]
        df_input = df_input.rename(columns=dict(zip(df_input.columns, years)))

        for year in years:
            fourthday = date(year, 1, 4) + pd.DateOffset(hours=12)
            df_year = df_input[year]
            if df_year.empty:
                continue
            for i in df_year.index:
                t = fourthday + i * pd.DateOffset(days=7)
                if t <= pd.to_datetime(self.end_date) and i < 52:
                    df_result.at[t, country + country_suffix] = df_year[i]
                else:
                    if i == 52:
                        t = pd.Timestamp(year, 12, 28) + pd.DateOffset(hours=12)
                        df_result.at[t, country + country_suffix] = df_year[51]

        # Interpolate missing values for min generation.
        df_result.interpolate(inplace=True, limit=84, limit_direction='both')

        # fill param_policy
        df_result['param_policy'] = 'userconstraintRHS'

        # create multi-index and return
        df_result = df_result.reset_index(names='time')
        df_result = df_result.set_index(['time', 'param_policy'])
        return df_result

    def process(self) -> pd.DataFrame:
        """
        Main processing logic that executes the processing steps:
          - Reads the global CSV input for non-Norway countries.
          - Processes each country (using the Norway method when appropriate).
          - Merges all individual DataFrames into a summary.
          
        Returns:
            pd.DataFrame: Summary DataFrame with hydro minimum generation limits
        """
        self.log("Reading input data...")
        
        # Read the global CSV input file.
        inputfile = os.path.join(self.input_folder, self.input_csv)
        try:
            global_df = pd.read_csv(inputfile)
        except Exception as e:
            self.log(f"Error reading hydro mingen input CSV '{inputfile}': {e}", level="warn")
            # Return empty DataFrame if input fails
            return pd.DataFrame()

        # Filter the global DataFrame by year range.
        global_df = global_df[(global_df["year"] >= self.start_year) & (global_df["year"] <= self.end_year)]
        global_df["year"] = pd.to_numeric(global_df["year"])
        global_df["week"] = pd.to_numeric(global_df["week"])

        # Prepare the summary DataFrame 
        summary_df = pd.DataFrame()
        self.log("Processing country level limits...")

        # Process each country.
        for country in self.country_codes:
            # choose correct processing function
            if country in self.norway_countries:
                filename = os.path.join(self.input_folder, f"{self.norway_file_first}{country}{self.norway_file_last}")
                result_df = self.process_norway_country(country, filename)
            else:
                country_df = global_df[global_df["zone"] == country].copy()
                if country_df.empty:
                    continue
                result_df = self.process_global_country(country, country_df)

            # join results
            if result_df is not None:
                # For the first DataFrame, assign it directly to summary_df
                if summary_df.empty:
                    summary_df = result_df.copy()
                else:
                    summary_df = summary_df.join(result_df, how='left')

        # Drop columns if their sum is zero
        summary_df = summary_df.loc[:, summary_df.sum() != 0]

        # Pick nodes with active mingen limits and store as secondary result
        hydro_mingen_nodes = summary_df.columns.tolist()
        self.secondary_result = hydro_mingen_nodes

        self.log("Hydro mingen time series built.")

        # Return the main result DataFrame
        return summary_df