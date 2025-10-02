import os
import pandas as pd
from datetime import date
from src.utils import log_status

class hydro_mingen_limits_MAF2019:
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
        summary_df: DateTime index from start_date to end_date
                    Processed countries as column names
                    hydro power minimum generation in MW as column values
        hydro_mingen_nodes: list of nodes that have minimum generation limits
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

        # Global input file (for non-Norway countries)
        self.input_csv = "PECD-hydro-weekly-reservoir-min-max-generation.csv"

        # Parameters for processing the generation limits.
        self.minvariable_header = "Minimum Generation in MW"
        self.suffix_reservoir = {'AL00': '',
                                'AT00': '',
                                'BA00': '',
                                'BG00': '',
                                'CH00': '',
                                'CZ00': '',
                                'DE00': '',
                                'ES00': '',
                                'FI00': '_reservoir',
                                'GR00': '',
                                'HR00': '',
                                'ITCN': '',
                                'ITCS': '',
                                'ITN1': '',
                                'ITS1': '',
                                'ITSA': '',
                                'LT00': '',
                                'LV00': '',
                                'MK00': '',
                                'RO00': '',
                                'RS00': '',
                                'SE01': '_reservoir',
                                'SE02': '_reservoir',
                                'SE03': '_reservoir',
                                'SE04': '_reservoir',
                                'SK00': '',
                                'TR00': '',
                                'FR00': '_reservoir',
                                'ME00': '',
                                'PL00': '',
                                'PT00': '',
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

        # Initialize log message list
        self.processor_log = []

    def process_global_country(self, country, country_df):
        """
        Processes global minimum generation data for one country from the CSV input.
        Parameters:  country     -- The country code (e.g., 'AT00')
                     country_df  -- The country-level CSV DataFrame (already filtered by year)

        Returns: A DataFrame with a time index, 
                                    column 'group' = 'UC_'<country>_<suffix_reservoir>, 
                                    column 'param_policy' = 'userconstraintRHS'
        """
        date_index = pd.date_range(self.start_date, self.end_date, freq='60 min')
        df_result = pd.DataFrame(index=date_index)
        country_suffix = self.suffix_reservoir[country]

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
          country  -- The Norway country code (e.g., 'NOM1')
          filename -- The path to the Norway Excel input file

        Returns:
          A DataFrame with a time index and a "genLimit" level containing the processed data.
          If process_maxGen is False, only the min generation data is returned.
        """
        date_index = pd.date_range(self.start_date, self.end_date, freq='60 min')
        df_result = pd.DataFrame(index=date_index)
        country_suffix = self.suffix_reservoir[country]

        try:
            df_input = pd.read_excel(
                filename,
                sheet_name='Pump storage - Open Loop',
                usecols="GN:HV",
                skiprows=12
            )
        except Exception as e:
            log_status(f"Error reading file {filename}: {e}", self.processor_log, level="warn")
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


    def run_processor(self):
        """
        Executes the processing steps:
          - Reads the global CSV input for non-Norway countries.
          - Processes each country (using the Norway method when appropriate).
          - Merges all individual DataFrames into a summary and writes it to a CSV file.
        """
        log_status("Reading input data...", self.processor_log)
        # Read the global CSV input file.
        inputfile = os.path.join(self.input_folder, self.input_csv)
        try:
            global_df = pd.read_csv(inputfile)
        except Exception as e:
            log_status(f"Error reading input CSV file: {e}", self.processor_log, level="warn")
            return

        # Filter the global DataFrame by year range.
        global_df = global_df[(global_df["year"] >= self.start_year) & (global_df["year"] <= self.end_year)]
        global_df["year"] = pd.to_numeric(global_df["year"])
        global_df["week"] = pd.to_numeric(global_df["week"])

        # Prepare the summary DataFrame 
        summary_df = pd.DataFrame()
        log_status(f"Processing country level limits...", self.processor_log, level="info")

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

        # Pick nodes with active mingen limits
        hydro_mingen_nodes = summary_df.columns.tolist()

        log_status("Hydro mingen time series built.", self.processor_log, level="info")

        # Return results
        return summary_df, hydro_mingen_nodes, "\n".join(self.processor_log)



# __main__ for testing this function directly
if __name__ == "__main__":
    input_folder = os.path.join("..\\src_files\\timeseries")
    output_folder = os.path.join("..\\inputData-test")
    output_file = os.path.join(output_folder, f'test_hydro_generation_limits.csv')
    country_codes = [
        'FI00', 'FR00', 'UK00', 'LT00', 'LV00', 'NL00', 'NOS0', 'NOM1',
        'NON1', 'PL00', 'SE01', 'SE02', 'SE03', 'SE04'
    ]
    start_date = "1982-01-01 00:00:00"
    end_date = "2021-01-01 00:00:00"

    kwargs_processor = {'input_folder': input_folder,
                        'country_codes': country_codes,
                        'start_date': start_date,
                        'end_date': end_date
    }

    # Set process_maxGen to True to process maximum generation limits,
    # or leave it as False (default) to process only minimum generation limits.
    processor = hydro_mingen_limits_MAF2019(**kwargs_processor)
    result = processor.run_processor()

    # Ensure the output directory exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Handle an optional second DataFrame and possible no result
    if isinstance(result, tuple) and len(result) == 2:
        summary_df, df_optional = result
    elif result is None:
        print(f"processor did not return any DataFrame.")
    else:
        summary_df = result  

    # Write to a csv.
    if summary_df is not None:
        print(f"writing {output_file}")
        summary_df.to_csv(output_file)