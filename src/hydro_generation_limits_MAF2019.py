import os
import time
import pandas as pd
import numpy as np
from datetime import date

class hydro_generation_limits_MAF2019:
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
    """

    def __init__(self, input_folder, country_codes, start_date, end_date, process_maxGen=False):
        self.input_folder = input_folder
        self.country_codes = country_codes
        self.start_date = start_date
        self.end_date = end_date
        self.process_maxGen = process_maxGen

        # Global input file (for non-Norway countries)
        self.input_csv = "PECD-hydro-weekly-reservoir-min-max-generation.csv"

        # Parameters for processing the generation limits.
        self.minvariable_header = "Minimum Generation in MW"
        self.maxvariable_header = "Maximum Generation in MW"
        self.minvariable = "minGen"
        self.maxvariable = "maxGen"
        self.suffix_reservoir = "_reservoir"

        # Norway-specific file parameters.
        self.norway_file_first = "PEMMDB_"
        self.norway_file_last = "_Hydro Inflow_SOR 20.xlsx"
        self.norway_countries = ['NOM1', 'NON1', 'NOS0']

        # Determine the start and end years.
        self.start_year = pd.to_datetime(self.start_date).year
        self.end_year = pd.to_datetime(self.end_date).year

    def process_global_country(self, country, country_df):
        """
        Processes global min/max generation data for one country from the CSV input.

        Parameters:
          country     -- The country code (e.g., 'AT00')
          country_df  -- The country-level CSV DataFrame (already filtered by year)

        Returns:
          A DataFrame with a time index and a "genLimit" level containing the processed data.
          If process_maxGen is False, only the min generation data is returned.
        """
        date_index = pd.date_range(self.start_date, self.end_date, freq='60 min')
        df_lowerBound = pd.DataFrame(index=date_index)
        if self.process_maxGen:
            df_upperBound = pd.DataFrame(index=date_index)

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
                    df_lowerBound.at[t, country + self.suffix_reservoir] = df_year.at[i, self.minvariable_header]
                    if self.process_maxGen:
                        df_upperBound.at[t, country + self.suffix_reservoir] = df_year.at[i, self.maxvariable_header]
                else:
                    if i == 52:
                        # For the final week of the year, use the value at index 51.
                        t = pd.Timestamp(year, 12, 28) + pd.DateOffset(hours=12)
                        df_lowerBound.at[t, country + self.suffix_reservoir] = df_year.at[51, self.minvariable_header]
                        if self.process_maxGen:
                            df_upperBound.at[t, country + self.suffix_reservoir] = df_year.at[51, self.maxvariable_header]

        # Interpolate missing values for min generation.
        df_lowerBound.interpolate(inplace=True, limit=84, limit_direction='both')
        df_lowerBound['genLimit'] = self.minvariable

        if self.process_maxGen:
            # Interpolate missing values for max generation.
            df_upperBound.interpolate(inplace=True, limit=84, limit_direction='both')
            df_upperBound['genLimit'] = self.maxvariable

            # Concatenate the min and max DataFrames.
            df_result = pd.concat([df_lowerBound, df_upperBound])
        else:
            df_result = df_lowerBound.copy()

        df_result = df_result.reset_index()
        df_result = df_result.sort_values(by=['index', 'genLimit'])
        df_result = df_result.set_index(['index', 'genLimit'])
        df_result = df_result.rename_axis(["", "genLimit"], axis="rows")
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
        df_lowerBound = pd.DataFrame(index=date_index)

        try:
            df_input = pd.read_excel(
                filename,
                sheet_name='Pump storage - Open Loop',
                usecols="GN:HV",
                skiprows=12
            )
        except Exception as e:
            print(f"Error reading file {filename}: {e}")
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
                    df_lowerBound.at[t, country + self.suffix_reservoir] = df_year[i]
                else:
                    if i == 52:
                        t = pd.Timestamp(year, 12, 28) + pd.DateOffset(hours=12)
                        df_lowerBound.at[t, country + self.suffix_reservoir] = df_year[51]

        df_lowerBound.interpolate(inplace=True, limit=84, limit_direction='both')
        df_lowerBound['genLimit'] = self.minvariable

        if self.process_maxGen:
            # There are no max values in the Norway Excel files; create a matching empty column.
            df_upperBound = pd.DataFrame(index=date_index)
            df_upperBound[country + self.suffix_reservoir] = np.nan
            df_upperBound['genLimit'] = self.maxvariable

            df_result = pd.concat([df_lowerBound, df_upperBound])
        else:
            df_result = df_lowerBound.copy()

        df_result = df_result.reset_index()
        df_result = df_result.sort_values(by=['index', 'genLimit'])
        df_result = df_result.set_index(['index', 'genLimit'])
        df_result = df_result.rename_axis(["", "genLimit"], axis="rows")
        return df_result

    def run(self):
        """
        Executes the processing steps:
          - Reads the global CSV input for non-Norway countries.
          - Processes each country (using the Norway method when appropriate).
          - Merges all individual DataFrames into a summary and writes it to a CSV file.
        """
        # Read the global CSV input file.
        inputfile = os.path.join(self.input_folder, self.input_csv)
        try:
            global_df = pd.read_csv(inputfile)
        except Exception as e:
            print(f"Error reading input CSV file: {e}")
            return

        # Filter the global DataFrame by year range.
        global_df = global_df[(global_df["year"] >= self.start_year) & (global_df["year"] <= self.end_year)]
        global_df["year"] = pd.to_numeric(global_df["year"])
        global_df["week"] = pd.to_numeric(global_df["week"])

        # Prepare the summary DataFrame with a MultiIndex.
        if self.process_maxGen:
            idx = pd.MultiIndex.from_product(
                [pd.date_range(self.start_date, self.end_date, freq="60 min"), [self.minvariable, self.maxvariable]],
                names=['time', 'genLimit']
            )
        else:
            idx = pd.MultiIndex.from_product(
                [pd.date_range(self.start_date, self.end_date, freq="60 min"), [self.minvariable]],
                names=['time', 'genLimit']
            )
        summary_df = pd.DataFrame(index=idx)
        summary_df = summary_df.rename_axis(["time", "genLimit"], axis="rows")

        # Process each country.
        for country in self.country_codes:
            if country in self.norway_countries:
                filename = os.path.join(self.input_folder, f"{self.norway_file_first}{country}{self.norway_file_last}")
                result_df = self.process_norway_country(country, filename)
            else:
                country_df = global_df[global_df["zone"] == country].copy()
                if country_df.empty:
                    print(f"   No hydro minGen or maxGen data for {country}")
                    continue
                result_df = self.process_global_country(country, country_df)

            if result_df is not None:
                # Ensure the index is named correctly.
                result_df.index = result_df.index.set_names(['time', 'genLimit'])
                summary_df = summary_df.join(result_df, how='left')

        return summary_df


# Example usage:
if __name__ == "__main__":
    input_folder = os.path.join("..\\inputFiles\\timeseries")
    output_folder = os.path.join("..\\inputData-test")
    output_file = os.path.join(output_folder, f'test_hydro_generation_limits.csv')
    country_codes = [
        'FI00', 'FR00', 'UK00', 'LT00', 'LV00', 'NL00', 'NOS0', 'NOM1',
        'NON1', 'PL00', 'SE01', 'SE02', 'SE03', 'SE04'
    ]
    start_date = "1982-01-01 00:00:00"
    end_date = "2021-01-01 00:00:00"

    # Set process_maxGen to True to process maximum generation limits,
    # or leave it as False (default) to process only minimum generation limits.
    processor = hydro_generation_limits_MAF2019(input_folder, country_codes, start_date, end_date, process_maxGen=False)
    summary_df = processor.run()

    # Ensure the output directory exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Write to a csv.
    print(f"writing {output_file}")
    summary_df.to_csv(output_file)