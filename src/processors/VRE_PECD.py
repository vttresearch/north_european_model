# src/processors/VRE_PECD.py

from datetime import datetime
import re
import os
import glob
import pandas as pd
import numpy as np
from src.processors.base_processor import BaseProcessor


class VRE_PECD(BaseProcessor):
    """
    Class to process capacity factor data from CSV files, adjust date indexing,
    and compile a time series DataFrame for a specified date range.

    Parameters:
        input_folder (str): Relative location of input files.
        input_file (str): Folder within input_folder containing the PECD CSV files.
        country_codes (list): List of country codes to filter.
        start_date (str): Start datetime (e.g., '1982-01-01 00:00:00').
        end_date (str): End datetime (e.g., '2021-12-31 23:00:00').
        attached_grid (str): Suffix to append to country codes in output columns.
        scaling_factor (float): Applies logit scaling for the timeseries, e.g. 0.8 reduces the annual average by 20%.

    Returns:
        main_result (pd.DataFrame): DataFrame with processed time series data indexed by datetime
        secondary_result: None (not used for this processor)
        # TBD: secondary_result (pd.DataFrame): Summary table with annual average capacity factors for each country
    """

    def __init__(self, **kwargs):
        # Initialize base class
        super().__init__(**kwargs)
        
        # List of required parameters
        required_params = [
            'input_folder',
            'input_file',
            'country_codes',
            'start_date',
            'end_date',
            'attached_grid'
        ]

        # Check if all required parameters are present
        missing_params = [param for param in required_params if param not in kwargs]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Unpack required parameters
        for param in required_params:
            setattr(self, param, kwargs.get(param))

        # Optional parameters
        self.scaling_factor = kwargs.get('scaling_factor', 1)

        # Ensure start_date and end_date are datetime objects
        if not isinstance(self.start_date, datetime):
            self.start_date = pd.to_datetime(self.start_date)
        if not isinstance(self.end_date, datetime):
            self.end_date = pd.to_datetime(self.end_date)

    def process(self) -> pd.DataFrame:
        """
        Main processing logic that reads, filters, and compiles CSV files into a single DataFrame.

        This method checks that the CSV folder exists and contains CSV files; if not,
        it logs a warning and returns an empty DataFrame. Otherwise, it calls the internal method
        to read and compile the CSV files and renames the country columns with the attached grid suffix.
        
        After processing, it creates a summary table with annual average capacity factors.

        Returns:
            pd.DataFrame: DataFrame indexed by the full date range (from start_date to end_date, hourly)
                         containing columns only for the specified countries with capacity factor values.
        """
        # Create the full path to the CSV folder
        self.csv_folder = os.path.join(self.input_folder, self.input_file)

        # Check if the folder exists
        if not os.path.isdir(self.csv_folder):
            self.log(f"The folder {self.csv_folder} does not exist.", level="warn")
            return pd.DataFrame()

        # Check that the folder contains at least one CSV file
        csv_files = glob.glob(os.path.join(self.csv_folder, "*.csv"))
        if not csv_files:
            self.log(f"No CSV files found in {self.csv_folder}.", level="warn")
            return pd.DataFrame()

        self.log(f"Processing input data in {self.csv_folder}...")
       
        # Extract and compile data using the split methods
        summary_df = self._read_and_compile_input_CSVs(
            self.csv_folder, self.country_codes, self.start_date, self.end_date
        )

        # Apply logit-normal scaling if scaling_factor differs from 1
        if self.scaling_factor != 1:
            self.log(f"Applying logit scaling with factor {self.scaling_factor}...")
            for col in summary_df.columns:
                summary_df[col] = self._apply_logit_scaling(summary_df[col], self.scaling_factor)

        # Rename country columns to indicate the attached grid
        for country in self.country_codes:
            if country in summary_df.columns:
                new_col = f"{country}_{self.attached_grid}"
                summary_df.rename(columns={country: new_col}, inplace=True)

        # Secondary result is None for this processor
        self.secondary_result = None

        self.log("Time series built.")

        return summary_df
    
    def _apply_logit_scaling(self, series, target_scaling, epsilon=1e-6):
        """
        Adjusts capacity factor using Logit-Normal transformation.
        Shifts the mean of a [0,1] bounded series by target_scaling multiplier
        while preserving the overall shape and bounds.
        """
        original_mean = series.mean()
        target_mean = original_mean * target_scaling

        if original_mean == 0:
            return series

        # Map to latent space (logit)
        clipped = np.clip(series.values, epsilon, 1 - epsilon)
        y = np.log(clipped / (1 - clipped))

        # Binary search for the offset
        low, high = -15.0, 15.0
        for _ in range(20):
            mid = (low + high) / 2
            transformed = 1 / (1 + np.exp(-(y + mid)))
            if transformed.mean() < target_mean:
                low = mid
            else:
                high = mid

        final_values = 1 / (1 + np.exp(-(y + mid)))

        # Preserve hard 0s and 1s
        final_values[series == 0] = 0
        final_values[series == 1] = 1

        return pd.Series(final_values, index=series.index)

    def _calculate_annual_summary(self, df):
        """
        Calculate annual average capacity factors from the entire timeseries.
        
        For each column in the DataFrame, this method:
        1. Groups data by year
        2. Calculates the mean for each year
        3. Takes the overall mean across all years
        
        Parameters:
            df (pd.DataFrame): The timeseries DataFrame with datetime index
            
        Returns:
            pd.DataFrame: Summary table with annual average capacity factors
        """
        annual_summary = {}
        
        for col in df.columns:
            # Calculate yearly means, then take the average across all years
            # This gives us the typical annual capacity factor
            yearly_means = df[col].groupby(df.index.year).mean()
            # Overall average across all years
            overall_mean = yearly_means.mean()
            annual_summary[col] = overall_mean
        
        # Create a DataFrame from the summary
        summary_table = pd.DataFrame.from_dict(
            annual_summary, 
            orient='index', 
            columns=['Annual Average']
        ).sort_index()
        
        return summary_table
    
    def _filter_csv_files(self, csv_folder, start_date, end_date):
        """
        Scan the folder for CSV files and filter them based on the date
        information encoded in their filenames.

        Parameters:
            csv_folder (str): Folder containing CSV files.
            start_date (datetime): Start date for filtering.
            end_date (datetime): End date for filtering.

        Returns:
            list: Filtered list of CSV file paths.
        """
        csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
        filtered_files = []

        for file in csv_files:
            filename = os.path.basename(file)
            match = re.search(r'S(\d{12})_E(\d{12})', filename)
            if match:
                file_start_str, file_end_str = match.groups()
                try:
                    file_start = pd.to_datetime(file_start_str, format='%Y%m%d%H%M%S')
                    file_end = pd.to_datetime(file_end_str, format='%Y%m%d%H%M%S')
                    # Check if this file's date range overlaps with our target range
                    if not (file_end < start_date or file_start > end_date):
                        filtered_files.append(file)
                except ValueError:
                    # In case of parsing errors, include the file to be safe
                    filtered_files.append(file)
            else:
                # If filename doesn't match the expected pattern, include the file to be safe
                filtered_files.append(file)

        return filtered_files

    def _process_country_code_mapping(self, df, country_codes):
        """
        Process the country code mapping from the DataFrame columns.
    
        When multiple columns match a country code (e.g., FI00_OFF1, FI00_OFF2, FI00_OFF3),
        this method selects the column with the highest sum of values to prioritize
        regions with better capacity factors and avoid empty timeseries.
        
        For each country code provided, the function performs:
          1. Exact match: Uses the code directly if it exists in the columns.
          2. 4-letter prefix match: Looks for columns that start with the first four letters.
             If multiple matches exist, selects the one with the highest sum.
          3. 2-letter prefix match: If the above fails, looks for columns that start with the first two letters.
             If multiple matches exist, selects the one with the highest sum.
    
        Parameters:
            df (pd.DataFrame): The DataFrame containing the data to evaluate.
            country_codes (list): List of country codes to match.
    
        Returns:
            dict: Mapping where keys are provided country codes and values are the actual column names found.
        """
        mapping = {}
        
        for country_code in country_codes:
            # Try exact match first
            if country_code in df.columns:
                mapping[country_code] = country_code
            else:
                # Try 4-letter prefix matching (if possible)
                prefix4 = country_code[:4]
                matching_columns = [col for col in df.columns if col.startswith(prefix4)]
                
                if len(matching_columns) > 1:
                    # Multiple matches found - select the one with highest sum
                    best_col = None
                    best_sum = -np.inf
                    sums = {}
                    
                    for col in matching_columns:
                        col_sum = df[col].sum()
                        sums[col] = col_sum
                        if col_sum > best_sum:
                            best_sum = col_sum
                            best_col = col
                    
                    if best_col is not None:
                        mapping[country_code] = best_col
                        self.log(f"   {country_code}: Selected '{best_col}' (sum={best_sum:.2f}) from {len(matching_columns)} options: {list(sums.keys())}")
                        
                elif len(matching_columns) == 1:
                    mapping[country_code] = matching_columns[0]
                else:
                    # Try 3-letter prefix matching
                    prefix3 = country_code[:3]
                    matching_columns = [col for col in df.columns if col.startswith(prefix3)]

                    if len(matching_columns) > 1:
                        best_col = None
                        best_sum = -np.inf
                        sums = {}

                        for col in matching_columns:
                            col_sum = df[col].sum()
                            sums[col] = col_sum
                            if col_sum > best_sum:
                                best_sum = col_sum
                                best_col = col

                        if best_col is not None:
                            mapping[country_code] = best_col
                            self.log(f"   {country_code}: Selected '{best_col}' (sum={best_sum:.2f}) from {len(matching_columns)} options: {list(sums.keys())}")

                    elif len(matching_columns) == 1:
                        mapping[country_code] = matching_columns[0]

                    else:
                        # Try 2-letter prefix matching
                        prefix2 = country_code[:2]
                        matching_columns = [col for col in df.columns if col.startswith(prefix2)]

                        if len(matching_columns) > 1:
                            # Multiple matches found - select the one with highest sum
                            best_col = None
                            best_sum = -np.inf
                            sums = {}

                            for col in matching_columns:
                                col_sum = df[col].sum()
                                sums[col] = col_sum
                                if col_sum > best_sum:
                                    best_sum = col_sum
                                    best_col = col

                            if best_col is not None:
                                mapping[country_code] = best_col
                                self.log(f"   {country_code}: Selected '{best_col}' (sum={best_sum:.2f}) from {len(matching_columns)} options: {list(sums.keys())}")

                        elif len(matching_columns) == 1:
                            mapping[country_code] = matching_columns[0]
                        
        return mapping

    def _read_and_process_csv(self, file, country_code_mapping, master_index):
        """
        Read and process a single CSV file.

        This method finds the header row (skipping comment lines that start with "#"),
        reads the CSV, checks for the 'Date' column (converting it to datetime and using it as the index),
        and builds a DataFrame using the country code mapping. The DataFrame is then filtered to include only rows
        present in the master index.

        Parameters:
            file (str): Path to the CSV file.
            country_code_mapping (dict): Mapping from country code to CSV column name.
            master_index (pd.Index): The complete date range index.

        Returns:
            pd.DataFrame or None: Processed DataFrame for the CSV file, or None if errors occur.
        """
        header_row = 0
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                if not line.startswith("#"):
                    header_row = i
                    break

        try:
            df_csv = pd.read_csv(file, skiprows=header_row)
        except Exception as e:
            self.log(f"Error reading file {file}: {e}", level="warn")
            return None

        if 'Date' not in df_csv.columns:
            self.log(f"File {file} does not have a 'Date' column. Skipping the file.", level="warn")
            return None

        df_csv['Date'] = pd.to_datetime(df_csv['Date'])
        df_csv.set_index('Date', inplace=True)

        # Build a temporary DataFrame for the selected country columns
        df_temp = pd.DataFrame(index=df_csv.index)
        for country_code, column_name in country_code_mapping.items():
            if column_name in df_csv.columns:
                df_temp[country_code] = df_csv[column_name]

        # Filter rows to only include indices from the master date range
        df_temp = df_temp[df_temp.index.isin(master_index)]
        return df_temp

    def _read_and_compile_input_CSVs(self, csv_folder, country_codes, start_date, end_date):
        """
        Compile input CSV files into a single DataFrame using the helper subroutines.

        1. Generates a complete date range to serve as the master index.
        2. Uses _filter_csv_files to get the valid CSV files.
        3. Derives the country code mapping from the first valid CSV file (now with sum-based selection).
        4. Iterates over the filtered CSV files, processing each with _read_and_process_csv and updating
           the master DataFrame.

        Parameters:
            csv_folder (str): Folder where CSV files are stored.
            country_codes (list): List of country codes to use.
            start_date (datetime): Start date for the complete date range.
            end_date (datetime): End date for the complete date range.

        Returns:
            pd.DataFrame: DataFrame indexed by the master date range containing the compiled data.
        """
        # Create the complete date range as master index
        date_range = pd.date_range(start=start_date, end=end_date, freq='60min')
        df_csv_summary = pd.DataFrame(index=date_range)

        # Filter CSV files based on date from their filenames
        filtered_files = self._filter_csv_files(csv_folder, start_date, end_date)
        csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
        self.log(f"Using {len(filtered_files)} files within date range from the found {len(csv_files)} files...")

        if not filtered_files:
            self.log(f"No valid CSV files found in '{csv_folder}' after date filtering.", level="warn")
            return df_csv_summary

        # Process country code mapping using the first valid CSV file
        # Now we pass the entire DataFrame instead of just columns to enable sum-based selection
        header_row = 0
        first_file = filtered_files[0]
        with open(first_file, 'r') as f:
            for i, line in enumerate(f):
                if not line.startswith("#"):
                    header_row = i
                    break
        try:
            df_first = pd.read_csv(first_file, skiprows=header_row)
        except Exception as e:
            self.log(f"Error reading the first file {first_file} for mapping: {e}", level="warn")
            return df_csv_summary

        if 'Date' not in df_first.columns:
            self.log(f"File {first_file} does not have a 'Date' column. Cannot determine country code mapping.", level="warn")
            return df_csv_summary

        # Pass the DataFrame instead of just columns for sum-based selection
        country_code_mapping = self._process_country_code_mapping(df_first, country_codes)

        if not country_code_mapping:
            self.log(f"No country code mappings found in '{csv_folder}'. Check CSV column headers vs. country_codes.", level="warn")

        # Process each filtered CSV file and update the master DataFrame
        for file in filtered_files:
            df_temp = self._read_and_process_csv(file, country_code_mapping, df_csv_summary.index)
            if df_temp is None:
                continue
            for col in df_temp.columns:
                if col not in df_csv_summary.columns:
                    df_csv_summary[col] = np.nan
                df_csv_summary.loc[df_temp.index, col] = df_temp[col]

        return df_csv_summary
