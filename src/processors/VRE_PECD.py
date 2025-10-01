from datetime import datetime
import re
import os
import glob
import pandas as pd
import numpy as np
from src.utils import log_status


class VRE_PECD:
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

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame with processed time series data indexed by datetime,
        or None if the input folder or CSV files are not found.
    """

    def __init__(self, **kwargs_processor):
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
        missing_params = [param for param in required_params if param not in kwargs_processor]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Unpack required parameters
        for param in required_params:
            setattr(self, param, kwargs_processor.get(param))
            
        # Initialize log message list
        self.processor_log = []


    def run_processor(self):
        """
        Process input CSV files by reading, filtering, and compiling them into a single DataFrame.

        This method checks that the CSV folder exists and contains CSV files; if not,
        it prints a warning and returns None. Otherwise, it calls the internal method
        to read and compile the CSV files and renames the country columns with the attached grid suffix.

        Returns:
            pandas.DataFrame
                DataFrame indexed by the full date range (from start_date to end_date, hourly)
                containing columns only for the specified countries with capacity factor values.
        """
        # Create the full path to the CSV folder
        self.csv_folder = os.path.join(self.input_folder, self.input_file)

        # Check if the folder exists
        if not os.path.isdir(self.csv_folder):
            log_status(f"The folder {self.csv_folder} does not exist.", self.processor_log, level="warn")
            return None

        # Check that the folder contains at least one CSV file
        csv_files = glob.glob(os.path.join(self.csv_folder, "*.csv"))
        if not csv_files:
            log_status(f"No CSV files found in {self.csv_folder}.", self.processor_log, level="warn")
            return None

        log_status(f"Processing input data in {self.csv_folder}...", self.processor_log, level="info")
       
        # Extract and compile data using the split methods
        summary_df = self._read_and_compile_input_CSVs(
            self.csv_folder, self.country_codes, self.start_date, self.end_date
        )

        # Rename country columns to indicate the attached grid
        for country in self.country_codes:
            if country in summary_df.columns:
                new_col = f"{country}_{self.attached_grid}"
                summary_df.rename(columns={country: new_col}, inplace=True)

        # Mandatory secondary results
        secondary_result = None

        log_status("Time series built.", self.processor_log, level="info")

        # Note: returning processor log as a string, because then we can distinct it from secondary results which might be a list of strings
        return summary_df, secondary_result, "\n".join(self.processor_log)
    

    def _filter_csv_files(self, csv_folder, start_date, end_date):
        """
        Scan the folder for CSV files and filter them based on the date
        information encoded in their filenames.

        Parameters:
            csv_folder (str): Folder containing CSV files.
            start_date (datetime or str): Start date for filtering.
            end_date (datetime or str): End date for filtering.

        Returns:
            list: Filtered list of CSV file paths.
        """
        csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
        filtered_files = []

        # Ensure start_date and end_date are datetime objects
        if not isinstance(start_date, datetime):
            start_date = pd.to_datetime(start_date)
        if not isinstance(end_date, datetime):
            end_date = pd.to_datetime(end_date)

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

    def _process_country_code_mapping(self, df_columns, country_codes):
        """
        Process the country code mapping from the DataFrame columns.
    
        It assumes that each CSV has the same mapping. For each country code provided,
        the function performs:
          1. Exact match: Uses the code directly if it exists in the columns.
          2. 4-letter prefix match: Looks for a column that starts with the first four letters.
          3. 2-letter prefix match: If the above fails, it looks for a column that starts with the first two letters.
    
        Parameters:
            df_columns (iterable): Column headers of the CSV file.
            country_codes (list): List of country codes to match.
    
        Returns:
            dict: Mapping where keys are provided country codes and values are the actual column names found.
        """
        mapping = {}
        for country_code in country_codes:
            # Try exact match first
            if country_code in df_columns:
                mapping[country_code] = country_code
            else:
                # Try 4-letter prefix matching (if possible)
                prefix4 = country_code[:4]
                matching_columns = [col for col in df_columns if col.startswith(prefix4)]
                if matching_columns:
                    mapping[country_code] = matching_columns[0]
                else:
                    # Try 2-letter prefix matching
                    prefix2 = country_code[:2]
                    matching_columns = [col for col in df_columns if col.startswith(prefix2)]
                    if matching_columns:
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
            master_index (pandas.Index): The complete date range index.

        Returns:
            pandas.DataFrame or None: Processed DataFrame for the CSV file, or None if errors occur.
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
            log_status(f"Error reading file {file}: {e}", self.processor_log, level="warn")
            return None

        if 'Date' not in df_csv.columns:
            log_status(f"File {file} does not have a 'Date' column. Skipping the file.", self.processor_log, level="warn")
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
        3. Derives the country code mapping from the first valid CSV file.
        4. Iterates over the filtered CSV files, processing each with _read_and_process_csv and updating
           the master DataFrame.

        Parameters:
            csv_folder (str): Folder where CSV files are stored.
            country_codes (list): List of country codes to use.
            start_date (str or datetime): Start date for the complete date range.
            end_date (str or datetime): End date for the complete date range.

        Returns:
            pandas.DataFrame: DataFrame indexed by the master date range containing the compiled data.
        """
        # Create the complete date range as master index
        date_range = pd.date_range(start=start_date, end=end_date, freq='60min')
        df_csv_summary = pd.DataFrame(index=date_range)

        # Ensure start_date and end_date are datetime objects
        if not isinstance(start_date, datetime):
            start_date = pd.to_datetime(start_date)
        if not isinstance(end_date, datetime):
            end_date = pd.to_datetime(end_date)

        # Filter CSV files based on date from their filenames
        filtered_files = self._filter_csv_files(csv_folder, start_date, end_date)
        csv_files = glob.glob(os.path.join(csv_folder, "*.csv"))
        log_status(f"Using {len(filtered_files)} files within date range from the found {len(csv_files)} files...", self.processor_log)

        if not filtered_files:
            log_status("No valid CSV files found after filtering.", self.processor_log, level="warn")
            return df_csv_summary

        # Process country code mapping using the first valid CSV file
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
            log_status(f"Error reading the first file {first_file} for mapping: {e}", self.processor_log, level="warn")
            return df_csv_summary

        if 'Date' not in df_first.columns:
            log_status(f"File {first_file} does not have a 'Date' column. Cannot determine country code mapping.", self.processor_log, level="warn")
            return df_csv_summary

        country_code_mapping = self._process_country_code_mapping(df_first.columns, country_codes)

        if not country_code_mapping:
            log_status("No country code mappings found.", self.processor_log, level="warn")

        # Extract only those mappings where the country code differs from the column name
        alternative_mappings = {code: col for code, col in country_code_mapping.items() if code != col}      
        if alternative_mappings:
            log_status("Alternative country code mappings used:", self.processor_log, level="info")
            for country_code, col in alternative_mappings.items():
                log_status(f"   {country_code}: {col}", self.processor_log)

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


# __main__ allows testing by calling this .py file directly.
if __name__ == "__main__":
    # Define parameters needed by the class
    params = {
        'input_folder': '../src_files/timeseries',
        # Change the input_file value based on the folder for PECD data.
        #'input_file': 'PECD-PV',
        #'input_file': 'PECD-onshore',
        'input_file': 'PECD-offshore',
        'country_codes': ['FI00', 'EE00', 'SE04'],  # example country codes
        'start_date': '2000-01-01 00:00:00',
        'end_date': '2009-12-31 23:00:00',
        'attached_grid': 'elec'
    }

    # Create an instance and run the processing method
    processor = VRE_PECD(**params)
    result_df = processor.run_processor()

    if result_df is not None:
        print(result_df.head())
