import os
import sys
import ast
import time
# gdxpds not needed here, but important to import before pandas
from gdxpds import to_gdx
import pandas as pd
import itertools
import configparser
import warnings


# ------------------------------------------------------
# Importing data building main functions
# ------------------------------------------------------

# Folder structure of this project
# NorthEuropeanBackbone/
#├── build_input_excel.py
#├── src/
#    ├── create_timeseries.py  
#    ├── create_input_excel.py  


# Import main processor classes
from src.build_input_excel import build_input_excel
from src.create_timeseries import create_timeseries



# ------------------------------------------------------
# Functions to merge input excels, carry a quality check, and simplify the code by calling these both via an aggregate fucntion
# ------------------------------------------------------
def merge_excel_files(excel_files, sheet_name_prefix):
    """
    Merges multiple Excel files by:
      1. Reading sheets starting with sheet_name_prefix from each file.
      2. Checking for duplicate entries within each sheet (using all columns except 'value', case insensitive).
      3. Merging all sheets into a single DataFrame.
      4. Overwriting duplicate 'value' entries when duplicates occur across sheets.
    
    Parameters:
      excel_files (list of str): List of Excel file names (assumed to be in input_folder).
      sheet_name_prefix (str): Prefix to identify sheets to read.
    
    Raises:
      FileNotFoundError: If any of the files cannot be opened.
      ValueError: If duplicate entries (based on all columns) are detected in any sheet.
      ValueError: If an Excel file has no sheet starting with sheet_name_prefix.
    """
    excel_sheets = []  # List to store DataFrames from each sheet
    
    for file_name in excel_files:
        file_path = os.path.join(input_folder, file_name)
        try:
            excel_file = pd.ExcelFile(file_path)
        except Exception as e:
            raise FileNotFoundError(f"Unable to open {file_path}: {e}")
        
        # Get sheets that match the given prefix (case-insensitive)
        sheet_names = [sheet for sheet in excel_file.sheet_names if sheet.lower().startswith(sheet_name_prefix.lower())]

        # Raise ValueError if no matching sheet is found
        if not sheet_names:
            raise ValueError(f"Excel file '{file_name}' does not contain any sheet starting with '{sheet_name_prefix}'.")
        
        # Process each matching sheet individually
        for sheet in sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet, header=0)
            # Define columns to check for duplicates, excluding any column named 'value' (case-insensitive)
            columns_to_check = [col for col in df.columns if col.lower() != 'value']
            
            if df.duplicated(subset=columns_to_check).any():
                duplicates = df[df.duplicated(subset=columns_to_check, keep=False)]
                raise ValueError(
                    f"Duplicate entries detected in file '{file_name}', sheet '{sheet}' based on columns {columns_to_check}.\n"
                    f"Duplicate rows:\n{duplicates}"
                )
            excel_sheets.append(df)
    
    # Merge all validated sheets into one DataFrame
    merged_df = pd.concat(excel_sheets, ignore_index=True) if excel_sheets else pd.DataFrame()
    return merged_df

def quality_check(df_input, df_identifier):
    # Check if the input DataFrame is empty
    if df_input.empty:
        raise TypeError(f"Abort: The input DataFrame '{df_identifier}' is empty, cannot proceed with input data generation.")

    # Convert all column names to lower case to ensure consistent naming
    df_input.columns = df_input.columns.str.lower()

    # Convert certain key columns to lower case if the columns exist
    cols = ['scenario', 'generator_id']
    for col in cols:
        if col in df_input.columns:
            df_input[col] = df_input[col].astype(str).str.lower()
    
    # Check that all columns with 'grid' in their name do not contain underscores
    grid_columns = [col for col in df_input.columns if 'grid' in col.lower()]
    for col in grid_columns:
        if df_input[col].astype(str).str.contains('_', na=False).any():
            bad_values = df_input.loc[df_input[col].astype(str).str.contains('_', na=False), col].unique()
            raise ValueError(f"Invalid values in dataframe '{df_identifier}' in column '{col}' containing underscores found: {bad_values}")
    
    # Check that if there is a column called 'from-to', it contains exactly one hyphen in each value
    if 'from-to' in df_input.columns:
        invalid_mask = df_input['from-to'].astype(str).apply(lambda x: x.count('-') != 1)
        if invalid_mask.any():
            bad_values = df_input.loc[invalid_mask, 'from-to'].unique()
            raise ValueError(f"Invalid 'from-to' values in dataframe '{df_identifier}': must contain exactly one '-' but found: {bad_values}")

    return df_input


def process_dataset(files, prefix):
    """
    Merge excel files and perform a quality check for a given dataset.
    
    Parameters:
        files (list): List of file paths.
        prefix (str): A string prefix to be used for the dataset.
    
    Returns:
        DataFrame: The processed DataFrame after merging and quality check.
    """
    try:
        df = merge_excel_files(files, prefix)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)
    
    try:
        df = quality_check(df, prefix)
    except (TypeError, ValueError) as e:
        print(e)
        sys.exit(1)
    
    return df



# ------------------------------------------------------
# Smaller utility functions
# ------------------------------------------------------

def log_time(message, log_start):
    # small helper function used to track the progress
    elapsed = time.perf_counter() - log_start
    print(f"[{elapsed:0.2f} s] {message}")


def keep_last_occurance(df_input, key_columns=None):
    # If key_columns is not provided, default to all columns except 'value'
    if key_columns is None:
        key_columns = [col for col in df_input.columns if col.lower() != 'value']
    else:
        # Keep only the columns that exist in the DataFrame
        key_columns = [col for col in key_columns if col in df_input.columns]
    
    # If the DataFrame is not empty and we have valid key columns, drop duplicates
    if not df_input.empty and key_columns:
        df_input = df_input.drop_duplicates(subset=key_columns, keep='last')
    return df_input


# ------------------------------------------------------
# Functions to filter dataframes
# ------------------------------------------------------

def filter_df_whitelist(df_input, df_name, filters):
    """
    Parameters:
        df_input (pd.DataFrame): The DataFrame to be filtered.
        filters (dict): Dictionary where keys are column names and values are the desired filter values.
                        For example: {'scenario': 'National Trends', 'year': 2025}. 
                        For example: {'scenario': ['National Trends', 'Global Ambition'], 'year': [2025, 2030]}.
        df_name (str): Name of the DataFrame (used in exception messages).
    """
    # Check that all filter columns exist in the DataFrame
    missing_cols = [col for col in filters if col not in df_input.columns]
    if missing_cols:
        raise Exception(f"Column(s) {missing_cols} are missing from {df_name if df_name else 'DataFrame'}.")

    # Apply the filters iteratively
    df_filtered = df_input.copy()
    for col, val in filters.items():
        if isinstance(val, list):
            if pd.api.types.is_string_dtype(df_filtered[col]):
                # Convert both the column and the list values to lower case for a case-insensitive comparison
                lowered_vals = [v.lower() for v in val if isinstance(v, str)]
                df_filtered = df_filtered[df_filtered[col].str.lower().isin(lowered_vals)]
            else:
                df_filtered = df_filtered[df_filtered[col].isin(val)]
        else:
            # If the column contains strings and the filter value is a string, do a case-insensitive comparison
            if pd.api.types.is_string_dtype(df_filtered[col]) and isinstance(val, str):
                df_filtered = df_filtered[df_filtered[col].str.lower() == val.lower()]
            else:
                df_filtered = df_filtered[df_filtered[col] == val]
    return df_filtered


def filter_df_blacklist(df_input, df_name, filters):
    """
    Parameters:
        df_input (pd.DataFrame): The DataFrame to be filtered.
        filters (dict): Dictionary where keys are column names and values are the values to blacklist.
                        For example: {'scenario': 'National Trends', 'year': 2025}.
                        For example: {'scenario': ['National Trends', 'Global Ambition'], 'year': [2025, 2030]}.
        df_name (str): Name of the DataFrame (used in exception messages).
    """
    # Check that all filter columns exist in the DataFrame
    missing_cols = [col for col in filters if col not in df_input.columns]
    if missing_cols:
        raise Exception(f"Column(s) {missing_cols} are missing from {df_name if df_name else 'DataFrame'}.")

    # Apply the filters iteratively to remove blacklisted rows
    df_filtered = df_input.copy()
    for col, val in filters.items():
        if isinstance(val, list):
            if pd.api.types.is_string_dtype(df_filtered[col]):
                # Convert both the column and the list values to lower case for a case-insensitive comparison
                lowered_vals = [v.lower() for v in val if isinstance(v, str)]
                df_filtered = df_filtered[~df_filtered[col].str.lower().isin(lowered_vals)]
            else:
                df_filtered = df_filtered[~df_filtered[col].isin(val)]
        else:
            # If the column contains strings and the filter value is a string, do a case-insensitive comparison
            if pd.api.types.is_string_dtype(df_filtered[col]) and isinstance(val, str):
                df_filtered = df_filtered[df_filtered[col].str.lower() != val.lower()]
            else:
                df_filtered = df_filtered[df_filtered[col] != val]
    return df_filtered


# ------------------------------------------------------
# Main run function to process input data and create timeseries
# ------------------------------------------------------
def run(input_folder, config_file):
    # Start the timer for the entire run.
    log_start = time.perf_counter()

    # Build the full path to the configuration file.
    config_file_path = os.path.join(input_folder, config_file)
    # Exit if the configuration file cannot be found.
    if not os.path.isfile(config_file_path):
        print(f"Config file not found from '{config_file_path}', check spelling and folder.")
        sys.exit(1)

    # Read and parse the configuration file.
    config = configparser.ConfigParser()
    config.read(config_file_path)

    # Extract parameters from the config file under the 'InputData' section.
    output_folder_prefix = config.get('InputData', 'output_folder_prefix')
    write_csv_files = config.getboolean('InputData', 'write_csv_files')
    start_date = config.get('InputData', 'start_date')
    end_date = config.get('InputData', 'end_date')
    # Convert string representations of lists into actual Python lists.
    scenarios = ast.literal_eval(config.get('InputData', 'scenarios'))
    scenario_years = ast.literal_eval(config.get('InputData', 'scenario_years'))
    country_codes = ast.literal_eval(config.get('InputData', 'country_codes'))
    exclude_grids = ast.literal_eval(config.get('InputData', 'exclude_grids'))
    demand_files = ast.literal_eval(config.get('InputData', 'demand_files'))
    transfer_files = ast.literal_eval(config.get('InputData', 'transfer_files'))
    techdata_files = ast.literal_eval(config.get('InputData', 'techdata_files'))
    unitcapacities_files = ast.literal_eval(config.get('InputData', 'unitcapacities_files'))
    fueldata_files = ast.literal_eval(config.get('InputData', 'fueldata_files'))
    emissiondata_files = ast.literal_eval(config.get('InputData', 'emissiondata_files'))
    # timeseries processing specs dictionary
    timeseries_specs = ast.literal_eval(config.get('InputData', 'timeseries_specs'))

    # Read and process input excels
    df_demands   = process_dataset(demand_files, 'demands')
    df_transfers = process_dataset(transfer_files, 'transfers')
    df_techs  = process_dataset(techdata_files, 'techdata') 
    df_unitcapacities  = process_dataset(unitcapacities_files, 'unitcapacities') 
    df_fuels  = process_dataset(fueldata_files, 'fueldata') 
    df_emissions  = process_dataset(emissiondata_files, 'emissiondata') 

    # exclude grids
    df_demands = filter_df_blacklist(df_demands, 'transfer_files', {'grid': exclude_grids})
    df_transfers = filter_df_blacklist(df_transfers, 'transfer_files', {'grid': exclude_grids})

    # remove duplicates
    df_demands = keep_last_occurance(df_demands, ['country', 'grid', 'node_suffix', 'scenario', 'year'])
    df_transfers = keep_last_occurance(df_transfers, ['from-to', 'grid', 'scenario', 'year'])
    df_techs = keep_last_occurance(df_techs, ['scenario', 'year', 'generator_id'])
    df_unitcapacities = keep_last_occurance(df_unitcapacities, ['country', 'generator_id', 'unit_name_prefix', 'scenario', 'year'])
    df_fuels = keep_last_occurance(df_fuels, ['scenario', 'year', 'fuel'])
    df_emissions = keep_last_occurance(df_emissions, ['scenario', 'year', 'emission'])



    # Loop over every combination of scenario and scenario year.
    for scenario, scenario_year in itertools.product(scenarios, scenario_years):
        print(f"\n--------------------------------------------------------------------------- ")
        log_time(f"---- processing {scenario}, {scenario_year} ---------------- ", log_start)

        # Process global input files to get the data for the current scenario and year.
        df_f_techs = filter_df_whitelist(df_techs, 'techdata_files', {'scenario':scenario, 'year':scenario_year})     
        df_f_fuels = filter_df_whitelist(df_fuels, 'fueldata_files', {'scenario':scenario, 'year':scenario_year})
        df_f_emissions = filter_df_whitelist(df_emissions, 'emissiondata_files' , {'scenario':scenario, 'year':scenario_year})
        df_f_transfers = filter_df_whitelist(df_transfers, 'transfer_files', {'scenario':scenario, 'year':scenario_year})

        # Process country specific input files to get the data for the current scenario, year, and country.
        df_f_demands = filter_df_whitelist(df_demands, 'demand_files', {'scenario':scenario, 'year':scenario_year, 'country': country_codes})
        df_f_unitcapacities = filter_df_whitelist(df_unitcapacities, 'unitcapacity_files', {'scenario':scenario, 'year':scenario_year, 'country': country_codes})
        
        # Create an output folder specific to the current scenario and year
        output_folder = os.path.join(f"{output_folder_prefix}_{scenario}_{str(scenario_year)}")
        if not os.path.exists(output_folder): 
            os.makedirs(output_folder)
        print(f"Writing output files to '.\\{output_folder}'")

        # Build input excel using the build_input_excel class.    
        build_input_excel(input_folder, output_folder, 
                          scenario, scenario_year, country_codes, exclude_grids,
                          df_f_transfers, df_f_techs, df_f_unitcapacities, df_f_fuels, df_f_emissions 
                          ).run()

        # Create the timeseries using the create_timeseries class.
        create_timeseries(timeseries_specs, input_folder, output_folder, 
                          start_date, end_date, country_codes, scenario, scenario_year, 
                          df_f_demands, log_start, write_csv_files
                          ).run()
        
    print("\n-----------------------------------")   
    log_time(f"All (scenario, year) pairs processed.", log_start)



def parse_args(argv):
    args = {}
    for arg in argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
    return args

# ------------------------------------------------------
# Main entry point for the script
# ------------------------------------------------------
if __name__ == '__main__':
    args = parse_args(sys.argv)
    
    if 'input_folder' not in args or 'config_file' not in args:
        print("Usage: python build_input_data.py input_folder=<input_folder> config_file=<config_file>, e.g. python build_input_data.py input_folder=src_files config_file=config_test.ini")
        sys.exit(1)
    
    input_folder = args['input_folder']
    filename = args['config_file']
    
    run(input_folder, filename)