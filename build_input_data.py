import os
import sys
import ast
import time
# gdxpds not needed here, but important to import before pandas
from gdxpds import to_gdx
import pandas as pd
import itertools
import configparser


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
# Utility function to log elapsed time for progress tracking
# ------------------------------------------------------
def log_time(message, run_start):
    # small helper function used to track the progress
    elapsed = time.perf_counter() - run_start
    print(f"[{elapsed:0.2f} s] {message}")


# ------------------------------------------------------
# Function to merge input excels
# ------------------------------------------------------
def merge_excel_files(excel_files, sheet_name_prefix):
    """
    Merges multiple Excel files by:
      1. Reading sheets starting with sheet_name_prefix from each file.
      2. Checking for duplicate entries within each sheet (using all columns except 'value', case insensitive).
      3. Merging all sheets into a single DataFrame.
    
    Parameters:
      excel_files (list of str): List of Excel file names (assumed to be in input_folder).
      sheet_name_prefix (str): Prefix to identify sheets to read.
    
    Raises:
      FileNotFoundError: If any of the files cannot be opened.
      ValueError: If duplicate entries (based on all columns) are detected in any sheet.
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


# ------------------------------------------------------
# Functions for quality check of read data
# ------------------------------------------------------

def quality_check(df_input):
    # Convert all column names to lower case to ensure consistent naming
    df_input.columns = df_input.columns.str.lower()

    # Convert all content in 'scenario' column to lower case if the column exists
    if 'scenario' in df_input.columns:
        df_input['scenario'] = df_input['scenario'].astype(str).str.lower()
    
    # Check that the grid values do not contain underscores
    if df_input['grid'].astype(str).str.contains('_', na=False).any():
        bad_values = df.loc[df['grid'].astype(str).str.contains('_', na=False), 'grid'].unique()
        raise ValueError(f"Invalid grid values containing underscores found: {bad_values}")
    
    # check that from-to contains one and only one -
    
    return df_input


# ------------------------------------------------------
# Function to filter input dataframe by scenario and scenario_year
# ------------------------------------------------------
def filter_by_scenario_and_year(df_input, scenario, scenario_year):

    # --- Filter for rows matching the given scenario and model year ---
    df_filtered = df_input[
        (df_input['scenario'] == scenario.lower()) &
        (df_input['year'] == scenario_year)
        ]

    return df_filtered


# ------------------------------------------------------
# Main run function to process input data and create timeseries
# ------------------------------------------------------
def run(input_folder, config_file):
    # Start the timer for the entire run.
    run_start = time.perf_counter()

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
    demand_files = ast.literal_eval(config.get('InputData', 'demand_files'))
    transfer_files = ast.literal_eval(config.get('InputData', 'transfer_files'))
    timeseries_specs = ast.literal_eval(config.get('InputData', 'timeseries_specs'))


    # Read input excels
    try:
        df_annual_demands = merge_excel_files(demand_files, 'demands_')
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)

    df_annual_demands = quality_check(df_annual_demands)

    try:
        df_transfers = merge_excel_files(transfer_files, 'transfers_')
    except FileNotFoundError as e:
        print(e)
        sys.exit(1)
    
    df_transfers = quality_check(df_transfers)


    # Loop over every combination of scenario and scenario year.
    for scenario, scenario_year in itertools.product(scenarios, scenario_years):
        print(f"\n--------------------------------------------------------------------------- ")
        log_time(f"---- processing {scenario}, {scenario_year} ---------------- ", run_start)

        # Process demand files to get the data for the current scenario and year.
        df_filtered_demands = filter_by_scenario_and_year(df_annual_demands, scenario, scenario_year)

        # Process transfer files to get the data for the current scenario and year.
        df_filtered_transfers = filter_by_scenario_and_year(df_transfers, scenario, scenario_year)

        # Create an output folder specific to the current scenario and year
        output_folder = os.path.join(f"{output_folder_prefix}_{scenario}_{str(scenario_year)}")
        if not os.path.exists(output_folder): 
            os.makedirs(output_folder)
        print(f"Writing output files to '.\\{output_folder}'")

        # Build input excel using the build_input_excel class.    
        build_input_excel(input_folder, output_folder, country_codes, df_filtered_transfers, scenario, scenario_year).run()

        # Create the timeseries using the create_timeseries class.
        create_timeseries(timeseries_specs, run_start, input_folder, output_folder, start_date, end_date,
                     country_codes, df_filtered_demands, scenario, scenario_year, write_csv_files).run()
        
    print("\n-----------------------------------")   
    log_time(f"All (scenario, year) pairs processed.", run_start)



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