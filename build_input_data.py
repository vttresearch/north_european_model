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
#├── build_input_data.py
#├── src/
#    ├── create_timeseries.py  
#    ├── create_input_excel.py  


# # Import main processor classes 
from src.create_timeseries import create_timeseries


# ------------------------------------------------------
# Utility function to log elapsed time for progress tracking
# ------------------------------------------------------
def log_time(message, run_start):
    # small helper function used to track the progress
    elapsed = time.perf_counter() - run_start
    print(f"[{elapsed:0.2f} s] {message}")


# ------------------------------------------------------
# Function to process multiple demand files
# ------------------------------------------------------
def process_demand_files(demand_files, scenario, scenario_year):
    """
    Processes multiple demand files by:
      1. Reading all sheets (except "readme") from each file.
      2. Merging sheets into a single DataFrame.
      3. Checking for duplicate entries based on key columns.
      4. Filtering for rows matching the specified scenario, grid, and scenario year.

    Parameters:
      demand_files (list of str): List of Excel file names (assumed to be in input_folder).

    Returns:
      pd.DataFrame: Merged and filtered DataFrame from all demand files.

    Raises:
      FileNotFoundError: If any of the files cannot be opened.
      ValueError: If duplicate entries are detected in any file.
    """
    all_filtered_demands = []  # List to store filtered DataFrames from each file

    # Loop over each file name in the demand_files list.
    for file_name in demand_files:
        file_path = os.path.join(input_folder, file_name)
        try:
            # Load the Excel file to access sheet names.
            excel_file = pd.ExcelFile(file_path)
            # Exclude any sheet named 'readme' (case insensitive).
            sheet_names = [sheet for sheet in excel_file.sheet_names if sheet.lower() != 'readme']
            # Read each sheet and store the resulting DataFrame in a list.
            df_list = [pd.read_excel(excel_file, sheet_name=sheet, header=0) for sheet in sheet_names]
            # Merge the DataFrames from all sheets.
            df_annual_demands = pd.concat(df_list, ignore_index=True)
        except Exception as e:
            raise FileNotFoundError(f"Unable to open {file_path}: {e}")

        # --- Check for invalid grid values containing an underscore ---
        if df_annual_demands['Grid'].astype(str).str.contains('_', na=False).any():
            bad_values = df_annual_demands.loc[
                df_annual_demands['Grid'].astype(str).str.contains('_', na=False), 'Grid'
            ].unique()
            raise ValueError(
                f"Invalid grid values containing underscores found in file {file_name}: {bad_values}"
            )

        # --- Check for duplicate entries based on key columns ---
        key_columns = ['Country', 'Grid', 'Node_suffix', 'Scenario', 'Year']
        if df_annual_demands.duplicated(subset=key_columns).any():
            duplicates = df_annual_demands[df_annual_demands.duplicated(subset=key_columns, keep=False)]
            raise ValueError(
                f"Duplicate entries detected in file {file_name} based on columns {key_columns}.\n"
                f"Duplicate rows:\n{duplicates}"
            )

        # --- Filter for rows matching the given scenario and model year ---
        df_filtered_demands = df_annual_demands[
            (df_annual_demands['Scenario'].str.lower() == scenario.lower()) &
            (df_annual_demands['Year'] == scenario_year)
        ]
        all_filtered_demands.append(df_filtered_demands)

    # Merge the filtered DataFrames from all files into one.
    if all_filtered_demands:
        merged_demands = pd.concat(all_filtered_demands, ignore_index=True)
    else:
        merged_demands = pd.DataFrame()  # Return an empty DataFrame if no data was found

    return merged_demands


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
    country_codes = ast.literal_eval(config.get('InputData', 'country_codes'))
    demand_files = ast.literal_eval(config.get('InputData', 'demand_files'))
    scenarios = ast.literal_eval(config.get('InputData', 'scenarios'))
    scenario_years = ast.literal_eval(config.get('InputData', 'scenario_years'))
    timeseries_specs = ast.literal_eval(config.get('InputData', 'timeseries_specs'))

    # Loop over every combination of scenario and scenario year.
    for scenario, s_year in itertools.product(scenarios, scenario_years):
        print(f"\n--------------------------------------------------------------------------- ")
        log_time(f"---- processing {scenario}, {s_year} ---------------- ", run_start)

        # Process the demand files to get the data for the current scenario and year.
        df_annual_demands = process_demand_files(demand_files, scenario, s_year)

        # Create an output folder specific to the current scenario and year
        output_folder = os.path.join(f"{output_folder_prefix}_{scenario}_{str(s_year)}")
        if not os.path.exists(output_folder): 
            os.makedirs(output_folder)
        print(f"Writing output files to '.\\{output_folder}'")

        # Create the timeseries using the create_timeseries class.
        # Pass all necessary parameters including timing, input/output paths, dates, country codes, etc.
        create_timeseries(timeseries_specs, run_start, input_folder, output_folder, start_date, end_date,
                     country_codes, df_annual_demands, scenario, s_year, write_csv_files).run()
        
    print("\n-----------------------------------")   
    log_time(f"All (scenario, year) pairs processed.", run_start)


# ------------------------------------------------------
# Main entry point for the script
# ------------------------------------------------------
if __name__ == '__main__':
    # Ensure the user provided the necessary command-line arguments.
    if len(sys.argv) < 3:
        print("Usage: python build_input_data.py <input_folder> <config_file>, e.g. python build_input_data.py inputFiles config.ini")
        sys.exit(1)
    
    # The first argument is the input folder.
    input_folder = sys.argv[1]
    # The second argument is the configuration file name.
    filename = sys.argv[2]
    # Start the main process with the provided arguments.
    run(input_folder, filename)