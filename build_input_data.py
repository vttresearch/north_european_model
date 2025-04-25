import os
import sys
import ast
import time
# gdxpds not needed by this file, but it is important to import it before pandas in whole project
from gdxpds import to_gdx
import pandas as pd
import itertools
import configparser
import shutil


# ------------------------------------------------------
# Importing data building main functions
# ------------------------------------------------------

# Folder structure of this project
# NorthEuropeanBackbone/
#├── build_input_excel.py
#├── src/
#    ├── create_timeseries.py  
#    ├── create_input_excel.py  
#    ├── multiple processor files 
#├── src_files/
#    ├── GAMS_files
#    ├── timeseries
#    ├── data_files

# Import main processor classes
from src.build_input_excel import build_input_excel
from src.create_timeseries import create_timeseries



# ------------------------------------------------------
# Functions to merge input excels, carry a quality check, and simplify the code by calling these both via an aggregate fucntion
# ------------------------------------------------------

def process_dataset(input_folder, files, prefix, isMandatory=True):
    """
    Merge excel files and perform a quality check for a given dataset.
    
    Parameters:
        input_folder (str): Location of files
        files (list): List of file paths.
        prefix (str): A string prefix to be used for the dataset.
    
    Returns:
        DataFrame: The processed DataFrame after merging and quality check.
    """
    try:
        df = merge_excel_files(input_folder, files, prefix, isMandatory)
    except (FileNotFoundError, ValueError) as e:
        print(e)
        sys.exit(1)
    
    try:
        df = quality_check(df, prefix)
    except (TypeError, ValueError) as e:
        print(e)
        sys.exit(1)
    
    return df

def merge_excel_files(input_folder, excel_files, sheet_name_prefix, isMandatory=True):
    """
    Merges multiple Excel files by:
      1. Reading sheets starting with sheet_name_prefix from each file.
      2. Checking for duplicate entries within each sheet (using all columns except 'value', case insensitive).
      3. Merging all sheets into a single DataFrame.
      4. Overwriting duplicate 'value' entries when duplicates occur across sheets.
    
    Parameters:
      input_folder (str): location of excel files
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
        if isMandatory and not sheet_names:
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
            
            # drop column "Note"
            df = df.drop(['note', 'Note'], axis=1, errors='ignore')

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

def parse_args(argv):
    args = {}
    for arg in argv[1:]:
        if '=' in arg:
            key, value = arg.split('=', 1)
            args[key] = value
    return args

def copy_GAMS_files(input_folder, output_folder):
    # Define the folder containing the GAMS files
    gams_files_folder = os.path.join(input_folder, "GAMS_files")
    
    # Define a list of tuples (source file, destination file)
    files_to_copy = [
        ("1_options.gms", "1_options.gms"),
        ("changes.inc", "changes.inc"),
        ("modelsInit_example.gms", "modelsInit.gms"),
        ("scheduleInit_example.gms", "scheduleInit.gms"),
        ("timeAndSamples.inc", "timeAndSamples.inc")
    ]
    
    # Loop through the files and copy them to the output folder with proper naming
    for src_filename, dst_filename in files_to_copy:
        src_path = os.path.join(gams_files_folder, src_filename)
        dst_path = os.path.join(output_folder, dst_filename)
        
        # Optionally, you can add error handling in case a file doesn't exist.
        if not os.path.isfile(src_path):
            print(f"Warning: {src_path} does not exist.")
            continue
        
        shutil.copy2(src_path, dst_path)
        print(f"Copied {src_path} to {dst_path}")


def build_node_column(df):
    """
    Add 'node' column to df: 
        * if node_suffix defined: <country>_<grid>_<node_suffix>  
        * else : <country>_<grid>
    requires following columns: country, grid
    optional column: node_suffix
    """
    # Define and check required columns
    required_columns = ["country", "grid"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError("DataFrame is missing required columns: " + ", ".join(missing_columns))
    
    # Build df['node'] based on available columns.
    if "node_suffix" in df.columns:
        # if node_suffix defined: <country>_<grid>_<node_suffix> 
        df['node'] = df.apply(
            lambda row: f"{row['country']}_{row['grid']}" + 
                        (f"_{row['node_suffix']}" if pd.notnull(row['node_suffix']) and row['node_suffix'] != "" else ""),
            axis=1
        )
    else:
        # If node_suffix column doesn't exist, <country>_<grid>
        df['node'] = df.apply(
            lambda row: f"{row['country']}_{row['grid']}",
            axis=1
        )

    return df


def build_unittype_unit_column(df, df_unittypedata):
    """
    Adds 'unittype' and 'unit' columns to df: 

    unittype is retrieved from df_unittypedata by matching 
    df_unittypedata['Generator_ID'] to df['generator_id'] in a case-insensitive manner.
    
    unit is constructed by following rules    
        * if unit_name_prefix is defined: <country>_<unit_name_prefix>_<unittype>
        * else : <country>_<unittype>
    
    Required columns in df: country, generator_id
    Optional column in df: unit_name_prefix
    """
    # Define and check the required columns
    required_columns = ["country", "generator_id"]
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        raise ValueError("DataFrame is missing required columns: " + ", ".join(missing_columns))

    # Create a case-insensitive mapping from generator_id to unittype in df_unittypedata.
    unit_mapping = df_unittypedata.set_index(df_unittypedata['generator_id'].str.lower())['unittype']
    
    # Map the generator_id in df to the corresponding unittype using case-insensitive matching.
    df['unittype'] = df['generator_id'].str.lower().map(unit_mapping)
    
    # Build df['unit'] based on available columns.
    if "unit_name_prefix" in df.columns:
        df['unit'] = df.apply(
            lambda row: f"{row['country']}"
                        + (f"_{row['unit_name_prefix']}" if pd.notnull(row['unit_name_prefix']) and row['unit_name_prefix'] != "" else "")
                        + f"_{row['unittype']}",
            axis=1
        )
    else:
        df['unit'] = df.apply(
            lambda row: f"{row['country']}_{row['unittype']}",
            axis=1
        )

    return df  


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
                        Exception: Always accept scenario all and year 1
        df_name (str): Name of the DataFrame (used in exception messages).
    """
    # Check that all filter columns exist in the DataFrame
    missing_cols = [col for col in filters if col not in df_input.columns]
    if missing_cols:
        raise Exception(f"Column(s) {missing_cols} are missing from {df_name if df_name else 'DataFrame'}.")

    # Apply the filters iteratively
    df_filtered = df_input.copy()
    for col, val in filters.items():
        # Convert a single value to a list of one item
        if not isinstance(val, list):
            val = [val]

        # Exception: Always accept scenario all and year 1
        if col == 'scenario': val.append('all')
        if col == 'year': val.append(1)

        # If the column is of string type, do a case-insensitive comparison
        if pd.api.types.is_string_dtype(df_filtered[col]):
            # Lower-case both the column values and filter values
            lowered_vals = [v.lower() if isinstance(v, str) else v for v in val]
            df_filtered = df_filtered[df_filtered[col].str.lower().isin(lowered_vals)]
        else:
            df_filtered = df_filtered[df_filtered[col].isin(val)]
            
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
        # Convert a single value to a list of one item
        if not isinstance(val, list):
            val = [val]

        # If the column is of string type, do a case-insensitive comparison
        if pd.api.types.is_string_dtype(df_filtered[col]):
            lowered_vals = [v.lower() if isinstance(v, str) else v for v in val]
            df_filtered = df_filtered[~df_filtered[col].str.lower().isin(lowered_vals)]
        else:
            df_filtered = df_filtered[~df_filtered[col].isin(val)]
    return df_filtered


# ------------------------------------------------------
# Main run function to process input data and create timeseries
# ------------------------------------------------------
def run(input_folder, config_file, input_excel_only=False):
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

    # Extract parameters from the config file under the 'inputdata' section.
    output_folder_prefix = config.get('inputdata', 'output_folder_prefix')
    start_date = config.get('inputdata', 'start_date')
    end_date = config.get('inputdata', 'end_date')
    write_csv_files = config.getboolean('inputdata', 'write_csv_files', fallback=False)

    # Convert string representations of lists into actual Python lists.
    scenarios = ast.literal_eval(config.get('inputdata', 'scenarios'))
    scenario_years = ast.literal_eval(config.get('inputdata', 'scenario_years'))
    scenario_alternatives = ast.literal_eval(config.get('inputdata', 'scenario_alternatives'))
    country_codes = ast.literal_eval(config.get('inputdata', 'country_codes'))
    exclude_grids = ast.literal_eval(config.get('inputdata', 'exclude_grids'))
    exclude_nodes = ast.literal_eval(config.get('inputdata', 'exclude_nodes'))
    demanddata_files = ast.literal_eval(config.get('inputdata', 'demanddata_files'))
    transferdata_files = ast.literal_eval(config.get('inputdata', 'transferdata_files'))
    unittypedata_files = ast.literal_eval(config.get('inputdata', 'unittypedata_files'))
    unitdata_files = ast.literal_eval(config.get('inputdata', 'unitdata_files'))
    storagedata_files = ast.literal_eval(config.get('inputdata', 'storagedata_files'))
    fueldata_files = ast.literal_eval(config.get('inputdata', 'fueldata_files'))
    emissiondata_files = ast.literal_eval(config.get('inputdata', 'emissiondata_files'))
    # timeseries processing specs dictionary
    timeseries_specs = ast.literal_eval(config.get('inputdata', 'timeseries_specs'))

    # Read and process input excels
    data_folder = os.path.join(input_folder, 'data_files')
    df_demanddata   = process_dataset(data_folder, demanddata_files, 'demanddata')
    df_transferdata = process_dataset(data_folder, transferdata_files, 'transferdata')
    df_unittypedata  = process_dataset(data_folder, unittypedata_files, 'unittypedata') 
    df_unitdata  = process_dataset(data_folder, unitdata_files, 'unitdata') 
    df_remove_units  = process_dataset(data_folder, unitdata_files, 'remove', isMandatory=False) 
    df_storagedata  = process_dataset(data_folder, storagedata_files, 'storagedata') 
    df_fueldata  = process_dataset(data_folder, fueldata_files, 'fueldata') 
    df_emissiondata  = process_dataset(data_folder, emissiondata_files, 'emissiondata') 
    
    # exclude grids
    df_demanddata = filter_df_blacklist(df_demanddata, 'demand_files', {'grid': exclude_grids})     
    df_transferdata = filter_df_blacklist(df_transferdata, 'transfer_files', {'grid': exclude_grids})

    # Build node columns
    df_demanddata = build_node_column(df_demanddata)
    df_storagedata = build_node_column(df_storagedata)

    # exclude nodes
    df_demanddata = filter_df_blacklist(df_demanddata, 'demand_files', {'node': exclude_nodes})  
    df_storagedata = filter_df_blacklist(df_storagedata, 'storage_files', {'node': exclude_nodes})      

    # Build unittype and unit columns
    df_unitdata = build_unittype_unit_column(df_unitdata, df_unittypedata)
    df_remove_units = build_unittype_unit_column(df_remove_units, df_unittypedata)


    # Loop over every combination of scenario and scenario year.
    for scenario, scenario_year, scenario_alternative in itertools.product(scenarios, scenario_years, scenario_alternatives or [None]):
        print(f"\n--------------------------------------------------------------------------- ")
        if scenario_alternative:
            log_time(f"---- processing '{scenario}', year {scenario_year}, and alternative '{scenario_alternative}'  ------------ ", log_start)
        else:
            log_time(f"---- processing '{scenario}' year {scenario_year} ---------------- ", log_start) 

        # Combine scenario with a possible alternative
        scen_and_alt = [scenario]
        if scenario_alternative:
            scen_and_alt.extend(scenario_alternatives)

        # Process global input files to get the data for the current scenario and year.
        df_f_unittypedata = filter_df_whitelist(df_unittypedata, 'techdata_files', {'scenario':scen_and_alt, 'year':scenario_year})     
        df_f_fueldata = filter_df_whitelist(df_fueldata, 'fueldata_files', {'scenario':scen_and_alt, 'year':scenario_year})
        df_f_emissiondata = filter_df_whitelist(df_emissiondata, 'emissiondata_files' , {'scenario':scen_and_alt, 'year':scenario_year})
        df_f_transferdata = filter_df_whitelist(df_transferdata, 'transfer_files', {'scenario':scen_and_alt, 'year':scenario_year})

        # Process country specific input files to get the data for the current scenario, year, and country.
        df_f_demanddata = filter_df_whitelist(df_demanddata, 'demand_files', {'scenario':scen_and_alt, 'year':scenario_year, 'country': country_codes})
        df_f_unitdata = filter_df_whitelist(df_unitdata, 'unitcapacity_files', {'scenario':scen_and_alt, 'year':scenario_year, 'country': country_codes})        
        df_f_remove_units = filter_df_whitelist(df_remove_units, 'unitcapacity_files', {'scenario':scen_and_alt, 'year':scenario_year, 'country': country_codes})
        df_f_storagedata = filter_df_whitelist(df_storagedata, 'unitcapacity_files', {'scenario':scen_and_alt, 'year':scenario_year, 'country': country_codes})  

        # remove duplicates. Keep the last value. This implicitly handles overwriting earlier values with the latest.
        df_f_demanddata = keep_last_occurance(df_f_demanddata, ['country', 'grid', 'node'])
        df_f_transferdata = keep_last_occurance(df_f_transferdata, ['from-to', 'grid'])
        df_f_unittypedata = keep_last_occurance(df_f_unittypedata, ['generator_id'])
        df_f_unitdata = keep_last_occurance(df_f_unitdata, ['country', 'generator_id', 'unit_name_prefix'])
        df_f_storagedata = keep_last_occurance(df_f_storagedata, ['country', 'grid', 'node'])
        df_f_fueldata = keep_last_occurance(df_f_fueldata, ['fuel'])
        df_f_emissiondata = keep_last_occurance(df_f_emissiondata, ['emission'])
        
        # Create an output folder specific to the current scenario and year
        if scenario_alternative:
            output_folder = os.path.join(f"{output_folder_prefix}_{scenario}_{str(scenario_year)}_{scenario_alternative}")
        else:
            output_folder = os.path.join(f"{output_folder_prefix}_{scenario}_{str(scenario_year)}")
        if not os.path.exists(output_folder): 
            os.makedirs(output_folder)
        print(f"Writing output files to '.\\{output_folder}'")

        # remove import_timeseries.inc if already exists in the output_folder
        file_path = os.path.join(output_folder, 'import_timeseries.inc')
        if os.path.exists(file_path): os.remove(file_path)

        # compile scen_tags
        scen_tags = {'scenario': scenario, 'year': scenario_year, 'alternative': scenario_alternative} 

        # Create the timeseries using the create_timeseries class.
        secondary_results = create_timeseries(timeseries_specs, input_folder, output_folder, 
                                start_date, end_date, 
                                country_codes, scen_tags, exclude_grids, exclude_nodes, 
                                df_f_demanddata, log_start, write_csv_files
                                ).run()

        # Build input excel using the build_input_excel class.    
        build_input_excel(input_folder, output_folder, 
                          country_codes, scen_tags, exclude_grids, exclude_nodes, 
                          df_f_transferdata, df_f_unittypedata, df_f_unitdata, df_f_remove_units, 
                          df_f_storagedata, df_f_fueldata, df_f_emissiondata, df_f_demanddata,
                          secondary_results
                          ).run()

        # Copy default GAMS files
        print(f"\n---- Copying GAMS files to '{output_folder}' ---------------- ")
        copy_GAMS_files(input_folder, output_folder)

        
    print("\n-----------------------------------")   
    if scenario_alternatives:
        log_time(f"All (scenario, alternative, year) tuples processed.", log_start)
    else:
        log_time(f"All (scenario, year) pairs processed.", log_start)


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
    # Extract input_excel_only; default is "False" if not provided.
    # Convert string values like "True", "true", "1" into a boolean.
    input_excel_only = args.get('input_excel_only', 'False').lower() in ('true', '1')
    
    run(input_folder, filename, input_excel_only)