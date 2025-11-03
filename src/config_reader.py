import configparser
import ast
from pathlib import Path
from typing import Dict, Any

def load_config(config_file: Path) -> Dict[str, Any]:
    """
    Load and validate a configuration file in .ini format.

    The function expects the .ini file to have an [inputdata] section,
    and requires the following fields within that section:
    - scenarios
    - scenario_years
    - start_date
    - end_date
    - country_codes

    Other keys are optional and have empty default values.

    Args:
        config_file (Path): Path to the .ini configuration file.

    Returns:
        Dict[str, Any]: Loaded and validated configuration dictionary.
    
    Raises:
        ValueError: 
            - If the file type is unsupported, 
            - the [inputdata] section is missing,
            - any of the mandatory fields is missing.
    """
    # Parse the config file
    parser = configparser.ConfigParser()
    read_files = parser.read(config_file) # Not wrapping this into try:, because configparser is very chatty already
    if not read_files:
        raise ValueError(f"Failed to read configuration file: {config_file}")

    # Import input data
    if 'inputdata' not in parser:
        raise ValueError("Missing required [inputdata] section in config file.")
    inputdata = parser['inputdata']

    # Check for missing mandatory fields
    mandatory_fields = ['scenarios', 'scenario_years', 'start_date', 'end_date', 'country_codes']
    missing_fields = [field for field in mandatory_fields if field not in inputdata]
    if missing_fields:
        raise ValueError(f"Missing mandatory fields in [inputdata]: {', '.join(missing_fields)}")

    # Build the config dictionary manually 
    # Insert correctly shaped default values in case of missing keys
    config: Dict[str, Any] = {
        # General settings
        'output_folder_prefix': inputdata.get('output_folder_prefix', 'output'),
        'write_csv_files': inputdata.getboolean('write_csv_files', False),
        'force_full_rerun': inputdata.getboolean('force_full_rerun', False),
        'print_all_elapsed_times': inputdata.getboolean('print_all_elapsed_times', False),
        'disable_all_ts_processors': inputdata.getboolean('disable_all_ts_processors', False),
        'disable_other_demand_ts': inputdata.getboolean('disable_other_demand_ts', False),

        # Scenario settings
        'scenarios': ast.literal_eval(inputdata.get('scenarios')),
        'scenario_years': ast.literal_eval(inputdata.get('scenario_years')),
        'scenario_alternatives': ast.literal_eval(inputdata.get('scenario_alternatives', '[""]')),

        # Climate years
        'start_date': inputdata.get('start_date'),
        'end_date': inputdata.get('end_date'),

        # Topology        
        'country_codes': ast.literal_eval(inputdata.get('country_codes')),
        'exclude_grids': ast.literal_eval(inputdata.get('exclude_grids', '[]')),
        'exclude_nodes': ast.literal_eval(inputdata.get('exclude_nodes', '[]')),

        # Data files
        'unittypedata_files': ast.literal_eval(inputdata.get('unittypedata_files', '[]')),
        'fueldata_files': ast.literal_eval(inputdata.get('fueldata_files', '[]')),
        'emissiondata_files': ast.literal_eval(inputdata.get('emissiondata_files', '[]')),
        'demanddata_files': ast.literal_eval(inputdata.get('demanddata_files', '[]')),
        'transferdata_files': ast.literal_eval(inputdata.get('transferdata_files', '[]')),
        'unitdata_files': ast.literal_eval(inputdata.get('unitdata_files', '[]')),
        'storagedata_files': ast.literal_eval(inputdata.get('storagedata_files', '[]')),
        'userconstraint_files': ast.literal_eval(inputdata.get('userconstraint_files', '[]')),

        # Timeseries specs
        'timeseries_specs': ast.literal_eval(inputdata.get('timeseries_specs', '{}'))
    }

    # If user has given scenario_alternatives = [], replace the value with [""]
    if not config['scenario_alternatives']: config['scenario_alternatives'] = [""] 

    return config
