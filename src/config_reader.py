import configparser
import ast
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Tuple


def _parse_climate_data(value: str) -> Tuple[int, int]:
    """
    Parse the climate_data config value into (start_year, end_year).

    Accepted formats:
        "YYYY"        -- single climate year (start_year == end_year)
        "YYYY-YYYY"   -- inclusive range of climate years

    Returns:
        (start_year, end_year) as integers.

    Raises:
        ValueError: if the format is invalid or the range is inverted.
    """
    value = value.strip()
    if not re.fullmatch(r'\d{4}(-\d{4})?', value):
        raise ValueError(
            f"Invalid climate_data format '{value}'. "
            "Expected 'YYYY' or 'YYYY-YYYY' (e.g. '2014' or '2014-2016')."
        )
    if '-' in value:
        start_str, end_str = value.split('-')
        start_year, end_year = int(start_str), int(end_str)
    else:
        start_year = end_year = int(value)

    if not (1982 <= start_year <= 2016 and 1982 <= end_year <= 2016):
        raise ValueError(
            f"Climate years must, for now, be between 1982 and 2016; got '{value}'."
        )
    if start_year > end_year:
        raise ValueError(
            f"climate_data start year ({start_year}) must not be later than "
            f"end year ({end_year})."
        )
    return start_year, end_year


def _parse_bb_timeseries_start(value: str) -> str:
    """
    Validate and return a bb_timeseries_start value ('mm-dd').

    Raises:
        ValueError: if the format or month/day values are out of range.
    """
    value = value.strip()
    if not re.fullmatch(r'\d{2}-\d{2}', value):
        raise ValueError(
            f"Invalid bb_timeseries_start format '{value}'. "
            "Expected 'mm-dd' (e.g. '01-01' or '07-01')."
        )
    mm, dd = int(value[:2]), int(value[3:])
    if not (1 <= mm <= 12):
        raise ValueError(
            f"bb_timeseries_start month {mm} is out of range (01-12)."
        )
    if not (1 <= dd <= 31):
        raise ValueError(
            f"bb_timeseries_start day {dd} is out of range (01-31)."
        )
    return value


_TIMESERIES_SPEC_DEFAULTS = {
    'demand_grid': '',
    'custom_column_value': None,
    'gdx_name_suffix': '',
    'rounding_precision': 0,
    'secondary_output_name': None,
    'input_sub_folder': '',
    'attached_grid': '',
    'is_input_data_dependent': True,
    'scaling_factor': 1,
}

_FORECAST_QUANTILES_DEFAULT = {0.5: 'f01', 0.1: 'f02', 0.9: 'f03'}

_TIMESERIES_SPEC_MANDATORY = ('processor_name', 'bb_parameter', 'bb_parameter_dimensions')


def _validate_timeseries_specs(specs: Any) -> Dict[str, Any]:
    """
    Validate timeseries_specs and inject defaults for optional fields.

    Each entry must be a dict with the mandatory keys:
        processor_name, bb_parameter, bb_parameter_dimensions

    Missing optional keys are filled from _TIMESERIES_SPEC_DEFAULTS.

    Returns:
        The validated and completed specs dict.

    Raises:
        ValueError: if specs is not a dict, any entry is not a dict,
                    or a mandatory field is missing.
    """
    if not isinstance(specs, dict):
        raise ValueError(
            f"timeseries_specs must be a dictionary; got {type(specs).__name__}."
        )
    for name, entry in specs.items():
        if not isinstance(entry, dict):
            raise ValueError(
                f"timeseries_specs entry '{name}' must be a dictionary; "
                f"got {type(entry).__name__}."
            )
        missing = [k for k in _TIMESERIES_SPEC_MANDATORY if k not in entry]
        if missing:
            raise ValueError(
                f"timeseries_specs entry '{name}' is missing mandatory "
                f"field(s): {', '.join(missing)}."
            )
        for key, default in _TIMESERIES_SPEC_DEFAULTS.items():
            entry.setdefault(key, default)
    return specs


def load_config(config_file: Path) -> Dict[str, Any]:
    """
    Load and validate a configuration file in .ini format.

    The function expects the .ini file to have an [inputdata] section,
    and requires the following fields within that section:
    - scenarios
    - scenario_years
    - climate_data
    - country_codes

    Other keys are optional and have default values.

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
    mandatory_fields = ['scenarios', 'scenario_years', 'climate_data', 'country_codes']
    missing_fields = [field for field in mandatory_fields if field not in inputdata]
    if missing_fields:
        raise ValueError(f"Missing mandatory fields in [inputdata]: {', '.join(missing_fields)}")

    # Parse climate_data
    start_year, end_year = _parse_climate_data(inputdata.get('climate_data'))

    # Parse optional bb_timeseries_start (default: '01-01')
    bb_ts_start_raw = inputdata.get('bb_timeseries_start', '01-01')
    bb_timeseries_start = _parse_bb_timeseries_start(bb_ts_start_raw)

    # Parse optional bb_timeseries_length (default: 365)
    bb_ts_length_raw = inputdata.get('bb_timeseries_length', '365')
    try:
        bb_timeseries_length = int(bb_ts_length_raw)
    except ValueError:
        raise ValueError(
            f"bb_timeseries_length must be a positive integer; got '{bb_ts_length_raw}'."
        )
    if not (1 <= bb_timeseries_length <= 365*35+9):
        raise ValueError(
            f"bb_timeseries_length must be between 1 and 365*35+9 = 12784"
             "(1982-2016 has 26 regular years, 9 leap years); "
             "got {bb_timeseries_length}."
        )

    # Validate that at least one climate year fits within the data range
    mm, dd = int(bb_timeseries_start[:2]), int(bb_timeseries_start[3:])
    data_end = datetime(end_year, 12, 31, 23)
    valid_years = []
    for yr in range(start_year, end_year + 1):
        try:
            window_last = datetime(yr, mm, dd) + timedelta(hours=bb_timeseries_length * 24 - 1)
            if window_last <= data_end:
                valid_years.append(yr)
        except ValueError:
            pass  # e.g. Feb 29 on a non-leap year -- skip silently
    if not valid_years:
        raise ValueError(
            f"No climate year in {start_year}-{end_year} has a complete {bb_timeseries_length}-day window "
            f"starting on {bb_timeseries_start} within the given data range. "
            f"Reduce bb_timeseries_length or extend the climate_data range."
        )

    # Parse optional forecast_quantiles (default: {0.5: 'f01', 0.1: 'f02', 0.9: 'f03'})
    forecast_quantiles_raw = inputdata.get('forecast_quantiles')
    if forecast_quantiles_raw is not None:
        forecast_quantiles = ast.literal_eval(forecast_quantiles_raw)
        if not isinstance(forecast_quantiles, dict):
            raise ValueError(
                f"forecast_quantiles must be a dict mapping float quantiles to f-labels; "
                f"got {type(forecast_quantiles).__name__}."
            )
        if "f00" in forecast_quantiles.values():
            raise ValueError(
                "forecast_quantiles contains 'f00', which is reserved for realized weather. "
                "Use f01, f02, … for forecast branches."
            )
    else:
        forecast_quantiles = _FORECAST_QUANTILES_DEFAULT

    # Build the config dictionary manually
    # Insert correctly shaped default values in case of missing keys
    config: Dict[str, Any] = {
        # General settings
        'output_folder_prefix': inputdata.get('output_folder_prefix', 'output'),
        'force_full_rerun': inputdata.getboolean('force_full_rerun', False),
        'print_all_elapsed_times': inputdata.getboolean('print_all_elapsed_times', False),

        # Scenario settings
        'scenarios': ast.literal_eval(inputdata.get('scenarios')),
        'scenario_years': ast.literal_eval(inputdata.get('scenario_years')),
        'scenario_alternatives': ast.literal_eval(inputdata.get('scenario_alternatives', '[""]')),
        'scenario_alternatives2': ast.literal_eval(inputdata.get('scenario_alternatives2', '[""]')),
        'scenario_alternatives3': ast.literal_eval(inputdata.get('scenario_alternatives3', '[""]')),
        'scenario_alternatives4': ast.literal_eval(inputdata.get('scenario_alternatives4', '[""]')),

        # Climate years
        'climate_data': inputdata.get('climate_data').strip(),
        'start_year': start_year,
        'end_year': end_year,

        # Timeseries window
        'bb_timeseries_start': bb_timeseries_start,
        'bb_timeseries_length': bb_timeseries_length,

        # Topology
        'country_codes': ast.literal_eval(inputdata.get('country_codes')),
        'exclude_grids': ast.literal_eval(inputdata.get('exclude_grids', '[]')),
        'exclude_nodes': ast.literal_eval(inputdata.get('exclude_nodes', '[]')),

        # Data files
        'unittypedata_files': ast.literal_eval(inputdata.get('unittypedata_files', '[]')),
        'nodedata_files': ast.literal_eval(inputdata.get('nodedata_files', '[]')),
        'emissiondata_files': ast.literal_eval(inputdata.get('emissiondata_files', '[]')),
        'demanddata_files': ast.literal_eval(inputdata.get('demanddata_files', '[]')),
        'transferdata_files': ast.literal_eval(inputdata.get('transferdata_files', '[]')),
        'unitdata_files': ast.literal_eval(inputdata.get('unitdata_files', '[]')),
        'userconstraintdata_files': ast.literal_eval(inputdata.get('userconstraintdata_files', '[]')),

        # Timeseries forecast quantiles and specs
        'forecast_quantiles': forecast_quantiles,
        'timeseries_specs': _validate_timeseries_specs(
            ast.literal_eval(inputdata.get('timeseries_specs', '{}'))
        )
    }

    # If user has given scenario_alternatives* = [], replace the value with [""]
    for key in ('scenario_alternatives', 'scenario_alternatives2', 'scenario_alternatives3', 'scenario_alternatives4'):
        if not config[key]:
            config[key] = [""]

    # Deprecation check: fueldata_files and storagedata_files were merged into nodedata_files.
    # Raise early so the user sees a clear message before any pipeline work begins.
    deprecated = [k for k in ('fueldata_files', 'storagedata_files') if inputdata.get(k) is not None]
    if deprecated:
        raise ValueError(
            f"Config key(s) {deprecated} are no longer supported. "
            "fueldata and storagedata have been merged into 'nodedata_files'. "
            "Rename the Excel sheets from 'fueldata'/'storagedata' to 'nodedata' "
            "and replace the two config entries with a single 'nodedata_files' list."
        )

    return config
