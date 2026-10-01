import configparser
import ast
import re
from itertools import product
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Iterable, List, Tuple


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


def _safe_eval_int(expr: str) -> int:
    """
    Evaluate a simple arithmetic expression to an int.

    Accepts numeric literals and + - * / // % ** and parentheses.
    Rejects names, calls, attribute access, and anything else.

    Raises:
        ValueError: if the expression is invalid or not integer-valued.
        SyntaxError: if the expression is not parseable as a Python expression.
    """
    tree = ast.parse(expr.strip(), mode='eval')
    allowed = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.USub, ast.UAdd,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed):
            raise ValueError(f"disallowed token in expression: {type(node).__name__}")
        if isinstance(node, ast.Constant) and not isinstance(node.value, (int, float)):
            raise ValueError(f"non-numeric constant: {node.value!r}")
    value = eval(compile(tree, '<config>', 'eval'))  # safe: AST whitelisted above
    if isinstance(value, float):
        if not value.is_integer():
            raise ValueError(f"expression is not integer-valued: {value}")
        value = int(value)
    return value


_TIMESERIES_SPEC_DEFAULTS = {
    'demand_grid': '',
    'custom_column_value': None,
    'gdx_name_suffix': '',
    'rounding_precision': 0,
    'input_sub_folder': '',
    'attached_grid': '',
    'scaling_factor': 1,
    'cutoff_below': None,
    'forecast_quantiles': None,
}

_FORECAST_QUANTILES_DEFAULT = {'f01': 0.5, 'f02': 0.1, 'f03': 0.9}
_FORECAST_WEIGHTS_DEFAULT   = {'f01': 0.6, 'f02': 0.2, 'f03': 0.2}

#: The branch the GAMS templates treat as the central forecast. scheduleInit.gms
#: names it in mf_central and changes.inc reads and writes its data by label, so
#: it is not a free choice of the config.
CENTRAL_FORECAST = 'f01'

#: How a branch beside the central one ends, as Backbone's boundForecastEnds.
FORECAST_BRANCH_ENDS = {'cut': 0, 'bound': 1, 'continue': 2}

#: A branch the config says nothing about: 149 days (3576 h), then cut.
_FORECAST_BRANCH_DEFAULT = {'length_days': 149, 'end': 'cut', 'blend_days': 0}

_TIMESERIES_SPEC_MANDATORY = ('processor_name', 'bb_parameter', 'bb_parameter_dimensions')


def _parse_forecast_branches(raw, forecast_quantiles: dict, bb_horizon_weeks: int) -> Dict[str, Any]:
    """
    Validate forecast_branches and fill in every branch beside the central one.

    The config states how long each branch carries its own data and how it ends:

        {'f01': 'central',
         'f02': {'length_days': 252, 'end': 'continue', 'blend_days': 28},
         'f04': {'length_days': 5, 'end': 'cut'}}

    ``'f01': 'central'`` is there for the reader of the config and is the only
    value f01 takes. ``end`` is one of FORECAST_BRANCH_ENDS, and ``blend_days``
    belongs to a continuing branch.

    Returns:
        {label: {'length_days', 'end', 'blend_days'}} for every label of
        forecast_quantiles except the central one, in that order.

    Raises:
        ValueError: on anything the GAMS templates could not honour.
    """
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError(
            f"forecast_branches must be a dict mapping f-labels to branch settings; "
            f"got {type(raw).__name__}."
        )
    if not forecast_quantiles:
        if raw:
            raise ValueError(
                "forecast_branches must be empty (or omitted) when "
                "forecast_quantiles is empty (deterministic mode)."
            )
        return {}
    if CENTRAL_FORECAST not in forecast_quantiles:
        raise ValueError(
            f"forecast_quantiles must include '{CENTRAL_FORECAST}': it is the central "
            f"forecast in scheduleInit.gms and changes.inc."
        )
    unknown = sorted(set(raw) - set(forecast_quantiles))
    if unknown:
        raise ValueError(
            f"forecast_branches names {unknown}, which forecast_quantiles does not have."
        )

    horizon_days = bb_horizon_weeks * 7
    branches = {}
    for label in forecast_quantiles:
        given = raw.get(label)
        if label == CENTRAL_FORECAST:
            if given not in (None, 'central'):
                raise ValueError(
                    f"forecast_branches['{CENTRAL_FORECAST}'] must be 'central': it always "
                    f"reaches the horizon and takes no settings; got {given!r}."
                )
            continue
        if given == 'central':
            raise ValueError(
                f"forecast_branches['{label}'] cannot be 'central': only "
                f"'{CENTRAL_FORECAST}' can, because scheduleInit.gms and changes.inc "
                f"treat '{CENTRAL_FORECAST}' as the central forecast."
            )
        if given is None:
            given = {}
        if not isinstance(given, dict):
            raise ValueError(
                f"forecast_branches['{label}'] must be a dict of branch settings; "
                f"got {given!r}."
            )
        unknown = sorted(set(given) - set(_FORECAST_BRANCH_DEFAULT))
        if unknown:
            raise ValueError(
                f"forecast_branches['{label}'] has unknown setting(s) {unknown}; "
                f"known: {sorted(_FORECAST_BRANCH_DEFAULT)}."
            )
        # The default length gives way to a shorter horizon; a stated one does not.
        default = {**_FORECAST_BRANCH_DEFAULT,
                   'length_days': min(_FORECAST_BRANCH_DEFAULT['length_days'], horizon_days)}
        branch = {**default, **given}

        if branch['end'] not in FORECAST_BRANCH_ENDS:
            raise ValueError(
                f"forecast_branches['{label}']['end'] must be one of "
                f"{sorted(FORECAST_BRANCH_ENDS)}; got {branch['end']!r}."
            )
        for key in ('length_days', 'blend_days'):
            if isinstance(branch[key], bool) or not isinstance(branch[key], int):
                raise ValueError(
                    f"forecast_branches['{label}']['{key}'] must be a whole number "
                    f"of days; got {branch[key]!r}."
                )
        # The first day of every solve is the realized one, so a branch needs a
        # second day to hold anything of its own.
        if not (2 <= branch['length_days'] <= horizon_days):
            raise ValueError(
                f"forecast_branches['{label}']['length_days'] must be between 2 and "
                f"the horizon, {horizon_days} days (bb_horizon_weeks = "
                f"{bb_horizon_weeks}); got {branch['length_days']}."
            )
        if branch['blend_days'] < 0:
            raise ValueError(
                f"forecast_branches['{label}']['blend_days'] must not be negative; "
                f"got {branch['blend_days']}."
            )
        if branch['blend_days'] and branch['end'] != 'continue':
            raise ValueError(
                f"forecast_branches['{label}'] has blend_days but ends with "
                f"'{branch['end']}': the blend into the central data belongs to "
                f"'continue'."
            )
        branches[label] = branch
    return branches


def _validate_spec_forecast_quantiles(specs: Dict[str, Any], forecast_quantiles: dict) -> None:
    """
    Check every timeseries_specs entry's own forecast_quantiles.

    A spec may give some branches a quantile of its own, which replaces the
    global forecast_quantiles value for that series and no other.

    Raises:
        ValueError: if an override is not a dict, names a branch the global map
                    does not have, or holds a value outside 0-1.
    """
    for name, entry in specs.items():
        override = entry.get('forecast_quantiles')
        if override is None:
            continue
        if not isinstance(override, dict):
            raise ValueError(
                f"timeseries_specs entry '{name}': forecast_quantiles must be a dict "
                f"mapping f-labels to probability quantiles; got {type(override).__name__}."
            )
        unknown = sorted(set(override) - set(forecast_quantiles))
        if unknown:
            raise ValueError(
                f"timeseries_specs entry '{name}': forecast_quantiles names {unknown}, "
                f"which the global forecast_quantiles does not have."
            )
        for label, value in override.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(
                    f"timeseries_specs entry '{name}': forecast_quantiles['{label}'] "
                    f"must be a probability between 0 and 1; got {value!r}."
                )


def spec_forecast_quantiles(config: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, float]:
    """The quantile of every forecast branch for one timeseries spec.

    The global forecast_quantiles, with the spec's own values in place of them
    where it has any. Labels keep the global order.
    """
    return {**config["forecast_quantiles"], **(spec.get("forecast_quantiles") or {})}


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
    # Accepts a plain integer or a simple arithmetic expression (e.g. "365*5").
    bb_ts_length_raw = inputdata.get('bb_timeseries_length', '365')
    try:
        bb_timeseries_length = _safe_eval_int(bb_ts_length_raw)
    except (ValueError, SyntaxError):
        raise ValueError(
            f"bb_timeseries_length must be a positive integer or simple "
            f"arithmetic expression (e.g. 365*5); got '{bb_ts_length_raw}'."
        )
    if not (1 <= bb_timeseries_length <= 365*35+9):
        raise ValueError(
            f"bb_timeseries_length must be between 1 and 365*35+9 = 12784 "
            f"(1982-2016 has 26 regular years, 9 leap years); "
            f"got {bb_timeseries_length}."
        )

    # Parse optional bb_horizon_weeks (default: 52). Whole weeks, because the last
    # interval block in scheduleInit.gms steps a week at a time from week 2.
    bb_horizon_raw = inputdata.get('bb_horizon_weeks', '52')
    try:
        bb_horizon_weeks = _safe_eval_int(bb_horizon_raw)
    except (ValueError, SyntaxError):
        raise ValueError(
            f"bb_horizon_weeks must be a whole number of weeks or a simple "
            f"arithmetic expression (e.g. 52+18); got '{bb_horizon_raw}'."
        )
    if not (3 <= bb_horizon_weeks <= 156):
        raise ValueError(
            f"bb_horizon_weeks must be between 3 and 156 weeks: below 3 the weekly "
            f"steps have no room after the first two weeks, and 156 is three years; "
            f"got {bb_horizon_weeks}."
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

    # Parse optional forecast_quantiles (default: {'f01': 0.5, 'f02': 0.1, 'f03': 0.9})
    forecast_quantiles_raw = inputdata.get('forecast_quantiles')
    if forecast_quantiles_raw is not None:
        forecast_quantiles = ast.literal_eval(forecast_quantiles_raw)
        if not isinstance(forecast_quantiles, dict):
            raise ValueError(
                f"forecast_quantiles must be a dict mapping f-labels to probability quantiles; "
                f"got {type(forecast_quantiles).__name__}."
            )
        if "f00" in forecast_quantiles:
            raise ValueError(
                "forecast_quantiles contains 'f00', which is reserved for realized weather. "
                "Use f01, f02, … for forecast branches."
            )
    else:
        forecast_quantiles = _FORECAST_QUANTILES_DEFAULT

    # Parse optional forecast_weights (default: equal weights, or established defaults for the standard 3-forecast setup)
    forecast_weights_raw = inputdata.get('forecast_weights')
    if not forecast_quantiles:
        # Deterministic mode: realized weather (f00) only, no forecast branches.
        if forecast_weights_raw is not None:
            forecast_weights_parsed = ast.literal_eval(forecast_weights_raw)
            if forecast_weights_parsed:
                raise ValueError(
                    "forecast_weights must be empty (or omitted) when "
                    "forecast_quantiles is empty (deterministic mode)."
                )
        forecast_weights = {}
    elif forecast_weights_raw is not None:
        forecast_weights = ast.literal_eval(forecast_weights_raw)
        if not isinstance(forecast_weights, dict):
            raise ValueError(
                f"forecast_weights must be a dict mapping f-labels to probability weights; "
                f"got {type(forecast_weights).__name__}."
            )
        if set(forecast_weights.keys()) != set(forecast_quantiles.keys()):
            raise ValueError(
                f"forecast_weights keys {sorted(forecast_weights)} must match "
                f"forecast_quantiles keys {sorted(forecast_quantiles)}."
            )
        total = sum(forecast_weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"forecast_weights values must sum to 1.0; got {total}."
            )
    else:
        if set(forecast_quantiles.keys()) == set(_FORECAST_WEIGHTS_DEFAULT.keys()):
            forecast_weights = _FORECAST_WEIGHTS_DEFAULT
        else:
            n = len(forecast_quantiles)
            forecast_weights = {label: 1.0 / n for label in forecast_quantiles}

    # Parse optional forecast_branches (default: every branch 149 days, then cut)
    forecast_branches_raw = inputdata.get('forecast_branches')
    forecast_branches = _parse_forecast_branches(
        None if forecast_branches_raw is None else ast.literal_eval(forecast_branches_raw),
        forecast_quantiles,
        bb_horizon_weeks,
    )

    timeseries_specs = _validate_timeseries_specs(
        ast.literal_eval(inputdata.get('timeseries_specs', '{}'))
    )
    _validate_spec_forecast_quantiles(timeseries_specs, forecast_quantiles)

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

        # Backbone schedule horizon
        'bb_horizon_weeks': bb_horizon_weeks,

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

        # Timeseries forecast quantiles, weights, branch shapes and specs
        'forecast_quantiles': forecast_quantiles,
        'forecast_weights': forecast_weights,
        'forecast_branches': forecast_branches,
        'timeseries_specs': timeseries_specs,
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


def output_folder_name(
    output_folder_prefix: str,
    scenario: str,
    year: Any,
    alternatives: Iterable[str] = (),
) -> str:
    """
    Name the folder one (scenario, year, alternatives) combination builds into.

    The rule is prefix, scenario, year, then every non-empty alternative, joined
    with underscores and with spaces removed from each part -- so 'Observed
    Trends' in config_OT2030.ini becomes input_ObservedTrends_2030.

    This is the single statement of that rule. `build_input_data.py` uses it to
    decide where to write, and `run_model.py` uses it to decide what to pass as
    Backbone's --input_dir, so a run cannot name a folder a build would not have
    written.
    """
    active = [a for a in alternatives if a]
    parts = [output_folder_prefix, scenario, str(year), *active]
    return "_".join(part.replace(" ", "") for part in parts)


def config_output_folder_names(config: Dict[str, Any]) -> List[str]:
    """
    Every folder name a build of this config writes, in the order it writes them.

    The combinations are the Cartesian product of scenarios, scenario_years and
    the four alternative axes, which is the loop `build_input_data.py` runs.
    """
    names: List[str] = []
    for scenario, year, alt1, alt2, alt3, alt4 in product(
        config['scenarios'], config['scenario_years'],
        config['scenario_alternatives'], config['scenario_alternatives2'],
        config['scenario_alternatives3'], config['scenario_alternatives4'],
    ):
        names.append(output_folder_name(
            config['output_folder_prefix'], scenario, year, [alt1, alt2, alt3, alt4]
        ))
    return names
