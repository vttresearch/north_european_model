import configparser
import ast
import re
from itertools import product
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Tuple


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
    'energy_quantiles': None,
}

_ENERGY_QUANTILES_DEFAULT  = {'f01': 0.5, 'f02': 0.1, 'f03': 0.9}
_FORECAST_WEIGHTS_DEFAULT   = {'f01': 0.6, 'f02': 0.2, 'f03': 0.2}

#: The branch the GAMS templates treat as the central forecast. scheduleInit.gms
#: names it in mf_central and changes.inc reads and writes its data by label, so
#: it is not a free choice of the config.
CENTRAL_FORECAST = 'f01'

#: How a branch beside the central one ends, as Backbone's boundForecastEnds.
FORECAST_BRANCH_ENDS = {'cut': 0, 'bound': 1, 'continue': 2}

#: A branch the config says nothing about: 149 days (3576 h), then cut.
_FORECAST_BRANCH_DEFAULT = {'length_days': 149, 'end': 'cut', 'blend_days': 0}

#: Said when a config still uses the key energy_quantiles replaced.
_RENAMED_QUANTILES = (
    "forecast_quantiles is now energy_quantiles, and its values mean energy: 0.5 is "
    "each series' mean energy, 0.1 a one-in-ten low over the branch's length. The "
    "old key took a quantile of every hour, which carries a different energy in "
    "every country. Rename the key; see docs/timeseries.md, 'Forecast branches'."
)

_TIMESERIES_SPEC_MANDATORY = ('processor_name', 'bb_parameter', 'bb_parameter_dimensions')

#: Said when a config still carries one of the four keys `variants` replaced.
_REPLACED_ALTERNATIVES = (
    "scenario_alternatives ... scenario_alternatives4 are replaced by 'variants', a dict "
    "of named builds that each choose options per country, e.g. "
    "variants = {'': {}, 'sens1': {'bio-price': {'FI00': 'high'}}}. Remove the "
    "scenario_alternatives* line(s), even an empty one; see docs/variants.md."
)

#: A variant, option or build name: letters, digits and '-'. No '_', which the
#: source tables read as a node-name separator and drop, and which separates the
#: parts of a folder name.
_VARIANT_NAME = re.compile(r"[A-Za-z0-9-]+")

#: In a data tag, 'all' means every option, so no variant or option may be named
#: it. As a country key in the config it means every country, which is a
#: different place in the dict.
RESERVED_VARIANT_WORD = 'all'

_VARIANTS_SHAPE = (
    "variants = {'build': {'variant': 'option'}} or "
    "{'build': {'variant': {'FI00' or 'FI' or 'all': 'option'}}}"
)


def _variant_name(value, what: str, where: str) -> str:
    """`value` checked as a variant or option name, lower-cased for matching."""
    if not isinstance(value, str):
        raise ValueError(
            f"variants: {where}: the {what} must be a string; got {value!r}. "
            f"Expected {_VARIANTS_SHAPE}."
        )
    if not _VARIANT_NAME.fullmatch(value):
        raise ValueError(
            f"variants: {where}: the {what} name {value!r} may hold only letters, digits "
            f"and '-': no spaces, no ':' or ',', which separate a tag's parts, and no "
            f"'_', which the source tables drop."
        )
    if value.lower() == RESERVED_VARIANT_WORD:
        raise ValueError(
            f"variants: {where}: '{value}' is reserved: in a data tag 'all' means "
            f"every option, so no {what} can be called that."
        )
    return value.lower()


def _expand_country_keys(choice: dict, country_codes: List[str], where: str) -> Dict[str, str]:
    """{'all' or exact code: option} from a per-country map.

    An exact code wins over a two-letter key, and a two-letter key over 'all';
    position in the dict never decides. A two-letter key is shorthand for every
    code it starts, so each of those codes is then judged as an exact code would be.
    """
    by_code = {code.lower(): code for code in country_codes}
    every, by_prefix, exact = None, {}, {}
    seen: Dict[str, str] = {}
    for key, option in choice.items():
        if not isinstance(key, str):
            raise ValueError(
                f"variants: {where}: country key {key!r} must be a string. "
                f"Expected {_VARIANTS_SHAPE}."
            )
        if isinstance(option, dict) or not isinstance(option, str):
            raise ValueError(
                f"variants: {where}: the value for {key!r} must be an option name; "
                f"got {option!r}. Expected {_VARIANTS_SHAPE}."
            )
        option = _variant_name(option, "option", f"{where}, {key!r}")
        folded = key.lower()
        if folded in seen:
            raise ValueError(
                f"variants: {where}: the country key {key!r} is given twice "
                f"(also as {seen[folded]!r})."
            )
        seen[folded] = key
        if folded == 'all':
            every = option
        elif folded in by_code:
            exact[by_code[folded]] = option
        elif len(key) == 2 and key.isalpha():
            codes = [code for code in country_codes if code.lower().startswith(folded)]
            if not codes:
                raise ValueError(
                    f"variants: {where}: the key {key!r} matches no code in country_codes."
                )
            by_prefix[folded] = (codes, option)
        else:
            raise ValueError(
                f"variants: {where}: the key {key!r} is neither 'all', a code in "
                f"country_codes, nor a two-letter prefix of one."
            )

    expanded: Dict[str, str] = {}
    if every is not None:
        expanded['all'] = every
    for codes, option in by_prefix.values():
        for code in codes:
            expanded[code] = option
    expanded.update(exact)
    return expanded


def parse_variants(raw: Any, country_codes: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    """
    Validate the config's `variants` dict and normalise it.

    Returns {build name as written: {variant, lower case: {'all' or an exact
    country code: option, lower case}}}. A plain string choice becomes
    {'all': option}, and a two-letter key becomes one entry per code it starts.

    Build names become folder suffixes: letters, digits and '-', unique
    case-insensitively, and '' for the build with no suffix.

    Raises:
        ValueError: on any mistake; config parsing runs before the logger exists.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"variants must be a dict of named builds; got {type(raw).__name__}. "
            f"Expected {_VARIANTS_SHAPE}."
        )
    if not raw:
        raise ValueError(
            "variants names no build, so nothing would be built. Write "
            "variants = {'': {}} for one build with every default, or leave the key out."
        )

    builds: Dict[str, Dict[str, Dict[str, str]]] = {}
    seen_builds: Dict[str, str] = {}
    for build, choices in raw.items():
        if not isinstance(build, str):
            raise ValueError(f"variants: build name {build!r} must be a string.")
        if build and not _VARIANT_NAME.fullmatch(build):
            raise ValueError(
                f"variants: the build name {build!r} cannot be a folder segment: use "
                f"letters, digits and '-' only. '_' separates the parts of a folder name."
            )
        if build.lower() in seen_builds:
            raise ValueError(
                f"variants: the build names {seen_builds[build.lower()]!r} and {build!r} "
                f"differ only in case, and would share one folder on Windows."
            )
        seen_builds[build.lower()] = build
        if not isinstance(choices, dict):
            raise ValueError(
                f"variants: build {build!r} must map variants to choices; got "
                f"{choices!r}. Expected {_VARIANTS_SHAPE}."
            )

        normalised: Dict[str, Dict[str, str]] = {}
        for variant, choice in choices.items():
            where = f"build {build!r}"
            name = _variant_name(variant, "variant", where)
            if name in normalised:
                raise ValueError(f"variants: {where}: the variant {variant!r} is given twice.")
            where = f"build {build!r}, variant {variant!r}"
            if isinstance(choice, str):
                normalised[name] = {'all': _variant_name(choice, "option", where)}
            elif isinstance(choice, dict):
                normalised[name] = _expand_country_keys(choice, country_codes, where)
            else:
                raise ValueError(
                    f"variants: {where}: the choice must be an option name or a "
                    f"per-country dict; got {choice!r}. Expected {_VARIANTS_SHAPE}."
                )
        builds[build] = normalised
    return builds


def selected_builds(config: Dict[str, Any], selection: str | None = None) -> List[str]:
    """
    The build names to build or run: every build in config order, or the one named.

    Matched case-insensitively, which is unambiguous because build names are
    unique that way.

    Raises:
        ValueError: when `selection` names no build of this config.
    """
    builds = list(config.get('variants') or {'': {}})
    if selection is None:
        return builds
    for build in builds:
        if build.lower() == selection.lower():
            return [build]
    known = ", ".join("'' (no suffix)" if not b else repr(b) for b in builds)
    raise ValueError(
        f"--variant {selection!r} is not a build of this config. Its builds are: {known}."
    )


def _parse_forecast_branches(raw, energy_quantiles: dict, bb_horizon_weeks: int) -> Dict[str, Any]:
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
        energy_quantiles except the central one, in that order.

    Raises:
        ValueError: on anything the GAMS templates could not honour.
    """
    raw = {} if raw is None else raw
    if not isinstance(raw, dict):
        raise ValueError(
            f"forecast_branches must be a dict mapping f-labels to branch settings; "
            f"got {type(raw).__name__}."
        )
    if not energy_quantiles:
        if raw:
            raise ValueError(
                "forecast_branches must be empty (or omitted) when "
                "energy_quantiles is empty (deterministic mode)."
            )
        return {}
    if CENTRAL_FORECAST not in energy_quantiles:
        raise ValueError(
            f"energy_quantiles must include '{CENTRAL_FORECAST}': it is the central "
            f"forecast in scheduleInit.gms and changes.inc."
        )
    unknown = sorted(set(raw) - set(energy_quantiles))
    if unknown:
        raise ValueError(
            f"forecast_branches names {unknown}, which energy_quantiles does not have."
        )

    horizon_days = bb_horizon_weeks * 7
    branches = {}
    for label in energy_quantiles:
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


def _validate_spec_energy_quantiles(specs: Dict[str, Any], energy_quantiles: dict) -> None:
    """
    Check every timeseries_specs entry's own energy_quantiles.

    A spec may give some branches an energy quantile of its own, which replaces
    the global energy_quantiles value for that series and no other.

    Raises:
        ValueError: if an entry still uses forecast_quantiles, or an override is
                    not a dict, names a branch the global map does not have, or
                    holds a value outside 0-1.
    """
    for name, entry in specs.items():
        if 'forecast_quantiles' in entry:
            raise ValueError(f"timeseries_specs entry '{name}': {_RENAMED_QUANTILES}")
        override = entry.get('energy_quantiles')
        if override is None:
            continue
        if not isinstance(override, dict):
            raise ValueError(
                f"timeseries_specs entry '{name}': energy_quantiles must be a dict "
                f"mapping f-labels to energy quantiles; got {type(override).__name__}."
            )
        unknown = sorted(set(override) - set(energy_quantiles))
        if unknown:
            raise ValueError(
                f"timeseries_specs entry '{name}': energy_quantiles names {unknown}, "
                f"which the global energy_quantiles does not have."
            )
        for label, value in override.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(
                    f"timeseries_specs entry '{name}': energy_quantiles['{label}'] "
                    f"must be between 0 and 1; got {value!r}."
                )


def spec_energy_quantiles(config: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, float]:
    """The energy quantile of every forecast branch for one timeseries spec.

    The global energy_quantiles, with the spec's own values in place of them
    where it has any. Labels keep the global order.
    """
    return {**config["energy_quantiles"], **(spec.get("energy_quantiles") or {})}


def branch_energy_days(config: Dict[str, Any]) -> Dict[str, int]:
    """Per branch beside the central one, the days its energy is measured over.

    A branch's energy quantile describes the energy it carries over its own
    length, so that is its forecast_branches length_days. The central branch is
    not named: it reaches the horizon and is measured over the whole window.
    """
    return {label: branch["length_days"]
            for label, branch in config.get("forecast_branches", {}).items()}


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

    # Parse optional energy_quantiles (default: {'f01': 0.5, 'f02': 0.1, 'f03': 0.9})
    if inputdata.get('forecast_quantiles') is not None:
        raise ValueError(_RENAMED_QUANTILES)
    energy_quantiles_raw = inputdata.get('energy_quantiles')
    if energy_quantiles_raw is not None:
        energy_quantiles = ast.literal_eval(energy_quantiles_raw)
        if not isinstance(energy_quantiles, dict):
            raise ValueError(
                f"energy_quantiles must be a dict mapping f-labels to energy quantiles; "
                f"got {type(energy_quantiles).__name__}."
            )
        if "f00" in energy_quantiles:
            raise ValueError(
                "energy_quantiles contains 'f00', which is reserved for realized weather. "
                "Use f01, f02, ... for forecast branches."
            )
        for label, value in energy_quantiles.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(
                    f"energy_quantiles['{label}'] must be between 0 and 1; got {value!r}."
                )
    else:
        energy_quantiles = _ENERGY_QUANTILES_DEFAULT

    # Parse optional forecast_weights (default: equal weights, or established defaults for the standard 3-forecast setup)
    forecast_weights_raw = inputdata.get('forecast_weights')
    if not energy_quantiles:
        # Deterministic mode: realized weather (f00) only, no forecast branches.
        if forecast_weights_raw is not None:
            forecast_weights_parsed = ast.literal_eval(forecast_weights_raw)
            if forecast_weights_parsed:
                raise ValueError(
                    "forecast_weights must be empty (or omitted) when "
                    "energy_quantiles is empty (deterministic mode)."
                )
        forecast_weights = {}
    elif forecast_weights_raw is not None:
        forecast_weights = ast.literal_eval(forecast_weights_raw)
        if not isinstance(forecast_weights, dict):
            raise ValueError(
                f"forecast_weights must be a dict mapping f-labels to probability weights; "
                f"got {type(forecast_weights).__name__}."
            )
        if set(forecast_weights.keys()) != set(energy_quantiles.keys()):
            raise ValueError(
                f"forecast_weights keys {sorted(forecast_weights)} must match "
                f"energy_quantiles keys {sorted(energy_quantiles)}."
            )
        total = sum(forecast_weights.values())
        if abs(total - 1.0) > 1e-9:
            raise ValueError(
                f"forecast_weights values must sum to 1.0; got {total}."
            )
    else:
        if set(energy_quantiles.keys()) == set(_FORECAST_WEIGHTS_DEFAULT.keys()):
            forecast_weights = _FORECAST_WEIGHTS_DEFAULT
        else:
            n = len(energy_quantiles)
            forecast_weights = {label: 1.0 / n for label in energy_quantiles}

    # Parse optional forecast_branches (default: every branch 149 days, then cut)
    forecast_branches_raw = inputdata.get('forecast_branches')
    forecast_branches = _parse_forecast_branches(
        None if forecast_branches_raw is None else ast.literal_eval(forecast_branches_raw),
        energy_quantiles,
        bb_horizon_weeks,
    )

    timeseries_specs = _validate_timeseries_specs(
        ast.literal_eval(inputdata.get('timeseries_specs', '{}'))
    )
    _validate_spec_energy_quantiles(timeseries_specs, energy_quantiles)

    # Named builds. Refused even when empty: every shipped config wrote
    # scenario_alternatives = [], and a stale config is told where it went.
    replaced = [k for k in ('scenario_alternatives', 'scenario_alternatives2',
                            'scenario_alternatives3', 'scenario_alternatives4')
                if inputdata.get(k) is not None]
    if replaced:
        raise ValueError(f"Config key(s) {replaced}: {_REPLACED_ALTERNATIVES}")
    variants_raw = inputdata.get('variants')
    variants = (
        {'': {}} if variants_raw is None
        else parse_variants(ast.literal_eval(variants_raw),
                            ast.literal_eval(inputdata.get('country_codes')))
    )

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
        'variants': variants,

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

        # Forecast branches: energy quantiles, weights, branch shapes; and specs
        'energy_quantiles': energy_quantiles,
        'forecast_weights': forecast_weights,
        'forecast_branches': forecast_branches,
        'timeseries_specs': timeseries_specs,
    }

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


def build_name(scenario: str, year: Any, variant: str = "") -> str:
    """
    One build's name: scenario, year and the build name, without the folder prefix.

    Joined with underscores and with spaces removed from each part, so 'Observed
    Trends' becomes ObservedTrends_2030, and the build 'sens1' adds _sens1. The
    build '' adds nothing.
    """
    parts = [scenario, str(year)] + ([variant] if variant else [])
    return "_".join(part.replace(" ", "") for part in parts)


def output_folder_name(
    output_folder_prefix: str,
    scenario: str,
    year: Any,
    variant: str = "",
) -> str:
    """
    Name the folder one (scenario, year, build) combination builds into.

    The rule is prefix, then build_name: input_ObservedTrends_2030 for the build
    '' of config_OT2030.ini, input_ObservedTrends_2030_sens1 for the build 'sens1'.

    This is the single statement of that rule. `build_input_data.py` uses it to
    decide where to write, and `run_model.py` uses it to say which config and build
    wrote a folder, so the two cannot disagree about a name.
    """
    return "_".join([output_folder_prefix.replace(" ", ""),
                     build_name(scenario, year, variant)])


def scenario_tags(scenario: str, year: Any, variant: str = "") -> List[str]:
    """
    The add_scen_tags row: scenario, year, build name and the build's own name.

    Spine Toolbox workflows read this row by cell position, so the order is an
    interface: (0, 0) and (0, 1) stay scenario and year, (0, 2) is the build
    name ('' for the unsuffixed build), and (0, 3) names this build uniquely.
    """
    return [scenario, str(year), variant, build_name(scenario, year, variant)]


def config_output_folder_names(config: Dict[str, Any]) -> List[str]:
    """
    Every folder name a build of this config writes, in the order it writes them.

    Scenarios, then years, then the builds of `variants` in the order written,
    which is the loop `build_input_data.py` runs.
    """
    return [
        output_folder_name(config['output_folder_prefix'], scenario, year, build)
        for scenario, year, build in product(
            config['scenarios'], config['scenario_years'], selected_builds(config)
        )
    ]
