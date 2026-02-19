# src/utils.py

import sys
import pandas as pd
import shutil
from pathlib import Path
import argparse
import time
from pandas.api.types import is_numeric_dtype, is_bool_dtype


# TODO: These module-level globals are a temporary half-measure. The intended next step
# is to move them into IterationLogger as instance attributes and make log_status() and elapsed_time() 
# methods of IterationLogger. That will require updating all sub-pipeline call sites
# (data_loader, GDX_exchange, processors, pipelines) to accept and use a logger instance
# instead of a plain list, eliminating the remaining global state entirely. This is a significant 
# refactoring as log_status is called more than 100 times throughout the code. After that the 
# IterationLogger can move out from the utils.py home.
_PRINT_ALL_ELAPSED_TIMES: bool = False
_START_TIME: float = time.time()
_WARNING_LOG: list[str] = []

class IterationLogger:
    """Encapsulates per-iteration log state.

    Creating an instance resets all per-iteration globals (_WARNING_LOG, _START_TIME,
    _PRINT_ALL_ELAPSED_TIMES) that log_status() uses internally.
    """

    def __init__(self, *, print_all_elapsed_times: bool) -> None:
        global _WARNING_LOG, _START_TIME, _PRINT_ALL_ELAPSED_TIMES
        _PRINT_ALL_ELAPSED_TIMES = bool(print_all_elapsed_times)
        _WARNING_LOG = []
        _START_TIME = time.time()
        self.messages: list[str] = []

    def log(self, message: str, level: str = "none", **kwargs) -> None:
        """Delegate to log_status, writing into this iteration's message list."""
        log_status(message, self.messages, level=level, **kwargs)

    def extend(self, more_messages: list[str]) -> None:
        """Append messages returned by a sub-pipeline into this iteration's log."""
        self.messages.extend(more_messages)

    @property
    def warnings(self) -> list[str]:
        """Return a copy of all warn/error/skip messages accumulated this iteration."""
        return list(_WARNING_LOG)


def elapsed_time(start_time):
    elapsed_seconds = time.time() - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = float(elapsed_seconds % 60)
    seconds = round(seconds,2)
    return minutes, seconds


def log_status(message: str, 
               log: list[str], 
               level: str = "none", 
               section_start_length: int = 0, 
               add_empty_line_before: bool = False, 
               add_empty_line_after: bool = False, 
               print_to_screen: bool = True):
    """
    Logs a formatted status message with emoji prefix and optional console print.

    If PRINT_ALL_ELAPSED_TIMES is True, prints elapsed time since _START_TIME.

    Parameters:
        message (str): The message to log.
        log (list[str]): List to accumulate messages.
        level (str): Status level (info, warn, run, done, skip).
        section_start_length (int): If larger than zero, format message as a section header with the min length of section_start_length
        print_now (bool): Whether to print the message immediately.
        add_empty_line_before (bool): If empty line is printed before the line
        add_empty_line_efter (bool): If empty line is printed after the line
    """
    prefix = {
        "info": "✓",
        "warn": "⚠️",
        "error": "❌",
        "run": "⚡",
        "done": "🎯",
        "skip": "⏩",
        "none": " "
    }.get(level, " ")

    # build elapsed prefix if enabled
    elapsed_prefix = ""
    if _PRINT_ALL_ELAPSED_TIMES:
        m, s = elapsed_time(_START_TIME)
        elapsed_prefix = f"{m} min {s} sec: "

    if add_empty_line_before:
        start = "\n"
    else:
        start = ""

    if add_empty_line_after:
        end = "\n"
    else:
        end = ""

    if section_start_length > 0:
        base = f"{start}---  {prefix} {elapsed_prefix} {message}{end} "
        padding_length = max(section_start_length, len(base))
        formatted = base + "-" * (padding_length - len(base))
    else:
        formatted = f"{start}{prefix} {elapsed_prefix}{message}{end}"

    log.append(formatted)

    if level in ("warn", "error", "skip"):
        _WARNING_LOG.append(formatted)

    if print_to_screen:
        print(formatted)


def parse_sys_args():
    # Instructions in case of mispelled input cmd
    USAGE_MSG = (
        "Usage: python build_input_data.py <input_folder> <config_file>,\n"
        "       e.g. python build_input_data.py src_files config_test.ini"
    )
        
    # detect legacy key=val syntax
    if any("=" in arg for arg in sys.argv[1:]):
        print(USAGE_MSG)
        sys.exit(1)
    else:
        # strict positional: both required
        parser = argparse.ArgumentParser(
            usage=USAGE_MSG,
            description="NorthEuropeanBackbone Input Builder"
        )
        parser.add_argument(
            "input_folder",
            type=str,
            help="Input folder (e.g. src_files)"
        )
        parser.add_argument(
            "config_file",
            type=str,
            help="Config file name (relative to input_folder)"
        )
        # argparse will print our USAGE_MSG if args are missing
        args = parser.parse_args()
        input_folder = Path(args.input_folder)
        config_file  = Path(input_folder, args.config_file)

        return (input_folder, config_file)
    

def check_dependencies():
    """
    Verifies required dependencies.
        - Python >= 3.12
        - pandas >= 2.2
        - pyarrow
        - tqdm
        - gams.transfer importable
        - gams executable accessible in PATH

    Raises RuntimeError if any requirement is not met.
    """
    import importlib

    errors = []

    # Check Python version
    py_major, py_minor = sys.version_info[:2]
    if (py_major, py_minor) < (3, 12):
        errors.append(f"Python {py_major}.{py_minor} detected (requires ≥3.12), see readme.md how to install/update the environment.")

    # Check pandas version
    try:
        import pandas as pd
        pd_major, pd_minor = map(int, pd.__version__.split('.')[:2])
        if (pd_major, pd_minor) < (2, 2):
            errors.append(f"pandas {pd_major}.{pd_minor} detected (requires ≥2.2)")
    except ImportError:
        errors.append("pandas not installed, see readme.md how to install/update the environment.")  

    # Check pyarrow availability
    try:
        importlib.import_module("pyarrow")
    except ImportError:
        errors.append("pyarrow not installed, see readme.md how to install/update the environment.")

    # Check tqdm availability
    try:
        importlib.import_module("tqdm")
    except ImportError:
        errors.append("tqdm not installed, see readme.md how to install/update the environment.")

    # Check gams.transfer importability
    try:
        importlib.import_module("gams.transfer")
    except ImportError:
        errors.append("gams.transfer not importable (GAMS Python API missing), see readme.md how to install/update the environment.")

    # Check gams executable availability in PATH
    gams_exec = shutil.which("gams") or shutil.which("gams.exe")
    if gams_exec is None:
        errors.append("GAMS not found in PATH")

    # Final decision
    if errors:
        msg = "Dependency check failed:\n  - " + "\n  - ".join(errors)
        raise RuntimeError(msg)


def standardize_df_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize DataFrame column dtypes to a consistent set:
    - Replace 'NaN' strings with pd.NA
    - Attempt to convert object columns containing numeric strings to numeric
    - Empty columns (all NA) → object
    - Numeric columns → Float64
    - Everything else → object

    NA values are preserved

    Examples
    --------
    >>> df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y'], 'c': [None, None]})
    >>> df = standardize_df_dtypes(df)
    >>> df.dtypes
    a    Float64
    b     object
    c     object
    dtype object

    # 'NaN' strings are treated as NA:
    #   ['NaN', '2'] becomes Float64 with [NA, 2.0]
    #   ['x', 'NaN'] becomes object with ['x', NA]
    """
    df = df.copy()

    # First pass: replace 'NaN' strings with pd.NA and identify empty columns
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].apply(
                lambda x: pd.NA if isinstance(x, str) and x.strip().lower() == 'nan' else x
            )
        if is_col_empty(df[col]):
            df[col] = df[col].astype("object")

    # Second pass: try to convert object columns to numeric
    for col in df.columns:
        if df[col].dtype == 'object' and not df[col].isna().all():
            converted = pd.to_numeric(df[col], errors="coerce")
            # Only convert if no new NAs were introduced
            if converted.isna().sum() == df[col].isna().sum():
                df[col] = converted

    # Third pass: standardize dtypes
    for col in df.columns:
        if df[col].isna().all():
            df[col] = df[col].astype("object")
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("Float64")
        else:
            df[col] = df[col].astype("object")

    # Fourth pass: replace numpy NaN with pd.NA in object columns
    # Float64 columns already use pd.NA natively, but object columns
    # can still contain float('nan').
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].fillna(pd.NA)

    return df


def fill_numeric_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill NA values with 0 in numeric (Float64) columns only.

    This avoids the FutureWarning from filling NA on mixed-dtype DataFrames.
    """
    df = df.copy()
    float_cols = df.select_dtypes(include=['Float64']).columns
    df[float_cols] = df[float_cols].fillna(0)
    return df


def collect_domains(df, possible_domains: list[str]) -> dict[str, list]:
    """
    Collect unique values for each domain column in the given list.

    Parameters:
    - df: pandas.DataFrame containing possible domain columns
    - possible_domains: list of domain column names to check in df

    Returns:
    - dict[str, list]: dictionary of domain -> unique values (unsorted)
    """
    result = {}

    for domain in possible_domains:
        if domain in df.columns:
            unique_values = df[domain].dropna().unique()
            if len(unique_values) > 0:
                result[domain] = list(unique_values)

    return result


def collect_domain_pairs(df, domain_pairs: list[list[str]]) -> dict[str, list[tuple]]:
    """
    Extract unique domain value pairs from the DataFrame for each domain pair,
    and return a dictionary of pair_key -> list of (value1, value2) tuples.

    Parameters:
    - df: pandas.DataFrame containing the domain columns
    - domain_pairs: list of domain pair lists, e.g. [['grid', 'node'], ['flow', 'node']]

    Returns:
    - dict[str, list[tuple]]: mapping from pair key like 'grid_node' to unique domain tuples
    """
    result = {}

    for pair in domain_pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("Each domain pair must be a list of exactly two domain names")

        domain1, domain2 = pair

        # Skip pair if any column is missing
        if domain1 not in df.columns or domain2 not in df.columns:
            continue

        # Extract and deduplicate
        pairs_df = df[[domain1, domain2]].drop_duplicates()
        pair_key = f"{domain1}_{domain2}"
        new_pairs = list(pairs_df.itertuples(index=False, name=None))

        if new_pairs:
            result[pair_key] = new_pairs

    return result


def is_val_empty(val) -> bool:
    """
    Return True if `val` is None or NaN-like (pd.NA, np.nan, etc.).

    Upstream normalization (normalize_dataframe + standardize_df_dtypes) guarantees
    that values reaching this function are only None, pd.NA/np.nan, float, or str
    — with whitespace already stripped and empty strings converted to pd.NA.
    """
    if val is None:
        return True
    try:
        return pd.isna(val)
    except (ValueError, TypeError):
        return False


def is_col_empty(s: pd.Series) -> bool:
    """
    Determine whether a pandas Series should be considered "empty."
    NaN values are always treated as empty.

    Rules:
    ------
    - Boolean columns: considered empty only if all values are NaN.
      (All-False is NOT empty.)
    - Numeric columns (excluding bool): NaNs treated as 0, so "all zero or NaN" means empty.
    - Non-numeric columns: NaN or "" (empty/whitespace-only string) counts as empty.

    Parameters
    ----------
    s : pd.Series
        The column (Series) to test.

    Returns
    -------
    bool
        True if the column is "empty" according to the above rules, False otherwise.
    """
    if len(s) == 0:
        return True

    # Booleans: usually don't drop just because all False; only NaNs count as empty
    if is_bool_dtype(s):
        return s.isna().all()

    # Numeric (excluding bool): empty if all zeros or NaN
    if is_numeric_dtype(s) and not is_bool_dtype(s):
        return (s.fillna(0) == 0).all()

    # Non-numeric: empty if all are NaN or whitespace-only strings
    na_mask = s.isna()
    # Safe elementwise test; no vectorized == on arbitrary objects
    empty_str_mask = s.map(lambda v: isinstance(v, str) and v.strip() == "")
    return (na_mask | empty_str_mask).all()