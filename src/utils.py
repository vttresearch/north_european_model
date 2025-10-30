import sys
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import argparse
import time
from pandas.api.types import is_numeric_dtype, is_bool_dtype
from typing import Optional


# --- module-level globals (defaults) ---
_PRINT_ALL_ELAPSED_TIMES: bool = False
_START_TIME: float = time.time()

def init_logging(*, print_all_elapsed_times: Optional[bool] = None,
                 start_time: Optional[float] = None) -> None:
    """
    Initialize module-wide logging switches without touching every call site.
    Call once at startup (after config is loaded).
    """
    global _PRINT_ALL_ELAPSED_TIMES, _START_TIME
    if print_all_elapsed_times is not None:
        _PRINT_ALL_ELAPSED_TIMES = bool(print_all_elapsed_times)
    if start_time is not None:
        _START_TIME = float(start_time)



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

    if section_start_length > 0:
        base = f"{start}---  {prefix} {elapsed_prefix} {message} "
        padding_length = max(section_start_length, len(base))
        formatted = base + "-" * (padding_length - len(base))
    else:
        formatted = f"{start}{prefix} {elapsed_prefix}{message}"

    log.append(formatted)

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
        - Python ≥ 3.12
        - pandas ≥ 2.2
        - gdxpds with accessible to_gdx
        - gams.transfer importable
        - gams executable accessible in PATH        

    Raises RuntimeError if any requirement is not met.
    """
    import importlib

    errors = []

    # 1) Check Python version
    py_major, py_minor = sys.version_info[:2]
    if (py_major, py_minor) < (3, 12):
        errors.append(f"Python {py_major}.{py_minor} detected (requires ≥3.12)")

    # 2) Check pandas version
    try:
        import pandas as pd
        pd_major, pd_minor = map(int, pd.__version__.split('.')[:2])
        if (pd_major, pd_minor) < (2, 2):
            errors.append(f"pandas {pd_major}.{pd_minor} detected (requires ≥2.2)")
    except ImportError:
        errors.append("pandas not installed")  

    # 3) Check gdxpds availability
    try:
        gdxpds = importlib.import_module("gdxpds")
    except ImportError:
        errors.append("gdxpds not installed")

    # 4) Check gams.transfer importability
    try:
        importlib.import_module("gams.transfer")
    except ImportError:
        errors.append("gams.transfer not importable (GAMS Python API missing)")

    # 5) Check gams executable availability in PATH
    gams_exec = shutil.which("gams") or shutil.which("gams.exe")
    if gams_exec is None:
        errors.append("GAMS not found in PATH")

    # Final decision
    if errors:
        msg = "Dependency check failed:\n  - " + "\n  - ".join(errors)
        raise RuntimeError(msg)



def trim_df(df, round_precision=0):
    # round to round_precision 
    df = df.round(round_precision)

    # drop empty columns
    df = df.loc[:, df.sum() != 0]

    # Remove leading and trailing rows that are fully empty/NaN.
    mask = ~df.isna().all(axis=1).to_numpy()
    if mask.any():
        first_valid_pos = np.where(mask)[0][0]
        last_valid_pos = np.where(mask)[0][-1]
        df = df.iloc[first_valid_pos:last_valid_pos + 1]

    # Convert dtypes (so that e.g. rounding to 0 decimals gives integers)
    df = df.convert_dtypes()

    return df


def standardize_df_dtypes(
    df: pd.DataFrame,
    *,
    convert_numeric: bool = False,
    fill_numeric_na: bool = False,
    treat_nan_string_as_na: bool = True,
) -> pd.DataFrame:
    """
    Standardize DataFrame column dtypes to a consistent set:
    - Empty columns (all NA) → object
    - Numeric columns → Float64
    - Everything else → object
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame to standardize
    convert_numeric : bool, default False
        If True, attempt to convert object columns to numeric using pd.to_numeric()
        before standardizing. This is useful when object columns contain numeric strings.
    fill_numeric_na : bool, default False
        If True, fill NA values in Float64 columns with 0 after conversion.
    treat_nan_string_as_na : bool, default True
        If True, replace string 'NaN' (case-insensitive) with pd.NA before processing.
        This allows numeric conversion to work properly on columns containing 'NaN' strings.
        
    Returns
    -------
    pd.DataFrame
        DataFrame with standardized dtypes
        
    Examples
    --------
    >>> # Basic standardization
    >>> df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y'], 'c': [None, None]})
    >>> df = standardize_df_dtypes(df)
    >>> df.dtypes
    a    Float64
    b     object
    c     object
    dtype object
    
    # Handle 'NaN' strings when treat_nan_string_as_na = True, 
       - ['NaN', '2'] becomes Float64 with [NA, 2.0]
       - ['x', 'NaN'] becomes object with ['x', NA]
    """
    df = df.copy()
    
    # First pass: replace 'NaN' strings with pd.NA and identify empty columns
    for col in df.columns:
        # Replace 'NaN' strings with pd.NA if requested
        if treat_nan_string_as_na and df[col].dtype == 'object':
            df[col] = df[col].apply(
                lambda x: pd.NA if isinstance(x, str) and x.strip().lower() == 'nan' else x
            )
        
        # Check if column is empty using the existing is_col_empty function
        if is_col_empty(df[col], treat_nan_as_empty=True):
            df[col] = df[col].astype("object")
    
    # Second pass: try to convert object columns to numeric if requested
    if convert_numeric:
        for col in df.columns:
            if df[col].dtype == 'object' and not df[col].isna().all():
                converted = pd.to_numeric(df[col], errors="coerce")
                # Only convert if no new NAs were introduced
                if converted.isna().sum() == df[col].isna().sum():
                    df[col] = converted
    
    # Third pass: standardize dtypes
    for col in df.columns:
        # Empty columns → object
        if df[col].isna().all():
            df[col] = df[col].astype("object")
        # Numeric columns → Float64
        elif pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("Float64")
        # Everything else → object
        else:
            df[col] = df[col].astype("object")
    
    # Fourth pass: fill NAs in Float64 columns if requested
    if fill_numeric_na:
        Float64_cols = df.select_dtypes(include=['Float64']).columns
        df[Float64_cols] = df[Float64_cols].fillna(0)
    
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


def copy_gams_files(input_folder: Path, output_folder: Path, logs: list[str]) -> list[str]:
    """
    Copy GAMS template files from src_files/GAMS_files/ to the given output_folder.

    Args:
        input_folder (Path): Base input folder (should contain src_files/GAMS_files).
        output_folder (Path): Output folder where GAMS files should be copied.
        logs (list[str]): a list of log string mutated if needed
    """
    gams_src_folder = input_folder / "GAMS_files"

    if not gams_src_folder.exists():
        log_status(f"⚠️ WARNING: GAMS source folder not found: {gams_src_folder}", logs, level="warn")
        return

    copied_any = False
    for file in gams_src_folder.glob("*.*"):
        dest = output_folder / file.name
        shutil.copy(file, dest)
        log_status(f"Copied {file.name} to {output_folder}", logs, level="info")
        copied_any = True

    if not copied_any:
        log_status(f"WARNING: No GAMS files were found to copy in {gams_src_folder}", logs, level="warn")


def is_val_empty(
    val,
    logs: list[str],
    treat_zero_as_empty: bool = True,
    ident: str = "is_val_empty",
) -> bool:
    """
    Return True if `val` is considered empty.

    Empties:
      - None
      - NaN-like scalars via pd.isna(val) (np.nan, pd.NA, pd.NaT, numpy scalar NaNs, Decimal('NaN'))
      - Empty or whitespace-only string (robust to zero-width/NBSP)
      - Empty bytes-like (bytes/bytearray/memoryview with len == 0)
      - Empty pandas objects (Series/DataFrame/Index) via `.empty`
      - Array-like with `.size == 0` (e.g., numpy arrays)
      - *Optionally* numeric zero scalars when `treat_zero_as_empty=True`

    Not empty by design:
      - Booleans (False is meaningful and never empty)
      - Numeric non-zero scalars

    Notes:
      - `treat_zero_as_empty` applies only to **scalar numerics**, not arrays/Series.
      - Any unexpected error is logged loudly via `log_status(..., level="error")`.
    """
    import numbers

    try:
        # 1) None
        if val is None:
            return True

        # 2) Strings (normalize tricky whitespace)
        if isinstance(val, str):
            s = val.replace("\u200b", "").replace("\ufeff", "").replace("\u00a0", " ")
            return s.strip() == ""

        # 3) Booleans always not empty
        if isinstance(val, bool):
            return False

        # 4) Numeric scalars (incl. numpy numbers, Decimal, etc.) 
        if isinstance(val, numbers.Number):
            if pd.isna(val):
                return True
            if treat_zero_as_empty and val == 0:
                return True
            return False

        # 5) Pandas containers
        if isinstance(val, (pd.Series, pd.DataFrame, pd.Index)):
            return val.empty

        # 6) Array-likes: empty if size == 0 (numpy arrays, etc.)
        if hasattr(val, "size"):
            return int(val.size) == 0

        # 7) Generic __len__ fallback
        if hasattr(val, "__len__"):
            return len(val) == 0

        # 8) Final NaN-like catch for other scalar types (e.g., pd.Timestamp(pd.NaT))
        if pd.isna(val):
            return True

        # Default: not empty
        return False

    except BaseException as e:
        # Loud logging on anything unexpected (no swallowing)
        log_status(
            f"[{ident}] ERROR while checking value {val!r} (type={type(val).__name__}): {e}",
            logs,
            level="error",
        )
        # Fail-safe: treat as NOT empty so we don't drop data silently
        return False



def is_col_empty(s: pd.Series, treat_nan_as_empty: bool = True) -> bool:
    """
    Determine whether a pandas Series should be considered "empty."

    Rules:
    ------
    - Boolean columns: considered empty only if all values are NaN.
      (All-False is NOT empty.)
    - Numeric columns (excluding bool):
        * With treat_nan_as_empty=True: NaNs treated as 0, so "all zero or NaN" means empty.
        * With treat_nan_as_empty=False: only strictly "all zero" counts as empty.
    - Non-numeric columns:
        * With treat_nan_as_empty=True: NaN or "" (empty/whitespace-only string) counts as empty.
        * With treat_nan_as_empty=False: only "" counts as empty.

    Parameters
    ----------
    s : pd.Series
        The column (Series) to test.
    treat_nan_as_empty : bool, default=True
        Whether NaN values should be treated as equivalent to empty.

    Returns
    -------
    bool
        True if the column is "empty" according to the above rules, False otherwise.
    """
    if len(s) == 0:
        return True

    # Booleans: usually don't drop just because all False; only NaNs count as empty
    if is_bool_dtype(s):
        return s.isna().all() if treat_nan_as_empty else False

    # Numeric (excluding bool): empty if all zeros (and optionally NaN→0)
    if is_numeric_dtype(s) and not is_bool_dtype(s):
        s_cmp = s.fillna(0) if treat_nan_as_empty else s
        return (s_cmp == 0).all()

    # Non-numeric: empty if all are "" (optionally allow NaN)
    if treat_nan_as_empty:
        na_mask = s.isna()
    else:
        na_mask = pd.Series(False, index=s.index)

    # Safe elementwise test; no vectorized == on arbitrary objects
    empty_str_mask = s.map(lambda v: isinstance(v, str) and v.strip() == "")
    # Combine the two masks (NaN OR empty string), and check if all entries satisfy
    return (na_mask | empty_str_mask).all()