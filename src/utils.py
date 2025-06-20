import sys
from gdxpds import to_gdx
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
import argparse
import time


def elapsed_time(start_time):
    elapsed_seconds = time.time() - start_time
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    return minutes, seconds


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
    

def log_status(message: str, 
               log: list[str], 
               level: str = "none", 
               section_start_length: int = 0, 
               add_empty_line_before: bool = False, 
               print_to_screen: bool = True):
    """
    Logs a formatted status message with emoji prefix and optional console print.

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
        "run": "⚡",
        "done": "🎯",
        "skip": "⏩",
        "none": ""
    }.get(level, "•")

    if add_empty_line_before:
        start = "\n"
    else:
        start = ""

    if section_start_length > 0:
        base = f"{start}--- {prefix} {message} "
        padding_length = max(section_start_length, len(base))
        formatted = base + "-" * (padding_length - len(base))
    else:
        formatted = f"{start}{prefix} {message}"

    log.append(formatted)

    if print_to_screen:
        print(formatted)

    

def check_dependencies():
    """
    Checks that Python ≥ 3.12 and pandas ≥ 2.2 are available.
    Prints warnings if either requirement is not met.
    """
    # 1) Check Python version
    py_major, py_minor = sys.version_info[:2]
    if (py_major, py_minor) < (3, 12):
        print(f"Warning: Detected Python {py_major}.{py_minor}. "
              "This code is tested on Python 3.12+. "
              "You may experience issues on older versions.")

    # 2) Check pandas version
    try:
        # split off any pre-release tags, take major/minor
        ver_parts = pd.__version__.split('.')
        pd_major, pd_minor = map(int, ver_parts[:2])
        if (pd_major, pd_minor) < (2, 2):
            print(f"Warning: Detected pandas {pd_major}.{pd_minor}. "
                  "This code is tested on pandas 2.2+. "
                  "You may experience issues on older versions.")
    except ImportError:
        print("Warning: pandas is not installed. "
              "Please install pandas ≥ 2.2 to ensure full functionality.")    


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


def copy_gams_files(input_folder: Path, output_folder: Path) -> list[str]:
    """
    Copy GAMS template files from src_files/GAMS_files/ to the given output_folder.

    Args:
        input_folder (Path): Base input folder (should contain src_files/GAMS_files).
        output_folder (Path): Output folder where GAMS files should be copied.
    """
    gams_src_folder = input_folder / "GAMS_files"

    logs = []

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

    return logs
