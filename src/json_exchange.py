import json
from pathlib import Path

def load_json(path: Path):
    """
    Load a JSON file from a given path.

    Args:
        path (Path): Path to the JSON file.

    Returns:
        dict: Parsed JSON content.
    """
    if path.exists():
        with open(path, "r") as f:
            return json.load(f)
    return {}

def save_json(path: Path, data: dict):
    """
    Save a dictionary to a JSON file.

    Args:
        path (Path): Path where JSON will be saved.
        data (dict): Data to save.
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)