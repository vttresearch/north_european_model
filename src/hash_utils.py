import json
import hashlib
from pathlib import Path

def compute_file_hash(file_path: Path) -> str:
    """
    Compute the SHA-256 hash of a file.

    Args:
        file_path (Path): The path to the file to be hashed.

    Returns:
        str: The SHA-256 hash of the file content.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def compute_folder_hash(folder_path: Path, extensions: list[str] = None) -> str:
    """
    Compute a combined SHA-256 hash for all files in a folder.

    Args:
        folder_path (Path): Path to the folder.
        extensions (list[str], optional): List of file extensions to include.

    Returns:
        str: Combined SHA-256 hash of all files.
    """
    sha256 = hashlib.sha256()
    for file in sorted(folder_path.rglob("*")):
        if file.is_file():
            if extensions and not any(file.name.endswith(ext) for ext in extensions):
                continue
            with open(file, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    sha256.update(chunk)
    return sha256.hexdigest()

def compute_processor_code_hash(processor_file: Path) -> str:
    """
    Compute the SHA-256 hash of a processor's code file.

    Args:
        processor_file (Path): Path to the processor code file.

    Returns:
        str: SHA-256 hash of the processor file content.
    """
    return compute_file_hash(processor_file)

