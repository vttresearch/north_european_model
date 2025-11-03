# src/hash_utils.py

import hashlib
from pathlib import Path

def compute_excel_sheets_hash(file_path: Path, sheet_name_prefix: str) -> dict[str, str]:
    """
    Compute SHA-256 hashes for sheets in an Excel file that match a prefix.
    
    Args:
        file_path: Path to Excel file
        sheet_name_prefix: Case-insensitive prefix to match sheet names
        
    Returns:
        dict mapping sheet names to their SHA-256 hashes
    """
    import pandas as pd
    import io
    
    sheet_hashes = {}
    
    try:
        xls = pd.ExcelFile(file_path)
        # Match sheets by prefix (case-insensitive), same logic as read_input_excels
        matched_sheets = [
            s for s in xls.sheet_names 
            if s.lower().startswith(sheet_name_prefix.lower())
        ]
        
        for sheet_name in matched_sheets:
            try:
                # Read the sheet
                df = pd.read_excel(xls, sheet_name=sheet_name, header=0)
                
                # Convert to CSV for consistent hashing (handles the same cleaning logic)
                # Use StringIO to avoid file I/O
                buffer = io.StringIO()
                df.to_csv(buffer, index=False)
                csv_content = buffer.getvalue().encode('utf-8')
                
                # Hash the CSV content
                sha256 = hashlib.sha256()
                sha256.update(csv_content)
                sheet_hashes[sheet_name] = sha256.hexdigest()
                
            except Exception:
                # If we can't read a sheet, skip it
                continue
                
    except Exception:
        # If we can't open the file, return empty dict
        pass
    
    return sheet_hashes

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

