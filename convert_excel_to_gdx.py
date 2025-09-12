import subprocess
import sys
import os
from pathlib import Path

def main():
    for input_path in input_paths:
        output_path = f'{modify_path_with_new_basename(input_path)[:-5]}.gdx'
        cmd_call = f'gdxxrw Input={input_path} Output={output_path} Index = Index!'
        os.system(cmd_call)

def modify_path_with_new_basename(file_path):
    """
    Combine the basename of the file and the last folder name into a new basename.
    Remove the last folder name from the path before returning.
    """
    file_path = Path(file_path)
    last_folder = file_path.parent.name  # Get the last folder name
    original_basename = file_path.stem  # Get the file name without extension
    new_basename = f"{last_folder}_{original_basename}"  # Combine last folder and basename
    # Remove the last folder from the path
    new_path = file_path.parent.parent / f"{new_basename}{file_path.suffix}"
    return str(new_path)

def find_files_with_suffix(folder_path, suffix):
    """
    Search for files in a given folder and its subfolders with a specific suffix and return them as an array.
    """
    folder = Path(folder_path)
    return [str(file) for file in folder.rglob(f"*{suffix}") if file.is_file()]

if __name__ == "__main__":
    input_paths = find_files_with_suffix('./toolbox_workflow/input', '.xlsx')
    print(input_paths)
    main()