import os
import sys
import shutil
import pathlib
import spinedb_api as api
import gams.transfer as gt

def main(include_db= False):
    if include_db:
        with api.DatabaseMapping(input_db) as source_db:
            folder_names = [scenario_alternative["alternative_name"] for scenario_alternative in source_db.get_scenario_alternative_items()]
    else:
        folder_names = get_input_file_folder(gdx_file)
    for folder_name in folder_names:
        input_folder = pathlib.Path(f"./{folder_name}")
        if input_folder.exists() and input_folder.is_dir():
            copy_files(input_folder)
        else:
            print(f"Warning: {input_folder} does not exist or is not a directory.")


def get_input_file_folder(gdx_file):
    #read timeseries folder
    m = gt.Container(gdx_file)
    if "scenario_year" in m:
        folder_names = m.data['scenario_year'].records["scenario_year"].array
        m.removeSymbols(symbols = "scenario_year")
    else:
        print("Data has no scenario_year selected")   
        exit(-1)
    m.write(gdx_file)
    shutil.copy(gdx_file, output_folder)
    return folder_names

def copy_files(input_folder):
    for filename in os.listdir(input_folder):
        print(f"Copying {filename} from {input_folder} to {output_folder}")
        if filename.endswith(".csv") or filename.endswith(".gdx") or filename.endswith(".inc") or filename.endswith(".gms"):
            file_path = os.path.join(input_folder, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, output_folder)
            else:
                print(f"Warning: {file_path} is not a file.")


if __name__ == "__main__":
    include_db = False
    if len(sys.argv) < 2:
        print("Usage: python convert_excel_to_gdx.py <gdx_file>")
        sys.exit(1)
    elif len(sys.argv) == 3:
        input_db = sys.argv[2]
        include_db = True
    gdx_file = sys.argv[1]
    output_folder = "../input"
    main(include_db = include_db)

