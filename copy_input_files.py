import os
import sys
import shutil
import pathlib
import spinedb_api as api
from spinedb_api.filters.tools import name_from_dict

def main():
    with api.DatabaseMapping(input_db) as source_db:
        scenario_name = name_from_dict(source_db.get_filter_configs()[0])
        folder_names = source_db.scenario(name=scenario_name)["alternative_name_list"]
    shutil.copy(gdx_file, output_folder)
    for folder_name in folder_names:
        ts_data_path = ne_model_path / folder_name
        if ts_data_path.exists():
            copy_files(ts_data_path)
        else:
            print(f"Warning: {ts_data_path} does not exist.")
    invest_schedule_path = pathlib.Path("input_invest_and_schedule") # target
    invest_schedule_path.mkdir(exist_ok=True)
    for file_path in input_invest_schedule.iterdir():
        shutil.copy(file_path, invest_schedule_path)


def copy_files(input_folder):
    for filename in os.listdir(input_folder):
        print(f"Copying {filename} from {input_folder} to {output_folder}")
        if any(filename.endswith(ext) for ext in [".csv", ".gdx", ".inc", ".gms"]) and not filename.startswith("inputData"):
            file_path = os.path.join(input_folder, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, output_folder)
            else:
                print(f"Warning: {file_path} is not a file.")


if __name__ == "__main__":
    include_db = False
    if len(sys.argv) != 5:
        print("Usage: python convert_excel_to_gdx.py <gdx_file> <input_db> <path_to_modelsInit>")
        sys.exit(1)
    ne_model_path = pathlib.Path(sys.argv[4])
    input_invest_schedule = pathlib.Path(sys.argv[3]) # source
    input_db = sys.argv[2]
    gdx_file = sys.argv[1]
    output_folder = pathlib.Path("input")
    main()

