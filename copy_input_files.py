import os
import sys
import shutil
import pathlib
import gams.transfer as gt

def main():
    folder_name = get_input_file_folder(gdx_file)
    input_folder = pathlib.Path(f"./{folder_name}")
    copy_files(input_folder)


def get_input_file_folder(gdx_file):
    #read timeseries folder
    m = gt.Container(gdx_file)
    if "scenario_year" in m:
        folder_name = m.data['scenario_year'].records["scenario_year"].array[0]
        m.removeSymbols(symbols = "scenario_year")
    else:
        print("Data has no scenario_year selected")   
        exit(-1)
    m.write(gdx_file)
    shutil.copy(gdx_file, output_folder)
    return folder_name

def copy_files(input_folder):
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv") or filename.endswith(".gdx") or filename.endswith(".inc") or filename.endswith(".gms"):
            file_path = os.path.join(input_folder, filename)
            if os.path.isfile(file_path):
                shutil.copy(file_path, output_folder)


if __name__ == "__main__":
    gdx_file = sys.argv[1]
    output_folder = "../input"
    main()

