import sys
import shutil
import os
import ast
import configparser
import build_input_data
from src.infrastructure.config_reader import config_output_folder_names, load_config
from src.utils import parse_sys_args
from openpyxl import load_workbook

if __name__ == '__main__':
    input_folder, config_file = parse_sys_args()
    #get the folder name
    config = load_config(config_file)
    # The builder's own rule, so the folders read here are the ones it wrote.
    output_folders = config_output_folder_names(config)

    #call the actual script
    build_input_data.main(input_folder, config_file)
    
    for j, output_folder in enumerate(output_folders):
        print(j)
        print(output_folder)
        #copy the inputdata excel to another folder for workflow use
        os.makedirs(f'./toolbox_workflow/input/{j}', exist_ok=True)
        shutil.copy(f'./{output_folder}/inputData.xlsx', f'./toolbox_workflow/input/{j}/inputData.xlsx')
        
        #add folder information
        wb = load_workbook(f'./toolbox_workflow/input/{j}/inputData.xlsx')          
        table_name = "scenario_folder"
        wb.create_sheet(table_name)
        for i, name in enumerate(wb.sheetnames):
            if name == "index":
                last_row = len(wb.worksheets[i]['A'])
                wb.worksheets[i].cell(row=last_row + 1, column=1).value = "Set"
                wb.worksheets[i].cell(row=last_row + 1, column=2).value = table_name
                wb.worksheets[i].cell(row=last_row + 1, column=3).value = f'{table_name}!A2'
                wb.worksheets[i].cell(row=last_row + 1, column=4).value = 1
            if name == table_name:
                wb.worksheets[i].cell(row=1, column=1).value = table_name
                wb.worksheets[i].cell(row=2, column=1).value = output_folder
        wb.save(f'./toolbox_workflow/input/{j}/inputData.xlsx')