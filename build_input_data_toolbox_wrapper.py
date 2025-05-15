import sys
import shutil
import os
import ast
import configparser
import build_input_data
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
from openpyxl.worksheet.table import Table, TableStyleInfo

if __name__ == '__main__':
    config_file = sys.argv[1]
    input_folder = "src_files"
    filename = os.path.basename(config_file)

    #get the folder name
    config = configparser.ConfigParser()
    config.read(config_file)
    output_folder_prefix = config.get('inputdata', 'output_folder_prefix')
    scenarios = ast.literal_eval(config.get('inputdata', 'scenarios'))
    scenario_years = ast.literal_eval(config.get('inputdata', 'scenario_years'))
    scenario_alternatives = ast.literal_eval(config.get('inputdata', 'scenario_alternatives'))
    if scenario_alternatives:
        output_folder = os.path.join(f"{output_folder_prefix}_{scenarios[0]}_{str(scenario_years[0])}_{scenario_alternatives[0]}")
    else:
        output_folder = os.path.join(f"{output_folder_prefix}_{scenarios[0]}_{str(scenario_years[0])}")

    #call the actual script
    build_input_data.run(input_folder, filename)
    
    #copy the inputdata excel to another folder for workflow use
    shutil.copy(f'{output_folder}/inputData.xlsx', "./toolbox_workflow/input/inputData.xlsx")
    
    #add folder information
    wb = load_workbook("./toolbox_workflow/input/inputData.xlsx")          
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
    wb.save("./toolbox_workflow/input/inputData.xlsx")