import os
import sys
from gdxpds import to_gdx
import pandas as pd
import numpy as np
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo



class build_input_excel:
    def __init__(self, input_folder, output_folder, 
                 scenario, scenario_year, country_codes, exclude_grids,
                 df_transfers, df_techs, df_capacities, df_fuels, df_emissions 
                 ):
        self.input_folder = os.path.join(input_folder, "timeseries")
        self.output_folder = output_folder
        self.country_codes = country_codes
        self.scenario = scenario
        self.scenario_year = scenario_year
        self.exclude_grids = exclude_grids

        self.df_transfers = df_transfers
        self.df_techs = df_techs
        self.df_capacities = df_capacities
        self.df_fuels = df_fuels
        self.df_emissions = df_emissions


# ------------------------------------------------------
# Functions creating the main Backbone input parameters: p_gnn, p_gnu, and p_unit
# ------------------------------------------------------
    def create_p_gnn(self, df):

        # Drop 'scenario' and 'year' columns.
        df = df.drop(columns=['scenario', 'year'])

        # Split the 'from-to' column into 'from' and 'to' columns.
        df[['from', 'to']] = df['from-to'].str.split('-', expand=True)
        # Drop the original 'from-to' column.
        df = df.drop(columns=['from-to'])
        # Filter rows to include only those where both 'from' and 'to' are in self.country_codes.
        df = df[df['from'].isin(self.country_codes) & df['to'].isin(self.country_codes)]

        # Define dimension columns for the GAMS parameter.
        dims = ['grid', 'from', 'to']
        # Define possible value columns. Note: all column names are converted to lower case in build_input_data.py
        value_cols = ['export_capacity', 'import_capacity', 'ramplimit', 'losses', 'vomcost']

        # Filter value columns existing in df
        existing_value_cols = [col for col in value_cols if col in df.columns]

        # Convert the DataFrame from wide format (multiple value columns) to long format (one value column).
        # This creates a new column 'param_gnn' that takes values from the names in value_cols
        # and numeric data of the new 'value' column from initial values.
        df_long = df.melt(id_vars=dims, value_vars=existing_value_cols, 
                      var_name='param_gnn', value_name='value')
        
        # Drop rows with NaN values and zeros.
        df_long = df_long.dropna(subset=['value'])
        df_long = df_long[df_long['value'] != 0]

        # rename certain parameters
        df_long['param_gnn'] = df_long['param_gnn'].replace({
            'export_capacity': 'transferCap',
            'vomcost': 'variableTransCost',
            'losses': 'transferLoss'
            })

        # For import_capacity rows, swap 'from' and 'to', then rename to 'transferCap'
        mask = df_long['param_gnn'] == 'import_capacity'
        df_long.loc[mask, ['from', 'to']] = df_long.loc[mask, ['to', 'from']].values
        df_long.loc[mask, 'param_gnn'] = 'transferCap'
      
        # Copy variableTransCost and transferLoss to swapped to-from dims.
        mask = df_long['param_gnn'].isin(['variableTransCost', 'transferLoss'])
        df_swapped = df_long[mask].copy()
        df_swapped[['from', 'to']] = df_swapped[['to', 'from']].values
        df_long = pd.concat([df_long, df_swapped], ignore_index=True)

        # Add a new parameter 'availability' with value 1 for each unique combination of dims.
        added_df = df_long[dims].drop_duplicates().assign(param_gnn='availability', value=1)
        df_long = pd.concat([df_long, added_df], ignore_index=True)

        # rename from->from_node, to->to_node
        df_long = df_long.rename(columns={'from': 'from_node', 'to': 'to_node'})
        return df_long


    def create_p_gnu(self, df_techs, df_capacities):
        """
        Creates a new DataFrame p_gnu with 
          dimension columns: [grid, node, unit, input_output]
          value columns: [availability, capacity, conversionCoeff, vomCosts]

        For each (country, node_suffix, generator_id) in df_capacities:
          - Lookup the corresponding generator_id row in df_techs.
          - Build unit name: <country>_<'unit_name_prefix' if defined>_<unit_short_name>
          - for puts=['input', 'output1', 'output2', 'output3']
              * fetch matching grid <grid>_<{put}> from df_techs
                  * Build node name <grid>_<country>_<matching 'node_suffix' if defined> 
                  * Fetch value column values            
                  * add row to p_gnu if the corresponding grid is defined
        """
        rows = []

        # Helper function to append a row with dimensions and value fields.
        def add_row(grid, node, unit, input_output, value_dims):
            row = {
                'grid': grid,
                'node': node,
                'unit': unit,
                'input_output': input_output,
                }
            row.update(value_dims)
            rows.append(row)

        # Process each row in the capacities DataFrame.
        for _, cap_row in df_capacities.iterrows():
            country = cap_row['country']
            generator_id = cap_row['generator_id']
            unit_name_prefix = cap_row['unit_name_prefix'] if 'unit_name_prefix' in cap_row.index else None

            # Get the matching technology row.
            tech_row = df_techs.loc[df_techs['generator_id'] == generator_id].iloc[0]
            unit_short_name = tech_row['unit_short_name'] 

            # Build unit name.
            if pd.notna(unit_name_prefix) and unit_name_prefix != '' and unit_name_prefix != '-':
                unit_name = f"{country}_{unit_name_prefix}_{unit_short_name}"
            else:
                unit_name = f"{country}_{unit_short_name}"
            
            # inputs and outputs
            puts = ['input', 'output1', 'output2', 'output3']
            for put in puts:
                # Extract grid value.
                grid = tech_row[f'grid_{put}'] if f'grid_{put}' in tech_row.index else None
                node_suffix = cap_row[f'node_suffix_{put}'] if f'node_suffix_{put}' in cap_row.index else None
                # if grid is defined.
                if pd.notna(grid) and grid != '' and grid != '-':
                    # Build node name 
                    if pd.notna(node_suffix) and node_suffix != '' and node_suffix != '-':
                        node_output = f"{grid}_{country}_{node_suffix}"
                    else:
                        node_output = f"{grid}_{country}"

                    # Fetch values
                    values = {
                        'availability': 1,
                        'capacity': cap_row[f'capacity_{put}'] if f'capacity_{put}' in cap_row.index else 0,
                        'conversionCoeff': tech_row[f'conversionCoeff_{put}'] if f'conversionCoeff_{put}' in tech_row.index else 1,
                        'vomCosts': tech_row[f'vomCosts_{put}'] if f'vomCosts_{put}' in tech_row.index else 0
                    }
                    # Add row
                    if put == 'input':
                        add_row(grid, node_output, unit_name, 'input', values)
                    else:
                        add_row(grid, node_output, unit_name, 'output', values)

        # Build and return the final DataFrame with the specified columns.
        p_gnu = pd.DataFrame(rows, columns=['grid', 'node', 'unit', 'input_output',
                                              'availability', 'capacity', 'conversionCoeff', 'vomCosts'])
        
        # sort by unit, input_output, node in case insensitive method
        p_gnu.sort_values(by=['unit', 'input_output', 'node'], 
                          key=lambda col: col.str.lower() if col.dtype == 'object' else col,
                          inplace=True
                          )

        return p_gnu


# ------------------------------------------------------
# Functions to create sets based on the main input tables
# ------------------------------------------------------

    def create_unittype(self, df_techdata):
        # Select only the 'generator_id' column and drop duplicate values
        df_unittype = df_techdata[['generator_id']].drop_duplicates().copy() 
        # Rename 'generator_id' to 'unittype'
        df_unittype.rename(columns={'generator_id': 'unittype'}, inplace=True)
        # Add a new column 'value' with all values set to 'yes'
        df_unittype['value'] = 'yes'
        # Sort the DataFrame alphabetically 
        df_unittype.sort_values(by='unittype', inplace=True)
        return df_unittype


# ------------------------------------------------------
# Utility functions
# ------------------------------------------------------
    def transform_to_wide(self, df_long):
        """
        Convert a long DataFrame to wide format if applicable
        """
        # Find the parameter column (only one is expected)
        param_col = next((col for col in df_long.columns if col.startswith('param_')), None)
        # Pivot to wide format if 'value' is numeric and the parameter column exists
        if pd.api.types.is_numeric_dtype(df_long['value']) and param_col:
            dimensions = [col for col in df_long.columns if col not in ['value', param_col]]
            df_wide = df_long.pivot(index=dimensions, columns=param_col, values='value').reset_index()
        else:
            df_wide = df_long.copy()
        return df_wide


    def check_if_file_open(self, file_path):
        """
        Checks if the file at file_path is locked (e.g. currently open in Excel).
        If it is, raises an exception with an informative message.
        """
        if os.path.exists(file_path):
            try:
                with open(file_path, 'a'):
                    pass
            except Exception as e:
                raise Exception(f"The Excel file '{file_path}' is currently open. Please close it before proceeding.")


    def write_gdx(self, df_long, output_file, param_name):       
        # Assume dataframe in gdx correct long format (N dimensions, 1 value column for parameter or 0 value volumns for set)
        # Package and export the DataFrame to a GDX file.
        dataframes = {param_name: df_long}
        to_gdx(dataframes, path=output_file)


# ------------------------------------------------------
# Function used to fine tune excel file after writing it
# ------------------------------------------------------

    def adjust_excel(self, output_file):
        """
        Adjusts the width of the columns and adds table formatting to all sheets in the specified Excel file.
        Each sheet gets its columns auto-adjusted based on the maximum length of the cell values,
        and an Excel table is created using the top row as headers.
        """
        wb = openpyxl.load_workbook(output_file)

        for ws in wb.worksheets:
            # Adjust each column's width in the worksheet
            for col_cells in ws.columns:
                max_length = 0
                col_letter = get_column_letter(col_cells[0].column)
                for cell in col_cells:
                    if cell.value is not None:
                        max_length = max(max_length, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = max_length + 6  # add extra padding

            # Freeze the first row by setting the freeze pane to cell A2
            ws.freeze_panes = "A2"

            # Determine the table range based on the worksheet's dimensions
            # Only add a table if there is more than a single cell with data.
            if ws.dimensions != "A1:A1" or (ws["A1"].value is not None):
                table_range = ws.dimensions
                # Generate a unique table name based on the sheet title (spaces replaced with underscores)
                table_name = f"Table_{ws.title.replace(' ', '_')}"
                table = Table(displayName=table_name, ref=table_range)

                # Define a table style (you can change the style name as needed)
                style = TableStyleInfo(name="TableStyleMedium9",
                                       showFirstColumn=False,
                                       showLastColumn=False,
                                       showRowStripes=True,
                                       showColumnStripes=False)
                table.tableStyleInfo = style

                # Add the table to the worksheet
                ws.add_table(table)

        wb.save(output_file)


# ------------------------------------------------------
# Functions for quality and consistency checks
# ------------------------------------------------------

    def check_if_values_defined(self, df1, df2, name1, name2, column_name):

        # checks that columns exist
        if column_name not in df1.columns:
            raise Exception(f"The {column_name} column is missing from {name1}.")
        if column_name not in df2.columns:
            raise Exception(f"The {column_name} column is missing from {name2}.")
        
        # converts values to lower case
        col_df1 = set(df1[column_name].astype(str).str.lower())
        col_df2 = set(df2[column_name].astype(str).str.lower())

        # Checks if all values used in df2['column_name'] are defined in df1['column_name']
        missing_ids = set(col_df2) - set(col_df1)
        if missing_ids:
            raise ValueError(f"{name2} have following values in column '{column_name}' that are not defined in {name1}: {missing_ids}")


# ------------------------------------------------------
# Main entry point for the script
# ------------------------------------------------------

    def run(self):

        # quality checks
        self.check_if_values_defined(self.df_techs, self.df_capacities, 'techdata_files', 'capacitydata_files', 'generator_id')

        # p_gnn
        p_gnn = self.create_p_gnn(self.df_transfers)
        p_gnn = self.transform_to_wide(p_gnn)

        # p_gnu
        p_gnu = self.create_p_gnu(self.df_techs, self.df_capacities)
        print(p_gnu)

        # unittype
        unittype = self.create_unittype(self.df_techs)


        # Define the merged output file
        merged_output_file = os.path.join(self.output_folder, 'inputData.xlsx')

        # Check if the Excel file is already open before proceeding
        try: 
            self.check_if_file_open(merged_output_file)
        except Exception as e:
            print(e)
            sys.exit(1)

        # Write DataFrames to different sheets of the merged Excel file
        with pd.ExcelWriter(merged_output_file) as writer:
            p_gnn.to_excel(writer, sheet_name='p_gnn', index=False)
            p_gnu.to_excel(writer, sheet_name='p_gnu', index=False)
            unittype.to_excel(writer, sheet_name='unittype', index=False)     

        # Apply the adjustment method on the Excel file
        self.adjust_excel(merged_output_file)
