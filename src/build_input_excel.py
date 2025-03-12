import os
import time
from gdxpds import to_gdx
import pandas as pd
import numpy as np


class build_input_excel:
    def __init__(self, input_folder, output_folder, country_codes, df_transfers, scenario, scenario_year):
        self.input_folder = os.path.join(input_folder, "timeseries")
        self.output_folder = output_folder
        self.country_codes = country_codes
        self.df_transfers = df_transfers
        self.scenario = scenario
        self.scenario_year = scenario_year


    def log_time(self, message):
        elapsed = time.perf_counter() - self.run_start
        print(f"[{elapsed:0.2f} s] {message}")


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

        # rename from->node, to->node, return df
        df_long = df_long.rename(columns={'from': 'node', 'to': 'node'} )
        return df_long


    def write_gdx(self, df_long, output_file):
        # Assume dataframe in gdx correct long format (N dimensions, 1 value column for parameter or 0 value volumns for set)
        # Package and export the DataFrame to a GDX file.
        dataframes = {'p_gnn': df_long}
        to_gdx(dataframes, path=output_file)


    def write_excel(self, df_long, output_file):
        # Transform back to to wide format
        df_wide = df_long.pivot(index=dims, columns='param_gnn', values='value').reset_index()


    def run(self):
        # p_gnn
        p_gnn = self.create_p_gnn(self.df_transfers)
        output_file = os.path.join(self.output_folder, f'p_gnn.gdx')
        self.write_gdx(p_gnn, output_file)
        print(p_gnn)
        print(f"   p_gnn written to {output_file}")