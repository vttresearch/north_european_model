import os
import sys
from gdxpds import to_gdx
import pandas as pd
import numpy as np
import openpyxl
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.table import Table, TableStyleInfo



class build_input_excel:
    def __init__(self, input_folder, output_folder, country_codes, exclude_grids,
                 df_transfers, df_techs, df_unitdata, df_remove_units, df_storagedata,
                 df_fueldata, df_emissions, df_demands,
                 secondary_results=None
                 ):
        self.input_folder = os.path.join(input_folder, "timeseries")
        self.output_folder = output_folder
        self.country_codes = country_codes
        self.exclude_grids = exclude_grids

        self.df_transfers = df_transfers
        self.df_techs = df_techs
        self.df_unitdata = df_unitdata
        self.df_remove_units = df_remove_units
        self.df_storagedata = df_storagedata
        self.df_fueldata = df_fueldata
        self.df_emissions = df_emissions
        self.df_demands = df_demands

        self.secondary_results = secondary_results        


# ------------------------------------------------------
# Functions creating the main Backbone input parameters: p_gnn and p_gnu
# ------------------------------------------------------

    def create_p_gnu(self, df_techs, df_unitdata, exclude_grids, df_remove_units):
        """
        Creates a new DataFrame p_gnu with 
        dimension columns: [grid, node, unit, input_output]
        value columns: [capacity, conversionCoeff, vomCosts]

        For each row in df_unitdata:
        - Lookup the corresponding generator_id row in df_techs.
        - skip processing if (country, unit_name_prefix, generator_id) exists in df_remove_units
        - Build unit name: <country>_<'unit_name_prefix' if defined>_<unit_short_name>
        - for puts=['input', 'output1', 'output2', 'output3']
            * fetch matching grid <grid>_<{put}> from df_techs
            * if grid is defined
                * Build node name <grid>_<country>_<matching 'node_suffix' if defined> 
                * Build mandatory base column values            
                * Build additional parameters 
                    * values can be e.g. vomCosts_input, vomCosts_output2. If no suffix is given, the code assumes output1
        - Build, sort, and return the final dataframe 

        Blacklists grids in self.exclude_grids
        """
        rows = []

        # Process each row in the capacities DataFrame.
        for _, cap_row in df_unitdata.iterrows():

            # Fetch values for country, unit_name_prefix, and generator_id
            country = cap_row['country']
            unit_name_prefix = cap_row['unit_name_prefix'] if 'unit_name_prefix' in cap_row.index else None
            generator_id = cap_row['generator_id']

            # Get the matching technology row.
            tech_row = df_techs.loc[df_techs['generator_id'] == generator_id].iloc[0]

            # skip processing if (country, unit_name_prefix, generator_id) exists in df_remove_units
            # Compare country and generator_id case-insensitively:
            mask_country = df_remove_units["country"].str.lower() == country.lower()
            mask_generator_id = df_remove_units["generator_id"].astype(str).str.lower() == str(generator_id).lower()
            # For unit_name_prefix, use a NaN-aware comparison:
            mask_unit_prefix = (
                (df_remove_units["unit_name_prefix"] == unit_name_prefix) |
                (df_remove_units["unit_name_prefix"].isna() & pd.isna(unit_name_prefix))
            )
            # Combine all masks:
            mask = mask_country & mask_unit_prefix & mask_generator_id
            # skip if (country, unit_name_prefix, generator_id) is in df_remove_units
            if mask.any():
                continue

            # Build unit name.
            if pd.notna(unit_name_prefix) and unit_name_prefix not in ['', '-']:
                unit_name = f"{country}_{unit_name_prefix}_{tech_row['unit_short_name']}"
            else:
                unit_name = f"{country}_{tech_row['unit_short_name']}"

            # Filter only available puts based on whether 'grid_{put}' exists.
            available_puts = [put for put in ['input1', 'input2', 'input3', 'output1', 'output2', 'output3'] 
                              if f'grid_{put}' in tech_row.index]

            for put in available_puts:
                # Extract put specific grid value.
                grid = tech_row[f'grid_{put}'] if f'grid_{put}' in tech_row.index else None

                # Only process if put specific grid is defined.
                if pd.notna(grid) and grid not in ['', '-'] and grid not in exclude_grids:
                    # Build node name: <country>_<grid>_<node_suffix if has value>
                    node_name = f"{country}"
                    node_suffix = cap_row[f'node_suffix_{put}'] if f'node_suffix_{put}' in cap_row.index else None
                    if pd.notna(node_suffix) and node_suffix not in ['', '-']:
                        node_name = f"{node_name}_{node_suffix}"
                    node_name = f"{node_name}_{grid}"
                            
                    # Build base row dictionary.
                    base_row = {
                        'grid': grid,
                        'node': node_name,
                        'unit': unit_name,
                        'input_output': 'input' if put.startswith('input') else 'output',
                        'capacity': cap_row[f'capacity_{put}'] if f'capacity_{put}' in cap_row.index else 0,
                        'conversionCoeff': tech_row[f'conversionCoeff_{put}'] if f'conversionCoeff_{put}' in tech_row.index else 1
                    }

                    # Build additional parameters with a dictionary comprehension.
                    additional_params = {
                        param_gnu: (
                            (tech_row[f'{param_gnu.lower()}_{put}'] if f'{param_gnu.lower()}_{put}' in tech_row.index else 0) +
                            (tech_row[param_gnu.lower()] if (param_gnu.lower() in tech_row.index and put == 'output1') else 0)
                        )
                        for param_gnu in ['vomCosts', 'startCostCold', 'startCostWarm', 
                                        'startCostHot', 'startColdAfterXhours', 'startWarmAfterXhours',
                                        'maxRampUp', 'maxRampDown', 'rampCost',
                                        'cb', 'cv', 'upperLimitCapacityRatio']
                    }

                    # Merge the dictionaries.
                    row = {**base_row, **additional_params}
                    rows.append(row)

        # Define the final columns order.
        final_columns = [
            'grid', 'node', 'unit', 'input_output', 'capacity', 'conversionCoeff',
            'vomCosts', 'startCostCold', 'startCostWarm', 'startCostHot', 
            'startColdAfterXhours', 'startWarmAfterXhours',
            'maxRampUp', 'maxRampDown', 'rampCost', 'cb', 'cv', 'upperLimitCapacityRatio'
        ]

        # Build the final DataFrame with all columns.
        p_gnu = pd.DataFrame(rows, columns=final_columns)
        
        # Sort by unit, input_output, node in a case-insensitive manner.
        p_gnu.sort_values(by=['unit', 'input_output', 'node'], 
                            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
                            inplace=True)

        # Replace NaN values with 0 and return.
        p_gnu = p_gnu.fillna(value=0)
        return p_gnu

    def create_p_gnn(self, df_transfers, p_gnu, exclude_grids):
        """
        Creates a new DataFrame p_gnn with 
          dimension columns: [grid, from_node, to_node]
          values: [capacity, availability, variableTransCost, transferLoss]
          additional value for from_node, to_node: [rampLimit]

          Blacklists grids in exclude_grids
          Requires that nodes are present in previously created p_gnu
        """                  
        # Split the 'from-to' column into 'from_node' and 'to_node'
        df_transfers[['from_node', 'to_node']] = df_transfers['from-to'].str.split('-', expand=True)

        # List to collect the new rows for the p_gnn DataFrame
        rows = []

        # Create a set of allowed nodes for faster lookup
        allowed_nodes = set(p_gnu['node'])

        # Iterate over each row in df_transfers
        for _, row in df_transfers.iterrows():
            # Check if both nodes are in the allowed country codes
            if row['grid'] not in exclude_grids and (row['from_node']+'_'+row['grid'] in allowed_nodes) and (row['to_node']+'_'+row['grid'] in allowed_nodes):

                # If export capacity is greater than 0, add row with from_node, to_node, export_capacity
                if row['export_capacity'] > 0:
                    rows.append({
                        'grid': row['grid'],
                        'from_node': row['from_node']+'_'+row['grid'],
                        'to_node': row['to_node']+'_'+row['grid'],
                        'capacity': row['export_capacity'], 
                        'rampLimit': row['ramplimit'] if 'ramplimit' in row.index else 0,                        
                        'availability': 1,                        
                        'variableTransCost': row['vomcost'] if 'vomcost' in row.index else 0,
                        'transferLoss': row['losses'] if 'losses' in row.index else 0
                    })

                # If import capacity is greater than 0, add row with to_node, from_node, import_capacity
                if row['import_capacity'] > 0:
                    rows.append({
                        'grid': row['grid'],
                        'from_node': row['to_node']+'_'+row['grid'],
                        'to_node': row['from_node']+'_'+row['grid'],
                        'capacity': row['import_capacity'],  
                        'rampLimit': 0,    
                        'availability': 1,                        
                        'variableTransCost': row['vomcost'] if 'vomcost' in row.index else 0,
                        'transferLoss': row['losses'] if 'losses' in row.index else 0
                    })

        # Create the resulting DataFrame with the specified columns
        p_gnn = pd.DataFrame(rows, columns=[
            'grid', 'from_node', 'to_node', 'availability',
            'capacity', 'variableTransCost', 'transferLoss', 'rampLimit'
        ])

        p_gnn = p_gnn.fillna(value=0)
        return p_gnn  


# ------------------------------------------------------
# Functions to create unittype based input tables: 
# unitUnittype, flowUnit, p_unit, effLevelGroupUnit
# ------------------------------------------------------

    def create_unitUnittype(self, p_gnu):
        # Get unique unit names from the 'unit' column of p_gnu
        unique_units = p_gnu['unit'].unique()
        # Create the DataFrame using a list comprehension, taking the last part as the unittype
        unitUnittype = pd.DataFrame(
            [{'unit': unit, 'unittype': unit.split('_')[-1]} for unit in unique_units],
            columns=['unit', 'unittype']
        )
        return unitUnittype
    
    def create_flowUnit(self, df_techs, unitUnittype):
        # Filter rows in df_techs that have a non-null 'flow' value
        flowtechs = df_techs[df_techs['flow'].notnull()].copy()
        # Merge unitUnittype with flowtechs based on matching 'unittype' and 'unit_short_name'
        merged = pd.merge(unitUnittype, flowtechs, left_on='unittype', right_on='unit_short_name', how='inner')
        # Create the final DataFrame with only 'flow' and 'unit' columns
        flowUnit = merged[['flow', 'unit']]
        return flowUnit
    
    def create_p_unit(self, df_techs, unitUnittype):
        rows = []
        # Iterate over each row in unitUnittype
        for _, u_row in unitUnittype.iterrows():
            # Retrieve the matching row from df_techs
            tech_row = df_techs[df_techs['unit_short_name'] == u_row['unittype']].iloc[0]

            # Build the param_unit dictionary using values from the matching row
            row_data = {
                'unit': u_row['unit'],
                'isSource': tech_row['isSource']  if 'isSource' in tech_row.index else 0,
                'isSink': tech_row['isSink']  if 'isSink' in tech_row.index else 0,
                'availability': 1,
                'eff00': tech_row['efficiency'],
                'eff01': tech_row['efficiency'],
                'op00': tech_row['minstablegen'] if 'minstablegen' in tech_row.index else 0,
                'op01': 1,
                'minOperationHours': tech_row['minOperationHours']  if 'minOperationHours' in tech_row.index else 0,
                'minShutdownHours': tech_row['minShutDownHours']  if 'minShutDownHours' in tech_row.index else 0,
            }

            # Append the unit and its parameter dictionary to the list
            rows.append(row_data)

        # Create the DataFrame with the desired column order
        columns = [
            'unit', 'isSource', 'isSink', 'availability', 
            'eff00', 'eff01', 'op00', 'op01', 
            'minOperationHours', 'minShutdownHours'
        ]
        p_unit = pd.DataFrame(rows, columns=columns)
        p_unit = p_unit.fillna(value=0)
        return p_unit
    
    def create_effLevelGroupUnit(self, df_techs, unitUnittype):
        # List to accumulate new rows
        rows = []

        # Iterate over each row in unitUnittype
        for _, u_row in unitUnittype.iterrows():
            unit = u_row['unit']
            unittype = u_row['unittype']

            # Retrieve the matching row from df_techs where 'unit_short_name' equals the unittype value
            tech_matches = df_techs[df_techs['unit_short_name'] == unittype]
            tech_row = tech_matches.iloc[0]

            # LP/MIP value
            lp_mip = tech_row['LP/MIP'] if 'minOperationHours' in tech_row.index else None

            if lp_mip in ['LP', 'MIP']:
                # effLevel1 = MIP/LP
                rows.append({
                    'effLevel': f'level{i}',
                    'effSelector': f'directOn{lp_mip}',
                    'unit': unit
                })
                # effLevel2-3 = LP
                for i in range(2,4):
                    rows.append({
                        'effLevel': f'level{i}',
                        'effSelector': 'directOnLP',
                        'unit': unit
                    })

        # Create a new DataFrame from the list of rows with the desired columns
        effLevelGroupUnit = pd.DataFrame(rows, columns=['effLevel', 'effSelector', 'unit'])
        return effLevelGroupUnit                                                                                                                     


# ------------------------------------------------------
# Functions to create node based input tables: 
# p_gn, p_gnBoundaryPropertiesForStates, ts_priceChange, 
# ------------------------------------------------------

    def create_p_gn(self, p_gnn, p_gnu, df_fueldata, df_demands, df_storagedata):
        # Build node name to df_storagedata: <country>_<grid>_<node_suffix>
        df_storagedata['node'] = df_storagedata.apply(
            lambda row: f"{row['country']}_{row['grid']}" + 
                        (f"_{row['node_suffix']}" if pd.notnull(row['node_suffix']) and row['node_suffix'] != "" else ""),
            axis=1
        )

        # Extract gn pairs from df_storagedata, p_gnn, and p_gnu
        pairs_df_storagedata = df_storagedata[['grid', 'node']]
        pairs_from = p_gnn[['grid', 'from_node']].rename(columns={'from_node': 'node'})
        pairs_to = p_gnn[['grid', 'to_node']].rename(columns={'to_node': 'node'})
        pairs_gnu = p_gnu[['grid', 'node']]
        # Concatenate and drop duplicates
        unique_gn_pairs = pd.concat([pairs_df_storagedata, pairs_from, pairs_to, pairs_gnu]).drop_duplicates()

        # Grids in df_demands for quick membership test
        demand_grids = df_demands['grid'].unique()

        # Process each (grid, node) pair:
        rows = []
        for idx, row in unique_gn_pairs.iterrows():
            grid = row['grid']
            node = row['node']

            # Determine usePrice: if any fuel record for this grid has price > 0.
            fuels_for_grid = df_fueldata[df_fueldata['grid'] == grid]
            if not fuels_for_grid.empty and (fuels_for_grid['price'] > 0).any():
                usePrice = True
            else:
                usePrice = False
            # useBalance is simply the opposite of usePrice.
            useBalance = not usePrice

            # check if node is a storage node based on multiple criteria:
            # 1. if in df_storagedata
            is_storage = ((pairs_df_storagedata['grid'] == grid) & 
                          (pairs_df_storagedata['node'] == node)).any()

            # 2. if 'upperLimitCapacityRatio' is defined to any (grid, node)
            subset_p_gnu = p_gnu[(p_gnu['grid'] == grid) & (p_gnu['node'] == node)]
            if not is_storage and not subset_p_gnu.empty:
                is_storage = ((subset_p_gnu['upperLimitCapacityRatio'].notnull()) 
                              & (subset_p_gnu['upperLimitCapacityRatio'] != 0)
                              ).any()
            # 3. if, in p_gnu, for this (grid, node), there are both 'input' and 'output' roles
            io_roles = set(subset_p_gnu['input_output'])
            has_both_io = ('input' in io_roles) and ('output' in io_roles)
            # Also, the grid should not appear in df_demands.
            grid_not_in_demands = grid not in demand_grids
            # Determine energyStoredPerUnitOfState:
            if is_storage or (has_both_io and grid_not_in_demands):
                energyStoredPerUnitOfState = 1
            else:
                energyStoredPerUnitOfState = 0

            # Build the record dictionary.
            row = {
                'grid': grid,
                'node': node,
                'usePrice': usePrice,
                'useBalance': useBalance,
                'energyStoredPerUnitOfState': energyStoredPerUnitOfState
            }
            rows.append(row)

        # Convert list of records to a DataFrame and return.
        p_gn = pd.DataFrame(rows)
        p_gn = p_gn.fillna(value=0)
        return p_gn

    def create_p_gnBoundaryPropertiesForStates(self, p_gn, df_storagedata):
        import pandas as pd  # Ensure pandas is imported if not already

        # Build the 'node' column in df_storagedata: <country>_<grid>_<node_suffix>
        df_storagedata['node'] = df_storagedata.apply(
            lambda row: f"{row['country']}_{row['grid']}" + 
                        (f"_{row['node_suffix']}" if pd.notnull(row['node_suffix']) and row['node_suffix'] != "" else ""),
            axis=1
        )

        rows = []
        # Loop through each row in p_gn that has energyStoredPerUnitOfState equal to 1
        for _, gn_row in p_gn.iterrows():
            if gn_row.get('energyStoredPerUnitOfState', 0) == 1:
                grid = gn_row['grid']
                node = gn_row['node']

                # Find the corresponding row in df_storagedata where both grid and node match
                mask = (df_storagedata['grid'] == grid) & (df_storagedata['node'] == node)
                if mask.any():
                    storage_row = df_storagedata[mask].iloc[0]
                else:
                    continue

                # Build the dictionary for the new row
                row_dict = {'grid': grid, 'node': node}

                # List of optional keys to include if they exist in df_storagedata
                optional_keys = ['upwardLimit', 'reference', 'balancePenalty', 'selfDischargeLoss', 'maxSpill']
                for key in optional_keys:
                    if key.lower() in df_storagedata.columns:
                        # If a matching storage_row was found, retrieve the value; otherwise, set to None
                        row_dict[key] = storage_row[key.lower()] if storage_row is not None and key.lower() in storage_row else None

                rows.append(row_dict)

        # Create a DataFrame from the list of row dictionaries
        p_gnBoundaryPropertiesForStates = pd.DataFrame(rows)
        p_gnBoundaryPropertiesForStates = p_gnBoundaryPropertiesForStates.fillna(value=0)
        return p_gnBoundaryPropertiesForStates
        
    def create_ts_priceChange(self, p_gn, df_fueldata):
        # Identify the price column in df_fueldata (case-insensitive)
        price_col = next((col for col in df_fueldata.columns if col.lower() == 'price'), None)
        if price_col is None:
            raise ValueError("No 'price' column found in df_fueldata (case-insensitive)")

        rows = []
        # Loop through each row in p_gn using the columns: grid, node, and usePrice
        for _, row in p_gn.iterrows():
            grid_value = row['grid']
            node_value = row['node']

            # Retrieve the node_price from df_fueldata where the grid value matches.
            matching_rows = df_fueldata[df_fueldata['grid'] == grid_value]
            if not matching_rows.empty:
                node_price = matching_rows.iloc[0][price_col]

                # Create a dictionary for the new row
                row_dict = {
                    'node': node_value,
                    't': 't000001',
                    'value': node_price                    
                }
                rows.append(row_dict)

        # Create the ts_priceChange DataFrame from the list of row dictionaries
        ts_priceChange = pd.DataFrame(rows)
        return ts_priceChange

    def create_p_userConstraint(self, p_gnu, **kwargs):
    
        # Filter kwargs {varname: var} where varname starts with 'mingen'
        mingen_vars = {key: value for key, value in kwargs.items() if key.startswith("mingen")}

        # Assuming each mingen_vars value is a list, use list comprehension to flatten them
        mingen_nodes = [item for sublist in mingen_vars.values() for item in sublist]

        # creating empty p_userconstraint df
        p_userConstraint = pd.DataFrame(columns=['group', '1st dimension', '2nd dimension', '3rd dimension', '4th dimension', 'parameter', 'value'])

        for node in mingen_nodes:
            # Filter rows in p_gnu where 'node' equals current node and 'input_output' is 'input'
            row_gnu = p_gnu[(p_gnu['node'] == node) & (p_gnu['input_output'] == 'input')]
            group_UC = 'UC_' + node

            # For each matching row, add a row to p_userConstraint
            for _, row in row_gnu.iterrows():
                p_userConstraint.loc[len(p_userConstraint)] = [
                    group_UC,          # group_UC
                    row['grid'],       # 1st dimension
                    node,              # 2nd dimension
                    row['unit'],       # 3rd dimension
                    "-",               # 4th dimension
                    "v_gen",           # parameter
                    1                  # value
                ]
            # Add row for parameter "GT"
            p_userConstraint.loc[len(p_userConstraint)] = [
                group_UC, "-", "-", "-", "-", "GT", 1
            ]
            # Add row for parameter "ts_groupPolicy"
            p_userConstraint.loc[len(p_userConstraint)] = [
                group_UC, "userconstraintRHS", "-", "-", "-", "ts_groupPolicy", 1
            ]

        return p_userConstraint



# ------------------------------------------------------
# Functions to create emission based input tables: 
# p_nEmission, ts_emissionPriceChange, 
# ------------------------------------------------------


# ------------------------------------------------------
# Functions to create various features: 
# minimum generation limits
# ------------------------------------------------------

# ------------------------------------------------------
# Functions to create domains: 
# grid, node, flow, unittype, unit, group
# ------------------------------------------------------


# ------------------------------------------------------
# Function used to fine tune excel file after writing it
# ------------------------------------------------------

    def adjust_excel(self, output_file):
        """
        Adjusts the width of the columns and adds table formatting to all sheets in the specified Excel file.
        Each sheet gets its columns auto-adjusted based on the maximum length of the cell values,
        and an Excel table is created using the top row as headers.
        Sheets with only one row will remain unchanged.
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

            # Skip the remaining adjustments if sheet has only one row
            if ws.max_row == 1:
                continue

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
   
    def check_if_values_defined(self, df1, name1, df2, name2, column_name):

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
        self.check_if_values_defined(self.df_techs, 'techdata_files', self.df_unitdata, 'unitdata_files', 'generator_id')

        # main input tables
        p_gnu = self.create_p_gnu(self.df_techs, self.df_unitdata, self.exclude_grids, self.df_remove_units)
        self.check_if_values_defined(p_gnu, 'techdata_files', self.df_transfers, 'transferdata', 'grid')
        p_gnn = self.create_p_gnn(self.df_transfers, p_gnu, self.exclude_grids)

        # unittype based input tables
        unitUnittype = self.create_unitUnittype(p_gnu)
        flowUnit = self.create_flowUnit(self.df_techs, unitUnittype)
        p_unit = self.create_p_unit(self.df_techs, unitUnittype)
        effLevelGroupUnit = self.create_effLevelGroupUnit(self.df_techs, unitUnittype)

        # node based input tables
        self.check_if_values_defined(p_gnu, 'techdata_files', self.df_storagedata, 'storagedata', 'grid')
        p_gn = self.create_p_gn(p_gnn, p_gnu, self.df_fueldata, self.df_demands, self.df_storagedata)
        p_gnBoundaryPropertiesForStates = self.create_p_gnBoundaryPropertiesForStates(p_gn, self.df_storagedata)
        ts_priceChange = self.create_ts_priceChange(p_gn, self.df_fueldata)
        p_userconstraint = self.create_p_userConstraint(p_gnu, **self.secondary_results)
        #print(p_userconstraint)

         # emission based input tables


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
            # main input tables
            p_gnn.to_excel(writer, sheet_name='p_gnn', index=False)
            p_gnu.to_excel(writer, sheet_name='p_gnu', index=False)
            # unittype based input tables
            unitUnittype.to_excel(writer, sheet_name='unitUnittype', index=False)     
            flowUnit.to_excel(writer, sheet_name='flowUnit', index=False)            
            p_unit.to_excel(writer, sheet_name='p_unit', index=False)       
            effLevelGroupUnit.to_excel(writer, sheet_name='effLevelGroupUnit', index=False)          
            # node based input tables
            p_gn.to_excel(writer, sheet_name='p_gn', index=False)      
            p_gnBoundaryPropertiesForStates.to_excel(writer, sheet_name='p_gnBoundaryPropertiesForStates', index=False)        
            ts_priceChange.to_excel(writer, sheet_name='ts_priceChange', index=False)   
            p_userconstraint.to_excel(writer, sheet_name='p_userconstraint', index=False)     

            # !!!!!!!!!!!!!!!
            # write scen_year_alt sheet        
            # !!!!!!!!!!!!!!!


        # Apply the adjustment method on the Excel file
        self.adjust_excel(merged_output_file)
