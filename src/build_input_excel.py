import os
import sys
#from gdxpds import to_gdx
import pandas as pd
#import numpy as np
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from openpyxl.styles import Alignment
from openpyxl.worksheet.table import Table, TableStyleInfo
import re


class build_input_excel:
    def __init__(self, input_folder, output_folder, country_codes, exclude_grids, scen_tags,
                 df_transfers, df_techs, df_units, df_remove_units, df_storages,
                 df_fuels, df_emissions, df_demands,
                 secondary_results=None
                 ):
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.country_codes = country_codes
        self.exclude_grids = exclude_grids

        self.scen_tags = scen_tags

        self.df_transfers = df_transfers
        self.df_techs = df_techs
        self.df_units = df_units
        self.df_remove_units = df_remove_units
        self.df_storages = df_storages
        self.df_fuels = df_fuels
        self.df_emissions = df_emissions
        self.df_demands = df_demands

        self.secondary_results = secondary_results 


# ------------------------------------------------------
# Functions creating the main Backbone input parameters: p_gnn and p_gnu
# ------------------------------------------------------

    def create_p_gnu_io(self, df_unittypes, df_units):
        """
        Creates a new DataFrame p_gnu_io with specified dimension and parameter columns

        For each row in df_units:
        - Lookup the corresponding generator_id row in df_techs.
        - Build unit name: <country>_<'unit_name_prefix' if defined>_<unittype>
        - for puts=['input', 'output1', 'output2', 'output3']
            * fetch matching grid <grid>_<{put}> from df_techs
            * if grid is defined (not NaN, empty, or -)
                * Build node name <country>_<grid>_<node_suffix if defined>
                * Build mandatory base column values            
                * Build additional parameters 
                    * values can be e.g. vomCosts_input, vomCosts_output2. If no suffix is given, the code assumes output1
        - Construct 2 levels of column headers: level1 from dimensions and level2 for param_gnu that are present. Using empty name for the counterpart.
        - Build, sort, and return the final dataframe 

        Blacklists grids in self.exclude_grids
        """
        # dimension and parameter columns
        dimensions = ['grid', 'node', 'unit', 'input_output']
        param_gnu = [ 'capacity', 'conversionCoeff', 'vomCosts', 'startCostCold', 
                      'startCostWarm', 'startCostHot', 
                      'maxRampUp', 'maxRampDown', 'rampCost', 
                      'cb', 'cv', 
                      'upperLimitCapacityRatio'
                    ]
        # List to collect the new rows 
        rows = []

        # Process each row in the capacities DataFrame.
        for _, cap_row in df_units.iterrows():
            
            # Fetch values for country, unit_name_prefix, and generator_id
            country = cap_row['country']
            unit_name_prefix = cap_row['unit_name_prefix'] if 'unit_name_prefix' in cap_row.index else None
            generator_id = cap_row['generator_id']

            # Get the matching technology row.
            try:
                tech_row = df_unittypes.loc[df_unittypes['generator_id'] == generator_id].iloc[0]
            # print warning and skip unit if unittype data not available
            except:
                print(f"   !!! Generator_ID '{generator_id}' from unitdata files does not have a matching generator_id in any of the unittypedata files. Check spelling of generator_id in all files.")
                continue

            # Build unit name.
            if pd.notna(unit_name_prefix) and unit_name_prefix not in ['', '-']:
                unit_name = f"{country}_{unit_name_prefix}_{tech_row['unittype']}"
            else:
                unit_name = f"{country}_{tech_row['unittype']}"

            # Filter only available puts based on whether 'grid_{put}' exists.
            available_puts = [put for put in ['input1', 'input2', 'input3', 'output1', 'output2', 'output3'] 
                              if f'grid_{put}' in tech_row.index]

            for put in available_puts:
                # Extract put specific grid value.
                grid = tech_row[f'grid_{put}'] if f'grid_{put}' in tech_row.index else None

                # Only process if put specific grid is defined.
                if pd.notna(grid) and grid not in ['', '-']:
                    # Build node name: <country>_<grid>_<node suffix if defined>
                    node_name = f"{country}_{grid}"
                    node_suffix = cap_row[f'node_suffix_{put}'] if f'node_suffix_{put}' in cap_row.index else None
                    if pd.notna(node_suffix) and node_suffix not in ['', '-']:
                        node_name = f"{node_name}_{node_suffix}"
                            
                    # Build base row dictionary.
                    base_row = {
                        'grid' : grid,
                        'node' : node_name,
                        'unit' : unit_name,
                        'input_output': 'input' if put.startswith('input') else 'output',
                        'capacity' : cap_row[f'capacity_{put}'] if f'capacity_{put}' in cap_row.index else 0,
                        # note: lower case conversioncoeff when picking values from tech_row
                        'conversionCoeff' : tech_row[f'conversioncoeff_{put}'] if f'conversioncoeff_{put}' in tech_row.index else 1
                    }

                    # Build additional parameters with a dictionary comprehension.
                    additional_params = {
                        param : (
                            (tech_row[f'{param.lower()}_{put}'] if f'{param.lower()}_{put}' in tech_row.index else 0) +
                            (tech_row[param.lower()] if (param.lower() in tech_row.index and put == 'output1') else 0)
                        )
                        for param in param_gnu if param not in ['capacity', 'conversionCoeff']
                    }

                    # Merge the dictionaries.
                    row = {**base_row, **additional_params}
                    rows.append(row)

        # create p_gnu
        p_gnu_io = self.create_fake_multiIndex(dimensions, param_gnu, rows)

        # Sort by unit, input_output, node in a case-insensitive manner.
        p_gnu_io.sort_values(by=['unit', 'input_output', 'node'], 
                            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
                            inplace=True)

        return p_gnu_io

    def create_p_gnn(self, df_transfers, p_gnu_io_flat, exclude_grids):
        """
        Creates a new DataFrame p_gnn with specified dimension and parameter columns

          Blacklists grids in exclude_grids
          Requires that nodes are present in previously created p_gnu
        """          
        # dimension and parameter columns
        dimensions = ['grid', 'from_node', 'to_node']
        param_gnn = ['transferCap', 'availability', 'variableTransCost', 'transferLoss', 'rampLimit']
        # List to collect the new rows 
        rows = []

        # Split the 'from-to' column into 'from_node' and 'to_node'
        df_transfers[['from_node', 'to_node']] = df_transfers['from-to'].str.split('-', expand=True)

        # Create a set of allowed nodes for faster lookup
        allowed_nodes = set(p_gnu_io_flat['node'])

        # Iterate over each row in df_transfers
        for _, row in df_transfers.iterrows():
            # Check if both nodes are in the allowed country codes
            if row['grid'] not in exclude_grids and (row['from_node']+'_'+row['grid'] in allowed_nodes) and (row['to_node']+'_'+row['grid'] in allowed_nodes):

                # If export capacity is greater than 0, add row with from_node, to_node, export_capacity
                if row['export_capacity'] > 0:
                    from_to_capacity = row['export_capacity']
                    rampLimit = row['ramplimit'] if 'ramplimit' in row.index else 0 
                    rows.append({
                        'grid' :                        row['grid'],
                        'from_node' :                   row['from_node']+'_'+row['grid'],
                        'to_node' :                     row['to_node']+'_'+row['grid'],
                        'transferCap' :                 from_to_capacity,                       
                        'availability' :                1,                        
                        'variableTransCost' :           row['vomcost'] if 'vomcost' in row.index else 0,
                        'transferLoss' :                row['losses'] if 'losses' in row.index else 0,
                        'rampLimit' :                   rampLimit  
                    })

                # If import capacity is greater than 0, add row with to_node, from_node, import_capacity
                if row['import_capacity'] > 0:
                    to_from_capacity = row['import_capacity']
                    rows.append({
                        'grid' :                        row['grid'],
                        'from_node' :                   row['to_node']+'_'+row['grid'],
                        'to_node' :                     row['from_node']+'_'+row['grid'],
                        'transferCap' :                 to_from_capacity,  
                        'availability' :                1,                        
                        'variableTransCost' :           row['vomcost'] if 'vomcost' in row.index else 0,
                        'transferLoss' :                row['losses'] if 'losses' in row.index else 0,
                        'rampLimit' :                   (rampLimit / to_from_capacity * from_to_capacity ) if to_from_capacity > 0 else 0
                    })

        # create p_gnn
        p_gnn = self.create_fake_multiIndex(dimensions, param_gnn, rows)
        
        # Sort by grid, from_node, to_node in a case-insensitive manner.
        p_gnn.sort_values(by=['grid', 'from_node', 'to_node'], 
                            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
                            inplace=True)

        return p_gnn  


# ------------------------------------------------------
# Functions to create unittype based input tables: 
# unitUnittype, flowUnit, p_unit, effLevelGroupUnit
# ------------------------------------------------------

    def create_unitUnittype(self, p_gnu_io_flat):
        # Get unique unit names from the 'unit' column of p_gnu
        unique_units = p_gnu_io_flat['unit'].unique()
        # Create the DataFrame using a list comprehension, taking the last part as the unittype
        unitUnittype = pd.DataFrame(
            [{'unit': unit, 'unittype': unit.split('_')[-1]} for unit in unique_units],
            columns=['unit', 'unittype']
        )
        return unitUnittype
    
    def create_flowUnit(self, df_techs, unitUnittype):
        # Filter rows in df_techs that have a non-null 'flow' value
        flowtechs = df_techs[df_techs['flow'].notnull()].copy()
        # Merge unitUnittype with flowtechs based on matching 'unittype' and 'unittype'
        merged = pd.merge(unitUnittype, flowtechs, left_on='unittype', right_on='unittype', how='inner')
        # Create the final DataFrame with only 'flow' and 'unit' columns
        flowUnit = merged[['flow', 'unit']]
        return flowUnit
    
    def create_p_unit(self, df_techs, unitUnittype):
        """
        Creates a new DataFrame p_unit with specified dimension and parameter columns
        """      
        # dimension and parameter columns
        dimensions = ['unit']
        param_unit = ['isSource', 'isSink', 'availability', 
                      'eff00', 'eff01', 'op00', 'op01', 
                      'startColdAfterXhours', 'startWarmAfterXhours',
                      'minOperationHours', 'minShutdownHours']
        # List to collect the new rows 
        rows = []

        def fetch_from_tech_row(tech_row, param, def_value=0):
            # Reads value from tech_row in case insensitive manner, 
            # accepts default value and uses 0 if not given
            # handles errors by using value 0
            try:
                value = tech_row[param.lower()]  if param.lower() in tech_row.index else def_value
            except:
                value = 0
            return value

        # Iterate over each row in unitUnittype
        for _, u_row in unitUnittype.iterrows():
            # Retrieve the matching row from df_techs
            tech_row = df_techs[df_techs['unittype'] == u_row['unittype']].iloc[0]

            minShutdownHours = fetch_from_tech_row(tech_row, 'minShutdownHours')
            # Build the param_unit dictionary using values from the matching row
            row_data = {
                'unit'              : u_row['unit'],
                'isSource'          : fetch_from_tech_row(tech_row, 'isSource'),
                'isSink'            : fetch_from_tech_row(tech_row, 'isSink'),
                'availability'      : fetch_from_tech_row(tech_row, 'availability', 1),
                'eff00'             : fetch_from_tech_row(tech_row, 'eff00', 1),
                'eff01'             : fetch_from_tech_row(tech_row, 'eff01', 1),
                'op00'              : fetch_from_tech_row(tech_row, 'op00'),
                'op01'              : fetch_from_tech_row(tech_row, 'op01', 1),
                'minOperationHours' : fetch_from_tech_row(tech_row, 'minOperationHours'),
                'minShutdownHours'  : minShutdownHours,
                'startColdAfterXhours' : max(minShutdownHours, fetch_from_tech_row(tech_row, 'startColdAfterXhours')),
                'startWarmAfterXhours' : fetch_from_tech_row(tech_row, 'startWarmAfterXhours')
            }

            # Append the unit and its parameter dictionary to the list
            rows.append(row_data)

        # create p_unit
        p_unit = self.create_fake_multiIndex(dimensions, param_unit, rows)
        
        # Sort by grid, from_node, to_node in a case-insensitive manner.
        p_unit.sort_values(by=['unit'], 
                            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
                            inplace=True)

        return p_unit
    
    def create_effLevelGroupUnit(self, df_techs, unitUnittype):
        # List to accumulate new rows
        rows = []

        # Iterate over each row in unitUnittype
        for _, u_row in unitUnittype.iterrows():
            unit = u_row['unit']
            unittype = u_row['unittype']

            # Retrieve the matching row from df_techs where 'unittype' equals the unittype value
            tech_matches = df_techs[df_techs['unittype'] == unittype]
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

    def create_p_gn(self, p_gnn_flat, p_gnu_io_flat, df_fuels, df_demands, df_storages, **kwargs):
        """
        Creates a new DataFrame p_gn with specified dimension and parameter columns

          Collects (grid, node) pairs from df_storages, p_gnn, and p_gnu
          classifies each gn based on a range of tests
            - if node has any fuel data in df_fuels -> price node
            - if node is not price node -> balance node
            - classifying node as a storage nodes if it passes any of the following three tests
                1. if gn has upwardLimit, downwardLimit, or reference in df_storages -> storage node
                2. if node is in ts_storage_limits
                3. if 'upperLimitCapacityRatio' is defined to any (grid, node) -> storage node
                4. if gn is in p_gnu with both 'input' and 'output' roles and grid does not appear in df_demands
        """      
        # dimension and parameter columns
        dimensions = ['grid', 'node']
        param_gn = ['usePrice', 'nodeBalance', 'energyStoredPerUnitOfState']
        # List to collect the new rows 
        rows = []

        # Filter kwargs {varname: var} where varname starts with 'ts_storage_limits'
        ts_storage_limits = {key: value for key, value in kwargs.items() if key.startswith("ts_storage_limits")}
        # Initialize an empty set for storage nodes
        ts_storage_nodes = set()   
        # Iterate through the filtered DataFrames
        for key, df in ts_storage_limits.items():
            # Verify that the DataFrame has the required columns
            if 'node' in df.columns and 'param_gnBoundaryTypes' in df.columns:
                # Add all nodes to the set
                ts_storage_nodes.update(df['node'].unique())

        # Get ts_domains from secondary_results (kwargs), pick grid_node and convert to DataFrame
        if 'ts_domains' in kwargs:
            ts_domains = kwargs['ts_domains']
            if 'grid_node' in ts_domains:
                ts_grid_node = pd.DataFrame(ts_domains['grid_node'], columns=['grid', 'node'])
        if ts_grid_node is None:
            ts_grid_node = pd.DataFrame()

        # Extract gn pairs from df_storages, p_gnn, and p_gnu
        pairs_df_storages = df_storages[['grid', 'node']]
        pairs_from = p_gnn_flat[['grid', 'from_node']].rename(columns={'from_node': 'node'})
        pairs_to = p_gnn_flat[['grid', 'to_node']].rename(columns={'to_node': 'node'})
        pairs_gnu = p_gnu_io_flat[['grid', 'node']]
        # Concatenate and drop duplicates
        unique_gn_pairs = pd.concat([ts_grid_node, pairs_df_storages, pairs_from, pairs_to, pairs_gnu]).drop_duplicates()

        # Grids in df_demands for quick membership test
        demand_grids = df_demands['grid'].unique()

        # Process each (grid, node) pair:
        for idx, row in unique_gn_pairs.iterrows():
            grid = row['grid']
            node = row['node']

            # Determine usePrice: if any fuel record for this grid has price > 0.
            fuels_for_grid = df_fuels[df_fuels['grid'] == grid]
            if not fuels_for_grid.empty and (fuels_for_grid['price'] > 0).any():
                usePrice = True
            else:
                usePrice = False
            # nodeBalance is simply the opposite of usePrice.
            nodeBalance = not usePrice

            # Initialize storage status to no storage
            is_storage = 0

            # check if node is a storage node based on multiple criteria:
            # 1. if gn has upwardLimit, downwardLimit, or reference in df_storages -> storage node
            cols_to_check = ['upwardLimit', 'downwardLimit', 'reference']
            existing_cols = [col for col in cols_to_check if col.lower() in df_storages.columns]
          
            # Only check further if at least one column exists
            if existing_cols:
                node_storage_data = df_storages[(df_storages['grid'] == grid) & (df_storages['node'] == node)]
                # Check each existing column: if any value is greater than 0, mark as storage
                for col in existing_cols:
                    if (node_storage_data[col.lower()] > 0).any():
                        is_storage = 1
                        break

            # 2. if node is in ts_storage_limits
            if not is_storage and node in ts_storage_nodes:
                is_storage = 1

            # 3. if 'upperLimitCapacityRatio' is defined to any (grid, node) -> storage node
            subset_p_gnu_io = p_gnu_io_flat[(p_gnu_io_flat['grid'] == grid) & (p_gnu_io_flat['node'] == node)]
            if not is_storage and not subset_p_gnu_io.empty:
                is_storage = ((subset_p_gnu_io['upperLimitCapacityRatio'].notnull()) 
                              & (subset_p_gnu_io['upperLimitCapacityRatio'] != 0)
                              ).any()

            # 4. if gn is in p_gnu with both 'input' and 'output' roles and grid does not appear in df_demands
            io_roles = set(subset_p_gnu_io['input_output'])
            has_both_io = ('input' in io_roles) and ('output' in io_roles)
            grid_not_in_demands = grid not in demand_grids
            if not is_storage and (has_both_io and grid_not_in_demands):
                is_storage = 1

            # Determine storage nodes based on previous tests:
            if is_storage: energyStoredPerUnitOfState = 1
            else: energyStoredPerUnitOfState = 0

            # Build the data row for (grid,node)
            row_dict = {
                'grid' :                        grid,
                'node' :                        node,
                'usePrice' :                    usePrice,
                'nodeBalance' :                 nodeBalance,
                'energyStoredPerUnitOfState' :  energyStoredPerUnitOfState
            }
            rows.append(row_dict)

        # create p_gn
        p_gn = self.create_fake_multiIndex(dimensions, param_gn, rows)
        
        # Sort by grid, from_node, to_node in a case-insensitive manner.
        p_gn.sort_values(by=['grid', 'node'], 
                            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
                            inplace=True)
        return p_gn

    def create_p_gnBoundaryPropertiesForStates(self, p_gn_flat, df_storages, **kwargs):
        """
        Creates a new DataFrame p_gnBoundaryPropertiesForStates 


        """   
        dimensions = ['grid', 'node', 'param_gnBoundaryTypes']
        param_gnBoundaryTypes = ['upwardLimit', 'downwardLimit', 
                                 'reference', 'balancePenalty', 'selfDischargeLoss', 'maxSpill',
                                 'downwardSlack01']
        param_gnBoundaryProperties = ['useConstant', 'constant', 'useTimeSeries', 'slackCost']
        # List to collect the new rows 
        rows = []   

        # Filter kwargs {varname: var} where varname starts with 'ts_storage_limits'
        ts_storage_limits = {key: value for key, value in kwargs.items() if key.startswith("ts_storage_limits")}

        # Initialize an empty dictionary for {(node, param_gnBoundaryTypes): average_value}
        ts_node_boundaryTypes = {}

        # Iterate through the filtered DataFrames
        for key, df in ts_storage_limits.items():
            # Verify that the DataFrame has the required columns
            if all(col in df.columns for col in ['node', 'param_gnBoundaryTypes', 'average_value']):
                # Extract (node, param_gnBoundaryTypes, average_value) tuples and add to dictionary
                for _, row in df[['node', 'param_gnBoundaryTypes', 'average_value']].iterrows():
                    node_boundaryType = (row['node'], row['param_gnBoundaryTypes'])
                    ts_node_boundaryTypes[node_boundaryType] = row['average_value']

        # Loop through each row in p_gn that has useBalance equal to 1
        for _, gn_row in p_gn_flat.iterrows():
            if gn_row.get('nodeBalance', 0) == 1:
                grid = gn_row['grid']
                node = gn_row['node']

                # Find the corresponding row in df_storages where both grid and node match
                mask = (df_storages['grid'] == grid) & (df_storages['node'] == node)
                if mask.any():
                    storage_row = df_storages[mask].iloc[0]
                else:
                    storage_row = None

                # Loop through each boundary type and add a row if a value exists
                for p_type in param_gnBoundaryTypes:

                    # looking if storage_row has constant value for this p_type
                    if storage_row is not None:
                        value = storage_row.get(p_type.lower(), None)
                    else:
                        value = None

                    # if (node, p_type) in ts_node_boundaryTypes, flag to use timeseries
                    if (node, p_type) in ts_node_boundaryTypes:          
                        row_dict = {
                            'grid':                     grid,
                            'node':                     node,
                            'param_gnBoundaryTypes':    p_type,
                            'useTimeSeries':            1,
                        }
                        rows.append(row_dict)  
                        # if timeseries format downwardLimit, activate downwardSlack01
                        if p_type == 'downwardLimit':
                            row_dict = {
                                'grid':                     grid,
                                'node':                     node,
                                'param_gnBoundaryTypes':    'downwardSlack01',
                                'useConstant':              1,
                                'constant':                 round(ts_node_boundaryTypes[(node, p_type)]/1000, 0),
                                'slackCost':                300
                            }
                            rows.append(row_dict)      

                    # writing a row for p_type if it has data (value different to zero)
                    elif value is not None:
                        if value != 0 and pd.notna(value):
                            row_dict = {
                                'grid':                     grid,
                                'node':                     node,
                                'param_gnBoundaryTypes':    p_type,
                                'useConstant':              1,
                                'constant':                 value
                            }
                            rows.append(row_dict)
                
            # Additional check for storage nodes
            if gn_row.get('energyStoredPerUnitOfState', 0) == 1:
                grid = gn_row['grid']
                node = gn_row['node']
                # Add a row for 'downwardLimit' with constant 'Eps' 
                # if there is no such row for the current grid and node
                if not any((r['grid'] == grid and 
                            r['node'] == node and 
                            r['param_gnBoundaryTypes'] == 'downwardLimit') for r in rows):
                    row_dict = {
                        'grid':                  grid,
                        'node':                  node,
                        'param_gnBoundaryTypes': 'downwardLimit',
                        'useConstant':           1,
                        'constant':              'Eps'
                    }
                    rows.append(row_dict)


        # create p_gnBoundaryPropertiesForStates
        p_gnBoundaryPropertiesForStates = self.create_fake_multiIndex(dimensions, param_gnBoundaryProperties, rows)
        
        # Sort by grid, from_node, to_node in a case-insensitive manner.
        p_gnBoundaryPropertiesForStates.sort_values(by=['grid', 'node'], 
                            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
                            inplace=True)

        return p_gnBoundaryPropertiesForStates
        
    def create_ts_priceChange(self, p_gn_flat, df_fuels):
        # Identify the price column in df_fuels (case-insensitive)
        price_col = next((col for col in df_fuels.columns if col.lower() == 'price'), None)
        if price_col is None:
            raise ValueError("No 'price' column found in df_fuels (case-insensitive)")

        rows = []
        # Loop through each row in p_gn using the columns: grid, node, and usePrice
        for _, row in p_gn_flat.iterrows():
            grid_value = row['grid']
            node_value = row['node']

            # Retrieve the node_price from df_fuels where the grid value matches.
            matching_rows = df_fuels[df_fuels['grid'] == grid_value]
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


# !!!!!!!!!!!!!
# convert p_userconstraint to ts_gnu(mingen)
# !!!!!!!!!!!!!

    def create_p_userConstraint(self, p_gnu_io_flat, **kwargs):
    
        # Filter kwargs {varname: var} where varname starts with 'mingen'
        mingen_vars = {key: value for key, value in kwargs.items() if key.startswith("mingen")}

        # Assuming each mingen_vars value is a list, use list comprehension to flatten them
        mingen_nodes = [item for sublist in mingen_vars.values() for item in sublist]

        # creating empty p_userconstraint df
        p_userConstraint = pd.DataFrame(columns=['group', '1st dimension', '2nd dimension', '3rd dimension', '4th dimension', 'parameter', 'value'])

        for node in mingen_nodes:
            # Filter rows in p_gnu where 'node' equals current node and 'input_output' is 'input'
            row_gnu = p_gnu_io_flat[(p_gnu_io_flat['node'] == node) & (p_gnu_io_flat['input_output'] == 'input')]
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
# Function to create compile domains 
# ------------------------------------------------------

    def compile_domain(self, dfs, domain, **kwargs):
        """
        Compile unique domain values from multiple DataFrames and optional time series domains.
        
        This function extracts values from a specified column across multiple DataFrames,
        and optionally combines them with domain values from time series data.
        
        Parameters:
        -----------
        dfs : list of pandas.DataFrame
            List of DataFrames from which to extract domain values.
        domain : str
            The column name representing the domain to compile.
        **kwargs : dict
            Additional keyword arguments:
            - secondary_results : dict
                - potentially including 'ts_domains'. If present, 'ts_domains' should 
                  be a dictionary mapping domain names to arrays of values.
        """
        # Get ts_domains from secondary_results (kwargs)
        all_ts_domains = []
        if 'ts_domains' in kwargs:
            ts_domains = kwargs['ts_domains']
            if domain in ts_domains:
                all_ts_domains = ts_domains[domain].tolist()

        # Initialize an empty list to collect domain values
        all_domains = []

        # Iterate over each DataFrame in the list
        for df in dfs:
            if domain not in df.columns:
                print(f"Warning: '{domain}' is not in DataFrame columns.")
            else:
                # Extend the list with values from the specified domain column
                all_domains.extend(df[domain].dropna().tolist())

        # Get unique domain values from both all_domains and all_ts_domains
        all_combined_domains = all_domains + all_ts_domains
        unique_domains = pd.unique(pd.Series(all_combined_domains))

        # Create a new DataFrame where each unique domain has a corresponding 'yes'
        compiled_df = pd.DataFrame({domain: unique_domains})

        # Sort the DataFrame alphabetically by the domain column
        compiled_df = compiled_df.sort_values(by=domain).reset_index(drop=True)
        
        return compiled_df


# ------------------------------------------------------
# Function used to fine tune excel file after writing it
# ------------------------------------------------------

    def add_index_sheet(self, input_folder, output_file):
        """
        Adds Index sheet to the excel
            * loads preconstructed 'indexSheet.xlsx'
            * picks rows where Symbol is in the sheet names
        """
        # Construct full path to the index sheet file
        index_path = os.path.join(input_folder, 'indexSheet.xlsx')

        # Read the index sheet file (assuming the first row contains headers)
        df_index = pd.read_excel(index_path, header=0)

        # Load the output Excel workbook which already has multiple sheets
        wb = load_workbook(output_file)
        existing_sheet_names = wb.sheetnames

        # Filter rows: keep only rows where the 'Symbol' exists among the workbook's sheet names
        df_filtered = df_index[df_index['Symbol'].isin(existing_sheet_names)]

        # Create a new sheet named 'index'
        new_sheet = wb.create_sheet(title='index')

        # Write header row (row 1)
        for col_num, header in enumerate(df_index.columns, start=1):
            new_sheet.cell(row=1, column=col_num, value=header)

        # Write the filtered data starting from row 2
        for row_num, row in enumerate(df_filtered.itertuples(index=False, name=None), start=2):
            for col_num, value in enumerate(row, start=1):
                new_sheet.cell(row=row_num, column=col_num, value=value)

        # Move the 'index' sheet to the first position in the workbook
        wb._sheets.insert(0, wb._sheets.pop(wb._sheets.index(new_sheet)))

        # Save the updated workbook back to the output file
        wb.save(output_file)

    def adjust_excel(self, output_file):
        """
        For each sheet in the Excel file
            * Adjust each column's width.
            * Skip remaining processing if sheet has only 1 row.
            * If A2 is empty, iterate non-empty cells in row 2:
                    - Rotate matching cell in row 1 if the length of the cell is more than 6 letters.
                    - Centre align columns
                    - set the column width to 6 
            * Freeze top row
            * Create and apply table formatting
            * Add explanatory texts after (right from) the generated table in case of "fake MultiIndex"

        Note: Empty A2 means the sheet has "fake MultiIndex" used as a compromize between excel and Backbone
        """
        wb = load_workbook(output_file)

        for ws in wb.worksheets:
            # Adjust each column's width
            for col_cells in ws.columns:
                max_length = 0
                col_letter = get_column_letter(col_cells[0].column)
                for cell in col_cells:
                    if cell.value is not None:
                        max_length = max(max_length, len(str(cell.value)))
                ws.column_dimensions[col_letter].width = max_length + 6  # add extra padding

            # Skip remaining processing if sheet has only 1 row
            if ws.max_row == 1:
                continue

            # If A2 is empty, the sheet has "fake MultiIndex" used as a compromize between excel and Backbone
            if ws["A2"].value is None:
                # Iterate cells in row 2 if cells are not empty
                for cell in ws[2]:
                    if cell.value is not None:
                        # Rotate matching cell in row 1 if the length of the cell is more than 6 letters.
                        if len(str(cell.value)) > 5:
                            ws.cell(row=1, column=cell.col_idx).alignment = Alignment(textRotation=90)

                        # Centre align entire columns that have text in row 2 
                        col_index = cell.column
                        # Iterate over all cells in this column
                        for row_cells in ws.iter_rows(min_col=col_index, max_col=col_index):
                            for c in row_cells:
                                # Preserve any existing text rotation
                                current_rotation = c.alignment.textRotation if c.alignment else 0
                                c.alignment = Alignment(horizontal='center', textRotation=current_rotation)
                        col_letter = get_column_letter(cell.column)

                        # set the column width to 6 
                        ws.column_dimensions[col_letter].width = 6       

            # Freeze the top row
            ws.freeze_panes = "A2"

            # Derive table name from sheet name: remove any non-word characters and append _table.
            table_name = re.sub(r'\W+', '_', ws.title) + "_table"
            # Apply Excel table formatting
            last_col_letter = get_column_letter(ws.max_column)
            table_ref = f"A1:{last_col_letter}{ws.max_row}"
            table = Table(displayName=table_name, ref=table_ref)
            style = TableStyleInfo(name="TableStyleMedium9",
                                   showFirstColumn=False,
                                   showLastColumn=False,
                                   showRowStripes=True,
                                   showColumnStripes=False)
            table.tableStyleInfo = style
            table.headerRowCount = 1
            ws.add_table(table)


            # If A2 is empty, the sheet has "fake MultiIndex" used as a compromize between excel and Backbone
            if ws["A2"].value is None:
                # Add explanatory texts after (right from) the generated table
                n = ws.max_column + 2
                ws.cell(row=1, column=n, value='The first row labels are for excel Table headers.')
                ws.cell(row=2, column=n, value='The Second row labels are for GDXXRW converting excel to GDX.')


        # save the adjusted file
        wb.save(output_file)


# ------------------------------------------------------
# Utility functions
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
   
    def create_fake_multiIndex(self, dim_cols, param_cols, rows):
        """
        Creates a fake MultiIndex by 
            * creating column names from dimension columns and parameter columms
            * adding the first row where dimension columns have value "" and parameter columns have value as column name
            * adding row values (the real data) after the first row

        """
        # Define the final columns titles and orders.
        final_cols = dim_cols.copy()
        final_cols.extend(param_cols)

        # Create an empty DataFrame with the desired columns.
        df_output = pd.DataFrame(columns=final_cols)

        # Create the first row:
        # For each column in dimensions: use an empty string ""
        # For each column in param_gnu: use the corresponding param_gnu value.
        first_row = [""] * len(dim_cols) + list(param_cols)
        df_output.loc[len(df_output)] = first_row

        # Create a DataFrame for the remaining rows.
        if rows:  # ensure rows is not empty
            rows_df = pd.DataFrame(rows, columns=final_cols)
            # Concatenate the header row with the rest of the rows.
            df_output = pd.concat([df_output, rows_df], ignore_index=True)
        
        # Replace NaN values with 0 and return
        df_output = df_output.fillna(value=0)
        return df_output

    def drop_row0(self, df):
        # Create a copy of the original DataFrame
        df_flat = df.copy()

        # Drop the first row by using its index position 0
        df_flat = df_flat.drop(df_flat.index[0]).reset_index(drop=True)

        return df_flat

    def remove_rows(self, df, key_col, values_to_exclude):
        """
        Remove rows from df where the value in key_col matches any value in df_remove

        Parameters:
        df (pandas.DataFrame): The DataFrame to remove rows from
        key_col (str): The column name to use for matching
        values_to_exclude (pandas.DataFrame or list): DataFrame containing values to match for removal,
                                             or a list of strings to match directly

        Returns:
        pandas.DataFrame: DataFrame with matching rows removed

        Raises:
        ValueError: If key_col is missing in DataFrame or if values_to_exclude is of unsupported type
        """
        # Check if key_col exists in main DataFrame
        if key_col not in df.columns:
            raise ValueError(f"Column '{key_col}' not found in the main DataFrame")

        # Build list of unique values based on the type of values_to_exclude
        if isinstance(values_to_exclude, pd.DataFrame):
            # Handle DataFrame case
            if key_col not in values_to_exclude.columns:
                raise ValueError(f"Column '{key_col}' not found in the removal DataFrame")
            unique_vals = values_to_exclude[key_col].unique()
        elif isinstance(values_to_exclude, list):
            # Handle list case
            unique_vals = values_to_exclude
        else:
            raise ValueError("values_to_exclude must be either a pandas DataFrame or a list")

        # Remove rows from df where key_col value is in unique_vals
        filtered_df = df[~df[key_col].isin(unique_vals)]

        return filtered_df

    def remove_units_by_excluding_grids(self, p_gnu_io, exclude_grids):
        """
        Remove units from p_gnu_io DataFrame based on excluded grids.

        Parameters:
        p_gnu_io (pandas.DataFrame): DataFrame containing grid, node, unit, and input_output columns
        exclude_grids (list): List of grid names to exclude

        Returns:
        pandas.DataFrame: Filtered DataFrame with excluded grids and affected units removed
        """
        import pandas as pd

        # Get unique units and their input/output counts before filtering
        units = pd.DataFrame({'unit': p_gnu_io['unit'].unique()})

        # Calculate original input counts for each unit
        input_counts = p_gnu_io[p_gnu_io['input_output'] == 'input']['unit'].value_counts().reset_index()
        input_counts.columns = ['unit', 'inputCount_orig']
        units = units.merge(input_counts, on='unit', how='left')
        units['inputCount_orig'] = units['inputCount_orig'].fillna(0).astype(int)

        # Calculate original output counts for each unit
        output_counts = p_gnu_io[p_gnu_io['input_output'] == 'output']['unit'].value_counts().reset_index()
        output_counts.columns = ['unit', 'outputCount_orig']
        units = units.merge(output_counts, on='unit', how='left')
        units['outputCount_orig'] = units['outputCount_orig'].fillna(0).astype(int)

        # Remove rows with excluded grids
        filtered_gnu_io = self.remove_rows(p_gnu_io, 'grid', exclude_grids)

        # Calculate updated input counts after grid removal
        input_counts_upd = filtered_gnu_io[filtered_gnu_io['input_output'] == 'input']['unit'].value_counts().reset_index()
        input_counts_upd.columns = ['unit', 'inputCount_upd']
        units = units.merge(input_counts_upd, on='unit', how='left')
        units['inputCount_upd'] = units['inputCount_upd'].fillna(0).astype(int)

        # Calculate updated output counts after grid removal
        output_counts_upd = filtered_gnu_io[filtered_gnu_io['input_output'] == 'output']['unit'].value_counts().reset_index()
        output_counts_upd.columns = ['unit', 'outputCount_upd']
        units = units.merge(output_counts_upd, on='unit', how='left')
        units['outputCount_upd'] = units['outputCount_upd'].fillna(0).astype(int)

        # Identify units to remove - those that lost all inputs or all outputs
        units_to_remove = []

        for _, row in units.iterrows():
            # Check if unit had inputs before but lost all of them
            if row['inputCount_upd'] == 0 and row['inputCount_orig'] >= 1:
                units_to_remove.append(row['unit'])
            # Check if unit had outputs before but lost all of them
            elif row['outputCount_upd'] == 0 and row['outputCount_orig'] >= 1:
                units_to_remove.append(row['unit'])

        # Remove units that lost all inputs or all outputs
        result = self.remove_rows(filtered_gnu_io, 'unit', units_to_remove)

        return result


# ------------------------------------------------------
# Main entry point for the script
# ------------------------------------------------------

    def run(self):

        print(f"\n------ Building input excel --------------------------------------------------------------- ")

        # p_gnu_io
        p_gnu_io = self.create_p_gnu_io(self.df_techs, self.df_units)
        p_gnu_io = self.remove_rows(p_gnu_io, 'unit', self.df_remove_units)
        p_gnu_io = self.remove_units_by_excluding_grids(p_gnu_io, self.exclude_grids)
        p_gnu_io_flat = self.drop_row0(p_gnu_io)

        # p_gnn
        p_gnn = self.create_p_gnn(self.df_transfers, p_gnu_io_flat, self.exclude_grids)
        p_gnn = self.remove_rows(p_gnn, 'grid', self.exclude_grids)
        p_gnn_flat = self.drop_row0(p_gnn)

        # p_gn
        p_gn = self.create_p_gn(p_gnn_flat, p_gnu_io_flat, self.df_fuels, self.df_demands, self.df_storages, **self.secondary_results)
        p_gn = self.remove_rows(p_gn, 'grid', self.exclude_grids)
        p_gn_flat = self.drop_row0(p_gn)

        # unittype based input tables
        unitUnittype = self.create_unitUnittype(p_gnu_io_flat)
        flowUnit = self.create_flowUnit(self.df_techs, unitUnittype)
        p_unit = self.create_p_unit(self.df_techs, unitUnittype)
        effLevelGroupUnit = self.create_effLevelGroupUnit(self.df_techs, unitUnittype)

        # node based input tables
        p_gnBoundaryPropertiesForStates = self.create_p_gnBoundaryPropertiesForStates(p_gn_flat, self.df_storages, **self.secondary_results)
        ts_priceChange = self.create_ts_priceChange(p_gn_flat, self.df_fuels)
        p_userconstraint = self.create_p_userConstraint(p_gnu_io_flat, **self.secondary_results)

        # emission based input tables


        # Compile domains
        grid = self.compile_domain([p_gnu_io_flat, p_gnn_flat, p_gn_flat], 'grid', **self.secondary_results)
        node = self.compile_domain([p_gnu_io_flat, p_gn_flat], 'node', **self.secondary_results)  # cannot use p_gnn as it has domains from_node, to_node
        flow = self.compile_domain([flowUnit], 'flow', **self.secondary_results)
        unit = self.compile_domain([unitUnittype], 'unit')
        unittype = self.compile_domain([unitUnittype], 'unittype')
        group = self.compile_domain([p_userconstraint], 'group', **self.secondary_results)
        restype = pd.DataFrame()
        emission = pd.DataFrame()

        # scenario tags to an excel sheet
        scen_tags_df = pd.DataFrame([self.scen_tags])

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
            # scenario tags
            scen_tags_df.to_excel(writer, sheet_name='add_scen_tags', index=False)  

            # main input tables
            p_gnn.to_excel(writer, sheet_name='p_gnn', index=False)
            p_gnu_io.to_excel(writer, sheet_name='p_gnu_io', index=False)
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

            # domains
            grid.to_excel(writer, sheet_name='grid', index=False)       
            node.to_excel(writer, sheet_name='node', index=False)   
            unit.to_excel(writer, sheet_name='unit', index=False)   
            unittype.to_excel(writer, sheet_name='unittype', index=False)    
            group.to_excel(writer, sheet_name='group', index=False)              
            flow.to_excel(writer, sheet_name='flow', index=False)                                   
            restype.to_excel(writer, sheet_name='restype', index=False)                                   
            emission.to_excel(writer, sheet_name='emission', index=False)                                   


        # Apply the adjustments on the Excel file
        self.add_index_sheet(self.input_folder, merged_output_file)
        self.adjust_excel(merged_output_file)

        print(f"Input excel for Backbone written to '{merged_output_file}'")
