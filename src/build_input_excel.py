import os
import sys
import pandas as pd
from pathlib import Path
from src.utils import log_status
from src.excel_exchange import add_index_sheet, adjust_excel, check_if_bb_excel_open
from src.pipeline.bb_excel_context import BBExcelBuildContext



class BuildInputExcel:
    def __init__(self, context: BBExcelBuildContext) -> None:
        self.context = context
        
        # From sys args
        self.input_folder = context.input_folder

        # Currently looped run parameters
        self.output_folder = context.output_folder
        self.scen_tags = context.scen_tags

        # Parameters from config file
        self.config = context.config
        self.country_codes = self.config.get("country_codes", [])
        self.exclude_grids = self.config.get("exclude_grids", [])
        self.exclude_nodes = self.config.get("exclude_nodes", [])

        # From InputDataPipeline
        self.df_transferdata = context.df_transferdata
        self.df_unittypedata = context.df_unittypedata
        self.df_unitdata = context.df_unitdata
        self.df_remove_units = context.df_remove_units
        self.df_storagedata = context.df_storagedata
        self.df_fueldata = context.df_fueldata
        self.df_emissiondata = context.df_emissiondata
        self.df_demanddata = context.df_demanddata

        # From TimeseriesPipeline
        self.secondary_results = context.secondary_results
        self.ts_domains = context.ts_domains
        self.ts_domain_pairs = context.ts_domain_pairs

        # Filter secondary_results {varname: var} where varname starts with 'ts_storage_limits'
        self.ts_storage_limits = {key: value for key, value in context.secondary_results.items() if key.startswith("ts_storage_limits")}
        # Filter secondary_results {varname: var} where varname starts with 'mingen'
        mingen_vars = {key: value for key, value in context.secondary_results.items() if key.startswith("mingen_nodes")}
        # flatten (and guard against None)
        self.mingen_nodes = [
            item
            for sublist in mingen_vars.values() or [] 
            if isinstance(sublist, list)
            for item in sublist
        ]     

        # initiate empty log 
        self.builder_logs = []

        # Define the merged output file
        self.output_file = os.path.join(self.output_folder, 'inputData.xlsx')

        # Initiate a flag for successful code excecution
        self.bb_excel_succesfully_built = False



# ------------------------------------------------------
# Functions create and modify p_gnu_io 
# ------------------------------------------------------

    def create_p_gnu_io(self, df_unittypedata, df_unitdata):
        """
        Creates a DataFrame representing generator input/output connections with parameters.

        This method processes unit data and their type specifications to build a relationship
        table between generation units and grid nodes, with associated parameters.

        Parameters:
        -----------
        df_unittypedata : DataFrame
            Contains technical specifications for different generator types (indexed by generator_id)
            Must include grid connection points (grid_input1, grid_output1, etc.)

        df_unitdata : DataFrame
            Contains specific unit instances with capacities and country locations
            Must include 'country', 'generator_id', and 'unit' columns

        Returns:
        --------
        DataFrame
            A multi-index DataFrame with dimensions (grid, node, unit, input_output)
            and parameter columns (capacity, conversionCoeff, vomCosts, etc.)

        Process:
        --------
        1. For each unit in df_unitdata:
           - Match with its technical specifications from df_unittypedata
           - For each defined input/output connection:
             - Build node names using country and grid information
             - Calculate parameter values, prioritizing unit-specific values over type defaults
           - Skip connections with undefined generator_id

        2. Structure the resulting data with a two-level column index:
           - First level: dimensions and parameters
           - Second level: parameter names within each category
        """
        # dimension and parameter columns
        dimensions = ['grid', 'node', 'unit', 'input_output']
        # Define param_gnu as a dictionary {param: def_value}
        param_gnu = {
            'isActive': 1,
            'capacity': 0,
            'conversionCoeff': 1,
            'useInitialGeneration': 0,
            'initialGeneration': 0,
            'maxRampUp': 0,
            'maxRampDown': 0,
            'rampUpCost': 0,
            'rampDownCost': 0,
            'upperLimitCapacityRatio': 0,
            'unitSize': 0,
            'invCosts': 0,
            'annuityFactor': 0,
            'invEnergyCost': 0,
            'fomCosts': 0,
            'vomCosts': 0,
            'inertia': 0,
            'unitSizeMVA': 0,
            'availabilityCapacityMargin': 0,
            'startCostCold': 0,
            'startCostWarm': 0,
            'startCostHot': 0,
            'startFuelConsCold': 0,
            'startFuelConsWarm': 0,
            'startFuelConsHot': 0,
            'shutdownCost': 0,
            'delay': 0,
            'cb': 0,
            'cv': 0
        }

        # List to collect the new rows 
        rows = []

        def is_empty(val):
            """Return True if val is 0, empty string, None, or NaN."""
            return val == 0 or val == "" or val is None or pd.isnull(val)
        
        def get_param_value(param, put, cap_row, tech_row, def_value):
            """
            Determine parameter value with fallback logic.
        
            Prioritizes:
            1. Connection-specific value from unit data (e.g., vomCosts_input2)
            2. Non-specific value from unit data (for output1 only)
            3. Connection-specific value from technology data
            4. Non-specific value from technology data (for output1 only)
            5. def_value as the last resort if both sources are empty.
            """
            # First try unit-specific value (with connection suffix or default)
            primary = (
                (cap_row[f'{param.lower()}_{put}'] if f'{param.lower()}_{put}' in cap_row.index else 0) +
                (cap_row[param.lower()] if (param.lower() in cap_row.index and put == 'output1') else 0)
            )
            # If no unit-specific value, fall back to technology specifications
            secondary = (
                (tech_row[f'{param.lower()}_{put}'] if f'{param.lower()}_{put}' in tech_row.index else 0) +
                (tech_row[param.lower()] if (param.lower() in tech_row.index and put == 'output1') else 0)
            )
            # Return the first non-empty value; if both are empty, fall back to def_value.
            if not is_empty(primary):
                return primary
            elif not is_empty(secondary):
                return secondary
            else:
                return def_value

        # Keep a set of generator_ids already warned about
        warned_generator_ids = set()

        # Process each row in the capacities DataFrame.
        for _, cap_row in df_unitdata.iterrows():

            # Fetch country, generator_id, and unit name
            country = cap_row['country']
            generator_id = cap_row['generator_id']
            unit_name = cap_row['unit']

            # Find the technical specifications for this generator type
            try:
                tech_row = df_unittypedata.loc[df_unittypedata['generator_id'] == generator_id].iloc[0]
            # print warning and skip unit if unittype data not available
            except IndexError:
                # Only warn once per generator_id
                if generator_id not in warned_generator_ids:
                    log_status(
                        f"Generator_ID '{generator_id}' does not have a matching generator_id "
                        "in any of the unittypedata files, check spelling.",
                        self.builder_logs,
                        level="warn"
                    )
                    warned_generator_ids.add(generator_id)
                continue

            # Identify all defined input/output connections for this generator type
            available_puts = [put for put in ['input1', 'input2', 'input3', 'output1', 'output2', 'output3'] 
                              if f'grid_{put}' in tech_row.index]

            # Process each available input/output connection
            for put in available_puts:
                # Get the grid this connection links to
                grid = tech_row[f'grid_{put}'] if f'grid_{put}' in tech_row.index else None

                 # Only process valid grid connections
                if pd.notna(grid) and grid not in ['', '-']:
                    # Build node name: <country>_<grid>_<node suffix if defined>
                    node_name = f"{country}_{grid}"
                    node_suffix = cap_row[f'node_suffix_{put}'] if f'node_suffix_{put}' in cap_row.index else None
                    if pd.notna(node_suffix) and node_suffix not in ['', '-']:
                        node_name = f"{node_name}_{node_suffix}"
                            
                    # Create base row with essential connection information
                    base_row = {
                        'grid' : grid,
                        'node' : node_name,
                        'unit' : unit_name,
                        'input_output': 'input' if put.startswith('input') else 'output',
                    }

                    # Add all other parameters using the value resolution function.
                    # Pass in the corresponding default value from param_gnu.
                    additional_params = {
                        param: get_param_value(param, put, cap_row, tech_row, def_value)
                        for param, def_value in param_gnu.items()
                    }

                    # Combine base and additional parameters
                    row = {**base_row, **additional_params}
                    rows.append(row)

        # Define the final columns titles and orders.
        final_cols = dimensions.copy()
        final_cols.extend(param_gnu)

        # Create p_gnu_io, fill NaN, and remove empty columns except certain mandatory columns
        p_gnu_io = pd.DataFrame(rows, columns=final_cols)
        p_gnu_io = p_gnu_io.fillna(value=0)
        p_gnu_io = self.remove_empty_columns(p_gnu_io, cols_to_keep=['capacity'])

        # if dataframe has content
        if not p_gnu_io.empty:
            # create fake MultiIndex
            p_gnu_io = self.create_fake_MultiIndex(p_gnu_io, dimensions)

            # Sort by unit, input_output, node in a case-insensitive manner.
            p_gnu_io.sort_values(by=['unit', 'input_output', 'node'], 
                                key=lambda col: col.str.lower() if col.dtype == 'object' else col,
                                inplace=True)
        
        return p_gnu_io


    def remove_units_by_excluding_col_values(self, p_gnu_io, exclude_col, exclude_list):
        """
        Remove units from p_gnu_io DataFrame based on exclusion of values in a specified column.
        Any unit for which the unique count of values in exclude_col decreases after exclusion is removed entirely.

        Parameters:
            p_gnu_io (pandas.DataFrame): DataFrame with at least the columns 'unit' and one additional column, exclude_col.
            exclude_col (str): Name of the column where exclusion should be applied.
            exclude_list (list): List of string values to exclude from the column specified by exclude_col.

        Returns:
            pandas.DataFrame: Filtered DataFrame with rows for the affected units removed.
        """
        # Compute the original unique count per unit for the specified column.
        orig_counts = (
            p_gnu_io.groupby('unit')[exclude_col]
            .nunique()
            .reset_index()
            .rename(columns={exclude_col: 'valueCount_orig'})
        )

        # Remove rows where the specified column's value is in the exclude_list.
        filtered_data = self.remove_rows_by_values(p_gnu_io, exclude_col, exclude_list)

        # Compute the updated unique count per unit after exclusion.
        updated_counts = (
            filtered_data.groupby('unit')[exclude_col]
            .nunique()
            .reset_index()
            .rename(columns={exclude_col: 'valueCount_upd'})
        )

        # Merge the original and updated counts.
        counts = orig_counts.merge(updated_counts, on='unit', how='left')
        counts['valueCount_upd'] = counts['valueCount_upd'].fillna(0).astype(int)

        # Identify units that lost one or more values.
        units_to_remove = counts.loc[
            counts['valueCount_upd'] < counts['valueCount_orig'], 'unit'
        ].tolist()

        # Remove all rows for units that lost values.
        result = self.remove_rows_by_values(filtered_data, 'unit', units_to_remove)

        return result


    def fill_capacities(self, p_gnu_io, p_unit):
        """
        Fills missing capacity values of units with a set of rules. 
        Currently calculates missing input capacity, if 
            * unit has 1 input without capacity and 1 output with capacity
            * unit has 1 input without capacity, 2 outputs with capacity, and no 'cv' parameter
            * unit has 1 input with capacity and 1 output without capacity

        """
        # Skip processing if either of the source dataframes are empty
        if p_gnu_io.empty or p_unit.empty:
            return

        # Get a flat version of the source dataframes without the fake multi-index
        p_gnu_io_flat = self.drop_fake_MultiIndex(p_gnu_io)
        p_unit_flat = self.drop_fake_MultiIndex(p_unit)

        # Create a dictionary mapping each unit to its maximum efficiency.
        # Efficiency is taken as the maximum among columns that start with 'eff' (e.g., eff00, eff01, ...)
        p_unit_eff = {}
        # Identify efficiency columns (assumes they all start with 'eff')
        eff_columns = [col for col in p_unit_flat.columns if col.startswith('eff')]
        for _, unit_row in p_unit_flat.iterrows():
            # For each unit, compute its max efficiency (assuming one row per unit)
            p_unit_eff[unit_row['unit']] = unit_row[eff_columns].max()

        # --- 1 input without capacity and 1 output with capacity ---
        # For each row in p_gnu_io_flat with type 'input' and capacity 0, if unit
        # has exactly one input row, try to find a matching output row for the same unit
        # that has exactly one output row and capacity > 0.
        for idx, row in p_gnu_io_flat.iterrows():
            if row['input_output'] == 'input' and row['capacity'] == 0:
                # Check that for this unit, there is only one input row.
                group_input = p_gnu_io_flat[
                    (p_gnu_io_flat['unit'] == row['unit']) &
                    (p_gnu_io_flat['input_output'] == 'input')
                ]
                if len(group_input) != 1:
                    continue

                # Look for a corresponding output row for the same unit with positive capacity
                candidate_outputs = p_gnu_io_flat[
                    (p_gnu_io_flat['unit'] == row['unit']) &
                    (p_gnu_io_flat['input_output'] == 'output') &
                    (p_gnu_io_flat['capacity'] > 0)
                ]
                # Require that there is exactly one output row.
                if not candidate_outputs.empty:
                    if len(candidate_outputs) == 1:
                        matched_row = candidate_outputs.iloc[0]
                        output_capacity = matched_row['capacity']
                        # Get efficiency for the unit, defaulting to 0 if not found
                        efficiency = p_unit_eff.get(row['unit'], 0)
                        if efficiency > 0:
                            # Update the missing input capacity; divide output capacity by efficiency.
                            new_capacity = output_capacity / efficiency if efficiency != 0 else 0
                            p_gnu_io_flat.at[idx, 'capacity'] = new_capacity

        # --- 1 input without capacity and 2 outputs with capacity (no 'cv' parameter) ---
        # For each row with type 'input' and capacity 0, if unit has exactly one input row, 
        # check if there are exactly 2 output rows with capacity > 0
        # and neither has a 'cv' parameter.
        for idx, row in p_gnu_io_flat.iterrows():
            if row['input_output'] == 'input' and row['capacity'] == 0:
                # Check that for this unit, there is only one input row.
                group_input = p_gnu_io_flat[
                    (p_gnu_io_flat['unit'] == row['unit']) &
                    (p_gnu_io_flat['input_output'] == 'input')
                ]
                if len(group_input) != 1:
                    continue

                # Look for output rows for the same unit with positive capacity
                candidate_outputs = p_gnu_io_flat[
                    (p_gnu_io_flat['unit'] == row['unit']) &
                    (p_gnu_io_flat['input_output'] == 'output') &
                    (p_gnu_io_flat['capacity'] > 0)
                ]

                # Check if there are exactly 2 outputs and neither has 'cv' parameter
                if len(candidate_outputs) == 2:
                    # Check if 'cv' column exists and if so, ensure neither output has it
                    has_cv_column = 'cv' in p_gnu_io_flat.columns
                    if has_cv_column:
                        # Skip if either output has a non-zero/non-null cv value
                        cv_values = candidate_outputs['cv'].fillna(0)
                        if any(cv_values > 0):
                            continue
                        
                    # Calculate total output capacity
                    total_output_capacity = candidate_outputs['capacity'].sum()

                    # Get efficiency for the unit, defaulting to 0 if not found
                    efficiency = p_unit_eff.get(row['unit'], 0)
                    if efficiency > 0:
                        # Calculate required input capacity
                        new_capacity = total_output_capacity / efficiency
                        # Update the missing input capacity
                        p_gnu_io_flat.at[idx, 'capacity'] = new_capacity


        # --- 1 output without capacity and 1 input with capacity ---
        # Now, for each row with type 'output' and zero capacity,
        # check that its (grid, node, unit) group has exactly one output row.
        # Then, try to find a matching input row from a different (grid, node) with capacity > 0 and exactly one input row.
        for idx, row in p_gnu_io_flat.iterrows():
            if row['input_output'] == 'output' and row['capacity'] == 0:
                # Check that within the (grid, node, unit) group there is only one output row.
                group_output = p_gnu_io_flat[
                    (p_gnu_io_flat['unit'] == row['unit']) &
                    (p_gnu_io_flat['input_output'] == 'output')
                ]
                if len(group_output) != 1:
                    continue

                # Find a candidate input row for the same unit with positive capacity.
                candidate_inputs = p_gnu_io_flat[
                    (p_gnu_io_flat['unit'] == row['unit']) &
                    (p_gnu_io_flat['input_output'] == 'input') &
                    (p_gnu_io_flat['capacity'] > 0)
                ]
                # Ensure the candidate input row's (grid, node, unit) group contains only one row.
                if not candidate_inputs.empty:
                    if len(candidate_inputs) == 1:
                        matched_row = candidate_inputs.iloc[0]
                        input_capacity = matched_row['capacity']
                        # Get efficiency for the unit, defaulting to 0 if not found
                        efficiency = p_unit_eff.get(row['unit'], 0)
                        if efficiency > 0:
                            # Update the missing output capacity; multiply input capacity by efficiency.
                            new_capacity = input_capacity * efficiency
                            p_gnu_io_flat.at[idx, 'capacity'] = new_capacity

        # Fill any remaining NaN values in the DataFrame
        p_gnu_io_flat = p_gnu_io_flat.fillna(value=0)

        # Recreate the fake multi-index using the specified columns.
        p_gnu_io = self.create_fake_MultiIndex(p_gnu_io_flat, ['grid', 'node', 'unit', 'input_output'])
        return p_gnu_io


# ------------------------------------------------------
# Functions create p_gnn, p_gn
# ------------------------------------------------------

    def create_p_gnn(self, df_transferdata: pd.DataFrame) -> pd.DataFrame:
        """
        Build p_gnn by looping over 'dimensions' and 'param_gnn'.
        Special cases:
          - transferCap: from export/import capacity depending on direction
          - rampLimit: forward = row['ramplimit'] (default 0),
                       reverse = scaled by (export/import) if both > 0
        """
        if df_transferdata.empty:
            return pd.DataFrame()

        dimensions = ['grid', 'from_node', 'to_node']
        param_gnn = [
            'transferCap',
            'availability',
            'variableTransCost',
            'transferLoss',
            'rampLimit',
            'diffCoeff',
            'diffLosses',
            'transferCapInvLimit',
            'investMIP',
            'invCost',
            'annuityFactor',
        ]

        # Defaults; others default to 0
        defaults = {
            'availability': 1,
        }

        def get_or_default(row, param):
            """Return row[param_lower[out_param]] unless NaN/missing → defaults[out_param] or 0."""
            key = param.lower()
            if key in row:
                val = row.get(key)
                return val if pd.notna(val) else defaults.get(param, 0)
            return defaults.get(param, 0)

        rows = []
        for _, row in df_transferdata.iterrows():
            # domains
            grid      = row.get('grid')
            from_node = row.get('from_node')
            to_node   = row.get('to_node')
            # specific values
            export_cap = get_or_default(row, 'export_capacity')
            import_cap = get_or_default(row, 'import_capacity')
            ramp_base = get_or_default(row, 'ramplimit')

            if not (pd.notna(grid) and pd.notna(from_node) and pd.notna(to_node)):
                continue  # skip incomplete defs

            def build_row(dir_from, dir_to, cap_value, is_reverse: bool):
                out = {
                    'grid': grid,
                    'from_node': dir_from,
                    'to_node': dir_to,
                    'transferCap': cap_value,
                }
                for p in param_gnn:
                    if p == 'transferCap':
                        continue
                    if p == 'rampLimit':
                        if not is_reverse:
                            out[p] = ramp_base or 0
                        else:
                            out[p] = (ramp_base * (export_cap / import_cap)) if (import_cap and export_cap) else 0
                    else:
                        out[p] = get_or_default(row, p)
                return out

            # left-to-right (export)
            rows.append(build_row(from_node, to_node, export_cap, is_reverse=False))

            # right-to-left (import)
            rows.append(build_row(to_node, from_node, import_cap, is_reverse=True))

        # construct p_gnn
        final_cols = dimensions + param_gnn
        p_gnn = pd.DataFrame(rows, columns=final_cols).fillna(0)

        # sort by grid, from_node, to_node
        p_gnn.sort_values(
            by=['grid', 'from_node', 'to_node'],
            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
            inplace=True
        )

        # add fake multi-index
        p_gnn = self.create_fake_MultiIndex(p_gnn, dimensions)

        return p_gnn




    def create_p_gn(self, p_gnu_io_flat, df_fueldata, 
                    df_demanddata, df_storagedata, 
                    ts_storage_limits, ts_domain_pairs):
        """
        Creates a new DataFrame p_gn with specified dimension and parameter columns

          Collects (grid, node) pairs from df_storagedata, p_gnn, and p_gnu
          classifies each gn based on a range of tests
            - if node has any fuel data in df_fueldata -> price node
            - if node is not price node -> balance node
            - classifying node as a storage nodes if it passes any of the following three tests
                1. if gn has upwardLimit, downwardLimit, or reference in df_storagedata -> storage node
                2. if node is in ts_storage_limits
                3. if 'upperLimitCapacityRatio' is defined to any (grid, node) -> storage node
                4. if gn is in p_gnu with both 'input' and 'output' roles and grid does not appear in df_demanddata
        """      
        # dimension and parameter columns
        dimensions = ['grid', 'node']
        param_gn = ['isActive',
                    'nodeBalance', 
                    'usePrice',
                    'energyStoredPerUnitOfState',
                    'selfDischargeLoss', 
                    'boundStart',
                    'boundStartOfSamples',     
                    'boundStartAndEnd',        
                    'boundStartToEnd',         
                    'boundEnd',                
                    'boundEndOfSamples',       
                    'boundAll',                
                    'boundSumOverInterval',    
                    'capacityMargin',          
                    'storageValueUseTimeSeries', 
                    'influx'
                    ]
        
        # List to collect the new rows 
        rows = []



        # --- Collect grid-node pairs ---

        # Get grid_node from ts_domain_pairs and convert to DataFrame
        if 'grid_node' in ts_domain_pairs:
            ts_grid_node = pd.DataFrame(ts_domain_pairs['grid_node'], columns=['grid', 'node'])
        else:
            ts_grid_node = pd.DataFrame(columns=['grid', 'node'])

        # Extract gn pairs from df_storagedata
        if not df_storagedata.empty:
            pairs_df_storagedata = df_storagedata[['grid', 'node']]
        else:
            pairs_df_storagedata = pd.DataFrame(columns=['grid', 'node'])

        # Extract gn pairs from p_gnu_io_flat
        if not p_gnu_io_flat.empty:
            pairs_gnu = p_gnu_io_flat[['grid', 'node']]
        else:
            pairs_gnu = pd.DataFrame(columns=['grid', 'node'])

        # Concatenate and drop duplicates
        parts = [ts_grid_node, pairs_df_storagedata, pairs_gnu]
        parts = [p for p in parts if not p.empty]
        unique_gn_pairs = (
            pd.concat(parts, ignore_index=True).drop_duplicates(ignore_index=True)
            if parts else pd.DataFrame(columns=['grid', 'node'])
        )

        # --- Preprocess data for the loop ---

        # Grids in df_demanddata for quick membership test
        if not df_demanddata.empty:
            demand_grids = {str(g).lower() for g in df_demanddata['grid'].dropna().unique()}
        else:
            demand_grids = set()
        print(demand_grids)
        # Build a lower-case column map for df_storagedata to support case-insensitive lookup
        if not df_storagedata.empty:
            storage_colmap = {c.lower(): c for c in df_storagedata.columns}
            storage_cols_lower = set(storage_colmap.keys())
        else:
            storage_colmap = {}
            storage_cols_lower = set()

        # Initialize an empty set for storage nodes
        ts_storage_nodes = set()   
        # Iterate through the filtered DataFrames
        for key, df in ts_storage_limits.items():
            # Verify that the DataFrame has the required columns
            if 'node' in df.columns and 'param_gnBoundaryTypes' in df.columns:
                # Add all nodes to the set
                ts_storage_nodes.update(df['node'].unique())


        # --- Process each (grid, node) pair ---
        for _, row in unique_gn_pairs.iterrows():
            grid = row['grid']
            node = row['node']

            # Subset of storagedata for this (grid,node)
            if not df_storagedata.empty:
                node_storage_data = df_storagedata[(df_storagedata['grid'] == grid) & (df_storagedata['node'] == node)]
            else:
                node_storage_data = pd.DataFrame()

            # ---- Basic classifications ----
            isActive = node_storage_data['isActive'].iloc[0] if 'isActive' in node_storage_data.columns else None
            usePrice = node_storage_data['usePrice'].iloc[0] if 'usePrice' in node_storage_data.columns else None
            nodeBalance = node_storage_data['nodeBalance'].iloc[0] if 'nodeBalance' in node_storage_data.columns else None
            energyStoredPerUnitOfState = node_storage_data['energyStoredPerUnitOfState'].iloc[0] if 'energyStoredPerUnitOfState' in node_storage_data.columns else None

            if usePrice == 1 and nodeBalance == 1:
                log_status(f"Storage data for (grid, node):({grid}, {node}) has 'usePrice'=1 and 'nodeBalance'=1, check the data.", self.builder_logs, level="warn")    

            if usePrice == 1 and energyStoredPerUnitOfState == 1:
                log_status(f"Storage data for (grid, node):({grid}, {node}) has 'usePrice'=1 and 'energyStoredPerUnitOfState'=1, check the data.", self.builder_logs, level="warn")    

            # --- Check if price node ---

            # usePrice if any fuel record for this grid
            if not usePrice and not df_fueldata.empty:
                usePrice = not df_fueldata[df_fueldata['grid'] == grid].empty
                usePrice = 1 if usePrice else 0

            # --- Check if balance node ---

            # if demand node
            if nodeBalance == None and grid in demand_grids:
                nodeBalance = 1

            # if has any data in node_storage_data
            if nodeBalance == None and not node_storage_data.empty:
                nodeBalance = 1

            # --- Check if storage node ---

            # 1) upwardLimit / downwardLimit / reference present and > 0
            cols_to_check = ['upwardLimit', 'downwardLimit', 'reference']
            existing_lower = [c.lower() for c in cols_to_check if c.lower() in storage_cols_lower]
            if not usePrice and existing_lower and not node_storage_data.empty:
                for low in existing_lower:
                    real_col = storage_colmap[low]
                    if (node_storage_data[real_col].fillna(0) > 0).any():
                        energyStoredPerUnitOfState = 1
                        break

            # 2) node in ts_storage_limits
            if not usePrice and not energyStoredPerUnitOfState and node in ts_storage_nodes:
                energyStoredPerUnitOfState = 1

            # 3) upperLimitCapacityRatio defined (non-null & non-zero) in p_gnu_io_flat
            if not usePrice and not energyStoredPerUnitOfState and not p_gnu_io_flat.empty and 'upperLimitCapacityRatio' in p_gnu_io_flat.columns:
                subset_p_gnu_io = p_gnu_io_flat[(p_gnu_io_flat['grid'] == grid) & (p_gnu_io_flat['node'] == node)]
                if not subset_p_gnu_io.empty:
                    energyStoredPerUnitOfState = ((subset_p_gnu_io['upperLimitCapacityRatio'].notnull()) &
                                  (subset_p_gnu_io['upperLimitCapacityRatio'] != 0)).any()
                    energyStoredPerUnitOfState = 1 if energyStoredPerUnitOfState else 0
            else:
                subset_p_gnu_io = pd.DataFrame()


            # ---- Build the data row ----            
            # Derivatice flags
            if energyStoredPerUnitOfState and usePrice == None: usePrice = 0 
            if energyStoredPerUnitOfState and nodeBalance == None: nodeBalance = 1            


            # Build row with options used for every gn
            row_dict = {
                'grid':                       grid,
                'node':                       node,
                'isActive':                   isActive,
                'usePrice':                   usePrice,
                'nodeBalance':                nodeBalance,
                'energyStoredPerUnitOfState': energyStoredPerUnitOfState
            }

            # Add optional params if present in storagedata (case-insensitive)
            if not node_storage_data.empty:
                for key in (k for k in param_gn if k not in row_dict):
                    low = key.lower()
                    if low in storage_colmap:
                        real_col = storage_colmap[low]
                        val = node_storage_data[real_col].iloc[0]
                        if val is not None:
                            row_dict[key] = val
            rows.append(row_dict)

      
        # Build p_gn
        final_cols = dimensions + param_gn
        p_gn = pd.DataFrame(rows, columns=final_cols)
        p_gn = p_gn.fillna(value=0)
        #p_gn = p_gn.replace({True: 1, False: 0})
        p_gn = self.remove_empty_columns(p_gn, cols_to_keep=['usePrice', 'nodeBalance', 'energyStoredPerUnitOfState'])
        p_gn = self.create_fake_MultiIndex(p_gn, dimensions)


        # Sort by grid, node in a case-insensitive manner.
        p_gn.sort_values(by=['grid', 'node'], 
                            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
                            inplace=True)
        return p_gn


# ------------------------------------------------------
# Functions to create unittype based input tables: 
# unitUnittype, flowUnit, p_unit, effLevelGroupUnit
# ------------------------------------------------------

    def create_unitUnittype(self, p_gnu_io_flat):
        # skip processing and return empty dataframe if no input data
        if p_gnu_io_flat.empty:
            return pd.DataFrame()

        # Get unique unit names from the 'unit' column of p_gnu
        unique_units = p_gnu_io_flat['unit'].unique()
        # Create the DataFrame using a list comprehension, taking the last part as the unittype
        unitUnittype = pd.DataFrame(
            [{'unit': unit, 'unittype': unit.split('_')[-1]} for unit in unique_units],
            columns=['unit', 'unittype']
        )
        return unitUnittype
    

    def create_flowUnit(self, df_unittypedata, unitUnittype):
        # skip processing and return empty dataframe if no input data
        if unitUnittype.empty:
            return pd.DataFrame()

        # Filter rows in df_unittypedata that have a non-null 'flow' value
        flowtechs = df_unittypedata[df_unittypedata['flow'].notnull()].copy()
        # Merge unitUnittype with flowtechs based on matching 'unittype' and 'unittype'
        merged = pd.merge(unitUnittype, flowtechs, left_on='unittype', right_on='unittype', how='inner')
        # Create the final DataFrame with only 'flow' and 'unit' columns
        flowUnit = merged[['flow', 'unit']]
        return flowUnit
    

    def create_p_unit(self, df_unittypedata, unitUnittype, df_unitdata):
        """
        Creates a new DataFrame p_unit with specified dimension and parameter columns.
        Retrieves parameter values from df_unitdata first, then from df_unittypedata if not present
        or if the primary value is deemed "empty" (0, "", NaN, or None).
        Matching for unittype and unit is done in a case-insensitive manner.
        Raises an error if no matching tech_row is found.
        """
        # Dimension column names.
        dimensions = ['unit']

        # parameter_unit names and their default values.
        param_unit_defaults = {
            'isActive': 1,  
            'isSource': 0,
            'isSink': 0,
            'fixedFlow': 0,
            'availability': 1,
            'unitCount': 0,
            'useInitialOnlineStatus': 0,
            'initialOnlineStatus': 0,
            'startColdAfterXhours': 0,
            'startWarmAfterXhours': 0,
            'rampSpeedToMinLoad': 0,
            'rampSpeedFromMinLoad': 0,
            'minOperationHours': 0,
            'minShutdownHours': 0,
            'eff00': 1,
            'eff01': 1,
            'opFirstCross': 0,
            'op00': 0,
            'op01': 1,
            'useTimeseries': 0,
            'useTimeseriesAvailability': 0,
            'investMIP': 0,
            'maxUnitCount': 0,
            'minUnitCount': 0,
            'becomeAvailable': 0,
            'becomeUnavailable': 0
        }

        # List to collect new rows.
        rows = []

        def is_empty(val):
            """Return True if val is 0, empty string, None, or NaN."""
            return val == 0 or val == "" or val is None or pd.isnull(val)

        def get_value(unit_row, tech_row, param, def_value=0):
            """
            Retrieves a value by primarily checking unit_row and then tech_row if the primary
            value is considered "empty" (0, "", NaN, or None). Uses a case-insensitive lookup
            by checking for param.lower() in the Series index.
            """
            lower_param = param.lower()
            primary = 0
            secondary = 0
            # Try retrieving from unit_row (primary source)
            if lower_param in unit_row.index:
                primary = unit_row[lower_param]

            # If the primary value is empty, check the tech_row (secondary source)
            if is_empty(primary): 
                if lower_param in tech_row.index:
                    secondary = tech_row[lower_param]
                    
            if not is_empty(primary): 
                return primary
            elif not is_empty(secondary):  
                return secondary
            else:
                return def_value


        # Process each row in unitUnittype.
        for _, u_row in unitUnittype.iterrows():
            # Case-insensitive matching for unittype.
            tech_matches = df_unittypedata[df_unittypedata['unittype'].str.lower() == u_row['unittype'].lower()]
            if tech_matches.empty:
                raise ValueError(f"No matching tech row found for unittype: {u_row['unittype']}")
            tech_row = tech_matches.iloc[0]

            # Case-insensitive matching for unit.
            unit_matches = df_unitdata[df_unitdata['unit'].str.lower() == u_row['unit'].lower()]
            if unit_matches.empty:
                raise ValueError(f"No matching unit row found for unit: {u_row['unit']}")
            unit_row = unit_matches.iloc[0]

            # Pre-fetch minShutdownHours using its default value from the dictionary.
            min_shutdown = get_value(unit_row, tech_row, 'minShutdownHours', param_unit_defaults['minShutdownHours'])

            # Start building the row data with the unit column.
            row_data = {'unit': u_row['unit']}

            # Loop through the parameters defined in param_defaults.
            for param, default in param_unit_defaults.items():
                if param == 'minShutdownHours':
                    row_data[param] = min_shutdown
                elif param == 'startColdAfterXhours':
                    # For startColdAfterXhours, compute the maximum of min_shutdown and the fetched value.
                    # In Backbone, Units should have p_unit(unit, minShutdownHours) <= p_unit(unit, startWarmAfterXhours) <= p_unit(unit, startColdAfterXhours)
                    fetched_value = get_value(unit_row, tech_row, param, default)
                    row_data[param] = max(min_shutdown, fetched_value)
                else:
                    row_data[param] = get_value(unit_row, tech_row, param, default)

            rows.append(row_data)

        # Define final column order.
        final_cols = dimensions.copy()
        final_cols.extend(list(param_unit_defaults.keys()))

        # Construct DataFrame, fill NaN, apply a fake MultiIndex, and sort.
        p_unit = pd.DataFrame(rows, columns=final_cols)
        p_unit = p_unit.fillna(0)
        p_unit = self.create_fake_MultiIndex(p_unit, dimensions)

        # Sort p_unit by the 'unit' column in a case-insensitive manner.
        p_unit.sort_values(
            by=['unit'],
            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
            inplace=True
        )

        return p_unit
    

    def create_effLevelGroupUnit(self, df_unittypedata, unitUnittype):
        # List to accumulate new rows
        rows = []

        # Iterate over each row in unitUnittype
        for _, u_row in unitUnittype.iterrows():
            unit = u_row['unit']
            unittype = u_row['unittype']

            # Retrieve the matching row from df_unittypedata where 'unittype' equals the unittype value
            tech_matches = df_unittypedata[df_unittypedata['unittype'] == unittype]
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
# p_gnBoundaryPropertiesForStates, ts_priceChange, p_userconstraint
# ------------------------------------------------------

    def create_p_gnBoundaryPropertiesForStates(self, p_gn_flat, df_storagedata, ts_storage_limits):
        """
        Creates a DataFrame that defines boundary properties for nodes in an energy grid system.

        This function generates boundary properties for nodes that have balance requirements 
        or energy storage capabilities. It processes both constant values from the storage 
        dataframe and time series data provided in ts_storage_limits.

        Parameters:
        -----------
        p_gn_flat : DataFrame containing node configurations.
            Must include columns: 'grid', 'node', 'nodeBalance', 'energyStoredPerUnitOfState'

        df_storagedata : DataFrame containing storage input data specifications.
            Must include columns: 'grid', 'node', and may include boundary values like
            'upwardlimit', 'downwardlimit', etc.

        ts_storage_limits

        Returns:
        --------
        DataFrame
            A fake-multi-indexed DataFrame with dimensions ['grid', 'node', 'param_gnBoundaryTypes'] 
            and param_gnBoundaryProperties ['useConstant', 'constant', 'useTimeSeries', 'slackCost'].
            Fake multi-index is a compromise between many aspects, see create_fake_MultiIndex()
            Defines boundary constraints for nodes in the energy system.
        """ 
        # Define the dimensions and parameters for the fake-multi-indexed output DataFrame
        dimensions = ['grid', 'node', 'param_gnBoundaryTypes']

        # List of supported boundary types to process
        param_gnBoundaryTypes = ['upwardLimit', 'downwardLimit', 
                                 'reference', 'balancePenalty', 'selfDischargeLoss', 'maxSpill',
                                 'downwardSlack01']
        
        # Properties that will be assigned to each boundary type
        param_gnBoundaryProperties = ['useConstant', 'constant', 'useTimeSeries', 'slackCost']

        # Initialize an empty list to collect all rows for the output DataFrame 
        rows = []   

        # Create a lookup dictionary to quickly find average values for node-boundary type combinations
        # Format: {(node, param_gnBoundaryTypes): average_value}
        ts_node_boundaryTypes = {}

        # Process all time series data frames to populate the lookup dictionary
        for key, df in ts_storage_limits.items():
            # Verify the DataFrame has all required columns before processing
            if all(col in df.columns for col in ['node', 'param_gnBoundaryTypes', 'average_value']):
                # Extract (node, param_gnBoundaryTypes, average_value) tuples and add to dictionary
                for _, row in df[['node', 'param_gnBoundaryTypes', 'average_value']].iterrows():
                    node_boundaryType = (row['node'], row['param_gnBoundaryTypes'])
                    ts_node_boundaryTypes[node_boundaryType] = row['average_value']

        # Process each node in the system that requires balance constraints
        for _, gn_row in p_gn_flat.iterrows():
            # Only process nodes with balance requirements (nodeBalance = 1)
            if gn_row.get('nodeBalance', 0) == 1:
                grid = gn_row['grid']
                node = gn_row['node']

                # Find the corresponding row in df_storagedata where both grid and node match
                storage_row = None
                if not df_storagedata.empty:
                    mask = (df_storagedata['grid'] == grid) & (df_storagedata['node'] == node)
                    if mask.any():
                        storage_row = df_storagedata[mask].iloc[0]

                # Process each boundary type for this node
                for p_type in param_gnBoundaryTypes:

                    # Check if there's a constant value in storage configuration
                    value = storage_row.get(p_type.lower(), None) if storage_row is not None else None

                    # Time series data takes precedence over constant values
                    if (node, p_type) in ts_node_boundaryTypes:
                        # Create entry for time series-based boundary         
                        row_dict = {
                            'grid':                     grid,
                            'node':                     node,
                            'param_gnBoundaryTypes':    p_type,
                            'useTimeSeries':            1,
                        }
                        rows.append(row_dict)  

                        ## Special case: For downwardLimit with time series, create a slack variable
                        ## to allow minor violations with a smaller penalty
                        #if p_type == 'downwardLimit':
                        #    row_dict = {
                        #        'grid':                     grid,
                        #        'node':                     node,
                        #        'param_gnBoundaryTypes':    'downwardSlack01',
                        #        'useConstant':              1,
                        #        # Scale down the average value and round it
                        #        'constant':                 round(ts_node_boundaryTypes[(node, p_type)]/1000, 0),
                        #        'slackCost':                1000 # Fixed penalty cost for violations
                        #    }
                        #    rows.append(row_dict)      

                    # If no time series but we have a non-zero constant value, use it
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
                # Ensure all storage nodes have at least an 'Eps' downward limit
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


        # Define the final columns titles and orders.
        final_cols = dimensions.copy()
        final_cols.extend(param_gnBoundaryProperties)

        # Create p_gnBoundaryPropertiesForStates, fill NaN, and add fake MultiIndex
        p_gnBoundaryPropertiesForStates = pd.DataFrame(rows, columns=final_cols)
        p_gnBoundaryPropertiesForStates = p_gnBoundaryPropertiesForStates.fillna(value=0)
        p_gnBoundaryPropertiesForStates = self.create_fake_MultiIndex(p_gnBoundaryPropertiesForStates, dimensions)

        # Sort by grid, from_node, to_node in a case-insensitive manner.
        p_gnBoundaryPropertiesForStates.sort_values(by=['grid', 'node'], 
                            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
                            inplace=True)

        return p_gnBoundaryPropertiesForStates


    def add_storage_starts(self, p_gn, p_gnBoundaryPropertiesForStates, p_gnu_io_flat, ts_storage_limits):
        """
        Adds storage start values to nodes with energy storage capabilities. 
        Converts p_gn and p_gnBoundaryPropertiesForStates to flat versions by removing the fake multi-index.
        Adds p_gn('boundStart') and p_gnBoundaryPropertiesForStates('reference') for storage nodes
        Recreates the fake multi-index

        Parameters:
            p_gn: DataFrame with columns ['grid', 'node'] and possibly 'energyStoredPerUnitOfState'
            p_gnBoundaryPropertiesForStates: DataFrame with columns ['grid', 'node', 'param_gnBoundaryTypes', 'param_gnBoundaryProperties']
            ts_storage_limits

        Returns:
            tuple: (p_gn, p_gnBoundaryPropertiesForStates) with updated values
        """
        # Create a lookup dictionary to quickly find average values for node-boundary type combinations
        # Format: {(node, param_gnBoundaryTypes): average_value}
        ts_node_boundaryTypes = {}

        # Process all time series data frames to populate the lookup dictionary
        for key, df in ts_storage_limits.items():
            # Verify the DataFrame has all required columns before processing
            if all(col in df.columns for col in ['node', 'param_gnBoundaryTypes', 'average_value']):
                # Extract (node, param_gnBoundaryTypes, average_value) tuples and add to dictionary
                for _, row in df[['node', 'param_gnBoundaryTypes', 'average_value']].iterrows():
                    node_boundaryType = (row['node'], row['param_gnBoundaryTypes'])
                    ts_node_boundaryTypes[node_boundaryType] = row['average_value']

        # Get a flat versions without the fake multi-index
        p_gn_flat = self.drop_fake_MultiIndex(p_gn)
        p_gnBoundaryPropertiesForStates_flat = self.drop_fake_MultiIndex(p_gnBoundaryPropertiesForStates)

        # Identify storage nodes - those where energyStoredPerUnitOfState is 1 or True
        storage_gn = []
        if 'energyStoredPerUnitOfState' in p_gn_flat.columns:
            for _, row in p_gn_flat.iterrows():
                if row.get('energyStoredPerUnitOfState') == 1 or row.get('energyStoredPerUnitOfState') is True:
                    storage_gn.append((row['grid'], row['node']))

        # Add 'boundStart' column to p_gn, initializing with 0
        p_gn_flat['boundStart'] = 0

        # Process each storage node
        for grid, node in storage_gn:

            # 1) calculate start_value based on the timeseries upwardLimit
            start_value = ts_node_boundaryTypes.get((node, 'upwardLimit'), 0)

            # 2) check if there's a constant value in p_gnBoundaryPropertiesForStates_flat
            if start_value == 0:
                # Find rows that match our criteria
                mask = ((p_gnBoundaryPropertiesForStates_flat['grid'] == grid) & 
                       (p_gnBoundaryPropertiesForStates_flat['node'] == node) &
                       (p_gnBoundaryPropertiesForStates_flat['param_gnBoundaryTypes'] == 'upwardLimit'))

                # Check if any matching rows exist and have constant values > 0
                if any(mask) and 'constant' in p_gnBoundaryPropertiesForStates_flat.columns:
                    constant_values = p_gnBoundaryPropertiesForStates_flat.loc[mask, 'constant']
                    if not constant_values.empty and constant_values.iloc[0] > 0:
                        start_value = constant_values.iloc[0]

            # 3) calculate maximum storage based on p_gnu_io('upperLimitCapacityRatio')
            if start_value == 0:
                subset_p_gnu_io = p_gnu_io_flat[(p_gnu_io_flat['grid'] == grid) & 
                                                (p_gnu_io_flat['node'] == node) & 
                                                (p_gnu_io_flat['upperLimitCapacityRatio'] > 0)
                                                ]
                if not subset_p_gnu_io.empty:
                        # Use the subset dataframe and get the first row if there are multiple matches
                        capacity = subset_p_gnu_io['capacity'].iloc[0]
                        upper_limit = subset_p_gnu_io['upperLimitCapacityRatio'].iloc[0]
                        start_value = capacity * upper_limit                      

            # Only proceed with adding/updating p_gn and boundary properties if we have a valid start_value
            if start_value != 0:
                # Set boundStart to 1 for storage nodes
                p_gn_flat.loc[(p_gn_flat['grid'] == grid) & (p_gn_flat['node'] == node), 'boundStart'] = 1

                new_constant = round(start_value * 0.7, 0)
                # Create a mask to find the 'reference' row for this grid and node
                ref_mask = (
                    (p_gnBoundaryPropertiesForStates_flat['grid'] == grid) &
                    (p_gnBoundaryPropertiesForStates_flat['node'] == node) &
                    (p_gnBoundaryPropertiesForStates_flat['param_gnBoundaryTypes'] == 'reference')
                )

                if not p_gnBoundaryPropertiesForStates_flat.loc[ref_mask].empty:
                    # If row exists, update the 'constant' value (and useConstant as needed)
                    p_gnBoundaryPropertiesForStates_flat.loc[ref_mask, 'constant'] = new_constant
                    p_gnBoundaryPropertiesForStates_flat.loc[ref_mask, 'useConstant'] = 1
                else:
                    # Create new row since one does not exist yet.
                    new_row = {
                        'grid': grid,
                        'node': node,
                        'param_gnBoundaryTypes': 'reference',
                        'useConstant': 1,
                        'constant': new_constant
                    }
                    new_row_df = pd.DataFrame([new_row])
                    # Use pandas concat instead of append (which is deprecated in newer pandas versions)
                    p_gnBoundaryPropertiesForStates_flat = pd.concat(
                        [p_gnBoundaryPropertiesForStates_flat, new_row_df],
                        ignore_index=True
                    )

        # fill NaN
        p_gn_flat = p_gn_flat.fillna(value=0)
        p_gnBoundaryPropertiesForStates_flat = p_gnBoundaryPropertiesForStates_flat.fillna(value=0)

        
        # Sort p_gnBoundaryPropertiesForStates alphabetically by [grid, node] in a case-insensitive manner
        p_gnBoundaryPropertiesForStates_flat = p_gnBoundaryPropertiesForStates_flat.sort_values(
                                                    by=['grid', 'node', 'param_gnBoundaryTypes'], 
                                                    key=lambda x: x.str.lower()
                                                    ).reset_index(drop=True)

        # recreate fake multi-indexes
        p_gn = self.create_fake_MultiIndex(p_gn_flat, ['grid', 'node'])
        p_gnBoundaryPropertiesForStates = self.create_fake_MultiIndex(p_gnBoundaryPropertiesForStates_flat, ['grid', 'node', 'param_gnBoundaryTypes'])

        return (p_gn, p_gnBoundaryPropertiesForStates)


    def create_ts_priceChange(self, p_gn_flat, df_fueldata):
        # skip processing if no price nodes (empty p_gn) or no price data (empty df_fueldata)
        if p_gn_flat.empty or df_fueldata.empty:
            return pd.DataFrame()

        # Identify the price column in df_fueldata (case-insensitive), skip processing if not found
        price_col = next((col for col in df_fueldata.columns if col.lower() == 'price'), None)
        if price_col is None:
            return pd.DataFrame()

        rows = []
        # Loop through each row in p_gn using the columns: grid, node, and usePrice
        for _, row in p_gn_flat.iterrows():
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


    def create_p_userConstraint(self, p_gnu_io_flat, mingen_nodes):

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
                    -1                  # value
                ]
            # Add row for parameter "GT"
            p_userConstraint.loc[len(p_userConstraint)] = [
                group_UC, "-", "-", "-", "-", "GT", -1
            ]
            # Add row for parameter "ts_groupPolicy"
            p_userConstraint.loc[len(p_userConstraint)] = [
                group_UC, "userconstraintRHS", "-", "-", "-", "ts_groupPolicy", 1
            ]
            # Add custom penalty value for userconstraint equations
            p_userConstraint.loc[len(p_userConstraint)] = [
                group_UC, "-", "-", "-", "-", "penalty", 2000
            ]

        return p_userConstraint


# ------------------------------------------------------
# Functions to create emission based input tables: 
# p_nEmission, ts_emissionPriceChange, 
# ------------------------------------------------------

    def create_p_nEmission(self, p_gn_flat, df_fueldata):
        """
        Create p_nEmission['node', 'emission', 'value'] emission factors (tEmission / MWh) for each node.

        Parameters   
        p_gn_flat : pandas DataFrame with columns 'grid' and 'node'.
        df_fueldata : pandas DataFrame with column 'grid' and optional columns 'emission_XX' 
            where XX is emission name (e.g., CO2, CH4).
        """
        # skip processing if no fuel nodes (empty p_gn) or no emission data (empty df_fueldata)
        if p_gn_flat.empty or df_fueldata.empty:
            return pd.DataFrame()

        # Extract emission names from column names. Skip processing if no emissions.
        emission_cols = [col for col in df_fueldata.columns if col.startswith('emission_')]
        if emission_cols is None:
            return pd.DataFrame()
        else:
            emissions = [col.replace('emission_', '') for col in emission_cols]

        # Create grid_emission DataFrame with (grid, emission) combinations
        grid_emission_data = []
        for grid in df_fueldata['grid'].unique():
            grid_row = df_fueldata[df_fueldata['grid'] == grid].iloc[0]
            for col, emission in zip(emission_cols, emissions):
                if col in grid_row:
                    if grid_row[col] > 0:
                        grid_emission_data.append({
                            'grid': grid,
                            'emission': emission,
                            'value': grid_row[col]
                        })

        grid_emission = pd.DataFrame(grid_emission_data)

        # Filter grid_node to only include grids that are in grid_emission
        valid_grids = grid_emission['grid'].unique()
        grid_node = p_gn_flat[p_gn_flat['grid'].isin(valid_grids)][['grid', 'node']]

        # Create p_nEmission by joining grid_node with grid_emission
        p_nEmission_data = []
        for _, row in grid_node.iterrows():
            for emission in emissions:
                grid_emission_row = grid_emission[(grid_emission['grid'] == row['grid']) & 
                                                 (grid_emission['emission'] == emission)]
                if not grid_emission_row.empty:
                    if grid_emission_row.iloc[0]['value'] > 0:
                        p_nEmission_data.append({
                            'node': row['node'],
                            'emission': emission,
                            'value': grid_emission_row.iloc[0]['value']
                        })

        p_nEmission = pd.DataFrame(p_nEmission_data)

        return p_nEmission


    def create_ts_emissionPriceChange(self, df_emissiondata):
        """
        Create ts_emissionPriceChange ['emission', 'group', 't', 'value'] DataFrame
        
        Parameters: 
            df_emissiondata : pandas DataFrame with columns 'emission', 'group', and optional 'price'.
        """
        # skip processing if no emission data (empty df_emissiondata)
        if df_emissiondata.empty:
            return pd.DataFrame()
        
        # Extract emission_group pairs
        emission_group = df_emissiondata[['emission', 'group']].drop_duplicates()

        # Check if price column exists
        has_price = 'price' in df_emissiondata.columns

        # Create ts_emissionPriceChange DataFrame
        ts_emissionPriceChange_data = []
        for _, row in emission_group.iterrows():
            emission = row['emission']
            group = row['group']

            # Get price value if it exists
            price_value = 0
            if has_price:
                emission_row = df_emissiondata[(df_emissiondata['emission'] == emission) & 
                                           (df_emissiondata['group'] == group)]
                if not emission_row.empty and not pd.isna(emission_row.iloc[0]['price']):
                    price_value = emission_row.iloc[0]['price']

            ts_emissionPriceChange_data.append({
                'emission': emission,
                'group': group,
                't': 't000001',
                'value': price_value
            })

        ts_emissionPriceChange = pd.DataFrame(ts_emissionPriceChange_data)

        return ts_emissionPriceChange


    def create_gnGroup(self, p_nEmission, ts_emissionPriceChange, p_gnu_io_flat, unitUnittype, df_unittypedata, input_dfs=[]):
        """
        Creates a gnGroup['grid', 'node', 'group'] DataFrame based on emission groups and input DataFrames.

        Parameters:
        -----------
        p_nEmission : DataFrame with columns ['node', 'emission']
        ts_emissionPriceChange : DataFrame with columns ['emission', 'group']
        p_gnu_io_flat : DataFrame with columns ['grid', 'node', 'unit']
        unitUnittype : DataFrame with columns ['unit', 'unittype']
        df_unittypedata : DataFrame with column ['unittype'] and possibly columns ['emission_group1', 'emission_group2', ...]
        input_dfs : list of DataFrames, optional. Each with columns ['grid', 'node', 'group'].
        """

        # Initialize an empty list to store rows
        rows_list = []

        # Step 1: Process emissions data
        for _, node_emission in p_nEmission.iterrows():
            node = node_emission['node']
            emission = node_emission['emission']

            # Check if emission exists in ts_emissionPriceChange
            if not ts_emissionPriceChange.empty:
                matching_emission = ts_emissionPriceChange[ts_emissionPriceChange['emission'].str.lower() == emission.lower()] 
            else:
                matching_emission = pd.DataFrame()            

            # Skip the rest of step 1 if no matching emission is found
            if matching_emission.empty:
                continue

            # Get the group value from ts_emissionPriceChange
            group = matching_emission['group'].iloc[0]

            # Find matching grid_node_unit tuples
            matching_gnu = p_gnu_io_flat[p_gnu_io_flat['node'] == node]

            for _, grid_node_unit in matching_gnu.iterrows():
                grid = grid_node_unit['grid']
                unit = grid_node_unit['unit']

                # Find matching unit_unittype tuple
                matching_unit_type = unitUnittype[unitUnittype['unit'] == unit]

                if not matching_unit_type.empty:
                    unittype = matching_unit_type['unittype'].iloc[0]

                    # Find correct unittype_row
                    unittype_rows = df_unittypedata[df_unittypedata['unittype'] == unittype]

                    if not unittype_rows.empty:
                        unittype_row = unittype_rows.iloc[0]

                        # Find if emission exists in any emission_group column
                        emission_group_cols = [col for col in df_unittypedata.columns if col.startswith('emission_group')]

                        for col in emission_group_cols:
                            if col in unittype_row and unittype_row[col] == group:
                                # Create row and add to rows_list
                                row = {'grid': grid, 'node': node, 'group': group}
                                rows_list.append(row)

        # Step 2: Process input DataFrames
        for df in input_dfs:
            # Check if df has required columns
            if all(col in df.columns for col in ['grid', 'node', 'group']):
                for _, row in df.iterrows():
                    rows_list.append({
                        'grid': row['grid'],
                        'node': row['node'],
                        'group': row['group']
                    })

        # Step 3: create the final DataFrame and drop duplicates
        gnGroup = pd.DataFrame(rows_list)
        gnGroup = pd.DataFrame(rows_list).drop_duplicates()

        return gnGroup


# ------------------------------------------------------
# Function to compile domains 
# ------------------------------------------------------

    def compile_domain(self, dfs, domain):
        """   
        Compiles unique domain values from a specified column across multiple DataFrames.

        Note: case-insensitive 
        
        Parameters:
        -----------
        dfs : list of pandas.DataFrame
            List of DataFrames from which to extract domain values.
        domain : str
            The column name representing the domain to compile.
        """
        # Initialize an empty list to collect domain values
        all_domains = []

        # Iterate over each DataFrame in the list
        for df in dfs:
            if domain in df.columns:
                # Extend the list with values from the specified domain column
                all_domains.extend(df[domain].dropna().tolist())

        # return empty df, if no domains found
        if not all_domains:
            return pd.DataFrame()

        else:
            # Convert to lowercase for comparison, but keep original case for the first occurrence
            domain_mapping = {}
            for d in all_domains:
                if d is not None and isinstance(d, str):
                    lower_d = d.lower()
                    if lower_d not in domain_mapping:
                        domain_mapping[lower_d] = d

            # Use values from the mapping (first occurrence of each case-insensitive unique domain)
            unique_domains = list(domain_mapping.values())

            # Create a new DataFrame where each unique domain has a corresponding 'yes'
            compiled_df = pd.DataFrame({domain: unique_domains})

            # Sort the DataFrame alphabetically by the domain column in a case-insensitive manner
            compiled_df = compiled_df.sort_values(by=domain, key=lambda x: x.str.lower()).reset_index(drop=True)

            return compiled_df


# ------------------------------------------------------
# Utility functions
# ------------------------------------------------------


    def create_fake_MultiIndex(self, df, dimensions):
        """
        Creates a fake MultiIndex by:
        1. Taking an existing DataFrame with single-layer column names
        2. Creating a new first row with empty strings for dimensions and parameter names for parameter columns
        3. Shifting existing data down by one row

        Parameters:
        - df: pandas DataFrame with single-layered column names
        - dimensions: list of column names that are dimension columns

        Returns:
        - DataFrame with fake MultiIndex structure
        """
        # Identify parameter columns (those not in dimensions)
        all_columns = list(df.columns)

        # Create a new DataFrame with the same columns
        df_output = pd.DataFrame(columns=all_columns)

        # Create the first row with empty strings for dimension columns
        # and parameter names for parameter columns
        first_row = []
        for col in all_columns:
            if col in dimensions:
                first_row.append("")
            else:
                first_row.append(col)

        # Add the first row to the DataFrame
        df_output.loc[0] = first_row

        # Reset the index of the input DataFrame and add it to the output
        df_reset = df.reset_index(drop=True)
        df_reset.index = df_reset.index + 1  # Shift indices to start from 1

        # Concatenate the first row with the original data
        df_output = pd.concat([df_output, df_reset], axis=0)

        return df_output
    

    def drop_fake_MultiIndex(self, df):
        # Create a copy of the original DataFrame
        df_flat = df.copy()

        # Drop the first row by using its index position 0
        df_flat = df_flat.drop(df_flat.index[0]).reset_index(drop=True)

        # convert data types to restore numeric dtypes. 
        # Fake multi-index sometimes converts dtype to object.
        df_flat = df_flat.convert_dtypes()

        return df_flat


    def remove_rows_by_values(self, df, key_col, values_to_exclude, printWarnings=False):
        """
        Remove rows from df where the value in key_col matches any value in df_remove,
        but warn if any requested values_to_exclude aren’t actually in df[key_col].

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

        # warn if any values_to_exclude aren't in df[key_col]
        if printWarnings:
            warning_log = []
            existing = set(df[key_col].unique())
            missing = set(unique_vals) - existing
            if missing:
                log_status(
                    f"The following value(s) were requested for exclusion but not found in '{key_col}': "
                    f"{sorted(missing)}",
                    warning_log,
                    level="warn"
                )

        # Remove rows from df where key_col value is in unique_vals
        filtered_df = df[~df[key_col].isin(unique_vals)]

        if printWarnings:
            return filtered_df, warning_log
        else:
            return filtered_df


    def remove_empty_columns(self, df: pd.DataFrame, cols_to_keep=None, treat_nan_as_empty=True):
        from pandas.api.types import is_numeric_dtype, is_bool_dtype

        cols_to_keep = set(cols_to_keep or [])

        def col_is_empty(s: pd.Series) -> bool:
            # Booleans: usually don't drop just because all False; only NaNs count as empty
            if is_bool_dtype(s):
                return s.isna().all() if treat_nan_as_empty else False

            # Numeric (excluding bool): empty if all zeros (and optionally NaN→0)
            if is_numeric_dtype(s) and not is_bool_dtype(s):
                s_cmp = s.fillna(0) if treat_nan_as_empty else s
                return (s_cmp == 0).all()

            # Non-numeric: empty if all are "" (optionally allow NaN)
            if treat_nan_as_empty:
                na_mask = s.isna()
            else:
                na_mask = pd.Series(False, index=s.index)

            # Safe elementwise test; no vectorized == on arbitrary objects
            empty_str_mask = s.map(lambda v: isinstance(v, str) and v.strip() == "")
            return (na_mask | empty_str_mask).all()

        empty_cols_mask = df.apply(col_is_empty, axis=0)

        # Keep protected columns even if empty
        keep = list(cols_to_keep & set(df.columns))
        if keep:
            empty_cols_mask.loc[keep] = False

        return df.loc[:, ~empty_cols_mask]



# ------------------------------------------------------
# Main entry point for the script
# ------------------------------------------------------

    def run(self):

        # Check if the Excel file is already open before proceeding
        try: 
            check_if_bb_excel_open(self.output_file)
        except Exception as e:
            log_status(f"{e}", self.builder_logs, level="warn")
            return self.builder_logs, self.bb_excel_succesfully_built

        # Instantiate warning log
        warning_log = []

        # Create p_gnu_io
        if not self.df_unittypedata.empty and not self.df_unitdata.empty:
            p_gnu_io = self.create_p_gnu_io(self.df_unittypedata, self.df_unitdata)     
        else:
            log_status(f"Missing unit data or unittype data, skipping p_gnu_io and derivatives.'", self.builder_logs, level="info")
            p_gnu_io = pd.DataFrame() 
        if not p_gnu_io.empty:
            # Filter out certain units, grids, and nodes
            if not self.df_remove_units.empty:
                p_gnu_io, warning_log = self.remove_rows_by_values(p_gnu_io, 'unit', self.df_remove_units, printWarnings=True)
            if warning_log != []: 
                self.builder_logs.extend(warning_log)
            if self.exclude_grids:
                p_gnu_io = self.remove_units_by_excluding_col_values(p_gnu_io, 'grid', self.exclude_grids)
            if self.exclude_nodes:
                p_gnu_io = self.remove_units_by_excluding_col_values(p_gnu_io, 'node', self.exclude_nodes)
            # Create flat version for easier use in other functions
            p_gnu_io_flat = self.drop_fake_MultiIndex(p_gnu_io)
        else:
            p_gnu_io_flat = pd.DataFrame()

        # unittype based input tables
        unitUnittype = self.create_unitUnittype(p_gnu_io_flat)
        p_unit = self.create_p_unit(self.df_unittypedata, unitUnittype, self.df_unitdata)
        flowUnit = self.create_flowUnit(self.df_unittypedata, unitUnittype)
        effLevelGroupUnit = self.create_effLevelGroupUnit(self.df_unittypedata, unitUnittype)

        if not p_gnu_io.empty:
            # Calculate missing capacities from p_gnu_io
            p_gnu_io = self.fill_capacities(p_gnu_io, p_unit)
            p_gnu_io_flat = self.drop_fake_MultiIndex(p_gnu_io)

        # p_gnn
        if not self.df_transferdata.empty:
            p_gnn = self.create_p_gnn(self.df_transferdata)
        else:
            log_status(f"Missing transfer data, skipping p_gnn and derivatives.'", self.builder_logs, level="info")
            p_gnn = pd.DataFrame()
        if not p_gnn.empty:
            # Filter out certain grids and nodes
            if self.exclude_grids:
                p_gnn = self.remove_rows_by_values(p_gnn, 'grid', self.exclude_grids)
            if self.exclude_nodes:
                p_gnn = self.remove_rows_by_values(p_gnn, 'from_node', self.exclude_nodes)
                p_gnn = self.remove_rows_by_values(p_gnn, 'to_node', self.exclude_nodes)
            # Create flat version for easier use in other functions
            p_gnn_flat = self.drop_fake_MultiIndex(p_gnn)
        else:
            p_gnn_flat = pd.DataFrame()

        # p_gn
        p_gn = self.create_p_gn(p_gnu_io_flat, self.df_fueldata, self.df_demanddata, self.df_storagedata, self.ts_storage_limits, self.ts_domain_pairs)
        if not p_gn.empty:
            # Filter out certain grids and nodes        
            if self.exclude_grids:
                p_gn = self.remove_rows_by_values(p_gn, 'grid', self.exclude_grids)
            if self.exclude_nodes:
                p_gn = self.remove_rows_by_values(p_gn, 'node', self.exclude_nodes)
            # Create flat version for easier use in other functions
            p_gn_flat = self.drop_fake_MultiIndex(p_gn)
        else:
            p_gn_flat = pd.DataFrame()            

        # node based input tables
        p_gnBoundaryPropertiesForStates = self.create_p_gnBoundaryPropertiesForStates(p_gn_flat, self.df_storagedata, self.ts_storage_limits)
        ts_priceChange = self.create_ts_priceChange(p_gn_flat, self.df_fueldata)
        if not p_gnu_io.empty:
            p_userconstraint = self.create_p_userConstraint(p_gnu_io_flat, self.mingen_nodes)
        else:
            p_userconstraint = pd.DataFrame()

        # add storage start levels to p_gn and p_gnBoundaryPropertiesForStates
        (p_gn, p_gnBoundaryPropertiesForStates) = self.add_storage_starts(p_gn, p_gnBoundaryPropertiesForStates, p_gnu_io_flat, self.ts_storage_limits)
        p_gn_flat = self.drop_fake_MultiIndex(p_gn)


        # emission based input tables
        p_nEmission = self.create_p_nEmission(p_gn_flat, self.df_fueldata)
        ts_emissionPriceChange = self.create_ts_emissionPriceChange(self.df_emissiondata)

        # group sets
        gnGroup = self.create_gnGroup(p_nEmission, ts_emissionPriceChange, p_gnu_io_flat, unitUnittype, self.df_unittypedata)

        # Compile domains
        ts_grids = pd.DataFrame()
        if self.ts_domains is not None and 'grid' in self.ts_domains:
            ts_grids = pd.DataFrame(self.ts_domains['grid'], columns=['grid'])
        grid = self.compile_domain([p_gnu_io_flat, p_gnn_flat, p_gn_flat, ts_grids], 'grid')

        ts_nodes = pd.DataFrame()
        if self.ts_domains is not None and 'node' in self.ts_domains:
            ts_nodes = pd.DataFrame(self.ts_domains['node'], columns=['node'])        
        node = self.compile_domain([p_gnu_io_flat, p_gn_flat, ts_nodes], 'node')  # cannot refer to p_gnn as it has domains from_node, to_node

        ts_flows = pd.DataFrame()
        if self.ts_domains is not None and 'flow' in self.ts_domains:
                ts_flows = pd.DataFrame(self.ts_domains['flow'], columns=['flow'])  
        flow = self.compile_domain([flowUnit, ts_flows], 'flow')

        ts_groups = pd.DataFrame()
        if self.ts_domains is not None and 'group' in self.ts_domains:
            ts_groups = pd.DataFrame(self.ts_domains['group'], columns=['group'])  
        group = self.compile_domain([p_userconstraint, ts_emissionPriceChange, gnGroup, ts_groups], 'group')

        unit = self.compile_domain([unitUnittype], 'unit')
        unittype = self.compile_domain([unitUnittype], 'unittype')
        emission = self.compile_domain([ts_emissionPriceChange, p_nEmission], 'emission')
        restype = pd.DataFrame()

        # scenario tags to an excel sheet
        scen_tags_df = pd.DataFrame([self.scen_tags], columns=['scenario', 'year', 'alternative'])

        # Write DataFrames to different sheets of the merged Excel file
        with pd.ExcelWriter(self.output_file) as writer:
            # scenario tags
            scen_tags_df.to_excel(writer, sheet_name='add_scen_tags', index=False)  

            # node based input tables
            grid.to_excel(writer, sheet_name='grid', index=False)       
            node.to_excel(writer, sheet_name='node', index=False) 
            p_gn.to_excel(writer, sheet_name='p_gn', index=False) 
            p_gnBoundaryPropertiesForStates.to_excel(writer, sheet_name='p_gnBoundaryPropertiesForStates', index=False)        
            ts_priceChange.to_excel(writer, sheet_name='ts_priceChange', index=False)   

            # transfer input tables
            p_gnn.to_excel(writer, sheet_name='p_gnn', index=False)

            # unit input tables
            unit.to_excel(writer, sheet_name='unit', index=False)   
            unittype.to_excel(writer, sheet_name='unittype', index=False)             
            unitUnittype.to_excel(writer, sheet_name='unitUnittype', index=False)     
            flowUnit.to_excel(writer, sheet_name='flowUnit', index=False)            
            effLevelGroupUnit.to_excel(writer, sheet_name='effLevelGroupUnit', index=False)  
            p_gnu_io.to_excel(writer, sheet_name='p_gnu_io', index=False)
            p_unit.to_excel(writer, sheet_name='p_unit', index=False)       
            p_userconstraint.to_excel(writer, sheet_name='p_userconstraint', index=False)     

            # emission based input tables
            p_nEmission.to_excel(writer, sheet_name='p_nEmission', index=False)   
            ts_emissionPriceChange.to_excel(writer, sheet_name='ts_emissionPriceChange', index=False)  

            # group sets
            gnGroup.to_excel(writer, sheet_name='gnGroup', index=False) 

            # remaining domains
            group.to_excel(writer, sheet_name='group', index=False)              
            flow.to_excel(writer, sheet_name='flow', index=False)                                   
            emission.to_excel(writer, sheet_name='emission', index=False)                                   
            restype.to_excel(writer, sheet_name='restype', index=False)    

        # Apply the adjustments on the Excel file
        add_index_sheet(self.input_folder, self.output_file)
        adjust_excel(self.output_file)

        log_status(f"Input excel for Backbone written to '{self.output_file}'", self.builder_logs, level="info")
        self.bb_excel_succesfully_built = True

        return self.builder_logs, self.bb_excel_succesfully_built
