import os
import pandas as pd
import src.utils as utils
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

        # From InputDataPipeline
        # Global
        self.df_unittypedata = context.df_unittypedata
        self.df_fueldata = context.df_fueldata
        self.df_emissiondata = context.df_emissiondata
        # Country specific
        self.df_transferdata = context.df_transferdata
        self.df_unitdata = context.df_unitdata
        self.df_storagedata = context.df_storagedata
        self.df_demanddata = context.df_demanddata
        # Custom
        self.df_userconstraintdata = context.df_userconstraintdata

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
        Creates a DataFrame representing unit input/output connections with parameters.

        This method processes unit data and their type specifications to build a relationship
        table between grids, nodes and units, with associated parameters.

        Parameters:
        -----------
        df_unittypedata : DataFrame
            Contains technical specifications for different unittypes (indexed by generator_id)

        df_unitdata : DataFrame
            Contains specific unit instances with capacities and country locations
            Must include 'generator_id', and 'unit' columns
            Must include grids (grid_input1, grid_output1, etc.)
            Must include matching nodes (node_input1, node_output1, etc.)

        Returns:
        --------
        DataFrame
            A multi-index DataFrame with dimensions (grid, node, unit, input_output)
            and parameter columns (capacity, conversionCoeff, vomCosts, etc.)

        Process:
        --------
        For each unit in df_unitdata:
            - Match with its technical specifications from df_unittypedata
            - For each defined input/output connection:
              - build base row
              - Calculate parameter values, prioritizing unit-specific values over type defaults
            - Skip connections with undefined generator_id

        """
        # dimension and parameter columns
        dimensions = ['grid', 'node', 'unit', 'input_output']
        # Define param_gnu as a dictionary {param: def_value}
        param_gnu = [
            'isActive',
            'capacity',
            'conversionCoeff',
            'useInitialGeneration',
            'initialGeneration',
            'maxRampUp',
            'maxRampDown',
            'rampUpCost',
            'rampDownCost',
            'upperLimitCapacityRatio',
            'unitSize',
            'invCosts',
            'annuityFactor',
            'invEnergyCost',
            'fomCosts',
            'vomCosts',
            'inertia',
            'unitSizeMVA',
            'availabilityCapacityMargin',
            'startCostCold',
            'startCostWarm',
            'startCostHot',
            'startFuelConsCold',
            'startFuelConsWarm',
            'startFuelConsHot',
            'shutdownCost',
            'delay',
            'cb',
            'cv'
        ]

        # Non-zero default values. Zero assumed for all others. 
        defaults = {'isActive': 1,
                    'conversionCoeff': 1
                    }

        # List to collect the new rows 
        rows = []

        # Process each row in the capacities DataFrame.
        for _, cap_row in df_unitdata.iterrows():

            # Fetch generator_id and unit name
            generator_id = cap_row['generator_id']
            unit = cap_row['unit']

            # Find the technical specifications for this generator type
            tech_matches = df_unittypedata.loc[df_unittypedata['generator_id'] == generator_id]
            if tech_matches.empty:
                utils.log_status(f"Missing tech data for generator_id: '{generator_id}', unit: '{unit}', not writing the p_gnu_io data. "
                            "Check spelling and files.'", self.builder_logs, level="warn")
                continue
            tech_row = tech_matches.iloc[0]

            # Identify all defined input/output connections for this generator type (grid_input1, grid_output1, ...)
            put_candidates = ['input1', 'input2', 'input3', 'input4', 'input5',
                              'output1', 'output2', 'output3', 'output4', 'output5']
            available_puts = [put for put in put_candidates if f'grid_{put}' in tech_row.index]

            # Process each available input/output connection
            for put in available_puts:
                # Construct looped column names
                grid_col = f"grid_{put}"
                node_col = f"node_{put}"

                # skip if the columns do not exist in unitdata
                if grid_col not in cap_row:
                    utils.log_status(f"Missing grid {grid_col} from unit {unit}, not writing the data. "
                               "Check spelling and files.'", self.builder_logs, level="warn")
                    continue
                if node_col not in cap_row:
                    utils.log_status(f"Missing node {node_col} from unit {unit}, not writing the data. "
                               "Check spelling and files.'", self.builder_logs, level="warn")
                    continue

                # get values from unitdata
                grid = cap_row.get(grid_col)
                node = cap_row.get(node_col)

                # skip undefined / blank grids
                if pd.isna(grid) or grid in ("", "-"):
                    continue

                # Construct base components needed for every row 
                base_row = {
                    "grid": grid,
                    "node": node,
                    "unit": unit,
                    "input_output": "input" if put.startswith("input") else "output",
                }

                # Add all other parameters using the get_param_value(..)
                # Pass in default values for each param_gnu, use zero if default is not defined.
                additional_params = {
                    param: self.get_param_value(param, put, cap_row, tech_row, defaults.get(param, 0))
                    for param in param_gnu
                }

                # Append base and additional parameters
                rows.append({**base_row, **additional_params})

        # Construct p_gnu_io
        final_cols = dimensions + param_gnu
        p_gnu_io = pd.DataFrame(rows, columns=final_cols)
        p_gnu_io = utils.standardize_df_dtypes(p_gnu_io, fill_numeric_na=True)

        #  Remove empty columns except mandatory 'capacity' column
        p_gnu_io = self.remove_empty_columns(p_gnu_io, cols_to_keep=['capacity'])

        # Sort by unit, input_output, node in a case-insensitive manner.
        p_gnu_io.sort_values(by=['unit', 'input_output', 'node'], 
                            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
                            inplace=True)

        # create fake MultiIndex
        p_gnu_io = self.create_fake_MultiIndex(p_gnu_io, dimensions)

        return p_gnu_io


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
            return pd.DataFrame()

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
                candidate_outputs = utils.standardize_df_dtypes(candidate_outputs, fill_numeric_na=True)

                # Check if there are exactly 2 outputs and neither has 'cv' parameter
                if len(candidate_outputs) == 2:
                    # Check if 'cv' column exists and if so, ensure neither output has it
                    has_cv_column = 'cv' in p_gnu_io_flat.columns
                    if has_cv_column:
                        # Skip if either output has a non-zero/non-null cv value
                        cv_values = candidate_outputs['cv']
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

        # Standardize dtypes and Fill any remaining NaN 
        p_gnu_io_flat = utils.standardize_df_dtypes(p_gnu_io_flat, fill_numeric_na=True)

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
        p_gnn = pd.DataFrame(rows, columns=final_cols)
        p_gnn = utils.standardize_df_dtypes(p_gnn, fill_numeric_na=True)

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

        # Extract gn pairs from df_demanddata
        if not df_demanddata.empty:
            pairs_df_demanddata = df_demanddata[['grid', 'node']]
        else:
            pairs_df_demanddata = pd.DataFrame(columns=['grid', 'node'])

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
        parts = [ts_grid_node, pairs_df_demanddata, pairs_df_storagedata, pairs_gnu]
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
                node_storage_data = utils.standardize_df_dtypes(node_storage_data, fill_numeric_na=True)
            else:
                node_storage_data = pd.DataFrame()

            # ---- Basic classifications ----
            isActive = node_storage_data['isActive'].iloc[0] if 'isActive' in node_storage_data.columns else None
            usePrice = node_storage_data['usePrice'].iloc[0] if 'usePrice' in node_storage_data.columns else None
            nodeBalance = node_storage_data['nodeBalance'].iloc[0] if 'nodeBalance' in node_storage_data.columns else None
            energyStoredPerUnitOfState = node_storage_data['energyStoredPerUnitOfState'].iloc[0] if 'energyStoredPerUnitOfState' in node_storage_data.columns else None

            if usePrice == 1 and nodeBalance == 1:
                utils.log_status(f"Storage data for (grid, node):({grid}, {node}) has 'usePrice'=1 and 'nodeBalance'=1, check the data.", self.builder_logs, level="warn")    

            if usePrice == 1 and energyStoredPerUnitOfState == 1:
                utils.log_status(f"Storage data for (grid, node):({grid}, {node}) has 'usePrice'=1 and 'energyStoredPerUnitOfState'=1, check the data.", self.builder_logs, level="warn")    

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
                    if (node_storage_data[real_col] > 0).any():
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
        p_gn = utils.standardize_df_dtypes(p_gn, fill_numeric_na=True)
        p_gn = self.remove_empty_columns(p_gn, cols_to_keep=['usePrice', 'nodeBalance', 'energyStoredPerUnitOfState'])

        # Sort by grid, node in a case-insensitive manner.
        p_gn.sort_values(by=['grid', 'node'], 
                            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
                            inplace=True)

        # Create the fake multi-index
        p_gn = self.create_fake_MultiIndex(p_gn, dimensions)

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
        if 'flow' in df_unittypedata.columns:
            flowtechs = df_unittypedata[df_unittypedata['flow'].notnull()].copy()
            # Merge unitUnittype with flowtechs based on matching 'unittype' and 'unittype'
            merged = pd.merge(unitUnittype, flowtechs, left_on='unittype', right_on='unittype', how='inner')
            # Create the final DataFrame with only 'flow' and 'unit' columns
            flowUnit = merged[['flow', 'unit']]
            return flowUnit
        else:
            return pd.DataFrame(columns=['flow', 'unit'])


    def create_p_unit(
        self,
        df_unittypedata: pd.DataFrame,
        unitUnittype: pd.DataFrame,
        df_unitdata: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create the `p_unit` DataFrame for Backbone model input.

        This method constructs a parameter table for each unit defined in
        `unitUnittype`, combining information from both `df_unitdata`
        (unit-specific parameters) and `df_unittypedata` (unittype-level defaults).
        Parameter values are read in the following priority order:
            1. From df_unitdata (specific to the unit)
            2. From df_unittypedata (shared for all units of the same unittype)
            3. From the defaults dictionary (if missing or empty)

        The method handles case-insensitive matching for both `unit` and `unittype`
        names, fills missing numeric values with zeros, and enforces consistency
        between certain operational parameters.

        Specifically, the following constraint is applied:
            minShutdownHours <= startWarmAfterXhours <= startColdAfterXhours

        Parameters
        ----------
        df_unittypedata : pd.DataFrame
            DataFrame containing technology-level (unittype) default parameters.
            Must include column 'unittype'.
        unitUnittype : pd.DataFrame
            DataFrame linking individual units to their unittype.
            Must include columns ['unit', 'unittype'].
        df_unitdata : pd.DataFrame
            DataFrame containing unit-level parameter values.
            Must include column 'unit'.

        Returns
        -------
        pd.DataFrame
            Finalized `p_unit` DataFrame with:
            - One row per unit in `unitUnittype`
            - Columns:
                ['unit'] + parameter list
            - NaN values replaced with zeros
            - Sorted case-insensitively by 'unit'
            - MultiIndex applied via `create_fake_MultiIndex`

        Notes
        -----
        - If a `unit` or `unittype` is missing from the input DataFrames,
          a warning is logged, and that row is skipped.
        """
        # Dimension column names.
        dimensions = ['unit']

        # parameter_unit names
        param_unit = [
            'isActive',  
            'isSource',
            'isSink',
            'fixedFlow',
            'availability',
            'unitCount',
            'useInitialOnlineStatus',
            'initialOnlineStatus',
            'startColdAfterXhours',
            'startWarmAfterXhours',
            'rampSpeedToMinLoad',
            'rampSpeedFromMinLoad',
            'minOperationHours',
            'minShutdownHours',
            'eff00',
            'eff01',
            'opFirstCross',
            'op00',
            'op01',
            'useTimeseries',
            'useTimeseriesAvailability',
            'investMIP',
            'maxUnitCount',
            'minUnitCount',
            'becomeAvailable',
            'becomeUnavailable'
        ]
        # Default values. Otherwise assuming zero.
        defaults = {
            'isActive': 1,
            'availability': 1,
            'eff00': 1,
            'op00': 1
        }

        # List to collect new rows.
        rows = []

        # Process each row in unitUnittype.
        for _, u_row in unitUnittype.iterrows():
            unit = str(u_row['unit'])
            unittype = str(u_row['unittype'])

            # Case-insensitive matching for unittype.
            tech_matches = df_unittypedata[df_unittypedata['unittype'].str.lower() == unittype.lower()]
            if tech_matches.empty:
                utils.log_status(f"No matching tech row found for unittype: '{unittype}', unit: '{unit}'. "
                            "Not writing the p_unit data, check spelling and files.'", 
                            self.builder_logs, level="warn")
                continue            
            tech_row = tech_matches.iloc[0]

            # Case-insensitive matching for unit.
            unit_matches = df_unitdata[df_unitdata['unit'].str.lower() == unit.lower()]
            if unit_matches.empty:
                utils.log_status(f"No unit data found unit: '{unit}'. "
                            "Not writing the p_unit data, check spelling and files.'", 
                            self.builder_logs, level="warn")
                continue  
            unit_row = unit_matches.iloc[0]

            # Pre-fetch minShutdownHours using its default value from the dictionary.
            # using get_param_value for 'output1' to prioritize following order: unit data, unittype data, defaul 
            min_shutdown = self.get_param_value('minShutdownHours', 'output1', 
                                                unit_row, tech_row, 
                                                defaults.get('minShutdownHours', 0)
                                                )

            # Start building the row data with the unit column.
            row = {'unit': unit}

            # Loop through the parameters defined in param_unit.
            for param in param_unit:
                # For startColdAfterXhours, compute the maximum of min_shutdown and the fetched value.
                # In Backbone, Units must have minShutdownHours <= startWarmAfterXhours <= startColdAfterXhours
                if param == 'startColdAfterXhours':
                    startColdAfterXhours = self.get_param_value(param, 'output1', 
                                                                unit_row, tech_row, 
                                                                defaults.get(param, 0))
                    row[param] = max(min_shutdown, startColdAfterXhours)
                    continue

                # using get_param_value for 'output1' to prioritize following order: unit data, unittype data, defaul 
                row[param] = self.get_param_value(param, 'output1', 
                                                  unit_row, tech_row, 
                                                  defaults.get(param, 0))

            rows.append(row)

        # Build p_unit
        final_cols = dimensions + param_unit
        p_unit = pd.DataFrame(rows, columns=final_cols)
        p_unit = utils.standardize_df_dtypes(p_unit, fill_numeric_na=True)

        # Sort p_unit by the 'unit' column in a case-insensitive manner.
        p_unit.sort_values(
            by=['unit'],
            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
            inplace=True
        )

        # Apply fake MultiIndex on dimensions
        p_unit = self.create_fake_MultiIndex(p_unit, dimensions)

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
            if tech_matches.empty:
                utils.log_status(f"No matching tech row found for unittype: '{unittype}', unit: '{unit}', not writing the effLevelGroupUnit data. "
                            "Check spelling and files.'", self.builder_logs, level="warn")
                continue            
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
                                 'reference', 'balancePenalty', 
                                 'maxSpill', 'downwardSlack01']
        
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
                    elif not utils.is_val_empty(value, self.builder_logs):
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


        # Build p_gnBoundaryPropertiesForStates
        final_cols = dimensions + param_gnBoundaryProperties
        p_gnBoundaryPropertiesForStates = pd.DataFrame(rows, columns=final_cols)
        p_gnBoundaryPropertiesForStates = utils.standardize_df_dtypes(p_gnBoundaryPropertiesForStates, fill_numeric_na=True)

        # Sort by grid, from_node, to_node in a case-insensitive manner.
        p_gnBoundaryPropertiesForStates.sort_values(by=['grid', 'node'], 
                            key=lambda col: col.str.lower() if col.dtype == 'object' else col,
                            inplace=True)

        # Create fake multi-index
        p_gnBoundaryPropertiesForStates = self.create_fake_MultiIndex(p_gnBoundaryPropertiesForStates, dimensions)

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
        for _, df in ts_storage_limits.items():
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

        # Standardize dtypes, fill NA
        p_gn_flat = utils.standardize_df_dtypes(p_gn_flat, fill_numeric_na=True)
        p_gnBoundaryPropertiesForStates = utils.standardize_df_dtypes(p_gnBoundaryPropertiesForStates, fill_numeric_na=True)
        
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
        ts_priceChange = utils.standardize_df_dtypes(ts_priceChange, fill_numeric_na=True)
        
        return ts_priceChange


    def create_p_userConstraint(
        self,
        uc_data: pd.DataFrame,
        p_gnu_io_flat: pd.DataFrame,
        mingen_nodes: list[str],
        logs: list[str]
    ) -> pd.DataFrame:
        """
        Creates the parameter DataFrame `p_userConstraint` defining user constraints.

        The function combines:
          1. Predefined user-constraint data from `uc_data` (added directly as-is,
             with case-insensitive column handling).
          2. Dynamically generated user-constraint rules based on `mingen_nodes`,
             linking nodes to units in `p_gnu_io_flat`.

        Parameters
        ----------
        uc_data : pd.DataFrame
            User-defined constraints, possibly with column names in arbitrary case.
            Expected logical columns (case-insensitive):
            ['group', '1st dimension', '2nd dimension', '3rd dimension',
             '4th dimension', 'parameter', 'value'].
        p_gnu_io_flat : pd.DataFrame
            Flattened DataFrame mapping grids, nodes, and units with columns
            ['grid', 'node', 'unit', 'input_output'].
        mingen_nodes : list[str]
            List of node names for which minimum-generation user constraints
            should be created.
        logs:
            An accumulating log event list

        Returns
        -------
        pd.DataFrame
            Combined DataFrame of all user constraints (`p_userConstraint`).
        """
        expected_cols = [
            'group', '1st dimension', '2nd dimension',
            '3rd dimension', '4th dimension', 'parameter', 'value'
        ]

        frames: list[pd.DataFrame] = []

        # ---- Phase 1: add uc_data, case-insensitive column alignment ----
        if uc_data is not None and not uc_data.empty:
            # map lowercased column name -> canonical expected column
            col_map = {c.lower(): c for c in expected_cols}
            rename_dict = {orig: col_map[orig.lower()] for orig in uc_data.columns if orig.lower() in col_map}
            uc_data_renamed = uc_data.rename(columns=rename_dict)

            missing = [c for c in expected_cols if c not in uc_data_renamed.columns]
            if missing:
                utils.log_status(f"uc_data missing required columns (after case-insensitive matching): {missing}", 
                           logs, level="info")

            uc_data_aligned = uc_data_renamed[expected_cols]

            # keep only rows that aren't entirely NA to avoid dtype inference warnings later
            uc_data_aligned = uc_data_aligned.dropna(how="all")

            if not uc_data_aligned.empty:
                frames.append(uc_data_aligned)

        # ---- Phase 2: generate rows for mingen_nodes ----
        generated_rows = []
        for node in mingen_nodes:
            row_gnu = p_gnu_io_flat[
                (p_gnu_io_flat['node'] == node) &
                (p_gnu_io_flat['input_output'] == 'input')
            ]
            group_UC = f"UC_{node}"

            for _, r in row_gnu.iterrows():
                generated_rows.append({
                    'group': group_UC,
                    '1st dimension': r['grid'],
                    '2nd dimension': node,
                    '3rd dimension': r['unit'],
                    '4th dimension': "-",
                    'parameter': "v_gen",
                    'value': -1,
                })

            # group-level rows
            generated_rows += [
                {'group': group_UC, '1st dimension': "-", '2nd dimension': "-", '3rd dimension': "-", '4th dimension': "-", 'parameter': "GT",             'value': -1},
                {'group': group_UC, '1st dimension': "userconstraintRHS", '2nd dimension': "-", '3rd dimension': "-", '4th dimension': "-", 'parameter': "ts_groupPolicy", 'value': 1},
                {'group': group_UC, '1st dimension': "-", '2nd dimension': "-", '3rd dimension': "-", '4th dimension': "-", 'parameter': "penalty",        'value': 2000},
            ]

        if generated_rows:
            frames.append(pd.DataFrame.from_records(generated_rows, columns=expected_cols))

        # ---- Finalize without ever concatenating with an empty frame ----
        if frames:
            p_userConstraint = pd.concat(frames, ignore_index=True)
        else:
            p_userConstraint = pd.DataFrame(columns=expected_cols)

        # Standardize dtypes, fill NA
        p_userConstraint = utils.standardize_df_dtypes(p_userConstraint, fill_numeric_na=True)

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
        p_nEmission = utils.standardize_df_dtypes(p_nEmission, fill_numeric_na=True)


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
        ts_emissionPriceChange = utils.standardize_df_dtypes(ts_emissionPriceChange, fill_numeric_na=True)


        return ts_emissionPriceChange


    def create_gnGroup(self,
                   p_nEmission: pd.DataFrame,
                   ts_emissionPriceChange: pd.DataFrame,
                   p_gnu_io_flat: pd.DataFrame,
                   unitUnittype: pd.DataFrame,
                   df_unittypedata: pd.DataFrame,
                   input_dfs=[]) -> pd.DataFrame:
        """
        Build gnGroup['grid','node','group'] by matching:
          p_nEmission(node, emission)
            -> ts_emissionPriceChange(emission -> group)
            -> p_gnu_io_flat(node -> grid, unit)
            -> unitUnittype(unit -> unittype)
            -> df_unittypedata(unittype has any emission_group* == group)

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
        df_output = utils.standardize_df_dtypes(df_output)


        return df_output
    

    def drop_fake_MultiIndex(self, df):
        # Create a copy of the original DataFrame
        df_flat = df.copy()

        # Drop the first row by using its index position 0
        df_flat = df_flat.drop(df_flat.index[0]).reset_index(drop=True)

        return df_flat


    def remove_empty_columns(self, df: pd.DataFrame, cols_to_keep=None, treat_nan_as_empty=True):
        cols_to_keep = set(cols_to_keep or [])

        empty_cols_mask = df.apply(utils.is_col_empty, axis=0)

        # Keep protected columns even if empty
        keep = list(cols_to_keep & set(df.columns))
        if keep:
            empty_cols_mask.loc[keep] = False

        return df.loc[:, ~empty_cols_mask]


    def get_param_value(self, param, put, cap_row, tech_row, def_value):
        """
        Determine parameter value with fallback logic (ALL lookups case-insensitive).

        Priority:
          1) cap_row connection-specific (e.g., "<param>_<put>")
          2) cap_row base (only when put == 'output1')
          3) tech_row connection-specific
          4) tech_row base (only when put == 'output1')
          5) def_value

        Notes:
          - All key comparisons are done case-insensitively.
          - If both candidates in a tier are empty, that tier yields None (not 0),
            so we only fall back to def_value when both sources are empty.
        """
        logs = getattr(self, "builder_logs", [])

        def _iter_items(row_like):
            if row_like is None:
                return []
            # Try dict-like first
            if hasattr(row_like, "items"):
                try:
                    return list(row_like.items())
                except Exception:
                    pass
            # Try pandas Series-like (index + values)
            if hasattr(row_like, "index"):
                try:
                    return list(zip(row_like.index, list(row_like)))
                except Exception:
                    pass
            # Last resort: try to coerce to dict
            try:
                return list(dict(row_like).items())
            except Exception:
                return []

        def _to_ci_map(row_like):
            """Map lowercased, stripped keys -> value. First occurrence wins."""
            ci = {}
            for k, v in _iter_items(row_like):
                kl = str(k).strip().lower()
                if kl not in ci:
                    ci[kl] = v
            return ci

        cap_ci = _to_ci_map(cap_row)
        tech_ci = _to_ci_map(tech_row)

        p = str(param).strip().lower()
        put_l = str(put).strip().lower()
        key_conn = f"{p}_{put_l}"
        key_base = p  # considered only when put == 'output1'

        def _combine(v1, v2, ident_suffix):
            """
            Combine two candidates (connection-specific and base) for a tier:
            - If both empty -> return None (signals 'tier empty')
            - Else treat empties as 0 and add; if addition fails, prefer first non-empty
            """
            a_empty = utils.is_val_empty(v1, logs, treat_zero_as_empty=False, ident=f"get_param_value/{ident_suffix}/a")
            b_empty = utils.is_val_empty(v2, logs, treat_zero_as_empty=False, ident=f"get_param_value/{ident_suffix}/b")
            if a_empty and b_empty:
                return None

            a_val = 0 if a_empty else v1
            b_val = 0 if b_empty else v2
            try:
                return a_val + b_val
            except Exception:
                # Fallback: return whichever is non-empty
                return a_val if not a_empty else b_val

        # --- Tier 1 & 2: cap_row ---
        cap_conn = cap_ci.get(key_conn, None)
        cap_base = cap_ci.get(key_base, None) if put_l == "output1" else None
        primary = _combine(cap_conn, cap_base, "primary")
        if not utils.is_val_empty(primary, logs, treat_zero_as_empty=False, ident="get_param_value/primary"):
            return primary

        # --- Tier 3 & 4: tech_row ---
        tech_conn = tech_ci.get(key_conn, None)
        tech_base = tech_ci.get(key_base, None) if put_l == "output1" else None
        secondary = _combine(tech_conn, tech_base, "secondary")
        if not utils.is_val_empty(secondary, logs, treat_zero_as_empty=False, ident="get_param_value/secondary"):
            return secondary

        # --- Tier 5: default ---
        return def_value


# ------------------------------------------------------
# Main entry point for the script
# ------------------------------------------------------

    def run(self):

        # Check if the Excel file is already open before proceeding
        try: 
            check_if_bb_excel_open(self.output_file)
        except Exception as e:
            utils.log_status(f"{e}", self.builder_logs, level="warn")
            return self.builder_logs, self.bb_excel_succesfully_built

        # Create p_gnu_io
        if not self.df_unittypedata.empty and not self.df_unitdata.empty:
            p_gnu_io = self.create_p_gnu_io(self.df_unittypedata, self.df_unitdata)     
        else:
            utils.log_status(f"Missing unit data or unittype data, skipping p_gnu_io and derivatives.'", 
                       self.builder_logs, level="info")
            p_gnu_io = pd.DataFrame() 
        if not p_gnu_io.empty:
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
            utils.log_status(f"Missing transfer data, skipping p_gnn and derivatives.'", self.builder_logs, level="info")
            p_gnn = pd.DataFrame()
        if not p_gnn.empty:
            # Create flat version for easier use in other functions
            p_gnn_flat = self.drop_fake_MultiIndex(p_gnn)
        else:
            p_gnn_flat = pd.DataFrame()

        # p_gn
        p_gn = self.create_p_gn(p_gnu_io_flat, self.df_fueldata, self.df_demanddata, 
                                self.df_storagedata, self.ts_storage_limits, self.ts_domain_pairs)
        if not p_gn.empty:
            # Create flat version for easier use in other functions
            p_gn_flat = self.drop_fake_MultiIndex(p_gn)
        else:
            p_gn_flat = pd.DataFrame()

        # node based input tables
        p_gnBoundaryPropertiesForStates = self.create_p_gnBoundaryPropertiesForStates(p_gn_flat, 
                                                                                      self.df_storagedata, 
                                                                                      self.ts_storage_limits)
        ts_priceChange = self.create_ts_priceChange(p_gn_flat, self.df_fueldata)
        p_userconstraint = self.create_p_userConstraint(self.df_userconstraintdata,
                                                        p_gnu_io_flat, 
                                                        self.mingen_nodes,
                                                        self.builder_logs)

        # add storage start levels to p_gn and p_gnBoundaryPropertiesForStates
        (p_gn, p_gnBoundaryPropertiesForStates) = self.add_storage_starts(p_gn, p_gnBoundaryPropertiesForStates, 
                                                                          p_gnu_io_flat, self.ts_storage_limits)
        p_gn_flat = self.drop_fake_MultiIndex(p_gn)

        # emission based input tables
        p_nEmission = self.create_p_nEmission(p_gn_flat, self.df_fueldata)
        ts_emissionPriceChange = self.create_ts_emissionPriceChange(self.df_emissiondata)

        # group sets
        gnGroup = self.create_gnGroup(p_nEmission, ts_emissionPriceChange, p_gnu_io_flat, unitUnittype, 
                                      self.df_unittypedata)

        # Compile domains
        ts_grids = pd.DataFrame()
        if self.ts_domains is not None and 'grid' in self.ts_domains:
            ts_grids = pd.DataFrame(self.ts_domains['grid'], columns=['grid'])
        grid = self.compile_domain([p_gnu_io_flat, p_gnn_flat, p_gn_flat, ts_grids], 'grid')

        ts_nodes = pd.DataFrame()
        if self.ts_domains is not None and 'node' in self.ts_domains:
            ts_nodes = pd.DataFrame(self.ts_domains['node'], columns=['node'])    
        # The current code cannot compile domains from p_gnn because it has columns from_node, to_node, 
        # but the same nodes should be available in other tables
        node = self.compile_domain([p_gnu_io_flat, p_gn_flat, ts_nodes], 'node')  

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

        utils.log_status(f"Input excel for Backbone written to '{self.output_file}'", self.builder_logs, level="info")
        self.bb_excel_succesfully_built = True

        return self.builder_logs, self.bb_excel_succesfully_built
