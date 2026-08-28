import math
import os

import pandas as pd

import src.backbone_params as backbone_params
import src.utils as utils
import src.bb_excel.bb_excel_writer as writer
from src.bb_excel.bb_excel_inputs import BBExcelInputs
from src.bb_excel.bb_excel_tables import (
    as_float64,
    build_parameter_sheet,
    coerce_numeric_columns,
    compile_domain_df,
    is_positive,
)
from src.source_workbook_shape import CONNECTION_SUFFIXES, base_column_name
from src.utils import summarise


class BBExcelPipeline:
    """
    Assembles the Backbone energy system model input Excel from pre-processed
    source DataFrames produced by the pipeline.

    `docs/input-excel.md` is the documentation and carries the reasoning: which
    sheets are written, why a parameter column is missing whenever nothing set it,
    how a node is decided to be a price or a balance or a storage node, and what
    each warning is asking its reader to change.

    Numeric value conventions
    -------------------------
    0 = NA = None = "parameter not set" throughout this class.
    Backbone treats an absent parameter and an explicit 0 identically for all
    parameters whose Backbone default is 0.

    Frames are flat; sheets are not
    -------------------------------
    Every frame a create_*() returns is an ordinary DataFrame, and stays one for
    as long as this class holds it. The fake MultiIndex -- the blank-dimension
    header row GDXXRW reads -- is a way of *writing* a sheet, not of holding one,
    so it is applied once, per sheet, in write_workbook. See SHEET_DIMENSIONS for
    which sheets get it.

    Processing sequence for numeric columns:
      1. _coerce_numeric_dtypes() — casts all known numeric parameter columns in the
         source DataFrames to Float64, coercing non-numeric values to NA.
      2. Each create_*() function — after building its output DataFrame, applies
         non-zero defaults from PARAM_*_DEFAULTS (e.g. isActive = 1) via fillna.
         Doing this in create_*() rather than in _coerce_numeric_dtypes() ensures
         defaults are applied to all rows regardless of which data source contributed
         them (source DataFrames, time series, inferred unit/demand data, etc.).
      3. fill_numeric_na() — called at the end of each create_*() to convert any
         remaining NA to 0 before writing to Excel.

    Local truthiness checks are used throughout (not val, val == 1) rather than
    any shared utility, because the 0 = NA equivalence is specific to this class.

    Any parameter column may be absent
    ----------------------------------
    p_gn, p_gnn and p_gnu_io drop their all-empty parameter columns before the
    workbook is written (utils.drop_empty_parameter_columns), because 0 = NA
    means an all-zero column states nothing. So **a PARAM_* column is present
    only if some row set it**, and code reading one of those frames must guard:

        if 'upperLimitCapacityRatio' in p_gnu_io.columns:

    or use .get(name, default) on a row. Dimension columns need no guard --
    the drop cannot reach them, and the sheets are meaningless without them.

    The guard is easy to forget precisely where it matters most: a column is
    missing only when *no row in the whole model* set it, which is the case a
    populated test fixture does not have. Two of the three consumers of
    upperLimitCapacityRatio were guarded and the third crashed the build.
    """

    # Backbone's parameter vocabulary, its non-zero defaults and the two naming
    # conventions the source workbooks use all live in src/backbone_params.py,
    # so that the source-data and timeseries stages can read them too -- a stage
    # that needs to know whether 'upwardlimit' is a parameter name cannot import
    # this class to find out. Kept as attributes here because the methods below
    # read them off self, and because that is what a reader looking for
    # "which parameters does p_gnu_io have" expects to find.
    PARAM_GNU = backbone_params.PARAM_GNU
    PARAM_UNIT = backbone_params.PARAM_UNIT
    PARAM_GNN = backbone_params.PARAM_GNN
    PARAM_GN = backbone_params.PARAM_GN
    PARAM_GN_BOUNDARY_TYPES = backbone_params.PARAM_GN_BOUNDARY_TYPES
    PARAM_GN_BOUNDARY_PROPERTIES = backbone_params.PARAM_GN_BOUNDARY_PROPERTIES
    PARAM_EMISSION = backbone_params.PARAM_EMISSION
    PARAM_USERCONSTRAINT = backbone_params.PARAM_USERCONSTRAINT

    EMISSION_COLUMN_PREFIX = backbone_params.EMISSION_COLUMN_PREFIX

    PARAM_GNU_DEFAULTS = backbone_params.PARAM_GNU_DEFAULTS
    PARAM_UNIT_DEFAULTS = backbone_params.PARAM_UNIT_DEFAULTS
    PARAM_GNN_DEFAULTS = backbone_params.PARAM_GNN_DEFAULTS
    PARAM_GN_DEFAULTS = backbone_params.PARAM_GN_DEFAULTS

    UC_DIMENSION_COLUMNS = backbone_params.UC_DIMENSION_COLUMNS
    UC_UNUSED_DIMENSION = backbone_params.UC_UNUSED_DIMENSION

    def __init__(self, context: BBExcelInputs) -> None:

        self.logger = context.logger
        self.input_folder = context.input_folder
        self.output_folder = context.output_folder
        self.scen_tags = context.scen_tags

        # The source data tables are the only data channel into this class.
        # Whatever the timeseries phase had to say about a node has already been
        # merged into them (source_data_contributions.apply_contributions), so
        # nothing here needs to know which stage a row came from.
        self.source_data = context.source_data
        # Global
        self.df_emissiondata = self.source_data.df_emissiondata
        # Country specific
        self.df_nodedata =     self.source_data.df_nodedata
        self.df_transferdata = self.source_data.df_transferdata
        self.df_unitdata =     self.source_data.df_unitdata
        self.df_demanddata =   self.source_data.df_demanddata
        self.df_boundarydata = self.source_data.df_boundarydata
        # Custom
        self.df_userconstraintdata = self.source_data.df_userconstraintdata

        # Define the merged output file
        self.output_file = os.path.join(self.output_folder, 'inputData.xlsx')

        # Initiate a flag for successful code excecution
        self.bb_excel_succesfully_built = False



    # ------------------------------------------------------
    # Functions create and modify p_gnu_io 
    # ------------------------------------------------------

    def create_p_gnu_io(
        self,
        df_unitdata: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Creates a DataFrame representing unit input/output connections with parameters.

        This method processes the merged df_unitdata (which already contains type-level
        defaults from merge_unittypedata_into_unitdata) to build a relationship table
        between grids, nodes and units, with associated parameters.

        Parameters:
        -----------
        df_unitdata : DataFrame
            Merged unit data. Must include 'generator_id' and 'unit' columns,
            grid_input1/grid_output1/... columns, and node_input1/node_output1/...
            columns (all added by build_unit_grid_and_node_columns).
            Type-level parameter defaults are pre-merged via
            merge_unittypedata_into_unitdata().

        Returns:
        --------
        DataFrame
            Dimensions (grid, node, unit, input_output) and parameter columns
            (capacity, conversionCoeff, vomCosts, etc.)
        """
        if df_unitdata.empty:
            return pd.DataFrame()

        # dimension and parameter columns
        dimensions = ['grid', 'node', 'unit', 'input_output']
        param_gnu = self.PARAM_GNU

        # List to collect the new rows
        rows = []
        connections_without_a_node = []

        # Process each row in the merged df_unitdata.
        for _, cap_row in df_unitdata.iterrows():

            # Fetch unit name
            unit = cap_row['unit']

            # Identify all defined input/output connections for this unit.
            # grid_* columns were added to df_unitdata by build_unit_grid_and_node_columns,
            # which is also what CONNECTION_SUFFIXES mirrors -- so the set of connections
            # is stated once, in source_workbook_shape, rather than spelled out again here.
            put_candidates = [suffix.lstrip('_') for suffix in CONNECTION_SUFFIXES]
            available_puts = [put for put in put_candidates if f'grid_{put}' in cap_row.index]

            # Process each available input/output connection
            for put in available_puts:
                # Construct looped column names
                grid_col = f"grid_{put}"
                node_col = f"node_{put}"

                # A connection with a grid but no node cannot be written. The
                # matching grid check that used to stand here could not fire:
                # available_puts is built from the grid_<put> columns that exist.
                if node_col not in cap_row:
                    connections_without_a_node.append(f"{unit} {put}")
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

                # Add all other parameters. pd.NA is the sentinel for an absent
                # column, so that NA and "column missing" are treated the same
                # way; build_parameter_sheet turns it into the parameter's
                # default, or into 0 where there is no non-zero default.
                additional_params = {
                    param: cap_row.get(f'{param.lower()}_{put}', pd.NA)
                    for param in param_gnu
                }

                # Append base and additional parameters
                rows.append({**base_row, **additional_params})

        if connections_without_a_node:
            self.logger.log_status(
                f"{len(connections_without_a_node)} unit connection(s) name a grid but no "
                f"node, and were not written: {summarise(connections_without_a_node)}. "
                "Check spelling and files.",
                level="warn"
            )

        # 'capacity' is kept even when empty, as the Cdim=1 column dimension.
        # Dimension columns are out of the drop's reach by construction.
        return build_parameter_sheet(
            rows, dimensions, param_gnu,
            sort_by=['unit', 'input_output', 'node'],
            defaults=self.PARAM_GNU_DEFAULTS,
            must_keep='capacity',
        )


    def fill_capacities(
        self, 
        p_gnu_io: pd.DataFrame, 
        p_unit: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Fills missing capacity values of units with a set of rules. 
        Currently calculates missing input capacity, if 
            * 1 input and 1 output, other one without capacity
            * unit has 1 input without capacity, 2 or more outputs with capacity, and no 'cv' parameter
        """

        # Nothing to derive from: hand back what came in. Returning an empty frame
        # here -- as this used to -- discarded the whole p_gnu_io sheet whenever
        # p_unit was empty, because run() assigns the result straight back.
        if p_gnu_io.empty or p_unit.empty:
            return p_gnu_io

        p_gnu_io = p_gnu_io.copy()

        # A derived capacity is a float, so the column has to be one before the
        # loop writes into it: assigning 187.5 into an int64 column is a pandas
        # FutureWarning today and an error tomorrow. create_p_gnu_io always hands
        # over Float64; a caller that assembled the frame by hand may not have,
        # and that used to be hidden by the fake MultiIndex making every column
        # object on the way in.
        if 'capacity' in p_gnu_io.columns:
            p_gnu_io['capacity'] = p_gnu_io['capacity'].astype('Float64')

        # Best efficiency per unit. The eff* columns are read across the row, so
        # a unit whose efficiencies are all empty yields NaN -- which the guard
        # below has to test for explicitly, see there.
        eff_columns = [col for col in p_unit.columns if col.startswith('eff')]
        unit_efficiency = dict(zip(p_unit['unit'], p_unit[eff_columns].max(axis=1)))

        # (unit, input_output) -> its rows, grouped once. The closure this replaces
        # re-filtered the whole frame twice per unit.
        rows_by_unit_and_io = dict(tuple(p_gnu_io.groupby(['unit', 'input_output'])))
        no_rows = p_gnu_io.iloc[0:0]

        # Process each unit only once
        for unit in p_gnu_io['unit'].unique():
            efficiency = unit_efficiency.get(unit, 0)
            # pd.isna first: a unit whose eff columns are all empty yields NaN,
            # and `NaN <= 0` is False, so it slipped past this guard and reached
            # math.ceil(capacity / NaN) -- an unhandled ValueError rather than a
            # skipped unit.
            if pd.isna(efficiency) or efficiency <= 0:
                continue

            inputs = rows_by_unit_and_io.get((unit, 'input'), no_rows)
            outputs = rows_by_unit_and_io.get((unit, 'output'), no_rows)

            # Rule 1: 1 input and 1 output, other one without capacity
            if len(inputs) == 1 and len(outputs) == 1:
                input_idx = inputs.index[0]
                output_idx = outputs.index[0]
                input_cap = inputs.iloc[0]['capacity']
                output_cap = outputs.iloc[0]['capacity']

                # If input cap not set (0) and output cap is set, derive input from output
                if not input_cap and output_cap:
                    p_gnu_io.at[input_idx, 'capacity'] = math.ceil(output_cap / efficiency * 10) / 10

                # If output cap not set (0) and input cap is set, derive output from input
                elif input_cap and not output_cap:
                    p_gnu_io.at[output_idx, 'capacity'] = math.ceil(input_cap * efficiency * 10) / 10

            # Rule 2: 1 input without capacity, 2 or more outputs with capacity (no 'cv')
            elif len(inputs) == 1 and len(outputs) > 1:
                input_idx = inputs.index[0]
                input_cap = inputs.iloc[0]['capacity']

                if not input_cap:
                    # Check both outputs have capacity
                    output_caps = outputs['capacity']
                    if all(cap for cap in output_caps):
                        # Check for 'cv' parameter if column exists
                        skip = False
                        if 'cv' in outputs.columns:
                            cv_values = outputs['cv']
                            if (cv_values > 0).any():
                                skip = True

                        if not skip:
                            total_output = output_caps.sum()
                            p_gnu_io.at[input_idx, 'capacity'] = math.ceil(total_output / efficiency * 10) / 10

        return p_gnu_io


    def drop_redundant_units(
        self, 
        p_gnu_io: pd.DataFrame, 
        p_unit: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Drops units that have no capacity and no investment parameters.

        A unit is dropped when both conditions are true:
            1. All capacity values are zero/NaN for every (grid, node, unit) row in p_gnu_io
            2. Investment parameters are zero/NaN: invCosts in p_gnu_io and maxUnitCount in p_unit

        Parameters
        ----------
        p_gnu_io : DataFrame
            Unit input/output table.
        p_unit : DataFrame
            Unit parameter table.

        Returns
        -------
        tuple of (DataFrame, DataFrame)
            Filtered p_gnu_io and p_unit with redundant units removed.
        """
        if p_gnu_io.empty or p_unit.empty:
            return p_gnu_io, p_unit

        # No copies: nothing here mutates, and the filter at the end returns a
        # new frame of its own. Both frames are grouped once rather than
        # re-filtered per unit.
        units_to_drop = []
        gnu_by_unit = dict(tuple(p_gnu_io.groupby('unit')))
        unit_by_unit = dict(tuple(p_unit.groupby('unit'))) if 'unit' in p_unit.columns else {}

        for unit, unit_rows in gnu_by_unit.items():

            # Condition 1: all capacity values are zero or NaN
            has_capacity = False
            if 'capacity' in unit_rows.columns:
                has_capacity = unit_rows['capacity'].notna().any() and (unit_rows['capacity'] != 0).any()
            if has_capacity:
                continue

            # Condition 2: invCosts zero/NaN in p_gnu_io AND maxUnitCount zero/NaN in p_unit
            has_inv_costs = False
            if 'invCosts' in unit_rows.columns:
                has_inv_costs = unit_rows['invCosts'].notna().any() and (unit_rows['invCosts'] != 0).any()
            if has_inv_costs:
                continue

            has_max_unit_count = False
            unit_row = unit_by_unit.get(unit)
            if unit_row is not None and 'maxUnitCount' in unit_row.columns:
                val = unit_row['maxUnitCount'].iloc[0]
                has_max_unit_count = pd.notna(val) and val != 0
            if has_max_unit_count:
                continue

            units_to_drop.append(unit)

        if units_to_drop:
            self.logger.log_status(
                f"Dropped {len(units_to_drop)} unit(s) with zero capacity and no investment "
                f"parameters: {summarise(units_to_drop)}.",
                level="skip"
            )
            p_gnu_io = p_gnu_io[~p_gnu_io['unit'].isin(units_to_drop)]
            p_unit = p_unit[~p_unit['unit'].isin(units_to_drop)]

        return p_gnu_io, p_unit


    # ------------------------------------------------------
    # Functions to create unit derived input tables: 
    # ------------------------------------------------------

    @staticmethod
    def _unit_rows_by_lowercase_name(df_unitdata: pd.DataFrame) -> dict:
        """``{lowercased unit name: its row}``, first occurrence winning.

        Three builders used to answer "what does df_unitdata say about this unit"
        by scanning the whole frame once per unit, which is the same question
        asked once per unit of a frame that has one row per unit. Built once and
        passed around instead.

        Lowercased because **GAMS is case-insensitive** and this build is run that
        way throughout: two spellings of a unit name are one unit by the time the
        workbook is read, so they have to be one unit here too. create_gnGroup
        matched exactly until 2026-08-28, which made it the odd one out rather
        than the strict one -- a unittype file spelling a name differently from
        unitdata silently cost that unit its emission group.

        First occurrence wins, matching the ``.iloc[0]`` the scans took, and
        matching compile_domain_df's rule that the first spelling seen is the one
        the model keeps.
        """
        rows = {}
        if df_unitdata.empty or 'unit' not in df_unitdata.columns:
            return rows
        for name, row in zip(df_unitdata['unit'], df_unitdata.to_dict('records')):
            key = str(name).lower()
            if key not in rows:
                rows[key] = row
        return rows


    def create_unitUnittype(
        self,
        df_unitdata: pd.DataFrame,
        active_units  # array-like of unit names
        ) -> pd.DataFrame:
        # return empty dataframe if no input data
        if df_unitdata.empty:
            return pd.DataFrame(columns=['unit', 'unittype'])
        if 'unittype' not in df_unitdata.columns:
            return pd.DataFrame(columns=['unit', 'unittype'])

        # Get unit-unittype pairs directly from df_unitdata for active units
        unitUnittype = (
            df_unitdata.loc[df_unitdata['unit'].isin(active_units), ['unit', 'unittype']]
            .dropna(subset=['unit', 'unittype'])
            .drop_duplicates(subset=['unit'])
            .reset_index(drop=True)
        )
        return unitUnittype
    

    def create_flowUnit(
        self,
        df_unitdata: pd.DataFrame,
        unitUnittype: pd.DataFrame
        ) -> pd.DataFrame:
        # return empty dataframe if no input data
        if unitUnittype.empty:
            return pd.DataFrame()

        # 'flow' is now directly in the merged df_unitdata.
        if 'flow' not in df_unitdata.columns:
            return pd.DataFrame(columns=['flow', 'unit'])

        # Keep only active units (those that made it through create_p_gnu_io)
        active_units = unitUnittype[['unit']]
        merged = active_units.merge(df_unitdata[['unit', 'flow']], on='unit', how='inner')
        # Exclude rows without a flow value ('' after fill_all_na, or explicit non-empty)
        flowUnit = merged[merged['flow'].notna() & (merged['flow'] != '')][['flow', 'unit']]
        return flowUnit


    def create_p_unit(
        self,
        unitUnittype: pd.DataFrame,
        df_unitdata: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Create the `p_unit` DataFrame for Backbone model input.

        Constructs a parameter table for each unit in `unitUnittype` using the
        merged df_unitdata, which already incorporates type-level defaults via
        merge_unittypedata_into_unitdata().  No separate df_unittypedata lookup
        is needed.

        Parameter values follow this priority:
            1. df_unitdata unit-specific values
            2. Type-level defaults (merged into df_unitdata in the pipeline)
            3. Non-zero parameter defaults from PARAM_UNIT_DEFAULTS (applied inline)
            4. 0 fallback for params absent from all input files

        Enforces the Backbone constraint:
            minShutdownHours <= startWarmAfterXhours <= startColdAfterXhours

        Parameters
        ----------
        unitUnittype : pd.DataFrame
            DataFrame linking individual units to their unittype.
            Must include columns ['unit', 'unittype'].
        df_unitdata : pd.DataFrame
            Merged unit data (unit-specific + type defaults).
            Must include column 'unit'.

        Returns
        -------
        pd.DataFrame
            Finalized `p_unit` DataFrame — one row per unit, sorted by unit name.
        """
        # Dimension column names.
        dimensions = ['unit']

        param_unit = self.PARAM_UNIT
        # List to collect new rows.
        rows = []
        units_without_data = []

        unit_rows = self._unit_rows_by_lowercase_name(df_unitdata)

        # Process each row in unitUnittype.
        for _, u_row in unitUnittype.iterrows():
            unit = str(u_row['unit'])

            unit_row = unit_rows.get(unit.lower())
            if unit_row is None:
                units_without_data.append(unit)
                continue

            # pd.NA where the unit says nothing; build_parameter_sheet resolves it
            # to the parameter's default, or to 0 where there is no non-zero one.
            rows.append({
                'unit': unit,
                **{param: unit_row.get(param.lower(), pd.NA) for param in param_unit},
            })

        if units_without_data:
            self.logger.log_status(
                f"{len(units_without_data)} unit(s) have no row in the unit data and were "
                f"left out of p_unit: {summarise(units_without_data)}. "
                "Check spelling and files.",
                level="warn"
            )

        p_unit = build_parameter_sheet(
            rows, dimensions, param_unit,
            sort_by=['unit'],
            defaults=self.PARAM_UNIT_DEFAULTS,
        )

        # In Backbone a unit must have minShutdownHours <= startWarmAfterXhours <=
        # startColdAfterXhours, so clamp startColdAfterXhours from below. Done on
        # the finished sheet rather than per row: both are p_unit parameters, so
        # this is the same comparison, once per column instead of once per unit.
        clamped = 'startColdAfterXhours'
        if clamped in p_unit.columns and 'minShutdownHours' in p_unit.columns:
            p_unit[clamped] = p_unit[[clamped, 'minShutdownHours']].max(axis=1)

        # The drop is deferred until after the clamp, which is why this sheet
        # does not hand must_keep to build_parameter_sheet: a startColdAfterXhours
        # that is non-zero only because of the clamp still has something to say,
        # and dropping the column first would silently lose it.
        # 'isActive' is kept even when empty so the Cdim=1 column dimension always
        # has a member; PARAM_UNIT_DEFAULTS fills it whenever there are rows.
        return utils.drop_empty_parameter_columns(p_unit, param_unit, 'isActive')
    

    def create_effLevelGroupUnit(
        self,
        df_unitdata: pd.DataFrame,
        unitUnittype: pd.DataFrame
        ) -> pd.DataFrame:
        # List to accumulate new rows
        rows = []
        unit_rows = self._unit_rows_by_lowercase_name(df_unitdata)

        # Iterate over each row in unitUnittype
        for _, u_row in unitUnittype.iterrows():
            unit = u_row['unit']

            unit_row = unit_rows.get(str(unit).lower())
            if unit_row is None:
                continue

            # LP/MIP value (column name lowercased after normalize_dataframe).
            lp_mip = unit_row.get('lp/mip', '')
            if not isinstance(lp_mip, str):
                lp_mip = ''

            if lp_mip in ['LP', 'MIP']:
                # effLevel1 = MIP/LP
                rows.append({
                    'effLevel': 'level1',
                    'effSelector': f'directOn{lp_mip}',
                    'unit': unit
                })
                # effLevel2-3 = LP
                for i in range(2, 4):
                    rows.append({
                        'effLevel': f'level{i}',
                        'effSelector': 'directOnLP',
                        'unit': unit
                    })

        # Create a new DataFrame from the list of rows with the desired columns
        effLevelGroupUnit = pd.DataFrame(rows, columns=['effLevel', 'effSelector', 'unit'])
        return effLevelGroupUnit



    # ------------------------------------------------------
    # Functions create transfer derived input tables
    # ------------------------------------------------------

    def create_p_gnn(
        self,
        df_transferdata: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Build p_gnn from df_transferdata where each row defines one directional link.
        Required columns: 'grid', 'from_node', 'to_node'.
        All PARAM_GNN columns are read directly by case-insensitive name matching.
        Rows with missing domain values are skipped.
        """
        if df_transferdata.empty:
            return pd.DataFrame()

        dimensions = ['grid', 'from_node', 'to_node']
        param_gnn = self.PARAM_GNN

        col_map = {c.lower(): c for c in df_transferdata.columns}

        rows = []
        for _, row in df_transferdata.iterrows():
            if not (pd.notna(row.get('grid')) and pd.notna(row.get('from_node')) and pd.notna(row.get('to_node'))):
                continue
            out = {
                'grid':      row['grid'],
                'from_node': row['from_node'],
                'to_node':   row['to_node'],
            }
            for p in param_gnn:
                actual = col_map.get(p.lower())
                out[p] = row[actual] if actual else pd.NA
            rows.append(out)

        # The optional transfer parameters (rampLimit, diffCoeff, invCost, ...) are
        # absent from a plain link and would otherwise be written as columns of
        # blanks. 'isActive' is kept even when empty so the Cdim=1 column dimension
        # always has a member.
        return build_parameter_sheet(
            rows, dimensions, param_gnn,
            sort_by=['grid', 'from_node', 'to_node'],
            defaults=self.PARAM_GNN_DEFAULTS,
            must_keep='isActive',
        )


    # ------------------------------------------------------
    # Functions create node derived input tables
    # ------------------------------------------------------


    def _collect_gn_pairs(
        self,
        p_gnu_io: pd.DataFrame,
        p_gnn: pd.DataFrame,
        df_nodedata: pd.DataFrame,
        df_demanddata: pd.DataFrame,
        ) -> pd.DataFrame:
        """
        Collect all unique (grid, node) pairs from df_demanddata, df_nodedata,
        p_gnu_io, and both ends of every transfer link in p_gnn.

        A node that exists only because a timeseries processor built a series
        for it arrives through df_nodedata like any other: the processor
        contributes the (grid, node) row and the merge folds it in.
        """
        pairs_demanddata = df_demanddata[['grid', 'node']] if not df_demanddata.empty else pd.DataFrame(columns=['grid', 'node'])
        pairs_nodedata   = df_nodedata[['grid', 'node']]   if not df_nodedata.empty   else pd.DataFrame(columns=['grid', 'node'])
        pairs_gnu        = p_gnu_io[['grid', 'node']] if not p_gnu_io.empty else pd.DataFrame(columns=['grid', 'node'])

        # Transfer nodes: both ends of each link become (grid, node) pairs
        pairs_gnn = []
        if not p_gnn.empty and 'grid' in p_gnn.columns:
            if 'from_node' in p_gnn.columns:
                pairs_gnn.append(p_gnn[['grid', 'from_node']].rename(columns={'from_node': 'node'}))
            if 'to_node' in p_gnn.columns:
                pairs_gnn.append(p_gnn[['grid', 'to_node']].rename(columns={'to_node': 'node'}))

        parts = [p for p in [pairs_demanddata, pairs_nodedata, pairs_gnu] + pairs_gnn if not p.empty]
        return (
            pd.concat(parts, ignore_index=True).drop_duplicates(ignore_index=True)
            if parts else pd.DataFrame(columns=['grid', 'node'])
        )


    #: The boundary types that imply a state variable. A node with only a
    #: maxSpill or a balancePenalty is not a storage: those say what may leave
    #: it and what an imbalance costs, not how much it holds.
    STATE_BOUNDARY_TYPES = ('upwardLimit', 'downwardLimit', 'reference')

    def _nodes_with_a_state_boundary(self, df_boundarydata: pd.DataFrame) -> set:
        """Nodes whose state is bounded, however the bound is given.

        A constant above zero and a time series count the same: both say the
        node holds something. Zero does not -- ``0`` is "not set" by the time a
        boundary reaches this class, so an all-zero limit is silence.

        A filter over _boundaries_by_node rather than a second scan of the table,
        so that "what the boundary table says" is read in one place.
        """
        return {
            node
            for (_grid, node, boundary_type), boundary
            in self._boundaries_by_node(df_boundarydata).items()
            if boundary_type in self.STATE_BOUNDARY_TYPES
            and (boundary['usetimeseries'] or is_positive(boundary['constant']))
        }

    def _boundaries_by_node(self, df_boundarydata: pd.DataFrame) -> dict:
        """``{(grid, node, type): {'usetimeseries': bool, 'constant': value}}``.

        Built once so that the per-node loops below can ask a question rather
        than scan the table again for every boundary type.

        ``usetimeseries`` is reduced to a plain bool here, because the column is
        ``Float64`` when something set it and all-NA ``object`` when nothing did
        -- and ``pd.NA == 1`` is neither True nor False, it raises when a caller
        puts it in an ``if``. Either property column may be absent entirely: a
        table built only from contributions carries what the processors stated
        and nothing else.
        """
        if df_boundarydata is None or df_boundarydata.empty:
            return {}
        if not {'grid', 'node', 'param_gnboundarytypes'} <= set(df_boundarydata.columns):
            return {}

        blank = pd.Series(pd.NA, index=df_boundarydata.index, dtype='object')
        use_series = df_boundarydata.get('usetimeseries', blank)
        constants = df_boundarydata.get('constant', blank)

        return {
            (grid, node, boundary_type): {
                'usetimeseries': bool(pd.notna(flag) and flag == 1),
                'constant': constant,
            }
            for grid, node, boundary_type, flag, constant in zip(
                df_boundarydata['grid'],
                df_boundarydata['node'],
                df_boundarydata['param_gnboundarytypes'],
                use_series,
                constants,
            )
        }

    def create_p_gn(
        self,
        unique_gn_pairs: pd.DataFrame,
        p_gnu_io: pd.DataFrame,
        df_nodedata: pd.DataFrame,
        df_demanddata: pd.DataFrame,
        df_boundarydata: pd.DataFrame,
        ) -> pd.DataFrame:
        """
        Creates p_gn from pre-collected (grid, node) pairs.

        Every node is unique and can have only one grid. Each node has only
        one row of data after merge_row_by_row.

        Phase 1 — Node classification:
            Each node is classified as a price node (usePrice = 1) or balance node
            (nodeBalance = 1), and balance nodes optionally as a storage node
            (energyStoredPerUnitOfState > 0).  Explicit user values in df_nodedata
            are used as-is; missing values are deduced:
                - usePrice: inferred if 'price' column is non-empty in df_nodedata
                - nodeBalance: inferred for demand grids or when nodedata is present
                - energyStoredPerUnitOfState: read from df_nodedata when provided;
                  otherwise inferred from an upwardLimit / downwardLimit / reference
                  boundary in df_boundarydata — set by a constant or by a time
                  series, it makes no difference — or from upperLimitCapacityRatio
                  in p_gnu_io. Deduced storage nodes default to 1; price nodes
                  and non-storage balance nodes default to 0.

        Remaining param_gn:
            All other PARAM_GN columns (including isActive) are read from df_nodedata.

        The deduction table, and why maxSpill and balancePenalty are deliberately
        not in it, is "How a node is classified" in docs/input-excel.md.
        """
        dimensions = ['grid', 'node']
        param_gn = self.PARAM_GN

        rows = []  

        # --- Preprocess data for the loop ---

        # Collected rather than logged per node: on a build with a missing input
        # file these fire for every node in the model at once.
        priced_and_balanced = []
        priced_and_stored = []
        neither_price_nor_balance = []

        # Nodes in df_demanddata — each node belongs to exactly one grid, so node alone is sufficient
        demand_nodes = (
            set(df_demanddata['node'].dropna())
            if not df_demanddata.empty else set()
        )

        # One nodedata row per node after merge_row_by_row, so this is a lookup
        # rather than a search -- it used to re-filter the whole frame for every
        # (grid, node) pair.
        nodedata_by_node = (
            dict(tuple(df_nodedata.groupby('node')))
            if not df_nodedata.empty and 'node' in df_nodedata.columns else {}
        )
        no_node_data = pd.DataFrame()

        # Nodes carrying a state boundary that implies a state variable. The
        # spill limits and balancePenalty are deliberately not in that set: a
        # node can have a maximum spill rate without storing anything.
        storage_bound_nodes = self._nodes_with_a_state_boundary(df_boundarydata)

        # --- Process each (grid, node) pair ---
        for _, pair in unique_gn_pairs.iterrows():
            grid = pair['grid']
            node = pair['node']

            node_data = nodedata_by_node.get(node, no_node_data)


            # ---- Phase 1: Node classification ----

            # Read explicit user-provided classification values from df_nodedata
            # (columns are lowercase after normalize_dataframe)
            usePrice = node_data['useprice'].iloc[0] if 'useprice' in node_data.columns and not node_data.empty else None
            nodeBalance = node_data['nodebalance'].iloc[0] if 'nodebalance' in node_data.columns and not node_data.empty else None
            energyStoredPerUnitOfState = node_data['energystoredperunitofstate'].iloc[0] if 'energystoredperunitofstate' in node_data.columns and not node_data.empty else None

            # Normalize flags: 0 and NA mean "not set" (same as absent column).
            # Collapse 0/None/NA into a single None sentinel so all downstream
            # is None / not checks behave identically regardless of source.
            usePrice    = 1 if pd.notna(usePrice)    and usePrice    == 1 else None
            nodeBalance = 1 if pd.notna(nodeBalance) and nodeBalance == 1 else None
            # energyStoredPerUnitOfState: preserve explicit positive numeric values from data;
            # only collapse 0/None/NA to None (deduction fallback applied further below).
            energyStoredPerUnitOfState = (
                float(energyStoredPerUnitOfState)
                if pd.notna(energyStoredPerUnitOfState) and energyStoredPerUnitOfState > 0
                else None
            )

            # Conflict checks on user-provided values
            if usePrice == 1 and nodeBalance == 1:
                priced_and_balanced.append(node)
            if usePrice == 1 and energyStoredPerUnitOfState is not None:
                priced_and_stored.append(node)

            # Deduction: usePrice — infer from 'price' column if not explicitly set
            if usePrice is None and not node_data.empty and 'price' in node_data.columns:
                price_val = node_data['price'].iloc[0]
                if pd.notna(price_val) and price_val > 0:
                    usePrice = 1

            # Deduction: nodeBalance — infer for demand nodes
            if nodeBalance is None and node in demand_nodes:
                nodeBalance = 1

            # Deduction: energyStoredPerUnitOfState — run only when not explicitly set
            # and only for non-price nodes
            if energyStoredPerUnitOfState is None and not usePrice:

                # 1) the node has an upwardLimit, downwardLimit or reference
                if node in storage_bound_nodes:
                    energyStoredPerUnitOfState = 1

                # 2) upperLimitCapacityRatio defined in p_gnu_io for any of the units.
                # check over sum is acceptable as upperLimitCapacityRatio is always positive
                if not energyStoredPerUnitOfState and not p_gnu_io.empty and 'upperLimitCapacityRatio' in p_gnu_io.columns:
                    if p_gnu_io.loc[p_gnu_io['node'] == node, 'upperLimitCapacityRatio'].sum() > 0:
                        energyStoredPerUnitOfState = 1

            # Derivative nodeBalance flag for storage nodes
            if energyStoredPerUnitOfState and nodeBalance is None:
                nodeBalance = 1

            # Warn if node passed all checks unresolved
            if not nodeBalance and not usePrice:
                neither_price_nor_balance.append(node)

            # Resolve energyStoredPerUnitOfState to its final value.
            # Deduction above sets a value for storage nodes; if still None the node
            # is a price node or a non-storage balance node — default to 0.
            if energyStoredPerUnitOfState is None:
                energyStoredPerUnitOfState = 0

            # ---- Remaining param_gn ----

            row_dict = {
                'grid':                       grid,
                'node':                       node,
                'usePrice':                   usePrice,
                'nodeBalance':                nodeBalance,
                'energyStoredPerUnitOfState': energyStoredPerUnitOfState,
            }

            # Add remaining param_gn (isActive + others) from nodedata.
            # Skip zeros: 0 means "not set" (same as absent) in BBExcelPipeline.
            if not node_data.empty:
                for key in (k for k in param_gn if k not in row_dict):
                    low = key.lower()
                    if low in node_data.columns:
                        val = node_data[low].iloc[0]
                        if pd.notna(val) and val != 0:
                            row_dict[key] = val

            rows.append(row_dict)

        for nodes, message in (
            (priced_and_balanced,
             "set both 'usePrice' and 'nodeBalance'"),
            (priced_and_stored,
             "set 'usePrice' together with 'energyStoredPerUnitOfState'"),
            (neither_price_nor_balance,
             "are neither price nor balance nodes, and nothing in the data says which"),
        ):
            if nodes:
                self.logger.log_status(
                    f"{len(nodes)} node(s) {message}: {summarise(nodes)}. Check the node data.",
                    level="warn"
                )

        # The defaults cover nodes with no df_nodedata entry at all. 'isActive' is
        # kept even when empty so the Cdim=1 column dimension always has a member;
        # with any row at all it is non-empty anyway, PARAM_GN_DEFAULTS having
        # filled it.
        return build_parameter_sheet(
            rows, dimensions, param_gn,
            sort_by=['grid', 'node'],
            defaults=self.PARAM_GN_DEFAULTS,
            must_keep='isActive',
        )


    def create_p_gnBoundaryPropertiesForStates(
        self,
        p_gn: pd.DataFrame,
        df_boundarydata: pd.DataFrame,
        ) -> pd.DataFrame:
        """
        Creates a DataFrame that defines boundary properties for nodes in an energy grid system.

        Reads ``df_boundarydata`` and nothing else. That table already carries
        every boundary the model has, whether the workbook wrote it as a constant
        column on nodedata or a timeseries processor contributed it, so this
        function no longer knows -- or needs to know -- which stage a row came
        from.

        Which flag reaches the sheet
        ---------------------------
        Backbone treats ``useConstant`` and ``useTimeseries`` as one either/or
        property (``useConstantOrTimeseries`` in inc/1a_definitions.gms), so
        exactly one of them is written per row. **A time series wins.** That
        single choice is the whole of the precedence rule, which used to be
        implicit in the order two data sources were consulted in.

        The other direction is a workbook's to take: the contribution merge fills
        only where the workbook said nothing, so ``usetimeseries = 0`` written by
        hand survives a processor claiming otherwise, and the constant is used.

        A constant of zero writes nothing: ``0`` is "not set" by the time a value
        reaches this class, and Backbone reads it the same way.

        Parameters:
        -----------
        p_gn : DataFrame containing node configurations.
            Must include columns: 'grid', 'node', 'nodeBalance', 'energyStoredPerUnitOfState'

        df_boundarydata : long-format boundary table with columns 'grid', 'node',
            'param_gnboundarytypes' and the lowercased param_gnBoundaryProperties.

        Returns:
        --------
        DataFrame
            Dimensions ['grid', 'node', 'param_gnBoundaryTypes'] and the
            param_gnBoundaryProperties ['useConstant', 'constant', 'useTimeseries',
            'slackCost']. Defines boundary constraints for nodes in the system.
        """
        if p_gn.empty:
            return pd.DataFrame()

        # Define the dimensions and parameters of the output DataFrame
        dimensions = ['grid', 'node', 'param_gnBoundaryTypes']

        # Properties that will be assigned to each boundary type
        param_gnBoundaryProperties = self.PARAM_GN_BOUNDARY_PROPERTIES

        # Initialize an empty list to collect all rows for the output DataFrame
        rows = []

        # {(node, param_gnBoundaryTypes): row}, so the loop below can ask about a
        # node's boundaries without scanning the table once per boundary type.
        boundaries = self._boundaries_by_node(df_boundarydata)

        # Process each node in the system that requires balance constraints
        for _, gn_row in p_gn.iterrows():
            # Only process nodes with balance requirements (nodeBalance = 1)
            nodeBalance = gn_row.get('nodeBalance', 0)
            if nodeBalance == 1:
                grid = gn_row['grid']
                node = gn_row['node']

                # Process each boundary type for this node
                for p_type in self.PARAM_GN_BOUNDARY_TYPES:
                    boundary = boundaries.get((grid, node, p_type))
                    if boundary is None:
                        continue

                    if boundary['usetimeseries']:
                        rows.append({
                            'grid':                     grid,
                            'node':                     node,
                            'param_gnBoundaryTypes':    p_type,
                            'useTimeseries':            1,
                        })
                        continue

                    value = boundary['constant']
                    if is_positive(value):
                        rows.append({
                            'grid':                     grid,
                            'node':                     node,
                            'param_gnBoundaryTypes':    p_type,
                            'useConstant':              1,
                            'constant':                 value,
                        })

            # Additional check for storage nodes
            isStorage = gn_row.get('energyStoredPerUnitOfState', 0)
            if isStorage and isStorage > 0:
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


        # No must_keep: add_storage_starts appends to this sheet and drops its
        # empty property columns afterwards, so dropping here would be undone.
        return build_parameter_sheet(
            rows, dimensions, param_gnBoundaryProperties,
            sort_by=['grid', 'node'],
        )


    def add_storage_starts(
        self, p_gn: pd.DataFrame, 
        p_gnBoundaryPropertiesForStates: pd.DataFrame, 
        p_gnu_io: pd.DataFrame,
        df_boundarydata: pd.DataFrame
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Adds p_gn('boundStart') and p_gnBoundaryPropertiesForStates('reference')
        for storage nodes.

        "Storage start levels" in docs/input-excel.md carries the rule and the
        two sources it reads, in order.

        The number this writes is provisional for hydro
        ----------------------------------------------
        ``changes.inc`` recomputes the reference of every ``psOpen`` and
        ``reservoir`` node from the *maximum* of its upwardLimit series, gated on
        ``boundStart = 1`` and a reference above zero. So for those nodes what
        matters here is that both gates are passed, not what the value is. The
        rule below is what the other storages -- batteries, closed pumped hydro,
        gas tanks -- actually get, and it is due a rewrite of its own: as written
        it cannot express a run that starts and ends in summer.

        Parameters:
            p_gn: DataFrame with columns ['grid', 'node'] and possibly 'energyStoredPerUnitOfState'
            p_gnBoundaryPropertiesForStates: DataFrame with columns ['grid', 'node', 'param_gnBoundaryTypes', 'param_gnBoundaryProperties']
            df_boundarydata: long-format boundary table, for the node's upwardLimit

        Returns:
            tuple: (p_gn, p_gnBoundaryPropertiesForStates) with updated values
        """
        if p_gn.empty or p_gnBoundaryPropertiesForStates.empty:
            return(p_gn, p_gnBoundaryPropertiesForStates)

        boundaries = self._boundaries_by_node(df_boundarydata)

        p_gn = p_gn.copy()
        p_gnBoundaryPropertiesForStates = p_gnBoundaryPropertiesForStates.copy()

        # Identify storage nodes - those where energyStoredPerUnitOfState > 0
        storage_gn = []
        if 'energyStoredPerUnitOfState' in p_gn.columns:
            for _, row in p_gn.iterrows():
                isStorage = row.get('energyStoredPerUnitOfState', 0)
                if isStorage and isStorage > 0:
                    storage_gn.append((row['grid'], row['node']))

        # Add 'boundStart' column to p_gn, initializing with 0
        p_gn['boundStart'] = 0
        unbounded_starts = []

        # Process each storage node
        for grid, node in storage_gn:

            # 1) the node's own upwardLimit, as df_boundarydata has it.
            # Read from the table rather than from the sheet built above,
            # because the sheet carries no constant on a row that resolved to
            # useTimeseries -- and a node whose limit comes from a series is
            # exactly the case that needs a start level.
            start_value = 0
            upward_limit = boundaries.get((grid, node, 'upwardLimit'))
            if upward_limit is not None:
                constant = upward_limit['constant']
                if is_positive(constant):
                    start_value = constant

            # 2) calculate maximum storage based on p_gnu_io('upperLimitCapacityRatio')
            # The column is absent whenever no unit in the whole model sets it:
            # drop_empty_parameter_columns removes an all-empty PARAM_GNU column
            # from p_gnu_io, and this is reached exactly when source 1 found
            # nothing, so a storage node declared any other way used to reach it
            # and die on a bare KeyError.
            if (start_value == 0
                    and not p_gnu_io.empty
                    and 'upperLimitCapacityRatio' in p_gnu_io.columns):
                subset_p_gnu_io = p_gnu_io[(p_gnu_io['grid'] == grid) &
                                                (p_gnu_io['node'] == node) &
                                                (p_gnu_io['upperLimitCapacityRatio'] > 0)
                                                ]
                if not subset_p_gnu_io.empty:
                        # Use the subset dataframe and get the first row if there are multiple matches
                        capacity = subset_p_gnu_io['capacity'].iloc[0]
                        upper_limit = subset_p_gnu_io['upperLimitCapacityRatio'].iloc[0]
                        if pd.notna(capacity):
                            start_value = capacity * upper_limit                      

            # A start value of 0 is not a start value. Backbone gates the bound on
            # the reference constant's own value (3d_setVariableLimits.gms), and 0
            # is indistinguishable from absent there, so writing boundStart=1 with
            # a 0 reference bound nothing while looking in the workbook as though
            # it did -- and the storage was free to initialise full.
            #
            # Skipping leaves exactly the same unconstrained model, so nothing
            # changes for the solve; what changes is that the user is told. The
            # node is named because the fix is in their data: give the node an
            # upwardLimit, or give one of its units an upperLimitCapacityRatio.
            if pd.notna(start_value) and start_value > 0:
                # Set boundStart to 1 for storage nodes
                p_gn.loc[(p_gn['grid'] == grid) & (p_gn['node'] == node), 'boundStart'] = 1

                new_constant = round(start_value * 0.7, 0)
                # Create a mask to find the 'reference' row for this grid and node
                ref_mask = (
                    (p_gnBoundaryPropertiesForStates['grid'] == grid) &
                    (p_gnBoundaryPropertiesForStates['node'] == node) &
                    (p_gnBoundaryPropertiesForStates['param_gnBoundaryTypes'] == 'reference')
                )

                if not p_gnBoundaryPropertiesForStates.loc[ref_mask].empty:
                    # If row exists, update the 'constant' value (and useConstant as needed)
                    p_gnBoundaryPropertiesForStates.loc[ref_mask, 'constant'] = new_constant
                    p_gnBoundaryPropertiesForStates.loc[ref_mask, 'useConstant'] = 1
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
                    p_gnBoundaryPropertiesForStates = pd.concat(
                        [p_gnBoundaryPropertiesForStates, new_row_df],
                        ignore_index=True
                    )
            else:
                unbounded_starts.append(node)

        if unbounded_starts:
            self.logger.log_status(
                f"No storage start level could be determined for {len(unbounded_starts)} "
                f"node(s): {summarise(unbounded_starts)}. Each keeps a state variable but "
                "starts unbounded, so the solver may start it full. Give the node an "
                "'upwardLimit', or one of its units an 'upperLimitCapacityRatio'.",
                level="warn"
            )

        # Standardize dtypes, fill NA. It has to happen after the loop, not
        # before it: the rows appended above carry only five keys, so every other
        # column comes through the concat as NaN and would reach the workbook
        # that way.
        p_gn = utils.fill_numeric_na(utils.standardize_df_dtypes(p_gn))
        p_gnBoundaryPropertiesForStates = utils.fill_all_na(
            utils.standardize_df_dtypes(p_gnBoundaryPropertiesForStates)
        )

        # Drop parameter columns nothing set. 'useConstant' and 'isActive' are kept
        # even when empty so each sheet's Cdim=1 column dimension has a member.
        #
        # p_gn is dropped again here, not only in create_p_gn: 'boundStart' is
        # added above after that drop has already run, so a model with no storage
        # node wrote a column of zeros -- exactly what "any parameter column may be
        # absent" in the class docstring says must not happen.
        p_gnBoundaryPropertiesForStates = utils.drop_empty_parameter_columns(
            p_gnBoundaryPropertiesForStates, self.PARAM_GN_BOUNDARY_PROPERTIES, 'useConstant'
        )
        p_gn = utils.drop_empty_parameter_columns(p_gn, self.PARAM_GN, 'isActive')


        # Sort p_gnBoundaryPropertiesForStates alphabetically by [grid, node] in a case-insensitive manner
        p_gnBoundaryPropertiesForStates = p_gnBoundaryPropertiesForStates.sort_values(
                                                    by=['grid', 'node', 'param_gnBoundaryTypes'], 
                                                    key=lambda x: x.str.lower()
                                                    ).reset_index(drop=True)

        return (p_gn, p_gnBoundaryPropertiesForStates)


    def create_p_nEmission(
        self,
        p_gn: pd.DataFrame,
        df_nodedata: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Create p_nEmission['node', 'emission', 'value'] emission factors (tEmission / MWh) for each node.

        Parameters
        p_gn : pandas DataFrame with columns 'grid' and 'node'.
        df_nodedata : pandas DataFrame with column 'grid' and optional columns 'emission_XX'
            where XX is emission name (e.g., CO2, CH4).
        """
        # Return empty dataframe if no nodes (empty p_gn) or no node data (empty df_nodedata)
        if p_gn.empty or df_nodedata.empty:
            return pd.DataFrame()

        # Extract emission names from column names. Return empty dataframe if no emissions.
        emission_cols = [col for col in df_nodedata.columns if col.startswith('emission_')]
        if not emission_cols:
            return pd.DataFrame()
        emissions = [col.replace('emission_', '') for col in emission_cols]

        # Build p_nEmission directly from df_nodedata, matched to p_gn nodes
        valid_nodes = set(p_gn['node'].dropna().unique())
        p_nEmission_data = []
        for _, fuel_row in df_nodedata.iterrows():
            node = fuel_row.get('node')
            if pd.isna(node) or node not in valid_nodes:
                continue
            for col, emission in zip(emission_cols, emissions):
                value = fuel_row.get(col)
                if pd.notna(value) and value > 0:
                    p_nEmission_data.append({
                        'node': node,
                        'emission': emission,
                        'value': value
                    })

        p_nEmission = pd.DataFrame(p_nEmission_data)
        p_nEmission = utils.fill_numeric_na(utils.standardize_df_dtypes(p_nEmission))

        return p_nEmission


    def create_ts_emissionPriceChange(
        self, 
        df_emissiondata: pd.DataFrame
        ) -> pd.DataFrame:
        """
        Create ts_emissionPriceChange ['emission', 'group', 't', 'value'] DataFrame
        
        Parameters: 
            df_emissiondata : pandas DataFrame with columns 'emission', 'group', and optional 'price'.
        """
        # Return empty dataframe if no emission data (empty df_emissiondata)
        if df_emissiondata.empty:
            return pd.DataFrame()
        
        # One row per (emission, group), carrying that pair's first price. The loop
        # this replaces took the pairs with drop_duplicates and then went back to
        # re-find each pair's first row -- which is the row drop_duplicates kept.
        pairs = df_emissiondata.drop_duplicates(subset=['emission', 'group'])

        ts_emissionPriceChange = pairs[['emission', 'group']].reset_index(drop=True)
        ts_emissionPriceChange['t'] = 't000001'
        ts_emissionPriceChange['value'] = (
            pairs['price'].reset_index(drop=True).fillna(0)
            if 'price' in df_emissiondata.columns else 0
        )

        return utils.fill_numeric_na(utils.standardize_df_dtypes(ts_emissionPriceChange))


    def create_gnGroup(
        self,
        p_nEmission: pd.DataFrame,
        ts_emissionPriceChange: pd.DataFrame,
        p_gnu_io: pd.DataFrame,
        df_unitdata: pd.DataFrame,
        ) -> pd.DataFrame:
        """
        Build gnGroup['grid', 'node', 'group'] from an emission-based five-step join:
          1. p_nEmission(node, emission)
          2. ts_emissionPriceChange(emission -> group)  [case-insensitive match]
          3. p_gnu_io(node -> grid, unit)
          4. unitUnittype(unit -> unittype)
          5. df_unitdata: unit row must have at least one emission_group*
             column whose value equals the group from step 2.
        A (grid, node, group) row is added for every combination that passes
        all five steps.

        Warnings are logged when:
          - p_nEmission is empty (no emission nodes defined).
          - ts_emissionPriceChange is empty (no emission price data loaded).
          - Some emission types in p_nEmission have no matching entry in
            ts_emissionPriceChange (logged with the unmatched names).
          - df_unitdata has no emission_group* columns at all.

        Duplicate rows are dropped before returning.
        """
        # Whether the unittype files declare emission groups at all is a fact about
        # the table, so it is asked once. It used to be asked -- and answered with
        # the same warning -- inside the innermost loop, once per matching row.
        emission_group_cols = [col for col in df_unitdata.columns if col.startswith('emission_group')]
        if not emission_group_cols:
            self.logger.log_status(
                "df_unitdata has no emission_group* columns, so no unit produces any "
                "emissions. Check that the unittype source file(s) include an "
                "emission_group column.",
                level="warn"
            )
            return pd.DataFrame()

        # emission -> group, lowercased, first entry winning. The same answer as
        # rescanning ts_emissionPriceChange for every node/emission pair.
        group_of_emission = {}
        if not ts_emissionPriceChange.empty:
            for name, group in zip(ts_emissionPriceChange['emission'], ts_emissionPriceChange['group']):
                group_of_emission.setdefault(str(name).lower(), group)

        unit_rows = self._unit_rows_by_lowercase_name(df_unitdata)

        gnu_by_node = dict(tuple(p_gnu_io.groupby('node'))) if not p_gnu_io.empty else {}

        rows_list = []
        for _, node_emission in p_nEmission.iterrows():
            node = node_emission['node']
            emission = node_emission['emission']

            group = group_of_emission.get(str(emission).lower())
            if group is None:
                continue

            for _, grid_node_unit in gnu_by_node.get(node, pd.DataFrame()).iterrows():
                unit_row = unit_rows.get(str(grid_node_unit['unit']).lower())
                if unit_row is None:
                    continue

                for col in emission_group_cols:
                    if unit_row.get(col) == group:
                        rows_list.append({
                            'grid': grid_node_unit['grid'],
                            'node': node,
                            'group': group,
                        })

        return pd.DataFrame(rows_list).drop_duplicates()


    # ------------------------------------------------------
    # Functions to create other input tables
    # ------------------------------------------------------


    def create_p_userconstraint(
        self,
        uc_data: pd.DataFrame,
        ) -> pd.DataFrame:
        """
        Creates the parameter DataFrame `p_userConstraint` defining user constraints.

        Every constraint comes from `uc_data`, with case-insensitive column
        handling. A processor that wants one of its own contributes rows to
        df_userconstraintdata like any other producer; there is no second,
        builder-generated source.

        Parameters
        ----------
        uc_data : pd.DataFrame
            User-defined constraints, possibly with column names in arbitrary case.
            Expected logical columns (case-insensitive):
            ['group', '1st dimension', '2nd dimension', '3rd dimension',
             '4th dimension', 'parameter', 'value'].

        Returns
        -------
        pd.DataFrame
            Combined DataFrame of all user constraints (`p_userConstraint`).
        """
        expected_cols = [
            'group', '1st dimension', '2nd dimension',
            '3rd dimension', '4th dimension', 'parameter', 'value'
        ]

        # Align uc_data onto the expected columns, case-insensitively.
        p_userConstraint = pd.DataFrame(columns=expected_cols)
        if uc_data is not None and not uc_data.empty:
            # map lowercased column name -> canonical expected column
            col_map = {c.lower(): c for c in expected_cols}
            rename_dict = {orig: col_map[orig.lower()] for orig in uc_data.columns if orig.lower() in col_map}
            uc_data_renamed = uc_data.rename(columns=rename_dict)

            missing = [c for c in expected_cols if c not in uc_data_renamed.columns]
            if missing:
                self.logger.log_status(f"uc_data missing required columns (after case-insensitive matching): {missing}",
                           level="warn")
                # Create them as NA rather than selecting strictly below, which
                # raised KeyError immediately after this warning was written.
                # A constraint using only the 1st and 2nd dimension is ordinary,
                # so a sheet that omits the unused columns must not fail a build.
                for column in missing:
                    uc_data_renamed[column] = pd.NA

            # keep only rows that aren't entirely NA to avoid dtype inference warnings later
            uc_data_aligned = uc_data_renamed[expected_cols].dropna(how="all")

            if not uc_data_aligned.empty:
                p_userConstraint = uc_data_aligned.reset_index(drop=True)

        # An unused uc slot must be '-', never blank: inc/1e_inputs.gms aborts the
        # run on anything else (see UC_UNUSED_DIMENSION). A user sheet that omits
        # the columns it does not need -- ordinary, and the columns are created as
        # NA above -- would otherwise write blanks that GAMS refuses.
        #
        # Done before the dtype pass on purpose: filled first, the slots are never
        # all-NA, so standardize_df_dtypes leaves them object rather than typing
        # them numeric and fill_numeric_na writing 0 into a set-element slot.
        for column in self.UC_DIMENSION_COLUMNS:
            if column in p_userConstraint.columns:
                filled = p_userConstraint[column].where(
                    p_userConstraint[column].notna(), self.UC_UNUSED_DIMENSION
                )
                # Blank and whitespace-only cells reach here as strings from sheets
                # that were never normalized, so emptiness is tested, not identity.
                p_userConstraint[column] = filled.map(
                    lambda v: self.UC_UNUSED_DIMENSION
                    if isinstance(v, str) and not v.strip()
                    else v
                )

        # group and parameter are not selector slots -- '-' would be a real label
        # there, naming a constraint or a Backbone variable that does not exist.
        for column in ('group', 'parameter'):
            if column in p_userConstraint.columns:
                blank = p_userConstraint[column].isna() | p_userConstraint[column].map(
                    lambda v: isinstance(v, str) and not v.strip()
                )
                if blank.any():
                    self.logger.log_status(
                        f"p_userconstraint has {int(blank.sum())} row(s) with an empty '{column}', "
                        "which Backbone cannot resolve. Check the userconstraintdata sheets.",
                        level="warn"
                    )

        # Standardize dtypes, fill NA
        p_userConstraint = utils.fill_numeric_na(utils.standardize_df_dtypes(p_userConstraint))

        return p_userConstraint



    # ------------------------------------------------------
    # Pre checks
    # ------------------------------------------------------

    def _normalize_unitdata_columns(self) -> None:
        """
        Restore the _output1 suffix on unsuffixed param_gnu columns in df_unitdata.

        normalize_dataframe (pipeline step 7) strips _output1 from numeric columns,
        so a bare column like 'capacity' implicitly represents output1.  This method
        makes that relationship explicit so downstream lookups can use the uniform
        '{param}_{put}' form (e.g. 'capacity_output1') for all connections.

        Params also in PARAM_UNIT (currently only 'isActive') are shared between
        create_p_gnu_io (needs '{param}_output1') and create_p_unit (needs base form).
        Those get a '{col}_output1' copy while the base column is kept.
        """
        _gnu  = {p.lower() for p in self.PARAM_GNU}
        _unit = {p.lower() for p in self.PARAM_UNIT}
        _gnu_only   = _gnu - _unit
        _gnu_shared = _gnu & _unit

        df = self.df_unitdata.copy()
        rename_map = {}
        copy_cols  = {}
        for col in list(df.columns):
            col_l = col.lower()
            if base_column_name(col) != col_l:
                continue  # already has a connection suffix
            if col_l in _gnu_only:
                output1_name = f'{col}_output1'
                if output1_name not in df.columns:
                    rename_map[col] = output1_name
            elif col_l in _gnu_shared:
                output1_name = f'{col}_output1'
                if output1_name not in df.columns:
                    copy_cols[output1_name] = df[col].copy()
        df = df.rename(columns=rename_map)
        for new_col, series in copy_cols.items():
            df[new_col] = series
        self.df_unitdata = df

    def _coerce_numeric_dtypes(self) -> None:
        """
        For each source DataFrame, casts every known numeric parameter column to Float64,
        coercing non-numeric values to NA.  NA values are left as-is; each create_*()
        function applies non-zero defaults (from PARAM_*_DEFAULTS) to its own output
        table so that defaults are enforced regardless of which source contributed a row.

        Columns that are not in any PARAM_* list are left untouched.
        param_unit columns that carry a connection suffix trigger a warning and are left
        untouched (ignored downstream).

        Coercion is silent here, and that is deliberate rather than an oversight: by
        the time a frame reaches this method, utils.gate_xlsx_frame has already
        reported and blanked every cell that looked like a number and was not, at the
        one place source workbooks are read. What is left for this to catch is a value
        that was never numeric in appearance either -- 'unknown' in a capacity column --
        which no content rule can distinguish from a legitimate label. Reporting it
        here would name a column rather than a cell and would fire on frames the user
        cannot act on, so the value becomes NA and the run carries on.

        Must be called after _normalize_unitdata_columns() so that param_gnu columns
        already carry explicit connection suffixes.
        """

        # --- 1) df_unitdata: PARAM_GNU + PARAM_UNIT ---
        # The only block that is not a plain name test: a param_gnu column may
        # carry a connection suffix, a param_unit column may not, and saying so
        # is the one thing this method warns about.
        _gnu  = {p.lower() for p in self.PARAM_GNU}
        _unit = {p.lower() for p in self.PARAM_UNIT}
        misplaced_suffix = []
        df = self.df_unitdata.copy()
        for col in df.columns:
            col_l = col.lower()
            # The suffix is named as well as stripped, because the param_unit
            # warning below quotes it back to whoever wrote the column.
            base = base_column_name(col)
            put = col_l[len(base) + 1:] if base != col_l else None
            if base in _gnu:
                df[col] = as_float64(df[col])
            elif base in _unit:
                if put is not None:
                    misplaced_suffix.append(f"{col} ('{base}' is not connection-specific)")
                else:
                    df[col] = as_float64(df[col])
        self.df_unitdata = df
        if misplaced_suffix:
            self.logger.log_status(
                f"{len(misplaced_suffix)} unit data column(s) name a unit-level parameter "
                f"with a connection suffix, which is not valid and leaves the column "
                f"ignored: {summarise(misplaced_suffix)}.",
                level="warn"
            )

        # --- 2) df_transferdata: PARAM_GNN ---
        self.df_transferdata = coerce_numeric_columns(self.df_transferdata, self.PARAM_GNN)

        # --- 3) df_nodedata + df_demanddata: PARAM_GN + PARAM_GN_BOUNDARY_TYPES ---
        # 'emission_XX' is included by prefix rather than by name: create_p_nEmission
        # discovers those columns the same way, and it compares each value with
        # `value > 0` -- which is a TypeError, not a bad number, if a string reaches
        # it. That was the most likely crash in the whole builder.
        for attr in ('df_nodedata', 'df_demanddata'):
            setattr(self, attr, coerce_numeric_columns(
                getattr(self, attr),
                self.PARAM_GN + self.PARAM_GN_BOUNDARY_TYPES,
                prefix=self.EMISSION_COLUMN_PREFIX,
            ))

        # --- 3b) df_boundarydata: the properties, not the types ---
        # The boundary type is a dimension value here rather than a column name,
        # so what needs coercing is the parameter block. Its 'constant' column is
        # a copy of a nodedata cell taken before the loop above could reach it,
        # which is the case this covers: 'unknown' in an upwardlimit column
        # would otherwise be compared with 0 and raise.
        # A property nothing set is left alone rather than typed Float64: an
        # all-NA column is object, and that is what says no assumption has been
        # made about it -- hence skip_all_na.
        self.df_boundarydata = coerce_numeric_columns(
            self.df_boundarydata, self.PARAM_GN_BOUNDARY_PROPERTIES, skip_all_na=True)

        # --- 4) df_emissiondata + df_userconstraintdata ---
        # Neither frame was coerced at all before. Their value columns are copied
        # straight into the output sheets, so anything non-numeric in them was
        # written to inputData.xlsx as a *text* cell -- which GDXXRW then reads as
        # a set label rather than a number, with no Python-side error anywhere.
        self.df_emissiondata = coerce_numeric_columns(self.df_emissiondata, self.PARAM_EMISSION)
        self.df_userconstraintdata = coerce_numeric_columns(
            self.df_userconstraintdata, self.PARAM_USERCONSTRAINT)

    # ------------------------------------------------------
    # Main entry point for the script
    # ------------------------------------------------------

    def run(self) -> None:

        # --- Pre-checks ---

        # Check if the Excel file is locked (e.g. open in Excel) before proceeding
        if os.path.exists(self.output_file):
            try:
                with open(self.output_file, 'a'):
                    pass
            except OSError:
                self.logger.log_status(
                    f"The Backbone input excel file '{self.output_file}' is currently open. Please close it and rerun the code.",
                    level="error"
                )
                return

        # Restore explicit _output1 suffix on unsuffixed param_gnu columns in df_unitdata.
        self._normalize_unitdata_columns()
        # Cast all known numeric columns to Float64; warn on invalid column patterns.
        self._coerce_numeric_dtypes()

        # --- Convert unit derived input data tables to DataFrames ---
        #
        # Every frame below is flat. The fake MultiIndex is a way of writing a
        # sheet, not a way of holding one, so it is applied once at the end --
        # see SHEET_DIMENSIONS and write_workbook.

        p_gnu_io = self.create_p_gnu_io(self.df_unitdata)

        # unit, unittype domain tables - derived from df_unitdata
        active_units = p_gnu_io['unit'].dropna().unique()
        unit = compile_domain_df(active_units.tolist(), 'unit')
        unittype_vals = (
            self.df_unitdata
            .loc[self.df_unitdata['unit'].isin(active_units), 'unittype']
            .dropna().unique().tolist()
        )
        unittype = compile_domain_df(unittype_vals, 'unittype')

        # unitUnittype - unit-unittype pairs from df_unitdata for active units
        unitUnittype = self.create_unitUnittype(self.df_unitdata, active_units)
        p_unit = self.create_p_unit(unitUnittype, self.df_unitdata)

        # Remove units without capacity or investment parameters
        p_gnu_io, p_unit = self.drop_redundant_units(p_gnu_io, p_unit)

        # Calculate missing input or output capacities in p_gnu_io
        p_gnu_io = self.fill_capacities(p_gnu_io, p_unit)

        # Create remaining unit related tables
        flowUnit = self.create_flowUnit(self.df_unitdata, unitUnittype)
        effLevelGroupUnit = self.create_effLevelGroupUnit(self.df_unitdata, unitUnittype)


        # --- Convert transfer derived input data tables to DataFrames ---

        p_gnn = self.create_p_gnn(self.df_transferdata)


        # --- Convert node derived input data tables to DataFrames ---

        # Collect unique (grid, node) pairs — covers units, transfers, nodedata, demanddata, timeseries
        unique_gn_pairs = self._collect_gn_pairs(
            p_gnu_io, p_gnn, self.df_nodedata, self.df_demanddata
        )

        # grid and node domains follow directly from unique_gn_pairs
        grid = compile_domain_df(unique_gn_pairs['grid'].dropna().tolist(), 'grid')
        node = compile_domain_df(unique_gn_pairs['node'].dropna().tolist(), 'node')

        # p_gn
        p_gn = self.create_p_gn(unique_gn_pairs, p_gnu_io, self.df_nodedata,
                                self.df_demanddata, self.df_boundarydata)

        # node based input tables
        p_gnBoundaryPropertiesForStates = self.create_p_gnBoundaryPropertiesForStates(
            p_gn, self.df_boundarydata
        )
        p_userconstraint = self.create_p_userconstraint(self.df_userconstraintdata)

        # add storage start levels to p_gn and p_gnBoundaryPropertiesForStates
        (p_gn, p_gnBoundaryPropertiesForStates) = self.add_storage_starts(p_gn, p_gnBoundaryPropertiesForStates,
                                                                          p_gnu_io, self.df_boundarydata)

        # emission based input tables
        p_nEmission = self.create_p_nEmission(p_gn, self.df_nodedata)
        ts_emissionPriceChange = self.create_ts_emissionPriceChange(self.df_emissiondata)


        # --- Compile remaining input tables ---

        # group sets
        gnGroup = self.create_gnGroup(p_nEmission, ts_emissionPriceChange, p_gnu_io,
                                      self.df_unitdata)

        # flow
        flow_vals = flowUnit['flow'].dropna().tolist() if 'flow' in flowUnit.columns else []
        flow = compile_domain_df(flow_vals, 'flow')

        # group
        group_vals = []
        for df in [p_userconstraint, ts_emissionPriceChange, gnGroup]:
            if 'group' in df.columns:
                group_vals.extend(df['group'].dropna().tolist())
        group = compile_domain_df(group_vals, 'group')

        # emission
        emission_vals = []
        for df in [ts_emissionPriceChange, p_nEmission]:
            if 'emission' in df.columns:
                emission_vals.extend(df['emission'].dropna().tolist())
        emission = compile_domain_df(emission_vals, 'emission')

        # restype
        restype = pd.DataFrame()

        # --- scenario tags to an excel sheet ---

        # Alternative columns are added only when present; column names follow the
        # pattern alternative, alternative2, alternative3, alternative4.
        _alt_col_names = ['alternative', 'alternative2', 'alternative3', 'alternative4']
        _n_alts = len(self.scen_tags) - 2
        scen_tags_df = pd.DataFrame([self.scen_tags], columns=['scenario', 'year'] + _alt_col_names[:_n_alts])


        # --- Write DataFrames to excel ---

        # Sheet order is the order of this mapping, and it is the order the
        # workbook gets.
        sheets = {
            # scenario tags
            'add_scen_tags': scen_tags_df,
            # node based input tables
            'grid': grid,
            'node': node,
            'p_gn': p_gn,
            'p_gnBoundaryPropertiesForStates': p_gnBoundaryPropertiesForStates,
            # transfer input tables
            'p_gnn': p_gnn,
            # unit input tables
            'unit': unit,
            'unittype': unittype,
            'unitUnittype': unitUnittype,
            'flowUnit': flowUnit,
            'effLevelGroupUnit': effLevelGroupUnit,
            'p_gnu_io': p_gnu_io,
            'p_unit': p_unit,
            'p_userconstraint': p_userconstraint,
            # emission based input tables
            'p_nEmission': p_nEmission,
            'ts_emissionPriceChange': ts_emissionPriceChange,
            # group sets
            'gnGroup': gnGroup,
            # remaining domains
            'group': group,
            'flow': flow,
            'emission': emission,
            'restype': restype,
        }
        writer.write_workbook(self.output_file, sheets)

        # --- Finishing touches ---

        # Apply the adjustments on the Excel file
        writer.add_index_sheet(self.output_file, self.input_folder, self.logger)
        writer.adjust_excel(self.output_file)

        self.logger.log_status(f"Input excel for Backbone written to '{self.output_file}'", level="info")
        self.bb_excel_succesfully_built = True
