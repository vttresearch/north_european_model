"""
Backbone's parameter vocabulary.

The names this project writes into ``inputData.xlsx``, the defaults it applies
before writing them, and the two naming conventions its source workbooks use.
Here rather than inside ``BBExcelPipeline`` so that the source-data and
timeseries stages can read them too: a stage that has to know whether
``upwardlimit`` is a parameter name cannot import the Excel builder to find out.

Where each value comes from
---------------------------
``../docs/dictionary.md`` in the Backbone repository is the authority for all of
it, and ``../inc/1a_definitions.gms`` is where Backbone declares the sets:

- the ``PARAM_*`` lists are its parameter tables, restricted to what this project
  actually writes. Backbone has more, and which more is measured per sheet in
  docs/identified-gaps.md rather than left as this sentence;
- the ``PARAM_*_DEFAULTS`` entries are the ``Default`` column of those same
  tables, for the parameters whose default is not 0;
- ``UC_DIMENSION_COLUMNS`` and ``UC_UNUSED_DIMENSION`` are the ``p_userconstraint``
  section: four selector slots, and the literal ``'-'`` an unused one must carry.

Nothing here is read from that file. Naming it is what makes deriving these
values from it a later pass rather than a rewrite -- when the builder took its
current shape the dictionary carried neither the defaults nor the slot contract,
which is why the lists grew inside the builder in the first place.
"""

#: p_gnu_io(grid, node, unit, input_output, param_gnu). Note that a connection
#: suffix (_input1 .. _output5) may be appended to any of these in a source
#: workbook; see build_unit_grid_and_node_columns.
PARAM_GNU = [
    'isActive',
    'capacity',
    'conversionCoeff',
    'useInitialGeneration',
    'initialGeneration',
    'maxRampUp',
    'maxRampDown',
    'rampPenalty',
    'rampUpPenalty',
    'rampDownPenalty',
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
    'cv',
]

#: p_unit(unit, param_unit). Unit-level, so a connection suffix on one of these
#: is an error rather than a refinement.
PARAM_UNIT = [
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
    'becomeUnavailable',
]

#: p_gnn(grid, from_node, to_node, param_gnn).
PARAM_GNN = [
    'isActive',
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

#: p_gn(grid, node, param_gn).
PARAM_GN = [
    'isActive',
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
    'storageValueUseTimeseries',
    'influx',
    'price',
]

#: The boundary types this project writes. Values of the param_gnBoundaryTypes
#: dimension of p_gnBoundaryPropertiesForStates, not columns of it -- Backbone
#: declares upwardSlack01*20 and downwardSlack01*20 as well, and a mod may add
#: more, so nothing here is a closed set.
PARAM_GN_BOUNDARY_TYPES = [
    'upwardLimit',
    'downwardLimit',
    'reference',
    'balancePenalty',
    'maxSpill',
    'downwardSlack01',
]

#: The parameter block of p_gnBoundaryPropertiesForStates. The boundary *types*
#: above are dimension values on that sheet; these are its columns.
#: 'multiplier' is deliberately absent: this project does not write it.
PARAM_GN_BOUNDARY_PROPERTIES = [
    'useConstant',
    'constant',
    'useTimeseries',
    'slackCost',
]

#: The numeric columns of the two source frames that carry no PARAM_* block of
#: their own. Without these, nothing coerced them and a stray string reached
#: inputData.xlsx as a text cell -- see BBExcelPipeline._coerce_numeric_dtypes.
PARAM_EMISSION = [
    'price',
]

PARAM_USERCONSTRAINT = [
    'value',
]

#: Emission factor columns on nodedata are named by their emission rather than
#: listed: create_p_nEmission derives the emission from the suffix of every
#: 'emission_XX' column it finds, so the set is open-ended by design.
EMISSION_COLUMN_PREFIX = 'emission_'

# --- Defaults -------------------------------------------------------------
#
# Only the parameters whose Backbone default is not 0. Applied inside each
# create_*() rather than during dtype coercion, so that they reach every row
# whichever source contributed it.

#: Applied per-connection in create_p_gnu_io.
PARAM_GNU_DEFAULTS = {
    'isActive':        1,
    'conversionCoeff': 1,
}

#: Applied per-unit in create_p_unit.
PARAM_UNIT_DEFAULTS = {
    'isActive':     1,
    'availability': 1,
    'eff00':        1,
    'op00':         1,
}

#: Applied in create_p_gnn after building the output DataFrame.
PARAM_GNN_DEFAULTS = {
    'isActive':     1,
    'availability': 1,
}

#: Applied in create_p_gn after building the output DataFrame.
PARAM_GN_DEFAULTS = {
    'isActive': 1,
}

# --- p_userconstraint slots -----------------------------------------------

#: p_userconstraint(group, uc1, uc2, uc3, uc4, param_userconstraint): the four
#: uc slots are optional selectors, but an unused one must carry the literal
#: '-' rather than a blank. inc/1e_inputs.gms aborts the run otherwise -- it
#: checks per parameter type that the slots it does not use are sameAs '-',
#: e.g. "should be '-' for <param> multiplier: (grid, node, '-', '-')".
UC_DIMENSION_COLUMNS = ['1st dimension', '2nd dimension', '3rd dimension', '4th dimension']
UC_UNUSED_DIMENSION = '-'
