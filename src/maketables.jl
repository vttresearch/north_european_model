using XLSX
using DataFrames
using Missings
using Printf
using Dates

"""
make_punit(capa, techdata)

Make the table for the p_unit parameter in backbone input
...
# Arguments
- `ct::DataFrame`: the table of units in processed with unit technical data entso-e format
...
"""
function make_punit(ct, options)

    #select columns in the right order
    if options["bb_version"] == "2.x" 
        #there are some additional startup constants in 2.x versions of Backbone
        punit = DataFrame(unit = ct.unit,
                        useTimeseries = 0,
                        outputCapacityTotal = 0,
                        availability = 1,
                        useInitialOnlineStatus = 0,
                        initialOnlineStatus = 0,
                        #shutdownCost = 0,
                        startCostCold = ct.startCostCold,
                        startFuelConsCold = ct.startFuelConsCold / 3.6,
                        rampSpeedToMinLoad = 0,
                        rampSpeedFromMinLoad = 0,
                        minOperationHours = ct.minOperationHours,
                        minShutDownHours = ct.minShutDownHours)
    else
        punit = DataFrame(unit = ct.unit,
                        useTimeseriesAvailability = coalesce.(ct.useTimeseriesAvailability,0),
                        useTimeseries = 0,
                        outputCapacityTotal = 0,
                        availability = 1,
                        useInitialOnlineStatus = 0,
                        initialOnlineStatus = 0,
                        shutdownCost = 0,
                        rampSpeedToMinLoad = 0,
                        rampSpeedFromMinLoad = 0,
                        minOperationHours = ct.minOperationHours,
                        minShutDownHours = ct.minShutDownHours)
        
    end
     
    if options["use_constrained_starttypes"] == false
        insertcols!(punit, :startWarmAfterXHours => 0)
        insertcols!(punit, :startColdAfterXHours => 0)

        if options["bb_version"] == "2.x"
            insertcols!(punit, :startCostHot => 0)
            insertcols!(punit, :startCostWarm => 0)
            insertcols!(punit, :startFuelConsHot => 0)
            insertcols!(punit, :startFuelConsWarm => 0)
        end

    else
        insertcols!(punit, :startWarmAfterXHours => ct.startWarmAfterXHours)
        insertcols!(punit, :startColdAfterXHours => ct.startColdAfterXhours)

        if options["bb_version"] == "2.x" 
            insertcols!(punit, :startCostHot => ct.startCostHot)
            insertcols!(punit, :startCostWarm => ct.startCostWarm)
            insertcols!(punit, :startFuelConsHot => ct.startFuelConsHot / 3.6)
            insertcols!(punit, :startFuelConsWarm => ct.startFuelConsWarm / 3.6)
        end
    end


    # efficiency                    
    insertcols!(punit, :section => 0)
    insertcols!(punit,	:opFirstCross => 0)
    insertcols!(punit, :eff00 => ct.efficiency)
    insertcols!(punit, :eff01 => ct.efficiency)
 
    for i in 2:12
        insertcols!(punit, Symbol("eff" * @sprintf("%02d",i)) => 0)
    end

    # operating points
    insertcols!(punit, :op00 => ct.minstablegen)
    insertcols!(punit, :op01 => 1)	
    for i in 2:12
        insertcols!(punit, Symbol("op" * @sprintf("%02d",i)) => 0)
    end

    #piecewise linear heat rates
    insertcols!(punit, :hrsection => 0)
    insertcols!(punit, :hr00 => 0) 
    insertcols!(punit, :hr01 => 0) 
    insertcols!(punit, :hr02 => 0) 
    insertcols!(punit, :hrop00 => 0) 
    insertcols!(punit, :hrop01 => 0) 
    insertcols!(punit, :hrop02 => 0) 

    #unit count, fixed or endogenous
    insertcols!(punit, :investMIP => 0)
    insertcols!(punit, :maxUnitCount => 0)
    insertcols!(punit, :minUnitCount => 0)
    
    #if given unitSize is 0, put also zero as the number of units
    ct = transform(ct, AsTable([:Elec_capa, :unitSize]) 
        => ByRow(x -> x[2] == 0 ? 0 : ceil(x[1] / x[2]) )  => :unitCount) 
    
    insertcols!(punit, :unitCount => ct.unitCount)		

    return punit
end

"""
    formatpgnuio(a, nodecol, capacityval::Number, bb_version; _conversionCoeff = 1)


Prepare a final format for part of p_gnu_io parameter for the purpose of unit outputs
...
# Arguments
- `a::DataFrame`: the table of units in processed entso-e format, joined with the techdata table
- `nodecol::Symbol`: the column where output node name sits
- `capacitycol::Symbol`: the column where capacity value sits
- `bb_version::String`: Backbone version
- `_conversionCoeff::Number`: constant which is send to the conversionCoeff column
...
"""
function formatpgnuio(a, nodecol, capacitycol::Symbol, options; _conversionCoeff = 1)
    
    b = formatpgnuio(a, nodecol, 1.0, options, _conversionCoeff = _conversionCoeff)
    b[:,:capacity] = a[:, capacitycol]     

    return b
end


"""
    formatpgnuio(a, nodecol, capacityval::Number, bb_version; _conversionCoeff = 1)


Prepare a final format for part of p_gnu_io parameter for the purpose of unit outputs
...
# Arguments
- `a::DataFrame`: the table of units in processed entso-e format, joined with the techdata table
- `nodecol::Symbol`: the column where output node name sits
- `capacityval::Number`: constant which is send to the capacity column
- `bb_version::String`: Backbone version
- `_conversionCoeff::Number`: constant which is send to the conversionCoeff column
...
"""
function formatpgnuio(a, nodecol, capacityval::Number, options; _conversionCoeff = 1)
    
    #if both conversionCoeff and capacity are zero, the flow would not be included in model
    if capacityval == 0 && _conversionCoeff == 0
        capacityval = 100000
    end

    b = DataFrame(grid = "all", 
                node = a[:, nodecol],
                unit = a.unit,
                input_output = "output",
                dummy = "",
                capacity = capacityval,
                useInitialGeneration = 0,	
                initialGeneration = 0,
                conversionCoeff = _conversionCoeff,
                maxRampUp = a.maxRampUp,
                maxRampDown = a.maxRampDown,
                upperLimitCapacityRatio = 0,
                unitSize = 0,
                invCosts = 0, 
                fomCosts = 0,	
                vomCosts = a.vomCosts,
                shutdownCost = 0)

    if options["bb_version"] ≠ "2.x"
        if options["use_constrained_starttypes"] == true
            insertcols!(b, :startCostHot => a.startCostWarm)
            insertcols!(b, :startCostWarm => a.startCostWarm)
            insertcols!(b, :startCostCold => a.startCostCold)
            insertcols!(b, :startFuelConsHot => a.startFuelConsHot / 3.6)
            insertcols!(b, :startfuelConsWarm => a.startfuelConsWarm / 3.6)
            insertcols!(b, :startFuelConsCold => a.startFuelConsCold / 3.6)
        else
            insertcols!(b, :startCostHot => 0)
            insertcols!(b, :startCostWarm => 0)
            insertcols!(b, :startCostCold => a.startCostCold)
            insertcols!(b, :startFuelConsHot => 0)
            insertcols!(b, :startfuelConsWarm => 0)
            insertcols!(b, :startFuelConsCold => a.startFuelConsCold / 3.6)
        end
        insertcols!(b, :annuity => 0)
    else
        insertcols!(b, :annuityFactor => 0)
    end

    return b
end

"""
    formatpgnuio(a, nodecol)


Prepare a final format for part of p_gnu_io parameter for the purpose of unit inputs
...
# Arguments
- `a::DataFrame`: the table of units in processed entso-e format, joined with the techdata table
- `nodecol::Symbol`: the column where input node name sits
...
"""
function formatpgnuio(a, nodecol, options)
 
    # input rows, arrange final format
    b = DataFrame(grid = "all", 
                    node = a[:, nodecol],
                    unit = a.unit,
                    input_output = "input",
                    dummy = "",
                    capacity = 0, 
                    useInitialGeneration = 0,	
                    initialGeneration = 0,
                    conversionCoeff = 1,
                    maxRampUp = 0,
                    maxRampDown = 0,
                    upperLimitCapacityRatio = 0,
                    unitSize = 0,
                    invCosts = 0, 
                    fomCosts = 0,	
                    vomCosts = 0,
                    shutdownCost = 0)

    if options["bb_version"] ≠ "2.x"
        insertcols!(b, :startCostHot => 0)
        insertcols!(b, :startCostWarm => 0)
        insertcols!(b, :startCostCold => 0)
        insertcols!(b, :startFuelConsHot => 0)
        insertcols!(b, :startfuelConsWarm => 0)
        insertcols!(b, :startFuelConsCold => 0)
        insertcols!(b, :annuity => 0)
    else
        insertcols!(b, :annuityFactor => 0)
    end

    return b
end


"""
    make_pgnuio(capa, techdata)


Make the table for the p_gnu_io parameter in backbone input
...
# Arguments
- `ct::DataFrame`: the table of units in processed entso-e format, joined with the techdata table
- `options::Dict`: the list of options
...
"""
function make_pgnuio(ct, options)
    pgnu_input, nodetypes_input, capa_stoinput = make_pgnuio_input(ct, options)
    pgnu_output, nodetypes_output = make_pgnuio_output(ct, options)

    #combine input and output rows
    pgnu = vcat(pgnu_input, pgnu_output)

    # combine node types
    # TBA: check conflicting node type declarations
    nodetypes = unique(vcat(nodetypes_input, nodetypes_output) )

    return pgnu, nodetypes, capa_stoinput
end

"""
    make_pgnuio_input(ct, options)


Make the table for the input rows of p_gnu_io parameter in backbone input
...
# Arguments
- `ct::DataFrame`: the table of units in processed entso-e format, joined with the techdata table
- `options::Dict`: the list of options
...
"""
function make_pgnuio_input(ct, options)
    #input rows of fuel-using units
    capa_conv = subset(ct, :inputtype => ByRow(==("fuel")))
    
    #deduce fuel names (remove spaces)
    capa_conv = transform(capa_conv, :Fuel => ByRow(x -> replace(x, " " => "")) => :inputnode) 

    #input rows of electricity-using units
    capa_elecinput = subset(ct, :inputtype => ByRow(==("elec")))
    capa_elecinput = transform(capa_elecinput, :Node => ByRow(x -> x * elec_suffix) => :inputnode)

    #input rows to units of hydro units
    capa_hydroinput = subset(ct, :inputtype => ByRow(==("hydro")))

    #format the input node name for hydro units, adding the regional node in the beginning
    capa_hydroinput = transform(capa_hydroinput, 
                    AsTable([:Node, :inputnodestub]) => ByRow(x -> x[1] * "_" * x[2]) => :inputnode)
                    
    #input rows to units of storage units of the bidding zone level
    capa_stoinput = subset(ct, :inputtype => ByRow(==("storage")))
    #format the input node name for storage units, adding the regional node in the beginning
    capa_stoinput = transform(capa_stoinput, 
                    AsTable([:Node, :inputnodestub]) => ByRow(x -> x[1] * "_" * x[2]) => :inputnode)

    #combine units with different input types 
    capa_1 = vcat(capa_conv, capa_elecinput, capa_hydroinput, capa_stoinput, cols = :union)

    # deduce backbone nodetypes
    nodetypes = DataFrame(node = capa_1.inputnode, type = capa_1.inputtype)  

    # input rows, arrange final format
    pgnu_input_1 = formatpgnuio(capa_1, :inputnode, options)

    return pgnu_input_1, nodetypes, capa_stoinput
end

"""
    make_pgnuio_output(ct, options)


Make the table for the output rows of p_gnu_io parameter in backbone input
...
# Arguments
- `ct::DataFrame`: the table of units in processed entso-e format, joined with the techdata table
- `options::Dict`: the list of options
...
"""
function make_pgnuio_output(ct, options)
    #go through specified outputs of units

    # units with electrical capacity
    capa_elecoutput = subset(ct, :Elec_capa => ByRow(!ismissing))
    capa_elecoutput = transform(capa_elecoutput, :Node => ByRow(x -> x * elec_suffix) => :outputnode)

    pgnu_output_1 = formatpgnuio(capa_elecoutput, :outputnode, :Elec_capa, options)
    nodetypes_output = DataFrame(node = capa_elecoutput.outputnode, type = "elec")

    # units with heat capacity 
    capa_heatoutput = subset(ct, :Heat_capa => ByRow(!ismissing))
    pgnu_output_2 = formatpgnuio(capa_heatoutput, :Heatnode, :Heat_capa, options)
    nodetypes_output = vcat(nodetypes_output, DataFrame(node = capa_heatoutput.Heatnode, type = "heat") )

    # units with custom output node which is not fuel
    # currently the node type is not set for these nodes
    capa_customoutput = subset(ct, :Other_capa => ByRow(!ismissing),
                                    :outputtype => ByRow(ismissing) )
    capa_customoutput = transform(capa_customoutput, AsTable([:Node, :outputnodestub]) 
                            => ByRow(x -> x[1] * "_" * x[2])  
                            => :outputnode)

    #units with custom output node which is fuel: node type is set for these nodes
    capa_customoutput2 = subset(ct, :Other_capa => ByRow(!ismissing),
                                :outputtype => ByRow(isequal("fuel") ) )
    
    insertcols!(capa_customoutput2, :outputnode => capa_customoutput2.outputnodestub)
    capa_customoutput = vcat(capa_customoutput, capa_customoutput2)                                           

    pgnu_output_3 = formatpgnuio(capa_customoutput, :outputnode, :Other_capa, options)
    nodetypes_output = vcat(nodetypes_output, DataFrame(node = capa_customoutput2.outputnode, type = "fuel") )

    # units with minimum generation limit (output to special mingen node)
    capa_mingenoutput = subset(ct, :Mingennode => ByRow(!ismissing))
    capa_mingenoutput = transform(capa_mingenoutput, AsTable([:Node, :Mingennode]) 
                            => ByRow(x -> x[1] * "_" * x[2])  
                            => :Mingennode)
    pgnu_output_4 = formatpgnuio(capa_mingenoutput, :Mingennode, 0, options, _conversionCoeff = 0)
    nodetypes_output = vcat(nodetypes_output, DataFrame(node = capa_mingenoutput.Mingennode, type = "mingen") )

    #combine output rows
    pgnu_output = vcat(pgnu_output_1, pgnu_output_2, pgnu_output_3, pgnu_output_4)

    return pgnu_output, nodetypes_output
end


"""
    make_unitconstraint_node(ct, pgnuio, nodetypes)


Make the table for the p_unitConstraintNode parameter in backbone input
...
# Arguments
- `ct::DataFrame`: the table of units in processed entso-e format, joined with the techdata table
- `pgnuio::DataFrame`: the table of units with their input and output nodes 
- `nodetypes::DataFrame`: the table of nodes and their types
...
"""
function make_unitconstraint_node(ct, pgnuio, nodetypes)

    #combine node type with input-output table
    pgnuio1 = innerjoin(pgnuio, nodetypes, on = :node)

    # equality constraints for cb units
    # extract the units where cb is defined
    cbunits = subset(ct, :cb => ByRow(!ismissing) )
    cbunits = select(cbunits, [:unit, :cb])
    #cbunits.constr = "eq" .* string.(axes(cbunits,1))
    insertcols!(cbunits, :constr => "eq1")

    # extract the units with electricity output (or input)
    pgnuio_e = subset(pgnuio1, :type => ByRow(==("elec")) )
    pgnuio_e = innerjoin(pgnuio_e, cbunits, on = :unit)

    # extract the units with heat output
    pgnuio_h = subset(pgnuio1, :type => ByRow(==("heat")) )
    pgnuio_h = innerjoin(pgnuio_h, cbunits, on = :unit)

    # format the unitconstraint_node table concerning the elec-heat relationship
    ucn_cb_e = DataFrame(unit = pgnuio_e.unit, constr = pgnuio_e.constr, 
                        node = pgnuio_e.node, value = 1)

    ucn_cb_h = DataFrame(unit = pgnuio_h.unit, constr = pgnuio_h.constr, 
                        node = pgnuio_h.node, value = -pgnuio_h.cb)

    # inequality constraints for units with minimum generation defined
    ucn_mingen_1, ucn_mingen_2 = make_unitconstraint_node_mingen(ct, pgnuio1)
    """
    mingenunits = select(subset(ct, :Mingennode => ByRow(!ismissing)), [:unit])
    insertcols!(mingenunits, :constr => "gt1")
    pgnuio_m = subset(pgnuio1, :type => ByRow(==("mingen")) )
    pgnuio_m = innerjoin(pgnuio_m, mingenunits, on = :unit)
    ucn_mingen_1 = DataFrame(unit = pgnuio_m.unit, constr = pgnuio_m.constr, 
                    node = pgnuio_m.node, value = -1)

    # the min gen is targeted to electricity generation
    pgnuio_e = subset(pgnuio1, :type => ByRow(==("elec")) )
    pgnuio_e = innerjoin(pgnuio_e, mingenunits, on = :unit)
    ucn_mingen_2 = DataFrame(unit = pgnuio_e.unit, constr = pgnuio_e.constr, 
                    node = pgnuio_e.node, value = 1)

    """
    return vcat(ucn_cb_e, ucn_cb_h, ucn_mingen_1, ucn_mingen_2)
end

"""
    make_unitconstraint_node_mingen(ct, pgnuio1)

Make the table for the p_unitConstraintNode parameter for minimum generation
...
# Arguments
- `ct::DataFrame`: the table of units in processed entso-e format, joined with the techdata table
- `pgnuio1::DataFrame`: the table of units with their input and output nodes, combined with nodetypes
...
"""
function make_unitconstraint_node_mingen(ct, pgnuio1)

    # inequality constraints for units with minimum generation defined
    mingenunits = select(subset(ct, :Mingennode => ByRow(!ismissing)), [:unit])
    insertcols!(mingenunits, :constr => "gt1")
    pgnuio_m = subset(pgnuio1, :type => ByRow(==("mingen")) )
    pgnuio_m = innerjoin(pgnuio_m, mingenunits, on = :unit)
    ucn_mingen_1 = DataFrame(unit = pgnuio_m.unit, constr = pgnuio_m.constr, 
                    node = pgnuio_m.node, value = -1)

    # the min gen is targeted to electricity generation
    pgnuio_e = subset(pgnuio1, :type => ByRow(==("elec")) )
    pgnuio_e = innerjoin(pgnuio_e, mingenunits, on = :unit)
    ucn_mingen_2 = DataFrame(unit = pgnuio_e.unit, constr = pgnuio_e.constr, 
                    node = pgnuio_e.node, value = 1)

    return ucn_mingen_1, ucn_mingen_2
end

function make_boundarypropertiesforstates(capa_stoinput, nodetypes, upwardlimit)
    #ask hydro reservoirs to follow ts_node limits
    bps_hydro = subset(nodetypes, :type => ByRow(==("hydro")))
    insertcols!(bps_hydro, :grid => "all")
    insertcols!(bps_hydro, :useTimeseries => 1)
    insertcols!(bps_hydro, :useConstant => 0)
    insertcols!(bps_hydro, :constant => 0)
    bps_hydro = crossjoin(bps_hydro, DataFrame(param_gnboundarytypes = ["upwardLimit", "downwardLimit"]) )
  
    # storage node limits
    # !notice that useConstant should not be set for storages which enable investments!

    bps_sto = subset(nodetypes, :type => ByRow(==("storage")))
     # calculate storage upper limit by multiplying sum output capacity by the relative capacity 
    temp = transform(capa_stoinput, 
                    AsTable([:Elec_capa, :Heat_capa, :Other_capa, :storage_rel_capa])
                    => ByRow(x -> sum(skipmissing([x[1], x[2], x[3]] ) ) * x[4] ) => :storage_abs_capa)
                 
    temp = combine(groupby(temp, :inputnode), :storage_abs_capa => sum => :storage_abs_capa)

    insertcols!(bps_sto, :grid => "all")
    insertcols!(bps_sto, :useTimeseries => 0)
    insertcols!(bps_sto, :useConstant => 1)
    bps_sto_d = copy(bps_sto) #make a copy for the downward limit

    # upward limits
    insertcols!(bps_sto, :param_gnboundarytypes => "upwardLimit")
    bps_sto = innerjoin(bps_sto, temp, on = :node => :inputnode)
    rename!(bps_sto, :storage_abs_capa => :constant)

    # downward limits
    insertcols!(bps_sto_d, :param_gnboundarytypes => "downwardLimit")
    insertcols!(bps_sto_d, :constant => 0)
    bps_sto = vcat(bps_sto, bps_sto_d)

    # prepare the final table in Backbone format
    bps = vcat(bps_hydro, bps_sto)
    bps = select(bps, [:grid, :node, :param_gnboundarytypes, :useTimeseries, :useConstant, :constant])

    return bps
  
end

function make_pusufuel(pgnuio, nodetypes)
    
    temp = subset(pgnuio, :input_output => ByRow(==("input")))
    temp = innerjoin(temp, nodetypes, on = :node)
    temp = subset(temp, :type => ByRow(==("fuel")))

    pusufuel = DataFrame(unit = temp.unit,
                    node = temp.node,
                    dummy = "",
                    fixedFuelFraction = 1.0
                    )
end

function make_eflunit(ct)
    #temp = subset(pgnuio, :input_output => ByRow(==("input")))
    #temp = innerjoin(temp, nodetypes, on = :node)

    # for the moment all units use directoff
    #eflunit = DataFrame(efflevel = "level1",
    #                    effselector = "directOff",
    #                    unit = temp.unit
    #                )

    temp = DataFrame(efflevel = ["level1", "level2"],
                        effselector = ["directOff","directOff"])

    
    ct = subset(ct, :inputtype => ByRow(!=("flow"))) 
    

    eflunit = DataFrame(unit = ct.unit)
    eflunit = crossjoin(temp, eflunit)
    eflunit = select(eflunit, [:efflevel, :effselector, :unit])    
               
end


"""
    make_pgn(nodetypes)

Make the table for the pgn parameter in backbone input
...
# Arguments
- `nodetypes::DataFrame`: the table of node - nodetype
...
"""
function make_pgn(nodetypes)

    # default values for all nodes
    pgn = DataFrame(grid = "all", node = nodetypes.node,
                dummy = "",
                selfDischargeLoss = 0,
                energyStoredPerUnitOfState = 0,	boundStart= 0,	boundStartAndEnd = 0,
                boundEnd = 0, boundAll = 0,	boundStartToEnd	= 0, nodeBalance = 0,
                capacityMargin = 0, storageValueUseTimeSeries = 0, usePrice = 0
                )
    
    #set node balance 
    pgn[nodetypes.type .== "elec", :nodeBalance] .= 1
    pgn[nodetypes.type .== "heat", :nodeBalance] .= 1
    pgn[nodetypes.type .== "hydro", :nodeBalance] .= 1
    pgn[nodetypes.type .== "mingen", :nodeBalance] .= 1
    pgn[nodetypes.type .== "storage", :nodeBalance] .= 1

    #node can store energy (may vary)
    pgn[nodetypes.type .== "hydro", :energyStoredPerUnitOfState] .= 1
    pgn[nodetypes.type .== "storage", :energyStoredPerUnitOfState] .= 1

    #node contents have price for fuels
    pgn[nodetypes.type .== "fuel", :usePrice] .= 1

    return pgn

#	


end

"""
    make_pgnn(entsoelines)

Make the table for the pgnn parameter in backbone input
...
# Arguments
- `entsoelines::DataFrame`: the table of lines in processed entso-e format
...
"""
function make_pgnn(entsoelines, options)

    #export capacity forms a set of lines in the input file
    e1 = subset(entsoelines, 
                :Parameter => ByRow(==("Export Capacity")), 
                )

    #check whether ramp limits have been defined for this set of capacities
    rl = subset(entsoelines, 
            :Parameter => ByRow(==("Ramp Limit")), 
            )
    select!(rl, :Startnode, :Endnode, :Value => :rampLimit1)

    # join ramp limits with export capacity data to match rows
    e1 = leftjoin(e1, rl, on = [:Startnode, :Endnode])
    e1.rampLimit1 = coalesce.(e1.rampLimit1, 0)

    # part of final table related to export capacity
    pgnn_1 = DataFrame(grid = "all", 
                        from_node = e1.Startnode .* "_elec",
                        to_node = e1.Endnode .* "_elec",
                        dummy = "",
                        transferCap = e1.Value,
                        rampLimit = e1.rampLimit1
                    )


    #import capacity forms another set of lines in the input file
    e2 = subset(entsoelines, 
                    :Parameter => ByRow(==("Import Capacity")), 
                    )
    
    # join ramp limits with import capacity data to match rows
    e2 = leftjoin(e2, rl, on = [:Startnode, :Endnode])
    e2.rampLimit1 = coalesce.(e2.rampLimit1, 0)

    pgnn_2 = DataFrame(grid = "all", 
                            from_node = e2.Endnode .* "_elec",
                            to_node = e2.Startnode .* "_elec",
                            dummy = "",
                            transferCap = -1 * e2.Value,
                            rampLimit = e2.rampLimit1
                        )
    
    pgnn = vcat(pgnn_1, pgnn_2)

    insertcols!(pgnn, :transferCapBidirectional => 0)
    insertcols!(pgnn, :transferLoss => 0.02)
    insertcols!(pgnn, :diffCoeff => 0)
    insertcols!(pgnn, :boundStateOffset => 0)
    insertcols!(pgnn, :boundStateMaxDiff => 0)
    insertcols!(pgnn, :transferCapInvLimit => 0)
    insertcols!(pgnn, :investMIP => 0)
    insertcols!(pgnn, :unitSize => 0)
    insertcols!(pgnn, :invCost => 0)
    insertcols!(pgnn, :portion_of_transfer_to_reserve => 0)
    insertcols!(pgnn, :variableTransCost => 0)
    insertcols!(pgnn, :availability => 1)
		
    # interconnection ramp limits depend on Backbone version
    if options["bb_version"] == "2.x"
        insertcols!(pgnn, :ICrampUp => 0)
        insertcols!(pgnn, :ICrampDown => 0)
        insertcols!(pgnn, :annuity => 0)
    else
        #insertcols!(pgnn, :rampLimit => 0)
        insertcols!(pgnn, :annuityFactor => 0)
    end

    return pgnn 									 
end

function make_tspricechange(fuelprices)

    # select columns
    tspricechange = DataFrame(fuel = fuelprices.fuel,
                            t000000 = fuelprices.price)

    return tspricechange
end

function make_tsemissionpricechange(emissionprices)

    # select columns
    tsemissionpricechange = DataFrame(emission = emissionprices.emission,
    Group = "emissionTaxGroup",
    t000001 = emissionprices.price)

    return tsemissionpricechange
end