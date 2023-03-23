using PyCall
using Statistics

"""
    convert_entsoe(includenodes)

Reads Entso-e unit and line data and prepares corresponding backbone input for Excel
...
# Arguments
- `includenodes::Array{String,1}`: list of entso-e format nodes for which data should be processed
...
"""
function convert_entsoe(includenodes,
                        filenames, 
                        options,
                        scenario = "National Trends", 
                        year = 2025)

    println("Preparing Backbone input file...")
    # formation of conversion units with their capacities 
    entsoecapa = readcapa(filenames["plantsourcefiles"], scenario, year)
    # select the units which reside in given nodes
    entsoecapa = subset(entsoecapa, :Node => ByRow(in(includenodes) ) )

    # read capacity filters which remove some units and remove the units
    capafilters = readcapafilters(filenames["plantsourcefiles"])
    entsoecapa = antijoin(entsoecapa, capafilters, on = [:Node, :Generator_ID])

    # reading fuel prices
    tspricechange = make_tspricechange(readfuelprices(filenames["fuelpricefile"], scenario, year) )

    # reading and preparing emission prices
    tsemissionpricechange = make_tsemissionpricechange(reademissionprices(filenames["fuelpricefile"], scenario, year) )

    # reading technology data for unit types and aggregated generator types
    techdata = readtechdata(filenames["techdatafile"])
    agg_gens = readagggens(filenames["techdatafile"])
    println("preparing aggregated generators...")
    #println(agg_gens)

    # disaggreagte aggregated units
    dcapa = disaggregate(entsoecapa, techdata, agg_gens)
   
    # add model names to units (generally "node_unitstub")
    ct = add_modelname(dcapa, filenames["unitdictfile"])

    #read flow types for unit types
    unitstubflow = readunitstubflow(filenames["unitdictfile"])

    # table pgnu_io
    pgnuio, nodetypes, capa_stoinput = make_pgnuio(ct, options)

    # table punit
    punit = make_punit(ct, options)

    # unitconstraint_node
    p_unitconstraintnode = make_unitconstraint_node(ct, pgnuio, nodetypes)

    # units
    units = select(punit, :unit)

    #pustartupfuel
    pusufuel = make_pusufuel(pgnuio, nodetypes)

    #unitflow
    flowunit = innerjoin(ct, unitstubflow, on = :unitstub)
    flowunit = select(flowunit, [:flow, :unit])

    #pgn
    pgn = make_pgn(nodetypes)
    
    # p_gnBoundaryPropertiesForStates
    p_bpfstates = make_boundarypropertiesforstates(capa_stoinput, nodetypes, Nothing)

    # efflevelgroupunit
    eflunit = make_eflunit(ct)

    #node
    allnodes = select(nodetypes, :node)

    # read line capacities and select relevant ones
    entsoelines = readalllines(filenames["linesourcefiles"], scenario, year)
    entsoelines = subset(entsoelines, :Startnode => ByRow(in(includenodes) ),
                                :Endnode => ByRow(in(includenodes) ) )
    pgnn = make_pgnn(entsoelines, options)
    
    println("Copying input file template...")

    # Copy Backbone input file template to be used as basis
    cp(filenames["bb_base_file"], filenames["bbinputfile"], force = true)

    println("Writing the input file...")

    #write the prepared tables to the Backbone input file
    XLSX.openxlsx(filenames["bbinputfile"], mode="rw") do xf
        
        writetable_and_clear(xf["p_gnu_io"], pgnuio, "B5")
        writetable_and_clear(xf["flowUnit"], flowunit, "B4")
          
        writetable_and_clear(xf["p_gn"], pgn, "B5")

        writetable_and_clear(xf["p_uStartupFuel"], pusufuel, "B5")
        
        writetable_and_clear(xf["node"], allnodes, "B4")
        writetable_and_clear(xf["unit"], units, "B4")
        writetable_and_clear(xf["effLevelGroupUnit"], eflunit, "B4")      

        writetable_and_clear(xf["p_gnn"], pgnn, "B5")
        writetable_and_clear(xf["p_unitConstraintNode"], p_unitconstraintnode, "B4")
        
        writetable_and_clear(xf["p_gnBoundaryPropertiesForStates"], p_bpfstates, "B5")
        
        writetable_and_clear(xf["p_unit"], punit, "B5")
        writetable_and_clear(xf["ts_PriceChange"], tspricechange, "B5") 
        writetable_and_clear(xf["ts_emissionPriceChange"], tsemissionpricechange, "B5") 
        
    end
end

"""
    convert_vre(includenodes)

Reads Entso-e format capacity factor files and converts them into backbone format
...
# Arguments
- `includenodes::Array{String,1}`: list of entso-e format nodes for which data should be processed
...
"""
function convert_vre(includenodes, filenames, options)

    # input directory
    a = filenames["timeseriesfolder"]

    # process original entso-e files
    println("Preprocessing VRE data...")
    process_maf(a * "PECD-MAF2019-wide-PV.csv", includenodes, a * "PECD-MAF2019-PV_byarea.csv")
    process_maf(a * "PECD-MAF2019-wide-WindOnshore.csv", includenodes, a * "PECD-MAF2019-WindOnshore_byarea.csv")
    process_maf(a * "PECD-MAF2019-wide-WindOffshore.csv", includenodes, a * "PECD-MAF2019-WindOffshore_byarea.csv")

    #load VRE data from processed entso-e files
    println("Loading VRE data...")
    cf = readVRE(a, options)

    #make the ts_cf_io table
    cf = make_tscfio(cf, timeorigin, includenodes)

    # write out csv and convert missing to zero
    println("Writing VRE data...")
    CSV.write(filenames["bbtimeseriesfile"], cf, dateformat="yyyy-mm-dd HH:MM:SS", missingstring = "0")

    # 10%,50%,90% measures
    println("Calculating statistics for VRE data...")
    @pyinclude("src/AverageVRE.py")
    py"conv_AverageVRE"(timeorigin, includenodes)

end

function make_other_demands(filenames,
                        scenario = "National Trends", 
                        year = 2025)

    # prepare process heat demand and defined other demands
    println("Preparing other demand data...")
    tsinfheat = make_dummy_ts_influx_heat(filenames, scenario, year, timeorigin)
    println("Writing other demand data...")
    CSV.write("output/bb_ts_influx_other.csv", tsinfheat, dateformat="yyyy-mm-dd HH:MM:SS", missingstring = "0")
end