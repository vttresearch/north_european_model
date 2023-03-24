using XLSX
using CSV
using DataFrames
using Missings
using Statistics

"""
    readcapa(plantsourcefiles, scenario, year)

Reads all capacity files for plant types and capacities and filters them
    according to given scenario and year.
...
# Arguments
- `plantsourcefiles::Array{String,1}`: list of file names of capacity files
- `scenario::String`: scenario name
- `year::Int`: year
...
"""
function readcapa(plantsourcefiles, scenario, year)

    capa = nothing
    
    #reading all the given files and combining
    for f in plantsourcefiles
        columns, labels = XLSX.readtable(f, "Capacity")
        newinput = DataFrame(columns, labels)

        #filter the capacities
        newinput  = subset(newinput , 
            :Scenario => ByRow(==(scenario)),
            :Year => ByRow(==(year))  )

        # set the combined table of capacities
        #a = findfirst(isequal(f), plantsourcefiles)

        if findfirst(isequal(f), plantsourcefiles) == 1
            capa = newinput
        else
            # remove those unit types in previous data which have been redefined in new input
            capa = antijoin(capa, newinput, on = [:Node, :Generator_ID])
            # add newly read capacities
            capa = vcat(capa, newinput, cols = :union)
        end

    end

    return capa
end

"""
    readcapafilters(plantsourcefiles)

Reads the unit types which should be removed from the input data.
...
# Arguments
- `plantsourcefiles::Array{String,1}`: list of file names of capacity files

...
"""
function readcapafilters(plantsourcefiles)
   
    filters = nothing

    for f in plantsourcefiles
        temp = DataFrame(Node = [], Generator_ID = [])
        try
            columns, labels = XLSX.readtable(f, "Remove_units_by_node")
            temp = DataFrame(columns, labels)
        catch
            nothing
        end


        if findfirst(isequal(f), plantsourcefiles) == 1
            filters = temp
        else
            filters = vcat(filters, temp, cols = :union)
        end
    end

    return filters
end

function readtechdata(techdatafile)
    #reading the whole sheet
    columns, labels = XLSX.readtable(techdatafile, "techdata")
    techdata = DataFrame(columns, labels)
end

function readagggens(techdatafile)
    #reading the whole sheet
    columns, labels = XLSX.readtable(techdatafile, "aggregates")
    agg_gens = DataFrame(columns, labels)
end


function readalllines(linesourcefiles, scenario, year)

    lines = nothing
    #reading all the given files and combining
    for f in linesourcefiles
        temp = readentsoelines(f, scenario, year)

        if isnothing(lines)
            lines = temp
        else
            # remove those line data in previous data which have been redefined in new input
            lines = antijoin(lines, temp, on = [:Startnode, :Endnode, :Parameter])
            # add newly read capacities
            lines = vcat(lines, temp, cols = :union)
        end

    end

    return lines
end

function readentsoelines(linesourcefile, scenario, year)
    #reading the whole sheet of entsoe lines
    columns, labels = XLSX.readtable(linesourcefile, "Line")
    entsoelines = DataFrame(columns, labels)

    #rename some cols
    rename!(entsoelines, Symbol("Node/Line") => :Line)

    #filter the capacities
    entsoelines = subset(entsoelines, 
                        :Scenario => ByRow(==(scenario)), 
                        :Year => ByRow(==(year)),
                        :Case => ByRow(==("Reference Grid")),
                        Symbol("Climate Year") => ByRow(==(1984)) )

    #get the start and end nodes    
    entsoelines = transform(entsoelines, :Line => (ByRow(x -> String.(split(x,"-")) ) ) => [:Startnode, :Endnode])

end

function readunitstubflow(unitdictfile)
    #reading the whole sheet s
    columns, labels = XLSX.readtable(unitdictfile, "unitflow")
    unitstubflow = DataFrame(columns, labels)
end

function readVRE(inputdir, options)

    #updated PV file
    filename_PV_update = "PECD_2021_PV_byarea.csv"
    filename_onshore_update = "PECD_2021_WindOnshore_byarea.csv"
    filename_offshore_update = "PECD_2021_WindOffshore_byarea.csv"

    # onshore wind
    maf = DataFrame(CSV.File(inputdir * "PECD-MAF2019-WindOnshore_byarea.csv", dateformat="yyyy-mm-dd HH:MM:SS", missingstring = "NA"))
    insertcols!(maf, 1, :flow => "onshorewind")

    #Pan-European wind and solar generation time series (PECD 2021 update) updates to onshore wind
    #Data requires a Preprocessing step which is not yet included in this repository
    if options["use_updated_VRE_series"] == true
        maf_b = DataFrame(CSV.File(inputdir * filename_onshore_update, dateformat="yyyy-mm-dd HH:MM:SS", missingstring = "NA"))
        updatecols = names(maf_b)
        deleteat!(updatecols, findfirst(isequal("time"), updatecols))
        maf = select(maf, Not(updatecols) )
        maf = innerjoin(maf, maf_b, on = :time)
    end 

    # PV
    maf2 = DataFrame(CSV.File(inputdir * "PECD-MAF2019-PV_byarea.csv", dateformat="yyyy-mm-dd HH:MM:SS", missingstring = "NA"))
    insertcols!(maf2, 1, :flow => "PV")
    #Pan-European wind and solar generation time series (PECD 2021 update) updates to onshore wind
    #Data requires a Preprocessing step which is not yet included in this repository
    if options["use_updated_VRE_series"] == true
        b = DataFrame(CSV.File(inputdir * filename_PV_update, dateformat="yyyy-mm-dd HH:MM:SS", missingstring = "NA"))
        maf2 = augment_series(maf2, b)
    end

    # offshore wind
    maf3 = DataFrame(CSV.File(inputdir * "PECD-MAF2019-WindOffshore_byarea.csv", dateformat="yyyy-mm-dd HH:MM:SS", missingstring = "NA"))
    insertcols!(maf3, 1, :flow => "offshorewind")
    if options["use_updated_VRE_series"] == true
        b = DataFrame(CSV.File(inputdir * filename_offshore_update, dateformat="yyyy-mm-dd HH:MM:SS", missingstring = "NA"))
        maf3 = augment_series(maf3, b)
    end

    maf = vcat(maf, maf2, maf3, cols = :union)

    return maf
end

function readfuelprices(filename, scenario, year)

    #reading the whole sheet 
    columns, labels = XLSX.readtable(filename, "fuelprices")
    fp = DataFrame(columns, labels)

    #filter the capacities
    fp = subset(fp, 
                :Scenario => ByRow(==(scenario)),
                :Year => ByRow(==(year))  )

end

function reademissionprices(filename, scenario, year)

    #reading the whole sheet 
    columns, labels = XLSX.readtable(filename, "emissionprices")
    fp = DataFrame(columns, labels)
 
    #filter the capacities
    fp = subset(fp, 
                :Scenario => ByRow(==(scenario)),
                :Year => ByRow(==(year))  )

end

function readdemands(demandfile, scenario, year)
    #reading the whole sheet of entsoe lines
    columns, labels = XLSX.readtable(demandfile, "demand")
    dem = DataFrame(columns, labels)

    #filter the demands
    dem = subset(dem, 
                :Scenario => ByRow(==(scenario)),
                :Year => ByRow(==(year))  )

    #create the model unit names
    dem = transform(dem, 
        AsTable([:Node, :Inputnodestub]) 
        => ByRow(x -> x[1] * "_" * (ismissing(x[2]) ? elec_suffix : x[2]) ) 
        => :inputnode)

end

function listfiles(dirname)
    # read filenames in the directory (without path)
    a = readdir(dirname)
    # select those with suitable pattern
    a = filter(x->occursin(".txt",x), a)

    return sort(a)
end