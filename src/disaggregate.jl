using DataFrames
using Missings
using Printf

"""
    make_ct(capa, techdata, agg_gens)


Combine aggregated generator types with their disaggregated components
...
# Arguments
- `capa::DataFrame`: the table of units in processed entso-e format
- `techdatal::DataFrame`: the table of unit technical data in processed entso-e format
- `agg:gens::DataFrame`: the table which ties aggregated generator types with their disaggregated components
...
"""
function disaggregate(capa, techdata, agg_gens)

    #add technology data to nonaggregated generator types
    ct = innerjoin(capa, techdata, on = :Generator_ID)

    # process aggregated generator ids
    temp = innerjoin(capa, agg_gens, on = :Generator_ID => :Aggregate_generator)
    rename!(temp, :Generator_ID => :Aggregate_generator)

    temp = innerjoin(temp, techdata, on = :Subgenerator_ID => :Generator_ID)
    rename!(temp, :Subgenerator_ID => :Generator_ID)

    # copy total aggregated capacity to temporary column
    temp = transform(temp, AsTable([:Elec_capa, :Heat_capa, :Other_capa])
                => ByRow(x -> sum(skipmissing([x[1], x[2], x[3]])) )
                => :temp_capa)

    #temp.temp_capa = temp.Elec_capa

    # deduce output capacities elec, heat, other
    temp = transform(temp, AsTable([:temp_capa, :elec_capashare]) 
                => ByRow(x -> x[1] * x[2])  
                => :Elec_capa)
            
    temp = transform(temp, AsTable([:temp_capa, :heat_capashare]) 
                => ByRow(x -> x[1] * x[2])  
                => :Heat_capa)

    temp = transform(temp, AsTable([:temp_capa, :other_capashare]) 
                => ByRow(x -> x[1] * x[2])  
                => :Other_capa)

    temp = select(temp, Not(:temp_capa))
    
    ct = vcat(ct, temp, cols = :union)


    return ct
end

"""
    add_modelname(capa, unitdictfile)


Create unit modelnames to be used in Backbone model
...
# Arguments
- `capa::DataFrame`: the table of units in processed entso-e format
- `unitdictfile::String`: the filename of the file where modelnames of units exist

"""
function add_modelname(capa, unitdictfile)

     #load stub model unit names and connect
     #columns, labels = XLSX.readtable(unitdictfile, "unitdict")
     unitdict = DataFrame(XLSX.readtable(unitdictfile, "unitdict") )
     unitdict.unitstub = replace(unitdict.unitstub, "NA" => missing)
     unitdict = dropmissing(unitdict)
     
     capa = innerjoin(capa, unitdict, on = :Generator_ID)
 
     #create the model unit names
     capa = transform(capa, 
                     AsTable([:Node, :Heatnode, :unitstub]) 
                     => ByRow(x -> "U_" * (ismissing(x[2]) ? x[1] : x[2]) * "_" * x[3]) 
                     => :unit)
end
                     