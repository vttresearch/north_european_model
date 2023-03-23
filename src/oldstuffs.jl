"""
    make_pgnuio(capa, techdata)

Make the table for the p_gnu_io parameter in backbone input
    ...
    # Arguments
    - `ct::DataFrame`: the table of units in processed entso-e format, joined with the techdata table
    - `options::Dict`: the list of options
    ...
    """
    function make_pgnuio_old(ct, options)
    
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
                        
        #input rows to units of storage units
        capa_stoinput = subset(ct, :inputtype => ByRow(==("storage")))
        #format the input node name for storage units, adding the regional node in the beginning
        capa_stoinput = transform(capa_stoinput, 
                        AsTable([:Node, :inputnodestub]) => ByRow(x -> x[1] * "_" * x[2]) => :inputnode)
    
        #combine units with different input types 
        capa_1 = vcat(capa_conv, capa_elecinput, capa_hydroinput, capa_stoinput, cols = :union)
    
        # deduce backbone nodetypes
        nodetypes = DataFrame(node = capa_1.inputnode, type = capa_1.inputtype)  
    
        # input rows, arrange final format
        pgnu_input_1 = DataFrame(grid = "all", 
                                node = capa_1.inputnode,
                                unit = capa_1.unit,
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
                                annuity = 0,
                                fomCosts = 0,	
                                vomCosts = 0)
                                                       
        #go through specified outputs of units
    
        # units with electrical capacity
        capa_elecoutput = subset(ct, :Elec_capa => ByRow(!ismissing))
        capa_elecoutput = transform(capa_elecoutput, :Node => ByRow(x -> x * elec_suffix) => :outputnode)
    
        pgnu_output_1 = formatpgnuio(capa_elecoutput, :outputnode, :Elec_capa)
        nodetypes_output = DataFrame(node = capa_elecoutput.outputnode, type = "elec")
    
        # units with heat capacity 
        capa_heatoutput = subset(ct, :Heat_capa => ByRow(!ismissing))
        pgnu_output_2 = formatpgnuio(capa_heatoutput, :Heatnode, :Heat_capa)
        nodetypes_output = vcat(nodetypes_output, DataFrame(node = capa_heatoutput.Heatnode, type = "heat") )
    
        # units with custom output node
        # currently the node type is not set for these nodes
        capa_customoutput = subset(ct, :Other_capa => ByRow(!ismissing))
        capa_customoutput = transform(capa_customoutput, AsTable([:Node, :outputnodestub]) 
                                => ByRow(x -> x[1] * "_" * x[2])  
                                => :outputnode)
        pgnu_output_3 = formatpgnuio(capa_customoutput, :outputnode, :Other_capa)
    
        # units with minimum generation limit (output to special mingen node)
        capa_mingenoutput = subset(ct, :Mingennode => ByRow(!ismissing))
        capa_mingenoutput = transform(capa_mingenoutput, AsTable([:Node, :Mingennode]) 
                                => ByRow(x -> x[1] * "_" * x[2])  
                                => :Mingennode)
        pgnu_output_4 = formatpgnuio(capa_mingenoutput, :Mingennode, 0, _conversionCoeff = 0)
        nodetypes_output = vcat(nodetypes_output, DataFrame(node = capa_mingenoutput.Mingennode, type = "mingen") )
    
        #combine input and output rows
        pgnu = vcat(pgnu_input_1, pgnu_output_1, pgnu_output_2, pgnu_output_3, pgnu_output_4)
    
        # combine node types
        nodetypes = unique(vcat(nodetypes, nodetypes_output) )
    
        #return the p_gnu table, nodetypes table and a list of storage discharging units with their inputnodes
        return pgnu, nodetypes, capa_stoinput
    end
    