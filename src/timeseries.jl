using Dates
using DataFrames, CSV
using Interpolations
using Missings
using Printf

function correct_leapyear(x)

    leap_years = [i for i in 1982:2017 if isleapyear(i)]

    for y in leap_years
        #shift all days after 28.2 backwards by 1 day
        x[(year.(x.time) .== y) .& (month.(x.time) .>= 3), :time] .-= Day(1)
        
        #interpolate values for 31.12. for each value column
        valuecols = filter(x-> x ∉ ["time"], names(x))
        timepoints = collect(DateTime(y,12,31,0):Hour(1):DateTime(y,12,31,23) )
        a = DataFrame(time = timepoints)

        #interpolate each value column
        for v in valuecols
            # check if 1.1. of next year exists
            if DateTime(y+1,1,1,0) ∈ x.time
                valend = x[x.time .== DateTime(y+1,1,1,0), v]
                valend = disallowmissing(valend)[1]
            else
                valend = 0
            end

            valbegin = x[x.time .== DateTime(y,12,30,23), v]
            valbegin = disallowmissing(valbegin)[1]
            #interpolate
            myinterp = LinearInterpolation(Dates.datetime2unix.([DateTime(y,12,30,23), DateTime(y+1,1,1,0)]), 
                                    [valbegin, valend])

            insertcols!(a, v => myinterp(Dates.datetime2unix.(timepoints) ) )                         
          
        end

        #add the interpolated values to x
        #duplicate values should not exist because time stamps were shifted backwards by 1 day
        x = vcat(x,a)
    end
    
    sort!(x, :time)

    return x
end


"""
    process_maf(filename, selected_areas)

the function processes the original MAF VRE data 
(e.g. PECD-MAF2019-wide-WindOnshore.csv) by casting the
area index and producing time stamp and correcting leap year data.

"""
function process_maf(filename, selected_areas, outfile = nothing)
    
    println("Processing file" * filename)

    # proceed with ENTSO-E historical data
    maf = DataFrame(CSV.File(filename, delim = ";", decimal=',', missingstring = "NA"))

    #select areas (selecting only a subset considerably reduces the memory requirement)
    maf = subset(maf, :area => ByRow(in(selected_areas)) )

    # melt years
    maf = stack(maf, Not([:area, :day, :month, :hour]))
    maf[!,:year] = parse.(Int64, maf[!, :variable])
    maf = select(maf, Not(:variable))

    # calculate timestamp as datetime
    maf = transform(maf, AsTable([:year, :month, :day, :hour]) => ByRow(x -> DateTime(x[1], x[2], x[3], x[4] - 1)  ) => :time)
    println(Base.summarysize(maf))

    #cast area
    maf = unstack(maf, :time, :area, :value)

    #correct leap years
    maf = correct_leapyear(maf)

    # Round the values in the DataFrame
    maf[:, Not(:time)] .= round.(maf[:, Not(:time)], digits=4)
    
    if isnothing(outfile)
        outfile = filename * "_byarea.csv"
    end

    # save results
    CSV.write(outfile, maf, dateformat="yyyy-mm-dd HH:MM:SS", missingstring = "NA")
   
end



function formattime(x, timeorigin)
    bb_t = convert(Dates.Hour, x - timeorigin)  / Hour(1) + 1
    bb_t = "t" * @sprintf("%06d",bb_t)
end

"""
    make_tscfio(cf, timeorigin, includenodes)

Make the table for the ts_cf_io parameter in backbone input
...
# Arguments
- `cf::DataFrame`: the table of capacity factors in processed entso-e format
- `timeorigin::DateTime`: the time of t000000
- `includenodes::Array{String,1}`: the entso-e nodes to be included
...
"""
function make_tscfio(cf, timeorigin, includenodes)

    #select those time stamps which are after time origin
    cf = subset(cf, :time => ByRow(>=(timeorigin)) )
    
    #convert time into backbone time index
    cf = transform(cf, :time
        => ByRow(x -> (formattime(x, timeorigin) ) )
        => :bb_t)

    # select specified entso-e nodes
    tscfio = select(cf, includenodes)

    # convert entso-e node names into backbone electricity nodes
    temp = Dict(s => s * "_elec" for s in includenodes)
    rename!(tscfio, temp)

    insertcols!(tscfio, 1, :flow => cf.flow)
    insertcols!(tscfio, 2, :f => "f00")
    insertcols!(tscfio, 3, :t => cf.bb_t)

    return tscfio
end


"""
    make_dummy_ts_influx_heat(timeorigin)

Make the table for the ts_influx_other parameter concerning some heat or other loads
...
# Arguments
- `timeorigin::DateTime`: the time of t000000
...
"""
function make_dummy_ts_influx_heat(filenames, scenario, year, timeorigin)

    # determine the time span for the demand time series, begins from timeorigin of course
    endtime = DateTime(2020,12,31)
    time = collect(timeorigin:Hour(1):endtime)
    bbtime = formattime.(time, timeorigin)

    # read constant demands for certain nodes
    dem = readdemands(filenames["demandfile"], scenario, year)

    # create the demand timeseries table and fill with some initial data
    # FI00industry: industry process heat
    demand = DataFrame(grid = "all", f = "f00", t = bbtime, 
                    #FI00industry = -2600,
                    #SE03_industry = -900
                    )
 
    # insert all specified constant demands to the demand timeseries table
    for row in eachrow(dem)
        insertcols!(demand, row["inputnode"] => -row["Value"])
    end

    return demand
end

"""
    augment_series(a, b)

the function augments dataframe timeseries in cross-tabulated format with another

"""
function augment_series(a, b)

    #check which data columns are present
    updatecols = names(b)
    deleteat!(updatecols, findfirst(isequal("time"), updatecols))
  
    # delete the columns to be updated from a and join
    a = select(a, Not(updatecols) )
    a = innerjoin(a, b, on = :time)

    return a
end
    
