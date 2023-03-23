using DataFrames, CSV
using Dates
using Interpolations
using Missings


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
    println(first(maf,6))

    #cast area
    maf = unstack(maf, :time, :area, :value)

    #correct leap years
    maf = correct_leapyear(maf)
    
    if isnothing(outfile)
        outfile = filename * "_byarea.csv"
    end

    # save results
    CSV.write(outfile, maf, dateformat="yyyy-mm-dd HH:MM:SS", missingstring = "NA")
   
end

# ---------------------------------------------------------------------------------


