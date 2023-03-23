using XLSX
using DataFrames

"""
    writetable_and_clear(sheet, df, _anchor_cell)


Writes a dataframe into excel sheet, clears the area before that
...
# Arguments
- `sheet::Worksheet`: the excel worksheet
- `df::dataframe`: the dataframe
- `_anchor_cell::String`: top left corner of the area
...
"""
function writetable_and_clear(sheet, df, _anchor_cell)

    # creates empty columns and labels and writes them
    columns, labels = create_empty_columns()
    XLSX.writetable!(sheet, columns, labels, 
                    anchor_cell=XLSX.CellRef(_anchor_cell) ) 

    # writes the data
    XLSX.writetable!(sheet, 
                    collect(DataFrames.eachcol(df)), 
                    DataFrames.names(df), 
                    anchor_cell=XLSX.CellRef(_anchor_cell) ) 
        
end

function create_empty_columns(n = 500, m = 100)

    columns = Vector()
    labels = ["" for j in 1:m]

    for j in 1:m
        push!(columns, ["" for i in 1:n])
    end
    
    return columns, labels
end
