$ontext
* --------------------------------------------------------------------------------
* PreprocessTimeseries.gms for the Northern European model
* --------------------------------------------------------------------------------

This file is created for Northern European Model (https://github.com/vttresearch/north_european_model)
and should be placed in the Backbone input folder, see instructions from 'Running Backbone' section
of the Northern European Model readme.

Contents
- General settings
- Set and parameter definitions for timeseries preprocessing
    - Sets, parameters, and scalars
    - Default values for command line parameters if not given
- reading input excel to get a list of grids, nodes, etc
- ts_cf
    - reading onshore, offshore, and PV data
    - creating ts_cf
    - adding missing forecasts when necessary
- ts_influx
    - reading elec data
    - reading hydro data
    - reading heat data
    - reading generation limit data
    - reading other demand data
    - creating ts_influx
    - adding missing forecasts when necessary
- ts_node
    - reading upwardLimit and downwardLimit data
    - reading historical values (reference)
    - creating ts_node
    - adding missing forecasts when necessary
- manual adjustments
    - FR00 and SE04 reservoir influx and downwardLimit adjustments
- write outputs



NOTE: additional project specific modifications are safest to add to the very end.

$offtext


* --------------------------------------------------------------------------------
* General settings
* --------------------------------------------------------------------------------

* Activate end of line comments and set comment character to '//'
$oneolcom
$eolcom //


* --------------------------------------------------------------------------------
* Set and parameter definitions for timeseries preprocessing
* --------------------------------------------------------------------------------

* --- Sets, parameters, and scalars --------------

Sets
    grid(*) "Forms of energy endogenously presented in the model"
    node(*) "Nodes maintain the energy balance or track exogenous commodities"
    unit(*) "Set of generators, storages and loads"
    gn_tmp(grid, node) "temporary (grid, node) set used in the input data manipulation"

    flow(*) "Flow based energy resources (time series)"
    flowNode_tmp(flow, node) "temporary (flow, node) set used in the input data manipulation"

    s "samples" / s000 /
    f "forecasts" / f00 * f03 /
    t "timesteps" / t000000 * t400000 /
    t_selectedYear(t) "Time steps selected by user in --tsYear and --modelledDays command line parameters"
    year "available model time series years" / y0001 * y2022 /

    param_gnBoundaryTypes "Types of boundaries that can be set for a node with balance" /
        upwardLimit      "Absolute maximum state of the node (MWh, unless modified by energyStoredPerUnitOfState parameter)"
        downwardLimit    "Absolute minimum energy in the node (MWh, unless modified by energyStoredPerUnitOfState parameter)"
        reference        "Reference value for a state that can be used to bound a state (MWh, unless modified by energyStoredPerUnitOfState parameter)"
    /

;

Parameters
    ts_influx_io(grid, f, t, node) "External power inflow/outflow during a time step (MWh/h) (reordered)"
    ts_cf_io(flow, f, t, node) "Available capacity factor time series (p.u. reordered)"
    ts_node_io(grid, param_gnBoundaryTypes, f, t, node) "Set node limits according to time-series form exogenous input (reordered)"

    p_yearStarts(year) "First time steps of available years"

    ts_influx(grid, node, f, t)
    ts_cf(flow, node, f, t)
    ts_node(grid, node, param_gnBoundaryTypes, f, t)
;

Scalars
    firstT "first t of the selected ts year"
;


* --- Default values for command line parameters if not given ------

// set values global to be available also when reading other files
$if not set modelYear $setglobal modelYear 2025
$if not set tsYear $setglobal tsYear 2015
$if not set forecasts $evalglobal forecastNumber 4
$if set forecasts $evalglobal forecastNumber %forecasts%

$if not set input_dir $setglobal input_dir input
$if not set input_file_gdx $setglobal input_file_gdx 'inputData.gdx'
$if not set input_excel_index $setglobal input_excel_index 'INDEX'
$if not set input_excel_checkdate $setglobal input_excel_checkdate ''



* --- Listing first time steps of each year and populating selectedYear --------------

// list the first time steps of the modelled years
p_yearStarts('y2004') = 192841;
p_yearStarts('y2005') = 201625;
p_yearStarts('y2006') = 210385;
p_yearStarts('y2007') = 219145;
p_yearStarts('y2008') = 227905;
p_yearStarts('y2009') = 236689;
p_yearStarts('y2010') = 245449;
p_yearStarts('y2011') = 254209;
p_yearStarts('y2012') = 262993;
p_yearStarts('y2013') = 271753;
p_yearStarts('y2014') = 280513;
p_yearStarts('y2015') = 289273;
p_yearStarts('y2016') = 298057;
p_yearStarts('y2017') = 306817;
p_yearStarts('y2018') = 315577;
p_yearStarts('y2019') = 324337;

// if user has given tsYear, compile t_selectedYear
if(%tsYear%>1980,
    // picking the first time step of the selected year to firstT variable
    firstT = sum(year $ (ord(year) = %tsYear%), p_yearStarts(year));

    // Determine time steps in the selected year to construct a year starting at t000001
    t_selectedYear(t)${ ord(t) >= (firstT + 1)
                        and ord(t) <= (firstT + 8760)
                      }
        = yes;
);


* --------------------------------------------------------------------------------
* reading input excel to get a list of grids, nodes, etc
* --------------------------------------------------------------------------------

* --- Converting %input_file_excel% to %input_file_gdx% --------------

// proceeding only if additional input data excel exists
$ifthen exist '%input_dir%/%input_file_excel%'
    $$call 'gdxxrw Input="%input_dir%/%input_file_excel%" Output="%input_dir%/%input_file_gdx%" Index=%input_excel_index%! %input_excel_checkdate%'
$elseif set input_file_excel

    $$abort 'Did not find input data excel from the given location, check path and spelling!'
$else
    $$abort 'Timeseries preprocessing needs --input_file_excel=<filename> to work!'
$endif

$ifthen exist '%input_dir%/bb_input_addData1.xlsx'
    $$call 'gdxxrw Input="%input_dir%/bb_input_addData1.xlsx" Output="%input_dir%/bb_input_addData1.gdx" Index=%input_excel_index%! %input_excel_checkdate%'
$endif

$ifthen exist '%input_dir%/bb_input_addData2.xlsx'
    $$call 'gdxxrw Input="%input_dir%/bb_input_addData2.xlsx" Output="%input_dir%/bb_input_addData2.gdx" Index=%input_excel_index%! %input_excel_checkdate%'
$endif

// reading only selected few tables required for time series preprocessing
$ifthen exist '%input_dir%/%input_file_gdx%'
    // load input data
    $$gdxin  '%input_dir%/%input_file_gdx%'
    $$loaddcm grid
    $$loaddcm node
    $$loaddcm flow
    $$loaddcm unit
    $$gdxin
$endif

$ifthen exist '%input_dir%/bb_input_addData1.gdx'
    // load input data
    $$gdxin  '%input_dir%/bb_input_addData1.gdx'
    $$loaddcm grid
    $$loaddcm node
    $$loaddcm flow
    $$loaddcm unit
    $$gdxin
$endif

$ifthen exist '%input_dir%/bb_input_addData2.gdx'
    // load input data
    $$gdxin  '%input_dir%/bb_input_addData2.gdx'
    $$loaddcm grid
    $$loaddcm node
    $$loaddcm flow
    $$loaddcm unit
    $$gdxin
$endif







* --------------------------------------------------------------------------------
* ts_cf
* --------------------------------------------------------------------------------

* onshore, offshore, and PV are all in the same files

* --- realization, ts_cf_io ----------------------------------

$call 'csv2gdx %input_dir%/bb_ts_cf_io.csv id=ts_cf_io index=1,2,3 values=(4..LastCol) useHeader=y output="%input_dir%/bb_ts_cf_io.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort  csv2gdx failed.

* Load ts_cf_io
$ifthen exist '%input_dir%/bb_ts_cf_io.gdx'
    $$gdxin  '%input_dir%/bb_ts_cf_io.gdx'
    $$loaddc ts_cf_io
    $$gdxin
$endif


* --- forecasts, ts_cf_io ------------------------------------

$ifThenE %forecastNumber%>1
    // convert csv to gdx with correct id
    $$call 'csv2gdx.exe %input_dir%/bb_ts_cf_io_50p.csv id=ts_cf_io index=1,2,3 values=(4..LastCol) useHeader=y output="%input_dir%/bb_ts_cf_io_50p.gdx" checkDate=TRUE'
    // Load ts_cf_io
    $$gdxin  '%input_dir%/bb_ts_cf_io_50p.gdx'
    $$loaddcm ts_cf_io
    $$gdxin
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_cf_io_50p failed! Check that your input file is valid and that your file path and file name are correct.

$ifThenE %forecastNumber%>3
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_cf_io_10p.csv id=ts_cf_io index=1,2,3 values=(4..LastCol) useHeader=y output="%input_dir%/bb_ts_cf_io_10p.gdx" checkDate=TRUE'
    // Load ts_cf_io
    $$gdxin  '%input_dir%/bb_ts_cf_io_10p.gdx'
    $$loaddcm ts_cf_io
    $$gdxin
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_cf_io_10p failed! Check that your input file is valid and that your file path and file name are correct.

$ifThenE %forecastNumber%>3
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_cf_io_90p.csv id=ts_cf_io index=1,2,3 values=(4..LastCol) useHeader=y output="%input_dir%/bb_ts_cf_io_90p.gdx" checkDate=TRUE'
    // Load ts_cf_io
    $$gdxin  '%input_dir%/bb_ts_cf_io_90p.gdx'
    $$loaddcm ts_cf_io
    $$gdxin
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_cf_io_90p failed! Check that your input file is valid and that your file path and file name are correct.


* --- creating ts_cf -------------------------------------

// convert ts_cf_io to ts_cf

// -1 as t000000 is the first one, but we want to map to t000001
firstT = firstT-1 ;

// pick only selected values if modelling a specific times series year
ts_cf(flow, node, f, t - firstT) = ts_cf_io(flow, f, t, node)$t_selectedYear(t) ;

// clear the large ts table
option clear = ts_cf_io;


* --- adding missing forecasts if necessary ---------

// if 2 forecasts, scheduleInit activates forecasts only when there is data in f01

// if 4 forecasts, make sure that f02 and f03 data exists when there is f01 data
if(%forecastNumber% >= 4,
    option flowNode_tmp < ts_cf;

    // copying f00 values to f02 and f03 if values are not given in earlier steps
    loop(flowNode_tmp(flow, node),
        if(sum(t, ts_cf(flow, node, 'f02', t))=0 and sum(t, ts_cf(flow, node, 'f01', t))>0,
            ts_cf(flow, node, 'f02', t) = ts_cf(flow, node, 'f01', t);
        );
        if(sum(t, ts_cf(flow, node, 'f03', t))=0 and sum(t, ts_cf(flow, node, 'f01', t))>0,
            ts_cf(flow, node, 'f03', t) = ts_cf(flow, node, 'f01', t);
        );
    );
);





* --------------------------------------------------------------------------------
* ts_influx
* --------------------------------------------------------------------------------

* --- realization, elec, ts_influx_io ---------

$call 'csv2gdx %input_dir%/bb_ts_influx_elec.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y output="%input_dir%/bb_ts_influx_elec.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort gdxxrw addition input failed! Check that your input Excel is valid and that your file path and file name are correct.

* Load ts_influx_io
$ifthen exist '%input_dir%/bb_ts_influx_elec.gdx'
    $$gdxin  '%input_dir%/bb_ts_influx_elec.gdx'
    $$loaddc ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif

* --- forecasts, elec, ts_influx_io ----------

$ifThenE %forecastNumber%>1
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_elec_50p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y output="%input_dir%/bb_ts_influx_elec_50p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_elec_50p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_elec_50p failed! Check that your input file is valid and that your file path and file name are correct.

$ifThenE %forecastNumber%>3
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_elec_10p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y output="%input_dir%/bb_ts_influx_elec_10p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_elec_10p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_elec_10p failed! Check that your input file is valid and that your file path and file name are correct.

$ifThenE %forecastNumber%>3
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_elec_90p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_influx_elec_90p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_elec_90p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_elec_90p failed! Check that your input file is valid and that your file path and file name are correct.



* --- realization, hydro, ts_influx_io ---------

$call 'csv2gdx %input_dir%/bb_ts_influx_hydro.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y output="%input_dir%/bb_ts_influx_hydro.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort gdxxrw ts_influx_io (hydro) failed! Check that your input Excel is valid and that your file path and file name are correct.

* Load ts_influx_io
$ifthen exist '%input_dir%/bb_ts_influx_hydro.gdx'
    $$gdxin  '%input_dir%/bb_ts_influx_hydro.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif

* --- forecasts, hydro, ts_influx_io ----------

$ifThenE %forecastNumber%>1
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_hydro_50p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y output="%input_dir%/bb_ts_influx_hydro_50p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_hydro_50p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_hydro_50p failed! Check that your input file is valid and that your file path and file name are correct.

$ifThenE %forecastNumber%>3
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_hydro_10p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y output="%input_dir%/bb_ts_influx_hydro_10p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_hydro_10p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_hydro_10p failed! Check that your input file is valid and that your file path and file name are correct.

$ifThenE %forecastNumber%>3
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_hydro_90p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y output="%input_dir%/bb_ts_influx_hydro_90p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_hydro_90p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_hydro_90p failed! Check that your input file is valid and that your file path and file name are correct.



* --- realization, heat, ts_influx_io ---------

$call 'csv2gdx %input_dir%/bb_ts_influx_heat_%modelYear%.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y valueDim=y output="%input_dir%/bb_ts_influx_heat_%modelYear%.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort gdxxrw ts_influx_io (heat) failed! Check that your input Excel is valid and that your file path and file name are correct.

* Load ts_influx_io
$ifthen exist '%input_dir%/bb_ts_influx_heat_%modelYear%.gdx'
    $$gdxin  '%input_dir%/bb_ts_influx_heat_%modelYear%.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif

* --- forecasts, heat, ts_influx_io ----------

$ifThenE %forecastNumber%>1
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_heat_%modelYear%_50p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y valueDim=y output="%input_dir%/bb_ts_influx_heat_%modelYear%_50p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_heat_%modelYear%_50p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_heat_%modelYear%_50p failed! Check that your input file is valid and that your file path and file name are correct.

$ifThenE %forecastNumber%>3
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_heat_%modelYear%_10p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y valueDim=y output="%input_dir%/bb_ts_influx_heat_%modelYear%_10p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_heat_%modelYear%_10p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_heat_%modelYear%_10p failed! Check that your input file is valid and that your file path and file name are correct.

$ifThenE %forecastNumber%>3
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_heat_%modelYear%_90p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y valueDim=y output="%input_dir%/bb_ts_influx_heat_%modelYear%_90p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_heat_%modelYear%_90p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_heat_%modelYear%_90p failed! Check that your input file is valid and that your file path and file name are correct.



* --- realization, generation limit nodes, ts_influx_io ---------

$call 'csv2gdx %input_dir%/bb_ts_influx_genlim.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y valueDim=y output="%input_dir%/bb_ts_influx_genlim.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort gdxxrw ts_influx_io (heat) failed! Check that your input Excel is valid and that your file path and file name are correct.

* Load ts_influx_io
$ifthen exist '%input_dir%/bb_ts_influx_genlim.gdx'
    $$gdxin  '%input_dir%/bb_ts_influx_genlim.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif

* --- forecasts, generation limit nodes, ts_influx_io ----------

* no forecasts



** --- realization, other demands, ts_influx_io ---------

$call 'csv2gdx %input_dir%/bb_ts_influx_other.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y valueDim=y output="%input_dir%/bb_ts_influx_other.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort gdxxrw ts_influx_io (other) failed! Check that your input Excel is valid and that your file path and file name are correct.

* Load ts_influx_io
$ifthen exist '%input_dir%/bb_ts_influx_other.gdx'
    $$gdxin  '%input_dir%/bb_ts_influx_other.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif

* --- forecasts, other demands, ts_influx_io ----------

* no forecasts



* --- converting ts_influx_io to ts_influx ---------------------

* convert ts_influx_io to ts_influx
$ifthen set convert_ts_influx
    if(%tsYear%>1980,
        // pick only selected values if modelling a specific times series year
        ts_influx(grid, node, f, t - firstT) = ts_influx_io(grid, f, t, node)$t_selectedYear(t);
        // clear the large ts table if modelling only a specific time series year
        option clear = ts_influx_io;
    else
        // otherwise rename the large table to the model input table
        ts_influx(grid, node, f, t) = ts_influx_io(grid, f, t, node);
    );
$endif


* --- adding missing forecasts if necessary ---------

// if 2 forecasts, scheduleInit activates forecasts only when there is data in f01

// if 4 forecasts, make sure that f02 and f03 data exists when there is f01 data
if(%forecastNumber% >= 4,
    option gn_tmp < ts_influx;

    // copying f01 values to f02 and f03 if f01 has values, but f02 or f03 does not
    loop(gn_tmp(grid, node),
        if(sum(t, ts_influx(grid, node, 'f02', t))=0 and sum(t, ts_influx(grid, node, 'f01', t))>0,
            ts_influx(grid, node, 'f02', t) = ts_influx(grid, node, 'f01', t);
        );
        if(sum(t, ts_influx(grid, node, 'f03', t))=0 and sum(t, ts_influx(grid, node, 'f01', t))>0,
            ts_influx(grid, node, 'f03', t) = ts_influx(grid, node, 'f01', t);
        );
    );
);





* --------------------------------------------------------------------------------
* ts_node
* --------------------------------------------------------------------------------

* --- realization, upwardLimits and downwardLimits, ts_node_io ---------

$call 'csv2gdx %input_dir%/bb_ts_node.csv id=ts_node_io index=1,2,3,4 values=(5..LastCol)  useHeader=y output="%input_dir%/bb_ts_node.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort csv2gdx ts_node_io (hydro downward and upward limits) failed! Check that your input file is valid and that your file path and file name are correct.

$ifthen exist '%input_dir%/bb_ts_node.gdx'
    $$gdxin  '%input_dir%/bb_ts_node.gdx'
    $$loaddc ts_node_io
    $$gdxin
$endif

* --- forecasts, upwardLimits and downwardLimits, ts_node_io  ----------

* no forecasts



* --- realization, reference, ts_node_io -----------------

// reading historical levels data reference.
// Used only to generate hydro storage starting levels then cleared.

$call 'csv2gdx %input_dir%/bb_ts_historical_levels.csv id=ts_node_io index=1,2,3,4 values=(5..LastCol) useHeader=y output="%input_dir%/bb_ts_historical_levels.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort csv2gdx ts_historical_levels (hydro) failed! Check that your input file is valid and that your file path and file name are correct.

$ifthen exist '%input_dir%/bb_ts_historical_levels.gdx'
    $$gdxin  '%input_dir%/bb_ts_historical_levels.gdx'
    $$loaddcm ts_node_io
    $$gdxin
$endif

* --- forecast, reference, ts_node_io ------------------

* no forecasts



* --- creating ts_node ----------------------------------

// convert ts_node_io to ts_node

// pick only selected values if modelling a specific times series year
ts_node(grid, node, param_gnBoundaryTypes, f, t-firstT) = ts_node_io(grid, param_gnBoundaryTypes, f, t, node)$t_selectedYear(t);

// clear the large ts table
option clear = ts_node_io;



* --- adding missing forecasts if necessary ---------

// no forecast data for hydro storage limits, but creating higher downwardLimit
// and lower upwardlimit to forecasts, helps the model to avoid over using hydro storages.

// Creating f01 (central) from realization by upwardlimit*0.98 and downwardLimit*1.02
// Creating f02 (low) from realization by upwardlimit*0.96 and downwardLimit*1.04
// Creating f03 (high) from realization by upwardlimit*1 and downwardLimit*1

if(%forecastNumber% >= 2,
    ts_node(grid, node, 'upwardLimit', 'f01', t)$ts_node(grid, node, 'upwardLimit', 'f00', t)
        = ts_node(grid, node, 'upwardLimit', 'f00', t) * 0.98;
    ts_node(grid, node, 'downwardLimit', 'f01', t)$ts_node(grid, node, 'downwardLimit', 'f00', t)
        = ts_node(grid, node, 'downwardLimit', 'f00', t) * 1.02;
);

if(%forecastNumber% >= 4,
    ts_node(grid, node, 'upwardLimit', 'f02', t)$ts_node(grid, node, 'upwardLimit', 'f00', t)
        = ts_node(grid, node, 'upwardLimit', 'f00', t) * 0.96;
    ts_node(grid, node, 'downwardLimit', 'f02', t)$ts_node(grid, node, 'downwardLimit', 'f00', t)
        = ts_node(grid, node, 'downwardLimit', 'f00', t) * 1.04;

    ts_node(grid, node, 'upwardLimit', 'f03', t)$ts_node(grid, node, 'upwardLimit', 'f00', t)
        = ts_node(grid, node, 'upwardLimit', 'f00', t) * 1;
    ts_node(grid, node, 'downwardLimit', 'f03', t)$ts_node(grid, node, 'downwardLimit', 'f00', t)
        = ts_node(grid, node, 'downwardLimit', 'f00', t) * 1;
);



* --------------------------------------------------------------------------------
* manual adjustments
* --------------------------------------------------------------------------------

* --- FR00 and SE04 reservoir influx and downwardLimit adjustments -------

// FR00 and SE04 reservoir nodes are particularly difficult to get dummy free with the current capacities and timeseries.
// As a patch, f02 influx is adjusted lower and f02 downwardLimit higher and than in other reservoirs.

ts_node('all', 'FR00_reservoir', 'downwardLimit', 'f02', t) = ts_node('all', 'FR00_reservoir', 'downwardLimit', 'f00', t) * 1.06;
ts_influx('all', 'FR00_reservoir', 'f02', t) = ts_influx('all', 'FR00_reservoir', 'f02', t) * 0.95;

if(%tsYear%=2011,
    ts_node('all', 'FR00_reservoir', 'downwardLimit', 'f02', t) = ts_node('all', 'FR00_reservoir', 'downwardLimit', 'f00', t) * 1.1;
    ts_influx('all', 'FR00_reservoir', 'f02', t) = ts_influx('all', 'FR00_reservoir', 'f02', t) * 0.9;
);

ts_node('all', 'SE04_reservoir', 'downwardLimit', 'f02', t) = ts_node('all', 'SE04_reservoir', 'downwardLimit', 'f00', t) * 1.06;
ts_influx('all', 'SE04_reservoir', 'f02', t) = ts_influx('all', 'SE04_reservoir', 'f02', t) * 0.95;

if(%tsYear%=2013,
    ts_node('all', 'SE04_reservoir', 'downwardLimit', 'f02', t) = ts_node('all', 'SE04_reservoir', 'downwardLimit', 'f00', t) * 1.1;
    ts_influx('all', 'SE04_reservoir', 'f02', t) = ts_influx('all', 'SE04_reservoir', 'f02', t) * 0.9;
);





* --------------------------------------------------------------------------------
* write outputs
* --------------------------------------------------------------------------------

// ts_cf.gdx
execute_unload '%input_dir%/ts_cf.gdx', ts_cf ;

// ts_influx.gdx
execute_unload '%input_dir%/ts_influx.gdx', ts_influx ;

// ts_node.gdx
execute_unload '%input_dir%/ts_node.gdx', ts_node ;
