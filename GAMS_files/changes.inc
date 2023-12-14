$ontext
* --------------------------------------------------------------------------------
* Changes.inc for the Northern European model
* --------------------------------------------------------------------------------

This file is created for Northern European Model (https://github.com/vttresearch/north_european_model)
and should be placed in the Backbone input folder, see instructions from 'Run specification files' section
of the Northern European Model readme.

Contents 
- Set and parameter definitions
    - New sets, parameters, and scalars
    - default values for command line parameters if not given
    - populating gn_hydroStorage manually
- Reading additional input data
    - bb_input_addData1.xlsx
- ts_cf
    - reading onshore, offshore, and PV data
    - creating ts_cf
    - adding missing forecasts when necessary
- ts_influx
    - reading elec data
    - reading hydro data
    - reading heat data
    - reading generation limit data
    - creating ts_influx
    - adding missing forecasts when necessary
- ts_node
    - reading upwardLimit and downwardLimit data
    - creating upwardLimits and downwardLimits forecasts
    - reading historical values -> reference ts
    - creating ts_node
    - creating hydro storage start levels, clearing reference ts
    - adding missing forecasts when necessary
- additional manual changes 
    - applying price multipliers
    - converting selected units to onlineLP, onlineMIP
    - clearing redundant values from p_gnu_io, p_unit and rounding remaining values
    - project specific modifications


NOTE: additional project specific modifications are safest to add to the very end.

$offtext



* --------------------------------------------------------------------------------
* Set and parameter definitions
* --------------------------------------------------------------------------------

* --- New sets, parameters, and scalars --------------

Sets
t_selectedYear(t) "Time steps selected by user in --tsYear and --modelledDays command line parameters"
year "available model time series years" / y0001 * y2022 /
gn_tmp(grid, node) "temporary (grid, node) set used in the input data manipulation"
flowNode_tmp(flow, node) "temporary (flow, node) set used in the input data manipulation"
gn_hydroStorage(grid, node) "helper set listing hydro nodes with storage"
;

Parameters
ts_influx_io(grid, f, t, node) "External power inflow/outflow during a time step (MWh/h) (reordered)"
ts_cf_io(flow, f, t, node) "Available capacity factor time series (p.u. reordered)"
ts_node_io(grid, param_gnBoundaryTypes, f, t, node) "Set node limits according to time-series form exogenous input (reordered)"
p_yearStarts(year) "First time steps of available years"
;

*define parameters for time checking
scalar now       Gregorian current date + time
       hour      hour,
       minute    minute,
       second    second;


* --- default values for command line parameters if not given ------

// set values global to be available also when reading other files
$if not set modelYear $setglobal modelYear 2025
$if not set tsYear $setglobal tsYear 2015
$if not set modelledDays $setglobal modelledDays 365
$if not set forecasts $setglobal forecastNumber 4
$if set forecasts $setglobal forecastNumber %forecasts%
$if not set priceMultiplier $setglobal priceMultiplier 1


* --- populating gn_hydroStorage manually -------------

// gn_hydroStorage(grid, node) = yes; // capacity, normal year inflow, calculated FLH

gn_hydroStorage('all', 'BE00_ror') = yes;  // 0.13 GW, 0.35 TWh, 2655 FLH
gn_hydroStorage('all', 'DE00_ror') = yes;  // 4.0 GW, 15.1 TWh, 3780 FLH
gn_hydroStorage('all', 'ES00_ror') = yes;  // 3.5 GW, 6.6 TWh, 1880 FLH
gn_hydroStorage('all', 'FR00_ror') = yes;  // 13.6 GW, 36.7 TWh, 2700 FLH
gn_hydroStorage('all', 'LT00_ror') = yes;  // 0.13 GW, 0.51 TWh, 3890 FLH
gn_hydroStorage('all', 'LV00_ror') = yes;  // 1.6 GW, 2.2 TWh, 1370 FLH
gn_hydroStorage('all', 'NL00_ror') = yes;  // 0.04 GW, 0.10 TWh, 2530 FLH
gn_hydroStorage('all', 'PL00_ror') = yes;  // 0.54 GW, 1.3 TWh, 2360 FLH
gn_hydroStorage('all', 'UK00_ror') = yes;  // 1.9 GW, 3.96 TWh, 2080 FLH

gn_hydroStorage('all', 'DE00_psOpen') = yes;  // 1.5 GW, 0.41 TWh, 275 FLH
gn_hydroStorage('all', 'ES00_psOpen') = yes;  // 1.7 GW, 0.35 TWh, 200 FLH
gn_hydroStorage('all', 'FR00_psOpen') = yes;  // 1.7 GW, 0 TWh   -> practically psClosed
gn_hydroStorage('all', 'NOM1_psOpen') = yes;  // 4.9 GW, 18.75 TWh, 3830 FLH
gn_hydroStorage('all', 'NON1_psOpen') = yes;  // 5.5 GW, 23.3 TWh, 4240 FLH
gn_hydroStorage('all', 'NOS0_psOpen') = yes;  // 24 GW, 88.6 TWh, 3690 FLH
gn_hydroStorage('all', 'PL00_psOpen') = yes;  // 0.15 GW, 0.07 TWh, 460 FLH

gn_hydroStorage('all', 'DE00_reservoir') = yes;  // 1.3 GW, 0.95 TWh, 730 FLH
gn_hydroStorage('all', 'ES00_reservoir') = yes;  // 6.4 GW, 16.5 TWH, 2580 FLH
gn_hydroStorage('all', 'FI00_reservoir') = yes;  // 2.4 GW, 13.5 TWh, 5630 FLH
gn_hydroStorage('all', 'FR00_reservoir') = yes;  // 8.0 GW, 16.3 TWh, 2040 FLH
gn_hydroStorage('all', 'PL00_reservoir') = yes;  // 0.20 GW, 0.24 TWh, 1220 FLH
gn_hydroStorage('all', 'SE01_reservoir') = yes;  // 5.3 GW, 19.8 TWh, 3730 FLH
gn_hydroStorage('all', 'SE02_reservoir') = yes;  // 6.8 GW, 36.8 TWh, 5410 FLH
gn_hydroStorage('all', 'SE03_reservoir') = yes;  // 2.0 GW, 10.4 TWh, 5200 FLH
gn_hydroStorage('all', 'SE04_reservoir') = yes;  // 0.24 GW, 1.4 TWh, 5680 FLH

gn_hydroStorage('all', 'BE00_psClosed') = yes;  // 1.4 GW, 0 TWh
gn_hydroStorage('all', 'DE00_psClosed') = yes;  // 8.4 GW, 0 TWh
gn_hydroStorage('all', 'ES00_psClosed') = yes;  // 4.2 GW, 0 TWh
gn_hydroStorage('all', 'FR00_psClosed') = yes;  // 1.8 GW, 0 TWh
gn_hydroStorage('all', 'LT00_psClosed') = yes;  // 0.9 GW, 0 TWh
gn_hydroStorage('all', 'PL00_psClosed') = yes;  // 1.3 GW, 0 TWh !!
gn_hydroStorage('all', 'UK00_psClosed') = yes;  // 3.0 GW, 0 TWh


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
    // picking the first time step of the selected year to tmp variable
    tmp = sum(year $ (ord(year) = %tsYear%), p_yearStarts(year));

    // Determine time steps in the selected year to construct a year starting at t000001
    t_selectedYear(t)${ ord(t) >= (tmp + 1)
                        and ord(t) <= (tmp + 8760)
                      }
        = yes;
);



* --------------------------------------------------------------------------------
* Reading additional input data
* --------------------------------------------------------------------------------

* --- bb_input_addData1.xlsx ----------------------------

// proceeding only if additional input data excel exists
$ifthen exist '%input_dir%/bb_input_addData1.xlsx'

// Convert additional excel input
$call 'gdxxrw Input="%input_dir%/bb_input_addData1.xlsx" Output="%input_dir%/bb_input_addData1.gdx" index=index!A1'
$ife %system.errorlevel%>0 $abort gdxxrw failed!

    // load input data
    $$gdxin  '%input_dir%/bb_input_addData1.gdx'
    $$loaddcm grid
    $$loaddcm node
    $$loaddcm flow
    $$loaddcm unittype
    $$loaddcm unit
    $$loaddcm unitUnittype
    $$loaddcm unit_fail
    $$loaddcm unitUnitEffLevel
    $$loaddcm effLevelGroupUnit
    $$loaddcm group
    $$loaddcm p_gn
    $$loaddcm p_gnn
    $$loaddcm p_gnu_io
    $$loaddcm p_gnuBoundaryProperties
    $$loaddcm p_unit
    $$loaddcm ts_unit
    $$loaddcm p_unitConstraint
    $$loaddcm p_unitConstraintNode
    $$loaddcm restype
    $$loaddcm restypeDirection
    $$loaddcm restypeReleasedForRealization
    $$loaddcm restype_inertia
    $$loaddcm p_groupReserves
    $$loaddcm p_groupReserves3D
    $$loaddcm p_groupReserves4D
    $$loaddcm p_gnuReserves
    $$loaddcm p_gnnReserves
    $$loaddcm p_gnuRes2Res
    $$loaddcm ts_reserveDemand
    $$loaddcm p_gnBoundaryPropertiesForStates
    $$loaddcm p_uStartupfuel
    $$loaddcm flowUnit
    $$loaddcm emission
    $$loaddcm p_nEmission
*    $$loaddcm ts_cf      // ts_cf, ts_influx, and ts_node constructed below. Cannot add content there like this.
    $$loaddcm ts_priceChange
*    $$loaddcm ts_influx
*    $$loaddcm ts_node
    $$loaddcm p_s_discountFactor
    $$loaddcm t_invest
    $$loaddcm utAvailabilityLimits
    $$loaddcm p_storageValue
    $$loaddcm ts_storageValue
    $$loaddcm uGroup
    $$loaddcm gnuGroup
    $$loaddcm gn2nGroup
    $$loaddcm gnGroup
    $$loaddcm sGroup
    $$loaddcm p_groupPolicy
    $$loaddcm p_groupPolicyEmission
    $$loaddcm p_groupPolicyUnit
    $$loaddcm gnss_bound
    $$loaddcm uss_bound
    $$gdxin
$endif



* --------------------------------------------------------------------------------
* ts_cf
* --------------------------------------------------------------------------------

* onshore, offshore, and PV are all in the same files

* --- realization, ts_cf_io ----------------------------------

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_cf_io.csv id=ts_cf_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_cf_io.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort  csv2gdx failed.

* Load ts_cf_io
$ifthen exist '%input_dir%/bb_ts_cf_io.gdx'
    $$gdxin  '%input_dir%/bb_ts_cf_io.gdx'
    $$loaddc ts_cf_io
    $$gdxin
$endif

* --- forecasts, ts_cf_io ------------------------------------

$ifthen set forecasts
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_cf_io_50p.csv id=ts_cf_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_cf_io_50p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_cf_io_50p.gdx'
    $$loaddcm ts_cf_io
    $$gdxin
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_cf_io_50p failed! Check that your input file is valid and that your file path and file name are correct.

$ifthen %forecastNumber% == 4
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_cf_io_10p.csv id=ts_cf_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_cf_io_10p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_cf_io_10p.gdx'
    $$loaddcm ts_cf_io
    $$gdxin
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_cf_io_10p failed! Check that your input file is valid and that your file path and file name are correct.

$ifthen %forecastNumber% == 4
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_cf_io_90p.csv id=ts_cf_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_cf_io_90p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_cf_io_90p.gdx'
    $$loaddcm ts_cf_io
    $$gdxin
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_cf_io_90p failed! Check that your input file is valid and that your file path and file name are correct.


* --- creating ts_cf -------------------------------------

* display time
now    = jnow;
hour   = ghour(now);
minute = gminute(now);
second = gsecond(now);
display  hour, minute, second;

* convert ts_cf_io to ts_cf
if(%tsYear%>1980,
    tmp = tmp-1 ;  // -1 as t000000 is the first one, but we want to map to t000001
    // pick only selected values if modelling a specific times series year
    ts_cf(flow, node, f, t - tmp) = ts_cf_io(flow, f, t, node)$t_selectedYear(t) ;
    // clear the large ts table if modelling only a specific time series year
    option clear = ts_cf_io;
else
    // otherwise rename the large table to the model input table
    ts_cf(flow, node, f, t) = ts_cf_io(flow, f, t, node);
);


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

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_influx_elec.csv id=ts_influx_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_influx_elec.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort gdxxrw addition input failed! Check that your input Excel is valid and that your file path and file name are correct.

* Load ts_influx_io
$ifthen exist '%input_dir%/bb_ts_influx_elec.gdx'
    $$gdxin  '%input_dir%/bb_ts_influx_elec.gdx'
    $$loaddc ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif

* --- forecasts, elec, ts_influx_io ----------

$ifthen set forecasts
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_elec_50p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_influx_elec_50p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_elec_50p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_elec_50p failed! Check that your input file is valid and that your file path and file name are correct.

$ifthen %forecastNumber% == 4
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_elec_10p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_influx_elec_10p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_elec_10p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_elec_10p failed! Check that your input file is valid and that your file path and file name are correct.

$ifthen %forecastNumber% == 4
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

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_influx_hydro.csv id=ts_influx_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_influx_hydro.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort gdxxrw ts_influx_io (hydro) failed! Check that your input Excel is valid and that your file path and file name are correct.

* Load ts_influx_io
$ifthen exist '%input_dir%/bb_ts_influx_hydro.gdx'
    $$gdxin  '%input_dir%/bb_ts_influx_hydro.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif

* --- forecasts, hydro, ts_influx_io ----------

$ifthen set forecasts
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_hydro_50p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_influx_hydro_50p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_hydro_50p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_hydro_50p failed! Check that your input file is valid and that your file path and file name are correct.

$ifthen %forecastNumber% == 4
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_hydro_10p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_influx_hydro_10p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_hydro_10p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_hydro_10p failed! Check that your input file is valid and that your file path and file name are correct.

$ifthen %forecastNumber% == 4
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_hydro_90p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_influx_hydro_90p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_hydro_90p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_hydro_90p failed! Check that your input file is valid and that your file path and file name are correct.



* --- realization, heat, ts_influx_io ---------

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_influx_heat_%modelYear%.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y valueDim=y output="%input_dir%/bb_ts_influx_heat_%modelYear%.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort gdxxrw ts_influx_io (heat) failed! Check that your input Excel is valid and that your file path and file name are correct.

* Load ts_influx_io
$ifthen exist '%input_dir%/bb_ts_influx_heat_%modelYear%.gdx'
    $$gdxin  '%input_dir%/bb_ts_influx_heat_%modelYear%.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif

* --- forecasts, heat, ts_influx_io ----------

$ifthen set forecasts
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_heat_%modelYear%_50p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y valueDim=y output="%input_dir%/bb_ts_influx_heat_%modelYear%_50p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_heat_%modelYear%_50p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_heat_%modelYear%_50p failed! Check that your input file is valid and that your file path and file name are correct.

$ifthen %forecastNumber% == 4
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_influx_heat_%modelYear%_10p.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y valueDim=y output="%input_dir%/bb_ts_influx_heat_%modelYear%_10p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_influx_heat_%modelYear%_10p.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_influx_heat_%modelYear%_10p failed! Check that your input file is valid and that your file path and file name are correct.

$ifthen %forecastNumber% == 4
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

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_influx_genlim.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y valueDim=y output="%input_dir%/bb_ts_influx_genlim.gdx" checkDate=TRUE'
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



* --- creating ts_influx -------------------------------------

* display time
now    = jnow;
hour   = ghour(now);
minute = gminute(now);
second = gsecond(now);
display  hour, minute, second;

* convert ts_influx_io to ts_influx
$ifthen set convert_ts_influx
    if(%tsYear%>1980,
        // pick only selected values if modelling a specific times series year
        ts_influx(grid, node, f, t - tmp) = ts_influx_io(grid, f, t, node)$t_selectedYear(t); // tmp from ts_cf
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

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_node.csv id=ts_node_io index=1,2,3,4 values=(5..LastCol)  useHeader=y output="%input_dir%/bb_ts_node.gdx" checkDate=TRUE'
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

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_historical_levels.csv id=ts_node_io index=1,2,3,4 values=(5..LastCol)  useHeader=y output="%input_dir%/bb_ts_historical_levels.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort csv2gdx ts_historical_levels (hydro) failed! Check that your input file is valid and that your file path and file name are correct.

$ifthen exist '%input_dir%/bb_ts_historical_levels.gdx'
    $$gdxin  '%input_dir%/bb_ts_historical_levels.gdx'
    $$loaddcm ts_node_io
    $$gdxin
$endif

* --- forecast, reference, ts_node_io ------------------

* no forecasts



* --- creating ts_node ----------------------------------

* display time
now    = jnow;
hour   = ghour(now);
minute = gminute(now);
second = gsecond(now);
display  hour, minute, second;

* convert ts_node_io to ts_node
if(%tsYear%>1980,
    // pick only selected values if modelling a specific times series year
    // tmp from ts_cf
    ts_node(grid, node, param_gnBoundaryTypes, f, t-tmp) = ts_node_io(grid, param_gnBoundaryTypes, f, t, node)$t_selectedYear(t);
    // clear the large ts table if modelling only a specific time series year
    option clear = ts_node_io;
else
    // otherwise rename the large table to the model input table
    ts_node(grid, node, param_gnBoundaryTypes, f, t) = ts_node_io(grid, param_gnBoundaryTypes, f, t, node);
);



* --- setting boundStart and clearing values from 'reference' timeseries -----

// if user has given tsYear
//  * loop gn_hydroStorages
//  * set boundStart = 1, useConstant =1
//  * calculate constant value as max(historical ts value, between downward and upward, user given value in the input data)
// else
//  * use value from time series
//  
// NOTE: will produce wrong values without tsYear

option gn_tmp < ts_node;

if(%tsYear%>1980,
    loop(gn_hydroStorage(grid, node),
        p_gn(grid, node, 'boundStart') = 1;
        p_gnBoundaryPropertiesForStates(grid, node, 'reference', 'useConstant') = 1;

        p_gnBoundaryPropertiesForStates(grid, node, 'reference', 'constant') 
            = max{ p_gnBoundaryPropertiesForStates(grid, node, 'reference', 'constant'),
                   [ts_node(grid, node, 'downwardLimit', 'f00', 't000001') + (ts_node(grid, node, 'upwardLimit', 'f00', 't000001')-ts_node(grid, node, 'downwardLimit', 'f00', 't000001'))/2]${gn_tmp(grid, node)},
                   ts_node(grid, node, 'reference', 'f00', 't000001')
                   };            
    );
else
    loop(node,
        if(sum(grid, sum(t, ts_node(grid, node, 'reference', 'f00', t))) > 0,
            p_gn('all', node, 'boundStart') = 1;
            p_gnBoundaryPropertiesForStates('all', node, 'reference', 'useTimeseries') = 1;
        );
    );
);

// remove data from ts_node(*, *, 'reference', f, t)
option gn_tmp < ts_node;
ts_node(gn_tmp(grid, node), 'reference', f, t) = 0;




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
* Additional manual changes to input data
* --------------------------------------------------------------------------------

* --- applying price multipliers ----------------------------------

ts_priceChange(node, t) = ts_priceChange(node, t) * %priceMultiplier%;
ts_emissionPriceChange(emission, group, t) = ts_emissionPriceChange(emission, group, t) * (1-(1-%priceMultiplier%)/2);


* --- converting selected units to onlineLP and onlineMIP --------------

// converting selected units to onlineLP
* unit efficiency approximations. If op00 > 0 -> LP
loop(unit,
    if(p_unit(unit, 'op00')>0,
        effLevelGroupUnit(effLevel, 'directOff', unit) = no;
        effLevelGroupUnit('level1', 'directOnLP', unit) = yes;
        effLevelGroupUnit('level2', 'directOnLP', unit) = yes;
    );
);

// converting FI and SE nuclear to MIP units. Deactivated by default 
$ontext 
effLevelGroupUnit('level1', 'directOnLP', 'U_FI00_nuclear') = no;
effLevelGroupUnit('level1', 'directOnMIP', 'U_FI00_nuclear') = yes;
effLevelGroupUnit('level1', 'directOnLP', 'U_SE03_nuclear') = no;
effLevelGroupUnit('level1', 'directOnMIP', 'U_SE03_nuclear') = yes;
$offtext

// adding all directOff level3 for all units
option unit_tmp < effLevelGroupUnit;
effLevelGroupUnit('level3', 'directOff', unit_tmp) = yes;


* --- clearing reduntant values from p_gnu_io, p_unit and rounding remaining values -------

// clearing reduntant values from p_gnu_io and rounding the remaining values to 4 digits
p_gnu_io(grid, node, unit, input_output, 'maxRampUp')$((p_gnu_io(grid, node, unit, input_output, 'maxRampUp')=1)$p_gnu_io(grid, node, unit, input_output, 'maxRampUp')) = 0;
p_gnu_io(grid, node, unit, input_output, 'maxRampDown')${(p_gnu_io(grid, node, unit, input_output, 'maxRampDown')=1)$p_gnu_io(grid, node, unit, input_output, 'maxRampDown') } = 0;
p_gnu_io(grid, node, unit, input_output, param_gnu) = round(p_gnu_io(grid, node, unit, input_output, param_gnu), 4);


// clearing reduntant values from p_unit and rounding the remaining values to 4 digits
p_unit(unit, 'minOperationHours')${(p_unit(unit, 'minOperationHours')=1)$p_unit(unit, 'minOperationHours')} = 0;
p_unit(unit, 'minShutdownHours')${(p_unit(unit, 'minShutdownHours')=1)$p_unit(unit, 'minShutdownHours')} = 0;
p_unit(unit, param_unit) = round(p_unit(unit, param_unit), 4);


* --- reservoir downwardLimit adjustments -------

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



* --- project specific modifications ----------