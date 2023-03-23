
* ------------------------
* Input data conversion file for Backbone with GAMS ver >= 25.0. Put this file in the input/ folder.
* ------------------------

*define parameters for time checking
scalar now       Gregorian current date + time
       hour      hour,
       minute    minute,
       second    second;

* ------------------------
* temporary set and parameter definitions
* ------------------------

Sets
t_selectedYear(t) "Time steps selected by user in --tsYear and --modelledDays command line parameters"
year "available model time series years" / y0001 * y2022 /
;

Parameters
ts_influx_io(grid, f, t, node) "External power inflow/outflow during a time step (MWh/h) (reordered)"
ts_cf_io(flow, f, t, node) "Available capacity factor time series (p.u. reordered)"
ts_node_io(grid, param_gnBoundaryTypes, f, t, node) "Set node limits according to time-series form exogenous input (reordered)"
p_yearStarts(year) "First time steps of available years"
;


* ------------------------
* Reading additional input data
* ------------------------

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



* ------------------------
* assuming default values for specific command line parameters if not given.
* ------------------------

// set global to be available also when reading other files

$if not set tsYear $setglobal tsYear 0
$if not set modelYear $setglobal modelYear 2025
$if not set modelledDays $setglobal modelledDays 365
$if not set forecasts $setglobal forecastNumber 1
$if set forecasts $setglobal forecastNumber %forecasts%
$if not set priceMultiplier $setglobal priceMultiplier 1


* ------------------------
* making few manual changes to input data
* ------------------------

ts_priceChange(node, t) = ts_priceChange(node, t) * %priceMultiplier%;
ts_emissionPriceChange(emission, group, t) = ts_emissionPriceChange(emission, group, t) * (1-(1-%priceMultiplier%)/2);


* updating VRE capacities to current 2025 projections
$ifthen exist '%input_dir%/VREscen-2025.inc'
   $$include '%input_dir%/VREscen-2025.inc'
$endif

* updating VRE capacities to current 2030 projections
$ifthen exist '%input_dir%/VREscen-2030.inc'
   $$include '%input_dir%/VREscen-2030.inc'
$endif


* unit efficiency approximations. If op00 > 0 -> LP
loop(unit,
    if(p_unit(unit, 'op00')>0,
        effLevelGroupUnit(effLevel, 'directOff', unit) = no;
        effLevelGroupUnit('level1', 'directOnLP', unit) = yes;
        effLevelGroupUnit('level2', 'directOnLP', unit) = yes;
    );
);

* converting Finnish and Swedish nuclear to MIP
effLevelGroupUnit('level1', 'directOnLP', 'U_FI00_nuclear') = no;
effLevelGroupUnit('level1', 'directOnMIP', 'U_FI00_nuclear') = yes;
effLevelGroupUnit('level1', 'directOnLP', 'U_SE03_nuclear') = no;
effLevelGroupUnit('level1', 'directOnMIP', 'U_SE03_nuclear') = yes;




* ------------------------
* Listing first time steps of each year and populating selectedYear
* ------------------------

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


if(%tsYear%>1980,
    // picking the first time step of the selected year to tmp variable
    tmp = sum(year $ (ord(year) = %tsYear%), p_yearStarts(year));

    // Determine time steps in the selected year to construct a year starting at t000001
    t_selectedYear(t)${ ord(t) >= (tmp + 1)
                        and ord(t) <= (tmp + 8760)
                      }
        = yes;
);



* ------------------------
* ts_cf_io
* ------------------------

* --- realization ---------

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_cf_io.csv id=ts_cf_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_cf_io.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort  csv2gdx failed.

* Load ts_cf_io
$ifthen exist '%input_dir%/bb_ts_cf_io.gdx'
    $$gdxin  '%input_dir%/bb_ts_cf_io.gdx'
    $$loaddc ts_cf_io
    $$gdxin
$endif

* --- forecasts ----------

$ifthen set forecasts
    // convert csv to gdx with correct id
    $$call 'csv2gdx %input_dir%/bb_ts_cf_io_50p.csv id=ts_cf_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_cf_io_50p.gdx" checkDate=TRUE'
    // Load ts_influx_io
    $$gdxin  '%input_dir%/bb_ts_cf_io_50p.gdx'
    $$loaddcm ts_cf_io
    $$gdxin
$endif
$ife %system.errorlevel%>0 $abort csv2gdx ts_cf_io_50p failed! Check that your input file is valid and that your file path and file name are correct.



* ------------------------
* ts_cf_io (tsYear conversion if defined)
* ------------------------

* convert ts_cf_io to ts_cf
* display time
now    = jnow;
hour   = ghour(now);
minute = gminute(now);
second = gsecond(now);
display  hour, minute, second;

if(%tsYear%>1980,
    tmp = tmp-1 ;  // -1 as t000000 is the first one, but we want to map to t000001
    ts_cf(flow, node, f, t - tmp) = ts_cf_io(flow, f, t, node)$t_selectedYear(t) ;
    option clear = ts_cf_io;
else
    ts_cf(flow, node, f, t) = ts_cf_io(flow, f, t, node);
);



* ------------------------
* ts_influx_io (elec)
* ------------------------

* --- realization ---------

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_influx_elec.csv id=ts_influx_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_influx_elec.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort gdxxrw addition input failed! Check that your input Excel is valid and that your file path and file name are correct.

* Load ts_influx_io
$ifthen exist '%input_dir%/bb_ts_influx_elec.gdx'
    $$gdxin  '%input_dir%/bb_ts_influx_elec.gdx'
    $$loaddc ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif

* --- forecasts ----------

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



* ------------------------
* ts_influx_io (hydro)
* ------------------------

* --- realization ---------

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_influx_hydro.csv id=ts_influx_io index=1,2,3 values=(4..LastCol)  useHeader=y output="%input_dir%/bb_ts_influx_hydro.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort gdxxrw ts_influx_io (hydro) failed! Check that your input Excel is valid and that your file path and file name are correct.

* Load ts_influx_io
$ifthen exist '%input_dir%/bb_ts_influx_hydro.gdx'
    $$gdxin  '%input_dir%/bb_ts_influx_hydro.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif

* --- forecasts ----------

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




* ------------------------
* ts_influx_io (heat)
* ------------------------

* --- realization ---------

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_influx_heat_%modelYear%.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y valueDim=y output="%input_dir%/bb_ts_influx_heat_%modelYear%.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort gdxxrw ts_influx_io (heat) failed! Check that your input Excel is valid and that your file path and file name are correct.

* Load ts_influx_io
$ifthen exist '%input_dir%/bb_ts_influx_heat_%modelYear%.gdx'
    $$gdxin  '%input_dir%/bb_ts_influx_heat_%modelYear%.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif

* --- forecasts ----------

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


* ------------------------
* ts_influx_io (generation limit nodes)
* ------------------------

* --- realization ---------

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_influx_genlim.csv id=ts_influx_io index=1,2,3 values=(4..LastCol) useHeader=y valueDim=y output="%input_dir%/bb_ts_influx_genlim.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort gdxxrw ts_influx_io (heat) failed! Check that your input Excel is valid and that your file path and file name are correct.

* Load ts_influx_io
$ifthen exist '%input_dir%/bb_ts_influx_genlim.gdx'
    $$gdxin  '%input_dir%/bb_ts_influx_genlim.gdx'
    $$loaddcm ts_influx_io
    $$gdxin
    $$set convert_ts_influx
$endif

* --- forecasts ----------


* ------------------------
* ts_influx_io (tsYear conversion if defined)
* ------------------------

$ifthen set convert_ts_influx
    if(%tsYear%>1980,
        ts_influx(grid, node, f, t - tmp) = ts_influx_io(grid, f, t, node)$t_selectedYear(t); // tmp from ts_cf
        option clear = ts_influx_io;
    else
         ts_influx(grid, node, f, t) = ts_influx_io(grid, f, t, node);
    );
$endif





* ------------------------
* ts_node
* ------------------------

* --- realization ---------

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_node.csv id=ts_node_io index=1,2,3,4 values=(5..LastCol)  useHeader=y output="%input_dir%/bb_ts_node.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort csv2gdx ts_node_io (hydro downward and upward limits) failed! Check that your input file is valid and that your file path and file name are correct.

$ifthen exist '%input_dir%/bb_ts_node.gdx'
    $$gdxin  '%input_dir%/bb_ts_node.gdx'
    $$loaddc ts_node_io
    $$gdxin
$endif

* --- forecasts ----------



* ------------------------
* ts_node (reference)
* ------------------------

* --- realization ---------

$if set input_file_excel $call 'csv2gdx %input_dir%/bb_ts_historical_levels.csv id=ts_node_io index=1,2,3,4 values=(5..LastCol)  useHeader=y output="%input_dir%/bb_ts_historical_levels.gdx" checkDate=TRUE'
$ife %system.errorlevel%>0 $abort csv2gdx ts_historical_levels (hydro) failed! Check that your input file is valid and that your file path and file name are correct.

$ifthen exist '%input_dir%/bb_ts_historical_levels.gdx'
    $$gdxin  '%input_dir%/bb_ts_historical_levels.gdx'
    $$loaddcm ts_node_io
    $$gdxin
$endif

* --- forecast ---------

* ------------------------
* ts_node (tsYear conversion if defined)
* ------------------------

if(%tsYear%>1980,
    // tmp from ts_cf
    ts_node(grid, node, param_gnBoundaryTypes, f, t-tmp) = ts_node_io(grid, param_gnBoundaryTypes, f, t, node)$t_selectedYear(t);
else
    ts_node(grid, node, param_gnBoundaryTypes, f, t) = ts_node_io(grid, param_gnBoundaryTypes, f, t, node);
);


$ifthen exist '%input_dir%/bb_ts_historical_levels.gdx'
    if(%tsYear%>1980,
        loop(node,
            if(sum(grid, sum(f, ts_node(grid, node, 'reference', f, 't000001'))) > 0,
                p_gn('all', node, 'boundStart') = 1;
                p_gnBoundaryPropertiesForStates('all', node, 'reference', 'useConstant') = 1;
                p_gnBoundaryPropertiesForStates('all', node, 'reference', 'constant') = max(0.7* sum(grid, sum(f, ts_node(grid, node, 'upwardLimit', f, 't000001'))),
                                                                                            sum(grid, sum(f, ts_node(grid, node, 'reference', f, 't000001')))
                                                                                            );
            );
        );

    else
        loop(node,
            if(sum(grid, sum(f, sum(t, ts_node(grid, node, 'reference', f, t)))) > 0,
                p_gn('all', node, 'boundStart') = 1;
                p_gnBoundaryPropertiesForStates('all', node, 'reference', 'useTimeseries') = 1;
            );
        );
    );
$endif

if(%tsYear%>1980,
    option clear = ts_node_io;
);

* --- forecasts ----------



* ------------------------
* ts_influx (adding missing forecasts)
* ------------------------

* projecting a temporary (grid, node) set for looping only required pairs
Set gn_tmp(grid, node);
option gn_tmp < ts_influx;

if(%forecastNumber% >= 2,
    // copying f00 values to f01 if values are not given in earlier steps
    loop(gn_tmp(grid, node),
        if(sum(t, ts_influx(grid, node, 'f01', t))=0,
            ts_influx(grid, node, 'f01', t) = ts_influx(grid, node, 'f00', t);
        );
    );
);

if(%forecastNumber% >= 4,
    // copying f00 values to f02 and f03 if values are not given in earlier steps
    loop(gn_tmp(grid, node),
        if(sum(t, ts_influx(grid, node, 'f02', t))=0,
            ts_influx(grid, node, 'f02', t) = ts_influx(grid, node, 'f00', t);
        );
        if(sum(t, ts_influx(grid, node, 'f03', t))=0,
            ts_influx(grid, node, 'f03', t) = ts_influx(grid, node, 'f00', t);
        );
    );
);



* ------------------------
* ts_cf (adding missing forecasts)
* ------------------------

* projecting a temporary (flow, node) set for looping only required pairs
Set flowNode_tmp(flow, node);
option flowNode_tmp < ts_cf;

if(%forecastNumber% >= 2,
    // copying f00 values to f01 if values are not given in earlier steps
    loop(flowNode_tmp(flow, node),
        if(sum(t, ts_cf(flow, node, 'f01', t))=0,
            ts_cf(flow, node, 'f01', t) = ts_cf(flow, node, 'f00', t);
        );
    );
);

if(%forecastNumber% >= 4,
    // copying f00 values to f02 and f03 if values are not given in earlier steps
    loop(flowNode_tmp(flow, node),
        if(sum(t, ts_cf(flow, node, 'f02', t))=0,
            ts_cf(flow, node, 'f02', t) = ts_cf(flow, node, 'f00', t);
        );
        if(sum(t, ts_cf(flow, node, 'f03', t))=0,
            ts_cf(flow, node, 'f03', t) = ts_cf(flow, node, 'f00', t);
        );
    );
);


* ------------------------
* ts_node (adding missing forecasts)
* ------------------------

* projecting a temporary (grid, node, param_gnBoundaryTypes) set for looping only required combinations
Set gnBoundaryTypes_tmp(grid, node, param_gnBoundaryTypes);
option gnBoundaryTypes_tmp < ts_node;

if(%forecastNumber% >= 2,
    // copying f00 values to f01 if values are not given in earlier steps
    loop(gnBoundaryTypes_tmp(grid, node, param_gnBoundaryTypes),
        if(sum(t, ts_node(grid, node, param_gnBoundaryTypes, 'f00', t))>0,
            if(sum(t, ts_node(grid, node, param_gnBoundaryTypes, 'f01', t))=0,
                ts_node(grid, node, param_gnBoundaryTypes, 'f01', t) = ts_node(grid, node, param_gnBoundaryTypes, 'f00', t);
            );
        );
    );
);

if(%forecastNumber% >= 4,
    // copying f00 values to f02 and f03 if values are not given in earlier steps
    loop(gnBoundaryTypes_tmp(grid, node, param_gnBoundaryTypes),
        if(sum(t, ts_node(grid, node, param_gnBoundaryTypes, 'f00', t))>0,
            if(sum(t, ts_node(grid, node, param_gnBoundaryTypes, 'f02', t))=0,
                ts_node(grid, node, param_gnBoundaryTypes, 'f02', t) = ts_node(grid, node, param_gnBoundaryTypes, 'f00', t);
            );
            if(sum(t, ts_node(grid, node, param_gnBoundaryTypes, 'f03', t))=0,
                ts_node(grid, node, param_gnBoundaryTypes, 'f03', t) = ts_node(grid, node, param_gnBoundaryTypes, 'f00', t);
            );
        );
    );
);
