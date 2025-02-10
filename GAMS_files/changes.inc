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
    - listing hydro storages
- Reading additional input data
    - optional bb_input_addData1.xlsx
    - optional bb_input_addData2.xlsx
- reading preprocessed input timeseries
    - ts_cf
    - ts_inclux
    - ts_node
    - creating hydro storage start levels, clearing reference ts
- additional manual changes 
    - converting selected units to onlineLP, onlineMIP
    - adding certain unit parameters
    - converting constant ts_influx to p_gn('influx')
- project specific modifications


NOTE: additional project specific modifications are safest to add to the very end.

$offtext



* --------------------------------------------------------------------------------
* Set and parameter definitions
* --------------------------------------------------------------------------------

* --- New sets, parameters, and scalars --------------

Sets
gn_hydroStorage(grid, node) "helper set listing hydro nodes with storage"
;

* --- default values for command line parameters if not given ------

// set values global to be available also when reading other files
$if not set modelYear $setglobal modelYear 2025
$if not set tsYear $setglobal tsYear 2015
$if not set modelledDays $setglobal modelledDays 365
$if not set forecasts $evalglobal forecastNumber 4
$if set forecasts $evalglobal forecastNumber %forecasts%


* --- listing hydro storages -------------

// a helper set used when generating start levels from hydro storages
// gn_hydroStorage(grid, node) = yes; // capacity, normal year inflow, calculated FLH
// Note: required also in scheduleInit.gms

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
gn_hydroStorage('all', 'PL00_psClosed') = yes;  // 1.3 GW, 0 TWh
gn_hydroStorage('all', 'UK00_psClosed') = yes;  // 3.0 GW, 0 TWh




* --------------------------------------------------------------------------------
* Reading additional input data
* --------------------------------------------------------------------------------

* --- optional bb_input_addData1.xlsx ----------------------------

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



* --- optional bb_input_addData2.xlsx ----------------------------

// proceeding only if additional input data excel exists
$ifthen exist '%input_dir%/bb_input_addData2.xlsx'

// Convert additional excel input
$call 'gdxxrw Input="%input_dir%/bb_input_addData2.xlsx" Output="%input_dir%/bb_input_addData2.gdx" index=index!A1'
$ife %system.errorlevel%>0 $abort gdxxrw failed!

    // load input data
    $$gdxin  '%input_dir%/bb_input_addData2.gdx'
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
* reading preprocessed timeseries
* --------------------------------------------------------------------------------


* --- ts_cf ----------------------------------

// reading only selected few tables required for time series preprocessing
$ifthen exist '%input_dir%/ts_cf.gdx'  
    // load input data
    $$gdxin  '%input_dir%/ts_cf.gdx'
    $$loaddcm ts_cf
    $$gdxin
$else
    $$abort 'Run timeseries preprocessing to create %input_dir%/ts_cf.gdx!'     
$endif


* --- ts_influx ----------------------------------

// reading only selected few tables required for time series preprocessing
$ifthen exist '%input_dir%/ts_influx.gdx'  
    // load input data
    $$gdxin  '%input_dir%/ts_influx.gdx'
    $$loaddcm ts_influx
    $$gdxin
$else
    $$abort 'Run timeseries preprocessing to create %input_dir%/ts_influx.gdx!'     
$endif


* --- ts_node ----------------------------------

// reading only selected few tables required for time series preprocessing
$ifthen exist '%input_dir%/ts_node.gdx'  
    // load input data
    $$gdxin  '%input_dir%/ts_node.gdx'
    $$loaddcm ts_node
    $$gdxin
$else
    $$abort 'Run timeseries preprocessing to create %input_dir%/ts_node.gdx!'     
$endif


* --- setting boundStart and clearing values from 'reference' timeseries -----

// if user has given tsYear
//  * loop gn_hydroStorage
//  * set boundStart = 1, useConstant =1
//  * calculate constant value as max(historical ts value, between downward and upward, user given value in the input data)
// else
//  * use value from time series
//  
// NOTE: will produce wrong values without tsYear

option gn_tmp < ts_node;

loop(gn_hydroStorage(grid, node),
    p_gn(grid, node, 'boundStart') = 1;
    p_gnBoundaryPropertiesForStates(grid, node, 'reference', 'useConstant') = 1;

    p_gnBoundaryPropertiesForStates(grid, node, 'reference', 'constant') 
        = max{ p_gnBoundaryPropertiesForStates(grid, node, 'reference', 'constant'),
               [ts_node(grid, node, 'downwardLimit', 'f00', 't000001') + (ts_node(grid, node, 'upwardLimit', 'f00', 't000001')-ts_node(grid, node, 'downwardLimit', 'f00', 't000001'))/2]${gn_tmp(grid, node)},
               ts_node(grid, node, 'reference', 'f00', 't000001')
               };            
    );


// remove data from ts_node(*, *, 'reference', f, t)
ts_node(gn_tmp(grid, node), 'reference', f, t) = 0;



* --------------------------------------------------------------------------------
* Additional manual changes to input data
* --------------------------------------------------------------------------------

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




* --- adding certain unit parameters ----------

// making sure all units have unitType
unitUnittype(unit,'all')${not unitUnittype(unit,'all')} = yes;

// adding directOff level3 for all units
option unit_tmp < effLevelGroupUnit;
effLevelGroupUnit('level3', 'directOff', unit_tmp) = yes;



* --- converting constant ts_influx to p_gn('influx') ---------

// timeseries with constant values are much slower to process than single constant parameter

// gn in ts_influx
option gn_tmp < ts_influx;
// f in ts_influx
option ff < ts_influx;
// t in ts_influx
option tt < ts_influx;

// gn that have constant ts_influx 
option clear = gn_tmp_;
loop(gn_tmp(grid, node),
    // if data only in 'f00'
    if(sum(tt, ts_influx(grid, node, 'f01', tt)) = 0,
        gn_tmp_(grid, node)
            ${ [smax(tt, ts_influx(grid, node, 'f00', tt)) = smin(tt, ts_influx(grid, node, 'f00', tt)) ]
              } = yes;
    // if data only also in other forecasts than 'f00'
    else
        gn_tmp_(grid, node)
            ${ [smax((ff, tt), ts_influx(grid, node, ff, tt)) = smin((ff, tt), ts_influx(grid, node, ff, tt)) ]
              } = yes;
    ); // END if(more than one f in ts_influx)

); // END loop(gn_tmp)

// picking value to p_gn('influx') for filtered gn from ts_influx
p_gn(gn_tmp_(grid, node), 'influx') = ts_influx(grid, node, 'f00', 't000001');

// clearing ts_influx of filtered gn 
ts_influx(gn_tmp_, ff, tt) = 0;




* --------------------------------------------------------------------------------
* project specific modifications
* --------------------------------------------------------------------------------


