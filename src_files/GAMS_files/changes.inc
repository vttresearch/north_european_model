$ontext
* --------------------------------------------------------------------------------
* Changes.inc for the Northern European model
* --------------------------------------------------------------------------------

This file is created for Northern European Model (https://github.com/vttresearch/north_european_model)
and should be placed in the Backbone input folder, see instructions from 'Run specification files' section
of the Northern European Model readme.

Contents 
- Default values for command line parameters if not given
- Reading additional input data
    - inputData_add1.xlsx
    - inputData_add2.xlsx
- reading timeseries
- additional manual changes 
    - converting selected units to onlineLP and onlineMIP
    - creating hydro storage start levels, clearing reference ts
    - converting constant ts_influx to p_gn('influx')
- project specific modifications


NOTE: additional project specific modifications should be added to the very end.

$offtext



* --------------------------------------------------------------------------------
* Default values for command line parameters if not given
* --------------------------------------------------------------------------------

// set values global to be available also when reading other files
$if not set modelYear $setglobal modelYear 2025
$if not set tsYear $setglobal tsYear 2015
$if not set modelledDays $setglobal modelledDays 365
$if not set forecasts $evalglobal forecastNumber 4
$if set forecasts $evalglobal forecastNumber %forecasts%





* --------------------------------------------------------------------------------
* Reading additional input data
* --------------------------------------------------------------------------------

* --- reading possible inputData_add1.xlsx ----------------------------

// proceeding only if additional input data excel exists
$ifthen.addData1 exist '%input_dir%/inputData_add1.xlsx'

    // setting quote mark for unix or windows (MSNT)
    $$ SET QTE "'"
    $$ IFI %SYSTEM.FILESYS%==MSNT $SET QTE '"'

    // Convert additional excel input
    $$call 'gdxxrw Input="%input_dir%/inputData_add1.xlsx" Output="%input_dir%/inputData_add1.gdx" index=index!A1'
    $$ife %system.errorlevel%>0 $abort gdxxrw failed!

    // query checking which data tables exists and writes the list to file inputDataInc
    $$hiddencall gdxdump "%input_dir%/inputData_add1.gdx"  NODATA > "%input_dir%/inputData_add1.inc"
    // Using sed to convert gdxdump output to a format that can be imported to backbone
    // This does the following:
    //    /^[$]/!d; - deletes any line that does not start with a dollar sign.
    //    /^\$LOAD add_/d; - drops any line starting with $LOAD add_.
    //    s/\$LOAD.. /\$LOADDCM /I - replaces occurrences of $LOAD followed by any two characters with $LOADDCM (case-insensitive).
    $$hiddencall sed %QTE%/^[$]/!d; /^\$LOAD add_/d; s/\$LOAD.. /\$LOADDCM /I%QTE%  "%input_dir%/inputData_add1.inc" > "%input_dir%/inputData_add1.inc"
    // importing data from the input data gdx as specified by the %inputDataInc_%
    $$INCLUDE %inputDataInc_%
    // closing the input file
    $$gdxin

$endif.addData1


* --- reading possible inputData_add2.xlsx ----------------------------

// proceeding only if additional input data excel exists
$ifthen.addData2 exist '%input_dir%/inputData_add2.xlsx'

    // setting quote mark for unix or windows (MSNT)
    $$ SET QTE "'"
    $$ IFI %SYSTEM.FILESYS%==MSNT $SET QTE '"'

    // Convert additional excel input
    $$call 'gdxxrw Input="%input_dir%/inputData_add2.xlsx" Output="%input_dir%/inputData_add2.gdx" index=index!A1'
    $$ife %system.errorlevel%>0 $abort gdxxrw failed!

    // query checking which data tables exists and writes the list to file inputDataInc
    $$hiddencall gdxdump "%input_dir%/inputData_add2.gdx"  NODATA > "%input_dir%/inputData_add2.inc"
    // Using sed to convert gdxdump output to a format that can be imported to backbone
    // This does the following:
    //    /^[$]/!d; - deletes any line that does not start with a dollar sign.
    //    /^\$LOAD add_/d; - drops any line starting with $LOAD add_.
    //    s/\$LOAD.. /\$LOADDCM /I - replaces occurrences of $LOAD followed by any two characters with $LOADDCM (case-insensitive).
    $$hiddencall sed %QTE%/^[$]/!d; /^\$LOAD add_/d; s/\$LOAD.. /\$LOADDCM /I%QTE%  "%input_dir%/inputData_add2.inc" > "%input_dir%/inputData_add2.inc"
    // importing data from the input data gdx as specified by the %inputDataInc_%
    $$INCLUDE %inputDataInc_%
    // closing the input file
    $$gdxin

$endif.addData2



* --------------------------------------------------------------------------------
* reading timeseries from GDX files
* --------------------------------------------------------------------------------

$ifthen exist '%input_dir%/import_timeseries.inc'
    $$include '%input_dir%/import_timeseries.inc'
$endif




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




* --- setting boundStart and clearing values from 'reference' timeseries -----

$ontext
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
$offtext


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


