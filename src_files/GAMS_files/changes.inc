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
- Converting constant timeseries to actual constants
- project specific modifications


NOTE: additional project specific modifications should be added to the very end.

$offtext



* --------------------------------------------------------------------------------
* Default values for command line parameters if not given
* --------------------------------------------------------------------------------

// set values global to be available also when reading other files
$if not set modelYear $setglobal modelYear 2025
$if not set climateYear $setglobal tsYear 2015
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
* Converting constant timeseries to actual constants
* --------------------------------------------------------------------------------


* --- converting constant ts_influx to p_gn('influx') ---------

// timeseries with constant values are much slower to process than single constant parameter

// gn in ts_influx
option gn_tmp < ts_influx;
// f in ts_influx
option ff < ts_influx;
// t in ts_influx
option tt < ts_influx;

// initialize gn_tmp_
option clear = gn_tmp_;

// find gn that have constant ts_influx 
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


* --- converting constant ts_node('upwardLimit') to actual constant in p_gnBoundaryProperties ---------
// gn in ts_node
option gn_tmp < ts_node;
// f in ts_node
option ff < ts_node;
// t in ts_node
option tt < ts_node;

// initialize gn_tmp_
option clear = gn_tmp_;

// find gn that have constant ts_node('upwardLimit')
loop(gn_tmp(grid, node),
    // if data only in 'f00'
    if(sum(tt, ts_node(grid, node, 'upwardLimit', 'f01', tt)) = 0,
        gn_tmp_(grid, node)
            ${ [smax(tt, ts_node(grid, node, 'upwardLimit', 'f00', tt)) = smin(tt, ts_node(grid, node, 'upwardLimit', 'f00', tt)) ]
              } = yes;
    // if data only also in other forecasts than 'f00'
    else
        gn_tmp_(grid, node)
            ${ [smax((ff, tt), ts_node(grid, node, 'upwardLimit', ff, tt)) = smin((ff, tt), ts_influx(grid, node, ff, tt)) ]
              } = yes;
    ); // END if(more than one f in ts_influx)
); // END loop(gn_tmp)

// replacing values in p_gnBoundaryPropertiesForStates
p_gnBoundaryPropertiesForStates(gn_tmp_, 'upwardLimit', 'useTimeSeries') = 0;
p_gnBoundaryPropertiesForStates(gn_tmp_, 'upwardLimit', 'useConstant') = 1;
p_gnBoundaryPropertiesForStates(gn_tmp_(grid, node), 'upwardLimit', 'constant') = ts_node(grid, node, 'upwardLimit', 'f00', 't000001');

// clear timeseries from ts_node to speed up compilation
ts_node(gn_tmp_, 'upwardLimit', ff, tt) = 0;



* --------------------------------------------------------------------------------
* project specific modifications
* --------------------------------------------------------------------------------


