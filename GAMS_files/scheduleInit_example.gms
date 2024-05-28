$ontext
This file is part of Backbone.

Backbone is free software: you can redistribute it and/or modify
it under the terms of the GNU Lesser General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

Backbone is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with Backbone.  If not, see <http://www.gnu.org/licenses/>.
$offtext


* =============================================================================
* --- Model Definition - Schedule ---------------------------------------------
* =============================================================================

if (mType('schedule'),
    m('schedule') = yes; // Definition, that the model exists by its name

* --- Define Key Execution Parameters in Time Indeces -------------------------

    // Define simulation start and end time indeces
    if(%tsYear%>1980,
        mSettings('schedule', 't_start') = 1;  // First time step to be solved, 1 corresponds to t000001 (t000000 will then be used for initial status of dynamic variables)
    else
        mSettings('schedule', 't_start') = 245449;  // First time step to be solved, 1 corresponds to t000001 (t000000 will then be used for initial status of dynamic variables)
    );

    mSettings('schedule', 't_end') =  mSettings('schedule', 't_start') + 24*%modelledDays% - 1; // Last time step to be included in the solve (may solve and output more time steps in case t_jump does not match)

    // Define simulation horizon and moving horizon optimization "speed"
    mSettings('schedule', 't_horizon') = 24*7*65;    // How many active time steps the solve contains (aggregation of time steps does not impact this, unless the aggregation does not match)
    mSettings('schedule', 't_jump') = 24;          // How many time steps the model rolls forward between each solve

    // Define length of data for proper circulation
    if(%tsYear%>1980,
        mSettings('schedule', 'dataLength') =  8760;
    else 
        mSettings('schedule', 'dataLength') =  min(mSettings('schedule', 't_start')-1 + 8760, 306815);
    );


* =============================================================================
* --- Model Time Structure ----------------------------------------------------
* =============================================================================

t_invest(t) = no;

* --- Define Samples ----------------------------------------------------------

    // Number of samples used by the model
    mSettings('schedule', 'samples') = 1;

    // Define Initial and Central samples
    ms_initial('schedule', 's000') = yes;
    ms_central('schedule', 's000') = yes;

    // Define time span of samples
    msStart('schedule', 's000') = mSettings('schedule', 't_start');
    msEnd('schedule', 's000') = mSettings('schedule', 't_start') + mSettings('schedule', 't_end') + mSettings('schedule', 't_horizon');

    // Define the probability (weight) of samples
    p_msAnnuityWeight('schedule', 's000') = 1;
    p_msProbability('schedule', s) = 0;
    p_msProbability('schedule', 's000') = 1;
    p_msWeight('schedule', s) = 0;
    p_msWeight('schedule', 's000') = 1;

* --- Define Time Step Intervals ----------------------------------------------

    // Define the duration of a single time-step in hours
    mSettings('schedule', 'stepLengthInHours') = 1;

    // Define the time step intervals in time-steps
    mInterval('schedule', 'stepsPerInterval', 'c000') = 1;
    mInterval('schedule', 'lastStepInIntervalBlock', 'c000') = 1*24;
    mInterval('schedule', 'stepsPerInterval', 'c001') = 3;
    mInterval('schedule', 'lastStepInIntervalBlock', 'c001') = 2*24;
    mInterval('schedule', 'stepsPerInterval', 'c002') = 6;
    mInterval('schedule', 'lastStepInIntervalBlock', 'c002') = 24*7;

    mInterval('schedule', 'stepsPerInterval', 'c003') = 24;
    mInterval('schedule', 'lastStepInIntervalBlock', 'c003') = 24*14;
    mInterval('schedule', 'stepsPerInterval', 'c004') = 168;
    mInterval('schedule', 'lastStepInIntervalBlock', 'c004') = 24*7*65;


* --- z-structure for superpositioned nodes ----------------------------------

    // add the candidate periods to model
    // no need to touch this part
    // The set is mainly used in the 'invest' model
    mz('schedule', z) = no;

    // Mapping between typical periods (=samples) and the candidate periods (z).
    // Assumption is that candidate periods start from z000 and form a continuous
    // sequence.
    // The set is mainly used in the 'invest' model
    zs(z,s) = no;

* =============================================================================
* --- Model Forecast Structure ------------------------------------------------
* =============================================================================

    Option clear = gn_forecasts;  // By default includes everything, so clear first
    Option clear = unit_forecasts;  // By default includes everything, so clear first

if(%forecastNumber%=1, 
    // Define the number of forecasts used by the model
    mSettings('schedule', 'forecasts') = 0;

    // Define Realized and Central forecasts
    mf_realization('schedule', 'f00') = yes;
    mf_central('schedule', 'f00') = yes;

    // Define forecast probabilities (weights)
    p_mfProbability('schedule', f) = 0;
    p_mfProbability(mf_realization('schedule', f)) = 1;

else

    // Define which nodes and timeseries use forecasts

    // loops flowNodes that have ts_cf data
    option flowNode_tmp < ts_cf;
    loop(flowNode_tmp,
        // if ts_cf has values in f01, use forecasts. It is ok to check only f01 because changes.inc makes sure that f02 and f03 also exist.
        if(sum(t, ts_cf(flowNode_tmp, 'f01', t))>0,
            gn_forecasts(flowNode_tmp, 'ts_cf') = yes;            
        );
    );

    // loops gridNodes that have ts_influx data
    option gn_tmp < ts_influx;
    loop(gn_tmp,
        // if ts_influx has values in f01, use forecasts. It is ok to check only f01 because changes.inc makes sure that f02 and f03 also exist
        if(sum(t, ts_influx(gn_tmp, 'f01', t)<>0),
            gn_forecasts(gn_tmp, 'ts_influx') = yes;            
        );
    );

    // loops gridNodes that have ts_node data
    option gn_tmp < ts_node;
    loop(gn_tmp,
        // if ts_node has values in f01, use forecasts. It is ok to check only f01 because changes.inc makes sure that f02 and f03 also exist
        if(sum(t, ts_node(gn_tmp, 'upwardLimit', 'f01', t)<>0),
            gn_forecasts(gn_tmp, 'ts_node') = yes;            
        );
        if(sum(t, ts_node(gn_tmp, 'downwardLimit', 'f01', t)<>0),
            gn_forecasts(gn_tmp, 'ts_node') = yes;            
        );        
    );


    // Define forecast properties and features
    mSettings('schedule', 't_forecastStart') = 1;                  // At which time step the first forecast is available ( 1 = t000001 )
    mSettings('schedule', 't_forecastLengthUnchanging') = 3576;       // Length of forecasts in time steps - this does not decrease when the solve moves forward 
    mSettings('schedule', 't_forecastLengthDecreasesFrom') = 0; // Length of forecasts in time steps - this decreases when the solve moves forward until the new forecast data is read
    mSettings('schedule', 't_perfectForesight') = 0;               // How many time steps after there is perfect foresight (including t_jump)
    mSettings('schedule', 't_forecastJump') = 24;                  // How many time steps before new forecast is available
    mSettings('schedule', 't_improveForecastNew') = 168;           // Number of time steps ahead of time that the forecast is improved on each solve, new method.
    mSettings('schedule', 'boundForecastEnds') = 0;                // 0/1 parameter if last v_state and v_online in f02,f03,... are bound to f01


    // Defining longer forecast improvement horizons for hydro nodes
    p_gn_improveForecastNew(gn_hydroStorage, 'ts_influx_')$gn_forecasts(gn_hydroStorage, 'ts_influx') = 168*2;


    // shorter improvement for upward and downward limits to reduce dummies in reservoirs
    option gn_tmp < ts_node;
    p_gn_improveForecastNew(gn_tmp, 'ts_node_') = 48;


    // Define the number of forecasts used by the model
    mSettings('schedule', 'forecasts') = 2;

    // Define Realized and Central forecasts
    mf_realization('schedule', f) = no;
    mf_realization('schedule', 'f00') = yes;
    mf_central('schedule', f) = no;
    mf_central('schedule', 'f01') = yes;

    // Define forecast probabilities (weights)
    p_mfProbability('schedule', f) = 0;
    p_mfProbability('schedule', 'f00') = 1;
    p_mfProbability('schedule', 'f01') = 1;
);

if(%forecastNumber%=4,
    // Define the number of forecasts used by the model
    mSettings('schedule', 'forecasts') = 4;

    // Define forecast probabilities (weights)
    p_mfProbability('schedule', f) = 0;
    p_mfProbability('schedule', 'f00') = 1;
    p_mfProbability('schedule', 'f01') = 0.6;
    p_mfProbability('schedule', 'f02') = 0.2;
    p_mfProbability('schedule', 'f03') = 0.2;
);


* =============================================================================
* --- Model Features ----------------------------------------------------------
* =============================================================================

* --- Define Reserve Properties -----------------------------------------------

    // Define whether reserves are used in the model
    mSettingsReservesInUse('schedule', restype, up_down) = no;


* --- Define Unit Approximations ----------------------------------------------

    // Define the last time step for each unit aggregation and efficiency level (3a_periodicInit.gms ensures that there is a effLevel until t_horizon)
    mSettingsEff('schedule', 'level1') = 24;
    mSettingsEff('schedule', 'level2') = 336;
    mSettingsEff('schedule', 'level3') = Inf;


* --- Control the solver ------------------------------------------------------

    // Control the use of advanced basis
    mSettings('schedule', 'loadPoint') = 0;  // 0 = no basis, 1 = latest solve, 2 = all solves, 3 = first solve
    mSettings('schedule', 'savePoint') = 0;  // 0 = no basis, 1 = latest solve, 2 = all solves, 3 = first solve


* --- additional data circulation rules ---------------------------------------

    option clear = ff;
    option ff < p_mfProbability;

    option flowNode_tmp < ts_cf;
    gn_tsCirculation('ts_cf', flowNode_tmp, ff, 'interpolateStepChange', 'isActive') = 1;
    gn_tsCirculation('ts_cf', flowNode_tmp, ff, 'interpolateStepChange', 'length') = 12;

    // not needed for PV as year changes overnight
    gn_tsCirculation('ts_cf', flowNode_tmp('PV', node), ff, 'interpolateStepChange', 'isActive') = 0;
    gn_tsCirculation('ts_cf', flowNode_tmp('PV', node), ff, 'interpolateStepChange', 'length') = 0;

    option gn_tmp < ts_influx;
    gn_tsCirculation('ts_influx', gn_tmp, ff, 'interpolateStepChange', 'isActive') = 1;
    gn_tsCirculation('ts_influx', gn_tmp, ff, 'interpolateStepChange', 'length') = 24;

    option gn_tmp < ts_node;
    gn_tsCirculation('ts_node', gn_tmp, ff, 'interpolateStepChange', 'isActive') = 1;
    gn_tsCirculation('ts_node', gn_tmp, ff, 'interpolateStepChange', 'length') = 48;



* --- Solver speed improvements -------------------------------

    mSettings('schedule', 'reducedDummies') = 0;
    mSettings('schedule', 'scalingMethod') = 2;
    mSettings('schedule', 'automaticRoundings') = 1;

    p_roundingTs('ts_influx_') = 1;
    p_roundingTs('ts_cf_') = 4;
    p_roundingTs('ts_node_') = 1;


); // END if(mType)
