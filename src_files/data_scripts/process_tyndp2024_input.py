# dependencies: pandas, openpyxl
# Tested with Python 3.12.7
import pandas as pd
import os
import sys

r"""

--- Required input ---

- TYNDP 2024 Excel files from https://2024.entsos-tyndp-scenarios.eu/download/
    * (1) "20231103 - Electricity and Hydrogen Reference Grid & Investment Candidates.xlsx"
        download "Electricity and Hydrogen Reference Grid & Investment Candidates After Public Consultation"
    From ENTSO-E & ENTSOG TYNDP 2024 Scenarios  - Outputs, 
    download "NT+ 2030 Modelling Results - Climate Year 2009" (and/or similar 2040 file) for 
    * (2) "MMStandardOutputFile_NT2030_Plexos_CY2009_2.5_v40.xlsx"
    * (3) "MMStandardOutputFile_NT2040_Plexos_CY2009_2.5_v40.xlsx"

- One additional file shared as a part of this repository (./src_files/data_scripts)
    * (4) historical_transfer_ramps.xlsx

--- What this script does ---

- Processes TYNDP 2024 data into NE model input.
- Adds ramp limits from historical_transfer_ramps.xlsx
- Renames and aggregates some TYNDP 2024 Generator_IDs as well as transmission lines to make them 
    match with existing IDs/lines in the NE model
    * see gen_id_renamings and gen_id_aggregations_loc
- Produces an output Excel file (Data_for_TYNDP-2024_National_Trends.xlsx) that 
    * contains NE model input sheets unitdata, demanddata_elec, transferdata
    * These should be copied into \src_files\data_files\TYNDP-2024_National_Trends.xlsx
    * additional documentation sheets trans_capacities, trans_missing_ref_capacities, trans_self_loops
- See TYNDP-2024_National_Trends.xlsx for further instructions    

--- Sheet-level info ---

- unitdata
    * Processed unit data from sources 2 and 3 ("Yearly Outputs" sheet)
- demanddata_elec
    * native electricity demands from sources 2 and 3 ("Yearly Outputs" sheet), values converted from GWh to TWh
    * ramp limits from historical_transfer_ramps.xlsx
- transferdata
    * processed transmission line capacities from sources 1, 2 and 3
    * reference grid capacities from source 1 are compared with modelled maximum flows in sources 2 and 3 -> whichever is higher is chosen as the final capacity

"""

# --- USER SETTINGS ---

# choose National Trends scenario years
chosen_NT_years = [2030, 2040]
assert len(chosen_NT_years) > 0, "No years entered in chosen_NT_years."
for year in chosen_NT_years:
    assert year in [2030, 2040], f"Invalid year entered in chosen_NT_years: {year}. Only 2030 and 2040 are allowed."

plexos_results_filename, path_to_MMStandardOutputFile = [{}, {}]
for year in chosen_NT_years:
    plexos_results_filename[year] = f'MMStandardOutputFile_NT{year}_Plexos_CY2009_2.5_v40.xlsx'
    path_to_MMStandardOutputFile[year] = os.path.normpath(f'./{plexos_results_filename[year]}')

ref_grid_inv_candidate_filename = '20231103 - Electricity and Hydrogen Reference Grid & Investment Candidates.xlsx'
path_to_ref_grid_inv_candidates_file = os.path.normpath(f'./{ref_grid_inv_candidate_filename}')

historical_ramp_limits_filename = 'historical_transfer_ramps.xlsx'
historical_ramp_limits_file_path = os.path.normpath(f'./{historical_ramp_limits_filename}')

# name for the output file of the script
output_filename = 'Data_for_TYNDP-2024_National_Trends.xlsx'
output_filename_path = os.path.normpath(f'./{output_filename}')


# --- MAPPING PLEXOS TO NE MODEL ---

# a dict to determine which country codes in the plexos results will be aggregated together in the NE model
# a slightly modified version of this is defined below for the reference grid excel file
agg_plexos_locations = {
    'AT00': ['AT00'],
    'BE00': ['BE00', 'BEOF'],
    'CH00': ['CH00'],
    'DE00': ['DE00', 'DEKF'],
    'DKE1': ['DKE1', 'DKBH', 'DKKF'],
    'DKW1': ['DKW1'],
    'EE00': ['EE00', 'EEOF'],
    'ES00': ['ES00'],
    'FI00': ['FI00'],
    'FR00': ['FR00', 'FR15'],
    'LT00': ['LT00', 'LTOF'],
    'LV00': ['LV00'],
    'NL00': ['NL00', 'NLLL'],
    'NOM1': ['NOM1'],
    'NON1': ['NON1'],
    'NOS0': ['NOS0'],
    'PL00': ['PL00'],
    'SE01': ['SE01'],
    'SE02': ['SE02'],
    'SE03': ['SE03'],
    'SE04': ['SE04'],
    'UK00': ['UK00', 'UKNI']
}

# the reference grid contains additional country codes for Poland
# the code uses either agg_plexos_locations or agg_ref_grid_locations depending on the Excel file
agg_ref_grid_locations = agg_plexos_locations.copy()
agg_ref_grid_locations['PL00'] = ['PL00', 'PL00I', 'PL00E']

# NOTE: the TYNDP2024 data has Generator_IDs with no direct equivalent in NE model data set
# this renames some of them to existing counterparts in the existing data
gen_id_renamings =    {
    # same value (as in key-value pair) twice -> capacities will be aggregated after renaming
    'Demand Side Response Explicit':        'DR cutoff tier 1', 
    'Demand Side Response Implicit':        'DR cutoff tier 1',

    # these choices are debatable
    # comments at end of line are alternative Generator_IDs in maf2020
    'Battery Storage charge (load)':        'Battery charger 4h',       # Battery charger
    'Battery Storage discharge (gen.)':     'Battery discharger 4h',    # Battery discharger
    'Hard Coal biofuel':                    'Hard coal new Bio',        # Hard coal old 1 Bio, Hard coal old 2 Bio
    'Lignite biofuel':                      'Lignite old 1 Bio',        # Lignite old 2 Bio
    'Gas biofuel':                          'Gas CCGT old 2 Bio',       # Gas conventional old 2 Bio

    'Electrolyser (load)':                  'Electrolyser',
    'Oil shale biofuel':                    'Oil shale new Bio',
    'Others non-renewable':                 'Industry non-renewable CHP',
    'Others renewable':                     'Industry renewable CHP',
    'Pondage':                              'Run-of-River',
    'Solar (Photovoltaic)':                 'Solar PV',
    'Solar (Thermal)':                      'Solar Thermal',
    'Wind Offshore':                        'Offshore Wind',
    'Wind Onshore':                         'Onshore Wind',
    'Pump Storage - Closed Loop (pump)':    'PS Closed pump',
    'Pump Storage - Closed Loop (turbine)': 'PS Closed turbine',
    'Pump Storage - Open Loop (pump)':      'PS Open pump',
    'Pump Storage - Open Loop (turbine)':   'PS Open turbine',
}

# location-specific capacity aggregations for Generator_IDs
# done after the renamings/aggregations specified in gen_id_renamings
gen_id_aggregations_loc = {
    # Reservoir and Run-of-River in NOM1, NON1 and NOS0 have no storagedata in the NE model
    # -> they are aggregated to PS Open turbine (perhaps not a good permanent solution)
    'NOM1': {'Reservoir': 'PS Open turbine',
             'Run-of-River': 'PS Open turbine'},
    'NON1': {'Reservoir': 'PS Open turbine',
             'Run-of-River': 'PS Open turbine'},
    'NOS0': {'Reservoir': 'PS Open turbine',
             'Run-of-River': 'PS Open turbine'}
}


# --- UTILITY FUNCTIONS ---

# check if the output will be overwritten
def check_chosen_output(output_filename_path:str):
    if os.path.exists(output_filename_path):
        print(f'\nWARNING: {output_filename_path} already exists.')
        user_answer = input('\n Do you want to overwrite it (write "yes" to proceed)?\n')
        if user_answer != 'yes':
            sys.exit('\nScript aborted to avoid overwriting output file.')


# --- NATIVE DEMANDS ---

def process_native_demands(plexos_caps_demands, year):
    # fill up missing values in 'Output type'
    plexos_caps_demands['Output type'] = plexos_caps_demands['Output type'].ffill()
    native_demands = plexos_caps_demands[plexos_caps_demands['Output type'] == 'Native Demand (excl. Pump load & Battery charge) [GWh]']
    
    for ne_location, plexos_locations in agg_plexos_locations.items():
        # for each ne_location column, sum together the native demands of to-be-aggregated locations and convert the result from GWh to TWh
        native_demands.loc[:, ne_location] = (native_demands[plexos_locations].sum(axis=1)/1000).copy()

    # remove columns that are not needed after aggregation
    unneeded_columns = [col for col in native_demands.columns if col not in agg_plexos_locations.keys()]
    native_demands = native_demands.drop(columns=unneeded_columns).reset_index(drop=True)

    # edit the demand data to format used by the NE model
    native_demands_ne_input = native_demands.melt(var_name='Country', value_name='TWh/year').copy()
    native_demands_ne_input.insert(1, 'Grid', 'elec')
    native_demands_ne_input.insert(2, 'Node_suffix', None)
    native_demands_ne_input.insert(3, 'Scenario', 'National Trends')
    native_demands_ne_input.insert(4, 'Year', year)
    native_demands_ne_input.insert(6, 'Constant_share', None)

    return native_demands_ne_input


# --- INSTALLED CAPACITIES ---

def process_installed_capacities(plexos_caps_demands, gen_id_renamings, gen_id_aggregations_loc, year):
    capacities = plexos_caps_demands[plexos_caps_demands['Output type'] == 'Installed Capacities [MW]']
    capacities = capacities.rename(columns={'Output type.1':'Generator_ID'}).copy()

    # aggregate columns and remove unnecessary ones, like in process_native_demands()
    for ne_location, plexos_locations in agg_plexos_locations.items():
        capacities.loc[:, ne_location] = capacities[plexos_locations].sum(axis=1).copy()
    unneeded_columns = [col for col in capacities.columns if col not in agg_plexos_locations.keys() and col != 'Generator_ID']
    capacities = capacities.drop(columns=unneeded_columns).reset_index(drop=True)

    # edit installed capacities data to format used by the NE model
    installed_capacities_ne_input = capacities.melt(id_vars=['Generator_ID'], var_name='Country', value_name='capacity_output1')
    installed_capacities_ne_input['Scenario'] = 'National Trends'
    installed_capacities_ne_input['Year'] = year
    installed_capacities_ne_input = installed_capacities_ne_input[['Country', 'Generator_ID', 'Scenario', 'Year', 'capacity_output1']]
    # remove zero-capacity rows
    installed_capacities_ne_input = installed_capacities_ne_input[installed_capacities_ne_input['capacity_output1'] > 0]
    installed_capacities_ne_input['node_suffix_output2'] = None
    installed_capacities_ne_input['capacity_input1'] = None
    installed_capacities_ne_input['Note'] = None

    # a check that entries in gen_id_renamings exist in data
    for k, v in gen_id_renamings.items():
        if k not in installed_capacities_ne_input['Generator_ID'].values:
            print(f"Warning: TYNDP2024 entry missing in gen_id_renamings: {k}")

    installed_capacities_ne_input['Generator_ID'] = installed_capacities_ne_input['Generator_ID'].replace(gen_id_renamings)

    # aggregate some TYNDP2024 entries to fit them into existing maf2020 data
    # in these cases, maf2020 naming will be used
    ins_cap_groupby_cols = ['Country', 'Generator_ID', 'Scenario', 'Year']
    ins_cap_aggs = {'capacity_output1': 'sum', 'node_suffix_output2': 'first', 'capacity_input1': 'first', 'Note': 'first'}
    installed_capacities_ne_input = installed_capacities_ne_input.groupby(ins_cap_groupby_cols, as_index=False).agg(ins_cap_aggs).copy()

    # Apply location-specific capacity aggregations based on gen_id_aggregations_loc
    for location, gen_aggregations in gen_id_aggregations_loc.items():
        for source_gen_id, target_gen_id in gen_aggregations.items():
            # Find rows matching the location and source Generator_ID
            mask = (installed_capacities_ne_input['Country'] == location) & (installed_capacities_ne_input['Generator_ID'] == source_gen_id)
            if mask.any():
                # Add the capacities to the target Generator_ID
                installed_capacities_ne_input.loc[
                    (installed_capacities_ne_input['Country'] == location) & 
                    (installed_capacities_ne_input['Generator_ID'] == target_gen_id), 
                    'capacity_output1'
                ] += installed_capacities_ne_input.loc[mask, 'capacity_output1'].sum()
                # Remove the source Generator_ID rows
                installed_capacities_ne_input = installed_capacities_ne_input[~mask]

    return installed_capacities_ne_input


# --- TRANSMISSION CAPACITIES ---

# add 'from' and 'to' columns to dataframes with index entries of type "BE00-UK00"
def add_from_to_columns(df):
    df.insert(0, 'from', [x.split('-')[0] for x in df.index])
    df.insert(1, 'to', [x.split('-')[1] for x in df.index])


# Create aggregated counterparts to country codes in lines (e.g. BEOF-UK00 -> BE00-UK00)
# agg_locations is either agg_plexos_locations or agg_ref_grid_locations depending on the Excel file
def get_aggregated_line(row, agg_locations, print_removals=False):
    reverse_agg = {v: k for k, values in agg_locations.items() for v in values}
    from_agg = reverse_agg.get(row['from'])
    to_agg = reverse_agg.get(row['to'])
    if from_agg == to_agg and print_removals:
        # e.g. 'BE00-BEOF' converts to 'BE00-BE00' after aggregation -> will be removed
        print(f'REMOVAL: {row["from"]}-{row["to"]} would become {from_agg}-{to_agg}.') 
    return f"{from_agg}-{to_agg}"


# "lost lines" are lines within aggregated areas that point to themselves (e.g. 'BE00-BE00') after aggregation
def remove_lost_lines(df):
    return df[df['agg_line'].str.split('-').apply(lambda x: x[0] != x[1])].copy()


def aggregate_lines(df, capacity_columns:iter, print_aggregations=False):
    if print_aggregations:
        to_be_aggregated = df[df.index != df['agg_line']]
        for i in to_be_aggregated.iterrows():
            print(f'AGGREGATION: {i[0]} -> {i[1]["agg_line"]}.')
    df = df.groupby('agg_line').agg(
        {next(capacity_columns): 'sum', next(capacity_columns): 'sum', 'agg_line':'first'}).set_index('agg_line').copy()
    df.index.name = 'from-to'
    print(df)
    add_from_to_columns(df)
    return df


def preprocess_transmission_data(ref_capacities, min_max_exchanges, agg_plexos_locations, agg_ref_grid_locations, transmission_link_headers):
    ref_capacities = ref_capacities.copy()
    min_max_exchanges = min_max_exchanges.copy()
    
    # lists of all relevant country codes in both plexos and the reference grid
    # these are either in the NE model or aggregated to nodes in it
    all_country_codes_plexos = list(set(code for codes in agg_plexos_locations.values() for code in codes))
    all_country_codes_ref_grid = list(set(code for codes in agg_ref_grid_locations.values() for code in codes))

    # rename the index to 'from-to' to match with Backbone terminology
    ref_capacities.index.name = 'from-to'

    min_max_exchanges.columns = list(transmission_link_headers.columns)
    min_max_exchanges = min_max_exchanges.T

    # remove the '>' symbols in the index of min_max_exchanges
    min_max_exchanges.index = min_max_exchanges.index.str.replace('>', '', regex=False)
    # rename index to Backbone format
    min_max_exchanges.index.name = 'from-to'

    # remove transmission links involving country codes that are not in the NE model
    add_from_to_columns(ref_capacities)
    add_from_to_columns(min_max_exchanges)
    ref_capacities.query(f'`from` in {all_country_codes_ref_grid} and `to` in {all_country_codes_ref_grid}', inplace=True)
    min_max_exchanges.query(f'`from` in {all_country_codes_plexos} and `to` in {all_country_codes_plexos}', inplace=True)

    # rename capacity columns to match with NE model
    ref_capacities = ref_capacities.rename(columns={'Summary Direction 1 (MW)':'export_capacity',
                                                    'Summary Direction 2 (MW)':'import_capacity'}).copy()

    # rename min_max_exchanges columns (just for clarity)
    min_max_exchanges = min_max_exchanges.rename(columns={'Min [MW]:': 'max_modelled_import',
                                                        'Max [MW]:': 'max_modelled_export'}).copy()
    
    # assert that all Min [MW] values are initially negative or zero (i.e. actual imports)
    assert (min_max_exchanges['max_modelled_import'] <= 0).all()

    # take absolute values for imports
    min_max_exchanges['max_modelled_import'] = min_max_exchanges['max_modelled_import'].abs()

    return ref_capacities, min_max_exchanges


def aggregate_transmission_data(ref_capacities, min_max_exchanges, agg_plexos_locations, agg_ref_grid_locations):
    # create copies of the original dataframes which we will aggregate
    ref_capacities_agg = ref_capacities.copy()
    min_max_exchanges_agg = min_max_exchanges.copy()

    print('\nRemoving lines lost during aggregation in the reference grid...\n')
    ref_capacities_agg['agg_line'] = ref_capacities_agg.apply(get_aggregated_line,
                                                            agg_locations=agg_ref_grid_locations,
                                                            print_removals=True,
                                                            axis=1)
    ref_capacities_agg = remove_lost_lines(ref_capacities_agg)

    print('\nRemoving lost lines in modelled crossborder exchanges...\n')
    min_max_exchanges_agg['agg_line'] = min_max_exchanges_agg.apply(get_aggregated_line,
                                                                    agg_locations=agg_plexos_locations,
                                                                    print_removals=True,
                                                                    axis=1)
    min_max_exchanges_agg = remove_lost_lines(min_max_exchanges_agg)

    # Group by the aggregated line and sum the capacities
    print('\nAggregating the lines of the reference grid...\n')
    ref_capacities_agg = aggregate_lines(ref_capacities_agg,
                                        iter(['export_capacity', 'import_capacity']),
                                        print_aggregations=True)
    print('\nAggregating the lines of the modelled crossborder exchanges...\n')
    min_max_exchanges_agg = aggregate_lines(min_max_exchanges_agg,
                                            iter(['max_modelled_export', 'max_modelled_import']),
                                            print_aggregations=True)
    
    return ref_capacities_agg, min_max_exchanges_agg


# --- combine reversed lines in the reference grid ---

def combine_reversed_lines(ref_capacities_agg, min_max_exchanges_agg):
    """
    The aggregation may lead to identical lines pointing in opposite directions.
    Combine capacities of such aggregated transmission lines.
    For each line, keep only the direction that exists in the Plexos DataFrame (plexos_df).
    
    For example, if we have lines:
        DE00-PL00 (export: 1000, import: 2000),
        PL00-DE00 (export: 3000, import: 4000),
    and DE00-PL00 exists in plexos_df, the result will be:
    DE00-PL00 (export: 1000 + 4000, import: 2000 + 3000)
    """
    combined = []  # holds the combined line data as dictionaries
    processed = set()  # keeps track of lines that are already processed

    for index, row in ref_capacities_agg.iterrows():
        if str(index) in processed:
            continue
            
        # Get the reverse direction of current line
        rev = f"{row['to']}-{row['from']}"
        
        # Check if:
        # 1. Reverse direction exists in our data AND
        # 2. Either current or reverse direction exists in reference data
        if rev in ref_capacities_agg.index and (index in min_max_exchanges_agg.index or rev in min_max_exchanges_agg.index):
            # Use the direction that exists in reference data
            canonical = index if index in min_max_exchanges_agg.index else rev
            
            if canonical in min_max_exchanges_agg.index:
                # Determine which is forward and which is reverse based on canonical direction
                fwd, rev_row = (row, ref_capacities_agg.loc[rev]) if canonical == index else (ref_capacities_agg.loc[rev], row)
                
                # Combine the capacities:
                # - Forward direction exports + Reverse direction imports = Total exports
                # - Forward direction imports + Reverse direction exports = Total imports
                combined.append({
                    'from-to': canonical,
                    'export_capacity': fwd['export_capacity'] + rev_row['import_capacity'],
                    'import_capacity': fwd['import_capacity'] + rev_row['export_capacity']
                })
                # Mark both directions as processed
                processed.update([str(index), str(rev)])
                
        # If no reverse exists but current line exists in reference data, keep it as is
        elif index in min_max_exchanges_agg.index:
            combined.append({
                'from-to': index,
                'export_capacity': row['export_capacity'],
                'import_capacity': row['import_capacity']
            })
    
    df_combined = pd.DataFrame(combined).set_index('from-to')
    add_from_to_columns(df_combined)
    
    return df_combined


# --- merging ref_capacities_agg with min_max_exchanges_agg -> choosing NE model capacities ---

def merge_transmission_data(ref_capacities_agg, min_max_exchanges_agg, year):

    merged_cap = ref_capacities_agg.merge(min_max_exchanges_agg, how='left', on=['from-to', 'from', 'to'])
    # positive values mean that modelled exports/imports are higher than the capacity
    merged_cap['export_diff'] = merged_cap['max_modelled_export'] - merged_cap['export_capacity']
    merged_cap['import_diff'] = merged_cap['max_modelled_import'] - merged_cap['import_capacity']
    #merged_cap.query('export_within_limits == False or import_within_limits == False')

    # for the NE model, we choose, as capacity, either the reference capacity or the maximum modelled flow, whichever is higher
    merged_cap['ne_model_export'] = merged_cap[['export_capacity', 'max_modelled_export']].max(axis=1)
    merged_cap['ne_model_import'] = merged_cap[['import_capacity', 'max_modelled_import']].max(axis=1)

    merged_cap.insert(2, 'year', year)

    return merged_cap


# --- process ramp limits ---

def process_ramp_limits(merged_cap, ramp_limits, year):
    merged_cap, ramp_limits = merged_cap.copy(), ramp_limits.copy()
    ramp_limits.rename(columns={
        'export_capacity':'export_capacity_2020',
        'import_capacity':'import_capacity_2020'},
        inplace=True)
    ramp_limits['import_capacity_2020'] = ramp_limits['import_capacity_2020'].abs()
    ramp_limits.set_index('from-to', inplace=True)

    ramp_limits = ramp_limits.query(f'Year == {year}').copy()

    # merge capacities with ramp limits
    merged_cap_ramp_limits = merged_cap.merge(ramp_limits, how='left', on=['from-to']).copy()

    # create the NE model transmission input data sheet

    trans_ne_input = merged_cap_ramp_limits[['grid', 'scenario', 'Year', 'ne_model_export', 'ne_model_import', 'max_ramp_rate']].copy()
    trans_ne_input = trans_ne_input.rename(columns={'ne_model_export':'export_capacity',
                                                    'ne_model_import':'import_capacity',
                                                    'max_ramp_rate':'rampLimit'})
    trans_ne_input['vomCost'] = 1
    trans_ne_input['losses'] = 0.01
    add_from_to_columns(trans_ne_input)
    trans_ne_input = trans_ne_input.query(f'scenario == "National Trends" and Year == {year}').reset_index()
    
    return trans_ne_input


# --- create a column of aggregated lines, for documentation ---

# Create lookup dictionary of which lines were aggregated to which agg_line
def get_aggregation_mapping(original_df, agg_df):
    """Map original lines to their aggregated counterparts by comparing indices"""
    mapping = {}
    for agg_line in agg_df.index:
        # Find all original lines that were aggregated to this line
        aggregated_lines = [line for line in original_df.index 
            if get_aggregated_line(
            {'from': line.split('-')[0], 'to': line.split('-')[1]}, agg_ref_grid_locations) == agg_line
            and line != agg_line]
        if aggregated_lines:
            mapping[agg_line] = aggregated_lines
    return mapping


# get lost lines from a DataFrame
def get_lost_lines(df, agg_locations, source_name, year):
    # Add agg_line column
    df = df.copy()
    df['agg_line'] = df.apply(get_aggregated_line, agg_locations=agg_locations, axis=1)
    
    # Filter lines that would become self-loops after aggregation
    lost_lines = df[df['agg_line'].str.split('-').apply(lambda x: x[0] == x[1])]
    
    # Select and rename columns for the result
    result = lost_lines[['from', 'to']].copy()
    # Add from-to column (will be the index)
    result['from-to'] = result['from'] + '-' + result['to']
    
    if 'export_capacity' in df.columns:
        result['export_capacity'] = lost_lines['export_capacity']
        result['import_capacity'] = lost_lines['import_capacity']
    else:
        result['export_capacity'] = lost_lines['max_modelled_export']
        result['import_capacity'] = lost_lines['max_modelled_import']
    
    # Add source column
    result['source'] = source_name
    
    # Set from-to as index
    result = result.set_index('from-to')
    result.index.name = 'from-to'

    result['year'] = year
    
    # Reorder columns
    result = result[['from', 'to', 'year', 'export_capacity', 'import_capacity', 'source']]
    return result


def create_doc_sheets(ref_capacities, ref_capacities_agg, min_max_exchanges, min_max_exchanges_agg, merged_cap, year):
    # Get mappings from both DataFrames
    ref_mapping = get_aggregation_mapping(ref_capacities, ref_capacities_agg)
    plexos_mapping = get_aggregation_mapping(min_max_exchanges, min_max_exchanges_agg)

    # Combine the mappings and add as a new column, join with commas
    merged_cap['aggregated_lines'] = merged_cap.index.map(lambda x: 
        ', '.join(sorted(set(ref_mapping.get(x, []) + plexos_mapping.get(x, [])))) if (x in ref_mapping or x in plexos_mapping) else None)

    # --- generate dataframes of missing capacities and lost lines, for documentation ---

    # Get lines that exist in min_max_exchanges but not in ref_capacities
    missing_lines = min_max_exchanges[~min_max_exchanges.index.isin(ref_capacities.index)].copy()
    missing_lines['year'] = year

    # Select only relevant columns and rename them for clarity
    missing_capacities = missing_lines[['year', 'max_modelled_export', 'max_modelled_import']].copy()
    missing_capacities = missing_capacities.rename(columns={
        'max_modelled_export': 'export_capacity',
        'max_modelled_import': 'import_capacity'
    })

    # Add from and to columns for better readability
    add_from_to_columns(missing_capacities)

    # Display the result
    print("Lines present in PLEXOS results but missing from reference grid:")
    print(missing_capacities.query('export_capacity > 0 or import_capacity > 0'))

    # Get lost lines from both datasets
    lost_plexos = get_lost_lines(min_max_exchanges, agg_plexos_locations, "PLEXOS results", year)
    lost_ref = get_lost_lines(ref_capacities, agg_ref_grid_locations, "Reference grid", year)

    # Combine the results + sort by source and line names for better readability
    lost_lines = pd.concat([lost_plexos, lost_ref]).sort_values(['source', 'from', 'to'])

    return merged_cap, missing_capacities, lost_lines


# this is the "super function" that processes all the data for a given year
# returns a dict of Excel sheets as dataframes
def process_year_data(year):
    plexos_caps_demands = pd.read_excel(path_to_MMStandardOutputFile[year], sheet_name='Yearly Outputs', skiprows=5)
    # maximum and minimum values of modelled transmissions in the Plexos results
    # in the "Crossborder exchanges" sheet, the table headers ('<from>-<to>') are below the min-max data
    # -> we take the headers (links) separately and add them afterwards in preprocess_transmission_data()
    transmission_link_headers = pd.read_excel(path_to_MMStandardOutputFile[year],
                                              sheet_name='Crossborder exchanges', skiprows=10, usecols='C:FB', nrows=0)
    min_max_exchanges = pd.read_excel(path_to_MMStandardOutputFile[year],
                                      sheet_name='Crossborder exchanges', skiprows=4, usecols='B:FB', index_col=0, nrows=2)

    native_demands_ne_input = process_native_demands(plexos_caps_demands, year)
    
    installed_capacities_ne_input = process_installed_capacities(plexos_caps_demands,
                                                                 gen_id_renamings,
                                                                 gen_id_aggregations_loc,
                                                                 year)
    
    ref_capacities_year, min_max_exchanges_year = preprocess_transmission_data(ref_capacities,
                                                                               min_max_exchanges,
                                                                               agg_plexos_locations,
                                                                               agg_ref_grid_locations,
                                                                               transmission_link_headers)
    
    ref_capacities_agg, min_max_exchanges_agg = aggregate_transmission_data(ref_capacities_year,
                                                                            min_max_exchanges_year,
                                                                            agg_plexos_locations,
                                                                            agg_ref_grid_locations)
    
    ref_capacities_agg = combine_reversed_lines(ref_capacities_agg,
                                                min_max_exchanges_agg)

    merged_cap = merge_transmission_data(ref_capacities_agg,
                                         min_max_exchanges_agg,
                                         year)

    trans_ne_input = process_ramp_limits(merged_cap,
                                         ramp_limits,
                                         year)

    merged_cap, missing_capacities, lost_lines = create_doc_sheets(ref_capacities_year,
                                                                   ref_capacities_agg,
                                                                   min_max_exchanges_year,
                                                                   min_max_exchanges_agg,
                                                                   merged_cap,
                                                                   year)

    return {
        'demanddata_elec': native_demands_ne_input,
        'unitdata': installed_capacities_ne_input,
        'transferdata': trans_ne_input,
        'trans_capacities': merged_cap,
        'trans_missing_ref_capacities': missing_capacities,
        'trans_self_loops': lost_lines
    }


if __name__ == "__main__":

    # check if output would be overwritten
    check_chosen_output(output_filename_path)
    # the reference grid (transmission capacities before investments in 2030)
    ref_capacities = pd.read_excel(path_to_ref_grid_inv_candidates_file, sheet_name='1. Elec Ref Grid').set_index('Border')

    # ramp limits from historical ramp limits
    ramp_limits = pd.read_excel(historical_ramp_limits_file_path, sheet_name='summary', skiprows=9, usecols='AZ:BG')

    processed_data = {
        year: process_year_data(year) for year in chosen_NT_years
    }

    # Concatenate data from both years for each sheet type
    sheet_names = list(processed_data.values())[0].keys()
    processed_data_concatenated = {
        sheet_name: pd.concat([processed_data[year][sheet_name] for year in chosen_NT_years], axis=0, ignore_index=True)
        for sheet_name in sheet_names
    }

    # Write concatenated data to Excel
    with pd.ExcelWriter(output_filename, engine='openpyxl') as writer:
        # also add a remove_units sheet with no content other than the headers
        remove_units_sheet = pd.DataFrame(columns=['Country', 'unit_name_prefix', 'Generator_ID', 'Scenario', 'Year'])
        
        processed_data_concatenated['unitdata'].to_excel(writer, index=False, sheet_name='unitdata')
        processed_data_concatenated['demanddata_elec'].to_excel(writer, index=False, sheet_name='demanddata_elec')
        processed_data_concatenated['transferdata'].to_excel(writer, index=False, sheet_name='transferdata')
        remove_units_sheet.to_excel(writer, index=False, sheet_name='remove_units')
        processed_data_concatenated['trans_capacities'].to_excel(writer, index=False, sheet_name='trans_capacities')
        processed_data_concatenated['trans_missing_ref_capacities'].to_excel(writer, index=False, sheet_name='trans_missing_ref_capacities')
        processed_data_concatenated['trans_self_loops'].to_excel(writer, index=False, sheet_name='trans_self_loops')