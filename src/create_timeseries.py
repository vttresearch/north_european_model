import os
import time
import importlib
from gdxpds import to_gdx
import pandas as pd
import numpy as np
import glob


class create_timeseries:
    def __init__(self, timeseries_specs, input_folder, output_folder, 
                 start_date, end_date, country_codes, scen_tags, 
                 df_annual_demands, log_start, write_csv_files=False
                 ):
        self.timeseries_specs = timeseries_specs
        self.input_folder = os.path.join(input_folder, "timeseries")
        self.output_folder = output_folder
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.country_codes = country_codes
        self.scenario_year = scen_tags['year']
        self.df_annual_demands = df_annual_demands
        self.log_start = log_start
        self.write_csv_files = write_csv_files


    def log_time(self, message):
        elapsed = time.perf_counter() - self.log_start
        print(f"[{elapsed:0.2f} s] {message}")

    def trim_summary(self, summary_df, round_precision=0):
        # round and drop empty columns
        summary_df = summary_df.round(round_precision)
        summary_df = summary_df.loc[:, summary_df.sum() != 0]

        # Remove leading and trailing rows that are fully empty/NaN.
        mask = ~summary_df.isna().all(axis=1).to_numpy()
        if mask.any():
            first_valid_pos = np.where(mask)[0][0]
            last_valid_pos = np.where(mask)[0][-1]
            summary_df = summary_df.iloc[first_valid_pos:last_valid_pos + 1]

        # Convert dtypes (so that e.g. rounding to 0 decimals gives integers)
        summary_df = summary_df.convert_dtypes()
        return summary_df

    def calculate_average_year(self, input_df, round_precision=0, **kwargs):
        """
        Assumes long format input_df with columns bb_parameter_dimensions and 'time' and 'value'
            * 'time' is datatime format presentation of time, e.g 2000-01-01 00:00:00 
            * 't' is t-index format presentation of time, e.g. t000001

        Calculates 'hour_of_year' as hours elapsed since Jan 1, starting at 1.
        Checks dim_cols as columns in bb_parameter_dimensions except 'f' and 't'
        for each unique dim_col:
            * Calculates quantiles for each hour_of_year of processed years using the quantile mapping.
            * Maps quantiles to 'f' based on quantile_map provided via kwargs

        Raises following errors
            * ValueError if input_df does not have 'time' columns
            * ValueError if input_df cover less than two years
            * ValueError if input_df columns do not have all bb_parameter_dimensions
            * ValueError if bb_parameter_dimensions cover only 'f' and 't'
        """
        # ---- Parameters and checks ----
        # Retrieve mandatory kwargs
        processor_name = kwargs.get('processor_name') # for error messages
        bb_parameter_dimensions = kwargs.get('bb_parameter_dimensions')
        quantile_map = kwargs.get('quantile_map')

        # Check for 'time' column.
        if 'time' not in input_df.columns:
            raise ValueError(f"processor '{processor_name}' does not have 'time' column. Check previous functions.")

        # Ensure time column is datetime and sort by time.
        input_df['time'] = pd.to_datetime(input_df['time'])
        input_df = input_df.sort_values(by='time')

        # Check that data covers more than one year.
        years = input_df['time'].dt.year.unique()
        if len(years) <= 1:
            raise ValueError(f"processor '{processor_name}' DataFrame covers only a year or less. Cannot calculate average year. Use 'calculate_average_year': False in config file?")

        # Check that bb_parameter_dimensions includes both 't' and 'f'
        if not ('t' in bb_parameter_dimensions and 'f' in bb_parameter_dimensions):
            raise ValueError(f"processor '{processor_name}' dimensions '{bb_parameter_dimensions}' do not include 'f' and 't'. Cannot calculate average year. Use 'calculate_average_year': False in config file?")

        # ---- Create helper columns ----
        # Create 'hour_of_year' column (hours elapsed since Jan 1, starting at 1).
        input_df['hour_of_year'] = (
            (input_df['time'] - pd.to_datetime(input_df['time'].dt.year.astype(str) + '-01-01')) /
             pd.Timedelta(hours=1)
        ).astype(int) + 1
        # Only process hours up to 8760 (ignore extra hours from leap years)
        input_df = input_df[input_df['hour_of_year'] <= 8760]

        # Determine additional dimension columns from bb_parameter_dimensions, excluding 'f' and 't'
        dim_cols = [col for col in bb_parameter_dimensions if col != 'f' and col != 't' and col in input_df.columns]
        if set(dim_cols + ['f'] + ['t']) != set(bb_parameter_dimensions):
            raise ValueError(f"Average year calculation did not find all bb_parameter_dimensions: '{bb_parameter_dimensions}' from processor '{processor_name}' DataFrame columns.")
        if not dim_cols:
            raise ValueError(f"processor '{processor_name}' dimensions '{bb_parameter_dimensions}' do not include anything but 'f' and 't'.")

        # ---- Quantile computation ----
        # Vectorized quantile computation:
        # Group by the additional dimensions and 'hour_of_year' then compute the three quantiles.
        df_quant = (
            input_df.groupby(dim_cols + ['hour_of_year'])['value']
            .quantile(list(quantile_map.keys()))
            .reset_index(name='value')
        )
        # Rename the quantile column (it comes from the groupby quantile call)
        df_quant = df_quant.rename(columns={'level_{}'.format(len(dim_cols) + 1): 'quantile'})

        # Create a complete Cartesian product for each group, each quantile, and every hour from 1 to 8760.
        # First, get unique group combinations.
        unique_dims = input_df[dim_cols].drop_duplicates()
        # DataFrame for quantile values.
        quantiles_df = pd.DataFrame({'quantile': list(quantile_map.keys())})
        # DataFrame for hours.
        hours_df = pd.DataFrame({'hour_of_year': np.arange(1, 8761)})

        # Create the full_grid (Cartesian product using a cross join).
        unique_dims['_key'] = 1
        quantiles_df['_key'] = 1
        hours_df['_key'] = 1
        full_grid = unique_dims.merge(quantiles_df, on='_key').merge(hours_df, on='_key')
        full_grid = full_grid.drop('_key', axis=1)

        # Merge the computed quantile results with the complete grid.
        df_full = full_grid.merge(df_quant, on=dim_cols + ['hour_of_year', 'quantile'], how='left')

        # ---- Prepare df_final ----
        # Add the 't' and 'f' columns vectorized.
        df_full['t'] = df_full['hour_of_year'].apply(lambda x: 't' + str(x).zfill(6))
        df_full['f'] = df_full['quantile'].map(quantile_map)

        # Fill missing quantile values with 0.
        df_full['value'] = df_full['value'].fillna(0)
        df_full['value'] = df_full['value'].round(round_precision)

        # Reorder columns to match bb_parameter_dimensions plus 'value'.
        # Here, bb_parameter_dimensions is assumed to include 't', 'f', and the remaining dimension columns.
        df_final = df_full[bb_parameter_dimensions + ['value']]
        return df_final


    def prepare_BB_df(self, df, start_date, country_codes, **kwargs):
        """
        Assumes wide format hourly input DataFrame where value column titles are node names
        
        Prepares the input DataFrame for Backbone-compatible GDX conversion.
            * Creates and processes 't' column if 't' in bb_parameter_dimensions
            * converts tables from wide format to long format
            * Creates and processes 'grid' column if 'grid' in bb_parameter_dimensions
            * Creates and processes 'flow' column if 'flow' in bb_parameter_dimensions
            * Creates and processes 'group' column if 'group' in bb_parameter_dimensions
            * Creates and processes 'f' column if 'f' in bb_parameter_dimensions
            * returns DataFrame based on bb_parameter_dimensions
        
        Raises ValueError if not all bb_parameter_dimensions are in the final returned df
        """
        # picking mandatory kwargs
        processor_name = kwargs.get('processor_name')
        bb_parameter_dimensions = kwargs.get('bb_parameter_dimensions')
        custom_column_value = kwargs.get('custom_column_value')
        # optional kwargs
        is_demand = kwargs.get('is_demand', False)

        # Create 't' column and convert it to t-index
        if 't' in bb_parameter_dimensions:
            # reset index, rename to 'time' if not already named
            df = df.reset_index()
            if 'time' not in df.columns and 'index' in df.columns:
                df = df.rename(columns={'index': 'time'})
            
            # calculate 't' column values 't000001' etc based on default_timeorigin
            default_timeorigin = pd.to_datetime(start_date, dayfirst=True)
            times = pd.to_datetime(df['time'], dayfirst=True)
            first_time = times.iloc[0]
            if first_time != default_timeorigin:
                time_diff = first_time - default_timeorigin
                times = times - time_diff
            unique_times = pd.unique(times)
            time_mapping = {time: 't' + str(i + 1).zfill(6) for i, time in enumerate(unique_times)}
            df['t'] = times.map(time_mapping)

        # Identify value columns based on country codes, treat other as dimensions
        value_vars = [col for col in df.columns if any(code in col for code in country_codes)]
        id_vars = [col for col in df.columns if col not in value_vars]
        # Melt the DataFrame to long format.
        melted_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                            var_name='node', value_name='value')

        # Manage possible grid column.
        if 'grid' in bb_parameter_dimensions:
            if custom_column_value is not None and custom_column_value.get('grid') is not None:
                melted_df['grid'] = custom_column_value['grid']
            else:
                melted_df['grid'] = melted_df['node'].apply(lambda x: x.split('_')[1] if '_' in x else x)

        # Manage possible flow column.
        if 'flow' in bb_parameter_dimensions:
            if custom_column_value is not None and custom_column_value.get('flow') is not None:
                melted_df['flow'] = custom_column_value['flow']
            else:
                melted_df['flow'] = melted_df['node'].apply(lambda x: x.split('_')[1] if '_' in x else x)

        # Manage possible group column.
        if 'group' in bb_parameter_dimensions:
            if custom_column_value is not None and custom_column_value.get('group') is not None:
                melted_df['group'] = custom_column_value['group']
            else:
                melted_df['group'] = "UC_"+melted_df['node']

        # Manage possible f column.
        if 'f' in bb_parameter_dimensions:
            if custom_column_value is not None and custom_column_value.get('f') is not None:
                melted_df['f'] = custom_column_value['f']
            else:
                melted_df['f'] = 'f00'

        # negative influx for demands
        if is_demand:
            melted_df['value'] = melted_df['value'] * -1

        # add year helper column
        melted_df['year'] = melted_df['time'].dt.year

        # add value, year, and time to final_cols
        final_cols = bb_parameter_dimensions.copy()
        final_cols.append('value')
        final_cols.append('year')
        final_cols.append('time')

        # Raise error if some of the final_cols not in melted_df
        missing_cols = set(final_cols) - set(melted_df.columns)
        if missing_cols:
            raise ValueError(f"The following columns are missing in the {processor_name} DataFrame: {missing_cols}")

        # pick only final_cols to returned df, polish, and return
        melted_df = melted_df[final_cols]
        melted_df['value'] = melted_df['value'].fillna(value=0)
        melted_df = melted_df.convert_dtypes()
        return melted_df


    def write_BB_gdx(self, df, output_file, **kwargs):
        """
        Writes the DataFrame to gdx

        Variables: output_file (DataFrame) with 'value' column. Other columns are treated as dimensions.
                   bb_parameter (string) used to create gdx parameter name
                   bb_parameter_dimensions (list of strings) used to filter written columns.  E.g. ['grid', 'node', 'f', 't']
        """
        if df is None:
            print("Abort: No data to write")
            return
        
        # Prepare required parameters
        bb_parameter = kwargs.get('bb_parameter')        
        bb_parameter_dimensions = kwargs.get('bb_parameter_dimensions')

        # add 'value' to final_cols
        final_cols = bb_parameter_dimensions.copy()
        final_cols.append('value')

        # write gdx
        dataframes = {bb_parameter: df}
        to_gdx(dataframes, path=output_file)


    def write_BB_gdx_annual(self, df, **kwargs):
        """
        Splits the processed DataFrame by year, remaps the time labels per year,
        and writes each year's data to a separate GDX file. 
    
        Variables: 
            df (DataFrame): DataFrame with 'value' column. Other columns are treated as dimensions.
            **kwargs: Additional parameters including:
                bb_parameter (string): Used to create GDX parameter name
                bb_parameter_dimensions (list of strings): Used to filter written columns. 
                                                           E.g. ['grid', 'node', 'f', 't']
                gdx_name_suffix (string): Used when generating the output file name
    
        Returns:
            None: Creates GDX files in the output folder
    
        Note: 
            - Drops hours after t8760 to handle leap years
        """
        if df is None:
            print("Abort: No data to write")
            return

        # Prepare required parameters
        bb_parameter = kwargs.get('bb_parameter')        
        bb_parameter_dimensions = kwargs.get('bb_parameter_dimensions')
        gdx_name_suffix = kwargs.get('gdx_name_suffix')

        # iterated years
        groups = df.groupby('year')
        single_year = groups.ngroups == 1

        # add 'value' to final_cols
        final_cols = bb_parameter_dimensions.copy()
        final_cols.append('value')

        # process year-by-year
        for yr, group in groups:
            # format dataframe and drop intermediate columns
            group = group.sort_values(by='t')
        
            # Get unique timestamps and create new mapping
            unique_times = pd.unique(group['t'])

            # If this is a leap year and we have more than 8760 hours
            if len(unique_times) > 8760:
                # Keep only the first 8760 hours
                unique_times = unique_times[:8760]
                # Filter the group to only include rows with times in unique_times
                group = group[group['t'].isin(unique_times)]

            # Create the time mapping
            new_time_mapping = {time: 't' + str(i + 1).zfill(6) for i, time in enumerate(unique_times)}
            group['t'] = group['t'].map(new_time_mapping)
            group_out = group[final_cols]

            # Construct filename
            if single_year:
                file_path = os.path.join(self.output_folder, f'{bb_parameter}_{gdx_name_suffix}.gdx') 
            else: 
                file_path = os.path.join(self.output_folder, f'{bb_parameter}_{gdx_name_suffix}_{yr}.gdx') 

            # Compile the parameter dataframe and write gdx
            dataframes = {bb_parameter: group_out}
            to_gdx(dataframes, path=file_path)


    def import_processor_class(self, processor_name):
        """
        Dynamically imports and returns a processor class from the src package.
        Assumes that the module and the class share the same name as processor_name.
        """
        module_path = f"src.{processor_name}"
        try:
            module = importlib.import_module(module_path)
            processor_class = getattr(module, processor_name)
            return processor_class
        except Exception as e:
            print(f"Error importing processor class '{processor_name}' from module '{module_path}': {e}")
            return None


    def create_other_demands(self, df_annual_demands, other_demands):
        """
        For each (grid, node) combination in df_annual_demands
        where the grid (case-insensitive) is in the set 'other_demands',
        create 8760 rows with columns [grid, node, f, t, value].      
        - 'f' is set to 'f00'.
        - 't' is a sequential time label from t000001 up to t008760.
        - 'value' is calculated as TWh/year * 1e6 / 8760.
        """
        # Filter for rows with unprocessed grid values (using lower-case for comparison)
        df_filtered = df_annual_demands[df_annual_demands["grid"].str.lower().isin(other_demands)]
        rows = []
        # Create t-index for a full year (8760 hours)
        t_index = [f"t{str(i).zfill(6)}" for i in range(1, 8760+1)]
        for _, row in df_filtered.iterrows():
            grid = row["grid"]
            node = row["node"]

            # Calculate hourly value (assume twh/year is numeric). 
            # Negative value for demands. Round to two decimals.
            hourly_value = round(row["twh/year"] * 1e6 / 8760 * -1, 2)
            row = pd.DataFrame({
                "grid": grid,
                "node": node,
                "f": "f00",
                "t": t_index,
                "value": hourly_value
            })
            rows.append(row)
        if rows:
            df_result = pd.concat(rows, ignore_index=True)
        else:
            df_result = pd.DataFrame(columns=["grid", "node", "f", "t", "value"])
        return df_result
        

    def update_import_timeseries_inc(self, output_folder, file_suffix=None, **kwargs):
        """
        Updates the import_timeseries.inc file by generating a GAMS code block that imports
        parameter data from GDX files. The function looks for matching GDX files in the output folder
        based on specified parameter names and patterns, then creates the necessary GAMS code to load
        parameters from these files.
        
        Args:
            output_folder (str): Directory path where GDX files are located and where import_timeseries.inc will be created/updated
            file_suffix (str, optional): Specific suffix for the GDX file. If None, searches for files with standard patterns
            **kwargs: Additional parameters including:
                - bb_parameter (str): Name of the Backbone parameter to import
                - gdx_name_suffix (str): Suffix to be used in the GDX filename
        
        Returns:
            None: Writes content to import_timeseries.inc file in the output_folder
        """        
        # Prepare required parameters
        bb_parameter = kwargs.get('bb_parameter')
        gdx_name_suffix = kwargs.get('gdx_name_suffix')

        # If file_suffix flag is True, search for the specific file.
        if file_suffix is not None:
            filename = os.path.join(output_folder, f'{bb_parameter}_{gdx_name_suffix}_{file_suffix}.gdx')
            if os.path.exists(filename):
                matching_files = filename
            else:
                raise FileNotFoundError(f"{bb_parameter}_{gdx_name_suffix}_{file_suffix}.gdx not found in {output_folder}.")

        else:
            # Check for the two patterns in the output_folder
            # Pattern a: a single file: f'{bb_parameter}_{gdx_name_suffix}.gdx'
            file_a = os.path.join(output_folder, f'{bb_parameter}_{gdx_name_suffix}.gdx')
            if os.path.exists(file_a):
                matching_files = file_a
                file_suffix = None
            else:
                # Pattern b: multiple files, e.g., f'{bb_parameter}_{gdx_name_suffix}_{yr}.gdx' where yr is four digit integer, e.g. 2014
                pattern_b = os.path.join(output_folder, f'{bb_parameter}_{gdx_name_suffix}_[0-9][0-9][0-9][0-9].gdx')
                matching_files = glob.glob(pattern_b)
                if matching_files:
                    file_suffix = "%climateYear%"

        if matching_files is None:
            raise FileNotFoundError(f"{bb_parameter}_{gdx_name_suffix}.gdx or {bb_parameter}_{gdx_name_suffix}_year.gdx not found in {output_folder}.")

        # Creating a text block with a specific structure to read GDX to Backbone
        if file_suffix is None:
            gdx_name = f"{bb_parameter}_{gdx_name_suffix}.gdx"
        else:
            gdx_name = f"{bb_parameter}_{gdx_name_suffix}_{file_suffix}.gdx"

        # Constructing text block: 
        text_block = "\n".join([
            f"$ifthen exist '%input_dir%/{gdx_name}'",
            f"    // If {gdx_name} exists, load input data",
            f"    $$gdxin '%input_dir%/{gdx_name}'",
            f"    $$loaddcm {bb_parameter}",
            "    $$gdxin",
            "$endIf",
            ""
        ])

        # Define the output file path
        output_file = os.path.join(output_folder, 'import_timeseries.inc')

        # Create the file if it does not exist and then append the text_block
        with open(output_file, 'a') as f:
            f.write(text_block + "\n")


    def update_unique_domains(self, df, possible_domains, ts_domains):
        """
        Updates the ts_domains dictionary with unique values from df for each domain in possible_domains.
        
        Parameters:
        - df: DataFrame containing domain columns
        - possible_domains: list of potential domain column names to check in df
        - ts_domains: dictionary to update with unique domain values
        
        Returns:
        - ts_domains: updated dictionary with unique domain values
        """
        # Find which of the possible domains exist in the DataFrame
        domains = [domain for domain in possible_domains if domain in df.columns]
        
        # For each existing domain
        for domain in domains:
            # Find unique values in the domain column
            domain_unique_values = df[domain].unique()
            
            if domain in ts_domains:
                # If the domain already exists in ts_domains, combine the unique values
                # Convert both arrays to sets for easy union operation, then back to numpy array
                ts_domains[domain] = np.array(list(set(ts_domains[domain]) | set(domain_unique_values)))
            else:
                # If this is the first time seeing this domain, just add it to ts_domains
                ts_domains[domain] = domain_unique_values
        
        return ts_domains


    def update_domain_pairs(self, df, domain_pair, ts_domains):
        """
        Extracts unique pairs of domain values from specified columns in a DataFrame
        and updates the ts_domains dictionary with these pairs.

        Parameters:
        - df: pandas.DataFrame containing the domain columns
        - domain_pair: a list containing exactly two domain column names, e.g. ['grid', 'node']
        - ts_domains: dictionary to update with unique domain pairs

        Returns:
        - ts_domains: updated dictionary with unique domain pairs

        Raises:
        - ValueError if domain_pair doesn't contain exactly two items
        """
        # Validate input
        if not isinstance(domain_pair, list) or len(domain_pair) != 2:
            raise ValueError("domain_pair must be a list containing exactly two domain names")

        domain1, domain2 = domain_pair

        # Check if both domains exist in the DataFrame
        for domain in domain_pair:
            if domain not in df.columns:
                # No update if domains don't exist
                return ts_domains

        # Extract unique combinations of domain values
        domain_pairs_df = df[[domain1, domain2]].copy()

        # Drop duplicate pairs
        domain_pairs_df = domain_pairs_df.drop_duplicates()

        # Create key for the domain pair in the dictionary
        pair_key = f"{domain1}_{domain2}"

        # Convert domain pairs to tuple list for easier combination
        new_pairs = list(domain_pairs_df.itertuples(index=False, name=None))

        if not new_pairs:
            # No pairs to add
            return ts_domains

        # Initialize if this is the first time seeing this pair combination
        if pair_key not in ts_domains:
            ts_domains[pair_key] = []

        # Get existing pairs
        existing_pairs = ts_domains[pair_key]

        # Combine existing and new pairs, ensuring uniqueness
        if len(existing_pairs) > 0:
            # Convert to sets of tuples for easy union operation
            all_pairs = list(set(existing_pairs) | set(new_pairs))
        else:
            all_pairs = new_pairs

        # Update ts_domains with the combined unique pairs
        ts_domains[pair_key] = all_pairs

        return ts_domains


    def run(self):
        
        # Secondary_results is an open dictionary for results dropped from timeseries processors
        secondary_results = {}
        # Initialize an empty dictionary to store merged domains from timeseries processors. Will be merged to secondary results.
        ts_domains = {}
        # Define domain titles used in compilations
        domain_titles = ['grid', 'node', 'flow', 'group']
        # Define domain pairs used in compilations
        grid_node = ['grid', 'node']
        flow_node = ['flow', 'node']
        
        # Processes each timeseries type according to specification in config file and creates GDX files for Backbone.
        for ts_key, ts_params in self.timeseries_specs.items():
            # --- Preparing the processing -------------------------------

            # Retrieve mandatory timeseries specifications
            processor_name = ts_params['processor_name']
            bb_parameter = ts_params['bb_parameter']
            bb_parameter_dimensions = ts_params['bb_parameter_dimensions']

            # warning and skip processing if any of above missing
            if processor_name is None:
                print(f"   Warning! {ts_key} does not have parameter 'processor_name'! Will skip processing.")
                continue
            if bb_parameter is None:
                print(f"   Warning! {ts_key} does not have parameter 'bb_parameter'! Will skip processing.")
                continue
            if bb_parameter_dimensions is None:
                print(f"   Warning! {ts_key} does not have parameter 'bb_parameter_dimensions'! Will skip processing.")
                continue
            
            # Optional values with their defaults
            demand_grid = ts_params.get('demand_grid', None)
            custom_column_value = ts_params.get('custom_column_value', None)
            gdx_name_suffix = ts_params.get('gdx_name_suffix', '')
            rounding_precision = ts_params.get('rounding_precision', 0)
            calculate_average_year = ts_params.get('calculate_average_year', False)
            process_only_single_year = ts_params.get('process_only_single_year', False)
            secondary_output_name = ts_params.get('secondary_output_name', None)
            quantile_map = ts_params.get('quantile_map', {0.5: 'f01', 0.1: 'f02', 0.9: 'f03'})

            # Processed values used in file names
            startyear = self.start_date.year
            endyear = self.end_date.year

            # Dynamically import the required processor class.
            processor_class = self.import_processor_class(processor_name)
            if processor_class is None:
                print(f"Skipping the processing of {processor_name} due to import error.")
                continue

            # Prepare kwargs for processors
            kwargs_processor = {'input_folder': self.input_folder,
                                'country_codes': self.country_codes,
                                'start_date': self.start_date,
                                'end_date': self.end_date,
                                'scenario_year': self.scenario_year
            }
            if demand_grid is not None: 
                # filter the demand of requested grid. df_annual_demands are already filtered by scenario and year
                df_filtered_demands = self.df_annual_demands[(self.df_annual_demands['grid'].str.lower() == demand_grid.lower())]
                kwargs_processor['df_annual_demands'] = df_filtered_demands

            # Prepare kwargs for bb conversion
            kwargs_bb_conversion = {'processor_name': processor_name,
                                    'bb_parameter': bb_parameter, 
                                    'bb_parameter_dimensions': bb_parameter_dimensions, 
                                    'custom_column_value': custom_column_value,
                                    'quantile_map': quantile_map,
                                    'gdx_name_suffix': gdx_name_suffix
            }
            if demand_grid is not None: 
                kwargs_bb_conversion['is_demand'] = True


            print(f"\n------ {processor_name} --------------------------------------------------------------- ")
            # --- Processing the main data -------------------------------
            self.log_time("Processing input data")

            # call timeseries processor
            processor = processor_class(**kwargs_processor)
            
            # Run the processor and check if a result was returned.
            result = processor.run()
            
            # Handle an optional second DataFrame and possible no result
            if isinstance(result, tuple) and len(result) == 2:
                summary_df, df_optional = result
                secondary_results[secondary_output_name] = df_optional
            elif result is None:
                print(f"{processor_name} did not return any DataFrame.")
                continue
            else:
                summary_df = result        

            # Trim summary_df and convert to BB format
            summary_df = self.trim_summary(summary_df, round_precision=rounding_precision)       
            summary_df_bb = self.prepare_BB_df(summary_df, self.start_date, self.country_codes, **kwargs_bb_conversion)

            # Update the lists of unique domains, gridNode pairs, and flowNode pairs
            ts_domains = self.update_unique_domains(summary_df_bb, domain_titles, ts_domains)
            ts_domains = self.update_domain_pairs(summary_df_bb, grid_node, ts_domains)
            ts_domains = self.update_domain_pairs(summary_df_bb, flow_node, ts_domains)


            # --- Writing output files
            self.log_time(f"Preparing {processor_name} output files...")

            # Write CSV if write_csv_files
            if self.write_csv_files:
                # Constructing the output csv filename
                if process_only_single_year:
                    output_file_csv = os.path.join(self.output_folder, f'{processor_name}.csv')
                else:
                    output_file_csv = os.path.join(self.output_folder, f'{processor_name}_{startyear}-{endyear}.csv')
                summary_df.to_csv(output_file_csv)
                print(f"   Summary csv written to '{output_file_csv}'")

            # Write annual gdx files      
            self.write_BB_gdx_annual(summary_df_bb, **kwargs_bb_conversion)
            print(f"   Annual GDX files for Backbone written to '{self.output_folder}'")
            
            # Expand timeseries reading instructions in import_timeseries.inc
            self.update_import_timeseries_inc(self.output_folder, **kwargs_bb_conversion)


            # --- Average year ----------------------------------------
            # If average-year calculations are enabled...
            if calculate_average_year:
                self.log_time(f"Calculating {processor_name} average year and preparing output files...")

                # Calculate average file
                avg_df = self.calculate_average_year(summary_df_bb, round_precision=rounding_precision, **kwargs_bb_conversion)

                # Write CSV if write_csv_files
                if self.write_csv_files:
                    if process_only_single_year:
                        output_file_avg_csv = os.path.join(self.output_folder, f'{processor_name}_average_year.csv')
                    else:
                        output_file_avg_csv = os.path.join(self.output_folder, f'{processor_name}_average_year_from_{startyear}-{endyear}.csv')
                    avg_df.to_csv(output_file_avg_csv)
                    print(f"   Average year csv written to '{output_file_avg_csv}'")

                # writing gdx file for forecasts
                output_file_gdx = os.path.join(self.output_folder, f'{bb_parameter}_{gdx_name_suffix}_forecasts.gdx')
                self.write_BB_gdx(avg_df, output_file_gdx, **kwargs_bb_conversion)
                print(f"   GDX for Backbone written to '{output_file_gdx}'")
                
                # Expand timeseries reading instructions in import_timeseries.inc
                self.update_import_timeseries_inc(self.output_folder, file_suffix="forecasts", **kwargs_bb_conversion)


        # --- Process Other Demands Not Yet Processed -------------------------------
        # Query all unique grids in df_annual_demands (convert to lower case for comparison)
        all_demand_grids = set(self.df_annual_demands["grid"].str.lower().unique())

        # Query all 'demand_grid' values from timeseries_specs that were processed previously
        processed_demand_grids = set()
        for ts_params in self.timeseries_specs.values():
            if ts_params.get("demand_grid"):
                processed_demand_grids.add(ts_params["demand_grid"].lower())

        # Determine the unprocessed (other) demands.
        unprocessed_grids = all_demand_grids - processed_demand_grids
        if unprocessed_grids:
            print("\n------ Processing Other Demands --------------------------------------------------------------- ")
            self.log_time("Processing other demands")
            # Creates other demands already in Backbone ready format (grid, node, f, t, value)
            df_other_demands = self.create_other_demands(self.df_annual_demands, unprocessed_grids)

            # Update the lists of unique domains
            ts_domains = self.update_unique_domains(df_other_demands, domain_titles, ts_domains)

            # Update the list of gridNode pairs
            ts_domains = self.update_domain_pairs(df_other_demands, grid_node, ts_domains)

            # Write CSV if write_csv_files
            if self.write_csv_files:
                output_csv = os.path.join(self.output_folder, 'Other_demands_1h_MWh.csv')
                df_other_demands.to_csv(output_csv)
                print(f"   Other demand csv written to {output_csv}")

            # Package and export the other demands DataFrame to a GDX file.
            output_file_other = os.path.join(self.output_folder, 'ts_influx_other_demands.gdx')
            dataframes = {'ts_influx': df_other_demands}
            to_gdx(dataframes, path=output_file_other)
            print(f"   GDX for Other Demands written to {output_file_other}")

            # Expand timeseries reading instructions in import_timeseries.inc
            kwargs_other = {'bb_parameter': 'ts_influx', 
                            'gdx_name_suffix': 'other_demands'
            } 
            self.update_import_timeseries_inc(self.output_folder, **kwargs_other)


        # --- Compile output ----------------------------------------------
        # Merging ts_domains to secondary results
        secondary_results['ts_domains'] = ts_domains
        return secondary_results