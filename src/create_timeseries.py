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

        Variables: df (DataFrame) with 'value' column. Other columns are treated as dimensions.
                   bb_parameter (string) used to create gdx parameter name
                   bb_parameter_dimensions (list of strings) used to filter written columns.  E.g. ['grid', 'node', 'f', 't']
                   gdx_name_suffix (string) used when generating the output file name
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
            unique_times = pd.unique(group['t'])
            new_time_mapping = {time: 't' + str(i + 1).zfill(6) for i, time in enumerate(unique_times)}
            group['t'] = group['t'].map(new_time_mapping)
            group_out = group[final_cols]

            # Construct filename
            if single_year:
                file_path = os.path.join(self.output_folder, f'{bb_parameter}_{gdx_name_suffix}.gdx') 
            else: 
                file_path = os.path.join(self.output_folder, f'{bb_parameter}_{gdx_name_suffix}_{yr}.gdx') 

            # write gdx
            dataframes = {bb_parameter: group_out}
            to_gdx(dataframes, path=file_path)


    def create_year_start_gdx(self, start_date, end_date, output_file):
            # Create an hourly datetime index
            dt_index = pd.date_range(start=start_date, end=end_date, freq='h')

            # Create a "t index" where the first hour is t000001, second t000002, etc.
            t_index = [f't{str(i).zfill(6)}' for i in range(1, len(dt_index)+1)]

            # Combine into a DataFrame
            df = pd.DataFrame({'datetime': dt_index, 't': t_index})

            # Pick the first hour of each year (January 1 at 00:00)
            first_hours = df[(df['datetime'].dt.month == 1) &
                             (df['datetime'].dt.day == 1) &
                             (df['datetime'].dt.hour == 0)]

            # Create the new DataFrame with columns 'year' and 't'
            year_starts = pd.DataFrame({
                'year': first_hours['datetime'].dt.year.apply(lambda x: f'y{x}'),
                't': first_hours['t']
            }).reset_index(drop=True)

            # Add the required 'value' column for a set, with True for each element
            year_starts['value'] = True

            # Package and export the DataFrame to a GDX file.
            dataframes = {'year_starts': year_starts}
            to_gdx(dataframes, path=output_file)


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
        For each (country, grid, node_suffix) combination in df_annual_demands
        where the grid (case-insensitive) is in the set 'other_demands',
        create 8760 rows with columns [grid, node, f, t, value].
        
        - 'node' is constructed as <country>_<grid>_<node_suffix> if node_suffix is provided,
          otherwise just <country>_<grid>.
        - 'f' is set to 'f00'.
        - 't' is a sequential time label from t000001 up to t008760.
        - 'value' is calculated as TWh/year * 1e6 / 8760.
        """
        # Filter for rows with unprocessed grid values (using lower-case for comparison)
        df_filtered = df_annual_demands[df_annual_demands["grid"].str.lower().isin(other_demands)]
        result_frames = []
        # Create t-index for a full year (8760 hours)
        t_index = [f"t{str(i).zfill(6)}" for i in range(1, 8760+1)]
        for _, row in df_filtered.iterrows():
            country = row["country"]
            grid = row["grid"]
            node_suffix = row.get("node_suffix", "")
            # Build node name <country>_<grid> or <country>_<grid>_<node_suffix>
            if pd.notna(node_suffix) and str(node_suffix).strip() != "":
                node = f"{country}_{grid}_{node_suffix}"
            else:
                node = f"{country}_{grid}"
            # Calculate hourly value (assume twh/year is numeric). 
            # Negative value for demands. Round to two decimals.
            hourly_value = round(row["twh/year"] * 1e6 / 8760 * -1, 2)
            df_temp = pd.DataFrame({
                "grid": grid,
                "node": node,
                "f": "f00",
                "t": t_index,
                "value": hourly_value
            })
            result_frames.append(df_temp)
        if result_frames:
            df_result = pd.concat(result_frames, ignore_index=True)
        else:
            df_result = pd.DataFrame(columns=["grid", "node", "f", "t", "value"])
        return df_result
        

    def update_import_timeseries_inc(self, output_folder, file_suffix=None, **kwargs):
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
            text_block = "\n".join([
                f"// If {bb_parameter}_{gdx_name_suffix} exists",
                f"$ifthen exist '%input_dir%/{bb_parameter}_{gdx_name_suffix}.gdx'",
                "    // load input data",
                f"    $$gdxin '%input_dir%/{bb_parameter}_{gdx_name_suffix}.gdx'",
                f"    $$loaddcm '{bb_parameter}'",
                "    $$gdxin",
                "$endIf",
                ""
            ])
        else:
            text_block = "\n".join([
                f"// If {bb_parameter}_{gdx_name_suffix}_{file_suffix} exists",
                f"$ifthen exist '%input_dir%/{bb_parameter}_{gdx_name_suffix}_{file_suffix}.gdx'",
                "    // load input data",
                f"    $$gdxin '%input_dir%/{bb_parameter}_{gdx_name_suffix}_{file_suffix}.gdx'",
                f"    $$loaddcm '{bb_parameter}'",
                "    $$gdxin",
                "$endIf",
                ""
            ])

        # Define the output file path
        output_file = os.path.join(output_folder, 'import_timeseries.inc')

        # Create the file if it does not exist and then append the text_block
        with open(output_file, 'a') as f:
            f.write(text_block + "\n")


    def run(self):
        
        ## Calculating first timesteps of each year and writing them to a gdx
        #output_file = os.path.join(self.output_folder, f'year_starts.gdx')
        #print(f"Writing the first timestep of each year to {output_file}")
        #self.create_year_start_gdx(self.start_date, self.end_date, output_file)
        
        secondary_results = {}
        # Processes each timeseries type according to specification in config file and creates GDX files for Backbone.
        for ts_key, ts_params in self.timeseries_specs.items():
            # --- Preparing the processing -------------------------------
            # Retrieve mandatory timeseries specifications
            processor_name = ts_params['processor_name']
            bb_parameter = ts_params['bb_parameter']
            bb_parameter_dimensions = ts_params['bb_parameter_dimensions']
            # warning and skip if any of above missing
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

            print(f"\n------ {processor_name} --------------------------------------------------------------- ")
            self.log_time("Processing input data")

            # --- Processing the main data -------------------------------
            # Dynamically import the required processor class.
            processor_class = self.import_processor_class(processor_name)
            if processor_class is None:
                print(f"Skipping the processing of {processor_name} due to import error.")
                continue

            # If a demand grid is defined, ...
            if demand_grid is not None:
                # filter the demand of requested grid. df_annual_demands are already filtered by scenario and year
                df_filtered_demands = self.df_annual_demands[(self.df_annual_demands['grid'].str.lower() == demand_grid.lower())]

                # call timeseries processing with additional parameters df_filtered_demands and self.scenario_year
                processor = processor_class(self.input_folder, self.country_codes,
                                            self.start_date, self.end_date,
                                            df_filtered_demands, self.scenario_year)
            else:
                # call timeseries processing with default parameters
                processor = processor_class(self.input_folder, self.country_codes,
                                            self.start_date, self.end_date)
            
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

            # Trim summary_df
            summary_df = self.trim_summary(summary_df, round_precision=rounding_precision)       

            # --- Writing output files -------------------------------
            self.log_time(f"Preparing {processor_name} output files...")
            # Write CSV if write_csv_files
            if self.write_csv_files:
                # File naming based on start/end year.
                startyear = self.start_date.year
                endyear = self.end_date.year
                # Constructing the output csv filename
                if process_only_single_year:
                    output_file_csv = os.path.join(self.output_folder, f'{processor_name}_1h_MWh.csv')
                else:
                    output_file_csv = os.path.join(self.output_folder, f'{processor_name}_{startyear}-{endyear}_1h_MWh.csv')
                summary_df.to_csv(output_file_csv)
                print(f"   Summary csv written to {output_file_csv}")

            # Prepare arguments.
            kwargs = {'processor_name': processor_name,
                      'bb_parameter': bb_parameter, 
                      'bb_parameter_dimensions': bb_parameter_dimensions, 
                      'custom_column_value': custom_column_value,
                      'quantile_map': quantile_map,
                      'gdx_name_suffix': gdx_name_suffix
                      }
            if demand_grid is not None: 
                kwargs['is_demand'] = True
              
            # Call BB conversion.
            summary_df = self.prepare_BB_df(summary_df, self.start_date, self.country_codes, **kwargs)
            # write annual gdx files.      
            self.write_BB_gdx_annual(summary_df, **kwargs)
            # add reading instructions to import_timeseries.inc
            self.update_import_timeseries_inc(self.output_folder, **kwargs)
            print(f"   Annual GDX files for Backbone written to {self.output_folder}")

            # --- Average year calculations -------------------------------
            # If average-year calculations are enabled...
            if calculate_average_year:
                self.log_time(f"Calculating {processor_name} average year and preparing output files...")

                # Calculate average file
                avg_df = self.calculate_average_year(summary_df, round_precision=rounding_precision, **kwargs)

                # Write CSV if write_csv_files
                if self.write_csv_files:
                    if process_only_single_year:
                        output_file_avg_csv = os.path.join(self.output_folder, f'{processor_name}_average_year_1h_MWh.csv')
                    else:
                        output_file_avg_csv = os.path.join(self.output_folder, f'{processor_name}_average_year_from_{startyear}-{endyear}_1h_MWh.csv')
                    avg_df.to_csv(output_file_avg_csv)
                    print(f"   Average year csv written to {output_file_avg_csv}")

                # writing gdx file for forecasts
                output_file_gdx = os.path.join(self.output_folder, f'{bb_parameter}_{gdx_name_suffix}_forecasts.gdx')
                self.write_BB_gdx(avg_df, output_file_gdx, **kwargs)
                self.update_import_timeseries_inc(self.output_folder, file_suffix="forecasts", **kwargs)
                print(f"   GDX for Backbone written to {output_file_gdx}")


        # --- Process Other Demands Not Yet Processed -------------------------------
        # Query all unique grids in df_annual_demands (convert to lower case for comparison)
        all_demand_grids = set(self.df_annual_demands["grid"].str.lower().unique())
        # Query all 'demand_grid' values from timeseries_specs that were already processed
        processed_demand_grids = set()
        for ts_params in self.timeseries_specs.values():
            if ts_params.get("demand_grid"):
                processed_demand_grids.add(ts_params["demand_grid"].lower())
        # Determine the unprocessed (other) demands.
        unprocessed_grids = all_demand_grids - processed_demand_grids
        if unprocessed_grids:
            print("\n------ Processing Other Demands --------------------------------------------------------------- ")
            self.log_time("Processing other demands")
            df_other_demands = self.create_other_demands(self.df_annual_demands, unprocessed_grids)

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

        return secondary_results