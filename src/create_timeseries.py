import os
import time
import importlib
from gdxpds import to_gdx
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import calendar


class create_timeseries:
    def __init__(self, timeseries_specs, input_folder, output_folder, 
                 start_date, end_date, country_codes, scenario, scenario_year, 
                 df_annual_demands, log_start, write_csv_files=False
                 ):
        self.timeseries_specs = timeseries_specs
        self.input_folder = os.path.join(input_folder, "timeseries")
        self.output_folder = output_folder
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.country_codes = country_codes
        self.scenario = scenario
        self.scenario_year = scenario_year
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

    def calculate_average_year(self, summary_df, round_precision=0):
        """
        Computes an average-year DataFrame based on quantiles.
        """
        df = summary_df.reset_index()
        if 'index' in df.columns:
            df['time'] = pd.to_datetime(df['index'])
            df.drop(columns=['index'], inplace=True)
        else:
            df['time'] = pd.to_datetime(df.iloc[:, 0])
            
        # Remove extra rows from leap years (December 31st in leap years).
        df['is_leap'] = df['time'].dt.year.apply(calendar.isleap)
        df = df[~((df['is_leap']) & (df['time'].dt.month == 12) & (df['time'].dt.day == 31))]

        # Create time-based grouping keys.
        df['month'] = df['time'].dt.month
        df['day']   = df['time'].dt.day
        df['hour']  = df['time'].dt.hour

        # Determine extra grouping dimensions (if any)
        extra_grouping = []
        if summary_df.index.nlevels > 1:
            index_names = summary_df.index.names
            extra_grouping = [name for name in index_names[1:] if name]
            
        group_keys = ['month', 'day', 'hour'] + extra_grouping

        exclude = ['time', 'is_leap', 'month', 'day', 'hour'] + extra_grouping
        value_cols = [col for col in df.columns if col not in exclude]

        grouped = df.groupby(group_keys)
        q_levels = [0.1, 0.5, 0.9]
        df_list = []
        for q in q_levels:
            df_q = grouped[value_cols].quantile(q).reset_index()
            df_q['quantile'] = q
            df_list.append(df_q)
        df_avg = pd.concat(df_list, ignore_index=True)

        df_avg.sort_values(by=group_keys + ['quantile'], inplace=True)
        df_avg.reset_index(drop=True, inplace=True)

        # Move quantile column to the desired position
        column_to_move = df_avg.pop("quantile")
        df_avg.insert(3 + len(extra_grouping), "quantile", column_to_move)

        # Round, except quantile columns
        round_cols = [col for col in df_avg.columns if col != 'quantile']
        df_avg[round_cols] = df_avg[round_cols].round(round_precision)
        
        # Convert dtypes (so that e.g. rounding to 0 decimals gives integers)
        df_avg = df_avg.convert_dtypes()

        return df_avg


    def prepare_BB_df(self, df, start_date, country_codes, bb_parameter,
                       add_column=None, add_column_bb_name=None,
                       replace_grid_title=None, replace_grid_values=None,
                       is_demand=False):
        """
        Prepares the DataFrame for Backbone-compatible GDX conversion.
        Processes time columns, renames/melts the DataFrame, and returns the modified DataFrame.

        Note: assuming hourly input data
        """
        if not bb_parameter:
            print("Abort: bb_parameter is empty")
            return None

        default_timeorigin = pd.to_datetime(start_date, dayfirst=True)

        # Process time information
        if set(['month', 'day', 'hour']).issubset(df.columns):
            df['dt'] = pd.to_datetime({'year': 2000, 'month': df['month'],
                                       'day': df['day'], 'hour': df['hour']})
            df['time_dt'] = df['dt']
            df['year'] = df['dt'].dt.year
            origin = df['dt'].iloc[0]
            df['hour_diff'] = ((df['dt'] - origin) / pd.Timedelta(hours=1)).astype(int)
            df['time'] = df['hour_diff'].apply(lambda x: 't' + f"{x+1:06d}")
            df.drop(columns=['month', 'day', 'hour', 'dt', 'hour_diff'], inplace=True)
        else:
            df = df.reset_index()
            if 'time' not in df.columns and 'index' in df.columns:
                df = df.rename(columns={'index': 'time'})
            times = pd.to_datetime(df['time'], dayfirst=True)
            df['time_dt'] = times
            df['year'] = times.dt.year
            first_time = times.iloc[0]
            if first_time != default_timeorigin:
                time_diff = first_time - default_timeorigin
                times = times - time_diff
            unique_times = pd.unique(times)
            time_mapping = {time: 't' + str(i + 1).zfill(6) for i, time in enumerate(unique_times)}
            df['time'] = times.map(time_mapping)

        # Handle quantile column conversion
        if 'quantile' in df.columns:
            df.rename(columns={'quantile': 'f'}, inplace=True)
            mapping = {0.1: 'f03', 0.5: 'f01', 0.9: 'f02'} if is_demand else {0.1: 'f02', 0.5: 'f01', 0.9: 'f03'}
            df['f'] = df['f'].map(mapping)
        else:
            df['f'] = 'f00'

        # Identify value columns based on country codes
        value_vars = [col for col in df.columns if any(code in col for code in country_codes)]
        id_vars = [col for col in df.columns if col not in value_vars]

        if is_demand:
            for col in value_vars:
                df[col] = -df[col]

        # Ensure that 'year' and 'time_dt' are retained.
        if 'year' not in id_vars:
            id_vars.append('year')
        if 'time_dt' not in id_vars:
            id_vars.append('time_dt')

        # Melt the DataFrame to long format.
        melted_df = pd.melt(df, id_vars=id_vars, value_vars=value_vars,
                            var_name='node', value_name='value')

        # Manage the grid column.
        if (replace_grid_title and not replace_grid_values) or (not replace_grid_title and replace_grid_values):
            print("Abort: Provide both replace_grid_title and replace_grid_values!")
            return None
        if replace_grid_title is not None:
            melted_df[replace_grid_title] = replace_grid_values
            grid_name = replace_grid_title
        else:
            melted_df['grid'] = melted_df['node'].apply(lambda x: x.split('_')[1] if '_' in x else x)
            grid_name = 'grid'

        # Process additional column if provided.
        if add_column and (add_column not in melted_df.columns):
            print(f"Abort: Additional column {add_column} not found in the input DataFrame")
            return None
        if (add_column and not add_column_bb_name) or (not add_column and add_column_bb_name):
            print("Abort: Provide both add_column and add_column_bb_name!")
            return None
        if add_column and add_column in melted_df.columns and add_column_bb_name:
            final_cols = [grid_name, 'node', add_column, 'f', 'value', 'year', 'time_dt']
            melted_df = melted_df[final_cols]
            melted_df.rename(columns={add_column: add_column_bb_name}, inplace=True)
        else:
            final_cols = [grid_name, 'node', 'f', 'value', 'year', 'time_dt']
            melted_df = melted_df[final_cols]

        return melted_df


    def write_BB_annual_gdx(self, df, output_file, bb_parameter, replace_grid_title=None, add_column=None, add_column_bb_name=None):
        """
        Splits the processed DataFrame by year, remaps the time labels per year,
        and writes each year's data to a separate GDX file.

        Uses the custom grid column name if provided via replace_grid_title; otherwise, defaults to 'grid'.
        """
        if df is None:
            print("Abort: No data to write")
            return

        # Dynamically determine the grid column name.
        grid_name = replace_grid_title if replace_grid_title is not None else 'grid'

        base, ext = os.path.splitext(output_file)
        groups = df.groupby('year')
        single_year = groups.ngroups == 1

        # Determine final column order based on whether an additional column is used.
        if add_column and add_column_bb_name and add_column_bb_name in df.columns:
            final_order = [grid_name, 'node', add_column_bb_name, 'f', 't', 'value']
        else:
            final_order = [grid_name, 'node', 'f', 't', 'value']

        for yr, group in groups:
            group = group.sort_values(by='time_dt')
            unique_times = pd.unique(group['time_dt'])
            new_time_mapping = {time: 't' + str(i + 1).zfill(6) for i, time in enumerate(unique_times)}
            group['t'] = group['time_dt'].map(new_time_mapping)
            group_out = group.drop(columns=['year', 'time_dt'])
            group_out = group_out[final_order]

            # Fill null values in required columns.
            for col in final_order:
                if group_out[col].isnull().any():
                    default_val = 0 if pd.api.types.is_numeric_dtype(group_out[col]) else ""
                    group_out[col] = group_out[col].fillna(default_val)
                    print(f"Null values in column '{col}' for year {yr} filled with '{default_val}'")

            file_path = output_file if single_year else f"{base}_{yr}{ext}"
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
        

    def run(self):
        
        ## Calculating first timesteps of each year and writing them to a gdx
        #output_file = os.path.join(self.output_folder, f'year_starts.gdx')
        #print(f"Writing the first timestep of each year to {output_file}")
        #self.create_year_start_gdx(self.start_date, self.end_date, output_file)
        

        # Processes each timeseries type according to specification in config file and creates GDX files for Backbone.
        for ts_key, ts_params in self.timeseries_specs.items():
            # Retrieve timeseries specifications, use default assumptions in case of missing specs
            processor_name = ts_params['processor_name']
            bb_parameter = ts_params['bb_parameter']
            gdx_name_suffix = ts_params['gdx_name_suffix']
            demand_grid = ts_params.get('demand_grid', None)
            rounding_precision = ts_params.get('rounding_precision', 0)
            calculate_average_year = ts_params.get('calculate_average_year', False)
            process_only_single_year = ts_params.get('process_only_single_year', False)
            add_column = ts_params.get('add_column', None)
            add_column_bb_name = ts_params.get('add_column_bb_name', None)
            replace_grid_title = ts_params.get('replace_grid_title', None)
            replace_grid_values = ts_params.get('replace_grid_values', None)

            print(f"\n------ {processor_name} --------------------------------------------------------------- ")
            self.log_time("Processing input data")

            # Dynamically import the required processor class.
            processor_class = self.import_processor_class(processor_name)
            if processor_class is None:
                print(f"Skipping the processing of {processor_name} due to import error.")
                continue

            # Adjust end_date if processing only one year.
            current_end_date = self.end_date
            if process_only_single_year:
                current_end_date = self.start_date + relativedelta(years=1, hours=-1)

            # If a demand grid is defined, ...
            if demand_grid is not None:
                # filter the demand of requested grid. df_annual_demands are already filtered by scenario and year
                df_filtered_demands = self.df_annual_demands[(self.df_annual_demands['grid'].str.lower() == demand_grid.lower())]

                # call timeseries processing with additional parameters
                processor = processor_class(self.input_folder, self.country_codes,
                                            self.start_date, current_end_date,
                                            df_filtered_demands, self.scenario_year)

            else:
                # call timeseries processing with default parameters
                processor = processor_class(self.input_folder, self.country_codes,
                                            self.start_date, current_end_date)
            
            # Run the processor
            summary_df = processor.run()

            # File naming based on start/end year.
            startyear = self.start_date.year
            endyear = current_end_date.year
            self.log_time(f"Preparing {processor_name} output files...")

            # Trim summary_df
            summary_df = self.trim_summary(summary_df, round_precision=rounding_precision)

            # Write CSV if write_csv_files
            if self.write_csv_files:
                if process_only_single_year:
                    output_file_csv = os.path.join(self.output_folder, f'{processor_name}_1h_MWh.csv')
                else:
                    output_file_csv = os.path.join(self.output_folder, f'{processor_name}_{startyear}-{endyear}_1h_MWh.csv')
                summary_df.to_csv(output_file_csv)
                print(f"   Summary csv written to {output_file_csv}")

            # Prepare arguments and call BB conversion.
            kwargs = {'bb_parameter': bb_parameter, 
                      'add_column': add_column, 'add_column_bb_name': add_column_bb_name,
                      'replace_grid_title': replace_grid_title, 'replace_grid_values': replace_grid_values
                      }
            if demand_grid is not None: 
                kwargs['is_demand'] = True
            output_file_gdx = os.path.join(self.output_folder, f'{bb_parameter}_{gdx_name_suffix}.gdx')
            processed_df = self.prepare_BB_df(summary_df, self.start_date, self.country_codes, **kwargs)
            # writing annual gdx files
            self.write_BB_annual_gdx(processed_df, output_file_gdx, bb_parameter, replace_grid_title, add_column, add_column_bb_name)

            print(f"   GDX for Backbone written to {output_file_gdx}")

            # If average-year calculations are enabled...
            if calculate_average_year:
                self.log_time(f"Calculating {processor_name} average year and preparing output files...")

                # Calculate average file
                avg_df = self.calculate_average_year(summary_df, round_precision=rounding_precision)

                # Write CSV if write_csv_files
                if self.write_csv_files:
                    if process_only_single_year:
                        output_file_avg_csv = os.path.join(self.output_folder, f'{processor_name}_average_year_1h_MWh.csv')
                    else:
                        output_file_avg_csv = os.path.join(self.output_folder, f'{processor_name}_average_year_from_{startyear}-{endyear}_1h_MWh.csv')
                    avg_df.to_csv(output_file_avg_csv)
                    print(f"   Average year csv written to {output_file_avg_csv}")

                # call BB conversion with previously created arguments
                output_file_gdx = os.path.join(self.output_folder, f'{bb_parameter}_{gdx_name_suffix}_forecasts.gdx')
                processed_df = self.prepare_BB_df(summary_df, self.start_date, self.country_codes, **kwargs)
                # writing annual gdx files
                self.write_BB_annual_gdx(processed_df, output_file_gdx, bb_parameter, replace_grid_title, add_column, add_column_bb_name)
                print(f"   GDX for Backbone written to {output_file_gdx}")


        # --- Process Other Demands Not Yet Processed ---
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
