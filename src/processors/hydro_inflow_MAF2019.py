import os
import calendar
import pandas as pd
import numpy as np
from src.utils import log_status

class hydro_inflow_MAF2019:
    """
    Class to process hydro inflows (reservoir, pump open cycle, and run-of-river) data.

    Parameters:
        input_folder (str): relative location of input files.
        country_codes (list): List of country codes.
        start_date (str): Start datetime (e.g., '1982-01-01 00:00:00').
        end_date (str): End datetime (e.g., '2021-01-01 00:00:00').
    """

    def __init__(self, **kwargs_processor):
        # List of required parameters
        required_params = [
            'input_folder', 
            'country_codes', 
            'start_date', 
            'end_date', 
        ]

        # Check if all required parameters are present
        missing_params = [param for param in required_params if param not in kwargs_processor]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Unpack parameters
        self.input_folder = kwargs_processor['input_folder']
        self.country_codes = kwargs_processor['country_codes']
        self.start_date = kwargs_processor['start_date']
        self.end_date = kwargs_processor['end_date']

        # Extract start and end years from the provided dates.
        self.startyear = pd.to_datetime(self.start_date).year
        self.endyear = pd.to_datetime(self.end_date).year

        # Define folders and file paths.
        self.file_weekly = os.path.join(self.input_folder, 'PECD-hydro-weekly-inflows-corrected.csv')
        self.file_daily = os.path.join(self.input_folder, 'PECD-hydro-daily-ror-generation.csv')

        # Define column headers and suffixes.
        self.inflow1_header = 'Cumulated inflow into reservoirs per week in GWh'
        self.inflow2_header = 'Cumulated NATURAL inflow into the pump-storage reservoirs per week in GWh'
        self.inflow3_header = 'Run of River Hydro Generation in GWh per day'
        self.inflow1_suffix = '_reservoir'
        self.inflow2_suffix = '_psOpen'
        self.inflow3_suffix = '_ror'

        # Initialize log message list
        self.processor_log = []

    def _process_reservoir_inflows(self, weekly_df):
        """
        Processes the weekly (reservoir and pump‐storage) data for each country.
        Returns one combined DataFrame with appropriately suffixed columns.
        """
        # Create a full hourly index for the given date range.
        full_index = pd.date_range(self.start_date, self.end_date, freq='60min')
        end_ts = pd.Timestamp(self.end_date)
        result_df = pd.DataFrame(index=full_index)

        for country in self.country_codes:
            col_name1 = country + self.inflow1_suffix
            col_name2 = country + self.inflow2_suffix

            # Filter for the country's data.
            df_country = weekly_df[weekly_df["zone"] == country].copy()
            if df_country.empty:
                result_df[col_name1] = np.nan
                result_df[col_name2] = np.nan
                continue

            # Sort by year and week, and fill missing values.
            df_country.sort_values(by=['year', 'week'], inplace=True)
            df_country.fillna(0, inplace=True)

            # Prepare lists to collect timestamps and inflow values.
            ts_list = []
            inflow1_list = []
            inflow2_list = []

            # Process data grouped by year.
            for year, group in df_country.groupby('year'):
                group = group.reset_index(drop=True)
                n = len(group)
                # For weekly data, use January 4th at 12:00 as the reference.
                fourthday = pd.Timestamp(year, 1, 4) + pd.Timedelta(hours=12)

                # Process "normal" weeks (up to week 52).
                normal_count = min(n, 52)
                if normal_count > 0:
                    i_array = np.arange(normal_count)
                    t_array = fourthday + i_array * pd.Timedelta(days=7)
                    valid_mask = t_array <= end_ts
                    t_valid = t_array[valid_mask]

                    values1 = (1000 * group[self.inflow1_header].iloc[:normal_count].to_numpy() / 168)[valid_mask]
                    values2 = (1000 * group[self.inflow2_header].iloc[:normal_count].to_numpy() / 168)[valid_mask]

                    ts_list.extend(t_valid)
                    inflow1_list.extend(values1)
                    inflow2_list.extend(values2)

                # Handle the potential extra (53rd) week.
                if n > 52:
                    leap_t = pd.Timestamp(year, 12, 28) + pd.Timedelta(hours=12)
                    if leap_t <= end_ts:
                        leap_val1 = 1000 * group[self.inflow1_header].iloc[51] / 168
                        leap_val2 = 1000 * group[self.inflow2_header].iloc[51] / 168
                        ts_list.append(leap_t)
                        inflow1_list.append(leap_val1)
                        inflow2_list.append(leap_val2)

            # Build Series for each inflow type.
            s_inflow1 = pd.Series(data=inflow1_list, index=ts_list)
            s_inflow2 = pd.Series(data=inflow2_list, index=ts_list)

            # Remove duplicate timestamps and sort the series.
            s_inflow1 = s_inflow1[~s_inflow1.index.duplicated(keep='first')].sort_index()
            s_inflow2 = s_inflow2[~s_inflow2.index.duplicated(keep='first')].sort_index()

            # Reindex to the full hourly index and interpolate missing values.
            s_inflow1 = s_inflow1.reindex(full_index).interpolate(limit=84, limit_direction='both')
            s_inflow2 = s_inflow2.reindex(full_index).interpolate(limit=84, limit_direction='both')

            result_df[col_name1] = s_inflow1
            result_df[col_name2] = s_inflow2

        # Drop empty columns, return result_df
        result_df = result_df.loc[:, result_df.sum() != 0]
        return result_df


    def _process_ror_inflows(self, daily_df):
        """
        Processes daily run‐of‐river data for each country.
        Returns one combined DataFrame with appropriately suffixed columns.
        """
        full_index = pd.date_range(self.start_date, self.end_date, freq='60min')
        end_ts = pd.to_datetime(self.end_date)
        result_df = pd.DataFrame(index=full_index)

        for country in self.country_codes:
            col_name = country + self.inflow3_suffix

            # Filter for the country's data.
            df_country = daily_df[daily_df["zone"] == country].copy()
            if df_country.empty:
                result_df[col_name] = np.nan
                continue

            # Sort and fill missing values.
            df_country.sort_values(by=['year', 'Day'], inplace=True)
            df_country.fillna(0, inplace=True)

            # Compute the day offset within each year.
            df_country['day_offset'] = df_country.groupby('year').cumcount()

            # Compute the timestamp for each row.
            df_country['timestamp'] = (
                pd.to_datetime(df_country['year'].astype(str), format='%Y') +
                pd.to_timedelta(df_country['day_offset'], unit='D') +
                pd.Timedelta(hours=12)
            )

            # Compute the flow values.
            df_country['value'] = 1000 * df_country[self.inflow3_header] / 24

            # Handle leap years: add an extra record for December 31 at 12:00 if needed.
            leap_rows = []
            for yr, grp in df_country.groupby('year'):
                if calendar.isleap(yr) and len(grp) > 364:
                    leap_ts = pd.Timestamp(yr, 12, 31, 12)
                    if leap_ts <= end_ts:
                        leap_val = 1000 * grp.iloc[364][self.inflow3_header] / 24
                        leap_rows.append({'timestamp': leap_ts, 'value': leap_val})

            ts_series = pd.Series(df_country['value'].values, index=df_country['timestamp'])
            ts_series = ts_series[ts_series.index <= end_ts]

            if leap_rows:
                leap_df = pd.DataFrame(leap_rows)
                leap_series = pd.Series(leap_df['value'].values, index=leap_df['timestamp'])
                ts_series = pd.concat([ts_series, leap_series])

            ts_series = ts_series[~ts_series.index.duplicated(keep='first')].sort_index()
            ts_series = ts_series.reindex(full_index)
            ts_series = ts_series.interpolate(limit=12, limit_direction='both')

            result_df[col_name] = ts_series

        # Drop empty columns, return result_df
        result_df = result_df.loc[:, result_df.sum() != 0]
        return result_df


    def run_processor(self):
        """
        Executes the full processing pipeline: reading input files, processing the inflow data,
        combining results, and writing the output CSV.
        """
        log_status("Reading input files...", self.processor_log)
        # Read the weekly CSV file and filter by year.
        weekly_df = pd.read_csv(self.file_weekly)
        weekly_df = weekly_df[(weekly_df["year"] >= self.startyear) & (weekly_df["year"] <= self.endyear)]
        weekly_df["year"] = pd.to_numeric(weekly_df["year"])
        weekly_df["week"] = pd.to_numeric(weekly_df["week"])

        # Read the daily CSV file and filter by year.
        daily_df = pd.read_csv(self.file_daily)
        daily_df = daily_df[(daily_df["year"] >= self.startyear) & (daily_df["year"] <= self.endyear)]
        daily_df["year"] = pd.to_numeric(daily_df["year"])
        daily_df["Day"] = pd.to_numeric(daily_df["Day"])

        log_status("Processing reservoir inflows for all countries...", self.processor_log)
        reservoir_all = self._process_reservoir_inflows(weekly_df)

        log_status("Processing run-of-river inflows for all countries...", self.processor_log)
        ror_all = self._process_ror_inflows(daily_df)

        # Combine the two DataFrames (they share the same hourly index).
        summary_df = pd.concat([reservoir_all, ror_all], axis=1)

        # Mandatory secondary results
        secondary_result = None

        log_status("Inflow time series built.", self.processor_log, level="info")

        # Note: returning processor log as a string, because then we can distinct it from secondary results which might be a list of strings
        return summary_df, secondary_result, "\n".join(self.processor_log)
    

# __main__ allows testing by calling this .py file directly. It contains some minimum data tables for testing.
if __name__ == '__main__':
    # Define the input parameters.
    input_folder = os.path.join("..\\inputFiles\\timeseries")
    output_folder = os.path.join("..\\inputData-test")
    output_file = os.path.join(output_folder, f'test_hydro_inflow.csv')
    country_codes = [
        'FI00', 'NOM1', 'PL00', 'SE01'
    ]
    start_date = '1982-01-01 00:00:00'
    end_date = '2021-01-01 00:00:00'

    kwargs_processor = {'input_folder': input_folder,
                        'country_codes': country_codes,
                        'start_date': start_date,
                        'end_date': end_date
    }

    # Create an instance of process_inflows and run the processing.
    processor = hydro_inflow_MAF2019(**kwargs_processor)
    summary_df = processor.run_processor()

    # Ensure the output directory exists.
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Write to a csv.
    print(f"writing {output_file}")
    summary_df.to_csv(output_file)