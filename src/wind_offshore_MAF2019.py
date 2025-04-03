import os
import calendar
import pandas as pd
import numpy as np


class wind_offshore_MAF2019:
    """
    Class to process wind offshore capacity factor timeseries for countries.

    Parameters:
        input_folder (str): Relative location of input files.
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
            'end_date'
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

    # Function to adjust date using row values.
    def adjust_date(self, row):
        # Extract and adjust the month
        y = int(row['Year'])
        m = int(row['Month'])
        d = int(row['Day'])
        h = int(row['Hour'])
        # If the year is leap and the month is after February, shift day by -1.
        if calendar.isleap(y) and m > 2:
            d = d - 1
            if d < 1:
                # If day becomes zero, roll back to the last day of the previous month.
                m = m - 1
                d = calendar.monthrange(y, m)[1]

        ts = pd.Timestamp(year=y, month=m, day=d, hour=h)
    
        # For leap years, if the row represents December 31
        # then return two timestamps: the one we computed, plus one with an extra day.
        if calendar.isleap(y) and int(row['Month']) == 12 and int(row['Day']) == calendar.monthrange(y, m)[1]:
            # In our leap-year adjustment the day gets shifted. Here, we assume that this row
            # (which originally was December 31) should be “duplicated.”
            extra_ts = ts + pd.Timedelta(days=1)
            return [ts, extra_ts]
    
        return [ts]

    def run(self):
        # Check the input file
        input_file = os.path.join(self.input_folder, "PECD-MAF2019-wide-WindOffshore.csv")
        print(f"   Reading wind offshore cf data from '{input_file}'.. ")
        if not os.path.isfile(input_file):
            print(f"Warning! Input file '{input_file}' not found, skipping wind offshore time series!!!")
            return

        # read the input file
        df_cf_ts = pd.read_csv(input_file, sep=";", decimal=",")
        year_cols = df_cf_ts.columns[3:]
        # Drop areas that are not in country_codes
        df_cf_ts = df_cf_ts[df_cf_ts["area"].isin(self.country_codes)]

        print(f"   Processing the input file.. ")
        # Melt the DataFrame so that the years become a single 'year' column with corresponding values.
        df_melted = df_cf_ts.melt(
            id_vars=["area", "day", "month", "hour"],
            value_vars=year_cols,
            var_name="year",
            value_name="value"
        )

        # Ensure that the time-related columns are numeric.
        for col in ["year", "month", "day", "hour"]:
            df_melted[col] = df_melted[col].astype(int)

        # Pivot so that each unique time appears once with areas as columns.
        df_pivot = df_melted.pivot_table(
            index=["year", "month", "day", "hour"],
            columns="area",
            values="value"
        ).reset_index()

        # Rename the time component columns:
        df_pivot.rename(columns={
        "year": "Year",
        "month": "Month",
        "day": "Day",
        "hour": "Hour"
        }, inplace=True)

        # converting hours from 1...24 to 0...23
        df_pivot["Hour"] = df_pivot["Hour"] - 1

        # Apply the date adjustment.
        df_pivot["Datetime"] = df_pivot.apply(self.adjust_date, axis=1)

        # If adjust_date returns lists (to handle the leap-year duplication),
        # explode the DataFrame on this column.
        df_pivot = df_pivot.explode("Datetime")

        # drop Year, Month, Day, Hour
        df_pivot = df_pivot.drop(labels=['Year', 'Month', 'Day', 'Hour'], axis=1)

        # Convert the Datetime column to a proper datetime type and set it as index.
        df_pivot["Datetime"] = pd.to_datetime(df_pivot["Datetime"])
        df_pivot = df_pivot.set_index("Datetime")
        df_pivot.sort_index(inplace=True)

        # Create a full hourly index for the given date range.
        full_index = pd.date_range(self.start_date, self.end_date, freq='60min')

        # Reindex df_pivot to the full_index.
        summary_df = df_pivot.reindex(full_index)

        # Renaming column titles from country to country_elec, e.g. FI00 -> FI00_elec
        # Note: This is beacuse ts_cf has dimensions (flow, node) in Backbone
        for country in self.country_codes:
            if country in summary_df.columns:
                summary_df = summary_df.rename({country: f"{country}_elec"}, axis=1)
            
        return summary_df


# __main__ allows testing by calling this .py file directly.
if __name__ == '__main__':
    # Define the input parameters.
    input_folder = os.path.join("..\\inputFiles\\timeseries")
    output_folder = os.path.join("..\\inputData-test")
    output_file = os.path.join(output_folder, 'test_output_wind_offshore_cf.csv')
    country_codes = [
        'FI00', 'FR00', 'NOM1'
    ]
    start_date = '1982-01-01 00:00:00'
    end_date = '2021-01-01 00:00:00'

    kwargs_processor = {'input_folder': input_folder,
                        'country_codes': country_codes,
                        'start_date': start_date,
                        'end_date': end_date
    }
    
    # Create an instance of processor and run the processing.
    processor = wind_offshore_MAF2019(**kwargs_processor)
    summary_df = processor.run()
    
    if summary_df is not None:
        # Ensure the output directory exists.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
        # Write the scaled profiles to a CSV file.
        print(f"writing {output_file}")
        summary_df.to_csv(output_file)
    else: 
        print("no outputs from processor")        