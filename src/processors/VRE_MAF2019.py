# src/processors/VRE_MAF2019.py

import os
import calendar
import pandas as pd
from src.processors.base_processor import BaseProcessor


class VRE_MAF2019(BaseProcessor):
    """
    Class to process capacity factor data from a CSV file, apply date adjustments,
    and prepare a time series DataFrame for the specified date range.

    Parameters:
        input_folder (str): Relative location of input files.
        input_file (str): Name of the input file in input_folder
        country_codes (list): List of country codes.
        start_date (str): Start datetime (e.g., '1982-01-01 00:00:00').
        end_date (str): End datetime (e.g., '2021-12-31 23:00:00').
        attached_grid (str): suffix to append to country codes in output columns

    Returns:
        main_result (pd.DataFrame): DataFrame with processed time series data indexed by datetime
        secondary_result: None (not used for this processor)
    """

    def __init__(self, **kwargs):
        # Initialize base class
        super().__init__(**kwargs)
        
        # List of required parameters
        required_params = [
            'input_folder', 
            'input_file',
            'country_codes', 
            'start_date', 
            'end_date',
            'attached_grid'
        ]

        # Check if all required parameters are present
        missing_params = [param for param in required_params if param not in kwargs]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Unpack required parameters
        for param in required_params:
            setattr(self, param, kwargs.get(param))

    def process(self) -> pd.DataFrame:
        """
        Main processing logic that organizes the workflow with subfunctions.
        
        Returns:
            pd.DataFrame: Processed time series data with datetime index
        """
        # Check the input file exists
        file_path = os.path.join(self.input_folder, self.input_file)
        self.log(f"Reading cf data from '{file_path}'..")
        
        if not os.path.isfile(file_path):
            self.log(f"Input file '{file_path}' not found, skipping VRE_MAF2019 time series.", level="warn")
            return pd.DataFrame()
        
        self.log("Processing the input file..")
        
        # Extract and transform data
        df_pivot = self._transform_input_data(file_path, self.country_codes)

        # Apply date adjustments and prepare final dataframe
        summary_df = self._finalize_dataframe(df_pivot, self.country_codes, 
                                              self.start_date, self.end_date, self.attached_grid)

        return summary_df

    def _transform_input_data(self, file_path, country_codes):
        """
        Read and transform the input CSV file into a pivoted DataFrame.

        Parameters:
            file_path (str): Path to the input CSV file.
            country_codes (list): List of country codes to filter

        Returns:
            pd.DataFrame: Transformed and pivoted DataFrame.
        """
        # Read the input file
        df_cf_ts = pd.read_csv(file_path, sep=";", decimal=",")
        year_cols = df_cf_ts.columns[3:]

        # Filter by country codes
        df_cf_ts = df_cf_ts[df_cf_ts["area"].isin(country_codes)]

        # Melt the DataFrame
        df_melted = df_cf_ts.melt(
            id_vars=["area", "day", "month", "hour"],
            value_vars=year_cols,
            var_name="year",
            value_name="value"
        )

        # Convert time columns to integers
        for col in ["year", "month", "day", "hour"]:
            df_melted[col] = df_melted[col].astype(int)

        # Pivot to create the required structure
        df_pivot = df_melted.pivot_table(
            index=["year", "month", "day", "hour"],
            columns="area",
            values="value"
        ).reset_index()

        # Rename columns and convert hours from 1-24 to 0-23
        df_pivot.rename(columns={
            "year": "Year",
            "month": "Month",
            "day": "Day",
            "hour": "Hour"
        }, inplace=True)

        df_pivot["Hour"] = df_pivot["Hour"] - 1

        return df_pivot

    def _finalize_dataframe(self, df_pivot, country_codes, start_date, end_date, attached_grid):
        """
        Apply date adjustments and prepare the final time series DataFrame.

        Parameters:
            df_pivot (pd.DataFrame): The pivoted DataFrame with time components.
            country_codes (list): List of country codes
            start_date (str): Start datetime for the time series
            end_date (str): End datetime for the time series
            attached_grid (str): Suffix to append to country codes in output columns.

        Returns:
            pd.DataFrame: The finalized DataFrame with proper datetime index and column names.
        """
        # Apply date adjustments
        df_pivot["Datetime"] = df_pivot.apply(self._adjust_date, axis=1)

        # Handle potential multiple timestamps from adjust_date
        df_pivot = df_pivot.explode("Datetime")

        # Clean up and prepare final structure
        df_pivot = df_pivot.drop(labels=['Year', 'Month', 'Day', 'Hour'], axis=1)
        df_pivot["Datetime"] = pd.to_datetime(df_pivot["Datetime"])
        df_pivot = df_pivot.set_index("Datetime")
        df_pivot.sort_index(inplace=True)

        # Ensure complete time series over the specified range
        full_index = pd.date_range(start_date, end_date, freq='60min')
        summary_df = df_pivot.reindex(full_index)

        # Rename country columns to indicate the attached node
        for country in country_codes:
            if country in summary_df.columns:
                summary_df = summary_df.rename({country: f"{country}_{attached_grid}"}, axis=1)

        # Secondary result is None for this processor
        self.secondary_result = None

        return summary_df
    
    def _adjust_date(self, row):
        """
        Adjusts dates to account for leap year effects in the dataset.

        For dates after February in leap years, shifts the day back by one to 
        compensate for leap day. For December 31 in leap years, returns both 
        the adjusted date and the following day to ensure proper data coverage.

        Parameters:
            row (pd.Series): A row containing 'Year', 'Month', 'Day', and 'Hour' columns
                            with integer or integer-like values.

        Returns:
            list: A list containing either one timestamp (normal case) or two timestamps
                  (special case for December 31 in leap years).
        """
        # Extract date components from the row
        y = int(row['Year'])
        m = int(row['Month'])
        d = int(row['Day'])
        h = int(row['Hour'])

        # For dates after February in leap years, adjust for leap day
        if calendar.isleap(y) and m > 2:
            d = d - 1
            # If day becomes zero, roll back to the last day of the previous month.
            if d < 1:
                m = m - 1
                d = calendar.monthrange(y, m)[1]

        # Create timestamp with adjusted date components
        ts = pd.Timestamp(year=y, month=m, day=d, hour=h)

        # Special handling for December 31 in leap years
        # We need to return both the adjusted date and the following day
        if calendar.isleap(y) and int(row['Month']) == 12 and int(row['Day']) == calendar.monthrange(y, m)[1]:
            # Add an extra timestamp for the next day to ensure complete data coverage
            extra_ts = ts + pd.Timedelta(days=1)
            return [ts, extra_ts]

        return [ts]