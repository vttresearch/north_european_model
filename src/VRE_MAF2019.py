import os
import calendar
import pandas as pd
import numpy as np


class VRE_MAF2019:
    """
    Class to rocess capacity factor data from a CSV file, apply date adjustments,
    and prepare a time series DataFrame for the specified date range.

    Parameters:
        input_folder (str): Relative location of input files.
        input_file : (str) Name of the input file in input_folder
        country_codes (list): List of country codes.
        start_date (str): Start datetime (e.g., '1982-01-01 00:00:00').
        end_date (str): End datetime (e.g., '2021-01-01 00:00:00').
        cf : (str) Type of capacity factor, e.g. 'PV', 'Wind'
        attached_grid : (str) suffix to append to country codes in output columns

    Returns:
    --------
    pandas.DataFrame or None
        DataFrame with processed time series data indexed by datetime,
        or None if the input file is not found.

    """

    def __init__(self, **kwargs_processor):
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
        missing_params = [param for param in required_params if param not in kwargs_processor]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Unpack required parameters
        for param in required_params:
            setattr(self, param, kwargs_processor.get(param))        

        # Extract start and end years from the provided dates.
        self.startyear = pd.to_datetime(self.start_date).year
        self.endyear = pd.to_datetime(self.end_date).year

    def run(self):
        """
        called by create_timeseries.py. 
        
        Organizing the workflow with subfunctions.
        """
        # Check the input file exists
        file_path = os.path.join(self.input_folder, self.input_file)
        print(f"   Reading cf data from '{file_path}'.. ")
        if not os.path.isfile(file_path):
            print(f"Warning! Input file '{file_path}' not found, skipping time series!!!")
            return None
        
        print(f"   Processing the input file.. ")
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
        -----------
        file_path : str
            Path to the input CSV file.

        Returns:
        --------
        pandas.DataFrame
            Transformed and pivoted DataFrame.
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
        -----------
        df_pivot : pandas.DataFrame
            The pivoted DataFrame with time components.
        attached_node : str
            Suffix to append to country codes in output columns.

        Returns:
        --------
        pandas.DataFrame
            The finalized DataFrame with proper datetime index and column names.
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

        return summary_df

    def _adjust_date(self, row):
            """
            Adjusts dates to account for leap year effects in the dataset.

            For dates after February in leap years, shifts the day back by one to 
            compensate for leap day. For December 31 in leap years, returns both 
            the adjusted date and the following day to ensure proper data coverage.

            Parameters:
            -----------
            row : pandas.Series or dict-like
                A row containing 'Year', 'Month', 'Day', and 'Hour' columns/keys
                with integer or integer-like values.

            Returns:
            --------
            list of pandas.Timestamp
                A list containing either one timestamp (normal case) or two timestamps
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


# __main__ allows testing by calling this .py file directly.
if __name__ == '__main__':
    # Define the input parameters.
    input_folder = os.path.join("..\\src_files\\timeseries")
    input_file = "PECD-MAF2019-wide-PV.csv"
    country_codes = [
        'FI00', 'FR00', 'NOM1'
    ]
    start_date = '1982-01-01 00:00:00'
    end_date = '2021-01-01 00:00:00'
    cf = 'PV'
    attached_grid = "elec"
    
    kwargs_processor = {'input_folder': input_folder,
                        'input_file': input_file,
                        'country_codes': country_codes,
                        'start_date': start_date,
                        'end_date': end_date,
                        'cf': cf,
                        'attached_grid': attached_grid
    }

    # Create an instance of processor and run the processing.
    processor = VRE_MAF2019(**kwargs_processor)
    summary_df = processor.run()

    output_folder = os.path.join("..\\inputData-test")
    output_file = os.path.join(output_folder, 'test_output_PV_cf.csv')

    if summary_df is not None:
        # Ensure the output directory exists.
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    
        # Write the scaled profiles to a CSV file.
        print(f"writing {output_file}")
        summary_df.to_csv(output_file)
    else: 
        print("no outputs from processor")