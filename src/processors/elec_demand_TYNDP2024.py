# src/processors/elec_demand_TYNDP2024.py

import os
import calendar
import pandas as pd
import numpy as np
from src.processors.base_processor import BaseProcessor


class elec_demand_TYNDP2024(BaseProcessor):
    """
    Process electricity demand timeseries for countries based on TYNDP 2024 profiles.
    
    This processor reads country-specific electricity demand profiles from Excel files,
    normalizes them, and scales them to match annual demand targets. The processor
    automatically selects the appropriate profile file (2030 or 2040) based on the
    scenario year.
    
    Required Parameters
    -------------------
    input_folder : str
        Relative location of input files containing TYNDP profiles
    country_codes : list[str]
        List of country codes to process (must match Excel sheet names)
    start_date : str
        Start datetime (e.g., '1982-01-01 00:00:00')
    end_date : str
        End datetime (e.g., '2021-01-01 00:00:00')
    df_annual_demands : pd.DataFrame
        DataFrame containing annual demand targets with columns:
        ['country', 'node', 'twh/year', 'constant_share' (optional)]
    scenario_year : int
        Target year for scaling profiles (determines which profile file to use)
    
    Profile Selection
    -----------------
    - scenario_year <= 2035: Uses 'elec_2030_National_Trends.xlsx'
    - scenario_year > 2035: Uses 'elec_2040_National_Trends.xlsx'
    
    Notes
    -----
    - Handles leap year adjustments in the TYNDP data
    - Normalizes profiles so annual sum equals number of valid years
    - Supports optional constant_share for baseload vs. variable demand split
    """

    def __init__(self, **kwargs_processor):
        """Initialize the TYNDP 2024 electricity demand processor."""
        # Initialize base class
        super().__init__(**kwargs_processor)
        
        # Define required parameters
        required_params = [
            'input_folder', 
            'country_codes', 
            'start_date', 
            'end_date', 
            'df_annual_demands', 
            'scenario_year'
        ]

        # Validate presence of required parameters
        missing_params = [param for param in required_params if param not in kwargs_processor]
        if missing_params:
            raise ValueError(f"Missing required parameters: {', '.join(missing_params)}")

        # Extract and store parameters
        self.input_folder = kwargs_processor['input_folder']
        self.country_codes = kwargs_processor['country_codes']
        self.start_date = kwargs_processor['start_date']
        self.end_date = kwargs_processor['end_date']
        self.df_annual_demands = kwargs_processor['df_annual_demands']
        self.scenario_year = kwargs_processor['scenario_year']

        # Extract start and end years
        self.startyear = pd.to_datetime(self.start_date).year
        self.endyear = pd.to_datetime(self.end_date).year

        # Choose the appropriate electricity profile file based on scenario_year
        if self.scenario_year <= 2035:
            self.input_file = os.path.join(self.input_folder, 'elec_2030_National_Trends.xlsx')
        else:
            self.input_file = os.path.join(self.input_folder, 'elec_2040_National_Trends.xlsx')

    def get_values_from_excel(self) -> pd.DataFrame:
        """
        Read Excel file for each country and pivot data into wide format.
        
        Reads country-specific sheets from the TYNDP Excel file, melts the data
        from wide to long format, then pivots it so each row represents a unique
        timestamp and each country has its own column.
        
        Returns
        -------
        pd.DataFrame
            Pivoted DataFrame with columns ['year', 'month', 'day', 'hour', ...country columns]
            Returns empty DataFrame if no valid country sheets are found
        """
        # Load Excel file and available sheet names
        try:
            xl = pd.ExcelFile(self.input_file)
            available_sheets = xl.sheet_names
        except Exception as e:
            raise FileNotFoundError(f"Unable to open {self.input_file}: {e}")

        melted_list = []
        for country in self.country_codes:
            if country not in available_sheets:
                self.log(
                    f"Sheet for country '{country}' not found in {self.input_file}. Skipping.",
                    level="warn"
                )
                continue

            # Read the sheet (headers are on row 8; zero-indexed header=7)
            df = pd.read_excel(self.input_file, sheet_name=country, header=7)
            df = df.drop(labels='Date', axis=1)

            # Rename the first three columns to: month, day, hour
            df.rename(columns={
                df.columns[0]: 'month', 
                df.columns[1]: 'day', 
                df.columns[2]: 'hour'
            }, inplace=True)
            
            # The remaining columns are the values for different years
            year_cols = df.columns[3:]
          
            # Melt the DataFrame so each row corresponds to a specific year and time
            df_melted = df.melt(
                id_vars=['month', 'day', 'hour'], 
                value_vars=year_cols,
                var_name='year', 
                value_name='value'
            )
         
            # Convert the "year" column to integer
            df_melted['year'] = df_melted['year'].astype(int)
            
            # Add a column to denote which country this row is from
            df_melted['country'] = country

            melted_list.append(df_melted)

        if not melted_list:
            return pd.DataFrame()

        # Combine the melted DataFrames and pivot
        combined_df = pd.concat(melted_list, ignore_index=True)
        pivot_df = combined_df.pivot_table(
            index=['year', 'month', 'day', 'hour'], 
            columns='country', 
            values='value',
            aggfunc='first'
        ).reset_index()
        
        return pivot_df

    def process_datetime_index(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the pivoted DataFrame to compute proper datetime index.
        
        This method handles the complexities of the TYNDP data format:
        - Adjusts for zero-indexed months in the source data
        - Handles leap year date shifts (data after Feb is shifted by -1 day)
        - Duplicates December 31 values for leap years to create 8760 hours
        - Reindexes to a complete hourly range
        
        Args
        ----
        input_df : pd.DataFrame
            DataFrame with columns ['year', 'month', 'day', 'hour', ...country columns]
        
        Returns
        -------
        pd.DataFrame
            DataFrame indexed by full hourly datetime range with country value columns
        """
        def adjust_date(row):
            """Adjust date accounting for zero-indexed months and leap years."""
            # Adjust month (Excel months are zero-indexed)
            y = int(row['year'])
            m = int(row['month']) + 1
            d = int(row['day'])
            h = int(row['hour'])
            
            # For leap years, if month is after February, shift the day by -1
            if calendar.isleap(y) and m > 2:
                d = d - 1
                if d < 1:
                    m = m - 1
                    d = calendar.monthrange(y, m)[1]
            
            ts = pd.Timestamp(year=y, month=m, day=d, hour=h)

            # If the row represents December 31 in a leap year, duplicate the timestamp
            if calendar.isleap(y) and int(row['month']) == 11 and int(row['day']) == calendar.monthrange(y, m)[1]:
                extra_ts = ts + pd.Timedelta(days=1)
                return [ts, extra_ts]

            return [ts]

        # Apply the date adjustment once per row
        input_df['datetime'] = input_df.apply(adjust_date, axis=1)
        
        # Explode rows with multiple timestamps
        input_df = input_df.explode('datetime')
        input_df = input_df.dropna(subset=['datetime'])
        
        # Set the datetime as the index and sort
        input_df = input_df.set_index('datetime').sort_index()

        # Create a full hourly index for the defined date range
        full_index = pd.date_range(self.start_date, self.end_date, freq='60min')
        result_df = input_df.reindex(full_index)

        return result_df

    def normalize_profiles(self, df_profiles: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize each country's profile so annual sum equals number of valid years.
        
        For each country column, the method:
        1. Identifies years with valid (positive) data
        2. Counts the number of valid years
        3. Scales the entire series so sum equals the count of valid years
        
        This normalization ensures that when scaled by annual demand, the profile
        will correctly distribute the demand across the year.
        
        Args
        ----
        df_profiles : pd.DataFrame
            DataFrame with datetime index and country columns containing hourly values
        
        Returns
        -------
        pd.DataFrame
            Normalized DataFrame with same structure as input
        """
        for country in self.country_codes:
            if country not in df_profiles.columns:
                continue
            
            s = df_profiles[country]

            # Group by year and check if at least one value is positive
            yearly_valid = s.groupby(s.index.year).apply(lambda x: (x.dropna() > 0).any())
            num_years = int(yearly_valid.sum())

            # If no valid year is found, mark the entire series as NaN
            if num_years == 0:
                df_profiles[country] = np.nan
                continue

            # Sum over the whole series (skipping NaNs)
            total = s.sum(skipna=True)

            if total and total != 0:
                # Scale so that the total equals the number of valid years
                df_profiles[country] = s * (num_years / total)
            else:
                df_profiles[country] = np.nan

        return df_profiles

    def build_demands(
        self, 
        df_profiles_norm: pd.DataFrame, 
        df_annual_demands: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build final demand timeseries by scaling normalized profiles to annual targets.
        
        For each country-node combination in df_annual_demands, this method:
        1. Retrieves the normalized hourly profile for the country
        2. Scales it by the annual demand target
        3. Optionally splits into variable and constant (baseload) components
        
        The scaling formula is: hourly_demand = A * profile + B
        where:
        - A = annual_demand * (1 - constant_share)
        - B = annual_demand * constant_share / 8760
        
        Args
        ----
        df_profiles_norm : pd.DataFrame
            Normalized profiles with datetime index and country columns
        df_annual_demands : pd.DataFrame
            Annual demand targets with columns: ['country', 'node', 'twh/year', 'constant_share']
        
        Returns
        -------
        pd.DataFrame
            Final demand timeseries with datetime index and node columns (in MWh)
        """
        # Create a new DataFrame to store the computed profiles
        df_profiles_result = pd.DataFrame(index=df_profiles_norm.index)

        for country in self.country_codes:
            # Check that the base hourly profile exists for this country
            if country not in df_profiles_norm.columns:
                continue

            # Filter demand rows for the country (case-insensitive)
            country_demand = df_annual_demands[
                df_annual_demands['country'].str.lower() == country.lower()
            ]
            
            if country_demand.empty:
                self.log(
                    f"No demand data for {country}, skipping.",
                    level="warn"
                )
                continue

            # Process each demand row for the country
            for _, row in country_demand.iterrows():
                # Pick node name as column name
                col_name = str(row['node'])

                # Retrieve annual demand from 'twh/year' and convert to MWh
                annual_demand = row['twh/year'] * 10**6

                # Retrieve constant_share if available; default to 0.0
                constant_share = row.get('constant_share', 0.0)
                if pd.isna(constant_share):
                    constant_share = 0.0

                # Validate constant_share
                if not (0 <= constant_share <= 1):
                    raise ValueError(
                        f"Constant_share for {col_name} must be between 0 and 1. Got {constant_share}."
                    )

                # Calculate hourly demands = A*x + B
                # A = variable component scaling factor
                # B = constant (baseload) component per hour
                A = annual_demand * (1 - constant_share)
                B = annual_demand * constant_share / 8760
                
                # Compute the new profile using the base profile
                df_profiles_result[col_name] = A * df_profiles_norm[country] + B

        return df_profiles_result

    def process(self) -> pd.DataFrame:
        """
        Main processing logic - implements the abstract method from BaseProcessor.
        
        Workflow:
        1. Read electricity demand profiles from TYNDP Excel file
        2. Process datetime index (handle leap years, etc.)
        3. Normalize profiles to per-year basis
        4. Build final demand timeseries scaled to annual targets
        
        Returns
        -------
        pd.DataFrame
            Final demand timeseries with datetime index and node columns (in MWh)
        """
        # Get the raw hourly profiles
        self.log(f"Reading electricity demand profiles from '{self.input_file}'...")
        df_out = self.get_values_from_excel()

        self.log("Processing datetime index, handling leap days, etc...")
        df_out = self.process_datetime_index(df_out)

        self.log("Normalizing demand profiles...")
        df_out = self.normalize_profiles(df_out)

        self.log("Building demand time series...")
        df_out = self.build_demands(df_out, self.df_annual_demands)

        # Set secondary result (none for this processor)
        self.secondary_result = None

        self.log("Demand time series built.", level="info")

        return df_out