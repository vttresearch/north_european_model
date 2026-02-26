import os
import calendar
import pandas as pd
import numpy as np
from pathlib import Path
from src.processors.base_processor import BaseProcessor
from tqdm import tqdm


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
            excel_filename = 'elec_2030_National_Trends.xlsx'
        else:
            excel_filename = 'elec_2040_National_Trends.xlsx'
            
        self.input_file = os.path.join(self.input_folder, excel_filename)
        
        # Define corresponding parquet cache file
        parquet_filename = excel_filename.replace('.xlsx', '.parquet')
        self.parquet_file = os.path.join(self.input_folder, parquet_filename)

    def need_to_create_parquet_cache(self) -> bool:
        """
        Check if the code needs to create parquet 
         * if cache does not exist or 
         * if cache is older than the Excel file.
        
        Returns
        -------
        bool
            True if parquet cache needs to be (re)created from the excel file
            False if parquet cache creation can be skipped
        """
        excel_path = Path(self.input_file)
        parquet_path = Path(self.parquet_file)
        
        # Check if both files exist
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel file not found: {self.input_file}")
        
        if not parquet_path.exists():
            self.logger.log_status("Parquet cache not found, will create from Excel file.", level="none")
            return True
        
        # Compare modification times
        excel_mtime = excel_path.stat().st_mtime
        parquet_mtime = parquet_path.stat().st_mtime
        
        if parquet_mtime > excel_mtime:
            self.logger.log_status("Using parquet cache.", level="none")
            return False
        else:
            self.logger.log_status("Excel file is newer than parquet cache, will rebuild cache.", level="none")
            return True

    def read_excel_to_parquet(self) -> None:
        """
        Reads all allowed_countries from Excel file and save as parquet for future use. 
        
        This method reads the entire Excel file (all allowed_countries sheets), processes 
        them into a single wide-format DataFrame, and saves it as parquet for faster
        subsequent loads.

        This  approach (having all working data in parqut) allows sharing the same parquet file 
        between different config files.
        
        Returns
        -------
        None

        Writes
        ------
        parquet file with the same name than input_file except .xlsx replaced with .parquet
        
        Notes
        -----
        Contains a hard-coded limitation to allowed_countries, because some of the non-included countries
        had some kind of an error in data and the code will crash for the full dataset. I didn't use time 
        to debug and fix this issue, because it does not impact currently needed countries or the next
        possible expansion set.
        """
        self.logger.log_status(f"Reading full Excel file: '{self.input_file}' ...", level="none")
        
        try:
            xl = pd.ExcelFile(self.input_file)
            available_sheets = xl.sheet_names
        except Exception as e:
            raise FileNotFoundError(f"Unable to open {self.input_file}: {e}")

        melted_list = []
        allowed_countries = ('AT00', 'BE00', 'CH00', 'DE00', 'DKW1', 'DKE1', 'EE00', 'ES00', 
                                  'FI00', 'FR00', 'LT00', 'LV00', 'NL00', 'NOS0', 'NOM1',
                                  'NON1', 'PL00', 'SE01', 'SE02', 'SE03', 'SE04', 'UK00',
                                  'ITN1', 'ITCN', 'ITCS', 'PT00'
                                  )

        sheets_to_process = set(allowed_countries) & set(available_sheets)

        # Read all allowed_countries 
        for sheet_name in tqdm(sheets_to_process, desc=f"Processing"):
            if sheet_name not in allowed_countries:
                continue 
            try:
                # Read the sheet (headers are on row 8; zero-indexed header=7)
                df = pd.read_excel(self.input_file, sheet_name=sheet_name, header=7)
                
                # Skip if 'Date' column doesn't exist (not a data sheet)
                if 'Date' not in df.columns:
                    continue
                    
                df = df.drop(labels='Date', axis=1)

                # Rename the first three columns to: month, day, hour
                df.rename(columns={
                    df.columns[0]: 'month', 
                    df.columns[1]: 'day', 
                    df.columns[2]: 'hour'
                }, inplace=True)
                
                # Convert month from 0-indexed to 1-indexed NOW to avoid complexity later
                df['month'] = df['month'] + 1
                
                # The remaining columns are the values for different years
                year_cols = df.columns[4:]  # Skip month, day, hour, and now the adjusted month
              
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
                df_melted['country'] = sheet_name

                melted_list.append(df_melted)
                
            except Exception as e:
                self.logger.log_status(f"Skipping sheet '{sheet_name}' in '{self.input_file}': {e}", level="warn")
                continue

        if not melted_list:
            raise ValueError("No valid data sheets found in Excel file")

        # Combine the melted DataFrames and pivot
        self.logger.log_status("Combining data from all sheets...", level="none")
        combined_df = pd.concat(melted_list, ignore_index=True)
        
        self.logger.log_status("Pivoting to wide format...", level="none")
        pivot_df = combined_df.pivot_table(
            index=['year', 'month', 'day', 'hour'], 
            columns='country', 
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Save to parquet
        self.logger.log_status(f"Saving parquet cache to: '{self.parquet_file}'...", level="none")
        pivot_df.to_parquet(self.parquet_file, index=False, engine='pyarrow', compression='snappy')


    def load_data_from_parquet(self) -> pd.DataFrame:
        """
        Load parquet file with only the required country columns.
        
        Does NOT filter years - that happens later in get_values_from_excel()
        after we have the full pivoted structure.
        
        Returns
        -------
        pd.DataFrame
            Wide-format DataFrame with columns ['year', 'month', 'day', 'hour', ...requested countries]
            Contains ALL available years from the parquet file
        """
        self.logger.log_status(f"Loading parquet cache: '{self.parquet_file}'...", level="none")
        
        try:
            # Load the full parquet file
            df_full = pd.read_parquet(self.parquet_file, engine='pyarrow')
            
            # Identify which columns are country data vs index columns
            index_cols = ['year', 'month', 'day', 'hour']
            available_countries = [col for col in df_full.columns if col not in index_cols]
            
            # Filter to only requested countries that exist
            countries_to_load = [c for c in self.country_codes if c in available_countries]
            missing_countries = [c for c in self.country_codes if c not in available_countries]
            
            if missing_countries:
                self.logger.log_status(f"Countries not found in cache: {missing_countries}", level="warn")
            
            # Select only the columns we need
            columns_to_keep = index_cols + countries_to_load
            df_filtered = df_full[columns_to_keep].copy()
            
            self.logger.log_status(
                f"Loaded {len(countries_to_load)} countries from parquet cache "
                f"({len(df_filtered)} total rows).", 
                level="info"
            )
            
            return df_filtered
            
        except Exception as e:
            raise RuntimeError(f"Failed to load parquet cache: {e}")

    def get_values_from_excel(self) -> pd.DataFrame:
        """
        Read electricity demand data - uses parquet cache if available and current.
        
        This method implements the caching logic:
        1. Check if parquet cache exists and is newer than Excel
        2. If yes: load from parquet (fast)
        3. If no: read full Excel and create parquet cache (slow, but only once)
        4. Filter to requested year range
        
        NOTE: Year filtering happens AFTER loading to preserve row structure.
        Not all countries have data for all years, so filtering by year at the
        DataFrame level would create gaps in the data.
        
        Returns
        -------
        pd.DataFrame
            Wide-format DataFrame with columns ['year', 'month', 'day', 'hour', ...country columns]
            Filtered to only years within [startyear, endyear] range
        """
        if self.need_to_create_parquet_cache():
            self.read_excel_to_parquet()

        # Load from parquet. This is fast enough (sub 1sec) that the code is simplified by always
        # taking this path.
        df_wide = self.load_data_from_parquet()
        
        # Filter by year on the complete row structure to preserve the standardized 8760 hours per year structure
        year_mask = (df_wide['year'] >= self.startyear) & (df_wide['year'] <= self.endyear)
        df_wide = df_wide[year_mask].copy()
        
        self.logger.log_status(
            f"Filtered to years {self.startyear}-{self.endyear}: {len(df_wide)} rows.",
            level="info"
        )

        return df_wide

    def process_datetime_index(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Process the pivoted DataFrame to compute proper datetime index.
        
        This method handles the complexities of the TYNDP data format:
        - Months are now 1-indexed (converted during Excel read)
        - All years use a standardized 365-day calendar (8760 hours)
        - For leap years: we need to map 365 days of standardized data to 366 actual days
        
        Mapping strategy for leap years (preserves chronological order):
        - Standard days 1-59 (Jan 1 - Feb 28): Direct 1:1 mapping
        - Standard day 60 (Mar 1 in std year): Maps to Feb 29 (the inserted leap day)
        - Standard days 61-365 (Mar 2 - Dec 31 std): Map to Mar 1 - Dec 30 actual
        - Standard day 365 is ALSO duplicated to create Dec 31
        
        Example for 2016 (leap year):
        - Std [month=2, day=28] → 2016-02-28 ✓
        - Std [month=3, day=1] → 2016-02-29 (Feb 29 gets Mar 1 data) ✓
        - Std [month=3, day=2] → 2016-03-01 (Mar 1 gets Mar 2 data) ✓
        - ...
        - Std [month=12, day=31] → 2016-12-30 AND 2016-12-31 (duplicated) ✓
        
        This maintains chronological order while inserting the leap day.
        
        Args
        ----
        input_df : pd.DataFrame
            DataFrame with columns ['year', 'month', 'day', 'hour', ...country columns]
            Months are 1-indexed
        
        Returns
        -------
        pd.DataFrame
            DataFrame indexed by full hourly datetime range with country value columns
        """
        if input_df.empty:
            raise ValueError("Input DataFrame is empty - cannot process datetime index")
        
        # Calculate standardized day-of-year for each row (using a non-leap reference year)
        # This gives us the "position" in the 365-day standardized year
        input_df['std_doy'] = input_df.apply(
            lambda row: pd.Timestamp(year=2001, month=int(row['month']), day=int(row['day'])).dayofyear,
            axis=1
        )
        
        def map_to_actual_date(row):
            """
            Map standardized day-of-year to actual calendar date.
            
            For non-leap years: Direct 1:1 mapping
            For leap years: 
                - Days 1-59 map directly (Jan 1 - Feb 28)
                - Day 60 becomes Feb 29 (inserted)
                - Days 61-365 become Mar 1 - Dec 30
                - Day 365 ALSO creates Dec 31 (duplicated)
            """
            y = int(row['year'])
            h = int(row['hour'])
            std_doy = int(row['std_doy'])
            
            if not calendar.isleap(y):
                # Non-leap year: direct mapping (std day N → actual day N)
                actual_doy = std_doy
                ts = pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=actual_doy - 1, hours=h)
                return [ts]
            else:
                # Leap year: need to insert Feb 29
                if std_doy <= 59:  # Jan 1 - Feb 28: direct mapping
                    actual_doy = std_doy
                elif std_doy == 60:  # Std Mar 1 → Feb 29 (the inserted day)
                    actual_doy = 60  # Feb 29
                else:  # std_doy 61-365: shift back by 1
                    actual_doy = std_doy  # Mar 1 comes from std day 61, etc.
                
                ts = pd.Timestamp(year=y, month=1, day=1) + pd.Timedelta(days=actual_doy - 1, hours=h)
                
                # For the last standardized day (Dec 31 in std year = day 365),
                # also create the actual Dec 31 by duplicating
                if std_doy == 365:
                    # This row creates BOTH Dec 30 (ts) and Dec 31 (duplicate)
                    dec_31_ts = pd.Timestamp(year=y, month=12, day=31, hour=h)
                    return [ts, dec_31_ts]
                
                return [ts]
        
        # Apply the date mapping
        input_df['datetime'] = input_df.apply(map_to_actual_date, axis=1)
        
        # Explode rows with multiple timestamps (Dec 30→31 duplication for leap years)
        input_df = input_df.explode('datetime')
        input_df = input_df.dropna(subset=['datetime'])
        
        # Set datetime as index and sort
        input_df = input_df.set_index('datetime').sort_index()
        
        # Drop helper columns
        input_df = input_df.drop(columns=['year', 'month', 'day', 'hour', 'std_doy'])
        
        # Check for duplicate timestamps before reindexing
        if not input_df.index.is_unique:
            n_dupes = input_df.index.duplicated().sum()
            self.logger.log_status(f"Found {n_dupes} duplicate timestamps in elec demand profiles, keeping first occurrence", level="warn")
            input_df = input_df[~input_df.index.duplicated(keep='first')]
        
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
                self.logger.log_status(
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
        1. Read electricity demand profiles (from parquet cache if available, else Excel)
        2. Process datetime index (handle leap years, etc.)
        3. Normalize profiles to per-year basis
        4. Build final demand timeseries scaled to annual targets
        
        Returns
        -------
        pd.DataFrame
            Final demand timeseries with datetime index and node columns (in MWh)

        Notes
        -----
        read_excel_to_parquet(..)
        Contains a hard-coded limitation to allowed_countries, because some of the non-included countries
        had some kind of an error in data and the code will crash for the full dataset. I didn't use time 
        to debug and fix this issue, because it does not impact currently needed countries or the next
        possible expansion set.
        """
        # Get the raw hourly profiles (uses parquet cache if available)
        self.logger.log_status(f"Reading electricity demand profiles...")
        df_out = self.get_values_from_excel()

        self.logger.log_status("Processing datetime index, handling leap days, etc...")
        df_out = self.process_datetime_index(df_out)

        self.logger.log_status("Normalizing demand profiles...")
        df_out = self.normalize_profiles(df_out)

        self.logger.log_status("Building demand time series...")
        df_out = self.build_demands(df_out, self.df_annual_demands)

        # Set secondary result (none for this processor)
        self.secondary_result = None

        self.logger.log_status("Demand time series built.", level="info")

        return df_out